# roi_extractor 设计文档

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人(gitId) |
| --- | --- | --- | --- |
| v1.0 | 基于 spec.yaml（11-stage PASS）与 req_analysis.md 生成 SE 方案设计 | 2026-09-10 | qq_52056150 |

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `roi_extractor`（注册名 `RoiExtractor`） |
| 算子类别 | Interpolation（category_enum.yaml 明确列出 roi_align） |
| 范式 | Interpolation |
| 条件范式路由 | 无 |
| 支持数据类型 | 输入 feats/rois：float16/float32；输出 y 与 feats 一致 |
| 目标芯片 | Ascend 950PR（主）、Ascend 910B（兼容） |
| 目标架构 | Ascend950PR → DAV_3510/a5；Ascend910B → DAV_2201/a3 |
| 编程框架 | PyPTO Pro（kernel）+ AscendC Host（registry-invoke 接入） |
| 目标仓 | ops-transformer，`detection/roi_extractor/`（新领域目录，根 CMakeLists 注册） |

### 1.2 算子功能

给定 FPN 多层级特征图与 RoI 列表，按 RoI 面积做层级映射后在其映射层级执行 RoIAlign 双线性插值池化，输出 `[K, C, pooled_h, pooled_w]` 固定尺寸特征。兼容 mmdet `SingleRoIExtractor` 语义与遗留 GE `RoiExtractor` 属性面，修复遗留 TBE 版三处语义缺口（仅方形池化、pool_mode 恒 avg、roi_scale_factor no-op）。

### 1.3 数学公式

记 `feats_l ∈ [N,C,H_l,W_l]`（NCHW，l∈[0,L)），`rois ∈ [K,5]`（batch_idx,x1,y1,x2,y2）：

**层级映射**：
`scale = sqrt((x2-x1)·(y2-y1)) / roi_scale_factor`（factor>0 时；否则不带除法）
`lvl = clamp(⌊log2(scale / finest_scale + 1e-6)⌋, 0, L-1)`

**坐标变换**（层 lvl 的 spatial_scale 记 `s_lvl`）：
`rx1 = x1·s_lvl − 0.5·aligned`，`rx2/ry1/ry2` 同理；`rw = max(rx2−rx1, 1)`，`rh = max(ry2−ry1, 1)`；`bw = rw/pooled_w`，`bh = rh/pooled_h`

**采样网格**（`n_h = sample_num>0 ? sample_num : max(⌈bh⌉,1)`，`n_w` 同理）：
`y(k,ph,ih) = ry1 + (ph+0.5)·bh + (ih+0.5)·bh/n_h`；`x(k,pw,iw) = rx1 + (pw+0.5)·bw + (iw+0.5)·bw/n_w`

**双线性插值**（边界规则与 torchvision 严格一致）：
- `y < −1 ∨ y > H ∨ x < −1 ∨ x > W` → 贡献 0（avg 计入分母）
- 否则 `y' = max(y,0)`，`y0 = ⌊y'⌋`，`y1 = min(y0+1, H−1)`，`λy = y'−y0`（x 同理）
- `val = (1−λy)(1−λx)·v[y0,x0] + (1−λy)λx·v[y0,x1] + λy(1−λx)·v[y1,x0] + λyλx·v[y1,x1]`

**bin 聚合**：`y[k,c,ph,pw] = (Σ_{ih,iw} val) / (n_h·n_w)`（avg）或 `max_{ih,iw} val`（max）

**[K,4]+index 变体**：batch_idx 由 index 携带，其余同上（数学等价，kernel 按 `hasIndex` 分支读列偏移）。

### 1.4 spec.yaml 一致性映射

| spec 字段 | 设计承接位置 | 状态 | 说明 |
| --- | --- | --- | --- |
| `op.paradigm_groups` | 无（未声明） | — | 单范式 Interpolation，无组合路由 |
| `reduction` | §3.2/§3.5 | 已承接 | bin 内 avg/max 归约 → TilingKey bit0 分模板 |
| `dtype_policy.supported_combinations` | §3.2/§3.5 | 已承接 | fp16/fp32 经 `input_dtype` datatype 映射单二进制覆盖，fp32 中间量累加 |
| `outputs[].shape_rule` / `broadcast` | §3.3/§3.4 | 已承接 | y=[K,C,ph,pw]，无广播；infershape 直推 |
| `math_semantics.formula` / `reference_oracle` | §1.3/§3.5 | 已承接 | 公式逐条翻译为 kernel 步骤；oracle 缺席（多层级无单一 callable），以 golden.py + ST 对拍 |
| `boundary_conditions` / `extreme_inputs` | §3.5/§6 | 已承接 | K=0、退化 RoI、越界、非方形、max、固定 sr、非法属性报错均入 UT/ST |
| `numerical_tolerance.per_dtype` | §5.2 | 已承接 | fp32 1e-5/1e-5，fp16 1e-3/1e-3 |
| `determinism` | §4.2 | 已承接 | 累加顺序固定（ih,iw 顺序串行），同输入同结果 |

## 2. 架构设计

### 2.1 逻辑视图

| 层 | 职责 | 核心文件 |
| --- | --- | --- |
| op_api | aclnn 两段式接口：aclTensorList 解包、空指针/越界/dtype 校验、构建 executor | `op_host/op_api/aclnn_roi_extractor.{h,cpp}`（ACLNNTYPE=aclnn，L2 构建系统自动生成框架 + 手写 L0） |
| op_host | OpDef（5 个 feats 槽位 + rois + index + y）、infershape、tiling（TilingData/TilingKey 计算） | `roi_extractor_def.cpp`、`roi_extractor_infershape.cpp`、`roi_extractor_tiling.{h,cpp}` |
| op_kernel | PyPTO Pro kernel：层级映射 + 双线性插值 + avg/max 池化 | `op_kernel/roi_extractor.py`（`@pl.jit`）、`roi_extractor_metadata.h` |
| op_graph | GEIR 原型（proto 声明） | `op_graph/roi_extractor_proto.h` |
| torch_extension | PyTorch 适配（list 输入 → 槽位填充） | `torch_extension/` |

**feats 输入承载方式（关键决策）**：GE DYNAMIC_INPUT 无法与 PyPTO 固定 kernel 签名对接（仓内 5 个 PyPTO 算子均为固定签名 + OPTIONAL_INPUT nullptr 模式）。故 OpDef 采用 **5 个 OPTIONAL_INPUT（feats0..feats4）**，`numLevels` 由 Host Tiling 统计非空槽位数写入 TilingData；aclnn 对外仍暴露 `aclTensorList*`（≤5 层），L0 解包后按槽位传入。遗留 GE 图（dynamic features）如需切换，待后续 fusion_pass 适配（本期不交付）。

### 2.2 开发视图

```text
ops-transformer/detection/roi_extractor/
├── CMakeLists.txt               # 与 attention/ 同款 glob 发现
├── README.md
├── docs/
│   ├── spec.yaml                # L0 数学契约（11-stage PASS）
│   ├── DESIGN.md                # 本文档（SE）
│   ├── MDE.md                   # MDE 详细设计
│   └── PLAN.md                  # 迭代计划
├── op_graph/
│   └── roi_extractor_proto.h
├── op_host/
│   ├── CMakeLists.txt           # require_pypto_pro + enable_pypto_kernel + add_modules_sources
│   ├── roi_extractor_def.cpp
│   ├── roi_extractor_infershape.cpp
│   ├── roi_extractor_tiling.h
│   ├── roi_extractor_tiling.cpp
│   ├── op_api/
│   │   ├── aclnn_roi_extractor.h
│   │   └── aclnn_roi_extractor.cpp
│   └── config/ascend950/        # binary.json + simplified_key.ini（构建生成/随仓）
├── op_kernel/
│   ├── roi_extractor.py         # PyPTO Pro kernel
│   └── roi_extractor_metadata.h
├── examples/
│   └── test_aclnn_roi_extractor.cpp
├── torch_extension/
└── tests/
    ├── ut/op_host/test_roi_extractor_tiling.cpp
    └── ut/op_api/test_aclnn_roi_extractor.cpp
```

无 arch22/arch35 子目录：PyPTO kernel 单 .py 双架构编译（a2a3/a5 由构建系统 --soc 决定）。

### 2.3 运行视图

1. **aclnn 调用**：`aclnnRoiExtractorGetWorkspaceSize(featsList, rois, index, attrs…, y, &ws, &exec)` → 校验 → `aclnnRoiExtractor(ws, wsSize, exec, stream)`
2. **Host Tiling**（编译期+运行期）：读 feats 槽位/rois/index shape + 8 个属性 → 校验（dtype 一致、L≤5、C 各层相同、rois 末维 4/5、pooled 1..56、sample_num 0..16、pool_mode 枚举、batch_idx<N 由 op_api 检查）→ 计算 `RoiExtractorTiling` 字段 + TilingKey（bit0=pool_mode）→ workspace（仅系统保留）
3. **Kernel 执行**（AICore，Vector）：
   - 各核领取 work item（k, wChunk, cBlock）
   - 标量段：读 rois[k] → 层级映射 → 坐标变换
   - 行加载：`pl.load` 把 `T[Cb, R·Wc]` 行块载入 UB（GM 张量视图，stride=(Hl·Wl,1)，逐行 R 次）
   - 采样循环：`vf.arange` 生成通道基址 + `vf.gather` 取四角 → 加权插值 → `vf.adds`/`vf.max` 累加
   - 写回：`pl.store` 输出 `[Cb, PWc]`（K,C,PH,PW 视图）
4. **Workspace**：仅 `GetLibApiWorkSpaceSize()` 系统保留，无自定义 workspace

### 2.4 用户视图

- **单算子（aclnn）**：`aclnnRoiExtractorGetWorkspaceSize/aclnnRoiExtractor`，feats 以 aclTensorList（≤5 层 NCHW）传入
- **PyTorch**：`torch.ops.npu_atb.RoiExtractor`（torch_extension 封装，list[Tensor] 入参）

## 3. 实现方案

### 3.1 技术路径决策

**Vector（PyPTO Pro vf 指令族），非 RegBase/Cube/SIMT**。理由：
1. 纯逐元素插值 + 小范围归约，无矩阵乘 → 不需要 Cube/RegBase
2. gather 寻址（vf.gather UB 内按索引取数）是核心原语，Vector 原生支持
3. a3(910B)/a5(950) 双端 vf.* 指令族一致，满足 910B 兼容（RegBase 为 950 专属，排除）
4. 仓内 PyPTO 基建（enable_pypto_kernel）对 vf 路径支持完整（dense_lightning/flash_attn_grad 先例）

### 3.2 模板划分总览

| 模板 ID | 范式分区 | 激活范式 | 触发条件 | dtype 覆盖 | shape 覆盖 | 原因 |
| --- | --- | --- | --- | --- | --- | --- |
| T0（bit0=0） | avg | Interpolation | pool_mode="avg"（默认） | fp16/fp32（input_dtype 单二进制） | L∈[1,5]、K∈[0,65536]、C≤4096、ph/pw∈[1,56]、sr∈[0,16] | 累加路径：acc += val，bin 末 ×(1/(n_h·n_w)) |
| T1（bit0=1） | max | Interpolation | pool_mode="max" | 同上 | 同上 | 归约路径：acc = max(acc,val)，内循环分支隔离避免逐点分支 |

dtype 不占 TilingKey 位：`@pl.jit(datatype={"feats0": "input_dtype"})` 由 datatype 映射在加载/存储时按输入 dtype 实例化（fp16 载入后 `pl.cast` 升 fp32 计算，输出前降回）。aligned/sample_num/finest_scale/roi_scale_factor/L/各层 shape 均为运行期 TilingData 标量，不产生模板分叉（只影响标量算术与循环边界）。

### 3.3 TilingData 结构体

Python 侧（op_kernel/roi_extractor.py）：

```python
@dataclass
class RoiExtractorTiling:
    numLevels: int        # L，非空 feats 槽位数 [1,5]
    roisNum: int          # K
    batchSize: int        # N（各层一致，Host 校验）
    channels: int         # C
    pooledHeight: int     # PH [1,56]
    pooledWidth: int      # PW [1,56]
    sampleNum: int        # 0=自适应 [0,16]
    finestScale: int      # 默认 56
    aligned: int          # 0/1
    hasIndex: int         # 0=[K,5]，1=[K,4]+index
    colOffset: int        # rois 列偏移（0 或 1）——hasIndex?1:0 的展开，kernel 免分支
    spatialScale0: float  # 各层缩放（乘法语义）
    spatialScale1: float
    spatialScale2: float
    spatialScale3: float
    spatialScale4: float  # 未用层填 1.0
    levelH0: int; levelH1: int; levelH2: int; levelH3: int; levelH4: int   # 各层 H_l
    levelW0: int; levelW1: int; levelW2: int; levelW3: int; levelW4: int   # 各层 W_l
    workPerCore: int      # 每核 work item 数（k×wChunk×cBlock 均分）
    workTailCore: int     # 尾核 work item 数
    cBlock: int           # 通道块（64，fp16 可 128）
    wChunk: int           # 列块（tiling 按 UB 预算推导）
    invSampleCount: float # 1/(n_h·n_w)（avg 模式，Host 预计算；自适应时逐 RoI 不同——见 3.5 注）
```

C++ 侧（op_host/roi_extractor_tiling.h）：`BEGIN_TILING_DATA_DEF(RoiExtractorTilingDataTmp)` + `TILING_DATA_FIELD_DEF` **逐字段同名同序**（codegen 生成的 `RoiExtractorTiling_tiling.h` force-include，两侧内存排布一致），`REGISTER_TILING_DATA_CLASS(RoiExtractor, RoiExtractorTilingDataTmp)`。

### 3.4 Host 侧 Tiling 方案

`RoiExtractorTilingFunc(gert::TilingContext*)`：

1. 平台：`platform_ascendc` → aivNum/aicNum，`CalcTschBlockDim`，`SetBlockDim`
2. 输入统计：`GetOptionalInputShape(0..4)` 非空者收集 (H_l, W_l)，L=count；空槽必须只在尾部（否则 shape_mismatch）
3. 校验：feats 各层 N/C 一致且 dtype ∈ {FP16,FP32} 且与 rois 一致；index dtype INT32 且 [K]；rois [K,5]/[K,4]（[K,4] 时 index 必须提供）；属性范围（finest_scale≥1、sample_num∈[0,16]、pooled∈[1,56]、pool_mode∈{avg,max}、roi_scale_factor∈[0,16]）
4. TilingData 填充：上述字段 + UB 预算推导：
   - `R = (sampleNum>0 ? sampleNum : 1) · PH + 2` 行预取上界（采样 y 跨度 ≤ PH·bh/n_h + 2 ≤ PH·sr + 2）
   - `wChunk = min(Wl, floor((UB_BUDGET − 输出/杂项) / (cBlock · R · dtypeSize)))`，UB_BUDGET = min(184KB(a3), 248KB(a5)) → 统一 180KB 保守值；不足时减小 cBlock 至 32
   - work items = K · ceil(PW 相关 wChunk 划分数) · ceil(C/cBlock)，`workPerCore/workTailCore` 均分
5. TilingKey：`bit0 = (pool_mode == "max")`，`SetTilingKey`
6. Workspace：`GetLibApiWorkSpaceSize()` 仅系统保留
7. `invSampleCount`：固定 sample_num>0 时为 1/(sr²)；自适应模式填 0，kernel 侧按逐 RoI 的 n_h·n_w 计算倒数（标量）

### 3.5 Kernel 侧计算方案（PyPTO Pro）

**签名**：

```python
@pl.jit(auto_mutex=True, tiling_key=RoiExtractorTilingKey,
        datatype={"feats0": "input_dtype", "rois": "input_dtype"})
def roi_extractor(
    feats0: pl.Ptr[pl.DT_UINT8], feats1: pl.Ptr[pl.DT_UINT8], feats2: pl.Ptr[pl.DT_UINT8],
    feats3: pl.Ptr[pl.DT_UINT8], feats4: pl.Ptr[pl.DT_UINT8],
    rois: pl.Ptr[pl.DT_UINT8], index: pl.Ptr[pl.DT_UINT8],
    y: pl.Ptr[pl.DT_UINT8], tiling: RoiExtractorTiling):
```

`RoiExtractorTilingKey`：`pool_mode = TilingKeyField(bits=1, values=[0, 1])`（bit0，与 Host SetTilingKey 位序一致）。

**执行框架**：

```text
core_id = pl.get_block_idx() // pl.get_subblock_num()
work_id 序列 = [workPerCore 或 workTailCore] 个 (k, wChunkId, cBlockId) 三维线性索引
for wid in pl.range(workCount):
    k, wc, cb = 解码(workId)
    # ① 标量段：pl.getval 读 rois[k, colOffset..colOffset+4]（fp16 经标量转 fp32）
    #    level = clamp(floor(log2(scale/finestScale+1e-6)), 0, L-1)
    #    rx1/ry1/rx2/ry2/rw/rh/bw/bh、xLo/xHi/yLo/yHi、R 行界、Wc 列界
    # ② GM 视图 + UB 行加载：make_tensor(featsL_ptr + base, [Cb, Wc], [Hl·Wl, 1])
    #    for r in range(R): pl.load(T 子块 [Cb, Wc], view, [0, r·Wc])   # T 布局 [Cb, R·Wc]
    # ③ cvec = vf.arange(0, DT_INT32) * (R·Wc)   # 通道基址向量（每 work item 一次）
    # ④ for ph in range(PH):
    #      acc[Cb] ← 0（avg）/ −inf（max）
    #      for ih in range(nh):  y/y0/y1/λy 标量
    #        for pw in range(PWc 对应列块):
    #          for iw in range(nw):
    #            x/x0/x1/λx 标量 → 四角列 j00..j11 = r(y)·Wc + (x − xLo)
    #            g00 = vf.gather(T, cvec + j00, preg)   # [Cb]
    #            … g01/g10/g11
    #            val = ((1−λy)(1−λx))·g00 + …  # vf.muls/adds
    #            acc = vf.add(acc, val)（avg） / vf.max(acc, val)（max）
    #        out_row[Cb] = acc·invCnt（avg；invCnt 逐 RoI 标量 1/(nh·nw)）
    #      pl.store(y 视图 [K,C,PH,PW] 的 [k, c0+cb, ph, pwLo..]，out_row)
```

**分支策略**：pool_mode 走 TilingKey 双二进制（④ 内无 mode 分支）；hasIndex 折叠为 colOffset 标量；aligned 折叠为 off=0.5/0.0 标量；层选择 `feats_ptr = [feats0,…,feats4][lvl]` 在标量段完成（lvl 为运行期标量，用 5 路 `if lvl == l` 运行期判断 + 编译期展开选取指针，与 make_tensor 的 ptr 参数绑定）。

**边界 case 与 spec 对齐**：
- K=0：workCount=0，kernel 空转
- 退化 RoI（w≤0）：rw=max(rw,1) 标量处理，输出非全 0
- 采样点越界：标量 inside 判断（y<−1∨y>H∨x<−1∨x>W）→ 该点 val=0（avg 计分母）；边界钳制 y'=max(y,0)、y1=min(y0+1,H−1)
- batch_idx：op_api 校验 <N（越界报错）；kernel 侧标量 clamp 兜底
- fp16：行加载 fp16 tile → `pl.cast` fp32 tile → 计算 → 输出 `pl.cast` 回 fp16（`RoundMode.CAST_RINT`）

### 3.6 API 映射

| 计算步骤 | Ascend C / PyPTO API | 参数签名 | 平台验证 | dtype 支持 | 约束说明 | 替代方案 |
| --- | --- | --- | --- | --- | --- | --- |
| 行块加载 | `pl.load(dst, src, offsets)` | `(Tile, Tensor, [r,c])` | a5（dense_lightning 已用） | fp16/fp32 | dst 为 UB tile；src 为 GM make_tensor 视图，stride=(H·W,1) | 逐行多次 load |
| 通道基址 | `vf.arange(start, dtype)` | `(0, DT_INT32)` | a5（官方 API 文档） | int32 | 步长固定 ±1，需 `vf.muls` 缩放 R·Wc | 常量 tile 广播 |
| 四角取数 | `vf.gather(src, index, preg)` | `(Tile, Reg, Preg)` | a5（API 文档） | 与 tile 同 | index 为 flat 线性索引；越界 index 未定义 → 标量段保证列界 | 逐元素标量读 |
| 标量读 | `pl.getval(tensor, offset)` | `(Tensor, int)` | a5（dense_lightning 已用） | 全 | offset 为编译期常量表达式 | — |
| 插值加权 | `vf.muls/adds(src, scalar, preg)` | `(Reg, scalar, Preg)` | a5 | fp32 | 标量参与逐 lane | vf.mul+vf.full |
| 累加 | `vf.add(r0, r1, preg)` / `vf.max(r0, r1, preg)` | `(Reg, Reg, Preg)` | a5 | fp32 | — | reduce 链 |
| dtype 转换 | `pl.cast(out, src, mode=CAST_RINT)` | `(Tile, Tile)` | a5（官方 API 文档） | fp16↔fp32 | out tile dtype 决定目标 | vf.muls_cast |
| 输出写回 | `pl.store(dst, src, offsets)` | `(Tensor, Tile, [k,c,ph,pw])` | a5 | fp16/fp32 | y 视图 stride 由 make_tensor 给出 | — |
| 运行期分支 | `if cond:`（pl.getval 标量） | — | a5（dense_lightning `if core_id < …` 先例） | — | 分支条件须为运行期标量 | TilingKey 分模板 |

### 3.7 API 验证记录

| API 名称 | 官方文档路径或来源 | 通配符搜索结果 | 验证状态 | 备注 |
| --- | --- | --- | --- | --- |
| pl.load/store/make_tensor | pypto_pro/language/_api.py（本机 CANN 9.2.0 安装包源码 docstring） | dense_lightning_indexer_softmax_lse_v2.py L839/977 | 已验证（仓内先例+包内源码） | — |
| vf.gather | pypto_pro/language/_vf_api.py L1029 | 仓内无先例 | **待验证**（a5 编译+单测首轮确认；a2a3 待 910B 验证） | Tile→Reg 逐元素模式 |
| vf.arange / vf.astype | _vf_api.py L1000 / L705 | roi_extractor CCE 编译通过 | **已验证** | astype 承担标量/向量 fp32→int32 转换（mod 与 int() 均不可用） |
| vf.muls/adds/max/add/mul/sub/mins | _vf_api.py | flash_attn_grad + roi_extractor | 已验证 | 结果型 vf 操作必须赋值形式（禁止嵌套） |
| pl.cast | _api.py L449 | flash_attn_grad cast 链 | 已验证 | RoundMode.CAST_RINT |
| TilingKeyField | pypto_pro/runtime/tilingkey.py | dense_lightning DenseLiLseTilingKey | 已验证 | 位序与 C++ SetTilingKey 手工对齐 |

### 3.8 数据流设计

```text
GM rois ──pl.getval(标量)──► 标量坐标/层级/边界 (寄存器标量)
GM featsL ──pl.load×R行──► UB T[Cb, R·Wc] (fp16 时先 cast 出 T32)
UB T ──vf.gather×4/采样点──► g00..g11[Cb]
        ──vf.muls/adds 加权──► val[Cb] ──vf.add/max──► acc[Cb]
acc ──×invCnt──► out_row[Cb] ──pl.store──► GM y[k, c0..c0+Cb, ph, pwLo..pwHi]
```

单 work item 内 MTE2（行加载，一次批量 R 行）与 VF（采样插值）串行；跨 work item 由 auto_mutex tile group 轮转双缓冲（见 3.9）。

### 3.9 内存管理

| 资源 | 用途 | 大小（fp32 为例） | 决策 |
| --- | --- | --- | --- |
| UB T | 行块缓存 | `Cb · R · Wc · dtypeSize` ≤ 150KB | `make_tile_group` 双缓冲（mutex_ids 轮转 2 块） |
| UB Tout | 输出行缓冲（fp16 目标时 cast 暂存） | `Cb · PW · dtypeSize` ≤ 64·56·4 ≈ 14KB | 单缓冲（store 后复用） |
| UB Tidx | 通道基址向量 | `Cb · 4B` = 256B | 单缓冲，work item 内常量 |
| L1 | 不使用 | 0 | — |
| Workspace | 系统保留 | `GetLibApiWorkSpaceSize()` | 无自定义 |
| 寄存器 | acc/g00..g11/权重标量 | ~16 × 256B/寄存器 | 编译器分配 |

fp16 输入：T 加载为 fp16 后立即 cast 到 fp32 侧 tile（T32），Wc 预算按 4B 计；输出前 cast 回 fp16（Tout_fp16）。

### 3.10 UB 容量验证

| 平台 | 可用 UB | Buffer 总和公式 | 验证结论 |
| --- | --- | --- | --- |
| DAV_3510 (950PR) | 248KB | Cb·R·Wc·4·2(双缓冲) + Cb·PW·4 + 256B ≤ 180KB（tiling 预算） | PASS（Wc 由 tiling 反推：Cb=64, R=9, Wc≤76；Cb=32, R=9, Wc≤152） |
| DAV_2201 (910B) | 192KB（向量可用 184KB） | 同上（统一 180KB 预算 → 双端一致） | PASS（Wc 上界同左，保守取 180KB） |

`Wc` 在 Host Tiling 按公式 `Wc = clamp(floor((180KB − Tout − 256B) / (2·Cb·R·4)), 1, Wl)` 推导，逐层 clamp 到该层 W_l。

## 4. 性能优化

### 4.1 并行策略

- **核间**：work item = (k, wChunkId, cBlockId) 三维线性均分到 aiv 核（`workPerCore/workTailCore`）；RoI 数 K 大时负载均衡良好
- **核内**：通道向量化（Cb lanes）+ 采样点标量循环；可选后续优化：pw 维向量化（等距 x 网格 conv 化）
- **L2 亲和（v2 方向）**：按层级聚拢 work item（同层 RoI 连续调度），提升 L2 命中

### 4.2 流水线设计

- 行加载（MTE2）与采样计算（VF）在 work item 间通过 `make_tile_group + auto_mutex` 双缓冲轮转重叠
- 无 Cube 参与无需多级流水；同步开销集中在 tile group mutex（框架自动）

### 4.3 性能目标与基线

需求未给出明确性能基线。验收起点：功能正确 + 对比基线（mmcv NPU 路径 = GE 内置 ROIAlign AICPU + Python 多层级循环；CANN 9.2.0 闭源 aclnnRoiAlign AICPU 单层路径）有正向收益；典型 shape（K=512/1000, C=256, 7×7, FPN 4 层）GM 流量 ≈ K·C·(PH·sr+2)·Wspan·4B，L2 复用后 ~0.1-0.4ms 量级。

## 5. 风险评估

### 5.1 API 风险

| 风险 | 等级 | 说明 |
| --- | --- | --- |
| vf.gather flat 索引语义 | 高 | 2D tile 的线性布局假设需首轮编译+单测证实；不符则退化为逐行 1D tile gather（每通道行独立 Wc 向量） |
| vf.arange + muls 组合在 a2a3 的可用性 | 中 | vci 指令跨架构存在，但 910B 实机未验证 |
| OPTIONAL_INPUT nullptr 传递 | 低 | dense_lightning seq_used_q 模式先例一致 |

### 5.2 精度风险

| 风险 | 等级 | 应对 |
| --- | --- | --- |
| fp16 输入插值累加误差 | 中 | 全链 fp32 累加（AP-007），仅 IO fp16；ST 对拍 fp64 golden（1e-3/1e-3） |
| 越界/边界钳制与 torchvision 差异 | 中 | 边界规则逐条对齐（y<−1∨y>H→0；y' =max(y,0)；y1=min(y0+1,H−1)）；golden.py 58 项已验证 |
| log2 层级映射 fp16 精度 | 低 | 层级计算全程 fp32 标量 |

### 5.3 性能风险

- 采样点逐点标量循环（pw·iw·ih·ph ≈ 49-784 次/work item）可能 VF 占用偏低 → v2 方向：pw 向量化（等距网格）
- wChunk 切碎导致行加载次数放大 → tiling 预算尽量放大 Wc

### 5.4 应对措施

1. 首轮编译后跑 UT 单点用例验证 vf.gather/arange 语义，不符即切换逐行 gather 备选方案
2. ST 全量对拍 golden.py（NumPy 双路径）+ torchvision 逐层拼装参考
3. 910B 二进制生成（--soc=ascend910b）验证 a2a3 编译；实机验证留待设备

## 6. 交付件清单

| 交付件 | 路径 | 状态 |
| --- | --- | --- |
| GEIR 原型 | op_graph/roi_extractor_proto.h | 本迭代 |
| OpDef | op_host/roi_extractor_def.cpp | 本迭代 |
| 信息库 | op_host/config/ascend950/{binary.json, simplified_key.ini} | 构建生成 |
| infershape | op_host/roi_extractor_infershape.cpp | 本迭代 |
| tiling | op_host/roi_extractor_tiling.{h,cpp} | 本迭代 |
| kernel（PyPTO Pro） | op_kernel/roi_extractor.py + metadata.h | 本迭代 |
| aclnnAPI | op_host/op_api/aclnn_roi_extractor.{h,cpp} | 本迭代 |
| torch_extension | torch_extension/ | 本迭代 |
| UT | tests/ut/op_host/test_roi_extractor_tiling.cpp、tests/ut/op_api/test_aclnn_roi_extractor.cpp | 本迭代 |
| examples | examples/test_aclnn_roi_extractor.cpp | 本迭代 |
| 文档 | docs/{spec.yaml, DESIGN.md, MDE.md, PLAN.md}、README.md | 本迭代 |
| fusion_pass / TF 插件 | — | 不交付 |

## 7. 迭代规划

本算子采用单迭代全量交付（iteration_count = 1），详见 PLAN.md。

| 迭代 | 目标 | 代码开发 | UT 开发 | ST 用例 |
| --- | --- | --- | --- | --- |
| 迭代一 | 全功能实现 | kernel（avg/max 双 TilingKey、fp16/fp32、L≤5、[K,5]/[K,4]+index）+ host 全量 + aclnn + torch_extension | tiling UT（shape/属性/TilingKey 分支全覆盖）+ op_api UT | L0+L1（golden 对拍：层级/aligned/sr/max/非方形/空 RoI/越界） |

关键依赖：穿刺任务 probe_gather_semantics（vf.gather flat 索引 + vf.arange 缩放语义）在主线开发前优先验证，Dtype 覆盖 fp16/fp32，Memory Strategy 为 UB 双缓冲（§3.9）；失败则切换逐行 gather 备选方案（§5.1/§5.4）。

## 8. Design Contract

| 约束类别 | DESIGN.md 来源 |
|---------|---------------|
| Programming Model | §3.1（Vector/PyPTO Pro，禁用 RegBase/Cube/SSBuf 保 910B 兼容） |
| Compute Chain | §3.5, §3.6（标量坐标段 → 行加载 → vf.gather 插值 → avg/max 累加 → store） |
| API Constraints | §3.6, §3.7（vf.gather/arange 待验证项必须首轮 UT 证实，否则启用替代方案） |
| Precision Strategy | §4.2, §5.2（fp32 累加、边界规则 torchvision 对齐、容差 1e-5/1e-3） |
| Memory Strategy | §3.9, §3.10（UB 预算 180KB 双端一致、tile group 双缓冲、无自定义 workspace） |
