# roi_extractor MDE 详细设计（模块级）

依据：docs/spec.yaml（11-stage PASS）、docs/DESIGN.md（SE，validate_design/completeness PASS）。
范围：PyPTO Pro kernel、Host Tiling/OpDef/infershape、aclnn L0、构建接入、UT 的实现级契约。

## 1. 模块结构（op_kernel/roi_extractor.py）

```python
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField
from dataclasses import dataclass

@dataclass
class RoiExtractorTiling:          # §2 三侧一致
    numLevels: int; roisNum: int; batchSize: int; channels: int
    pooledHeight: int; pooledWidth: int; sampleNum: int; finestScale: int
    aligned: int; hasIndex: int; colOffset: int
    spatialScale0: float; spatialScale1: float; spatialScale2: float
    spatialScale3: float; spatialScale4: float
    levelH0: int; levelH1: int; levelH2: int; levelH3: int; levelH4: int
    levelW0: int; levelW1: int; levelW2: int; levelW3: int; levelW4: int
    workPerCore: int; workTailCore: int; cBlock: int; maxSpan: int

class RoiExtractorTilingKey:
    pool_mode = TilingKeyField(bits=1, values=[0, 1])   # bit0：0=avg，1=max

@pl.jit(auto_mutex=True, tiling_key=RoiExtractorTilingKey,
        datatype={"feats0": "input_dtype", "feats1": "input_dtype",
                  "feats2": "input_dtype", "feats3": "input_dtype",
                  "feats4": "input_dtype", "rois": "input_dtype"})
def roi_extractor(
        feats0: pl.Ptr[pl.DT_UINT8], feats1: pl.Ptr[pl.DT_UINT8],
        feats2: pl.Ptr[pl.DT_UINT8], feats3: pl.Ptr[pl.DT_UINT8],
        feats4: pl.Ptr[pl.DT_UINT8],
        rois: pl.Ptr[pl.DT_UINT8], index: pl.Ptr[pl.DT_UINT8],
        out: pl.Ptr[pl.DT_UINT8], tiling: RoiExtractorTiling): ...
```

- `datatype` 映射使 fp16/fp32 共享单二进制（DESIGN §3.2）；`y` 输出 dtype 与 feats 相同，由 `input_dtype` 视图决定。
- kernel 入口仅做视图构造与分发，主体在 `_roi_extractor_inner`（普通 Python 函数，参照 dense_lightning `_inner` 先例）。

## 2. 三侧一致性契约（强约束）

### 2.1 TilingData 字段（Python dataclass ↔ C++ TILING_DATA_FIELD_DEF）

| # | Python（op_kernel/roi_extractor.py） | C++（op_host/roi_extractor_tiling.h，同名同序） | 类型 | 语义 |
|---|---|---|---|---|
| 1 | numLevels | TILING_DATA_FIELD_DEF(int64_t, numLevels) | int64 | L，非空 feats 槽位数 [1,5] |
| 2 | roisNum | int64_t roisNum | int64 | K |
| 3 | batchSize | int64_t batchSize | int64 | N（各层一致） |
| 4 | channels | int64_t channels | int64 | C |
| 5 | pooledHeight | int64_t pooledHeight | int64 | PH |
| 6 | pooledWidth | int64_t pooledWidth | int64 | PW |
| 7 | sampleNum | int64_t sampleNum | int64 | sr，0=自适应 |
| 8 | finestScale | int64_t finestScale | int64 | 层级基准 |
| 9 | aligned | int64_t aligned | int64 | 0/1 |
| 10 | hasIndex | int64_t hasIndex | int64 | 0=[K,5]，1=[K,4]+index |
| 11 | colOffset | int64_t colOffset | int64 | hasIndex?1:0 |
| 12-16 | spatialScale0..4 | float spatialScale0..4 | float | 各层缩放，未用层 1.0 |
| 17-21 | levelH0..4 | int64_t levelH0..4 | int64 | 各层 H_l，未用层 1 |
| 22-26 | levelW0..4 | int64_t levelW0..4 | int64 | 各层 W_l，未用层 1 |
| 27 | workPerCore | int64_t workPerCore | int64 | 每核 work item 数 |
| 28 | workTailCore | int64_t workTailCore | int64 | 尾核 work item 数 |
| 29 | cBlock | int64_t cBlock | int64 | 通道块 Cb |
| 30 | maxSpan | int64_t maxSpan | int64 | 行加载列预算 Wc |

C++ 侧：`BEGIN_TILING_DATA_DEF(RoiExtractorTilingDataTmp)` … `REGISTER_TILING_DATA_CLASS(RoiExtractor, RoiExtractorTilingDataTmp)`；tiling.cpp 使用 codegen force-include 的 `RoiExtractorTiling`（`context->GetTilingData<RoiExtractorTiling>()`）。**两侧字段名、顺序、类型必须逐项一致（本表为验收基准，UT 断言 sizeof/字段偏移）。**

### 2.2 TilingKey 位序（Python TilingKeyField ↔ C++ SetTilingKey）

| 位 | Python TilingKeyField | C++ 计算 | 值域 |
|---|---|---|---|
| bit0 | pool_mode = TilingKeyField(bits=1, values=[0, 1]) | `tilingKey = static_cast<uint64_t>(poolModeIsMax)` 后 `context->SetTilingKey(tilingKey)` | 0=avg，1=max |

无其他位。dtype 不占位（input_dtype 单二进制）；aligned/hasIndex/sample_num 等为运行期标量（TilingData），不产生模板分叉。

## 3. Kernel 详细算法（_roi_extractor_inner）

### 3.1 视图与工作分发

```python
core_id = pl.get_block_idx() // pl.get_subblock_num()
workCount = workPerCore if core_id < foreCores else workTailCore   # foreCores 由 blockDim 与总数推出
numCBlocks = (C + cBlock - 1) // cBlock
roisT   = pl.make_tensor(rois,  [K, 5], [5, 1], dtype=input_dtype)
indexT  = pl.make_tensor(index, [K],    [1],    dtype=pl.DT_INT32)
outBase = C * PH * PW
for wid in pl.range(workCount):
    k  = wid // numCBlocks
    cb = wid % numCBlocks
    c0 = cb * cBlock
    cbValid = min(cBlock, C - c0)                     # 尾块
    ...
```

### 3.2 标量段（每 work item）

```python
bi = pl.getval(roisT, k * 5 + colOffset + 0)          # batch_idx（fp16 输入时标量转 fp32）
x1 = pl.getval(roisT, k * 5 + colOffset + 1); y1 = ...; x2 = ...; y2 = ...
# [K,4]+index 变体：bi = pl.getval(indexT, k)，坐标从第 0 列读（colOffset=0）
scale = sqrt((x2-x1)*(y2-y1)) / roiScaleFactor        # factor>0 才除（DESIGN §1.3）
lvl = clamp(floor(log2(scale / finestScale + 1e-6)), 0, L-1)
```

层级参数取值（Python 列表 + 编译期索引）：
```python
levelHs = [tiling.levelH0, tiling.levelH1, tiling.levelH2, tiling.levelH3, tiling.levelH4]
levelWs = [tiling.levelW0, ...]; sss = [tiling.spatialScale0, ...]
ptrs    = [feats0, feats1, feats2, feats3, feats4]
```

### 3.3 坐标与采样网格（标量）

```python
off = 0.5 if tiling.aligned else 0.0
rx1 = x1 * s_lvl - off; ry1 = y1 * s_lvl - off
rx2 = x2 * s_lvl - off; ry2 = y2 * s_lvl - off
rw = max(rx2 - rx1, 1.0); rh = max(ry2 - ry1, 1.0)
bw = rw / PW; bh = rh / PH
nh = sampleNum if sampleNum > 0 else max(ceil(bh), 1)
nw = sampleNum if sampleNum > 0 else max(ceil(bw), 1)
invCnt = 1.0 / (nh * nw)                               # avg 用
```

### 3.4 主循环：per (ph, ih) 行对加载 + per (pw, iw) gather 插值

```python
acc = tile [PW, Cb] FP32 @UB_A0（make_tile_group，mutex 轮转）   # 行 pw 连续，便于 [Cb] 列更新
for ph in pl.range(PH):
    vf 清零 acc（store vf.full(0)）
    for ih in pl.range(nh):
        yy = ry1 + (ph + 0.5) * bh + (ih + 0.5) * bh / nh
        # 行界（边界规则 = spec formula）
        insideY = (yy >= -1.0) and (yy <= Hl)
        yyc = max(yy, 0.0); y0 = floor(yyc); y1r = min(y0 + 1, Hl - 1); lamY = yyc - y0
        # 该 (ph,ih) 的 x 全跨度（覆盖全部 pw/iw 采样点 ±1 列）
        xLoF = floor(rx1 + 0.5 * bw / nw - 1.0); xHiF = ceil(rx1 + (PW - 0.5) * bw + 1.0)
        span = min(xHiF - xLoF + 1, Wl)
        if span <= maxSpan:                            # 主路径：行对加载
            xLo = clamp(xLoF, 0, Wl - 1)
            T0 = tile [Cb, maxSpan] @UB_T0；T1 = tile [Cb, maxSpan] @UB_T1
            view0 = make_tensor(ptr_l + ((bi * C + c0) * Hl + y0) * Wl + xLo,
                                [cbValid, maxSpan], [Hl * Wl, 1], dtype=input_dtype)
            view1 = 同上 y1r
            pl.load(T0, view0, [0, 0]); pl.load(T1, view1, [0, 0])
            # fp16：pl.cast(T0f, T0, CAST_RINT) 至 fp32 侧 tile 后 gather（T0f/T1f @UB_T0F/UB_T1F）
            cvec = vf.arange(0, DT_INT32); cvec = vf.muls(cvec, maxSpan, preg)   # i*maxSpan
            for pw in pl.range(PW):
                for iw in pl.range(nw):
                    xx = rx1 + (pw + 0.5) * bw + (iw + 0.5) * bw / nw
                    insideX = (xx >= -1.0) and (xx <= Wl)
                    if insideY and insideX:            # 完全越界 → 贡献 0（avg 计分母）
                        xxc = max(xx, 0.0); x0 = floor(xxc); x1r = min(x0 + 1, Wl - 1)
                        lamX = xxc - x0
                        j0 = x0 - xLo; j1 = x1r - xLo
                        g00 = vf.gather(T0, vf.adds(cvec, j0, preg), preg)
                        g01 = vf.gather(T0, vf.adds(cvec, j1, preg), preg)
                        g10 = vf.gather(T1, vf.adds(cvec, j0, preg), preg)
                        g11 = vf.gather(T1, vf.adds(cvec, j1, preg), preg)
                        # val = (1-lamY)(1-lamX)g00 + (1-lamY)lamX g01 + lamY(1-lamX)g10 + lamY lamX g11
                        val = 四次 vf.muls + 三次 vf.add（fp32）
                        # acc 行 pw 更新（avg：加；max：取大——TilingKey 双二进制各一份内循环体）
                        accRow = vf.load(acc, offset=pw * Cb …)     # [Cb]
                        accNew = vf.add(accRow, val)（avg）/ vf.max(accRow, val)（max）
                        vf.store(acc, accNew, …)                      # 写回行 pw
        else:                                           # 回退路径：逐采样点四角 [Cb] GM 负载
            for pw in pl.range(PW):
                for iw in pl.range(nw):
                    ...同上坐标/inside...
                    for (cornerY, cornerX, w) in [(y0, x0, wA), (y0, x1r, wB), (y1r, x0, wC), (y1r, x1r, wD)]:
                        view = make_tensor(ptr_l + ((bi * C + c0) * Hl + cornerY) * Wl + cornerX,
                                          [cbValid], [Hl * Wl], dtype=input_dtype)
                        Tc = tile [1, cbValid]；pl.load(Tc, view, [0, 0])
                        累加 val（fp32 侧）
                    acc 更新同上
    # 输出该 ph 行：acc × invCnt（avg）
    if pool_mode_avg: accTile 乘 invCnt（vf.muls 逐行或存前统一）
    for pw in pl.range(PW):
        outView = make_tensor(out + ((k * C + c0) * PH + ph) * PW + pw,
                              [1, cbValid], [PH * PW], dtype=input_dtype)
        outTile = tile [1, cbValid] input_dtype
        pl.cast(outTile, acc 行 pw 的 fp32 tile, CAST_RINT)   # fp16 输出；fp32 直用
        pl.store(outView, outTile, [0, 0])
```

**边界规则与 spec formula 逐条对齐**：`y<-1∨y>H→0`（insideY 判定）；`y'=max(y,0)`；`y1=min(y0+1,H-1)`；`λy=y'-y0`；avg 分母 nh·nw 含越界点；退化 RoI `max(rw/rh,1)`；batch_idx 标量 clamp(bi, 0, N-1) 兜底（op_api 层已校验报错）。

### 3.5 层级指针选择（5 路运行期分支）

```python
if lvl == 0: _process_roi(feats0, levelHs[0], levelWs[0], sss[0], 0, ...)
if lvl == 1: _process_roi(feats1, levelHs[1], levelWs[1], sss[1], 1, ...)
...（共 5 路，lvl 为运行期标量；_process_roi 为普通 Python 函数，编译期各实例化一次）
```

### 3.6 Tile/UB 地址分配（fp32 主路径，tiling 预算 180KB）

| Tile | 形状 | dtype | 地址 | 说明 |
|---|---|---|---|---|
| acc | [PW, Cb] | FP32 | UB_A0 | ≤ 56·64·4 = 14KB |
| T0/T1 | [Cb, maxSpan] | input_dtype | UB_T0/UB_T1 | fp32：2·64·Wc·4；fp16 原始 + cast 副本 |
| T0f/T1f | [Cb, maxSpan] | FP32 | UB_T0F/UB_T1F | 仅 fp16 路径 |
| outTile | [1, Cb] | input_dtype | UB_OUT | 256B |

`maxSpan = clamp((180KB − acc − out − 512B) / (2·Cb·4)，16, Wmax)`（fp16 时除以 (2·4+2·4+2·2)=…保守取 fp32 公式减半）。tile group 用 `make_tile_group(addrs=…, mutex_ids=…)` 双缓冲轮转（acc 单缓冲）。

## 4. Host 侧详细设计

### 4.1 roi_extractor_def.cpp（OpDef）

```cpp
REG_OP(RoiExtractor)
    .OPTIONAL_INPUT(feats0, TensorType({DT_FLOAT16, DT_FLOAT}))   // NCHW
    .OPTIONAL_INPUT(feats1, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(feats2, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(feats3, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(feats4, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(rois, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(index, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(finest_scale, Int, 56)
    .ATTR(roi_scale_factor, Float, 0.0f)
    .ATTR(spatial_scale, ListFloat, {0.25f, 0.125f, 0.0625f, 0.03125f})
    .ATTR(pooled_height, Int, 7)
    .ATTR(pooled_width, Int, 7)
    .ATTR(sample_num, Int, 0)
    .ATTR(pool_mode, String, "avg")
    .ATTR(aligned, Bool, true)
    .OP_END_FACTORY_REG(RoiExtractor)
```

属性名/默认值与遗留 GE RoiExtractor 完全一致（req_analysis §7 接口设计）。AICore().AddConfig("ascend950")（910B 编译走 --soc=ascend910b 时由构建注入对应 config）。

### 4.2 infershape（roi_extractor_infershape.cpp）

- 输出 y：`[rois.dim(0), feats0.dim(1), pooled_height, pooled_width]`；UnknownRank 传导 rois.dim(0)/C
- dtype：与 feats 一致；index 存在时校验 rois 末维=4，否则=5

### 4.3 tiling.cpp（RoiExtractorTilingFunc）

1. `platform_ascendc` → aivNum/aicNum → `blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum)` → SetBlockDim
2. 收集非空 feats 槽位（`GetOptionalInputShape(0..4)`）：L、Hl/Wl；**非空槽必须连续在前**（否则报 shape_mismatch）；各层 dim0(N)/dim1(C) 一致校验
3. 校验清单（报错 → `ge::GRAPH_PARAM_INVALID` + OP_LOGE）：
   - dtype：feats ∈ {FP16,FP32} 且各层一致且 rois 一致（否则 dtype_not_supported）
   - rois rank=2 且末维 ∈ {4,5}；[K,4] 时 index 必须提供且 [K] INT32；[K,5] 时 index 若提供也须 [K] INT32（忽略或校验一致）
   - 属性：finest_scale≥1；sample_num∈[0,16]；pooled_height/width∈[1,56]；pool_mode∈{avg,max}；roi_scale_factor∈[0,16]；spatial_scale 长度=L 且各值∈(0,1]
4. TilingData 填充：§2.1 全字段；`cBlock = min(C, 64)`；`maxSpan` 按 §3.6 公式（用 min(a3,a5) 预算 180KB）；`workPerCore/workTailCore` 由 `totalWork = K × ceil(C/cBlock)` 对 blockDim 均分；未用层 levelH/W=1、spatialScale=1.0
5. `tilingKey = (pool_mode == "max") ? 1 : 0`；`context->SetTilingKey(tilingKey)`
6. workspace：`GetLibApiWorkSpaceSize()` 仅系统保留
7. IMPL_OP_OPTILING(RoiExtractor).Tiling(Func).TilingParse<CompileInfo>(Parse)

### 4.4 aclnn L0（op_api/aclnn_roi_extractor.cpp）

```cpp
aclnnStatus aclnnRoiExtractorGetWorkspaceSize(
    const aclTensorList *feats, const aclTensor *rois, const aclTensor *index,
    int64_t finestScale, double roiScaleFactor, const double *spatialScales,
    int64_t spatialScalesLen, int64_t pooledHeight, int64_t pooledWidth,
    int64_t samplingRatio, char *poolMode, bool aligned,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);
```

- feats 解包 ≤5 个 aclTensor（>5 报 ACL_ERROR_INVALID_PARAM）；逐个 ACLNN 装载为 feats0..feats4 槽位
- 属性/shape 校验同 §4.3（含 batch_idx 全量扫描在 device 不可行 → 由 kernel 标量 clamp 兜底 + rois[N] 越界仅告警？——不：op_api 不扫 device 数据，依赖 tiling 静态校验 + kernel clamp；spec 的"报错"case 以 UT mock 层验证 tiling 路径）
- 双段式调用 `ADD_TO_LAUNCHER_LIST_AICORE`（纯 AICore 算子，无 AICPU fallback）

### 4.5 构建接入

- `detection/CMakeLists.txt`：复制 mhc/CMakeLists.txt 的 glob 发现模式
- 根 `CMakeLists.txt`：`should_add_module("detection" ADD_DETECTION)` + `add_subdirectory(detection)`（mamba 同款 3 行）
- `op_host/CMakeLists.txt`（对齐 dense_lightning）：

```cmake
require_pypto_pro(roi_extractor)
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE roi_extractor_def.cpp)
    add_modules_sources(OPTYPE roi_extractor ACLNNTYPE aclnn)
    enable_pypto_kernel(roi_extractor)          # 用户要求项：PyPTO 接入
endif()
add_ops_compile_options(OP_NAME RoiExtractor OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
add_modules_sources(OPTYPE roi_extractor ACLNNTYPE aclnn)
add_tiling_modules()
```

## 5. UT 设计

### 5.1 tiling UT（tests/ut/op_host/test_roi_extractor_tiling.cpp）

框架：gtest + `tiling_context_faker.h` + `tiling_case_executor.h`（attention_update 先例）。用例矩阵：

| 用例 | 覆盖 | 期望 |
|---|---|---|
| 基本 fp32 4 层 [K,5] | L=4、K=8、C=256、7×7、avg | GRAPH_SUCCESS + TilingKey=0 + 字段断言 |
| max 模式 | pool_mode=max | TilingKey=1 |
| fp16 | feats/rois fp16 | 成功（datatype 单二进制） |
| [K,4]+index | hasIndex=1 | colOffset=1 |
| 非方形 | pooled 7×14 | 成功 |
| L=1 单层 | 退化 roi_align | 成功 |
| K=0 | 空 RoIs | 成功（workCount=0） |
| 槽位断续 | feats1 空 feats2 非空 | GRAPH_PARAM_INVALID |
| dtype 不一致 | feats fp16 + rois fp32 | GRAPH_PARAM_INVALID |
| 属性越界 | sample_num=-1 / pooled=64 / pool_mode=median / spatial_scale 长度≠L | GRAPH_PARAM_INVALID |

### 5.2 op_api UT（tests/ut/op_api/test_aclnn_roi_extractor.cpp）

- 单点数值用例（fp32：L=1 N=1 C=64 H=W=16 K=2 pooled 2×2 sr=1 aligned，对照 NumPy golden 固定值）
- TensorList 解包/越界（>5 层报错）
- 空指针（feats=null → ACL_ERROR_INVALID_PARAM）

### 5.3 编译验证目标

- `bash build.sh --ophost_test --ops=roi_extractor --noexec`（UT 编译）
- `bash build.sh --ophost --opkernel --soc=ascend950 --ops=roi_extractor`（950 二进制）
- `bash build.sh --opkernel --soc=ascend910b --ops=roi_extractor`（910B 二进制，a2a3 编译验证）

## 6. MDE 自审（范式偏离检查）

| 检查项 | 结论 |
|---|---|
| 与 spec formula 语义一致 | PASS：边界规则/层级映射/avg-max/退化 RoI/invCnt 逐条对应（§3.4 注） |
| 三侧一致（dataclass/TILING_DATA_FIELD/TilingKey） | PASS：§2 表为验收基准，UT 断言 |
| UB 预算双端一致 | PASS：统一 180KB 保守预算，maxSpan 动态推导（§3.6） |
| 910B 兼容 | PASS：纯 vf/pl API，无 RegBase/Cube/SSBuf；a2a3 待编译验证（§5.3） |
| PyPTO API 已验证/待验证项 | vf.gather flat 索引、vf.arange+muls 组合为首轮 UT 必验项（DESIGN §3.7）；失败启用回退路径（§3.4 else 分支已设计） |
| 空 K/退化/越界边界 | 与 spec boundary_conditions 一一对应（§5.1 用例矩阵） |


## 7. 实现验证结论（开发完成后回填）

| 项 | 结论 |
|---|---|
| vf.gather flat 索引（[Cb, SPAN] tile，cvec·SPAN + j） | 已验证：CCE 编译通过（a5） |
| vf.arange + vf.muls 通道基址向量 | 已验证 |
| vf.astype fp32→int32（寄存器级 vcvt） | 已验证；标量 floor/ceil 统一走 floor_vf（astype + getval 往返 INT32 scratch tile）——`x % 1.0`（mod-floor）在 C++ 后端非法（float %），int() 在 inline helper 的 full-trace 阶段被拒，往返模式双端合法 |
| pl.load order=[1,3] 跨轴（C,W）窗口加载 | 已验证；[CB, SPAN] tile 合法，[CB, 1] 列向量 tile 触发布局 static_assert——fallback 改为 2 列窗口 + 相对坐标复用 accumulate_*_vf |
| pl.store 列输出 | 已验证；order 必须递增，输出视图改为置换轴 [K,PH,PW,C]（stride 置换），[1, CB] tile 默认 order 即可 |
| 穿刺结果 | 成功：probe_gather_semantics 通过（codegen + CCE 全链路） |
| **910B 支持情况（最终结论）** | **代码路径纯 vf 兼容 a3/a5，但当前 CANN 9.2.0 的 PyPTO 二进制编译器不支持 compute unit 'ascend910b'（RuntimeError: PyPTO binary compilation does not support compute unit 'ascend910b'）——仅 Ascend950 系列可产出交付二进制；910B 需等待 PyPTO 二进制路径支持后启用（def.cpp 已预留单行 AddConfig 扩展点）** |
| UT | tiling UT 15/15 PASS；op_api UT 10/10 PASS（含 TilingKey=1 max 分支、[K,4]+index、空 RoIs、非法属性拒绝） |
| 二进制 | ascend950/bin/roi_extractor/ 下 2 个 .o（fp32/fp16 两个 dtype 变体，各含 tilingKey 0/1 avg/max kernel），opParaSize=208 与 TilingData 一致 |
