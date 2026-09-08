# Cumprod 算子设计（DESIGN.md）

> Stage 4 输出。平台 Ascend 950PR/DT（a5, DAV_3510），纯 Vector 算子（无 Cube）。
> 参考范式：`dense_lightning_indexer_softmax_lse_v2`（pypto-pro + TilingKey + enable_pypto_kernel）。

## 1. 总体结构

```
custom_transformer 包
└── attention/cumprod/
    ├── op_kernel/cumprod.py                  # pypto-pro kernel（@pl.jit）
    ├── op_host/
    │   ├── cumprod_def.cpp                   # OpDef（GEIR + aclnn 生成源）
    │   ├── cumprod_infershape.cpp            # y.shape = x.shape, y.dtype = x.dtype
    │   ├── cumprod_tiling.h/.cpp             # TilingData 注册 + TilingFunc
    │   ├── CMakeLists.txt                    # require_pypto_pro + enable_pypto_kernel
    │   └── config/ascend950/                 # binary.json（3 dtype 变体）+ simplified_key.ini
    ├── tests/ttk/                            # TTK 用例（GE + aclnn）+ golden 插件
    └── docs/aclnnCumprod.md
```

## 2. 布局参数化（对齐竞品 [dimx, dimy, dimz] 三维化）

任意 rank/axis 归一化为三维视图 `[outer, scan_len, inner]`：

- `outer = prod(shape[:dim])`（扫描轴之前的维度积）
- `scan_len = shape[dim]`（扫描轴长度）
- `inner = prod(shape[dim+1:])`（扫描轴之后的维度积）

`inner == 1` ⇔ 扫描线在内存中连续（含 1D 与末轴，及尾部维度全 1 的中间轴）。

## 3. TilingData（对应 kernel `@dataclass CumprodTiling`）

| 字段 | 类型 | 含义 |
|---|---|---|
| outer | int64 | dimx |
| scan_len | int64 | dimy |
| inner | int64 | dimz |

host 侧 `BEGIN_TILING_DATA_DEF(CumprodTilingData)`（int64_t ×3，顺序一致）+ `REGISTER_TILING_DATA_CLASS(Cumprod, CumprodTilingData)`；TilingFunc 用 force-include 的生成头 `CumprodTiling`（`GetTilingData<CumprodTiling>()`）。

## 4. TilingKey（对应 kernel `class CumprodTilingKey`）

| 字段 | bits | values | 位偏移 | 语义 |
|---|---|---|---|---|
| is_last_axis | 1 | [0, 1] | bit0 | 1: inner==1（连续扫描线，chunk 内 Hillis-Steee 扫描）；0: inner≥2（inner 维向量化 + 扫描轴串行） |

host 打包（与 dense 相同的手工位布局）：`tilingKey = is_last_axis`（单字段 values=[0,1] 时编码值==实际值）。

## 5. Kernel 算法

### 5.1 分支 1：is_last_axis = 1（连续扫描线）

视图 `[outer, scan_len]`（行 stride = scan_len）。行（扫描线）按核循环划分：
`for line in pl.range(block_idx, outer, block_num)`。

每条扫描线按 `VEC=64`（fp32 寄存器 lane 数）分 chunk，chunk 间用标量 carry（fp32）串行传播：

1. `pl.load(x_tile[1,VEC] input_dtype, validshape=[1,cnt]) ← x[line, off]`（cnt = min(scan_len-off, VEC)）
2. `pl.cast(state_tile_fp32[1,2*VEC] ← x_tile)`（fp32 直通时 `pl.move`）
3. VF 内 Hillis-Steele 扫描（log2(VEC)=6 步）+ carry 乘：
   ```
   acc = vf.load_align(state, 0); acc = acc * full(carry)
   for s in (1,2,4,8,16,32):            # 编译期展开
       if cnt > s:                        # 运行时分支
           vf.store_align(state, acc, preg_cnt)      # 只写 [0,cnt)
           vf.mem_bar(VST_VLD)
           sh = vf.load_align(state, s)              # sh[i] = state[i+s]
           part = vf.mul(acc, sh, preg_step)         # preg_step = update_mask(cnt-s)，ZEROING
           acc = vf.select(part, acc, preg_step)     # 未激活 lane 保持原值
   vf.store_align(state, acc, preg_cnt)
   ```
4. `carry = pl.getval(state_tile, cnt-1)`（保持 fp32 精度跨 chunk）
5. `pl.cast(out_tile input_dtype ← state_tile)`（CAST_RINT）→ `pl.store(y[line, off])`

正确性要点：
- 步进 s 只影响 lane i∈[0,cnt-s)（i+s≤cnt-1，state 内均为本 chunk 新值）；lane [cnt-s,cnt) 由 select 保持 → 数学上等价于顺序前缀积，仅结合顺序并行化（与竞品 CUDA Sklansky 同口径）。
- state tile 尺寸 2*VEC：offset=s 的 load 最远读到 s+VEC-1 < 1.5*VEC ≤ 2*VEC，不越 tile。
- VST→VLD 同 pipe RAW 由 `vf.mem_bar(VST_VLD)` 消除；MTE2/V/MTE3 跨 pipe 依赖由 `auto_mutex=True` tile group 互斥自动管理。
- cast 写 state 与 VF 读 state 之间：VF 函数入口 `vf.mem_bar(VST_VLD)` 覆盖。

### 5.2 分支 0：is_last_axis = 0（inner 向量化）

视图 `[outer, scan_len, inner]`（stride = [scan_len*inner, inner, 1]）。工作项 = outer × ceil(inner/VEC)：
`for wi in pl.range(block_idx, total, block_num)`，`i = wi // inner_chunks, c = wi % inner_chunks`。

每个工作项：acc 标量（fp32）沿 scan_len 串行累乘，向量维为 inner 的 [1,VEC] chunk：

```
for j in pl.range(scan_len):
    pl.load(x_tile, x, [i, j, off])          # validshape [1, cnt]
    cast/move → state_tile_fp32
    scan_step_vf(state_tile, carry, cnt):    # acc = full(carry)*x（preg_cnt 掩码）
        vf.store_align(state, acc_reg, preg_cnt)
    carry = pl.getval(state_tile, cnt-1)
    cast/move → out_tile
    pl.store(y, out_tile, [i, j, off])
```

（与 PyTorch CUDA `scan_outer_dim`（线程绑内行串行累乘）同构，inner 维由 64-lane 向量承载。）

### 5.3 数值口径实现

- fp16/bf16：load → `pl.cast` 拓宽 fp32 → 全程 fp32 累乘 → `pl.cast(CAST_RINT)` 逐元素写回。carry 在 fp32 域跨 chunk/步保持（`pl.getval` 于 fp32 state tile）。
- fp32：`pl.move` 直通，无 cast。
- dtype 分发：`@pl.jit(datatype={"x": "input_dtype"})` + def.cpp 三 dtype → 三个 fatbin（simplifiedKey d=0/1/2）。

## 6. Host Tiling 策略（cumprod_tiling.cpp）

1. 校验：x 非空指针、rank ∈ [1,8]、dim 归一化（负轴 +rank）后 ∈ [0, rank)、y 与 x 同形（GE 侧 infershape 已保证，防御性校验输出 shape）。
2. 计算 outer / scan_len / inner；写入 TilingData。
3. `is_last_axis = (inner == 1)`；`SetTilingKey(is_last_axis)`。
4. `blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum)`（对齐 dense/sparse 两个 pypto 参考算子；多余核空转 range 循环自然退出）。
5. workspace：仅 `GetLibApiWorkSpaceSize()` 系统预留（kernel 无自定义 workspace）。
6. 空Tensor：numel==0 时直接置 blockDim=1、TilingData 全 0，kernel 循环体空转。

## 7. 精度与测试策略

- golden：`torch.cumprod`（fp16/bf16 先 fp32 promote，等价 CPU acc_type 语义）——`cumprod_golden.py`（自检通过）+ TTK TestSpec 插件（Kernel/GEIR: `Cumprod`；ACLNN: `aclnnCumprod`）。
- 容差：fp32 1e-4 / fp16 2e-3 / bf16 4e-3（CSV precision_tolerances + 插件 tolerance 双保险）。
- 输入域：长链（scan_len>32）取 [0.98, 1.02] 防系统性下溢；短链取 [-1.5, 1.5] 覆盖负数/符号翻转。
- 通路：TTK `geir`（GE 图编译执行）+ TTK `aclnn`（aclnnCumprod C API），均默认静态 shape 编译。

## 8. 风险点与对策

| 风险 | 对策 |
|---|---|
| 与内置 Cumprod（AICPU，950 未支持）重名冲突 | 参考 flash_attention_score 先例（custom 包同名覆盖）；GEIR/aclnn 实测验证，失败则改名 CumprodV2 |
| 自定义算子同名覆盖后 GE 静态编译仍选到内置 op | 950 ops-info 无内置 Cumprod AiCore/AiCPU 条目（已核实），冲突面实际不存在 |
| vf.mul 不支持 MERGING | 已规避：ZEROING + select 组合 |
| V pipe 内同 tile 读写乱序 | VF 入口/步进间 `vf.mem_bar(VST_VLD)` |
| 混合 pipe（MIX 1:2）下 block_idx 语义 | 采用 `range(block_idx, total, block_num)` 通用划分，任何 workers≥1 均正确；实测以二进制 json taskRation 确认 |
| 1D 超长链单核串行 | v1 已知性能边界（正确性优先），Stage 7 可扩展分段+两遍扫描 |
| fp16 长链下溢 0 | 数学语义；用例输入域收敛到近 1 区间规避比对抖动 |
