# LayerNormGradV3 精度问题分析

## 1. 现象

- **用例**: case_id=0, input_shape=[128, 256], normalized_shape=[256], dtype=float16
- **结果**: 编译通过，精度未通过
  - avg_abs_error=0.0199, avg_rel_error=0.0525
  - **max_abs_error=11.82, max_rel_error=350.25** → 存在少量元素误差极大，说明存在逻辑/实现 bug，而非单纯 float16 舍入

## 2. Tiling 路径

日志显示：

```
LayerNormGradV3SingleReadTiling: col value(=256) is not support now, please check.
LayerNormGradV3Transpose Template only support colSize <= 64 and not 32B align, colSize: 256
Do general op tiling success priority=3000
```

即：

| 模板            | 优先级 | 条件                     | 本用例 col=256      |
|-----------------|--------|--------------------------|---------------------|
| SingleRead      | 1000   | 768 ≤ col ≤ 12176        | 不满足 (256 < 768)  |
| Transpose       | 2000   | col ≤ 64 且非 32B 对齐   | 不满足 (256 > 64)   |
| **Common**      | 3000   | col ≤ 768                | **命中**            |

因此当前 [128, 256] 走的是 **Common（通用）tiling + Common kernel**，而不是为大数据量优化的 SingleRead/Transpose 路径。

## 3. 可能原因

### 3.1 Common 路径覆盖“中间”col，易暴露实现细节

- SingleRead 只支持 **col ≥ 768**，Transpose 只支持 **col ≤ 64**。
- **64 < col < 768**（如 256）只由 Common 一条路径承担，该路径实现复杂（多 Phase、LastReduceSum 分支、对齐与尾块处理），更容易在特定 (row, col) 下出现精度或越界问题。

### 3.2 Common kernel 中 LastReduceSum 的 col>64 分支

- `layer_norm_grad_v3_common.h` 中 `LastReduceSum` 在 **colAlignV > 64** 时走 else 分支（约 728–756 行）。
- 该分支使用 `DataCopy` + 多段 `Add` 做按行 reduce，涉及：
  - `blockLen = COMMON_CONSTANT_EIGHT`（8）在 DataCopy 中的单位（字节 vs 元素）需与 `Add` 的 `COMMON_B32_REPEAT_SIZE`（64）一致，否则会导致错误归约或越界。
  - `srcStride`、`dstStride` 与 `blockCount = curRowsNum` 的配合若不当，会在某些 row/col 下产生错误或未定义行为。
- 建议重点核对：col=256 时该分支的 DataCopy 与 Add 的步长、元素个数是否与“每行 256 个 float 的 reduce”一致。

### 3.3 参考与误差阈值

- 参考实现：`torch.ops.aten.native_layer_norm_backward`（或 fallback 手写公式）。
- 验证：`prepare_inputs.py` 中 float16 使用 max_abs_error=0.02, max_rel_error=0.02；当前 **max_rel_error=350** 远超阈值，说明存在明显实现错误，而非可接受的舍入误差。

## 4. 已修复：SingleRead 中 broadcast 索引错误（2025-02-09）

**根因**：`layer_norm_grad_v3_single_read.h` 中 `BinElemWithInlinedLastBrcFP32` 对 broadcast 源的索引错误。

- `BlockBroadcast(buffer6, buffer4, curRowsNum)` 的语义是：`(curRowsNum+7)/8` 个块，每块 1 个源元素重复 8 次，得到 **curRowsNum 个元素**，且 **第 i 行的值在 `dst[i]`**（即 `buffer6[i]`）。
- 原代码使用 **`src1[i * B32_BLOCK_SIZE]`**（即 `src1[i*8]`）作为第 i 行的 broadcast 源：
  - 行 0 用 `src1[0]` ✓
  - 行 1 用 `src1[8]`（应是行 8 的值）✗
  - 导致 mean/rstd 错行，dx/dgamma/dbeta 出现大误差甚至 inf。

**修改**：将两处 `src1[i * B32_BLOCK_SIZE]` 改为 **`src1[i]`**，使第 i 行使用第 i 个 broadcast 值（与 BlockBroadcast 布局一致）。

## 5. 建议方案（若仍走 Common 路径时参考）

### 方案 A：扩展 SingleRead 支持 col=256（推荐，改动小）

- 目的：让 [128, 256] 走更成熟、为“大 col”设计的 SingleRead 路径，避免 Common 在“中间 col”上的潜在问题。
- 修改：在 `layer_norm_grad_v3_single_read_tiling.cpp` 中，将 `LNG_MIN_COL` 从 **768** 降为 **256**（或 128），使 256 落在 SingleRead 的 col 范围内。
- 注意：需确认 SingleRead kernel 内部没有对“col ≥ 768”的假设（如 UB 估算、循环边界）；若仅 tiling 限制可放宽，则只改 tiling 即可。

### 方案 B：扩展 Transpose 支持 col=256

- Transpose 当前限制 col ≤ 64，若扩展到 256 需要较大改动（模板设计、UB、寄存器等），一般不作为首选。

### 方案 C：修复 Common kernel 中 col>64 的 LastReduceSum

- 在保持当前 tiling 不变的前提下，专门针对 **colAlignV > 64** 分支：
  - 核对 DataCopy 的 blockLen/blockCount/srcStride/dstStride 与“每行 colAlignV 个 float”的语义一致；
  - 核对与 `Add` 的块大小、repeat 数是否匹配；
  - 增加单测：仅 col=256（及 128、384 等）的 Common 路径，对比 PyTorch ref，定位是 pd_x 还是 pd_gamma/pd_beta 出错。

## 6. 建议的下一步

1. **已做**：① SingleRead tiling 扩展 `LNG_MIN_COL=256`；② SingleRead kernel 中 `BinElemWithInlinedLastBrcFP32` 的 broadcast 索引由 `src1[i*8]` 改为 `src1[i]`。
2. **请重新跑精度**：`python3 run_multi_test.py -task_name LayerNormGradV3 -stages compile precision`。若仍失败，再排查 Common 路径或其它 shape。

## 7. 相关文件

- Tiling: `op_host/layer_norm_grad_v3_single_read_tiling.cpp`, `layer_norm_grad_v3_common_tiling.cpp`, `layer_norm_grad_v3_transpose_tiling.cpp`
- Kernel: `op_kernel/layer_norm_grad_v3_common.h`（LastReduceSum 等）
- 用例: `validation/test_cases.csv` case_id=0
