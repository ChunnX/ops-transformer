# SFA golden脚本 S2 分块计算模式 修改说明

## 修改文件

仅涉及一个文件：

```
attention/sparse_flash_attention/tests/pytest/sparse_flash_attention_golden.py
```

## 主要修改内容

### 1. 新增 S2 分块计算模式（flash-attention online softmax）

在黄金脚本中新增了两种运行模式，通过顶部新增的两个全局开关控制：

- `RUN_MODE`：`0` 表示不切分 S2（保留原有一次性计算逻辑），`1` 表示切分 S2，采用 flash-attention 的 online softmax 分块计算方式。
- `S2_BASE_SIZE`：S2 维度的分块大小（仅 `RUN_MODE == 1` 时生效），默认 `512`。

在 `t_increattention_bnsd` 中，原本一次性完成的 `bmm1 -> scale -> softmax -> bmm2` 流程被拆分为：

- `RUN_MODE == 0`：保持原逻辑不变。
- `RUN_MODE == 1`：调用新增的 `tiled_flash_attention` 函数，按 `S2_BASE_SIZE` 对 K/V 进行分块循环，逐块维护 `row_max`（行最大值）与 `row_sum`（softmax 分母），并用 `update_mul = exp(row_max_old - row_max)` 对累计输出进行在线缩放，最终得到与整段 softmax 等价的 attention 输出及 LSE（`row_max`、`row_sum`）。
- 其它取值：抛出 `ValueError`。

### 2. 新增辅助函数

- `_cast_softmax(softmax_res, q_dtype)`：将 softmax 结果按 `q_dtype`（`float16`/`bfloat16`）量化后再转回 `float`，用于贴近 kernel 实际的中间量化行为。
- `tiled_flash_attention(...)`：实现上述分块 online softmax 计算逻辑，并在 `cur_sinks` 存在时复用 sink 对应的初始 `row_max`/`row_sum`。

### 3. 数据类型精度修复

- 分块计算中的矩阵乘（`q_curr @ k_tile.T` 与 `cur_softmax @ v_tile`）改为在 `float64` 下进行，再转回 `float32`，避免长序列分块累加时的精度损失。
- 输出写回 `y` 时根据 `q_dtype` 做 `float16`/`bfloat16` 的量化再转回 `float`，与 kernel 行为对齐。
