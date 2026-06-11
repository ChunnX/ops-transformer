# moe_token_permute_with_routing_map 需求分析文档

## 1. 算子概述

### 1.1 功能描述

`moe_token_permute_with_routing_map` 是 MOE（Mixture of Experts）场景下的 Token 重排算子。根据路由映射（routing_map）将输入的 token 按选中的专家重新排列，使得分配到同一专家的 token 连续存放。

### 1.2 数学描述

给定：
- `tokens`: `[num_tokens, hidden]` — 输入 token
- `routing_map`: `[num_experts, num_tokens]` (bool/int8) — 稀疏的 token 到 expert 的映射

输出：
- `permuted_tokens`: `[num_out_tokens, hidden]` — 重排后的 token
- `sorted_indices`: `[num_out_tokens]` — 排序后的索引

参考 PyTorch 逻辑 (`mtprm.py:77-89`):

```python
routing_map = routing_map.bool().T.contiguous()  # [num_experts, num_tokens]
token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
sorted_indices = token_indices.masked_select(routing_map)  # 1-D, 按expert分组
sorted_indices2 = torch.sort(sorted_indices.float(), stable=True)[1]
permuted_input = tokens.index_select(0, sorted_indices)
```

### 1.3 输入输出规格

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| tokens | T | [num_tokens, hidden] | 输入 token |
| routing_map | bool/int8 | [num_experts, num_tokens] | 路由映射 |
| probs | T (optional) | [num_tokens, num_experts] | 路由概率（可选） |
| permuted_tokens | T | [num_out_tokens, hidden] | 输出重排 token |
| permuted_probs | T (optional) | [num_out_tokens] | 输出重排概率 |
| sorted_indices | int32 | [num_out_tokens] | 排序索引 |

### 1.4 支持数据类型

- tokens/permuted_tokens: float16, bfloat16, float32, int32, int64
- routing_map: int8, uint8, bool
- probs/permuted_probs: float16, float32

## 2. 当前缺陷

### 2.1 问题描述

当 `drop_and_pad=false` 时，当前实现假设 `num_out_tokens = topK * num_tokens`，即每个 token 必须恰好选中 topK 个专家。但实际上 `routing_map` 中每行可能是任意数量的 True 值（≤ topK）。

**当前代码硬编码**（`moe_token_permute_with_routing_map_tiling.cpp:218`）：
```cpp
numOutTokens = totalLength;  // totalLength = numTokens * topK
```

这导致：
1. 如果某些 token 选中的专家少于 topK 个，`sorted_indices` 实际长度小于 `numOutTokens`
2. 多余的输出位置是未初始化的数据
3. `tokencopy` 会读取越界的 indices，导致错误

### 2.2 影响范围

- **maskselect**: 需要输出实际选中的 token 总数（不是 topK * numTokens）
- **sort**: 实际排序的元素数量可能小于 topK * numTokens，需要重新计算 tiling 参数
- **tokencopy**: 需要知道实际的 num_out_tokens 以正确搬移

## 3. 功能需求

### 3.1 maskselect — 新增总输出 token 计数输出

**需求描述**：在 maskselect 完成后，输出实际的 `num_out_tokens`（即 `routing_map` 中 True 的总数）。

**实现要点**：
- maskselect 阶段各核已计算自己处理的 offset，汇总可得 total 非零元素个数
- 将此值写入 workspace 或新增输出 tensor，供后续 sort 和 tokencopy 使用

### 3.2 sort — 支持非对齐的排序长度

**需求描述**：当实际 token 数量 != topK * numTokens 时：
1. 强制使用多核排序（不再使用单核排序分支）
2. 重新计算 sort 的 tiling 参数（基于实际 token 数量而非 topK * numTokens）
3. 归并排序最后单核输出时，用 -1 补齐每个 token 的 topK 个专家

**补齐逻辑详解**：

假设：3 个 token，topK=3，但实际选中的专家索引为 `[0,2,0,4,1,5,1,6,1,7]`（即 token0 选中 expert0、expert2；token1 选中 expert4、expert5；token2 选中 expert6、expert7）。

maskselect 输出 sorted_indices: `[0, 0, 1, 1, 1, 2, 2]`（7 个元素，分别对应 token0×2, token1×3, token2×2）

排序后 Extract 前：`[0,2, 0,4, 1,5, 1,6, 1,7]`（按 token index 排序的 (token, position) 对）

需要补齐为：`[2, 4, -1, 5, 6, 7, -1, -1, -1]`（每个 token 补齐到 topK=3 个，不足的位置填 -1）

**补齐策略**：
- **策略1（推荐）**：Scalar 操作。Extract 后，遍历当前 UB 内的数据，判断每个 token 是否有 topK 个输出，不足的位置写入 -1。
- **策略2**：创造一条补齐值的 UB（填满 -1），与原始 UB 再次归并排序。

### 3.3 tokencopy — 跳过 -1 位置

**需求描述**：在 tokencopy 阶段，当 sorted_indices 中某个位置的值为 -1 时，跳过不搬出。

**实现要点**：
- 在 `MoeindexCopyOp::CopyOut()` 和 `MoeindexCopySplitDOp::CopyInAndOut()` 中
- 在遍历 `topKId` 时，检查 `indicesValue` 是否等于 -1
- 若等于 -1，跳过该位置的 DataCopyPadCustom 调用

## 4. 非功能性需求

### 4.1 精度要求

- 排序精度：sort 本身为稳定排序，补齐 -1 不引入精度误差
- token copy 精度：与原有逻辑一致

### 4.2 性能要求

- 新增的补齐逻辑应尽量高效，避免大幅增加 UB 占用或计算时间
- 补齐后的 sorted_indices 长度仍为 topK * numTokens，tokencopy 的搬移量不变（只是部分位置跳过）

### 4.3 兼容性

- 不影响现有 `drop_and_pad=true` 的 padded 模式（tiling key 9/19）
- 当所有 token 恰好选中 topK 个专家时，行为与原逻辑完全一致
- 支持现有所有数据类型

## 5. 芯片类型

- 目标芯片：Ascend 910B/910C 系列
- arch: 35（DAV_3510）

## 6. 可行性评估

| 评估项 | 结论 |
|--------|------|
| maskselect 输出 total count | 可行：各核已计算 offset，汇总即可 |
| sort 多核分支 | 可行：多核排序框架已有，复用 Tiling4VBSCompute + Tinlig4VBSMultiCoreCompute |
| -1 补齐逻辑 | 可行：参考策略1，在 Extract 后做 scalar 填充 |
| tokencopy 跳过 -1 | 可行：在已有 topKId 循环中加条件判断 |
| 向后兼容 | 可行：当 total == topK * numTokens 时行为不变 |