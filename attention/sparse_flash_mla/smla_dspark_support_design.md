# SparseFlashMla 支持 DSpark 特性设计文档

## 1. 背景

DeepSpec 是 DeepSeek 开源的 speculative decoding 训练与评测工程，覆盖 DSpark、DFlash、Eagle3 等 draft model 算法。vLLM-Ascend PR [vllm-project/vllm-ascend#11196](https://github.com/vllm-project/vllm-ascend/pull/11196) 给出了 DSpark 在线推理落地方案：target model 仍使用 DeepSeek V4 原有混合注意力路径，draft model 的 DSpark attention 使用标准 SWA paged KV cache，并通过 `ori_sparse_indices` 描述每个 draft query 可见的上游 slot。

当前 `SparseFlashMla` 已支持共享 KV 的三类计算路径：

- SWA（Sliding Window Attention）：仅使用 `ori_kv`。
- CSA（Compressed Sparse Attention）：使用 `ori_kv`、`cmp_kv` 和 `cmp_sparse_indices`。
- HCA（Heavily Compressed Attention）：使用 `ori_kv` 和连续 `cmp_kv`。

但当前 SMLA 对 DSpark draft attention 仍有差距：

- `ori_sparse_indices` 已在 OpDef 原型中预留，但文档和校验语义仍是“不支持传入非空 Tensor”。
- A2/A3 资料与部分校验将 `ori_win_left` 固定为 127、`ori_win_right` 固定为 0，不能表达当前 draft block 内的非因果可见。
- SWA 路径默认按连续 band window 从 `ori_kv` 取数，不能按每行可见 slot 做 gather。
- metadata 任务分配尚未把 ori 侧稀疏 topk 作为一级调度语义。

本设计目标是在尽量不新增主算子 ABI 的前提下，使能 DSpark draft attention 所需的 ori 侧稀疏 SWA 能力。

## 2. 目标与非目标

### 2.1 目标

1. 支持 DSpark draft attention 语义：每个 draft query 可见 trailing context window 与完整当前 draft block。
2. 支持 `ori_win_right > 0`，典型值为 `draft_block_size - 1`。
3. 支持 `ori_sparse_indices` 在 SWA 路径生效，用于精确描述每行 query 的 ori_kv 可见集合。
4. 支持 TND + PA_BBND 作为 DSpark 推理主路径，同时设计上兼容 BSND 和 TND 非 PA 场景。
5. 保持现有 SWA、CSA、HCA 行为不变；未传 `ori_sparse_indices` 时走原连续窗口路径。
6. metadata 与 kernel 均能处理 ori 侧稀疏 gather、无效 `-1` slot、变长 visible length、PA block table 映射。

### 2.2 非目标

1. 不在 SMLA 内实现 DSpark proposer、Markov sampling、draft logits、rejection sampling 等解码流程。
2. 不改变 target verification 的 causal 语义。target verification 仍由 target model 按原路径验证候选 token。
3. 不在本阶段新增新的公开输入参数；优先复用已有预留输入。
4. 不改变 `cmp_kv`、`cmp_sparse_indices`、`cmp_residual_kv` 的既有 CSA/HCA 语义。

## 3. DSpark Attention 语义

### 3.1 target verification 与 draft attention 的区别

| 场景 | 语义 | 是否需要看当前 q 右侧 token | SMLA 影响 |
| --- | --- | --- | --- |
| target verification | target model 验证候选 token，仍是 causal attention。 | 通常不需要。 | 若 q 是 cache 尾部连续片段，`ori_win_right = 0` 仍成立。 |
| DSpark draft attention | draft model 内部对一个 draft block 做非因果可见。 | 需要。 | 每行 q 需要看到完整当前 draft block，因此需要 `ori_win_right = draft_block_size - 1` 或 `ori_sparse_indices` 精确列出可见 slot。 |

### 3.2 DSpark draft block 可见范围

设：

- `window_size`：DSpark 使用的 trailing context window，典型为 127。
- `draft_block_size`：当前一次 draft 的 token 数。
- `draft_start`：当前 draft block 的首 token 全局位置。
- `draft_end = draft_start + draft_block_size - 1`。

则每一行 draft query 的可见全局 token 范围为：

```text
visible_global_range = [max(seq_start, draft_start - window_size), draft_end]
```

该范围由两部分组成：

```text
trailing context window: [max(seq_start, draft_start - window_size), draft_start - 1]
current draft block:     [draft_start, draft_end]
```

对 draft block 内任意第 `i` 行 query：

```text
q_global_pos = draft_start + i
right_visible_count = draft_end - q_global_pos
max_right_visible_count = draft_block_size - 1
```

因此 packed TND 场景可设置：

```text
ori_win_left  = window_size + draft_block_size - 1
ori_win_right = draft_block_size - 1
```

边界裁剪后，每行 q 都能覆盖整段 `trailing context + draft block`。

### 3.3 `ori_sparse_indices` 可见 slot 语义

DSpark 的 paged KV cache 场景下，更推荐由 `ori_sparse_indices` 显式表达可见集合。逻辑上每行 q 的可见集合为：

```text
visible_indices[row] = [
    context_pos_0,
    context_pos_1,
    ...,
    draft_start,
    draft_start + 1,
    ...,
    draft_end,
    -1,
    -1,
    ...
]
```

其中 `-1` 表示无效 slot，不参与 score、softmax 和 PV 计算。

对 PA_BBND，建议 SMLA 统一采用“逻辑 KV 行号”作为 `ori_sparse_indices` 的公共语义：

```text
logical_ori_idx = token position relative to current sequence
block_id        = ori_block_table[b, logical_ori_idx / page_block_size]
in_block_idx    = logical_ori_idx % page_block_size
physical_offset = (((block_id * page_block_size + in_block_idx) * N2 + n2) * D + d)
```

如果上游引擎已经生成 vLLM 风格的 physical slot list，应在 PyTorch API 适配层或专用 wrapper 中转换为 SMLA 约定的 logical index；若后续需要直接消费 physical slot，可通过 `topk_value_mode` 扩展区分，但不建议在本阶段混用两种语义。

## 4. 当前 SMLA 接口复用方案

### 4.1 主算子 `SparseFlashMla`

当前 OpDef 已有如下相关输入，可直接复用：

| 参数 | 当前状态 | DSpark 支持后的语义 |
| --- | --- | --- |
| `ori_kv` | 可选输入 | DSpark 使用的标准 SWA KV cache，可为 TND、BSND 或 PA_BBND。 |
| `ori_sparse_indices` | 可选输入，当前预留 | ori 侧稀疏可见集合。传入非空时启用 ori sparse SWA。 |
| `ori_block_table` | 可选输入 | PA_BBND 下逻辑 KV 行到物理 cache block 的映射。 |
| `cu_seqlens_q` | 可选输入 | TND 下 query batch 边界。 |
| `cu_seqlens_ori_kv` | 可选输入 | TND 下 ori_kv batch 边界。 |
| `seqused_q` | 可选输入 | BSND/PA_BBND 下每个 batch 的 query 有效长度。 |
| `seqused_ori_kv` | 可选输入 | BSND/PA_BBND 下每个 batch 的 ori_kv 有效长度。PA_BBND 必传。 |
| `ori_topk_length` | 可选输入，当前预留 | 可选的每行 ori sparse 有效长度；未传时通过 `-1` sentinel 判断。 |
| `metadata` | 可选输入 | `SparseFlashMlaMetadata` 生成的任务切分结果，DSpark 仍必传。 |

相关属性复用：

| 属性 | DSpark draft attention 推荐值 | 说明 |
| --- | --- | --- |
| `cmp_ratio` | 1 | DSpark draft attention 是 ori-only SWA 扩展，不使用 cmp_kv。 |
| `ori_mask_mode` | 4 | 必须显式保留为 band/window 模式。连续 SWA 下它决定真实 ori window；DSpark ori sparse SWA 下它用于进入 ori 侧 window/sparse 调度路径，最终可见集合由 `ori_sparse_indices` 精确决定。 |
| `cmp_mask_mode` | 0 或兼容 3 | 未传 `cmp_kv` 时不参与 cmp 侧计算；为兼容 vLLM-Ascend 调用可接受 3 但不生效。 |
| `ori_win_left` | `window_size + draft_block_size - 1` | packed TND 下覆盖左侧 context；PA sparse 下作为调度上界。 |
| `ori_win_right` | `draft_block_size - 1` | 支持当前 draft block 内右侧 token 可见。 |
| `layout_q` | `TND` 优先 | DSpark 在线推理通常按变长请求打包。 |
| `layout_kv` | `PA_BBND` 优先 | DSpark 推理主路径为 paged KV cache；TND/BSND 可用于单测和 fallback。 |
| `topk_value_mode` | 1 | 第一阶段保持与现有 sparse index 语义一致。 |

### 4.2 metadata 算子 `SparseFlashMlaMetadata`

metadata 侧不新增公开 ABI，复用已有输入：

| 参数 | DSpark 支持后的语义 |
| --- | --- |
| `ori_topk_length` | 可选。传入时 shape 与 `ori_sparse_indices` 去掉最后一维后一致，用于避免 kernel 扫描 `-1`。 |
| `cmp_topk_length` | 仍保持 CSA/HCA 预留或原语义，不参与 DSpark draft attention。 |
| `has_ori_kv` | true。 |
| `has_cmp_kv` | false。 |
| `ori_topk` | `ori_sparse_indices` 最后一维 K，要求不小于最大可见长度。 |
| `cmp_topk` | 0。 |

推荐 shape：

| layout_q | `ori_sparse_indices` shape | `ori_topk_length` shape |
| --- | --- | --- |
| BSND | `(B, S1, N2, K_ori)` | `(B, S1, N2)` |
| TND | `(T, N2, K_ori)` | `(T, N2)` |

其中 `K_ori >= window_size + draft_block_size`。未传 `ori_topk_length` 时，kernel 从每行 `ori_sparse_indices` 中第一个 `-1` 推导实际长度。

## 5. 模式识别与校验规则

### 5.1 新增内部模式

建议新增内部模式标识，不改变公开模式命名：

```text
SWA_CONTIGUOUS:  现有连续 ori window。
SWA_ORI_SPARSE:  ori_sparse_indices 非空，ori-only sparse SWA，用于 DSpark draft attention。
CSA:             cmp_sparse_indices 非空。
HCA:             cmp_kv 非空且 cmp_sparse_indices 为空。
```

模式推导优先级：

```text
if cmp_kv is not null:
    if cmp_sparse_indices is not null:
        mode = CSA
    else:
        mode = HCA
else:
    if ori_sparse_indices is not null:
        mode = SWA_ORI_SPARSE
    else:
        mode = SWA_CONTIGUOUS
```

`SWA_ORI_SPARSE` 不应被误判为 CSA。其核心区别是：CSA 的稀疏索引作用于 `cmp_kv`，DSpark draft attention 的稀疏索引作用于 `ori_kv`。

### 5.2 正向校验

新增/调整校验建议：

1. `ori_sparse_indices` 非空时，必须传入 `ori_kv`。
2. `ori_sparse_indices` 非空且 `layout_kv = "PA_BBND"` 时，必须传入 `ori_block_table` 和 `seqused_ori_kv`。
3. `ori_sparse_indices` dtype 必须为 INT32，shape 最后一维为 `K_ori`，`K_ori > 0`。
4. `ori_topk_length` 非空时，dtype 必须为 INT32，shape 必须与 `ori_sparse_indices` 去掉最后一维后一致。
5. `ori_topk_length[row]` 必须在 `[0, K_ori]` 范围内。
6. 未传 `ori_topk_length` 时，`ori_sparse_indices` 每行允许通过 `-1` 表示提前结束；`-1` 后的元素必须全部视为无效。
7. `ori_sparse_indices` 中非负 index 必须小于当前 batch 的 `ori_kv` 有效长度。
8. DSpark ori sparse SWA 下 `cmp_kv` 必须为空；若同时传入 `cmp_kv` 和 `ori_sparse_indices`，本阶段建议拦截，避免语义混合。
9. `ori_win_right` 支持非负数。A2/A3 如原本固定 0，需要放开到非负值；DSpark 推荐 `draft_block_size - 1`。
10. `ori_win_left` 支持非负数；DSpark 推荐 `window_size + draft_block_size - 1`。
11. DSpark ori sparse SWA 下 `ori_mask_mode` 必须为 4。`ori_mask_mode = 3` 会按 causal 语义屏蔽当前 q 右侧 draft token，不满足 DSpark draft attention；`ori_mask_mode = 0` 虽然理论上可由 `ori_sparse_indices` 完全表达可见集合，但会绕开现有 SWA/window 调度语义，第一阶段不建议支持。

### 5.3 反向校验

需要新增反向用例覆盖：

| 场景 | 期望 |
| --- | --- |
| `ori_sparse_indices` 非空但 `ori_kv` 为空 | 拦截。 |
| PA_BBND 下 `ori_sparse_indices` 非空但缺 `ori_block_table` | 拦截。 |
| PA_BBND 下 `ori_sparse_indices` 非空但缺 `seqused_ori_kv` | 拦截。 |
| `ori_topk_length` shape 与 `ori_sparse_indices` 不匹配 | 拦截。 |
| `ori_topk_length` 值大于 `K_ori` | 拦截。 |
| index 超过 `seqused_ori_kv[b]` 或 TND batch length | 拦截或 kernel 安全屏蔽，建议 host 可静态检查时拦截。 |
| DSpark 模式下同时传入 `cmp_kv` | 拦截。 |
| `ori_win_right < 0` 且不是兼容的默认值 | 拦截。 |

## 6. Tiling 与 Metadata 设计

### 6.1 TilingData 扩展

当前 `SparseFlashMlaSwaParams` 中已有 `oriSparseBlockCount`、`sparseBlockSize` 等字段，可在不破坏 A2/A3 字段顺序的前提下复用或追加：

```text
oriSparseEnabled     // 是否启用 ori_sparse_indices
oriTopk              // ori_sparse_indices 最后一维 K
oriTopkLengthEnabled // 是否传入 ori_topk_length
oriSparseBlockCount  // ceil(oriTopk / s2BaseSize)
```

如果当前字段不足，A5 可以追加字段；A2/A3 需谨慎保持 tilingdata ABI 兼容。

### 6.2 Metadata 分核原则

SWA_CONTIGUOUS 当前按 `ori_win_left/right` 推导每个 s1 tile 对应的连续 s2 范围。SWA_ORI_SPARSE 需要按 sparse K 维切分：

```text
for each b, n2, s1 tile:
    K_ori_eff = ori_topk_length[row] if provided else K_ori upper bound
    split K_ori_eff into s2 tiles by s2BaseSize
    metadata item records:
        b, n2
        s1Start, s1End
        oriSparseStart, oriSparseEnd
        oriTopkBaseOffset
        qOffset / outputOffset
```

若 `ori_topk_length` 未传，metadata 不能逐行读取 `ori_sparse_indices` 的实际 `-1` 位置，只能按 `K_ori` 上界分配；kernel 在 Vector0/gather 阶段过滤 `-1`。因此：

- 第一阶段可只按 `K_ori` 上界分配，开发量小但可能有无效计算。
- 第二阶段启用 `ori_topk_length` 后，metadata 可按每行有效长度做更细任务裁剪，减少无效 tile。

### 6.3 多 batch 与 TND

TND 下 `ori_sparse_indices` 的第 0 维按 packed q row 排列，和 `q` 的 T 维一致：

```text
packed_q_row = cu_seqlens_q[b] + local_q_row
ori_sparse_indices[packed_q_row, n2, k]
```

`ori_sparse_indices` 中的 index 为 batch 内 logical ori row：

```text
0 <= logical_ori_idx < ori_len[b]
```

TND GM offset：

```text
ori_tensor_row = cu_seqlens_ori_kv[b] + logical_ori_idx
gm_offset = ((ori_tensor_row * N2 + n2) * D + d)
```

PA_BBND GM offset：

```text
block_idx = logical_ori_idx / page_block_size
in_block  = logical_ori_idx % page_block_size
physical_block = ori_block_table[b, block_idx]
gm_offset = (((physical_block * page_block_size + in_block) * N2 + n2) * D + d)
```

BSND GM offset：

```text
gm_offset = (((b * S2 + logical_ori_idx) * N2 + n2) * D + d)
```

实际实现中还需要叠加 `ori_kv` 的 stride0 逻辑，避免非连续 0 轴输入取错。

## 7. Kernel 设计

### 7.1 整体流程

SWA_ORI_SPARSE 可复用 SWA 模板主体，但 Vector0 和 mask 逻辑需要新增分支：

```text
Metadata task
  -> V0: 按 ori_sparse_indices gather ori K/V 到 L1/UB
  -> C1: Q x K_ori^T
  -> V1: 对 -1 slot / 越界 slot 写 -inf，做 softmax
  -> C2: P x V_ori
  -> V2: 写 attn_out / softmax_lse
```

### 7.2 V0 gather

V0 负责把 `ori_sparse_indices[row, n2, k]` 对应的 K/V 从 GM 搬到连续局部 buffer，形成 matmul 可消费的右矩阵。

处理规则：

1. `idx = -1`：该 k 无效，V0 可填零，V1 对应 score 必须 mask 为 `-inf`。
2. `idx >= ori_len[b]`：安全起见应在 host 拦截；kernel 仍需防御，按无效处理，避免越界。
3. PA_BBND：通过 `ori_block_table` 将 logical idx 映射到 physical block。
4. TND：通过 `cu_seqlens_ori_kv[b]` 加 batch 前缀。
5. BSND：通过 `b * ori_stride0` 或连续 fallback 计算。

### 7.3 C1 matmul

输入矩阵：

```text
Left:  Q tile        [M, D]
Right: K_ori tile^T  [D, K_tile]
Out:   score         [M, K_tile]
```

SWA_ORI_SPARSE 下，K 维是 sparse topk 维，不再是连续 s2 坐标。score 的每一列对应 `ori_sparse_indices` 中的一个 visible idx。

### 7.4 V1 mask 与 softmax

mask 来源分两层：

1. sparse index 有效性：`idx == -1` 或 `idx >= ori_len[b]` 必须 mask。
2. window 上界：如果保留 `ori_win_left/right` 作为调度上界，可额外检查 idx 是否落在 `[q_pos - ori_win_left, q_pos + ori_win_right]` 内。

DSpark PA 路径中，`ori_sparse_indices` 是真正语义来源；`ori_win_left/right` 只是 metadata 调度上界。因此建议：

- 对 DSpark ori sparse SWA，只要 index 有效，就参与计算。
- `ori_win_left/right` 只用于限制 metadata 最大调度范围和 host 校验，不在 V1 重复裁剪 index，避免把上游已经精确给出的 full draft block slot 错误 mask 掉。
- `ori_mask_mode` 仍需要为 4，用来选择 ori 侧 window/sparse 路径；但在 `ori_sparse_indices` 非空时，kernel 不能再按 causal 右边界把 `q_pos` 右侧的 draft token mask 掉。

### 7.5 C2 matmul

输入矩阵：

```text
Left:  P = softmax(score) [M, K_tile]
Right: V_ori tile         [K_tile, D]
Out:   O tile             [M, D]
```

与现有 SWA 路径一致，区别仅在 V_ori 由 sparse gather 得到。

### 7.6 V2 写回

输出 offset 仍由 q row 决定，不受 sparse idx 影响：

```text
BSND: out[b, q_row, n1, d]
TND:  out[cu_seqlens_q[b] + q_row, n1, d]
```

padding row 或 `seqused_q` 外的 row 输出置零，`softmax_lse` 按现有约定写出。

## 8. Workspace 与局部内存

SWA_ORI_SPARSE 需要在现有 SWA workspace 基础上增加或复用以下 buffer：

| Buffer | 位置 | 用途 |
| --- | --- | --- |
| `oriIndexUb` | UB | 当前 s1 tile、n2、K_tile 的 sparse index。 |
| `oriValidMaskUb` | UB | `idx != -1` 且未越界的有效标记。 |
| `kGatherL1/Ub` | L1/UB | gather 后连续 K tile。 |
| `vGatherL1/Ub` | L1/UB | gather 后连续 V tile。 |
| `scoreMaskUb` | UB | 写 `-inf` 前的 mask。 |

内存复用建议：

1. `oriIndexUb` 与 `cmp_sparse_indices` 的 index buffer 可复用，但模式必须互斥，避免同一任务同时消费 ori 与 cmp sparse。
2. K/V gather buffer 可复用现有 SWA 连续搬运后的 L1 右矩阵 buffer，只改变填充方式。
3. `ori_topk_length` 若传入，只需每行读一个 INT32，可放入小 UB buffer。

## 9. 平台与规格约束建议

第一阶段建议规格：

| 规格项 | 建议范围 |
| --- | --- |
| dtype | FP16、BF16 |
| D | 512 |
| N2 | 1 |
| N1/N2 | 沿用现有 SMLA 平台约束 |
| layout_q | BSND、TND，DSpark 主推 TND |
| layout_kv | BSND、TND、PA_BBND，DSpark 主推 PA_BBND |
| ori sparse topk | `K_ori <= 1024`，推荐覆盖 `window_size + draft_block_size` |
| ori_mask_mode | 4 |
| ori_win_left/right | 非负数；DSpark 推荐 `window_size + draft_block_size - 1` / `draft_block_size - 1` |
| cmp_ratio | DSpark draft attention 固定 1 |
| cmp_kv | DSpark draft attention 不传 |

若 A2/A3 当前资料范围仍要求 `ori_win_left=127`、`ori_win_right=0`，需要同步更新资料和代码拦截；否则 DSpark draft attention 无法合法传入 `ori_win_right > 0`。

## 10. PyTorch API 适配

PyTorch API 建议保持两段式调用：

```python
metadata = cann_ops_transformer.sparse_flash_mla_metadata(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    seqused_ori_kv=seqused_ori_kv,
    ori_topk_length=ori_topk_length,
    num_heads_q=num_heads_q,
    num_heads_kv=1,
    head_dim=512,
    batch_size=batch_size,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_ori_kv=max_seqlen_ori_kv,
    ori_topk=ori_topk,
    cmp_topk=0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=window_size + draft_block_size - 1,
    ori_win_right=draft_block_size - 1,
    layout_q="TND",
    layout_kv="PA_BBND",
    has_ori_kv=True,
    has_cmp_kv=False,
)

attn_out, softmax_lse = cann_ops_transformer.sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    ori_sparse_indices=ori_sparse_indices,
    ori_block_table=ori_block_table,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    seqused_ori_kv=seqused_ori_kv,
    ori_topk_length=ori_topk_length,
    metadata=metadata,
    softmax_scale=softmax_scale,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=0,
    ori_win_left=window_size + draft_block_size - 1,
    ori_win_right=draft_block_size - 1,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=True,
)
```

上游 DSpark wrapper 负责构造：

```text
ori_sparse_indices[row] = logical visible ori row list
ori_topk_length[row]    = visible length
```

如果 wrapper 从 vLLM 拿到的是 physical slot list，需要先转换为 SMLA logical index，或在 SMLA 中新增明确的 physical slot mode。

## 11. Golden 与验证方案

### 11.1 Golden 公式

对每个 batch、head、row：

```python
visible = [idx for idx in ori_sparse_indices[row] if idx >= 0]
k = ori_kv[visible]
v = ori_kv[visible]
score = q[row] @ k.T * softmax_scale
p = softmax(score)
out[row] = p @ v
```

DSpark draft attention 的 visible 构造：

```python
visible = range(max(seq_start, draft_start - window_size), draft_end + 1)
```

### 11.2 正向用例

| 类别 | 覆盖点 |
| --- | --- |
| 基础 | `draft_block_size=1`，退化为 `ori_win_right=0`。 |
| 右窗口 | `draft_block_size=2/4/8/16`，验证当前 q 右侧 token 可见。 |
| 序列开头 | `draft_start < window_size`，验证左边界裁剪。 |
| PA block 边界 | visible range 跨多个 PA block。 |
| block table 非连续 | PA physical block 重排。 |
| TND 多 batch | 多序列 packed，`cu_seqlens_q` 和 `cu_seqlens_ori_kv` 不等长。 |
| padding | `ori_sparse_indices` 尾部 `-1`，`ori_topk_length` 与 `-1` 一致。 |
| dtype | FP16、BF16。 |
| head ratio | 覆盖现有 SMLA 支持的 head ratio。 |
| lse | `return_softmax_lse=True/False`。 |

### 11.3 反向用例

见 5.3，重点验证 `ori_sparse_indices` 相关拦截和错误日志。

### 11.4 性能与内存

1. 性能以 `msopprof` 为准，不能用 Python wall time 作为结论。
2. 对比对象：
   - 现有 SWA_CONTIGUOUS。
   - PyTorch gather + matmul golden。
   - vLLM-Ascend SAS DSpark path，如可用。
3. 内存检查使用 mssanitizer，至少覆盖 PA_BBND + ori sparse + `-1` padding + block table 非连续。

## 12. 开发拆分建议

### 12.1 PR 1：host 校验与文档

- 放开 `ori_sparse_indices` 非空校验。
- 增加 SWA_ORI_SPARSE 模式推导。
- 放开 DSpark 所需 `ori_win_right > 0`。
- 文档同步说明 `ori_sparse_indices`、`ori_topk_length`、`ori_win_left/right` 的 DSpark 语义。

### 12.2 PR 2：metadata 支持 ori sparse

- metadata 支持按 `ori_topk` 上界分配任务。
- 支持 `ori_topk_length` 后，裁剪无效 sparse tile。
- 增加 host UT。

### 12.3 PR 3：kernel V0/V1 支持 ori sparse

- SWA 模板新增 ori sparse gather。
- PA_BBND/TND/BSND offset 统一。
- V1 mask 支持 `-1` 与越界防御。
- 完成基础 pytest 与 ATK 正向精度。

### 12.4 PR 4：泛化、性能和 DSpark wrapper

- 增加 DSpark 场景 ATK 用例。
- PyTorch API 示例与文档。
- 性能优化：使用 `ori_topk_length` 减少无效 tile。
- mssanitizer 与 msopprof 报告。

## 13. 风险与待确认项

1. `ori_sparse_indices` 使用 logical index 还是 physical slot 需要和上游 DSpark wrapper 明确。建议 SMLA 公共语义使用 logical index，wrapper 负责转换。
2. `ori_topk_length` 当前是预留输入，启用后需要同步更新所有 aclnn 文档、PyTorch API 文档和反向拦截。
3. A2/A3 若存在固定 `ori_win_left=127`、`ori_win_right=0` 的资料约束，需要产品规格确认后放开。
4. metadata 若只按 `K_ori` 上界分配，短序列或 `-1` 较多时会有无效计算，性能可能不理想。
5. PA_BBND 下 block table 非连续会增加 gather 随机访存，需评估 L1/UB 排布和 DataCopy 粒度。
6. 如果后续 DSpark engine 支持跨请求重排或非尾部拼接，单靠 `cu_seqlens` 推导 q 位置不够，需要 wrapper 传入明确的 visible indices；本设计通过 `ori_sparse_indices` 可以覆盖该情况。
7. **遗留（性能）**：arch22 SWA ori sparse 路径在 `GetOriSparseActualSeqLen()`（扫描 `-1`）与 `DataCopyPABySlots()`（按 K tile 循环 `GetValue` 读 indices）中逐元素 GM 读 index，会打断 MTE/计算流水。后续应改为批量搬 index 到 UB/L1 再 gather，或绑定 `ori_topk_length` 避免扫描；本阶段先保证 logical 语义与精度，性能优化单列。

## 14. 结论

DSpark draft attention 的关键不是修改 target verification 的 causal 逻辑，而是让 draft model attention 能表达“每个 draft query 可见 trailing context window + 完整当前 draft block”。对 SMLA 来说，最小可行方案是使能已有的 `ori_sparse_indices` 预留输入，并放开 `ori_win_right > 0`，形成 ori sparse SWA 路径。该方案不需要新增主算子公开 ABI，能覆盖 vLLM-Ascend DSpark 的核心 attention 需求，同时保持现有 SWA、CSA、HCA 路径兼容。

---

## 附录：实现偏差说明（2026-07-23）

> 正文设计原则不变；以下为相对 §7.2 / §7.4 / §12.3「V1 mask 支持 `-1` 与越界防御」的**落地取舍**（方案 A），便于与代码对照。

| 设计原文 | 实际实现 |
| --- | --- |
| V1 对每个 `-1` / 越界 slot **逐列**写 `-inf` | **尾部 `-1`**：仍由 `actualSingleProcessSInnerSize` 对 tile 尾列统一 mask（DSpark 仅尾部 padding，与 golden 一致）。**越界 index**：V1 对 `idx < 0` 或 `idx >= ori_len[b]` 的列补 `-inf`。 |
| V0 无效 slot「可填零，V1 必须 mask」 | **仅 V1 负责 mask**：V0 对 `idx < 0` 仅 `continue`（不 gather）；越界 index 由 V1 单列 `-inf` 屏蔽 softmax，V0 不做 `Duplicate(0)`（冗余）。 |
| 中间位置 `-1`（非尾部） | **未单独处理**；与 DSpark / vLLM 约定（`-1` 仅作尾部 padding）一致，不做方案 B 式全量逐列扫描。 |
| host 拦截 `index >= ori_len` | **kernel 防御已做（V1）**；host 侧仍仅 shape/dtype 校验，运行时数值越界由 V1 mask，不在 tiling 读 indices。 |

**影响范围**：arch22 SWA + TND + PA_BBND 的 DSpark 主路径；连续 SWA / 无 `ori_sparse_indices` 路径无行为变化。
