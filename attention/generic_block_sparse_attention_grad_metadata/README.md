# GenericBlockSparseAttentionGradMetadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |

## 功能说明

`GenericBlockSparseAttentionGradMetadata` 算子根据 KV2Q 稀疏块索引表 `rsvd_block_idx` / `rsvd_block_count`，按 `B → N2 → J → G` 顺序展开 `(b, n2, j, g)` 任务列表，并在 AIC 核间做负载均衡，供后续 `GenericBlockSparseAttentionGrad` 算子消费。

设计要点（对齐详细设计 §3.1 / §6.1 / §6.2）：

- 主循环遍历序：**B → N2 → J → G**（先 S2 块，再组内 Q Head）
- 调度原子单元：**`(b, n2, j)` KV 块**（同块内 `g=0..G-1` 连续展开，保证 KV L1 复用）
- 负载单位：**`[baseM, baseN] = [128, 128]` 基本块个数**
  - 单 KV 块开销 = `G × ⌈count/baseM⌉ × ⌈kvBlockLen/baseN⌉`
  - `kvBlockLen = min(By, actS2 - j×By)`
- 分核策略：按 KV 块组做**贪心负载均衡**（参考 QBSA `ScheduleFa`），不把同一 `(b,n2,j)` 拆到不同核
- 输出 `metadata`（int64，长度 4096）包含分核边界与任务列表

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 |
| :--- | :--- | :--- |
| rsvd_block_idx | 输入 | int32，shape `[B, N2, J, maxS1]`，KV 块选中的 Q token 索引 |
| rsvd_block_count | 输入 | int32，shape `[B, N2, J]`，每个 KV 块实际选中的 Q 数量 |
| cu_seq_lengths | 可选输入 | int64，TND 布局必选 |
| cu_seq_lengths_kv | 可选输入 | int64，TND 布局必选 |
| seqused_q / seqused_kv | 可选输入 | int32，变长序列实际长度 |
| max_q_seqlen / max_kv_seqlen | 属性 | 最大 Q/KV 序列长度 |
| num_q_heads / num_kv_heads | 属性 | Q/KV head 数，要求 N1 % N2 == 0 |
| head_dim | 属性 | 当前固定 128 |
| block_shape | 属性 | `{Bx, By}`，当前 `{1, 128}` |
| is_packed_gqa | 属性 | 当前仅支持 1 |
| q_input_layout / kv_input_layout | 属性 | TND / BNSD / BSND，须一致 |
| mask_type / softmax_precision | 属性 | 传递给主算子 |
| window_size_left / window_size_right | 属性 | 当前须为 -1 |
| metadata | 输出 | int64，shape `[metaSize]`，metaSize = 80 + B × N1 × J × 4 |

## Metadata 布局

| 偏移 | 字段 | 含义 |
| :--- | :--- | :--- |
| 0 | total_num | 有效任务总数（= KV 块组数 × G） |
| 1 | total_block_cost | 全部 `[128,128]` 基本块总数 |
| 2 | max_core_block_cost | 单核最大基本块数 |
| 3 | base_m | 128 |
| 4 | base_n | = blockShapeY（默认 128） |
| 5 | used_core_num | 实际启用核数 |
| 6 | group_size | G = N1 / N2 |
| 7 | max_task_count | 单 KV 块最大 count |
| 8..43 | core_task_start | 每核任务起始下标 |
| 44..79 | core_task_end | 每核任务结束下标（开区间） |
| 80+ | task_list | 每任务 4 元组 `(b, n2, j, g)` |

## aclnn 接口

- `aclnnGenericBlockSparseAttentionGradMetadataGetWorkspaceSize`
- `aclnnGenericBlockSparseAttentionGradMetadata`

详细接口说明见 [docs/aclnnGenericBlockSparseAttentionGradMetadata.md](docs/aclnnGenericBlockSparseAttentionGradMetadata.md)。

## 目录结构

```
generic_block_sparse_attention_grad_metadata/
├── op_host/           # L2 aclnn + 参数校验
├── op_kernel/         # metadata 布局常量（与主 Grad 共享）
└── op_kernel_aicpu/   # AICPU 任务展开与分核
```
