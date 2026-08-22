# AlltoAllvGroupedMatMulV2 用例设计文档

## 1. 算子标杆

PyTorch 参考实现：

```python
import torch


def allto_allv_grouped_mat_mul_v2_reference(
    rank_id,
    rank_inputs,
    rank_send_counts,
    local_weight,
    trans_gmm_weight=False,
    permute_out_flag=False,
    mm_x=None,
    mm_weight=None,
    trans_mm_weight=False,
):
    """
    rank_inputs[src] 已按目标全局专家编号升序排列。
    rank_send_counts[src] 的长度为 world_size * expert_per_rank。
    输出顺序固定为：本地专家优先，专家内按 source rank 升序。
    """
    world_size = len(rank_inputs)
    expert_per_rank = local_weight.shape[0]
    local_expert_chunks = []
    expert_token_counts = []

    for local_expert in range(expert_per_rank):
        global_expert = rank_id * expert_per_rank + local_expert
        source_chunks = []
        for src_rank in range(world_size):
            counts = rank_send_counts[src_rank]
            start = int(sum(counts[:global_expert]))
            length = int(counts[global_expert])
            if length != 0:
                source_chunks.append(rank_inputs[src_rank][start:start + length])

        if source_chunks:
            expert_x = torch.cat(source_chunks, dim=0)
        else:
            expert_x = rank_inputs[0].new_empty((0, rank_inputs[0].shape[1]))
        local_expert_chunks.append(expert_x)
        expert_token_counts.append(expert_x.shape[0])

    non_empty = [chunk for chunk in local_expert_chunks if chunk.shape[0] != 0]
    permute_out = (
        torch.cat(non_empty, dim=0)
        if non_empty
        else rank_inputs[0].new_empty((0, rank_inputs[0].shape[1]))
    )

    gmm_parts = []
    for expert_id, expert_x in enumerate(local_expert_chunks):
        if expert_x.shape[0] == 0:
            continue
        weight = local_weight[expert_id]
        if trans_gmm_weight:
            weight = weight.transpose(0, 1)
        # 以 FP32 作为高精度标杆，最终结果转换回 BF16。
        gmm_parts.append(
            torch.matmul(expert_x.float(), weight.float()).to(torch.bfloat16)
        )

    logical_n = (
        local_weight.shape[1] if trans_gmm_weight else local_weight.shape[2]
    )
    gmm_y = (
        torch.cat(gmm_parts, dim=0)
        if gmm_parts
        else rank_inputs[0].new_empty((0, logical_n))
    )

    mm_y = None
    if mm_x is not None:
        shared_weight = mm_weight.transpose(0, 1) if trans_mm_weight else mm_weight
        mm_y = torch.matmul(mm_x.float(), shared_weight.float()).to(torch.bfloat16)

    return gmm_y, mm_y, permute_out if permute_out_flag else None
```

---

## 2. 用例说明

### 2.1 测试配置

```python
SUPPORTED_DTYPES = ["bfloat16"]

# 字段：
# (id, ep, expert_per_rank, local_m, k, n, distribution,
#  trans_gmm_weight, permute_out, shared_mm, trans_mm_weight)
TEST_SHAPES = [
    ("basic_ep2",          2, 2, 16, 256, 128, "uniform",                False, False, False, False),
    ("permute_ep2",        2, 2, 15, 256,  96, "skewed",                 False, True,  False, False),
    ("transpose_ep2",      2, 2, 13, 320, 128, "uniform",                True,  False, False, False),
    ("shared_ff_ep2",      2, 2, 16, 256, 128, "uniform",                False, False, True,  False),
    ("shared_ft_ep2",      2, 2, 16, 256, 128, "uniform",                False, True,  True,  True),
    ("uniform_ep4",        4, 2, 24, 256, 128, "uniform",                False, False, False, False),
    ("zero_peer_ep4",      4, 2, 19, 320, 128, "source_rank_2_zero",     False, True,  False, False),
    ("zero_expert_ep4",    4, 4, 23, 256, 192, "local_expert_1_zero",    True,  True,  False, False),
    ("skew_ep4",           4, 4, 29, 384, 128, "ninety_percent_one_exp", False, False, False, False),
    ("single_expert_ep8",  8, 2, 31, 512, 256, "all_to_one_global_exp",  False, False, False, False),
    ("tail_tile_ep8",      8, 2, 37, 272, 130, "uneven",                 True,  True,  False, False),
    ("shared_ep8",         8, 2, 32, 512, 192, "uniform",                False, True,  True,  False),
]

GENERAL_SHAPES = [
    ("tiny_m1_ep2",        2, 1,  1,  16,  16, "single_token",           False, False, False, False),
    ("tiny_tail_ep2",      2, 2,  3,  17,  19, "uneven",                 False, True,  False, False),
    ("small_trans_ep2",    2, 2,  5,  32,  48, "uniform",                True,  False, False, False),
    ("small_shared_ep2",   2, 4,  7,  64,  32, "skewed",                 False, True,  True,  True),
    ("prime_dims_ep2",     2, 2, 11, 113, 127, "uneven",                 True,  True,  False, False),
    ("m64_ep2",            2, 4, 64, 256, 256, "uniform",                False, False, False, False),
    ("m127_ep2",           2, 4,127, 320, 192, "skewed",                 False, True,  False, False),
    ("m129_ep2",           2, 4,129, 256, 128, "uneven",                 True,  False, False, False),
    ("tiny_ep4",           4, 1,  4,  16,  32, "one_token_per_rank",     False, True,  False, False),
    ("expert8_ep4",        4, 8, 32, 128,  64, "uniform",                False, False, False, False),
    ("alternate_zero_ep4", 4, 4, 17, 192,  96, "alternate_peer_zero",    True,  True,  False, False),
    ("one_rank_hot_ep4",   4, 2, 33, 256, 160, "only_source_rank_0",     False, True,  True,  False),
    ("k_tail_ep4",         4, 2, 21, 273, 128, "uneven",                 False, False, False, False),
    ("n_tail_ep4",         4, 2, 21, 256, 131, "uneven",                 True,  True,  False, False),
    ("m255_ep4",           4, 4,255, 128, 128, "skewed",                 False, False, False, False),
    ("tiny_ep8",           8, 1,  8,  32,  32, "one_token_per_rank",     False, True,  False, False),
    ("expert4_ep8",        8, 4, 41, 128,  96, "uniform",                True,  False, False, False),
    ("many_zero_ep8",      8, 2,  9, 256, 128, "six_source_ranks_zero",  False, True,  False, False),
    ("m257_ep8",           8, 2,257, 256, 128, "uneven",                 False, False, True,  False),
    ("wide_n_ep8",         8, 2, 48, 256, 512, "uniform",                True,  True,  True,  True),
]

BOUNDARY_VALUES = [
    ("invalid_comm_ai_cpu",      {"comm_mode": "ai_cpu"},       "ACLNN_ERR_PARAM_INVALID"),
    ("invalid_comm_ccu",         {"comm_mode": "ccu"},          "ACLNN_ERR_PARAM_INVALID"),
    ("invalid_fp16",             {"dtype": "float16"},          "GRAPH_FAILED"),
    ("invalid_count_length",     {"count_length_delta": -1},     "GRAPH_FAILED"),
    ("invalid_count_sum",        {"send_sum_delta": 1},          "GRAPH_FAILED"),
    ("invalid_negative_count",   {"send_count_value": -1},       "GRAPH_FAILED"),
    ("invalid_window_size",      {"window_bytes_delta": -1},     "ACLNN_ERR_PARAM_INVALID"),
    ("invalid_partial_shared_mm",{"only_mm_x_present": True},     "ACLNN_ERR_PARAM_INVALID"),
    ("invalid_permute_output",   {"flag": True, "output": None},"ACLNN_ERR_PARAM_INVALID"),
]
```

所有有效用例满足：

```text
sendCounts length = EP * expert_per_rank
recvCounts length = EP * expert_per_rank
sum(sendCounts) = local_m
gmmY rows = sum(recvCounts)
output order = local expert major, source rank minor
zero-token experts participate in synchronization but emit no rows
```

### 2.2 用例覆盖统计

| 类别 | Shape数量 | 边界值数量 | dtype数量 | 总用例数 |
|------|----------:|-----------:|----------:|---------:|
| 常规形状 | 12 | 0 | 1 | 12 |
| 泛化形状 | 20 | 0 | 1 | 20 |
| 边界值 | 0 | 9 | 1 | 9 |
| **有效精度用例合计** | **32** | **0** | **1** | **32** |

---

## 3. 使用说明

### 生成测试数据示例

```python
import random
import torch


def make_rank_counts(world_size, expert_per_rank, local_m, seed):
    random.seed(seed)
    total_experts = world_size * expert_per_rank
    cut_points = sorted(random.sample(range(local_m + total_experts - 1), total_experts - 1))
    counts = []
    previous = -1
    for point in cut_points + [local_m + total_experts - 1]:
        counts.append(point - previous - 1)
        previous = point
    assert len(counts) == total_experts
    assert sum(counts) == local_m
    return counts


def make_bf16_inputs(world_size, expert_per_rank, local_m, k, n, seed):
    generator = torch.Generator().manual_seed(seed)
    rank_inputs = [
        torch.randn((local_m, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    weights = [
        torch.randn((expert_per_rank, k, n), generator=generator, dtype=torch.float32).to(torch.bfloat16)
        for _ in range(world_size)
    ]
    counts = [
        make_rank_counts(world_size, expert_per_rank, local_m, seed + rank)
        for rank in range(world_size)
    ]
    return rank_inputs, weights, counts
```

### 注意事项

1. 运行多卡用例时，所有 rank 必须使用相同的 `epWorldSize`、转置属性和专家数。
2. `gmmX` 必须按目标全局专家顺序排列；生成随机输入后应按 `sendCounts` 分段保存。
3. `gmmWeight` 的逻辑形状为 `[E,K,N]`；`transGmmWeight=true` 时物理形状为 `[E,N,K]`。
4. 单个专家可以接收零 token，但 `gmmX` 本身不能为空 Tensor。
5. `permuteOut` 与 `gmmY` 使用相同的 token 行顺序。
6. 共享专家的 `mmX/mmWeight/mmY` 必须同时存在或同时为空。
7. EP=2/4/8 用例分别在对应数量的物理 NPU 上启动，不得用单进程伪造 peer window。
