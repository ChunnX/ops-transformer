# AllToAllV GroupedMatMul SHMEM 用例设计

## 1. 算子标杆

本算子没有可直接表达多 rank 变长路由的单一 PyTorch API。标杆由高精度
PyTorch 小算子组合：先按 `sendCounts[dstRank, dstLocalExpert]` 从每个源 rank
切分输入行，再按 `recvCounts[srcRank, localExpert]` 拼出本 rank 各本地专家的
输入，最后使用 FP32 `torch.matmul`，按本地专家和源 rank 的路由顺序写入输出。

```python
def alltoallv_gmm_golden(xs, weights, send_counts, recv_counts, rank):
    world = len(xs)
    local_experts = weights[rank].shape[0]
    received = [[] for _ in range(local_experts)]

    for src in range(world):
        cursor = 0
        for dst in range(world):
            for dst_expert in range(local_experts):
                count = send_counts[src][dst * local_experts + dst_expert]
                rows = xs[src][cursor:cursor + count].float()
                cursor += count
                if dst == rank:
                    expected = recv_counts[rank][src * local_experts + dst_expert]
                    assert count == expected
                    received[dst_expert].append(rows)
        assert cursor == xs[src].shape[0]

    output = []
    for expert in range(local_experts):
        weight = weights[rank][expert].float()
        for src_rows in received[expert]:
            output.append(torch.matmul(src_rows, weight))
    return torch.cat(output, dim=0).to(torch.bfloat16)
```

## 2. 用例说明

### 2.1 测试配置

```python
SUPPORTED_DTYPES = [torch.bfloat16]

# 元组字段：name, M, K, N, EP, local_experts, route_pattern
TEST_SHAPES = [
    ("source_reference", 4084, 2048, 4096, 2, 4, "uneven"),
    ("ep1_local_only", 128, 256, 256, 1, 4, "balanced"),
    ("ep2_balanced_small", 128, 256, 256, 2, 1, "balanced"),
    ("ep2_balanced_multi_expert", 512, 512, 1024, 2, 4, "balanced"),
    ("ep4_balanced", 1024, 1024, 2048, 4, 4, "balanced"),
    ("ep8_balanced", 2048, 2048, 4096, 8, 8, "balanced"),
    ("m0_128_family", 640, 1024, 4096, 2, 8, "uneven"),
    ("m0_256_family", 1280, 1024, 512, 4, 8, "uneven"),
    ("large_local_expert", 4096, 2048, 4096, 8, 64, "sparse"),
]

GENERAL_SHAPES = [
    ("tiny_one_token", 1, 16, 16, 2, 1, "single_route"),
    ("tiny_less_than_ep", 3, 32, 48, 4, 1, "sparse"),
    ("tiny_ep8", 8, 64, 64, 8, 1, "diagonal"),
    ("odd_ep3", 192, 256, 256, 3, 2, "uneven"),
    ("odd_ep5", 320, 256, 512, 5, 2, "uneven"),
    ("odd_ep7", 448, 512, 512, 7, 2, "uneven"),
    ("m_residue_1", 129, 256, 256, 2, 2, "uneven"),
    ("m_residue_63", 191, 256, 512, 2, 4, "uneven"),
    ("ep6_balanced", 384, 256, 512, 6, 2, "balanced"),
    ("k_residue_16", 256, 272, 512, 2, 4, "balanced"),
    ("k_residue_48", 384, 304, 768, 4, 4, "uneven"),
    ("k_less_than_k0", 96, 128, 256, 2, 2, "uneven"),
    ("n_residue_16", 256, 512, 272, 2, 4, "balanced"),
    ("n_residue_80", 512, 1024, 592, 4, 4, "uneven"),
    ("n_less_than_n0", 192, 512, 96, 2, 4, "uneven"),
    ("all_residue", 333, 560, 336, 4, 3, "uneven"),
    ("zero_expert_ep2", 256, 256, 512, 2, 8, "zero_expert"),
    ("zero_expert_ep4", 512, 512, 1024, 4, 8, "zero_expert"),
    ("zero_expert_ep8", 1024, 1024, 2048, 8, 8, "zero_expert"),
    ("zero_peer_payload", 512, 512, 1024, 4, 4, "zero_peer"),
    ("single_hot_peer", 512, 1024, 1024, 8, 4, "single_peer"),
    ("highly_uneven", 2047, 1024, 2048, 8, 8, "first_route_heavy"),
    ("one_local_expert_ep2", 512, 1024, 2048, 2, 1, "uneven"),
    ("one_local_expert_ep8", 1024, 1024, 2048, 8, 1, "uneven"),
    ("expert_63", 2016, 512, 1024, 4, 63, "sparse"),
    ("expert_64", 2048, 512, 1024, 4, 64, "sparse"),
    ("workspace_near_limit", 1230000, 256, 128, 2, 1, "balanced"),
]

BOUNDARY_VALUES = [
    ("ep_below_min", 64, 256, 256, 0, 1, "reject"),
    ("ep_above_max", 64, 256, 256, 9, 1, "reject"),
    ("expert_zero", 64, 256, 256, 2, 0, "reject"),
    ("expert_above_max", 64, 256, 256, 2, 65, "reject"),
    ("counts_too_short", 64, 256, 256, 2, 4, "reject"),
    ("counts_too_long", 64, 256, 256, 2, 4, "reject"),
    ("negative_count", 64, 256, 256, 2, 4, "reject"),
    ("send_sum_mismatch", 64, 256, 256, 2, 4, "reject"),
    ("output_rows_mismatch", 64, 256, 256, 2, 4, "reject"),
    ("workspace_over_limit", 1240000, 256, 128, 2, 1, "reject"),
]
```

`balanced`、`uneven`、`sparse` 等 route pattern 必须由确定性种子生成完整的
`EP * local_experts` 数组。有效用例要求每个源 rank 的 `sum(sendCounts) == M`，
并通过所有 rank 的 send 表反推每个目的 rank 的 recv 表，禁止分别随机生成两张
不一致的路由表。

### 2.2 用例覆盖统计

| 类别 | Shape 数量 | 边界值数量 | dtype 数量 | 总用例数 |
|---|---:|---:|---:|---:|
| 典型形状 | 9 | 0 | 1 | 9 |
| 泛化形状 | 27 | 0 | 1 | 27 |
| 非法边界 | 0 | 10 | 1 | 10 |
| **总计** | **36** | **10** | **1** | **46** |

## 3. 使用说明

### 生成测试数据示例

```python
generator = torch.Generator().manual_seed(20260810)
x = torch.randn((m, k), generator=generator, dtype=torch.float32).to(torch.bfloat16)
weight = torch.randn(
    (local_experts, k, n), generator=generator, dtype=torch.float32
).to(torch.bfloat16)
send_counts = make_route_pattern(route_pattern, m, ep, local_experts)
recv_counts = transpose_all_rank_routes(all_send_counts, ep, local_experts)
```

多 rank 执行时每个 rank 使用独立输入和本地专家权重，但使用同一份全局路由
描述生成互相一致的 send/recv 表。`gmmX` 与执行 workspace 必须来自同一个已
初始化的 SHMEM heap；`shmemContext` 使用该初始化流程返回的 FFTS 配置地址。
