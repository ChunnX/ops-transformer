# MQA Logits + TopK 融合 Kernel 修改说明

## 概述

在 DeepGEMM 的 `sm100_mqa_logits.cuh` 基础上，新增 TopK 选取阶段，使 kernel 直接输出 `sparse_indices`（稀疏 KV token 索引），功能对齐 `quant_lightning_indexer` 算子。

## 文件结构

```
topk_utils.cuh                      # 新增: warp级流式TopK工具函数
sm100_mqa_logits_with_topk.cuh      # 新增: 融合TopK的完整kernel
```

## 两阶段设计

### Phase 1: 计算 logits（复用原 kernel 逻辑）

- 完全保留原始的 MQA logits 计算流程
- 唯一变更：store 目标从 `logits` 改为 `logits_buffer`（float 类型临时缓冲）
- 所有 warp 分工不变：TMA Q / TMA KV / MMA Issue / Math Warpgroups

### Phase 2: TopK 选取（新增）

- Phase 1 完成后 `__syncthreads()` 全 CTA 同步
- 仅 Math Warps 参与 TopK（专用 warps 闲置）
- 每个 math warp 按轮询方式处理 Q tokens
- 每个 Q token 由一个 warp 执行流式 TopK

## TopK 算法（warp_batch_topk）

```
输入: logits[k_len], kSparseCount (Phase 1 已计算完毕的完整 logits 数组)
输出: out_indices[kSparseCount] (全局KV索引, 按logit降序)

Step 1: 初始化 SMEM top-K buffer 为 -inf
Step 2: 用前 kSparseCount 个 logits 填充 buffer, 单线程冒泡排序
Step 3: 扫描剩余 logits (每轮32个):
  - 每个 lane 加载一个 logit
  - __ballot_sync 找出超过当前最小值的 lane
  - 逐个 qualifying lane 执行: 二分查找 + 位移插入
Step 4: 输出排序后的索引
```

## 新增模板参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `kSparseCount` | uint32_t | TopK 数量 (对应 quant_lightning_indexer 的 sparse_count) |

## 新增接口参数

| 参数 | 方向 | Shape | 说明 |
|---|---|---|---|
| `logits_buffer` | 输入/输出 | [aligned_seq_len, stride] | Phase 1 临时缓冲 (float) |
| `sparse_indices` | 输出 | [num_q_tokens, kSparseCount] | 最终输出 (int32, 全局KV索引) |

## 与 quant_lightning_indexer 的对应

| quant_lightning_indexer | 本 kernel |
|---|---|
| query (INT8/FP8) | q (FP8_E4M3 / FP4) |
| key (INT8/FP8) | kv (FP8_E4M3 / FP4) |
| query_dequant_scale | sf_q (FP32 / UE8M0) |
| key_dequant_scale | sf_kv (FP32 / UE8M0) |
| weights | weights (FP32 / BF16) |
| sparse_count | kSparseCount |
| sparse_indices [B, Q_S, K_N, sparse_count] | sparse_indices [num_q_tokens, kSparseCount] |
| actual_seq_lengths_key | cu_seq_len_k_start / cu_seq_len_k_end |
| block_table (PA_BSND) | block_table (paged mode) |

## Scheduler 需要新增的接口

Phase 2 需要从 scheduler 获取每个 Q token 的 KV 范围和总 Q token 数：

```cpp
// 需要在 SM100MQALogitsScheduler / SM100PagedMQALogitsScheduler 中新增:

// 获取 Q token 总数
CUTLASS_DEVICE uint32_t get_num_q_tokens() const {
    return num_q_tokens;
}

// 获取第 q_token 个 Q token 的 KV 起始索引
CUTLASS_DEVICE uint32_t get_k_start_for_token(const uint32_t& q_token) const {
    if constexpr (kIsPaged) {
        // paged mode: return 0, context_lens determines range
        return 0;
    } else {
        return cute::min(cu_seq_len_k_start[cute::min(q_token, num_q_tokens - 1)], num_kv_tokens);
    }
}

// 获取第 q_token 个 Q token 的 KV 结束索引
CUTLASS_DEVICE uint32_t get_k_end_for_token(const uint32_t& q_token) const {
    if constexpr (kIsPaged) {
        // paged mode: use context_lens
        if constexpr (kIsContextLens2D) {
            const uint32_t batch = q_token / kTokensPerRequest;
            return context_lens[batch * 1 + 0];  // adjust based on layout
        }
        return context_lens[q_token];
    } else {
        return cute::min(cu_seq_len_k_end[cute::min(q_token, num_q_tokens - 1)], num_kv_tokens);
    }
}
```

## 调用示例

```cpp
// 1. 分配 logits_buffer (临时)
const int aligned_seq_len = align(seq_len, 128);
const int stride = align(seq_len_kv, 256);
auto logits_buffer = torch::empty({aligned_seq_len, stride}, 
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

// 2. 分配 sparse_indices (输出)
auto sparse_indices = torch::empty({seq_len, kSparseCount},
    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

// 3. 启动 kernel
sm100_mqa_logits_with_topk<...>(
    seq_len, seq_len_kv, stride,
    cu_seq_len_k_start, cu_seq_len_k_end,
    logits_buffer.data_ptr<float>(),
    sparse_indices.data_ptr<int32_t>(),
    tensor_map_q, tensor_map_sf_q,
    tensor_map_kv, tensor_map_sf_kv,
    tensor_map_weights);
```

## SMEM 复用

Phase 1 使用 SMEM 存储 Q/KV/SF/Weights/Barriers。
Phase 2 复用同一块 SMEM 作为 TopK scratch buffer：

```
Phase 1 SMEM:                     Phase 2 SMEM (reuse):
┌──────────────┐                  ┌──────────────────┐
│ Q data       │                  │ topk_vals        │ [kSparseCount * 4B]
│ KV data      │     reuse        │ topk_idxs        │ [kSparseCount * 4B]
│ SF data      │ ──────────→      └──────────────────┘
│ Weights      │                  Total: kSparseCount * 8 bytes
│ Barriers     │                  e.g. kSparseCount=2048 → 16KB
└──────────────┘
```

## 性能考量

### 当前实现的瓶颈

1. **TopK 初始排序**: 单线程冒泡排序 O(k²), k=2048 时约 4M 次比较
2. **流式插入**: 串行处理 qualifying lanes, 每次插入需要 shift O(k)
3. **logits_buffer 回读**: Phase 2 需要从 GMEM 读回 Phase 1 写入的 logits

### 优化方向（后续）

1. **初始排序**: 用 bitonic sort 替代冒泡排序 (并行 O(k log²k))
2. **流式插入**: 用并行 merge 替代串行插入
3. **流式 TopK**: 在 Phase 1 计算过程中维护 SMEM top-K, 避免 Phase 2 回读 (复杂度高, 后续优化方向)
4. **多 warp 协作**: 多个 warp 共同处理一个 Q token 的 TopK
5. **TMA 加载 logits**: 用 TMA 批量加载 logits 到 SMEM, 减少逐元素 GMEM 访问

## 约束

- `kSparseCount` 需 ≤ 2048 (SMEM 容量限制)
- `logits_buffer` 需为 float32 (Phase 1 输出精度)
- 输出索引为全局 KV token 索引 (非相对索引)
- 无效的 Q token (k_len=0) 输出索引为 -1
