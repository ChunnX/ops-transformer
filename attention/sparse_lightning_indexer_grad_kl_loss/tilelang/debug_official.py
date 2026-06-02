#!/usr/bin/env python3
"""
调试官方 npu_sparse_lightning_indexer_grad_kl_loss 调用
严格对齐C++示例参数
"""

import torch
import torch_npu
import math

print("=" * 70)
print("调试官方 npu_sparse_lightning_indexer_grad_kl_loss")
print("=" * 70)

# 严格对齐C++示例: s1=1, s2=1, n1=128, n2=1, nQueryIndex=64, topK=2048, TND
t1 = 1
t2 = 1
n1 = 128
n2 = 1
n_query_index = 64
d_query = 512
d_query_index = 128
d_rope = 64
topk = 2048
G = n1 // n2  # G=128

print(f"\n参数: t1={t1}, t2={t2}, n1={n1}, n2={n2}, nQueryIndex={n_query_index}")
print(f"  d_query={d_query}, d_query_index={d_query_index}, d_rope={d_rope}, topk={topk}, G={G}")

torch.manual_seed(42)

# TND layout tensors
query = torch.randn(t1, n1, d_query, dtype=torch.float16).npu()
key = torch.randn(t2, n2, d_query, dtype=torch.float16).npu()
query_index = torch.randn(t1, n_query_index, d_query_index, dtype=torch.float16).npu()
key_index = torch.randn(t2, n2, d_query_index, dtype=torch.float16).npu()
weights = torch.randn(t1, n_query_index, dtype=torch.float16).npu()

# queryRope/keyRope (C++示例中提供了)
query_rope = torch.randn(t1, n1, d_rope, dtype=torch.float16).npu()
key_rope = torch.randn(t2, n2, d_rope, dtype=torch.float16).npu()

# sparseIndices: TND (T1, N2, topK) = (1, 1, 2048)
# C++示例: 前 t2 个设为 0..t2-1, 其余设为 -1
sparse_indices = torch.full((t1, n2, topk), -1, dtype=torch.int32).npu()
for i in range(t2):
    sparse_indices[0, 0, i] = i

# softmaxMax/Sum: TND (N2, T1, G) = (1, 1, 128), dtype=float32
# C++示例: softmaxMax=25.4483, softmaxSum=1.0
softmax_max = torch.full((n2, t1, G), 25.4483, dtype=torch.float32).npu()
softmax_sum = torch.full((n2, t1, G), 1.0, dtype=torch.float32).npu()

scale_value = 1.0 / math.sqrt(d_query)

# actual_seq_lengths: API签名要求 SymInt[]? (list[int]), 不是 tensor
actual_seq_qlen = [t1]
actual_seq_klen = [t2]

print(f"\nTensor shapes:")
print(f"  query: {query.shape}, key: {key.shape}")
print(f"  query_index: {query_index.shape}, key_index: {key_index.shape}")
print(f"  weights: {weights.shape}")
print(f"  query_rope: {query_rope.shape}, key_rope: {key_rope.shape}")
print(f"  sparse_indices: {sparse_indices.shape}, dtype={sparse_indices.dtype}")
print(f"  softmax_max: {softmax_max.shape}, softmax_sum: {softmax_sum.shape}")
print(f"  actual_seq_qlen: {actual_seq_qlen} (type={type(actual_seq_qlen).__name__}), actual_seq_klen: {actual_seq_klen} (type={type(actual_seq_klen).__name__})")
print(f"  scale_value: {scale_value}")

print("\n调用 npu_sparse_lightning_indexer_grad_kl_loss (TND)...")
try:
    d_qi, d_ki, dw, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        query, key, query_index, key_index, weights, sparse_indices,
        softmax_max, softmax_sum, scale_value,
        query_rope=query_rope,
        key_rope=key_rope,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_klen=actual_seq_klen,
        layout="TND",
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print(f"  成功!")
    print(f"  d_query_index: shape={d_qi.shape}, max={d_qi.abs().max().item():.6f}")
    print(f"  d_key_index: shape={d_ki.shape}, max={d_ki.abs().max().item():.6f}")
    print(f"  d_weights: shape={dw.shape}, max={dw.abs().max().item():.6f}")
    print(f"  loss: shape={loss.shape}, value={loss.item():.6f}")
except Exception as e:
    print(f"  失败: {e}")
    import traceback
    traceback.print_exc()

# Also try BSND layout (unsqueeze B=1)
print("\n调用 npu_sparse_lightning_indexer_grad_kl_loss (BSND)...")
try:
    query_b = query.unsqueeze(0)
    key_b = key.unsqueeze(0)
    query_index_b = query_index.unsqueeze(0)
    key_index_b = key_index.unsqueeze(0)
    weights_b = weights.unsqueeze(0)
    sparse_indices_b = sparse_indices.unsqueeze(0)
    query_rope_b = query_rope.unsqueeze(0)
    key_rope_b = key_rope.unsqueeze(0)
    # softmaxMax/Sum: TND (N2,T1,G) -> BSND (B,N2,T1,G)
    softmax_max_b = softmax_max.unsqueeze(0)
    softmax_sum_b = softmax_sum.unsqueeze(0)
    d_qi2, d_ki2, dw2, loss2 = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        query_b, key_b, query_index_b, key_index_b, weights_b, sparse_indices_b,
        softmax_max_b, softmax_sum_b, scale_value,
        query_rope=query_rope_b,
        key_rope=key_rope_b,
        layout="BSND",
        sparse_mode=3,
    )
    torch.npu.synchronize()
    print(f"  成功!")
    print(f"  d_query_index: shape={d_qi2.shape}, max={d_qi2.abs().max().item():.6f}")
    print(f"  d_key_index: shape={d_ki2.shape}, max={d_ki2.abs().max().item():.6f}")
    print(f"  d_weights: shape={dw2.shape}, max={dw2.abs().max().item():.6f}")
    print(f"  loss: shape={loss2.shape}, value={loss2.item():.6f}")
except Exception as e:
    print(f"  失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)