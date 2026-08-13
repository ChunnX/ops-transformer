"""test_topk_debug.py — Debug TopK accuracy issue"""
import torch
import deep_gemm

def reference_impl(q, kv, kv_sf, weights, cu_start, cu_end, sparse_count):
    seq_len, num_heads, head_dim = q.shape
    kv_f32 = kv.float() * kv_sf.unsqueeze(1).float()
    q_f32 = q.float()
    sparse_indices = torch.full((seq_len, sparse_count), -1, dtype=torch.int32, device='cuda')
    for i in range(seq_len):
        ks = int(cu_start[i])
        ke = int(cu_end[i])
        if ke <= ks:
            continue
        q_i = q_f32[i]
        kv_slice = kv_f32[ks:ke]
        scores = torch.matmul(q_i, kv_slice.T)
        scores = torch.relu(scores)
        scores = scores * weights[i].unsqueeze(1)
        logits_i = scores.sum(dim=0)
        actual_k = min(sparse_count, ke - ks)
        if actual_k > 0:
            _, topk_local = torch.topk(logits_i, actual_k)
            sparse_indices[i, :actual_k] = (topk_local + ks).to(torch.int32)
    return sparse_indices

def test_simple():
    print("=== Debug Test: small seq, large KV ===")
    seq_len = 4
    seq_len_kv = 1024
    num_heads = 64
    head_dim = 128
    sparse_count = 16
    device = 'cuda'

    q = torch.randn(seq_len, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_data = torch.randn(seq_len_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_sf = torch.ones(seq_len_kv, device=device, dtype=torch.float32) * 0.1
    weights = torch.randn(seq_len, num_heads, device=device, dtype=torch.float32)

    cu_start = torch.zeros(seq_len, device=device, dtype=torch.int32)
    cu_end = torch.full((seq_len,), seq_len_kv, device=device, dtype=torch.int32)

    sparse_indices = deep_gemm.fp8_mqa_logits_with_topk(
        q, (kv_data, kv_sf), weights, cu_start, cu_end,
        sparse_count=sparse_count, max_seqlen_k=0)

    print(f"  Output shape: {sparse_indices.shape}")
    ref_indices = reference_impl(q, kv_data, kv_sf, weights, cu_start, cu_end, sparse_count)

    for i in range(seq_len):
        kernel_list = sparse_indices[i].tolist()
        ref_list = ref_indices[i].tolist()
        kernel_set = set(kernel_list) - {-1}
        ref_set = set(ref_list) - {-1}
        overlap = len(kernel_set & ref_set)
        total = len(ref_set)

        # Check for out-of-range indices
        bad_indices = [idx for idx in kernel_list if idx >= seq_len_kv or idx < -1]
        print(f"  Token {i}: overlap={overlap}/{total}, bad_indices={bad_indices}")
        print(f"    kernel (first 8): {kernel_list[:8]}")
        print(f"    ref    (first 8): {ref_list[:8]}")

if __name__ == '__main__':
    test_simple()
