"""
test_topk.py - Test MQA logits + TopK fused kernel
"""
import torch
import math

def reference_impl(q, kv, kv_sf, weights, cu_start, cu_end, sparse_count):
    """PyTorch reference implementation (non-paged)"""
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

def test_non_paged():
    import deep_gemm
    print("=== Test: Non-paged MQA logits + TopK ===")
    # QLI params: batch=156, s1=4, n1=64, d=128, s2=8192, topk=512, block_size=64
    batch_size = 156
    s1 = 4
    seq_len = batch_size * s1           # 624
    seq_len_kv = 8192
    num_heads = 64
    head_dim = 128
    sparse_count = 512
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
    print(f"  Output dtype: {sparse_indices.dtype}")
    print(f"  First row (top 10): {sparse_indices[0, :10].tolist()}")

    ref_indices = reference_impl(q, kv_data, kv_sf, weights, cu_start, cu_end, sparse_count)
    match_count = 0
    total_count = 0
    for i in range(min(20, seq_len)):
        kernel_set = set(sparse_indices[i].tolist()) - {-1}
        ref_set = set(ref_indices[i].tolist()) - {-1}
        match_count += len(kernel_set & ref_set)
        total_count += len(ref_set)
    accuracy = match_count / max(total_count, 1)
    print(f"  TopK accuracy: {accuracy:.2%}")
    print(f"  {'PASSED' if accuracy > 0.95 else 'FAILED'}")
    return accuracy > 0.95

def reference_impl_paged(q, kv_data, kv_sf, weights, context_lens, block_table, block_kv, sparse_count):
    """PyTorch reference implementation for paged KV cache"""
    batch_size, next_n, num_heads, head_dim = q.shape
    q_f32 = q.float()
    sparse_indices = torch.full((batch_size * next_n, sparse_count), -1, dtype=torch.int32, device='cuda')
    for b in range(batch_size):
        for n in range(next_n):
            q_idx = b * next_n + n
            ctx_len = int(context_lens[b, n])
            if ctx_len <= 0:
                continue
            q_i = q_f32[b, n]
            num_blocks = (ctx_len + block_kv - 1) // block_kv
            kv_blocks = block_table[b, :num_blocks]
            kv_flat = kv_data[kv_blocks, :block_kv].reshape(-1, head_dim)[:ctx_len]
            sf_flat = kv_sf[kv_blocks, :block_kv].reshape(-1)[:ctx_len]
            kv_f32 = kv_flat.float() * sf_flat.unsqueeze(1).float()
            scores = torch.matmul(q_i, kv_f32.T)
            scores = torch.relu(scores)
            scores = scores * weights[q_idx].unsqueeze(1)
            logits_i = scores.sum(dim=0)
            actual_k = min(sparse_count, ctx_len)
            if actual_k > 0:
                _, topk_local = torch.topk(logits_i, actual_k)
                sparse_indices[q_idx, :actual_k] = topk_local.to(torch.int32)
    return sparse_indices

def test_paged():
    import deep_gemm
    print("\n=== Test: Paged MQA logits + TopK ===")
    # QLI params: batch=156, s1=4, n1=64, d=128, s2=8192, topk=512, block_size=64
    batch_size = 156
    next_n = 4
    num_heads = 64
    head_dim = 128
    block_kv = 64
    max_context_len = 8192
    sparse_count = 512
    device = 'cuda'

    num_kv_blocks_per_batch = (max_context_len + block_kv - 1) // block_kv   # 128
    num_kv_blocks = batch_size * num_kv_blocks_per_batch                      # 19968

    q = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_data = torch.randn(num_kv_blocks, block_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_sf = torch.ones(num_kv_blocks, block_kv, device=device, dtype=torch.float32) * 0.1
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)
    context_lens = torch.full((batch_size, next_n), max_context_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_kv_blocks, device=device, dtype=torch.int32).reshape(batch_size, -1)

    # Host expects: each block = [KV data (block_kv*head_dim bytes) | SF (block_kv*4 bytes)]
    # stride(0) = block_kv * (head_dim + sf_bytes), stride(1) = head_dim + sf_bytes, stride(3) = 1
    # But KV data stride(1) = head_dim, SF starts at offset block_kv * head_dim
    # So we need a custom layout, not standard reshape
    block_bytes = block_kv * (head_dim + 4)
    fused_kv_raw = torch.empty(num_kv_blocks, block_bytes, device=device, dtype=torch.uint8)
    fused_kv_raw[:, :block_kv * head_dim] = kv_data.view(torch.uint8).reshape(num_kv_blocks, -1)
    fused_kv_raw[:, block_kv * head_dim:] = kv_sf.view(torch.uint8).reshape(num_kv_blocks, -1)
    fused_kv = fused_kv_raw.reshape(num_kv_blocks, block_kv, 1, head_dim + 4)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms=num_sms)

    sparse_indices = deep_gemm.fp8_paged_mqa_logits_with_topk(
        q, fused_kv, weights,
        context_lens, block_table, schedule_meta,
        max_context_len, sparse_count)

    print(f"  Output shape: {sparse_indices.shape}")
    print(f"  Output dtype: {sparse_indices.dtype}")
    print(f"  First row (top 10): {sparse_indices[0, :10].tolist()}")

    ref_indices = reference_impl_paged(
        q, kv_data, kv_sf, weights, context_lens, block_table, block_kv, sparse_count)

    match_count = 0
    total_count = 0
    for i in range(batch_size * next_n):
        kernel_set = set(sparse_indices[i].tolist()) - {-1}
        ref_set = set(ref_indices[i].tolist()) - {-1}
        match_count += len(kernel_set & ref_set)
        total_count += len(ref_set)
    accuracy = match_count / max(total_count, 1)
    print(f"  TopK accuracy: {accuracy:.2%}")
    print(f"  {'PASSED' if accuracy > 0.95 else 'FAILED'}")
    return accuracy > 0.95

if __name__ == '__main__':
    test_paged()
