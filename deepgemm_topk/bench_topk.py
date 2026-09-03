"""bench_topk.py — Performance benchmark + MFU for paged & non-paged MQA logits + TopK"""
import torch
import deep_gemm

B200_PEAK_FP8_TFLOPS = 2250.0

def bench_paged(q, fused_kv, weights, context_lens, block_table,
                schedule_meta, max_context_len, sparse_count, warmup=10, iters=100):
    for _ in range(warmup):
        deep_gemm.fp8_paged_mqa_logits_with_topk(
            q, fused_kv, weights,
            context_lens, block_table, schedule_meta,
            max_context_len, sparse_count)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        deep_gemm.fp8_paged_mqa_logits_with_topk(
            q, fused_kv, weights,
            context_lens, block_table, schedule_meta,
            max_context_len, sparse_count)
    end.record()
    torch.cuda.synchronize()

    avg_us = start.elapsed_time(end) / iters * 1000
    return avg_us

def bench_paged_phase1(q, fused_kv, weights, context_lens, block_table,
                       schedule_meta, max_context_len, warmup=10, iters=100):
    for _ in range(warmup):
        deep_gemm.fp8_paged_mqa_logits(
            q, fused_kv, weights,
            context_lens, block_table, schedule_meta,
            max_context_len)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        deep_gemm.fp8_paged_mqa_logits(
            q, fused_kv, weights,
            context_lens, block_table, schedule_meta,
            max_context_len)
    end.record()
    torch.cuda.synchronize()

    avg_us = start.elapsed_time(end) / iters * 1000
    return avg_us

def bench_non_paged(q, kv, kv_sf, weights, cu_start, cu_end,
                    sparse_count, warmup=10, iters=100):
    for _ in range(warmup):
        deep_gemm.fp8_mqa_logits_with_topk(
            q, (kv, kv_sf), weights, cu_start, cu_end,
            sparse_count=sparse_count, max_seqlen_k=0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        deep_gemm.fp8_mqa_logits_with_topk(
            q, (kv, kv_sf), weights, cu_start, cu_end,
            sparse_count=sparse_count, max_seqlen_k=0)
    end.record()
    torch.cuda.synchronize()

    avg_us = start.elapsed_time(end) / iters * 1000
    return avg_us

def bench_non_paged_phase1(q, kv, kv_sf, weights, cu_start, cu_end,
                           warmup=10, iters=100):
    for _ in range(warmup):
        deep_gemm.fp8_mqa_logits(
            q, (kv, kv_sf), weights, cu_start, cu_end)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        deep_gemm.fp8_mqa_logits(
            q, (kv, kv_sf), weights, cu_start, cu_end)
    end.record()
    torch.cuda.synchronize()

    avg_us = start.elapsed_time(end) / iters * 1000
    return avg_us

def compute_flops(batch, next_n, num_heads, head_dim, seq_len_kv):
    num_q_tokens = batch * next_n
    matmul_flops = num_q_tokens * 2 * seq_len_kv * num_heads * head_dim
    reduce_flops = num_q_tokens * 2 * seq_len_kv * num_heads
    return matmul_flops + reduce_flops

def print_result(label, us, flops):
    tflops = flops / us * 1e-6
    mfu = tflops / B200_PEAK_FP8_TFLOPS * 100
    print(f"  {label:<30} {us:>8.1f} us  {tflops:>7.1f} TFLOPS  MFU={mfu:>5.1f}%")

def main():
    device = 'cuda'
    batch_size = 156
    next_n = 4
    num_heads = 64
    head_dim = 128
    block_kv = 64
    seq_len_kv = 8192
    sparse_count = 512

    num_q_tokens = batch_size * next_n
    flops = compute_flops(batch_size, next_n, num_heads, head_dim, seq_len_kv)

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # ===== Shared inputs =====
    q_paged = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(num_q_tokens, num_heads, device=device, dtype=torch.float32)

    # ===== Paged inputs =====
    num_kv_blocks_per_batch = (seq_len_kv + block_kv - 1) // block_kv
    num_kv_blocks = batch_size * num_kv_blocks_per_batch
    kv_data = torch.randn(num_kv_blocks, block_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_sf = torch.ones(num_kv_blocks, block_kv, device=device, dtype=torch.float32) * 0.1
    context_lens = torch.full((batch_size, next_n), seq_len_kv, device=device, dtype=torch.int32)
    block_table = torch.arange(num_kv_blocks, device=device, dtype=torch.int32).reshape(batch_size, -1)

    block_bytes = block_kv * (head_dim + 4)
    fused_kv_raw = torch.empty(num_kv_blocks, block_bytes, device=device, dtype=torch.uint8)
    fused_kv_raw[:, :block_kv * head_dim] = kv_data.view(torch.uint8).reshape(num_kv_blocks, -1)
    fused_kv_raw[:, block_kv * head_dim:] = kv_sf.view(torch.uint8).reshape(num_kv_blocks, -1)
    fused_kv = fused_kv_raw.reshape(num_kv_blocks, block_kv, 1, head_dim + 4)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms=num_sms)

    # ===== Non-paged inputs =====
    q_non_paged = torch.randn(num_q_tokens, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_non_paged = torch.randn(seq_len_kv, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_sf_non_paged = torch.ones(seq_len_kv, device=device, dtype=torch.float32) * 0.1
    cu_start = torch.zeros(num_q_tokens, device=device, dtype=torch.int32)
    cu_end = torch.full((num_q_tokens,), seq_len_kv, device=device, dtype=torch.int32)

    print("=" * 70)
    print(f"  batch={batch_size}, next_n={next_n}, heads={num_heads}, dim={head_dim}")
    print(f"  seq_len_kv={seq_len_kv}, sparse_count={sparse_count}, block_kv={block_kv}")
    print(f"  num_q_tokens={num_q_tokens}, num_sms={num_sms}")
    print(f"  FLOPS={flops/1e9:.1f}G")
    print("=" * 70)

    # ===== Paged =====
    print("\n[Paged Mode]")
    try:
        p1 = bench_paged_phase1(q_paged, fused_kv, weights, context_lens, block_table,
                                schedule_meta, seq_len_kv)
        print_result("Phase 1 (no TopK)", p1, flops)
    except Exception as e:
        print(f"  Phase 1: ERROR: {e}")
        p1 = None

    total_p = bench_paged(q_paged, fused_kv, weights, context_lens, block_table,
                          schedule_meta, seq_len_kv, sparse_count)
    print_result("Phase 1+2 (TopK)", total_p, flops)
    if p1:
        p2 = total_p - p1
        print_result("Phase 2 only", p2, flops)
        print(f"  {'Overhead':<30} {p2/p1:>8.1f}x of Phase 1")

    # ===== Non-paged =====
    print("\n[Non-paged Mode]")
    try:
        n1 = bench_non_paged_phase1(q_non_paged, kv_non_paged, kv_sf_non_paged, weights,
                                    cu_start, cu_end)
        print_result("Phase 1 (no TopK)", n1, flops)
    except Exception as e:
        print(f"  Phase 1: ERROR: {e}")
        n1 = None

    total_n = bench_non_paged(q_non_paged, kv_non_paged, kv_sf_non_paged, weights,
                              cu_start, cu_end, sparse_count)
    print_result("Phase 1+2 (TopK)", total_n, flops)
    if n1:
        n2 = total_n - n1
        print_result("Phase 2 only", n2, flops)
        print(f"  {'Overhead':<30} {n2/n1:>8.1f}x of Phase 1")

    print()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'l2':
        bench_l2_reuse_experiment()
    else:
        main()


def bench_l2_reuse_experiment():
    """A/B: next_n=4 shared-KV vs next_n=1 no-sharing (equal work), Phase 1 only."""
    import deep_gemm
    device = 'cuda'
    num_heads, head_dim, block_kv = 64, 128, 64
    max_ctx, sparse_count = 8192, 512
    nq = 624  # total tokens, equal in both cases

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    def make_meta(ctx_lens, bk):
        return deep_gemm.get_paged_mqa_logits_metadata(ctx_lens, bk, num_sms)

    # ===== A: 156 batches x 4 tokens, per-batch shared KV (1MB each, 156MB total) =====
    ba = 156
    nb_a = (max_ctx + block_kv - 1) // block_kv      # 128 blocks/batch
    nblocks_a = ba * nb_a
    q_a = torch.randn(ba, 4, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    kv_a = torch.randn(nblocks_a, block_kv, 1, head_dim, device=device).to(torch.bfloat16)
    kv_amax = kv_a.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf_a = (kv_amax / 448.0)
    kv_fp8_a = (kv_a * (1.0 / sf_a)).to(torch.float8_e4m3fn)
    fused_a = torch.empty(nblocks_a, block_kv, 1, head_dim + 4, device=device, dtype=torch.uint8)
    fused_a[:, :, 0, :head_dim] = kv_fp8_a.view(torch.uint8).reshape(nblocks_a, block_kv, head_dim)
    fused_a[:, :, 0, head_dim:head_dim+4] = sf_a.view(torch.uint8).reshape(nblocks_a, block_kv, 4)
    w_a = torch.randn(ba * 4, num_heads, device=device)
    cl_a = torch.full((ba, 4), max_ctx, device=device, dtype=torch.int32)
    bt_a = torch.arange(nblocks_a, device=device, dtype=torch.int32).reshape(ba, -1)
    meta_a = make_meta(cl_a, block_kv)

    # ===== B: 624 batches x 1 token, each own KV (1MB each, 624MB total) =====
    bb = 624
    nb_b = (max_ctx + block_kv - 1) // block_kv
    nblocks_b = bb * nb_b
    ku = torch.randn(nblocks_b, block_kv, 1, head_dim, device=device).to(torch.bfloat16)
    ku_amax = ku.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf_b = (ku_amax / 448.0)
    ku_fp8 = (ku * (1.0 / sf_b)).to(torch.float8_e4m3fn)
    fused_b = torch.empty(nblocks_b, block_kv, 1, head_dim + 4, device=device, dtype=torch.uint8)
    fused_b[:, :, 0, :head_dim] = ku_fp8.view(torch.uint8).reshape(nblocks_b, block_kv, head_dim)
    fused_b[:, :, 0, head_dim:head_dim+4] = sf_b.view(torch.uint8).reshape(nblocks_b, block_kv, 4)
    q_b = torch.randn(bb, 1, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    w_b = torch.randn(bb * 1, num_heads, device=device)
    cl_b = torch.full((bb, 1), max_ctx, device=device, dtype=torch.int32)
    bt_b = torch.arange(nblocks_b, device=device, dtype=torch.int32).reshape(bb, -1)
    meta_b = make_meta(cl_b, block_kv)

    def t1(q, kv, w, cl, bt, meta):
        for _ in range(5):
            deep_gemm.fp8_paged_mqa_logits(q, kv, w, cl, bt, meta, max_ctx)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            deep_gemm.fp8_paged_mqa_logits(q, kv, w, cl, bt, meta, max_ctx)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / 50 * 1000

    ta = t1(q_a, fused_a, w_a, cl_a, bt_a, meta_a)
    tb = t1(q_b, fused_b, w_b, cl_b, bt_b, meta_b)
    print(f"[L2REUSE] A(next_n=4 shared KV, 156MB): {ta:.1f} us")
    print(f"[L2REUSE] B(next_n=1 no-share,   624MB): {tb:.1f} us")
    print(f"[L2REUSE] A/B = {ta/tb:.2f}  (若显著<1 → 复用已在生效; 若≈1 → 有优化空间)")
