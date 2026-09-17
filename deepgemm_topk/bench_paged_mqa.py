"""bench_paged_mqa.py — Benchmark DeepGEMM paged MQA logits (official API)
Tests multiple shapes matching official test_attention.py ranges.
"""
import torch
import deep_gemm

def bench(q, fused_kv, weights, context_lens, block_table, schedule_meta,
          max_context_len, warmup=10, iters=100):
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
    return start.elapsed_time(end) / iters * 1000  # us

def make_inputs(batch_size, next_n, num_heads, head_dim, block_kv, avg_kv, device='cuda'):
    context_len = avg_kv
    num_blocks_per_batch = (context_len + block_kv - 1) // block_kv
    num_total_blocks = batch_size * num_blocks_per_batch

    q = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(torch.float8_e4m3fn)
    kv = torch.randn(num_total_blocks, block_kv, 1, head_dim, device=device).to(torch.bfloat16)
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    # Cast KV to FP8 + per-token float scale
    kv_amax = kv.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    kv_sf = kv_amax / 448.0
    kv_fp8 = (kv * (1.0 / kv_sf)).to(torch.float8_e4m3fn)

    block_bytes = block_kv * (head_dim + 4)
    fused_kv = torch.empty(num_total_blocks, block_kv, 1, head_dim + 4, device=device, dtype=torch.uint8)
    fused_kv[:, :, 0, :head_dim] = kv_fp8.view(torch.uint8).reshape(num_total_blocks, block_kv, head_dim)
    fused_kv[:, :, 0, head_dim:head_dim+4] = kv_sf.view(torch.uint8).reshape(num_total_blocks, block_kv, 4)

    context_lens = torch.full((batch_size, next_n), context_len, device=device, dtype=torch.int32)
    block_table = torch.arange(num_total_blocks, device=device, dtype=torch.int32).reshape(batch_size, -1)

    return q, fused_kv, weights, context_lens, block_table, context_len

def main():
    device = 'cuda'
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU: {torch.cuda.get_device_name(0)}, SMs: {num_sms}")
    print(f"{'Config':<70} {'us':>6} {'TFLOPS':>7} {'GB/s':>6} {'MFU%':>5}")
    print("-" * 100)

    # Official test shapes + your custom shape
    configs = [
        # (batch, next_n, heads, head_dim, block_kv, avg_kv, label)
        (256, 1, 64, 128, 64, 8192,  "official: BSZ=256 N=1 H=64 D=128 KV=8K blk=64"),
        (256, 1, 64, 128, 128, 8192, "official: BSZ=256 N=1 H=64 D=128 KV=8K blk=128"),
        (256, 1, 64, 128, 32,  8192, "official: BSZ=256 N=1 H=64 D=128 KV=8K blk=32"),
        (256, 2, 64, 128, 64,  8192, "official: BSZ=256 N=2 H=64 D=128 KV=8K blk=64"),
        (256, 6, 64, 128, 64,  8192, "official: BSZ=256 N=6 H=64 D=128 KV=8K blk=64"),
        (256, 1, 32, 128, 64,  8192, "official: BSZ=256 N=1 H=32 D=128 KV=8K blk=64"),
        (256, 1, 64, 64,  64,  8192, "official: BSZ=256 N=1 H=64 D=64  KV=8K blk=64"),
        (256, 1, 64, 128, 64,  65536,"official: BSZ=256 N=1 H=64 D=128 KV=64K blk=64"),
        (4096,1, 64, 128, 64,  8192, "official: BSZ=4096 N=1 H=64 D=128 KV=8K blk=64"),
        (156, 4, 64, 128, 64,  8192, "yours:    BSZ=156 N=4 H=64 D=128 KV=8K blk=64"),
    ]

    B200_PEAK_FP8 = 2250.0  # TFLOPS
    HBM_BW = 8.0  # TB/s

    for batch, next_n, heads, head_dim, block_kv, avg_kv, label in configs:
        try:
            q, fused_kv, weights, ctx_lens, block_table, max_ctx = make_inputs(
                batch, next_n, heads, head_dim, block_kv, avg_kv, device)

            num_q = batch * next_n
            sm = deep_gemm.get_paged_mqa_logits_metadata(
                ctx_lens, block_kv, num_sms)

            us = bench(q, fused_kv, weights, ctx_lens, block_table, sm, max_ctx)

            # FLOPS: 2 * sum(context_lens) * next_n * heads * head_dim
            # For uniform context_len: 2 * batch * avg_kv * next_n * heads * head_dim
            flops = 2 * batch * avg_kv * next_n * heads * head_dim
            tflops = flops / us / 1e6

            # Bytes: Q + KV + weights + logits output
            kv_bytes = batch * avg_kv * (head_dim + 4)  # FP8 + float SF
            q_bytes = num_q * heads * head_dim  # FP8
            w_bytes = num_q * heads * 4  # float32
            out_bytes = num_q * avg_kv * 4  # float32 logits
            total_bytes = kv_bytes + q_bytes + w_bytes + out_bytes
            gbs = total_bytes / us / 1e3

            mfu = tflops / B200_PEAK_FP8 * 100

            print(f"{label:<70} {us:>6.0f} {tflops:>7.0f} {gbs:>6.0f} {mfu:>5.1f}")
        except Exception as e:
            print(f"{label:<70} ERROR: {e}")

    print("-" * 100)

if __name__ == '__main__':
    main()
