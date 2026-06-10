"""Stem block scoring + mask generation — quant_type=0 verification & benchmark.

Companion to `test_stem.py`.  test_stem.py exercises the legacy
`QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR` path; this file exercises the new
`QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD` path:

  - K-scale: 4D fp32 [num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale],
             effective semantic = per-KV-token fp32.
  - V-scale: per-head fp32 vector [num_head_kv].

Verification modes:

  --verify-prep     KV prep + Q prep correctness + speed (vs FP32 PyTorch ref)
  --verify-gemm     GEMM kernel accuracy + speed vs PyTorch matmul
  --verify-oam      Full OAM pipeline accuracy vs FP32 ref + CUDA-only speed
  --verify-tpd      TPD mask accuracy + speed (TPD is scale-agnostic; kept for
                    end-to-end coverage)
  --verify-stemAll  End-to-end OAM+TPD pipeline accuracy vs FP32 ref + speed
  --chunked         Chunked-prefill D-semantic invariance (quant_type=K_PERTOKEN_V_PERHEAD)

Triton baselines are intentionally omitted from --verify-oam / --verify-stemAll
because the upstream vllm Triton implementation only supports per-tensor K-scale.

Usage:
  python test_stem_scale.py --verify-prep --seq-lens 1024 4096 16384 65536
  python test_stem_scale.py --verify-gemm
  python test_stem_scale.py --verify-oam
  python test_stem_scale.py --verify-tpd
  python test_stem_scale.py --verify-stemAll
  python test_stem_scale.py --chunked
"""

import argparse
import math
import os
import sys
import types

# Try real hpc first; fall back to mock so vllm.stem can still be imported
try:
    import hpc

    _HPC_AVAILABLE = True
except (ImportError, AssertionError, OSError):
    sys.modules["hpc"] = types.ModuleType("hpc")
    _HPC_AVAILABLE = False

# Use local vllm source if present (kept for parity with test_stem.py;
# only needed by --verify-tpd and --verify-stemAll which use the Python
# `_generate_sparse_mask_per_req` reference).
_LOCAL_VLLM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm")
if os.path.isdir(_LOCAL_VLLM):
    sys.path.insert(0, _LOCAL_VLLM)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:
    from vllm.v1.attention.backends.stem import _generate_sparse_mask_per_req
except Exception:
    # Allow file to import even when vllm is unavailable; verify-tpd/stemAll
    # will then error explicitly if attempted.
    _generate_sparse_mask_per_req = None


# ---------------------------------------------------------------------------
# Shared sequence-length lists across all verification / benchmark functions
# ---------------------------------------------------------------------------
VERIFY_SEQ_LENS = [
    14, 190, 500, 1030, 2999, 5123, 7777, 12345, 24576, 51200, 99999, 131067,
    524199, 1048499,
]
BENCH_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


# ---------------------------------------------------------------------------
# Tolerance presets (torch.allclose style: pass iff |a-b| <= atol + rtol*|b|)
# ---------------------------------------------------------------------------
TOL_PREP = dict(atol=1e-3, rtol=0.0)
TOL_GEMM = dict(atol=0.01, rtol=0.05)
TOL_OAM = dict(atol=0.01, rtol=0.05)


def allclose_check(ref: torch.Tensor, test: torch.Tensor, atol: float, rtol: float):
    """Compare two tensors on finite entries using allclose-style tolerance."""
    ref_f = ref.float()
    test_f = test.float()
    finite = torch.isfinite(ref_f) & torch.isfinite(test_f)
    n_finite = finite.sum().item()
    n_total = ref_f.numel()

    if n_finite == 0:
        return dict(
            pass_rate=0.0, max_abs=float("nan"), mean_abs=float("nan"),
            max_rel=float("nan"), mean_rel=float("nan"),
            n_finite=0, n_total=n_total, all_pass=False,
            inf_match=(~torch.isfinite(ref_f)).eq(~torch.isfinite(test_f)).all().item(),
        )

    r = ref_f[finite]
    t = test_f[finite]
    diff = (r - t).abs()
    threshold = atol + rtol * r.abs()
    within = diff <= threshold

    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    r_abs = r.abs().clamp(min=1e-8)
    rel = diff / r_abs
    max_rel = rel.max().item()
    mean_rel = rel.mean().item()

    pass_rate = within.float().mean().item()
    inf_match = (~torch.isfinite(ref_f)).eq(~torch.isfinite(test_f)).all().item()

    return dict(
        pass_rate=pass_rate, max_abs=max_abs, mean_abs=mean_abs,
        max_rel=max_rel, mean_rel=mean_rel,
        n_finite=n_finite, n_total=n_total,
        all_pass=(pass_rate > 0.999) and inf_match,
        inf_match=inf_match,
    )


def _print_check(stats, label=""):
    """Pretty-print allclose_check results."""
    tag = f"  {label}" if label else "  "
    pct = stats["pass_rate"] * 100
    status = "PASS" if stats["all_pass"] else "FAIL"
    print(f"{tag}allclose: {pct:.2f}%  max_abs={stats['max_abs']:.4e}  "
          f"mean_abs={stats['mean_abs']:.4e}  max_rel={stats['max_rel']:.4e}  "
          f"mean_rel={stats['mean_rel']:.4e}  "
          f"inf_ok={'Y' if stats['inf_match'] else 'N'}  "
          f"[{status}]")


# ---------------------------------------------------------------------------
# Benchmark utility
# ---------------------------------------------------------------------------
def benchmark_fn(fn, warmup=10, iters=50):
    """CUDA-event timing.  Returns average latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        fn()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / iters


# ====================================================================
# Data setup — quant_type=0 layout (4D fp32 K-scale + per-head V-scale)
# ====================================================================
def _setup_paged_fp8_data(
    num_req,
    seq_len,
    num_head_q,
    num_head_kv,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
):
    """Random paged FP8 data with K_PERTOKEN_V_PERHEAD scale layout."""
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    total_tokens = num_req * seq_len

    q_fp8 = (
        torch.randn(
            total_tokens, num_head_q, dim_qk, dtype=torch.bfloat16, device=device
        )
        / math.sqrt(dim_qk)
    ).to(T_fp8)

    num_blocks_per_req = (seq_len + kv_block_size - 1) // kv_block_size
    total_blocks = num_req * num_blocks_per_req

    kcache_fp8 = (
        torch.randn(
            total_blocks, kv_block_size, num_head_kv, dim_qk,
            dtype=torch.bfloat16, device=device,
        )
        / math.sqrt(dim_qk)
    ).to(T_fp8)
    vcache_fp8 = torch.randn(
        total_blocks, kv_block_size, num_head_kv, dim_v,
        dtype=torch.bfloat16, device=device,
    ).to(T_fp8)

    kv_indices = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(
        num_req, num_blocks_per_req
    )

    max_seqlen_pad128 = ((seq_len + 127) // 128) * 128
    qscale = torch.ones(
        num_req, num_head_q, max_seqlen_pad128, dtype=torch.float32, device=device
    )

    # ------- New scale layout -------
    # K-scale: 4D fp32 [num_blocks, kScaleBlockSize, num_head_kv, num_dim_scale]
    #   * kScaleBlockSize = kv_block_size / 32 (1 or 2 for our supported block sizes)
    #   * num_dim_scale  = dim_v / sizeof(float) = 32 for dim_v=128
    #   * effective semantic: 1 fp32 K-scale per KV token
    num_dim_scale = dim_v // 4
    assert kv_block_size % num_dim_scale == 0, (
        f"kv_block_size={kv_block_size} must be a multiple of num_dim_scale={num_dim_scale}"
    )
    scale_block_size = kv_block_size // num_dim_scale
    kscale_fp32 = (
        torch.abs(
            torch.randn(
                total_blocks, scale_block_size, num_head_kv, num_dim_scale,
                dtype=torch.float32, device=device,
            )
        )
        / 10.0
    )
    # V-scale: per-head fp32 vector [num_head_kv]
    vscale = torch.abs(torch.ones(num_head_kv, dtype=torch.float32, device=device))

    seqlens = torch.full((num_req,), seq_len, dtype=torch.int32, device=device)
    cu_q_seqlens = torch.zeros(num_req + 1, dtype=torch.int32, device=device)
    cu_q_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    kv_seqlens = seqlens.clone()

    return dict(
        q_fp8=q_fp8,
        kcache_fp8=kcache_fp8,
        vcache_fp8=vcache_fp8,
        qscale=qscale,
        kscale_fp32=kscale_fp32,
        vscale=vscale,
        kv_indices=kv_indices,
        cu_q_seqlens=cu_q_seqlens,
        kv_seqlens=kv_seqlens,
        kv_block_size=kv_block_size,
    )


# ====================================================================
# k_schedule helper (mirrors stem.py logic; identical to test_stem.py)
# ====================================================================
def _compute_k_val(q_block_num):
    """Return the k_val from the default k_schedule for a given q_block_num."""
    if q_block_num < 56:
        return q_block_num
    elif q_block_num < 160:
        return int(q_block_num * 0.2 + 30)
    else:
        return int(q_block_num * 0.1 + 30)


# ====================================================================
# Mask generation with optional k_val override
# (identical to test_stem.py — TPD and mask gen are scale-agnostic)
# ====================================================================
def _generate_sparse_mask_with_budget(
    block_logits: torch.Tensor,
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    block_size: int,
    device: torch.device,
    k_val_override=None,
):
    """Mask generation from stem.py with optional k_val budget override."""
    num_req, num_heads, max_Qb, max_Kb = block_logits.shape
    approx_simple_mask = torch.zeros_like(block_logits, dtype=torch.bool)

    for i in range(num_req):
        PAD_ALIGN = 128
        qi_padded = ((q_seq_lens[i].item() + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        ki_padded = ((kv_seq_lens[i].item() + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        qi_blocks = min((qi_padded + block_size - 1) // block_size, max_Qb)
        ki_blocks = min((ki_padded + block_size - 1) // block_size, max_Kb)

        if qi_blocks == 0 or ki_blocks == 0:
            continue

        req_logits = block_logits[i:i + 1, :, :qi_blocks, :ki_blocks]

        alpha = 1.0
        q_block_num = qi_blocks
        k_block_num = ki_blocks

        if k_val_override is not None:
            k_val = min(k_val_override, k_block_num)
        else:
            if q_block_num < 56:
                k_val = q_block_num
            elif q_block_num < 160:
                k_val = int((q_block_num * 0.2) + 30)
            else:
                k_val = int(q_block_num * 0.1) + 30

        sched = torch.full((q_block_num,), k_val, dtype=torch.long, device=device)
        if q_block_num > k_val:
            decay_len = q_block_num - k_val
            k_end = k_val * alpha
            ideal_vals = torch.linspace(
                float(k_val), float(k_end),
                steps=decay_len, dtype=torch.float64, device=device,
            )
            sched[k_val:] = torch.clamp(
                torch.floor(ideal_vals).long(), min=1, max=k_val,
            )

        growth = max(1.0, alpha)
        needed_k = int(min(math.ceil(sched[0].item() * growth), k_block_num))
        budget = sched.view(1, 1, -1, 1)

        req_mask = torch.zeros(
            (1, num_heads, q_block_num, k_block_num),
            device=device, dtype=torch.bool,
        )

        if needed_k > 0:
            topk_vals, topk_idx = torch.topk(req_logits, needed_k, dim=-1)
            rank = torch.arange(needed_k, device=device).view(1, 1, 1, -1)
            keep = (rank < budget) & torch.isfinite(topk_vals)

            if keep.any():
                req_mask.scatter_(3, topk_idx, keep)
                initial_blocks = 4
                window_size = 4
                req_mask[..., :initial_blocks] = True
                q_range = torch.arange(q_block_num, device=device).unsqueeze(1)
                k_range = torch.arange(k_block_num, device=device).unsqueeze(0)
                local_mask = (k_range <= q_range) & (
                    k_range > (q_range - window_size)
                )
                req_mask.masked_fill_(local_mask, True)

        approx_simple_mask[i:i + 1, :, :qi_blocks, :ki_blocks] = req_mask

    return approx_simple_mask


# ====================================================================
# Per-token K-scale gather (vectorized — must be fast for 256K seqs)
# ====================================================================
def _gather_per_token_kscale_one_req(
    kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
):
    """Materialize per-token K-scale for one request: shape [k_padded, num_head_kv].

    Padding rows (>= kv_len) get zero.  All ops vectorized — O(k_padded) memory,
    no per-token Python loop.
    """
    device = kscale_fp32.device
    num_head_kv = kscale_fp32.shape[2]
    num_kv_blocks = block_ids.shape[0]

    token_ids = torch.arange(k_padded, device=device, dtype=torch.long)
    valid_mask = token_ids < kv_len
    page_idx = (token_ids // kv_block_size).clamp(max=num_kv_blocks - 1)
    row_in_page = token_ids % kv_block_size
    phys_blocks = block_ids[page_idx].to(torch.long)
    row_chunk = row_in_page // 32
    row_offset = row_in_page % 32
    # Advanced indexing: kscale_fp32[phys_block, row_chunk, :, row_offset]
    # → shape [k_padded, num_head_kv]
    gathered = kscale_fp32[phys_blocks, row_chunk, :, row_offset]
    gathered = torch.where(
        valid_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered),
    )
    return gathered


# ====================================================================
# OAM Prep — PyTorch reference (qkpertoken_perhead_vperhead path)
# ====================================================================
def _prep_kv_reference(
    kcache_fp8: torch.Tensor,
    vcache_fp8: torch.Tensor,
    kscale_fp32: torch.Tensor,
    vscale: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    kv_block_size: int,
    lambda_mag: float = 0.3,
    stem_block_size: int = 128,
    stem_stride: int = 16,
):
    """Pure PyTorch reference matching CUDA stem_prep_paged_kv_qkpertoken_perhead_vperhead_kernel.

    K processing: per-KV-token K-scale folded into FP32 group sum (per-row scaling).
    V processing: per-head V-scale element-wise applied before L2 norm.
    Vbias path is shared with legacy (log normalize is invariant to per-head scale).

    Returns:
        kflat: [num_batch, num_head_kv, max_Kb, stem_stride * dim_qk] BF16
        v_bias: [num_batch, num_head_kv, max_Kb] FP32
    """
    device = kcache_fp8.device
    num_batch = kv_seq_lens.shape[0]
    num_head_kv = kcache_fp8.shape[2]
    dim_qk = kcache_fp8.shape[3]
    dim_v = vcache_fp8.shape[3]
    R = stem_block_size // stem_stride

    k_padded_lens = (
        (kv_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    ).to(torch.int32)
    max_k_padded = k_padded_lens.max().item()
    max_Kb = max_k_padded // stem_block_size

    kflat_out = torch.zeros(
        num_batch, num_head_kv, max_Kb, stem_stride * dim_qk,
        dtype=torch.bfloat16, device=device,
    )
    v_bias_out = torch.zeros(
        num_batch, num_head_kv, max_Kb,
        dtype=torch.float32, device=device,
    )

    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        k_padded = k_padded_lens[b].item()
        num_stem_blocks = k_padded // stem_block_size
        k_down_len = k_padded // stem_stride

        # De-page K and V into dense FP32 (zero-padded beyond kv_len)
        num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size
        block_ids = kv_indices[b, :num_kv_blocks]
        k_paged = kcache_fp8[block_ids].float()
        v_paged = vcache_fp8[block_ids].float()
        k_dense = torch.zeros(k_padded, num_head_kv, dim_qk, dtype=torch.float32, device=device)
        v_dense = torch.zeros(k_padded, num_head_kv, dim_v, dtype=torch.float32, device=device)
        actual_rows = min(kv_len, num_kv_blocks * kv_block_size)
        k_dense[:actual_rows] = k_paged.reshape(-1, num_head_kv, dim_qk)[:actual_rows]
        v_dense[:actual_rows] = v_paged.reshape(-1, num_head_kv, dim_v)[:actual_rows]

        # Per-token K-scale gather: [k_padded, num_head_kv]
        kscale_per_row = _gather_per_token_kscale_one_req(
            kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
        )

        # Apply per-token K-scale to K (broadcast on dim) before group sum
        k_scaled = (
            k_dense[:num_stem_blocks * stem_block_size]
            * kscale_per_row[:num_stem_blocks * stem_block_size].unsqueeze(-1)
        )

        for h in range(num_head_kv):
            # ---- Kflat ----
            k_h = k_scaled[:, h, :]  # [k_padded, dim_qk]
            # row index within stem block = r * stem_stride + g
            k_blocks = k_h.reshape(num_stem_blocks, R, stem_stride, dim_qk)
            k_group_sum = k_blocks.sum(dim=1)  # [num_stem_blocks, stride, dim_qk]
            k_group_rev = k_group_sum.flip(1)
            kflat_out[b, h, :num_stem_blocks] = k_group_rev.reshape(
                num_stem_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

            # ---- V L2 norm with per-head V-scale ----
            vscale_h = vscale[h].item()
            v_h = v_dense[:num_stem_blocks * stem_block_size, h, :] * vscale_h
            v_rows = v_h.reshape(k_down_len, stem_stride, dim_v)
            norms = torch.sqrt((v_rows ** 2).sum(dim=-1))  # [k_down_len, stride]
            row_ids = torch.arange(
                num_stem_blocks * stem_block_size, device=device,
            ).reshape(k_down_len, stem_stride)
            norms = torch.where(row_ids < kv_len, norms, torch.zeros_like(norms))
            v_norm_down = norms.max(dim=-1).values

            # ---- v_bias (FP32 throughout, matches kernel + scale-invariant)
            if k_down_len > 0:
                log_vals = torch.log(v_norm_down + 1e-6)
                v_mean = log_vals.mean()
                if k_down_len > 1:
                    v_std = log_vals.std(unbiased=True)
                else:
                    v_std = torch.tensor(0.0, device=device)
                inv_std = 1.0 / (v_std + 1e-6)
                normalized = (log_vals - v_mean) * inv_std
                v_final = lambda_mag * torch.relu(normalized)
                v_final_blocks = v_final[:num_stem_blocks * R].reshape(num_stem_blocks, R)
                v_bias_out[b, h, :num_stem_blocks] = v_final_blocks.mean(dim=1)

    return kflat_out, v_bias_out


def _prep_q_reference(
    q_fp8: torch.Tensor,
    qscale: torch.Tensor,
    q_seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
):
    """Pure PyTorch reference matching CUDA stem_oam_prep_varlen_q kernel.

    Q prep is identical between legacy and new quant_type paths — Q-scale is
    per-token-per-head in both cases and is independent of K/V scale layout.
    """
    device = q_fp8.device
    num_head_q = q_fp8.shape[1]
    dim_qk = q_fp8.shape[2]
    num_batch = q_seq_lens.shape[0]
    R = stem_block_size // stem_stride

    q_padded_lens = (
        (q_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    ).to(torch.int32)
    max_q_padded = q_padded_lens.max().item()
    max_Qb = max_q_padded // stem_block_size

    qflat_out = torch.zeros(
        num_batch, num_head_q, max_Qb, stem_stride * dim_qk,
        dtype=torch.bfloat16, device=device,
    )

    for b in range(num_batch):
        q_len = q_seq_lens[b].item()
        q_padded = q_padded_lens[b].item()
        num_q_blocks = q_padded // stem_block_size
        cu_off = cu_seqlens_q[b].item()

        q_dense = torch.zeros(
            q_padded, num_head_q, dim_qk, dtype=torch.float32, device=device,
        )
        q_dense[:q_len] = q_fp8[cu_off:cu_off + q_len].float()

        for h in range(num_head_q):
            q_h = q_dense[:num_q_blocks * stem_block_size, h, :]
            q_blocks = q_h.reshape(num_q_blocks, R, stem_stride, dim_qk)
            positions = torch.arange(
                num_q_blocks * stem_block_size, dtype=torch.long, device=device,
            ).reshape(num_q_blocks, R, stem_stride)
            scales = qscale[b, h, positions.clamp(max=qscale.shape[2] - 1)]
            q_group_sum = (q_blocks * scales.unsqueeze(-1)).sum(dim=1)
            qflat_out[b, h, :num_q_blocks] = q_group_sum.reshape(
                num_q_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

    return qflat_out


# ====================================================================
# OAM math reference (Frobenius GEMM, K_PERTOKEN_V_PERHEAD path)
# ====================================================================
def _oam_math_reference(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kcache_fp8: torch.Tensor,
    vcache_fp8: torch.Tensor,
    kscale_fp32: torch.Tensor,
    v_scale: torch.Tensor,
    kv_indices: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    q_seq_lens: torch.Tensor,
    stride: int,
    block_size: int,
    lambda_mag: float,
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
    kv_block_size: int,
):
    """FP32 Frobenius GEMM reference for the K_PERTOKEN_V_PERHEAD path.

    Differences from legacy `_oam_math_reference` in test_stem.py:
      - K group-sum applies per-KV-token K-scale (4D fp32 layout) before sum
      - V_bias uses per-head V-scale via this file's `_prep_kv_reference`
      - No qk_scale fusion: Q has its own per-token-per-head scale, K has its
        own per-token scale; they are independent
      - No `vbias_mode="triton"` branch (Triton path doesn't support new layout)
    """
    num_req = kv_seq_lens.shape[0]
    num_q_heads = q_fp8.shape[1]
    num_kv_heads = kcache_fp8.shape[2]
    head_dim = q_fp8.shape[2]
    R = block_size // stride  # 8

    PAD_ALIGN = 128
    q_padded_lens = (
        (q_seq_lens.to(torch.int64) + PAD_ALIGN - 1) // PAD_ALIGN * PAD_ALIGN
    ).to(torch.int32)
    k_padded_lens = (
        (kv_seq_lens.to(torch.int64) + PAD_ALIGN - 1) // PAD_ALIGN * PAD_ALIGN
    ).to(torch.int32)
    q_down_lens = (q_padded_lens // stride).to(torch.int32)
    k_down_lens = (k_padded_lens // stride).to(torch.int32)

    max_q_down_len = q_down_lens.max().item()
    max_k_down_len = k_down_lens.max().item()
    max_q_down_aligned = ((max_q_down_len + R - 1) // R) * R
    max_k_down_aligned = ((max_k_down_len + R - 1) // R) * R
    max_Qb = max_q_down_aligned // R
    max_Kb = max_k_down_aligned // R

    # --- V processing: FP32 reference matching new path ---
    _, v_bias_per_kv = _prep_kv_reference(
        kcache_fp8, vcache_fp8, kscale_fp32, v_scale,
        kv_indices, kv_seq_lens, kv_block_size,
        lambda_mag=lambda_mag,
        stem_block_size=block_size, stem_stride=stride,
    )
    if num_q_heads != num_kv_heads:
        ratio = num_q_heads // num_kv_heads
        v_bias_expanded = v_bias_per_kv.repeat_interleave(ratio, dim=1)
    else:
        v_bias_expanded = v_bias_per_kv

    # --- QK scoring via Frobenius GEMM (per request, per-token K-scale folded)
    block_logits = torch.full(
        (num_req, num_q_heads, max_Qb, max_Kb),
        float("-inf"), dtype=torch.float32, device=device,
    )

    for req in range(num_req):
        q_start = cu_q_seqlens[req].item()
        q_len = q_seq_lens[req].item()
        kv_len = kv_seq_lens[req].item()
        q_padded = q_padded_lens[req].item()
        k_padded = k_padded_lens[req].item()
        Qb = q_padded // block_size
        Kb = k_padded // block_size

        # -- De-page Q: [q_padded, Hq, D] FP32 --
        q_dense = torch.zeros(
            q_padded, num_q_heads, head_dim,
            dtype=torch.float32, device=device,
        )
        q_dense[:q_len] = q_fp8[q_start:q_start + q_len].float()

        # -- De-page K: [k_padded, Hkv, D] FP32 --
        num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size
        block_ids = kv_indices[req, :num_kv_blocks]
        k_paged = kcache_fp8[block_ids]
        k_flat_raw = k_paged.reshape(-1, num_kv_heads, head_dim).float()
        k_dense = torch.zeros(
            k_padded, num_kv_heads, head_dim,
            dtype=torch.float32, device=device,
        )
        k_dense[:min(kv_len, k_flat_raw.shape[0])] = k_flat_raw[:kv_len]

        # Apply per-token K-scale to K (broadcast on dim) BEFORE group sum
        kscale_per_row = _gather_per_token_kscale_one_req(
            kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
        )  # [k_padded, Hkv]
        k_scaled = k_dense * kscale_per_row.unsqueeze(-1)

        # -- Build Q̃ [Hq, Qb, stride, D] (per-token per-head Q-scale folded in)
        q_blocks = q_dense[:Qb * block_size].reshape(Qb, R, stride, num_q_heads, head_dim)
        positions = torch.arange(
            Qb * block_size, dtype=torch.long, device=device,
        ).reshape(Qb, R, stride)
        scale_vals = q_scale[req, :, positions.clamp(max=q_scale.shape[2] - 1)]
        q_grouped = q_blocks.permute(3, 0, 1, 2, 4)  # [Hq, Qb, R, stride, D]
        q_tilde = (q_grouped * scale_vals.unsqueeze(-1)).sum(dim=2)
        # q_tilde: [Hq, Qb, stride, D]

        # -- Build K̃' [Hkv, Kb, stride, D] (reversed groups) --
        k_blocks = k_scaled[:Kb * block_size].reshape(Kb, R, stride, num_kv_heads, head_dim)
        k_grouped = k_blocks.permute(3, 0, 1, 2, 4)  # [Hkv, Kb, R, stride, D]
        k_tilde = k_grouped.sum(dim=2)  # [Hkv, Kb, stride, D]
        k_tilde_rev = k_tilde.flip(2)

        # -- Flatten + GEMM --
        q_flat = q_tilde.reshape(num_q_heads, Qb, stride * head_dim)
        k_flat = k_tilde_rev.reshape(num_kv_heads, Kb, stride * head_dim)
        k_flat_exp = k_flat.repeat_interleave(num_q_heads // num_kv_heads, dim=0)
        qk = (1.0 / (R * R)) * torch.bmm(q_flat, k_flat_exp.transpose(1, 2))
        # qk: [Hq, Qb, Kb]

        # -- V bias --
        vb_Kb = min(Kb, v_bias_expanded.shape[2])
        v_block = v_bias_expanded[req, :, :vb_Kb]  # [Hq, Kb]
        qk[:, :, :vb_Kb] += v_block.unsqueeze(1)

        block_logits[req, :, :Qb, :Kb] = qk

    # --- Causal mask ---
    if causal:
        qi_blocks_t = torch.clamp(
            (q_down_lens.to(torch.int64) + R - 1) // R, max=max_Qb,
        )
        ki_blocks_t = torch.clamp(
            (k_down_lens.to(torch.int64) + R - 1) // R, max=max_Kb,
        )
        block_offsets = ki_blocks_t - qi_blocks_t
        qb_range = torch.arange(max_Qb, device=device)
        kb_range = torch.arange(max_Kb, device=device)
        causal_mask = kb_range.unsqueeze(0).unsqueeze(0) > (
            qb_range.unsqueeze(0).unsqueeze(-1)
            + block_offsets.unsqueeze(-1).unsqueeze(-1)
        )
        invalid_q = qb_range.unsqueeze(0) >= qi_blocks_t.unsqueeze(-1)
        invalid_k = kb_range.unsqueeze(0) >= ki_blocks_t.unsqueeze(-1)
        invalid_mask = invalid_q.unsqueeze(-1) | invalid_k.unsqueeze(-2)
        full_mask = (causal_mask | invalid_mask).unsqueeze(1)
        block_logits.masked_fill_(full_mask, float("-inf"))

    return block_logits.to(dtype)


# ====================================================================
# verify_prep — KV prep + Q prep correctness + benchmark
# ====================================================================
def verify_prep(
    seq_lens=None,
    num_req=1,
    num_head_q=8,
    num_head_kv=1,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
    stride=16,
    stem_block_size=128,
    lambda_mag=0.3,
):
    """Compare CUDA stem_oam_prep_paged_kv (quant_type=0) + stem_oam_prep_varlen_q
    vs PyTorch reference."""
    if not _HPC_AVAILABLE:
        print("ERROR: hpc module not available. Build hpc-ops first.")
        sys.exit(1)

    if seq_lens is None:
        seq_lens = VERIFY_SEQ_LENS

    device = "cuda"
    print()
    print("=" * 80)
    print("  Prep Operators — Correctness (KV prep + Q prep, quant_type=0)")
    print(f"  num_req={num_req}, Hq={num_head_q}, Hkv={num_head_kv}, Dqk={dim_qk}, Dv={dim_v}")
    print(f"  kv_block_size={kv_block_size}, stride={stride}, "
          f"stem_bs={stem_block_size}, lambda={lambda_mag}")
    print(f"  Scale layout: K = per-KV-token (4D fp32), V = per-head fp32")
    print("=" * 80)

    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        # ---- KV prep ----
        ref_kflat, ref_vbias = _prep_kv_reference(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"], d["kv_block_size"],
            lambda_mag=lambda_mag,
            stem_block_size=stem_block_size, stem_stride=stride,
        )
        cuda_kflat, cuda_vbias = hpc.stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=lambda_mag,
            stem_block_size=stem_block_size, stem_stride=stride,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )

        # ---- Q prep ----
        ref_qflat = _prep_q_reference(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )
        cuda_qflat = hpc.stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )

        # ---- Compare valid blocks per request ----
        max_Kb = ref_kflat.shape[2]
        max_Qb = ref_qflat.shape[2]

        sk = allclose_check(
            torch.cat([
                ref_kflat[b, :, :((d["kv_seqlens"][b].item() + stem_block_size - 1) // stem_block_size)]
                for b in range(num_req)
            ]),
            torch.cat([
                cuda_kflat[b, :, :((d["kv_seqlens"][b].item() + stem_block_size - 1) // stem_block_size)]
                for b in range(num_req)
            ]),
            **TOL_PREP,
        )
        sv = allclose_check(
            torch.cat([
                ref_vbias[b, :, :((d["kv_seqlens"][b].item() + stem_block_size - 1) // stem_block_size)]
                for b in range(num_req)
            ]),
            torch.cat([
                cuda_vbias[b, :, :((d["kv_seqlens"][b].item() + stem_block_size - 1) // stem_block_size)]
                for b in range(num_req)
            ]),
            **TOL_PREP,
        )
        sq = allclose_check(
            torch.cat([
                ref_qflat[b, :, :((q_seqlens[b].item() + stem_block_size - 1) // stem_block_size)]
                for b in range(num_req)
            ]),
            torch.cat([
                cuda_qflat[b, :, :((q_seqlens[b].item() + stem_block_size - 1) // stem_block_size)]
                for b in range(num_req)
            ]),
            **TOL_PREP,
        )

        print(f"\n  seq_len={seq_len}  (Qb={max_Qb}, Kb={max_Kb})")
        _print_check(sk, "Kflat ")
        _print_check(sv, "V_bias")
        _print_check(sq, "Qflat ")

    # ---- Second round: random qscale (non-trivial weights) ----
    print(f"\n  {'─' * 76}")
    print("  Re-running Q prep with random qscale (non-uniform weights)...")

    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        max_seqlen_pad128 = ((seq_len + 127) // 128) * 128
        d["qscale"] = 0.5 + torch.rand(
            num_req, num_head_q, max_seqlen_pad128,
            dtype=torch.float32, device=device,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        ref_qflat = _prep_q_reference(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )
        cuda_qflat = hpc.stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )

        q_padded = (
            (q_seqlens.to(torch.int64) + stem_block_size - 1)
            // stem_block_size * stem_block_size
        ).to(torch.int32)
        ref_cat = torch.cat([
            ref_qflat[b, :, :q_padded[b].item() // stem_block_size] for b in range(num_req)
        ])
        cuda_cat = torch.cat([
            cuda_qflat[b, :, :q_padded[b].item() // stem_block_size] for b in range(num_req)
        ])
        s = allclose_check(ref_cat, cuda_cat, **TOL_PREP)
        print(f"  seq_len={seq_len:>6} (rand qscale): "
              f"max_abs={s['max_abs']:.4e}  mean_abs={s['mean_abs']:.4e}  "
              f"[{'PASS' if s['all_pass'] else 'FAIL'}]")

    # ---- Benchmark table: Q / KV / Total, CUDA new vs CUDA legacy ----
    bench_lens = BENCH_SEQ_LENS
    print(f"\n  {'─' * 90}")
    print("  Performance: CUDA Q + KV (quant_type=0 new)  vs  legacy (quant_type=1)")
    print(f"  {'─' * 90}")
    print(f"  {'SeqLen':>8}  {'Blocks':>6}  "
          f"{'Q':>8}  {'KV_new':>8}  {'Total_new':>9}  "
          f"{'KV_legacy':>10}  {'Total_legacy':>12}  {'KV_overhead':>11}")
    print(f"  {'-' * 90}")

    for seq_len in bench_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)
        Kb = ((seq_len + stem_block_size - 1) // stem_block_size)

        # Legacy-path scalar K/V scales for the apples-to-apples KV bench.
        kscale_legacy = torch.ones(1, dtype=torch.float32, device=device)
        vscale_legacy = torch.ones(1, dtype=torch.float32, device=device)

        def _cuda_q(dd=d, qs=q_seqlens):
            return hpc.stem_oam_prep_varlen_q(
                dd["q_fp8"], dd["qscale"], qs, dd["cu_q_seqlens"],
                stem_block_size, stride,
            )

        def _cuda_kv_new(dd=d):
            return hpc.stem_oam_prep_paged_kv(
                dd["kcache_fp8"], dd["vcache_fp8"], dd["kscale_fp32"], dd["vscale"],
                dd["kv_indices"], dd["kv_seqlens"],
                lambda_mag, stem_block_size, stride,
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            )

        def _cuda_kv_legacy(dd=d, ks=kscale_legacy, vs=vscale_legacy):
            return hpc.stem_oam_prep_paged_kv(
                dd["kcache_fp8"], dd["vcache_fp8"], ks, vs,
                dd["kv_indices"], dd["kv_seqlens"],
                lambda_mag, stem_block_size, stride,
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            )

        def _cuda_both_new(dd=d, qs=q_seqlens):
            hpc.stem_oam_prep_paged_kv(
                dd["kcache_fp8"], dd["vcache_fp8"], dd["kscale_fp32"], dd["vscale"],
                dd["kv_indices"], dd["kv_seqlens"],
                lambda_mag, stem_block_size, stride,
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            )
            hpc.stem_oam_prep_varlen_q(
                dd["q_fp8"], dd["qscale"], qs, dd["cu_q_seqlens"],
                stem_block_size, stride,
            )

        def _cuda_both_legacy(dd=d, qs=q_seqlens, ks=kscale_legacy, vs=vscale_legacy):
            hpc.stem_oam_prep_paged_kv(
                dd["kcache_fp8"], dd["vcache_fp8"], ks, vs,
                dd["kv_indices"], dd["kv_seqlens"],
                lambda_mag, stem_block_size, stride,
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            )
            hpc.stem_oam_prep_varlen_q(
                dd["q_fp8"], dd["qscale"], qs, dd["cu_q_seqlens"],
                stem_block_size, stride,
            )

        q_ms = benchmark_fn(_cuda_q, warmup=10, iters=50)
        kv_new_ms = benchmark_fn(_cuda_kv_new, warmup=10, iters=50)
        kv_leg_ms = benchmark_fn(_cuda_kv_legacy, warmup=10, iters=50)
        total_new_ms = benchmark_fn(_cuda_both_new, warmup=10, iters=50)
        total_leg_ms = benchmark_fn(_cuda_both_legacy, warmup=10, iters=50)
        kv_overhead = (kv_new_ms / kv_leg_ms - 1.0) * 100 if kv_leg_ms > 0 else float("nan")

        print(f"  {seq_len:>8}  {Kb:>6}  "
              f"{q_ms:>7.3f}  {kv_new_ms:>7.3f}  {total_new_ms:>8.3f}  "
              f"{kv_leg_ms:>9.3f}  {total_leg_ms:>11.3f}  "
              f"{kv_overhead:>+9.1f}%")

    print(f"  {'─' * 90}")
    print("  Q          = stem_oam_prep_varlen_q (Q prep, identical for both quant_types)")
    print("  KV_new     = stem_oam_prep_paged_kv (quant_type=0, per-token K + per-head V)")
    print("  KV_legacy  = stem_oam_prep_paged_kv (quant_type=1, per-tensor K + V)")
    print("  KV_overhead = (KV_new / KV_legacy - 1) × 100")
    print(f"\n{'=' * 80}\n")


# ====================================================================
# verify_gemm — GEMM kernel accuracy + speed (causal=False)
# ====================================================================
def verify_gemm(
    seq_lens=None,
    num_req=1,
    num_head_q=8,
    num_head_kv=1,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
    stride=16,
    stem_block_size=128,
    lambda_mag=0.3,
):
    """Compare stem_oam_gemm (CUDA) vs PyTorch matmul + FrobScale + V_bias.

    Inputs (qflat, kflat, v_bias) come from the CUDA prep kernels under
    quant_type=0, so the GEMM is tested on realistic per-token-K-scaled data.
    """
    if not _HPC_AVAILABLE:
        print("ERROR: hpc module not available. Build hpc-ops first.")
        sys.exit(1)

    if seq_lens is None:
        seq_lens = VERIFY_SEQ_LENS

    R = stem_block_size // stride
    frob_scale = 1.0 / (R * R)
    ratio = num_head_q // num_head_kv

    print()
    print("=" * 80)
    print("  stem_oam_gemm — Correctness (causal=False, quant_type=0 inputs)")
    print(f"  num_req={num_req}, Hq={num_head_q}, Hkv={num_head_kv}, Dqk={dim_qk}")
    print(f"  stem_bs={stem_block_size}, stride={stride}, FrobScale=1/{R * R}")
    print("=" * 80)

    # ---- Accuracy ----
    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        kflat, v_bias = hpc.stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=lambda_mag,
            stem_block_size=stem_block_size, stem_stride=stride,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )
        qflat = hpc.stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )

        cuda_out = hpc.stem_oam_gemm(
            qflat, kflat, v_bias, q_seqlens, d["kv_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
            causal=False,
        )

        max_Qb, max_Kb = qflat.shape[2], kflat.shape[2]
        kflat_exp = kflat.repeat_interleave(ratio, dim=1)
        v_bias_exp = v_bias.repeat_interleave(ratio, dim=1)

        raw = torch.bmm(
            qflat.reshape(-1, max_Qb, qflat.shape[3]).float(),
            kflat_exp.reshape(-1, max_Kb, kflat_exp.shape[3]).float().transpose(-1, -2),
        ).reshape(num_req, num_head_q, max_Qb, max_Kb)

        ref_out = frob_scale * raw + v_bias_exp.unsqueeze(2)

        for b in range(num_req):
            q_padded = (
                (q_seqlens[b].item() + stem_block_size - 1)
                // stem_block_size * stem_block_size
            )
            kv_padded = (
                (d["kv_seqlens"][b].item() + stem_block_size - 1)
                // stem_block_size * stem_block_size
            )
            ref_out[b, :, q_padded // stem_block_size:, :] = float("-inf")
            ref_out[b, :, :, kv_padded // stem_block_size:] = float("-inf")

        ref_out = ref_out.to(torch.bfloat16)

        s = allclose_check(ref_out, cuda_out, **TOL_GEMM)
        print(
            f"\n  seq_len={seq_len}  (Qb={max_Qb}, Kb={max_Kb}, "
            f"finite={s['n_finite']}/{s['n_total']})"
        )
        _print_check(s)

    # ---- Benchmark ----
    bench_lens = BENCH_SEQ_LENS
    print(f"\n  {'─' * 76}")
    print("  Performance: CUDA stem_oam_gemm vs PyTorch matmul (quant_type=0 inputs)")
    print(f"  {'─' * 76}")
    print(f"  {'SeqLen':>8}  {'Qb':>5}  {'Kb':>5}  "
          f"{'CUDA':>8}  {'PyTorch':>9}  {'Speedup':>8}")
    print(f"  {'-' * 58}")

    for seq_len in bench_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        kflat, v_bias = hpc.stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=lambda_mag,
            stem_block_size=stem_block_size, stem_stride=stride,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )
        qflat = hpc.stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )

        max_Qb, max_Kb = qflat.shape[2], kflat.shape[2]
        kflat_exp = kflat.repeat_interleave(ratio, dim=1)
        v_bias_exp = v_bias.repeat_interleave(ratio, dim=1)
        qf_bf16 = qflat.reshape(-1, max_Qb, qflat.shape[3])
        kf_bf16 = kflat_exp.reshape(-1, max_Kb, kflat_exp.shape[3]).transpose(-1, -2)

        def _cuda(q=qflat, k=kflat, vb=v_bias, qs=q_seqlens, kvs=d["kv_seqlens"]):
            return hpc.stem_oam_gemm(
                q, k, vb, qs, kvs, stem_block_size, stride, False,
            )

        def _pytorch(q=qf_bf16, k=kf_bf16, vbe=v_bias_exp, nr=num_req, nh=num_head_q,
                     mq=max_Qb, mk=max_Kb, fs=frob_scale):
            raw = torch.bmm(q, k).reshape(nr, nh, mq, mk).float()
            return (fs * raw + vbe.unsqueeze(2)).to(torch.bfloat16)

        cuda_ms = benchmark_fn(_cuda, warmup=10, iters=50)
        ref_ms = benchmark_fn(_pytorch, warmup=10, iters=50)
        speedup = ref_ms / cuda_ms if cuda_ms > 0 else float("inf")

        print(f"  {seq_len:>8}  {max_Qb:>5}  {max_Kb:>5}  "
              f"{cuda_ms:>7.3f}   {ref_ms:>8.3f}   {speedup:>7.1f}x")

    print(f"  {'─' * 58}")
    print("  CUDA    = stem_oam_gemm kernel (ms)")
    print("  PyTorch = torch.bmm (BF16) + FrobScale + V_bias (ms)")
    print(f"\n{'=' * 80}\n")


# ====================================================================
# verify_oam — Full OAM pipeline accuracy + speed (causal=True, no Triton)
# ====================================================================
def verify_oam(
    seq_lens=None,
    num_req=1,
    num_head_q=8,
    num_head_kv=1,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
    stride=16,
    stem_block_size=128,
    lambda_mag=0.3,
):
    """Compare full CUDA OAM pipeline (quant_type=0) vs FP32 PyTorch reference.

    Triton baseline is intentionally omitted: vllm Triton path uses per-tensor
    K/V scale only and cannot consume our 4D K-scale layout; speed comparison
    here is CUDA-only across seq lens.
    """
    if not _HPC_AVAILABLE:
        print("ERROR: hpc module not available. Build hpc-ops first.")
        sys.exit(1)

    if seq_lens is None:
        seq_lens = VERIFY_SEQ_LENS

    print()
    print("=" * 80)
    print("  Full OAM Pipeline — Accuracy vs FP32 PyTorch Reference (quant_type=0)")
    print(f"  num_req={num_req}, Hq={num_head_q}, Hkv={num_head_kv}, Dqk={dim_qk}")
    print(f"  stem_bs={stem_block_size}, stride={stride}, lambda_mag={lambda_mag}")
    print("=" * 80)

    dtype = torch.bfloat16

    # ---- Accuracy ----
    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        ref_out = _oam_math_reference(
            d["q_fp8"], d["qscale"],
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], q_seqlens,
            stride=stride, block_size=stem_block_size,
            lambda_mag=lambda_mag, causal=True,
            dtype=dtype, device=d["q_fp8"].device,
            kv_block_size=d["kv_block_size"],
        )

        kflat, v_bias = hpc.stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=lambda_mag,
            stem_block_size=stem_block_size, stem_stride=stride,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )
        qflat = hpc.stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )
        cuda_out = hpc.stem_oam_gemm(
            qflat, kflat, v_bias, q_seqlens, d["kv_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
            causal=True,
        )

        r_Qb, r_Kb = ref_out.shape[2], ref_out.shape[3]
        c_Qb, c_Kb = cuda_out.shape[2], cuda_out.shape[3]
        min_Qb, min_Kb = min(r_Qb, c_Qb), min(r_Kb, c_Kb)
        ref_cmp = ref_out[:, :, :min_Qb, :min_Kb]
        cuda_cmp = cuda_out[:, :, :min_Qb, :min_Kb]

        s = allclose_check(ref_cmp, cuda_cmp, **TOL_OAM)
        print(
            f"\n  seq_len={seq_len}  (Ref: {r_Qb}\u00d7{r_Kb}, "
            f"CUDA: {c_Qb}\u00d7{c_Kb}, finite={s['n_finite']})"
        )
        _print_check(s)

    # ---- Benchmark (CUDA only across seq lens) ----
    bench_lens = BENCH_SEQ_LENS
    print(f"\n  {'─' * 76}")
    print("  Performance: CUDA OAM pipeline (quant_type=0, causal=True)")
    print(f"  {'─' * 76}")
    print(f"  {'SeqLen':>8}  {'Blocks':>6}  "
          f"{'Prep_KV':>8}  {'Prep_Q':>8}  {'GEMM':>8}  {'Total':>8}")
    print(f"  {'-' * 58}")

    for seq_len in bench_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)
        Kb = (seq_len + stem_block_size - 1) // stem_block_size

        kflat_pre, vb_pre = hpc.stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=lambda_mag,
            stem_block_size=stem_block_size, stem_stride=stride,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )
        qflat_pre = hpc.stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=stem_block_size, stem_stride=stride,
        )

        def _kv(dd=d):
            return hpc.stem_oam_prep_paged_kv(
                dd["kcache_fp8"], dd["vcache_fp8"], dd["kscale_fp32"], dd["vscale"],
                dd["kv_indices"], dd["kv_seqlens"],
                lambda_mag, stem_block_size, stride,
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            )

        def _q(dd=d, qs=q_seqlens):
            return hpc.stem_oam_prep_varlen_q(
                dd["q_fp8"], dd["qscale"], qs, dd["cu_q_seqlens"],
                stem_block_size, stride,
            )

        def _gemm(qf=qflat_pre, kf=kflat_pre, vb=vb_pre, qs=q_seqlens,
                  kvs=d["kv_seqlens"]):
            return hpc.stem_oam_gemm(
                qf, kf, vb, qs, kvs, stem_block_size, stride, True,
            )

        def _cuda_all(dd=d, qs=q_seqlens):
            kf, vb = hpc.stem_oam_prep_paged_kv(
                dd["kcache_fp8"], dd["vcache_fp8"], dd["kscale_fp32"], dd["vscale"],
                dd["kv_indices"], dd["kv_seqlens"],
                lambda_mag, stem_block_size, stride,
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
            )
            qf = hpc.stem_oam_prep_varlen_q(
                dd["q_fp8"], dd["qscale"], qs, dd["cu_q_seqlens"],
                stem_block_size, stride,
            )
            return hpc.stem_oam_gemm(
                qf, kf, vb, qs, dd["kv_seqlens"],
                stem_block_size, stride, True,
            )

        kv_ms = benchmark_fn(_kv, warmup=10, iters=50)
        q_ms = benchmark_fn(_q, warmup=10, iters=50)
        gemm_ms = benchmark_fn(_gemm, warmup=10, iters=50)
        total_ms = benchmark_fn(_cuda_all, warmup=10, iters=50)

        print(f"  {seq_len:>8}  {Kb:>6}  "
              f"{kv_ms:>7.3f}   {q_ms:>7.3f}   {gemm_ms:>7.3f}   {total_ms:>7.3f}")

    print(f"  {'─' * 58}")
    print("  Prep_KV / Prep_Q / GEMM = individual kernel time (ms)")
    print("  Total   = all 3 kernels launched sequentially (ms)")
    print(f"\n{'=' * 80}\n")


# ====================================================================
# TPD verification helpers (identical to test_stem.py — TPD is scale-agnostic)
# ====================================================================
def _setup_tpd_data(num_req, seq_len, num_head_q, block_size=128):
    device = "cuda"
    max_Qb = (seq_len + block_size - 1) // block_size
    max_Kb = max_Qb

    block_logits = torch.randn(
        num_req, num_head_q, max_Qb, max_Kb,
        dtype=torch.bfloat16, device=device,
    )

    q_idx = torch.arange(max_Qb, device=device).view(-1, 1)
    k_idx = torch.arange(max_Kb, device=device).view(1, -1)
    block_logits.masked_fill_(k_idx > q_idx, float("-inf"))

    q_seq_lens = torch.full((num_req,), seq_len, dtype=torch.int32, device=device)
    kv_seq_lens = torch.full((num_req,), seq_len, dtype=torch.int32, device=device)
    return block_logits, q_seq_lens, kv_seq_lens


def _tpd_mask_compare(ref_mask, cuda_mask, qi_blocks, ki_blocks):
    ref = ref_mask[:, :, :qi_blocks, :ki_blocks].bool()
    cuda = cuda_mask[:, :, :qi_blocks, :ki_blocks].bool()

    total = ref.numel()
    match = (ref == cuda).sum().item()
    fp = (cuda & ~ref).sum().item()
    fn = (~cuda & ref).sum().item()

    return {
        "total": total,
        "match": match,
        "fp": fp,
        "fn": fn,
        "match_pct": match / total * 100 if total > 0 else 100.0,
        "ref_density": ref.float().mean().item() * 100,
        "cuda_density": cuda.float().mean().item() * 100,
    }


# ====================================================================
# verify_tpd
# ====================================================================
def verify_tpd(
    seq_lens=None,
    num_req=1,
    num_head_q=8,
    stem_block_size=128,
    alpha=1.0,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
    **_kwargs,
):
    """Compare CUDA stem_tpd vs Python _generate_sparse_mask_per_req.

    Identical to test_stem.py's verify_tpd — TPD is scale-agnostic.
    """
    if not _HPC_AVAILABLE:
        print("ERROR: hpc module not available. Build hpc-ops first.")
        sys.exit(1)
    if _generate_sparse_mask_per_req is None:
        print("ERROR: vllm.v1.attention.backends.stem._generate_sparse_mask_per_req "
              "not available — verify-tpd needs it as the Python reference.")
        sys.exit(1)

    if seq_lens is None:
        seq_lens = VERIFY_SEQ_LENS

    device = "cuda"

    print()
    print("=" * 80)
    print("  TPD Mask Generation — Accuracy vs PyTorch Ref, Speed vs Python")
    print(f"  num_req={num_req}, Hq={num_head_q}, block_size={stem_block_size}")
    print(f"  alpha={alpha}, initial_blocks={initial_blocks}, window_size={window_size}")
    print("=" * 80)

    print(f"\n  {'SeqLen':>8}  {'Qb':>5} {'Kb':>5}  "
          f"{'Match%':>7}  {'FP':>8}  {'FN':>8}  "
          f"{'Ref%':>6}  {'CUDA%':>6}  {'Status':>6}")
    print(f"  {'-' * 72}")

    all_pass = True
    for seq_len in seq_lens:
        torch.cuda.empty_cache()

        block_logits, q_seq_lens_t, kv_seq_lens_t = _setup_tpd_data(
            num_req, seq_len, num_head_q, stem_block_size,
        )
        max_Qb = block_logits.shape[2]

        PAD_ALIGN = 128
        qi_padded = ((seq_len + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        qi_blocks = min(
            (qi_padded + stem_block_size - 1) // stem_block_size, max_Qb,
        )
        ki_blocks = qi_blocks

        # vllm reference only exposes the `large` regime knobs (medium hard-coded
        # to 0.2/30 upstream); pass-through `_large` here keeps the comparison fair.
        ref_mask = _generate_sparse_mask_per_req(
            block_logits, q_seq_lens_t, kv_seq_lens_t,
            block_size=stem_block_size, alpha=alpha,
            k_block_num_rate=k_block_num_rate_large,
            k_block_num_bias=k_block_num_bias_large,
            device=device,
        )

        cuda_mask = hpc.stem_tpd(
            block_logits, q_seq_lens_t, kv_seq_lens_t,
            kv_seq_lens_t,  # num_prompt_tokens (normal prefill: equals kv_seq_lens)
            block_size=stem_block_size, alpha=alpha,
            initial_blocks=initial_blocks, window_size=window_size,
            k_block_num_rate_medium=k_block_num_rate_medium,
            k_block_num_bias_medium=k_block_num_bias_medium,
            k_block_num_rate_large=k_block_num_rate_large,
            k_block_num_bias_large=k_block_num_bias_large,
        )

        s = _tpd_mask_compare(ref_mask, cuda_mask, qi_blocks, ki_blocks)
        ok = s["fn"] == 0
        all_pass &= ok
        status = "PASS" if ok else "FAIL"

        print(f"  {seq_len:>8}  {qi_blocks:>5} {ki_blocks:>5}  "
              f"{s['match_pct']:>6.2f}%  {s['fp']:>8}  {s['fn']:>8}  "
              f"{s['ref_density']:>5.1f}%  {s['cuda_density']:>5.1f}%  "
              f"{'  ' + status:>6}")

    print(f"  {'-' * 72}")
    print("  FP = false positives (CUDA selects extra blocks, harmless)")
    print("  FN = false negatives (CUDA misses ref blocks, would be a bug)")
    print("  Ref% / CUDA% = mask density in valid causal region")
    if all_pass:
        print("  ** ALL PASSED (FN == 0 for every test case) **")
    else:
        print("  ** SOME TESTS FAILED — check FN column **")

    bench_lens = BENCH_SEQ_LENS
    print(f"\n  {'─' * 60}")
    print("  Performance: CUDA stem_tpd vs Python _generate_sparse_mask_per_req")
    print(f"  {'─' * 60}")
    print(f"  {'SeqLen':>8}  {'Blocks':>6}  "
          f"{'CUDA':>8}  {'Python':>8}  {'Speedup':>8}")
    print(f"  {'-' * 48}")

    for seq_len in bench_lens:
        torch.cuda.empty_cache()
        block_logits, q_seq_lens_t, kv_seq_lens_t = _setup_tpd_data(
            num_req, seq_len, num_head_q, stem_block_size,
        )
        Kb = block_logits.shape[3]

        def _cuda(bl=block_logits, qs=q_seq_lens_t, kvs=kv_seq_lens_t):
            return hpc.stem_tpd(
                bl, qs, kvs,
                kvs,  # num_prompt_tokens (normal prefill: equals kv_seq_lens)
                block_size=stem_block_size, alpha=alpha,
                initial_blocks=initial_blocks, window_size=window_size,
                k_block_num_rate_medium=k_block_num_rate_medium,
                k_block_num_bias_medium=k_block_num_bias_medium,
                k_block_num_rate_large=k_block_num_rate_large,
                k_block_num_bias_large=k_block_num_bias_large,
            )

        def _python(bl=block_logits, qs=q_seq_lens_t, kvs=kv_seq_lens_t):
            return _generate_sparse_mask_per_req(
                bl, qs, kvs,
                block_size=stem_block_size, alpha=alpha,
                k_block_num_rate=k_block_num_rate_large,
                k_block_num_bias=k_block_num_bias_large,
                device=device,
            )

        cuda_ms = benchmark_fn(_cuda, warmup=10, iters=50)
        python_ms = benchmark_fn(_python, warmup=5, iters=20)
        speedup = python_ms / cuda_ms if cuda_ms > 0 else float("inf")

        print(f"  {seq_len:>8}  {Kb:>6}  "
              f"{cuda_ms:>7.3f}   {python_ms:>7.3f}   {speedup:>7.1f}x")

    print(f"  {'─' * 60}")
    print("  CUDA   = hpc.stem_tpd (ms)")
    print("  Python = _generate_sparse_mask_per_req from stem.py (ms)")
    print(f"\n{'=' * 80}\n")


# ====================================================================
# verify_stem_all — End-to-end OAM + TPD pipeline (quant_type=0)
# ====================================================================
def verify_stem_all(
    seq_lens=None,
    num_req=1,
    num_head_q=8,
    num_head_kv=1,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
    stride=16,
    stem_block_size=128,
    lambda_mag=0.3,
    alpha=1.0,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
    **_kwargs,
):
    """End-to-end Stem pipeline verification (quant_type=0).

    Part 1: Mask accuracy — CUDA pipeline (quant_type=0) vs FP32 PyTorch reference.
    Part 2: End-to-end speed — CUDA pipeline only across seq lens
            (Triton baseline omitted: vllm Triton path doesn't support quant_type=0)
    """
    if not _HPC_AVAILABLE:
        print("ERROR: hpc module not available. Build hpc-ops first.")
        sys.exit(1)
    if _generate_sparse_mask_per_req is None:
        print("ERROR: vllm.v1.attention.backends.stem._generate_sparse_mask_per_req "
              "not available — verify-stemAll needs it as the mask reference.")
        sys.exit(1)

    if seq_lens is None:
        seq_lens = VERIFY_SEQ_LENS

    device = "cuda"

    print()
    print("=" * 88)
    print("  Stem End-to-End — OAM + TPD Pipeline (quant_type=0)")
    print(f"  num_req={num_req}, Hq={num_head_q}, Hkv={num_head_kv}, Dqk={dim_qk}")
    print(f"  stem_bs={stem_block_size}, stride={stride}, lambda_mag={lambda_mag}")
    print(f"  alpha={alpha}, initial_blocks={initial_blocks}, window_size={window_size}")
    print("=" * 88)

    def _cuda_pipeline(d):
        return hpc.stem_paged_kv(
            d["q_fp8"],
            d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"],
            d["kv_seqlens"],  # num_prompt_tokens (normal prefill: equals kv_seqlens)
            lambda_mag=lambda_mag, alpha=alpha,
            stem_block_size=stem_block_size, stem_stride=stride,
            causal=True,
            initial_blocks=initial_blocks, window_size=window_size,
            k_block_num_rate_medium=k_block_num_rate_medium,
            k_block_num_bias_medium=k_block_num_bias_medium,
            k_block_num_rate_large=k_block_num_rate_large,
            k_block_num_bias_large=k_block_num_bias_large,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )

    def _valid_blocks(seq_len, *shapes):
        PAD_ALIGN = 128
        qi_padded = ((seq_len + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        qi = (qi_padded + stem_block_size - 1) // stem_block_size
        for Qb, Kb in shapes:
            qi = min(qi, Qb, Kb)
        return qi, qi

    # ── Part 1: Mask Accuracy vs FP32 Ground Truth ────────────────────
    print("\n  Part 1: Mask Accuracy vs FP32 Ground Truth")
    print("  (Ground truth = FP32 OAM logits → Python top-k mask)")
    print(f"  {'─' * 76}")
    print(f"  {'SeqLen':>8}  {'Qb':>5} {'Kb':>5}  "
          f"{'Match%':>7} {'FP':>8} {'FN':>8} {'Status':>6}")
    print(f"  {'-' * 64}")

    # New-path floor: per-token K-scale multiplies every row before group-sum
    # (vs legacy path's single scalar multiply at the end).  Every row's FMA
    # produces ~ulp-level drift between CUDA (fp32 accum) and PyTorch reference
    # (also fp32 but with different op ordering), which in turn lets a handful
    # of near-tie block_logits cross the top-k boundary.  Effect caps out
    # around 0.05-0.10% as Kb grows.  Legacy path does not suffer this because
    # a scalar multiply commutes with sum and doesn't change top-k ordering.
    PASS_THRESHOLD = 99.8
    all_pass = True

    for seq_len in seq_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        ref_logits = _oam_math_reference(
            d["q_fp8"], d["qscale"],
            d["kcache_fp8"], d["vcache_fp8"], d["kscale_fp32"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], q_seqlens,
            stride=stride, block_size=stem_block_size,
            lambda_mag=lambda_mag, causal=True,
            dtype=torch.float32, device=d["q_fp8"].device,
            kv_block_size=d["kv_block_size"],
        )
        ref_mask = _generate_sparse_mask_per_req(
            ref_logits, q_seqlens, d["kv_seqlens"],
            block_size=stem_block_size, alpha=alpha,
            k_block_num_rate=k_block_num_rate_large,
            k_block_num_bias=k_block_num_bias_large,
            device=device,
        )

        cuda_mask = _cuda_pipeline(d)

        qi, ki = _valid_blocks(
            seq_len,
            (ref_mask.shape[2], ref_mask.shape[3]),
            (cuda_mask.shape[2], cuda_mask.shape[3]),
        )

        s = _tpd_mask_compare(ref_mask, cuda_mask, qi, ki)
        ok = s["match_pct"] >= PASS_THRESHOLD
        all_pass &= ok
        status = "PASS" if ok else "FAIL"

        print(f"  {seq_len:>8}  {qi:>5} {ki:>5}  "
              f"{s['match_pct']:>6.2f}% {s['fp']:>7} {s['fn']:>7} "
              f"{'  ' + status:>6}")

    print(f"  {'-' * 64}")
    print("  FP = extra blocks vs FP32 ground truth (harmless)")
    print("  FN = missed blocks from FP32 reference")
    print(f"  PASS threshold: match% >= {PASS_THRESHOLD}%")
    if all_pass:
        print("  ** CUDA pipeline ALL PASSED **")
    else:
        print("  ** CUDA pipeline SOME FAILED — check Match% column **")

    # ── Part 2: End-to-End Speed (CUDA only across seq lens) ─────────
    bench_lens = BENCH_SEQ_LENS
    print("\n  Part 2: End-to-End Speed — CUDA Pipeline (quant_type=0)")
    print(f"  {'─' * 50}")
    print(f"  {'SeqLen':>8}  {'Blocks':>6}  {'CUDA':>9}")
    print(f"  {'-' * 32}")

    for seq_len in bench_lens:
        torch.cuda.empty_cache()
        d = _setup_paged_fp8_data(
            num_req, seq_len, num_head_q, num_head_kv,
            dim_qk, dim_v, kv_block_size,
        )
        Kb = (seq_len + stem_block_size - 1) // stem_block_size

        def _cuda_fn(dd=d):
            return _cuda_pipeline(dd)

        cuda_ms = benchmark_fn(_cuda_fn, warmup=10, iters=50)
        print(f"  {seq_len:>8}  {Kb:>6}  {cuda_ms:>8.3f}")

    print(f"  {'─' * 32}")
    print("  CUDA = prep_kv + prep_q + gemm + tpd end-to-end (ms)")
    print(f"\n{'=' * 88}\n")


# ====================================================================
# verify_chunked — chunked-prefill correctness (D semantic, quant_type=0)
# ====================================================================


def _three_stage_k_start(n,
                         k_block_num_rate_medium=0.2, k_block_num_bias_medium=30,
                         k_block_num_rate_large=0.1, k_block_num_bias_large=30):
    """Three-stage k_start selector matching stem_kernels.cuh::compute_budget."""
    if n < 56:
        return n
    if n < 160:
        return int(n * k_block_num_rate_medium) + k_block_num_bias_medium
    return int(n * k_block_num_rate_large) + k_block_num_bias_large


def _generate_sparse_mask_D_semantic(
    block_logits,        # [num_req, num_heads, max_Qb, max_Kb], float
    q_seq_lens,          # [num_req], int — current chunk's Q tokens
    kv_seq_lens,         # [num_req], int — cumulative KV tokens (current chunk)
    num_prompt_tokens,   # [num_req], int — full prompt KV tokens (chunk-invariant)
    block_size,
    alpha,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
    device="cuda",
):
    """D-semantic Python ref: k_start + decay keyed on prompt_kv_blocks."""
    num_req, num_heads, max_Qb, max_Kb = block_logits.shape
    out = torch.zeros(num_req, num_heads, max_Qb, max_Kb,
                      dtype=torch.uint8, device=device)
    PAD_ALIGN = 128
    for i in range(num_req):
        qi_padded = ((int(q_seq_lens[i].item()) + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        ki_padded = ((int(kv_seq_lens[i].item()) + PAD_ALIGN - 1) // PAD_ALIGN) * PAD_ALIGN
        qi_blocks = min((qi_padded + block_size - 1) // block_size, max_Qb)
        ki_blocks = min((ki_padded + block_size - 1) // block_size, max_Kb)
        if qi_blocks == 0 or ki_blocks == 0:
            continue

        prompt_kv_blocks = (int(num_prompt_tokens[i].item()) + block_size - 1) // block_size
        kb_offset = ki_blocks - qi_blocks
        k_start = _three_stage_k_start(
            prompt_kv_blocks,
            k_block_num_rate_medium, k_block_num_bias_medium,
            k_block_num_rate_large, k_block_num_bias_large,
        )

        budget = torch.zeros(qi_blocks, dtype=torch.long, device=device)
        decay_len = prompt_kv_blocks - k_start
        for q_row in range(qi_blocks):
            q_pos = q_row + kb_offset
            if q_pos < k_start or decay_len <= 1:
                b = k_start
            else:
                t = (q_pos - k_start) / (decay_len - 1)
                k_end = k_start * alpha
                b_f = k_start + t * (k_end - k_start)
                b = int(math.floor(b_f))
                b = max(1, min(k_start, b))
            budget[q_row] = b

        req_logits = block_logits[i, :, :qi_blocks, :ki_blocks]
        req_mask = torch.zeros(num_heads, qi_blocks, ki_blocks,
                               dtype=torch.bool, device=device)
        for q_row in range(qi_blocks):
            b = int(budget[q_row].item())
            for h in range(num_heads):
                row = req_logits[h, q_row]
                finite = torch.isfinite(row)
                n_finite = int(finite.sum().item())
                if n_finite == 0:
                    pass
                elif b >= n_finite:
                    req_mask[h, q_row] = finite
                else:
                    finite_vals = row[finite]
                    topk_vals, _ = torch.topk(finite_vals, b)
                    threshold = topk_vals[-1].item()
                    req_mask[h, q_row] = (row >= threshold) & finite
                upper = min(initial_blocks, ki_blocks)
                req_mask[h, q_row, :upper] = True
                diag_col = q_row + kb_offset
                if 0 <= diag_col < ki_blocks:
                    win_lo = max(0, diag_col - window_size + 1)
                    req_mask[h, q_row, win_lo:diag_col + 1] = True

        out[i, :, :qi_blocks, :ki_blocks] = req_mask.to(torch.uint8)

    return out


def _compute_expected_budget(q_pos, prompt_kv_blocks, alpha,
                             k_block_num_rate_medium=0.2, k_block_num_bias_medium=30,
                             k_block_num_rate_large=0.1, k_block_num_bias_large=30):
    """Mirror compute_budget(): D-semantic per-row budget."""
    k_val = _three_stage_k_start(
        prompt_kv_blocks,
        k_block_num_rate_medium, k_block_num_bias_medium,
        k_block_num_rate_large, k_block_num_bias_large,
    )
    decay_len = prompt_kv_blocks - k_val
    if q_pos < k_val or decay_len <= 1:
        return k_val
    t = (q_pos - k_val) / (decay_len - 1)
    k_end = k_val * alpha
    return max(1, min(k_val, int(math.floor(k_val + t * (k_end - k_val)))))


def verify_chunked(
    num_req=1,
    num_head_q=8,
    num_head_kv=1,
    dim_qk=128,
    dim_v=128,
    kv_block_size=64,
    stride=16,
    stem_block_size=128,
    lambda_mag=0.3,
    alpha=0.8,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
    **_kwargs,
):
    """D-semantic chunked-prefill invariance for stem_paged_kv (quant_type=0).

    Three oracles per chunk:
      (inv)    chunked CUDA vs sliced 1-chunk normal CUDA mask
      (ref)    chunked CUDA vs FP32 OAM + Python D-ref mask
      (budget) mask-count per row >= min(expected_budget, q_pos+1)
    """
    if not _HPC_AVAILABLE:
        print("ERROR: hpc module not available. Build hpc-ops first.")
        sys.exit(1)

    device = "cuda"
    L_blocks = 512
    L_tokens = L_blocks * stem_block_size  # 65536

    chunk_specs = [
        ("c4",     [128, 128, 128, 128]),
        ("c8",     [64] * 8),
        ("edge",   [511, 1]),
        ("uneven", [100, 50, 250, 112]),
    ]
    for label, sizes in chunk_specs:
        assert sum(sizes) == L_blocks, f"{label} chunk sizes must sum to {L_blocks}"

    PASS_THRESHOLD = 99.0

    print()
    print("=" * 100)
    print("  Stem Chunked-Prefill D-semantic Invariance (quant_type=0) — "
          f"prompt L={L_blocks} blocks ({L_tokens} tokens), alpha={alpha}")
    print(f"  Hq={num_head_q}, Hkv={num_head_kv}, Dqk={dim_qk}, stem_bs={stem_block_size}")
    print(f"  PASS threshold: match% >= {PASS_THRESHOLD}% (BF16 OAM tail noise allowed)")
    print("=" * 100)

    torch.cuda.empty_cache()
    d_full = _setup_paged_fp8_data(
        num_req, L_tokens, num_head_q, num_head_kv,
        dim_qk, dim_v, kv_block_size,
    )

    def _slice_chunk(q_start_tok, qi_tok, kv_cum_tok):
        max_pad = ((qi_tok + 127) // 128) * 128
        qscale_chunk = d_full["qscale"][:, :, q_start_tok:q_start_tok + max_pad].contiguous()
        q_chunk = d_full["q_fp8"][q_start_tok:q_start_tok + qi_tok].contiguous()
        cu_q = torch.tensor([0, qi_tok], dtype=torch.int32, device=device)
        kv_seq = torch.tensor([kv_cum_tok], dtype=torch.int32, device=device)
        num_prompt = torch.tensor([L_tokens], dtype=torch.int32, device=device)
        return dict(
            q_fp8=q_chunk,
            kcache_fp8=d_full["kcache_fp8"],
            vcache_fp8=d_full["vcache_fp8"],
            qscale=qscale_chunk,
            kscale_fp32=d_full["kscale_fp32"],
            vscale=d_full["vscale"],
            kv_indices=d_full["kv_indices"],
            cu_q_seqlens=cu_q,
            kv_seqlens=kv_seq,
            num_prompt_tokens=num_prompt,
            kv_block_size=d_full["kv_block_size"],
        )

    def _run_cuda(d_chunk):
        return hpc.stem_paged_kv(
            d_chunk["q_fp8"],
            d_chunk["kcache_fp8"], d_chunk["vcache_fp8"],
            d_chunk["qscale"], d_chunk["kscale_fp32"], d_chunk["vscale"],
            d_chunk["kv_indices"], d_chunk["cu_q_seqlens"], d_chunk["kv_seqlens"],
            d_chunk["num_prompt_tokens"],
            lambda_mag=lambda_mag, alpha=alpha,
            stem_block_size=stem_block_size, stem_stride=stride,
            causal=True,
            initial_blocks=initial_blocks, window_size=window_size,
            k_block_num_rate_medium=k_block_num_rate_medium,
            k_block_num_bias_medium=k_block_num_bias_medium,
            k_block_num_rate_large=k_block_num_rate_large,
            k_block_num_bias_large=k_block_num_bias_large,
            quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD,
        )

    print()
    print("  [1] Building ground-truth mask via 1-chunk normal prefill ...")
    d_gt = _slice_chunk(0, L_tokens, L_tokens)
    mask_normal = _run_cuda(d_gt)
    print(f"      mask_normal shape: {tuple(mask_normal.shape)}")

    overall_pass = True

    for spec_name, chunk_sizes in chunk_specs:
        print()
        print(f"  [2] spec={spec_name}  chunk_sizes={chunk_sizes}")
        print(f"      {'chunk':>5}  {'qi_blk':>6} {'ki_blk':>6} {'kb_off':>6}  "
              f"{'inv%':>7} {'ref%':>7} {'budget':>10}  {'verdict':>8}")
        print(f"      {'-' * 78}")

        q_start_blk = 0
        for ci, qi_blocks_c in enumerate(chunk_sizes):
            q_start_tok = q_start_blk * stem_block_size
            qi_tok = qi_blocks_c * stem_block_size
            kv_cum_blk = q_start_blk + qi_blocks_c
            kv_cum_tok = kv_cum_blk * stem_block_size
            kb_off = kv_cum_blk - qi_blocks_c
            q_start_blk = kv_cum_blk

            torch.cuda.empty_cache()
            d_chunk = _slice_chunk(q_start_tok, qi_tok, kv_cum_tok)
            q_seqlens_chunk = (d_chunk["cu_q_seqlens"][1:] - d_chunk["cu_q_seqlens"][:-1]
                               ).to(torch.int32)

            mask_cuda = _run_cuda(d_chunk)

            qi_blk = min(qi_blocks_c, mask_cuda.shape[2])
            ki_blk = min(kv_cum_blk, mask_cuda.shape[3])

            # (i) invariance vs sliced normal mask.
            q_lo = q_start_tok // stem_block_size
            q_hi = q_lo + qi_blk
            normal_slice = mask_normal[:, :, q_lo:q_hi, :ki_blk].contiguous()
            inv_stats = _tpd_mask_compare(normal_slice, mask_cuda, qi_blk, ki_blk)

            # (ii) D-ref vs chunked CUDA.
            ref_logits = _oam_math_reference(
                d_chunk["q_fp8"], d_chunk["qscale"],
                d_chunk["kcache_fp8"], d_chunk["vcache_fp8"],
                d_chunk["kscale_fp32"], d_chunk["vscale"],
                d_chunk["kv_indices"], d_chunk["cu_q_seqlens"], d_chunk["kv_seqlens"],
                q_seqlens_chunk,
                stride=stride, block_size=stem_block_size,
                lambda_mag=lambda_mag, causal=True,
                dtype=torch.float32, device=device,
                kv_block_size=d_chunk["kv_block_size"],
            )
            mask_ref = _generate_sparse_mask_D_semantic(
                ref_logits, q_seqlens_chunk, d_chunk["kv_seqlens"],
                d_chunk["num_prompt_tokens"],
                block_size=stem_block_size, alpha=alpha,
                initial_blocks=initial_blocks, window_size=window_size,
                k_block_num_rate_medium=k_block_num_rate_medium,
                k_block_num_bias_medium=k_block_num_bias_medium,
                k_block_num_rate_large=k_block_num_rate_large,
                k_block_num_bias_large=k_block_num_bias_large,
                device=device,
            )
            ref_stats = _tpd_mask_compare(mask_ref, mask_cuda, qi_blk, ki_blk)

            # (iii) budget lower-bound sanity: cnt >= min(expected, q_pos+1).
            row_counts = mask_cuda[0, 0, :qi_blk, :ki_blk].sum(dim=-1).long().tolist()
            bad_rows = 0
            for r, cnt in enumerate(row_counts):
                q_pos = r + kb_off
                exp_b = _compute_expected_budget(
                    q_pos, L_blocks, alpha,
                    k_block_num_rate_medium, k_block_num_bias_medium,
                    k_block_num_rate_large, k_block_num_bias_large,
                )
                eff_b = min(exp_b, q_pos + 1)
                if cnt < eff_b:
                    bad_rows += 1
            budget_str = f"{qi_blk - bad_rows}/{qi_blk}"

            ok_inv = inv_stats["match_pct"] >= PASS_THRESHOLD
            ok_ref = ref_stats["match_pct"] >= PASS_THRESHOLD
            ok_budget = bad_rows == 0
            ok = ok_inv and ok_ref and ok_budget
            overall_pass &= ok

            print(f"      {ci:>5}  {qi_blk:>6} {ki_blk:>6} {kb_off:>6}  "
                  f"{inv_stats['match_pct']:>6.3f}% {ref_stats['match_pct']:>6.3f}% "
                  f"{budget_str:>10}  {'PASS' if ok else 'FAIL':>8}")

    print()
    print("  " + "-" * 98)
    print("  Oracles: (inv) chunked CUDA vs sliced 1-chunk normal CUDA;"
          "  (ref) chunked CUDA vs FP32 OAM + Python D-ref;")
    print("           (budget) rows where mask-count >= min(expected_budget, q_pos+1).")
    print(f"  ** Chunked D-semantic {'ALL PASSED' if overall_pass else 'FAILED'} **")
    print("=" * 100 + "\n")


# ====================================================================
# CLI
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stem block scoring + mask generation — quant_type=0 verification",
    )
    parser.add_argument(
        "--verify-prep",
        action="store_true",
        help="Run prep operators (KV + Q) correctness + benchmark (quant_type=0)",
    )
    parser.add_argument(
        "--verify-gemm",
        action="store_true",
        help="GEMM kernel accuracy + speed vs PyTorch matmul (quant_type=0 inputs)",
    )
    parser.add_argument(
        "--verify-oam",
        action="store_true",
        help="Full OAM pipeline accuracy vs FP32 ref + CUDA-only speed (quant_type=0)",
    )
    parser.add_argument(
        "--verify-tpd",
        action="store_true",
        help="TPD mask generation accuracy (vs PyTorch) + speed (vs Python) — scale-agnostic",
    )
    parser.add_argument(
        "--verify-stemAll",
        action="store_true",
        help="End-to-end pipeline: OAM+TPD accuracy vs FP32 ref + speed (quant_type=0)",
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Chunked-prefill D-semantic invariance (quant_type=0)",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths (default: built-in lists for verify / bench)",
    )
    parser.add_argument("--num-req", type=int, default=1)
    parser.add_argument("--num-head-q", type=int, default=8)
    parser.add_argument("--num-head-kv", type=int, default=1)
    parser.add_argument("--dim-qk", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--kv-block-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    _common_kwargs = dict(
        seq_lens=args.seq_lens,
        num_req=args.num_req,
        num_head_q=args.num_head_q,
        num_head_kv=args.num_head_kv,
        dim_qk=args.dim_qk,
        dim_v=args.dim_v,
        kv_block_size=args.kv_block_size,
        stride=args.stride,
        stem_block_size=args.block_size,
    )

    if args.chunked:
        verify_chunked(**_common_kwargs)
    elif args.verify_stemAll:
        verify_stem_all(**_common_kwargs)
    elif args.verify_tpd:
        verify_tpd(**_common_kwargs)
    elif args.verify_oam:
        verify_oam(**_common_kwargs)
    elif args.verify_gemm:
        verify_gemm(**_common_kwargs)
    elif args.verify_prep:
        verify_prep(**_common_kwargs)
    else:
        parser.print_help()
        print(
            "\nNo verification mode selected.  Pick one of:\n"
            "  --verify-prep      KV/Q prep correctness + benchmark\n"
            "  --verify-gemm      GEMM accuracy + speed\n"
            "  --verify-oam       OAM pipeline accuracy + CUDA-only speed\n"
            "  --verify-tpd       TPD mask accuracy + speed\n"
            "  --verify-stemAll   End-to-end pipeline accuracy + speed\n"
            "  --chunked          Chunked-prefill D-semantic invariance\n"
        )
