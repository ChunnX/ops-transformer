"""
Independent formula verification for stem_oam_prep_paged_kv golden.

This script implements the mathematical formulas from the requirement document
FROM SCRATCH (without importing or reusing the golden), then compares results.

Formulas source: StemOamPrepPagedKV需求分析和方案设计说明书.md (阶段1-3)

Key formulas:
  阶段1: K_scaled = K_dense * kscale_per_row  (per-token per-head, broadcast on dim)
  阶段2: K_group_sum[kb,g,:] = Σ_{r=0}^{R-1} K_blocks[kb,r,g,:]
         K_group_rev[kb,g',:] = K_group_sum[kb, S-1-g', :]  (anti-diag flip)
         kFlat = Cast<BF16>(reshape(K_group_rev, [S*D]))
  阶段3a: norms[idx,s] = sqrt(Σ_d (V_rows[idx,s,d] * vScale[h])^2)
          v_norm_down[idx] = max_s norms[idx,s]  (zero beyond kv_len)
          log_vals[idx] = log(v_norm_down[idx] + ε)
  阶段3b: μ = mean(log_vals), σ = sqrt(Σ(log_vals-μ)^2 / (k_down_len-1))  (unbiased)
  阶段3c: normalized = (log_vals - μ) / (σ + ε)
          v_final = λ * ReLU(normalized)
          vBias[kb] = (1/R) * Σ_{r=0}^{R-1} v_final[kb*R + r]
"""
import torch
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_oam_prep_paged_kv_golden import stem_oam_prep_paged_kv_golden

EPSILON = 1e-6


def manual_prep_kv(
    kcache_fp8, vcache_fp8, kscale_fp32, vscale,
    kv_indices, kv_seq_lens, kv_block_size,
    lambda_mag, kv_layout, stem_block_size, stem_stride,
):
    """Manual implementation following the requirement doc formulas EXACTLY.

    Each step is annotated with the corresponding formula from the doc.
    Returns (kflat_expected, vbias_expected) in FP32 for comparison.
    """
    device = kcache_fp8.device
    num_batch = kv_seq_lens.shape[0]

    if kv_layout == 0:
        H_kv = kcache_fp8.shape[2]
    else:
        H_kv = kcache_fp8.shape[1]
    D = kcache_fp8.shape[3]
    R = stem_block_size // stem_stride

    # Cast to FP32 upfront (FP8 doesn't support indexing on CPU)
    kcache_f32 = kcache_fp8.to(torch.float32)
    vcache_f32 = vcache_fp8.to(torch.float32)

    # --- 阶段1派生值 ---
    # k_padded[b] = ceil(kv_len[b] / B) * B
    kv_lens = [kv_seq_lens[b].item() for b in range(num_batch)]
    k_padded_list = [((l + stem_block_size - 1) // stem_block_size) * stem_block_size for l in kv_lens]
    max_k_padded = max(k_padded_list)
    max_Kb = max_k_padded // stem_block_size

    kflat_exp = torch.zeros(num_batch, H_kv, max_Kb, stem_stride * D, dtype=torch.float32, device=device)
    vbias_exp = torch.zeros(num_batch, H_kv, max_Kb, dtype=torch.float32, device=device)

    for b in range(num_batch):
        kv_len = kv_lens[b]
        k_padded = k_padded_list[b]
        num_Kb = k_padded // stem_block_size
        k_down_len = num_Kb * R  # = k_padded // stem_stride

        if kv_len == 0:
            continue

        # ===== 阶段1: De-page + Cast FP32 + kScaleCache Gather =====
        num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size
        block_ids = kv_indices[b, :num_kv_blocks].to(torch.long)

        # De-page K: gather pages via kv_indices, then flatten to [rows, H_kv, D]
        # 阶段1公式: K_dense = Cast(gather(kcache[kv_indices[b]]), FP32), zero-pad beyond kv_len
        if kv_layout == 0:
            # Layout A: [total_blocks, kvBlockSize, H_kv, D] -> gather -> [num_kv_blocks, kvBlockSize, H_kv, D]
            # -> reshape to [num_kv_blocks*kvBlockSize, H_kv, D]
            k_paged = kcache_f32[block_ids].reshape(-1, H_kv, D)
            v_paged = vcache_f32[block_ids].reshape(-1, H_kv, D)
        else:
            # Layout B: [total_blocks, H_kv, kvBlockSize, D] -> gather -> [num_kv_blocks, H_kv, kvBlockSize, D]
            # -> permute(0,2,1,3) -> [num_kv_blocks, kvBlockSize, H_kv, D] -> reshape
            k_paged = kcache_f32[block_ids].permute(0, 2, 1, 3).reshape(-1, H_kv, D)
            v_paged = vcache_f32[block_ids].permute(0, 2, 1, 3).reshape(-1, H_kv, D)

        # Zero-pad to k_padded
        k_dense = torch.zeros(k_padded, H_kv, D, dtype=torch.float32, device=device)
        v_dense = torch.zeros(k_padded, H_kv, D, dtype=torch.float32, device=device)
        actual_rows = min(kv_len, num_kv_blocks * kv_block_size)
        k_dense[:actual_rows] = k_paged[:actual_rows]
        v_dense[:actual_rows] = v_paged[:actual_rows]

        # Gather kScaleCache: same layout logic as kCache, but D=1
        # 阶段1公式: kscale_per_row[b] = gather(kScaleCache[kv_indices[b]])
        kscale_per_row = torch.zeros(k_padded, H_kv, dtype=torch.float32, device=device)
        for row in range(k_padded):
            if row >= kv_len:
                continue  # zero-pad
            logical_block = row // kv_block_size
            row_in_block = row % kv_block_size
            phys_block = kv_indices[b, logical_block].item()
            if kv_layout == 0:
                # kScaleCache[phys, row_in_block, h, 0]
                kscale_per_row[row, :] = kscale_fp32[phys_block, row_in_block, :, 0]
            else:
                # kScaleCache[phys, h, row_in_block, 0]
                kscale_per_row[row, :] = kscale_fp32[phys_block, :, row_in_block, 0]

        # K_scaled[b,h] = K_dense[b,h,:] * kscale_per_row[b,h,:]  (broadcast on dim)
        k_scaled = k_dense[:num_Kb * stem_block_size] * kscale_per_row[:num_Kb * stem_block_size].unsqueeze(-1)

        # ===== 阶段2: K Processing (Group Sum + Anti-diagonal Flip) =====
        for h in range(H_kv):
            k_h = k_scaled[:, h, :]  # [num_Kb*B, D]

            # K_blocks = reshape(K_scaled[:num_Kb*B, h, :], [num_Kb, R, S, D])
            k_blocks = k_h.reshape(num_Kb, R, stem_stride, D)

            # K_group_sum[kb,g,:] = Σ_{r=0}^{R-1} K_blocks[kb,r,g,:]
            k_group_sum = torch.zeros(num_Kb, stem_stride, D, dtype=torch.float32, device=device)
            for kb in range(num_Kb):
                for g in range(stem_stride):
                    for r in range(R):
                        k_group_sum[kb, g, :] += k_blocks[kb, r, g, :]

            # K_group_rev[kb,g',:] = K_group_sum[kb, S-1-g', :]  (anti-diagonal flip)
            k_group_rev = torch.zeros(num_Kb, stem_stride, D, dtype=torch.float32, device=device)
            for kb in range(num_Kb):
                for g_prime in range(stem_stride):
                    g = stem_stride - 1 - g_prime
                    k_group_rev[kb, g_prime, :] = k_group_sum[kb, g, :]

            # kFlat[kb,:] = reshape(K_group_rev[kb,:,:], [S*D]) -> Cast BF16 (keep FP32 for comparison)
            kflat_exp[b, h, :num_Kb, :] = k_group_rev.reshape(num_Kb, stem_stride * D)

        # ===== 阶段3: V Processing =====
        for h in range(H_kv):
            vscale_h = vscale[h].item()

            # --- Step 3a: L2 Norm + Max Pool + Log ---
            # V_rows = reshape(V_dense[:k_down_len*S, h, :], [k_down_len, S, D])
            v_h = v_dense[:num_Kb * stem_block_size, h, :]
            v_rows = v_h.reshape(k_down_len, stem_stride, D)

            # norms[idx,s] = sqrt(Σ_d (V_rows[idx,s,d] * vScale[h])^2)
            norms = torch.zeros(k_down_len, stem_stride, dtype=torch.float32, device=device)
            for idx in range(k_down_len):
                for s in range(stem_stride):
                    row_idx = idx * stem_stride + s
                    if row_idx >= kv_len:
                        norms[idx, s] = 0.0  # zero beyond kv_len
                    else:
                        val = v_rows[idx, s, :] * vscale_h
                        norms[idx, s] = math.sqrt(float((val ** 2).sum()))

            # v_norm_down[idx] = max_s norms[idx,s]
            v_norm_down = norms.max(dim=-1).values

            # log_vals[idx] = log(v_norm_down[idx] + ε)
            log_vals = torch.tensor(
                [math.log(float(v) + EPSILON) for v in v_norm_down],
                dtype=torch.float32, device=device,
            )

            # --- Step 3b: Global Normalize (μ/σ over ALL k_down_len) ---
            if k_down_len > 0:
                mu = float(log_vals.mean())
                if k_down_len > 1:
                    # σ = sqrt(Σ(log_vals-μ)^2 / (k_down_len-1))  (unbiased std)
                    sigma = math.sqrt(float(((log_vals - mu) ** 2).sum()) / (k_down_len - 1))
                else:
                    sigma = 0.0
            else:
                mu = 0.0
                sigma = 0.0

            # --- Step 3c: Normalize + ReLU + Block Average ---
            # normalized[idx] = (log_vals[idx] - μ) / (σ + ε)
            normalized = (log_vals - mu) / (sigma + EPSILON)

            # v_final[idx] = λ * ReLU(normalized[idx])
            v_final = lambda_mag * torch.relu(normalized)

            # vBias[kb] = (1/R) * Σ_{r=0}^{R-1} v_final[kb*R + r]
            for kb in range(num_Kb):
                s = 0.0
                for r in range(R):
                    s += float(v_final[kb * R + r])
                vbias_exp[b, h, kb] = s / R

    # Cast kFlat through BF16 to match golden output precision
    # (requirement doc: kFlat output dtype = BF16, so both sides must go through BF16 rounding)
    kflat_exp = kflat_exp.to(torch.bfloat16).to(torch.float32)

    return kflat_exp, vbias_exp


def compare_results(name, golden_out, expected_out, rtol, atol):
    """Compare golden output with manually computed expected output."""
    if golden_out.dtype == torch.bfloat16:
        golden_f32 = golden_out.float()
    else:
        golden_f32 = golden_out

    diff = (golden_f32 - expected_out).abs()
    max_diff = diff.max().item()
    max_abs = expected_out.abs().max().item()

    # Element-wise rtol/atol check
    within_tol = torch.isclose(golden_f32, expected_out, rtol=rtol, atol=atol)
    pass_rate = within_tol.float().mean().item() * 100
    all_pass = bool(within_tol.all())

    status = "PASS" if all_pass else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.2e}, max_abs={max_abs:.2e}, "
          f"pass_rate={pass_rate:.1f}%, tol=(rtol={rtol}, atol={atol})")

    if not all_pass:
        fail_mask = ~within_tol
        fail_indices = torch.nonzero(fail_mask, as_tuple=False)
        num_show = min(5, fail_indices.shape[0])
        print(f"         First {num_show} mismatches:")
        for i in range(num_show):
            idx = fail_indices[i]
            idx_tuple = tuple(idx.tolist())
            g = golden_f32[idx_tuple].item()
            e = expected_out[idx_tuple].item()
            print(f"           idx={idx_tuple}: golden={g:.6f}, expected={e:.6f}, diff={abs(g-e):.2e}")

    return all_pass


# ============================================================
# Test configurations: (name, batch, H_kv, D, kv_block_size, total_blocks,
#                        max_kv_blocks, kv_layout, kv_seq_lens, stem_block_size, stem_stride, lambda_mag)
# ============================================================
configs = [
    ("Basic Layout B",        2, 2, 128, 64,  8, 4, 1, [200, 100], 128, 16, 0.3),
    ("Basic Layout A",        2, 2, 128, 64,  8, 4, 0, [200, 100], 128, 16, 0.3),
    ("kv_block_size=128",     2, 2, 128, 128, 8, 4, 1, [300, 150], 128, 16, 0.3),
    ("B=256 S=32 (R=8)",      1, 2, 128, 128, 8, 4, 1, [500],      256, 32, 0.3),
    ("B=32 S=16 (R=2)",       1, 1, 128, 32,  4, 4, 1, [100],      32,  16, 0.3),
    ("kv_len=0 boundary",     2, 2, 128, 64,  8, 4, 1, [0, 100],   128, 16, 0.3),
    ("kv_len=1 boundary",     1, 1, 128, 64,  4, 1, 1, [1],        128, 16, 0.3),
    ("lambda=0.5",            1, 2, 128, 64,  8, 8, 1, [300],      128, 16, 0.5),
    ("Unaligned kv_len=100",  1, 1, 128, 64,  8, 4, 1, [100],      128, 16, 0.3),
]

all_pass = True
for i, (name, batch, H_kv, D, kv_bs, tb, mkb, layout, kv_lens, B_sz, S_sz, lam) in enumerate(configs):
    print(f"\n{'='*70}")
    print(f"Config {i+1}: {name}")
    print(f"  batch={batch}, H_kv={H_kv}, D={D}, kv_block_size={kv_bs}, "
          f"layout={'A' if layout==0 else 'B'}, B={B_sz}, S={S_sz}, lambda={lam}")
    print(f"  kv_seq_lens={kv_lens}, total_blocks={tb}, max_kv_blocks={mkb}")
    print(f"{'='*70}")

    torch.manual_seed(42 + i)
    kv_seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    kv_indices = torch.randint(0, tb, (batch, mkb), dtype=torch.int32)

    if layout == 0:
        kcache = (torch.rand(tb, kv_bs, H_kv, D) * 4 - 2).to(torch.float8_e4m3fn)
        vcache = (torch.rand(tb, kv_bs, H_kv, D) * 4 - 2).to(torch.float8_e4m3fn)
        kscale = torch.rand(tb, kv_bs, H_kv, 1, dtype=torch.float32) * 5
    else:
        kcache = (torch.rand(tb, H_kv, kv_bs, D) * 4 - 2).to(torch.float8_e4m3fn)
        vcache = (torch.rand(tb, H_kv, kv_bs, D) * 4 - 2).to(torch.float8_e4m3fn)
        kscale = torch.rand(tb, H_kv, kv_bs, 1, dtype=torch.float32) * 5

    vscale = torch.rand(H_kv, dtype=torch.float32) * 4

    # Run golden
    kflat_g, vbias_g = stem_oam_prep_paged_kv_golden(
        kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
        kv_block_size=kv_bs, lambda_mag=lam, kv_layout=layout,
        stem_block_size=B_sz, stem_stride=S_sz,
    )

    # Run manual formula computation
    kflat_m, vbias_m = manual_prep_kv(
        kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
        kv_block_size=kv_bs, lambda_mag=lam, kv_layout=layout,
        stem_block_size=B_sz, stem_stride=S_sz,
    )

    # Compare
    # kFlat: BF16 precision -> rtol=1e-3, atol=1e-3 (per requirement doc)
    # vBias: FP32 precision -> rtol=1e-5, atol=1e-5
    p1 = compare_results("kFlat", kflat_g, kflat_m, rtol=1e-3, atol=1e-3)
    p2 = compare_results("vBias", vbias_g, vbias_m, rtol=1e-5, atol=1e-5)

    if not (p1 and p2):
        all_pass = False

# ============================================================
print(f"\n{'='*70}")
print("Cross-check: Layout A vs Layout B produce identical results")
print(f"{'='*70}")
torch.manual_seed(100)
D, H_kv, kv_bs, tb, mkb = 128, 2, 64, 4, 4
B_sz, S_sz, lam = 128, 16, 0.3

# Create same data in both layouts
data_a = torch.randn(tb, kv_bs, H_kv, D)
data_b = data_a.permute(0, 2, 1, 3).contiguous()
kscale_a = torch.rand(tb, kv_bs, H_kv, 1, dtype=torch.float32) * 5
kscale_b = kscale_a.permute(0, 2, 1, 3).contiguous()
kv_idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
kv_sl = torch.tensor([200], dtype=torch.int32)
vs = torch.rand(H_kv, dtype=torch.float32) * 4

kf_a, vb_a = stem_oam_prep_paged_kv_golden(
    data_a.to(torch.float8_e4m3fn), data_a.to(torch.float8_e4m3fn),
    kscale_a, vs, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=lam, kv_layout=0,
    stem_block_size=B_sz, stem_stride=S_sz,
)
kf_b, vb_b = stem_oam_prep_paged_kv_golden(
    data_b.to(torch.float8_e4m3fn), data_b.to(torch.float8_e4m3fn),
    kscale_b, vs, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=lam, kv_layout=1,
    stem_block_size=B_sz, stem_stride=S_sz,
)
# Also verify both against manual
kf_m_a, vb_m_a = manual_prep_kv(
    data_a.to(torch.float8_e4m3fn), data_a.to(torch.float8_e4m3fn),
    kscale_a, vs, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=lam, kv_layout=0,
    stem_block_size=B_sz, stem_stride=S_sz,
)
kf_m_b, vb_m_b = manual_prep_kv(
    data_b.to(torch.float8_e4m3fn), data_b.to(torch.float8_e4m3fn),
    kscale_b, vs, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=lam, kv_layout=1,
    stem_block_size=B_sz, stem_stride=S_sz,
)

print(f"  Golden A vs Golden B: kflat diff={((kf_a.float()-kf_b.float()).abs().max().item()):.2e}, "
      f"vbias diff={((vb_a-vb_b).abs().max().item()):.2e}")
print(f"  Manual A vs Manual B: kflat diff={((kf_m_a-kf_m_b).abs().max().item()):.2e}, "
      f"vbias diff={((vb_m_a-vb_m_b).abs().max().item()):.2e}")
p1 = compare_results("Golden-A vs Manual-A kFlat", kf_a, kf_m_a, rtol=1e-3, atol=1e-3)
p2 = compare_results("Golden-A vs Manual-A vBias", vb_a, vb_m_a, rtol=1e-5, atol=1e-5)
p3 = compare_results("Golden-B vs Manual-B kFlat", kf_b, kf_m_b, rtol=1e-3, atol=1e-3)
p4 = compare_results("Golden-B vs Manual-B vBias", vb_b, vb_m_b, rtol=1e-5, atol=1e-5)
if not (p1 and p2 and p3 and p4):
    all_pass = False

# ============================================================
print(f"\n{'='*70}")
print("Manual step-by-step spot check (single element trace)")
print(f"{'='*70}")
# Small case where we can trace every step by hand
torch.manual_seed(0)
D, H_kv, kv_bs, tb, mkb = 128, 1, 32, 1, 1
B_sz, S_sz, lam = 32, 16, 0.3
R = B_sz // S_sz  # =2

# Set all K rows to known values, all scales to 1
kcache = torch.zeros(tb, H_kv, kv_bs, D, dtype=torch.float32)
for r in range(32):
    kcache[0, 0, r, :] = float(r + 1)
kcache_fp8 = kcache.to(torch.float8_e4m3fn)
kcache_rt = kcache_fp8.to(torch.float32)  # FP8 roundtrip

vcache = torch.ones(tb, H_kv, kv_bs, D, dtype=torch.float32).to(torch.float8_e4m3fn)
kscale = torch.ones(tb, H_kv, kv_bs, 1, dtype=torch.float32)
vscale = torch.ones(H_kv, dtype=torch.float32)
kv_idx = torch.tensor([[0]], dtype=torch.int32)
kv_sl = torch.tensor([32], dtype=torch.int32)

kf, vb = stem_oam_prep_paged_kv_golden(
    kcache_fp8, vcache, kscale, vscale, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=lam, kv_layout=1,
    stem_block_size=B_sz, stem_stride=S_sz,
)

print(f"\n  K Processing trace (R={R}, S={S_sz}, D={D}):")
print(f"  k_group_sum[g] = K_scaled[g] + K_scaled[g+16]  (sum over r=0,1)")
print(f"  k_group_rev[g'] = k_group_sum[15-g']  (anti-diag flip)")
print(f"  kFlat[g'*128 + 0] should = k_group_sum[15-g'][0]\n")

kf_f32 = kf[0, 0, 0].float()
spot_ok = True
for g_prime in range(S_sz):
    g = S_sz - 1 - g_prime
    # Use FP8-roundtripped values
    row_g = kcache_rt[0, 0, g, 0].item()
    row_g16 = kcache_rt[0, 0, g + 16, 0].item()
    expected = row_g + row_g16
    actual = kf_f32[g_prime * D].item()
    match = abs(actual - expected) < 0.2
    if not match:
        print(f"    g'={g_prime}: golden={actual}, manual={expected} (rows {g}+{g+16}: {row_g}+{row_g16}) MISMATCH")
        spot_ok = False
status = "PASS" if spot_ok else "FAIL"
print(f"  [{status}] K spot check (16 groups, anti-diag flip verified)")

print(f"\n  V Processing trace:")
print(f"  All V rows = ones(128), vScale=1")
print(f"  norms[idx,s] = sqrt(128 * 1^2) = sqrt(128) = {math.sqrt(128):.4f}")
print(f"  v_norm_down[idx] = max_s norms = {math.sqrt(128):.4f} (all same)")
print(f"  log_vals[idx] = log({math.sqrt(128):.4f} + 1e-6) = {math.log(math.sqrt(128)+1e-6):.6f}")
print(f"  All log_vals identical -> mu=that value, sigma=0")
print(f"  normalized = (log_val - mu)/(0+1e-6) = 0")
print(f"  ReLU(0) = 0, v_final = 0.3*0 = 0")
print(f"  vBias = mean(0, 0) = 0")
v0 = vb[0, 0, 0].item()
status = "PASS" if abs(v0) < 1e-5 else "FAIL"
print(f"  [{status}] vBias[0,0,0] = {v0:.8f} (expected 0.0)")
if not spot_ok or abs(v0) >= 1e-5:
    all_pass = False

# ============================================================
print(f"\n{'='*70}")
if all_pass:
    print("ALL FORMULA VERIFICATION TESTS PASSED")
else:
    print("SOME TESTS FAILED - see details above")
print(f"{'='*70}")
