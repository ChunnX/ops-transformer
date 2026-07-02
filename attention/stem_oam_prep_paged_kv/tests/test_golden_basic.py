import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stem_oam_prep_paged_kv_golden import stem_oam_prep_paged_kv_golden


def make_inputs(batch, H_kv, D, kv_block_size, total_blocks, max_kv_blocks,
                kv_layout, kv_seq_lens_list, device="cpu", seed=42):
    torch.manual_seed(seed)
    kv_seq_lens = torch.tensor(kv_seq_lens_list, dtype=torch.int32, device=device)
    kv_indices = torch.randint(0, total_blocks, (batch, max_kv_blocks), dtype=torch.int32, device=device)

    if kv_layout == 0:
        kcache = (torch.rand(total_blocks, kv_block_size, H_kv, D, device=device) * 4 - 2).to(torch.float8_e4m3fn)
        vcache = (torch.rand(total_blocks, kv_block_size, H_kv, D, device=device) * 4 - 2).to(torch.float8_e4m3fn)
        kscale = torch.rand(total_blocks, kv_block_size, H_kv, 1, dtype=torch.float32, device=device) * 5
    else:
        kcache = (torch.rand(total_blocks, H_kv, kv_block_size, D, device=device) * 4 - 2).to(torch.float8_e4m3fn)
        vcache = (torch.rand(total_blocks, H_kv, kv_block_size, D, device=device) * 4 - 2).to(torch.float8_e4m3fn)
        kscale = torch.rand(total_blocks, H_kv, kv_block_size, 1, dtype=torch.float32, device=device) * 5

    vscale = torch.rand(H_kv, dtype=torch.float32, device=device) * 4
    return kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens


def check_no_nan_inf(name, tensor):
    has_nan = tensor.isnan().any().item()
    has_inf = tensor.isinf().any().item()
    status = "PASS" if not (has_nan or has_inf) else "FAIL"
    print(f"  [{status}] {name}: no NaN={not has_nan}, no Inf={not has_inf}")
    return not (has_nan or has_inf)


def check_dtype(name, tensor, expected_dtype):
    status = "PASS" if tensor.dtype == expected_dtype else "FAIL"
    print(f"  [{status}] {name} dtype: got={tensor.dtype}, expected={expected_dtype}")
    return tensor.dtype == expected_dtype


def check_shape(name, tensor, expected_shape):
    status = "PASS" if list(tensor.shape) == list(expected_shape) else "FAIL"
    print(f"  [{status}] {name} shape: got={list(tensor.shape)}, expected={list(expected_shape)}")
    return list(tensor.shape) == list(expected_shape)


def calc_expected_shapes(kv_seq_lens, H_kv, D, stem_block_size, stem_stride):
    k_padded_lens = ((kv_seq_lens.to(torch.int64) + stem_block_size - 1) // stem_block_size * stem_block_size)
    max_k_padded = k_padded_lens.max().item()
    max_Kb = max_k_padded // stem_block_size
    kflat_dim = stem_stride * D
    return [kv_seq_lens.shape[0], H_kv, max_Kb, kflat_dim], [kv_seq_lens.shape[0], H_kv, max_Kb]


# ============================================================
print("=" * 70)
print("Test 1: Basic Layout B (kv_block_size=64, B=128, S=16)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=2, H_kv=2, D=128, kv_block_size=64, total_blocks=8, max_kv_blocks=4,
    kv_layout=1, kv_seq_lens_list=[200, 100],
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.3, kv_layout=1, stem_block_size=128, stem_stride=16,
)
exp_kflat, exp_vbias = calc_expected_shapes(kv_seq_lens, 2, 128, 128, 16)
check_shape("kflat", kflat, exp_kflat)
check_shape("vbias", vbias, exp_vbias)
check_dtype("kflat", kflat, torch.bfloat16)
check_dtype("vbias", vbias, torch.float32)
check_no_nan_inf("kflat", kflat)
check_no_nan_inf("vbias", vbias)
print(f"  kflat nonzero: {kflat.abs().sum().item() > 0}")
print(f"  vbias nonzero: {vbias.abs().sum().item() > 0}")
print(f"  vbias[0,0,:]: {[round(v, 4) for v in vbias[0, 0, :].tolist()]}")
print(f"  vbias[1,0,:]: {[round(v, 4) for v in vbias[1, 0, :].tolist()]}")

# ============================================================
print("\n" + "=" * 70)
print("Test 2: Basic Layout A (kv_block_size=64, B=128, S=16)")
print("=" * 70)
kcache_a, vcache_a, kscale_a, vscale_a, kv_indices_a, kv_seq_lens_a = make_inputs(
    batch=2, H_kv=2, D=128, kv_block_size=64, total_blocks=8, max_kv_blocks=4,
    kv_layout=0, kv_seq_lens_list=[200, 100], seed=42,
)
kflat_a, vbias_a = stem_oam_prep_paged_kv_golden(
    kcache_a, vcache_a, kscale_a, vscale_a, kv_indices_a, kv_seq_lens_a,
    kv_block_size=64, lambda_mag=0.3, kv_layout=0, stem_block_size=128, stem_stride=16,
)
exp_kflat_a, exp_vbias_a = calc_expected_shapes(kv_seq_lens_a, 2, 128, 128, 16)
check_shape("kflat", kflat_a, exp_kflat_a)
check_shape("vbias", vbias_a, exp_vbias_a)
check_no_nan_inf("kflat", kflat_a)
check_no_nan_inf("vbias", vbias_a)
print(f"  kflat nonzero: {kflat_a.abs().sum().item() > 0}")
print(f"  vbias nonzero: {vbias_a.abs().sum().item() > 0}")

# ============================================================
print("\n" + "=" * 70)
print("Test 3: Layout A vs B consistency (same data, different memory layout)")
print("=" * 70)
D, H_kv, kv_bs, tb, mkb = 128, 2, 64, 4, 4
data_a = torch.randn(tb, kv_bs, H_kv, D)
data_b = data_a.permute(0, 2, 1, 3).contiguous()
kscale_a = torch.rand(tb, kv_bs, H_kv, 1, dtype=torch.float32) * 5
kscale_b = kscale_a.permute(0, 2, 1, 3).contiguous()
kv_idx = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
kv_sl = torch.tensor([200], dtype=torch.int32)
vs = torch.rand(H_kv, dtype=torch.float32) * 4
kf_a, vb_a = stem_oam_prep_paged_kv_golden(
    data_a.to(torch.float8_e4m3fn), data_a.to(torch.float8_e4m3fn), kscale_a, vs, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=0.3, kv_layout=0, stem_block_size=128, stem_stride=16,
)
kf_b, vb_b = stem_oam_prep_paged_kv_golden(
    data_b.to(torch.float8_e4m3fn), data_b.to(torch.float8_e4m3fn), kscale_b, vs, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=0.3, kv_layout=1, stem_block_size=128, stem_stride=16,
)
kflat_diff = (kf_a.float() - kf_b.float()).abs().max().item()
vbias_diff = (vb_a - vb_b).abs().max().item()
status = "PASS" if kflat_diff == 0 and vbias_diff == 0 else "FAIL"
print(f"  [{status}] kflat max diff: {kflat_diff:.2e}")
print(f"  [{status}] vbias max diff: {vbias_diff:.2e}")

# ============================================================
print("\n" + "=" * 70)
print("Test 4: kv_block_size=128")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=2, H_kv=2, D=128, kv_block_size=128, total_blocks=8, max_kv_blocks=4,
    kv_layout=1, kv_seq_lens_list=[300, 150],
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=128, lambda_mag=0.3, kv_layout=1, stem_block_size=128, stem_stride=16,
)
exp_kflat, exp_vbias = calc_expected_shapes(kv_seq_lens, 2, 128, 128, 16)
check_shape("kflat", kflat, exp_kflat)
check_shape("vbias", vbias, exp_vbias)
check_no_nan_inf("kflat", kflat)
check_no_nan_inf("vbias", vbias)
print(f"  kflat nonzero: {kflat.abs().sum().item() > 0}")
print(f"  vbias nonzero: {vbias.abs().sum().item() > 0}")

# ============================================================
print("\n" + "=" * 70)
print("Test 5: Different stem_block_size=256, stem_stride=32 (R=8)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=1, H_kv=2, D=128, kv_block_size=128, total_blocks=8, max_kv_blocks=4,
    kv_layout=1, kv_seq_lens_list=[500],
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=128, lambda_mag=0.3, kv_layout=1, stem_block_size=256, stem_stride=32,
)
exp_kflat, exp_vbias = calc_expected_shapes(kv_seq_lens, 2, 128, 256, 32)
check_shape("kflat", kflat, exp_kflat)
check_shape("vbias", vbias, exp_vbias)
check_no_nan_inf("kflat", kflat)
check_no_nan_inf("vbias", vbias)
print(f"  kflat_dim = 32*128 = {kflat.shape[3]}")
print(f"  R = 256/32 = 8")
print(f"  kflat nonzero: {kflat.abs().sum().item() > 0}")
print(f"  vbias nonzero: {vbias.abs().sum().item() > 0}")

# ============================================================
print("\n" + "=" * 70)
print("Test 6: Boundary kv_seq_lens=0 (output should be all zero)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=2, H_kv=2, D=128, kv_block_size=64, total_blocks=8, max_kv_blocks=4,
    kv_layout=1, kv_seq_lens_list=[0, 100],
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.3, kv_layout=1, stem_block_size=128, stem_stride=16,
)
b0_kflat_zero = kflat[0].abs().sum().item() == 0
b0_vbias_zero = vbias[0].abs().sum().item() == 0
b1_vbias_nonzero = vbias[1].abs().sum().item() > 0
status = "PASS" if b0_kflat_zero and b0_vbias_zero and b1_vbias_nonzero else "FAIL"
print(f"  [{status}] batch0 kflat all zero: {b0_kflat_zero}")
print(f"  [{status}] batch0 vbias all zero: {b0_vbias_zero}")
print(f"  [{status}] batch1 vbias nonzero: {b1_vbias_nonzero}")

# ============================================================
print("\n" + "=" * 70)
print("Test 7: Boundary k_down_len=1 (R=1, std=0 -> normalized=0 -> VBias=0)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=1, H_kv=1, D=128, kv_block_size=64, total_blocks=4, max_kv_blocks=1,
    kv_layout=1, kv_seq_lens_list=[1],
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.3, kv_layout=1, stem_block_size=32, stem_stride=32,
)
R = 32 // 32
k_down_len = 1 * R
print(f"  R={R}, k_down_len={k_down_len}")
check_no_nan_inf("kflat", kflat)
check_no_nan_inf("vbias", vbias)
v0_zero = vbias[0, 0, 0].item() == 0.0
status = "PASS" if v0_zero else "FAIL"
print(f"  [{status}] vbias[0,0,0]=0 (mu=logVal, normalized=0, ReLU(0)=0): {v0_zero}")
print(f"  kflat nonzero (K processing independent of V): {kflat.abs().sum().item() > 0}")

# ============================================================
print("\n" + "=" * 70)
print("Test 8: Boundary kv_len=1, R=8 (k_down_len=8, std!=0)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=1, H_kv=1, D=128, kv_block_size=64, total_blocks=4, max_kv_blocks=1,
    kv_layout=1, kv_seq_lens_list=[1],
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.3, kv_layout=1, stem_block_size=128, stem_stride=16,
)
R = 128 // 16
k_down_len = 1 * R
print(f"  R={R}, k_down_len={k_down_len} (only 1 valid row, 7 are zero-padded)")
check_no_nan_inf("kflat", kflat)
check_no_nan_inf("vbias", vbias)
print(f"  vbias[0,0,0] = {vbias[0, 0, 0].item():.6f} (should be 0: only 1 nonzero, but mu=single value, normalized=0)")

# ============================================================
print("\n" + "=" * 70)
print("Test 9: lambda_mag effect (larger lambda -> larger vBias)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=1, H_kv=1, D=128, kv_block_size=64, total_blocks=8, max_kv_blocks=8,
    kv_layout=1, kv_seq_lens_list=[300],
)
_, vbias_low = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.1, kv_layout=1, stem_block_size=128, stem_stride=16,
)
_, vbias_high = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.9, kv_layout=1, stem_block_size=128, stem_stride=16,
)
vlow = vbias_low.abs().sum().item()
vhigh = vbias_high.abs().sum().item()
status = "PASS" if vhigh >= vlow else "FAIL"
print(f"  [{status}] lambda=0.1 vBias sum={vlow:.4f}")
print(f"  [{status}] lambda=0.9 vBias sum={vhigh:.4f}")
print(f"  larger lambda -> larger/equal VBias: {vhigh >= vlow}")

# ============================================================
print("\n" + "=" * 70)
print("Test 10: Padding correctness (kv_len not aligned to kv_block_size)")
print("=" * 70)
kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens = make_inputs(
    batch=1, H_kv=1, D=128, kv_block_size=64, total_blocks=8, max_kv_blocks=4,
    kv_layout=1, kv_seq_lens_list=[100],  # 100 tokens, 2 kv_blocks (128 rows), 28 padding rows
)
kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
    kv_block_size=64, lambda_mag=0.3, kv_layout=1, stem_block_size=128, stem_stride=16,
)
check_no_nan_inf("kflat", kflat)
check_no_nan_inf("vbias", vbias)
print(f"  kv_len=100, kv_block_size=64 -> 2 kv_blocks (128 rows), 28 padding")
print(f"  num_Kb = ceil(100/128) = 1")
print(f"  kflat shape: {list(kflat.shape)}")
print(f"  vbias shape: {list(vbias.shape)}")
print(f"  kflat nonzero: {kflat.abs().sum().item() > 0}")

# ============================================================
print("\n" + "=" * 70)
print("Test 11: Manual K processing verification (small case)")
print("=" * 70)
# Manually verify K processing with known values
torch.manual_seed(0)
B_sz, S_sz, D_sz, H_sz = 32, 16, 128, 1  # R=2
kv_bs = 32
total_b = 1
kv_idx = torch.tensor([[0]], dtype=torch.int32)
kv_sl = torch.tensor([32], dtype=torch.int32)  # exactly 1 stem block

# Create kcache where each row has known values (use powers of 2 for FP8 exactness)
kcache = torch.zeros(total_b, H_sz, kv_bs, D_sz, dtype=torch.float32)
for r in range(32):
    kcache[0, 0, r, :] = float(r + 1)  # row r filled with value (r+1)
kcache_fp8 = kcache.to(torch.float8_e4m3fn)
kcache_roundtrip = kcache_fp8.to(torch.float32)  # FP8 quantized values

vcache = torch.ones(total_b, H_sz, kv_bs, D_sz, dtype=torch.float32).to(torch.float8_e4m3fn)
kscale = torch.ones(total_b, H_sz, kv_bs, 1, dtype=torch.float32)  # all scale=1
vscale = torch.ones(H_sz, dtype=torch.float32)

kflat, vbias = stem_oam_prep_paged_kv_golden(
    kcache_fp8, vcache, kscale, vscale, kv_idx, kv_sl,
    kv_block_size=kv_bs, lambda_mag=0.3, kv_layout=1, stem_block_size=B_sz, stem_stride=S_sz,
)
R = B_sz // S_sz  # =2
# K processing with FP8-quantized values:
# k_group_sum[g] = roundtrip[g] + roundtrip[g+16]
# flip: k_group_rev[g'] = k_group_sum[15-g']
kflat_f32 = kflat[0, 0, 0].float()
manual_ok = True
for g_prime in range(16):
    g = 15 - g_prime  # un-flip
    row_g_val = kcache_roundtrip[0, 0, g, 0].item()
    row_g16_val = kcache_roundtrip[0, 0, g + 16, 0].item()
    expected = row_g_val + row_g16_val
    actual = kflat_f32[g_prime * 128].item()
    if abs(actual - expected) > 0.15:
        print(f"  [FAIL] group {g_prime}: got {actual}, expected {expected} "
              f"(FP8 vals: {row_g_val}+{row_g16_val})")
        manual_ok = False
status = "PASS" if manual_ok else "FAIL"
print(f"  [{status}] K group sum + anti-diag flip manual verification (FP8-aware)")
if manual_ok:
    actual_vals = [round(kflat_f32[g * 128].item(), 1) for g in range(16)]
    print(f"  Actual kflat values (per group, first element): {actual_vals}")
    print(f"  (values reflect FP8 quantization rounding)")

# ============================================================
print("\n" + "=" * 70)
print("Test 12: V Processing L2 Norm + Max + Log manual verification")
print("=" * 70)
# With kscale=1, vscale=1, vcache all ones
# v_h[row, :] = ones(128) -> L2 norm = sqrt(128*1^2) = sqrt(128) = 11.3137
# All rows same norm -> max = 11.3137 for all idx
# log(11.3137 + 1e-6) = 2.4277
# All log_vals same -> mu=2.4277, sigma=0, normalized=0, ReLU(0)=0 -> VBias=0
v0 = vbias[0, 0, 0].item()
expected_v0 = 0.0
status = "PASS" if abs(v0 - expected_v0) < 1e-5 else "FAIL"
print(f"  [{status}] V manual: vbias[0,0,0]={v0:.6f}, expected={expected_v0} (all-ones V -> sigma=0 -> VBias=0)")

# ============================================================
print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
