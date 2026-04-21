#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detailed breakdown of MM1 under Split-KN N4×K7 (perf_decode_bs1, BF16, 950).

Shows every sub-component that estimate_matmul_split_kn computes:
  - Per-core tile geometry (M × K_per_core × N_per_core)
  - Outer MTE1 load breakdown (A vs B, HBM vs L2)
  - Inner loop: L0A, L0B, MMAD per stepK
  - FixPipe (to GM since k_groups > 1 → FP32 partials)
  - Vector accumulation (AccumCq) on AIV side
"""
from perf.cli.formatting import params_from_testcase
from perf.model.hardware import ASCEND_950
from perf.model.matmul_split_kn import SplitKNSpec, estimate_matmul_split_kn
from perf.model.block_spec import MatmulBlockSpec
from perf.model.matmul_detailed import estimate_matmul_detailed
from perf.tiling.config import compute_tiling

params = params_from_testcase("perf_decode_bs1")
hw = ASCEND_950
tiling = compute_tiling(params, hw)

# MM1 baseline: M=T, K=He, N=Hcq
M, K, N = tiling.step_batch_size, params.He, params.Hcq
print(f"=== MM1 (C_q) Baseline geometry ===")
print(f"  M={M}, K=He={K}, N=Hcq={N}   dtype: BF16 (2B in, 2B out)")
print()

# ---- Baseline split-N (no split-KN) ----
# baseline uses mm1BlockNum=24 cores with singleN=Hcq/24=64
# Let's recompute via estimate_matmul_detailed
from perf.model.orchestration import get_default_block_specs
blocks = get_default_block_specs(params, hw)
mm1_block = blocks["MM1_Cq"]
print(f"Block spec: baseM={mm1_block.baseM} baseN={mm1_block.baseN} "
      f"baseK={mm1_block.baseK} stepK={mm1_block.stepK}")
print()

baseline_cores = tiling.mm1_block_num  # 24 on 950
baseline = estimate_matmul_detailed(
    "MM1_Cq_baseline", M, K, N,
    active_cores=baseline_cores, hw=hw, block=mm1_block, a_reused=False,
    a_shared_cores=baseline_cores,   # all 24 cores share A matrix
    fixpipe_dest="UB",
)

def print_mm(t, label):
    print(f"--- {label} ---")
    print(f"  shape: M={t.M} K={t.K} N={t.N}  active_cores={t.active_cores}")
    print(f"  kL1_loops={t.kL1_loops}  inner_bound={t.inner_bound}")
    print(f"  Outer MTE1 (HBM/L2 → L1):")
    print(f"    load_A_hbm = {t.load_A_hbm_us*1000:7.2f} ns   (a_source={t.a_source})")
    print(f"    load_A_l2  = {t.load_A_l2_us *1000:7.2f} ns")
    print(f"    load_B_hbm = {t.load_B_hbm_us*1000:7.2f} ns")
    print(f"    mte1_outer = {t.mte1_outer_us*1000:7.2f} ns  (effective per-kL1)")
    print(f"  Inner loop per kL1 (stepK iterations of L1→L0 + MMAD):")
    print(f"    l0a  = {t.l0a_us *1000:7.2f} ns")
    print(f"    l0b  = {t.l0b_us *1000:7.2f} ns")
    print(f"    mmad = {t.mmad_us*1000:7.2f} ns")
    print(f"    cube_us (kL1 × per_kl1) = {t.cube_us:6.3f} us")
    print(f"  FixPipe (L0C → dest):")
    print(f"    fixpipe = {t.fixpipe_us*1000:7.2f} ns")
    print(f"  => total = {t.total_us:.3f} us    bound={t.bound}")
    print()

print_mm(baseline, "BASELINE  split-N, 24 cores, singleN=64")

# ---- Split-KN N4×K7 ----
# n_groups=4 on N-axis, k_groups=7 on K-axis, total=28 cores
# single_n = N/4 = 384, single_k = K/7 = 1024
spec = SplitKNSpec("MM1_Cq", n_groups=4, k_groups=7,
                   single_n=N // 4, single_k=K // 7)
print(f"=== Split-KN N4×K7 config ===")
print(f"  n_groups={spec.n_groups}, k_groups={spec.k_groups}, total_cores={spec.total_cores}")
print(f"  per-core tile: M={M}  K_per_core={spec.single_k}  N_per_core={spec.single_n}")
print(f"  FixPipe dest: GM (k_groups > 1 → FP32 partial sums to GM for AIV accum)")
print(f"  A-matrix sharing: n_groups={spec.n_groups} cores within each k-group share A")
print(f"                    → a_source = MIX (1 cold HBM + 3 warm L2)")
print()

cube_t, accum_t = estimate_matmul_split_kn(
    "MM1_Cq", M, K, N, spec, hw, mm1_block,
    a_reused=False, vec_cores=hw.aiv_num, out_size=2,
)
print_mm(cube_t, "SPLIT-KN CUBE  per-core tile on 28 cores (FP32 partials to GM)")

print(f"--- Vector accumulation (AccumCq on AIV side) ---")
print(f"  Partials to accumulate per (M, N_full): k_groups=7 FP32 slices")
print(f"  Total bytes loaded: k_groups × M × (N_per_core × n_groups) × 4")
print(f"                    = 7 × {M} × {spec.single_n * spec.n_groups} × 4 = "
      f"{7*M*spec.single_n*spec.n_groups*4} B")
print(f"  mte2_load  = {accum_t.mte2_load_us*1000:7.2f} ns")
print(f"  vec_compute= {accum_t.vec_compute_us*1000:7.2f} ns")
print(f"  mte3_store = {accum_t.mte3_store_us*1000:7.2f} ns")
print(f"  => accum total = {accum_t.total_us:.3f} us   bound={accum_t.bound}")
print()

# ---- Summary delta ----
print(f"=== Delta vs baseline ===")
print(f"  Cube:    {baseline.total_us:6.3f} → {cube_t.total_us:6.3f} us  "
      f"({cube_t.total_us - baseline.total_us:+.3f})")
print(f"  + Accum overhead (AIV): {accum_t.total_us:.3f} us (hidden by next AIC matmul)")
print(f"  MM1 end-to-end AIC:     {cube_t.total_us:.3f} us  (vs baseline {baseline.total_us:.3f})")
