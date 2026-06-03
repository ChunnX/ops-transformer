# MixInverse Handoff

Date: 2026-05-11

Read this file first in a new window. For background evidence, read
`TRI_INVERSE_STAGE_SUMMARY.md`.

## Current State

`tri_inverse` is runnable and already split into two kernel headers:

```text
op_kernel/tri_inverse.h      # baseline TriInverseOp; do not optimize
op_kernel/tri_inverse_mix.h  # MixInverseOp; optimize here
op_kernel/tri_inverse.cpp    # compile-time include switch
```

Default build path:

```text
TRI_INVERSE_USE_MIX_INVERSE=1 -> include tri_inverse_mix.h
TRI_INVERSE_USE_MIX_INVERSE=0 -> include tri_inverse.h
```

`tri_inverse_mix.h` is currently Phase31a: P7 has a passing packed64 resident
diagonal-inverse boundary, and `zero64`/`identity64` are prepared once after
`InitRawMmadLocalBuffers()` for later reuse. P8 L0C accumulation has not been
implemented yet.

Last verified command sequence:

```bash
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tri_inverse -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet
cd experimental/attention/tri_inverse/tests/pytest
export ASCEND_RT_VISIBLE_DEVICES=1
bash test_run.sh single
```

Result:

```text
30 passed
```

## Hard Boundaries

For real MixInverse implementation, only edit:

```text
op_kernel/tri_inverse_mix.h
```

plus necessary docs.

Do not modify:

```text
op_host/
op_host/op_api/
op_kernel/tri_inverse_tiling_data.h
tests/pytest/tri_inverse_golden.py
tests/pytest/tri_inverse_operator_single.py
torch_extension/
Python call/interface semantics
```

`op_kernel/tri_inverse.h` is the baseline reference. Do not optimize it.

Do not use `PTO`, `Pto`, `MixInverse2`, `LX`, or `RX` naming in the new path.

## Math Contract

Input/output stay unchanged:

```text
cc_in  : BF16, ND, [tile_num, 64, 64]
cc_out : BF16, ND, [tile_num, 64, 64]
```

`cc_in` is positive strict-lower `A`, not raw `KKT` and not already-negated
`-A`.

Current baseline flow:

```text
cc_in -> AIV DataCopy/Cast -> Muls(-1.0)
      -> 32x32 top/bottom inverse -> cc_out
      -> AIC lower-left compose -> cc_out
```

Lower-triangular compose order is fixed:

```text
outLower = bottomInv @ lowerWork @ topInv
lowerWork = -LOW
```

## Optimization Rules

- Use the simplest viable implementation first. Precision-passing pure-cube
  correctness may come before removing redundant movement or compute.
- Solve the active problem directly. Do not replace the direct MixInverse path
  with a circuitous workaround and call that the main plan.
- Final MixInverse performance must be better than the reference/baseline
  performance, not just correctness-equivalent.
- Change one dataflow at a time.
- Run precision before moving to the next optimization.
- Do not use GM dump or tensor roundtrip debugging.
- Profile fields must keep CANN original names.
- Packed64 diagonal inverse means four independent `16x16` diagonal lanes in a
  `64x64` packed container. It is still four `16x16` inverses, not a full
  `64x64` triangular inverse; inner-lower/cross-lower blocks must stay in the
  existing lower-triangular compose flow.
- L0C accumulation and DataCopy/movement merge are allowed only as later
  optimization phases, after correctness phases and the first 6-shape profile.
- For L0C accumulation, it is acceptable to inspect
  `/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann`
  as a technique reference.
- Do not fully follow or port the Neumann inverse implementation. It has a
  working version, but its performance is poor; reuse only the specific L0C
  accumulation idea after MixInverse correctness is already stable.

Precision gate:

```text
shape=64 seed=0/21
shape=256 seed=0/21
shape=1024 seed=0/21
full 30 cases
```

Performance shapes:

```text
64, 65, 1024, 1025, 10240, 10241
```

## Progress Since Handoff

All validated baselines are saved under `op_kernel/baseline/`. The active
source is `op_kernel/tri_inverse_mix.h`; `op_kernel/tri_inverse.h` remains the
protected reference baseline.

Validation gate used after each phase:

```text
shape=64/256/1024 seed=0/21 PASS
bash test_run.sh single -> 30 passed
```

Saved PASS phases:

| phase | saved file | key change |
| --- | --- | --- |
| Phase2 | `op_kernel/baseline/tri_inverse_mix_phase2_diag16_cube_full30_pass.h` | four `16x16` diagonal inverse blocks moved to FP16 raw-MMAD |
| Phase3 | `op_kernel/baseline/tri_inverse_mix_phase3_compose16to32_full30_pass.h` | local `16 -> 32` lower-left compose moved to cube |
| Phase4 | `op_kernel/baseline/tri_inverse_mix_phase4_compose32to64_full30_pass.h` | `32 -> 64` lower-left compose moved to cube |
| Phase5 | `op_kernel/baseline/tri_inverse_mix_phase5_drop_aiv_baseline_compute_full30_pass.h` | removed old AIV baseline inverse compute path |
| Phase6 | `op_kernel/baseline/tri_inverse_mix_phase6_compose32to64_packed32_full30_pass.h` | packed32 compose landing |
| Phase7 | `op_kernel/baseline/tri_inverse_mix_phase7_diag32_cube_full30_pass.h` | two direct `32x32` diagonal cube inverses |
| Phase8 | `op_kernel/baseline/tri_inverse_mix_phase8_skip_diag_topright_zero_full30_pass.h` | skipped redundant diagonal top-right zero writes |
| Phase9 | `op_kernel/baseline/tri_inverse_mix_phase9_coalesce_scratch_topright_zero_full30_pass.h` | coalesced scratch top-right zero writes |
| Phase10 | `op_kernel/baseline/tri_inverse_mix_phase10_relocate_scratch_skip_aic_zero_full30_pass.h` | relocated diagonal scratch and removed AIC zero-MMAD top-right write |
| Phase11 | `op_kernel/baseline/tri_inverse_mix_phase11_load_compose32_once_full30_pass.h` | loaded `32x32` compose lowerWork once; reduced `aic_mte2_time(us)` |
| Phase12 | `op_kernel/baseline/tri_inverse_mix_phase12_drop_top_diag_reload_full30_pass.h` | dropped top diagonal scratch reload after reordering AIV writes |
| Phase13 | `op_kernel/baseline/tri_inverse_mix_phase13_load_compose32_once_after_phase12_full30_pass.h` | reapplied the Phase11 `32x32` lowerWork load on top of Phase12 |
| Phase14 | `op_kernel/baseline/tri_inverse_mix_phase14_l1_result64_shadow_full30_pass.h` | assembled top-left, bottom-right, and lower-left `32x32` blocks into L1 `result64`; final GM writeback is verification-only |
| Phase15b | `op_kernel/baseline/tri_inverse_mix_phase15b_drop_final_result64_replay_full30_pass.h` | removed only final `result64 @ I32` verification replay; diagonal/compose GM writebacks remain |
| Phase16 | `op_kernel/baseline/tri_inverse_mix_phase16_identity32_once_per_tile_full30_pass.h` | generated and loaded `identity32` once per tile, reused by both top/bottom diagonal32 solves |
| Phase17 | `op_kernel/baseline/tri_inverse_mix_phase17_diag32_single_replay_full30_pass.h` | removed the second final `INV32 @ I32` replay in diagonal32 solve; one L0C result now feeds L1 result64 and BF16 GM |
| Phase18 | `op_kernel/baseline/tri_inverse_mix_phase18_compose32_single_replay_full30_pass.h` | removed the second final compose32 replay; one L0C result now feeds L1 result64 and BF16 GM |
| Phase19 | `op_kernel/baseline/tri_inverse_mix_phase19_bottom_diag_subblock1_full30_pass.h` | subBlock1 writes bottom diagonal power scratch directly; subBlock0 no longer reloads/casts the bottom half |
| Phase22 | `op_kernel/baseline/tri_inverse_mix_phase22_identity32_batch_reuse_full30_pass.h` | AIC loads `identity32` once per `ProcessParaTiles` batch and reuses it for every tile in that batch |
| Phase23 | `op_kernel/baseline/tri_inverse_mix_phase23_identity32_aiv_batch_reuse_full30_pass.h` | AIV writes `identity32` scratch only for the first tile in each batch; AIC reuses that L1 identity for the batch |
| Phase27 | `op_kernel/baseline/tri_inverse_mix_phase27_identity32_core_lifetime_full30_pass.h` | AIV/AIC prepare `identity32` only for the first batch and hoist raw-MMAD init/restore/reset outside the batch loop |
| Phase29a | `op_kernel/baseline/tri_inverse_mix_phase29a_identity32_first_batch_l1_reuse_full30_pass.h` | isolates first-batch-only `identity32` L1 reuse while keeping Phase23 per-batch init/restore/reset behavior |
| Phase30 | `op_kernel/baseline/tri_inverse_mix_phase30_result64_topright_zero_only_full30_pass.h` | zeroes only the final top-right `32x32` of L1 `result64` instead of the full `64x64` result |
| Phase31 | `op_kernel/baseline/tri_inverse_mix_phase31_p7_4x16_packed64_diag_full30_pass.h` | P7 writes four independent `16x16` diagonal inverses into packed64 L0C lanes; `cInv=aicL0C_` semantically holds `D0/D1/D2/D3` at the P7 boundary |
| Phase31a | `op_kernel/baseline/tri_inverse_mix_phase31a_zero64_identity64_once_full30_pass.h` | prepares L1 resident `zero64` and packed64 `identity64` once after raw-MMAD local buffer init; no P8 compose accumulation yet |

Current active hash:

```text
sha256(op_kernel/tri_inverse_mix.h) = 866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf
sha256(op_kernel/baseline/tri_inverse_mix_phase31a_zero64_identity64_once_full30_pass.h) = 866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf
```

Phase31 notes:

- P7 uses four `16x16` solves as the compute unit, then writes each final
  `INV @ I` result into packed64 diagonal L0C offsets `0/1280/2560/3840`.
- Off-diagonal packed64 L0C lanes are cleared with zero MMAD and immediately
  Fixpipe-consumed into L1 `result64` shadow.
- Each nonzero diagonal lane is also immediately Fixpipe-consumed into L1
  `result64`. This was required because direct nonzero L0C offset writes
  without a Fixpipe consumer caused very slow/hanging behavior.
- The later Phase31 compatibility output path still uses the older diagonal32
  and compose32 flow, so Phase32 must start from the P7 resident boundary and
  replace the first-layer compose path. Do not treat Phase31 as already having
  P8 `cTmp + cInv` accumulation.
- Failed directions to avoid: full64 `result64 @ identity64` materialization,
  full64 zero-MMAD initialization, and repeated full64 MMAD for every inverse
  update all hung or timed out.

Phase31a adds only the resident constant lifecycle needed by later P8 work:
`InitRawMmadLocalBuffers()` allocates `zero64` and `identity64` after
`result64`, then `PreparePacked64Constants()` zeroes both full packed64 L1
buffers and copies four static `I16` NZ fractals into packed64 diagonal lanes
using offsets `0/1280/2560/3840`. This uses the Neumann packed64 copy parameter
shape, but it does not start P8 first-layer or second-layer accumulation.

Phase31a validation:

```text
source/build/package/installed tri_inverse_mix.h hash = 866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf
shape=64/256/1024 seed=0/21 PASS
bash test_run.sh single -> 30 passed in 6.05s
```

### Phase31 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase31_p7_4x16_packed64_diag_full30_6shape/tri_inverse_mixinverse_phase31_p7_4x16_packed64_diag_full30_6shape.csv
```

Final profiled hashes were all aligned to
`c9c61652e99d717b260873c3e8655a4a60bac69ea222328c8ec6a527c80c6c5c`
for source, packaged build header, and installed header.

| shape | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 25.560 | 0.168 | 0.022 | 0.176 | 0.115 | 30.952 |
| 65 | 28.020 | 0.335 | 0.027 | 0.330 | 0.236 | 33.405 |
| 1024 | 28.661 | 2.681 | 0.097 | 2.173 | 1.839 | 79.522 |
| 1025 | 28.980 | 2.849 | 0.102 | 2.285 | 1.947 | 81.882 |
| 10240 | 152.083 | 26.808 | 0.817 | 18.318 | 18.366 | 98.289 |
| 10241 | 167.203 | 26.973 | 0.822 | 18.411 | 18.467 | 90.012 |

Interpretation: Phase31 is intentionally not a performance win by itself. It
does P7 packed64 resident setup and then still runs the legacy diagonal32 plus
compose32 output path. Treat it only as the correctness boundary needed before
Phase32. The next performance-relevant step is to replace first-layer legacy
compose with resident `cTmp + cInv` accumulation.

## Performance Record

### v0 Reference

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_v0_6shape/tri_inverse_mixinverse_v0_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 11.82 | 0.004 | 0.257 | 0.081 | 0.000 |
| 65 | 2 | 16.12 | 0.009 | 0.497 | 0.121 | 0.001 |
| 1024 | 16 | 16.90 | 0.069 | 3.858 | 0.644 | 0.005 |
| 1025 | 17 | 16.98 | 0.073 | 4.098 | 0.676 | 0.005 |
| 10240 | 160 | 66.26 | 0.688 | 38.427 | 5.847 | 0.011 |
| 10241 | 161 | 66.04 | 0.693 | 38.667 | 5.891 | 0.011 |

Different-scheme total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase10 | 14.44 | 17.04 | 17.92 | 17.70 | 73.36 | 74.92 |
| phase11 | 14.96 | 17.72 | 17.74 | 17.88 | 73.74 | 77.38 |
| phase12 | 18.32 | 18.10 | 18.78 | 18.88 | 72.90 | 78.44 |
| phase13 | 16.24 | 17.78 | 18.44 | 18.60 | 71.52 | 76.50 |

### Phase12 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase12_drop_top_diag_reload_full30_6shape/tri_inverse_mixinverse_phase12_drop_top_diag_reload_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 18.32 | 0.061 | 0.025 | 0.164 | 0.008 |
| 65 | 2 | 18.10 | 0.122 | 0.033 | 0.287 | 0.014 |
| 1024 | 16 | 18.78 | 0.974 | 0.149 | 1.948 | 0.065 |
| 1025 | 17 | 18.88 | 1.034 | 0.157 | 2.114 | 0.068 |
| 10240 | 160 | 72.90 | 9.731 | 1.333 | 19.025 | 0.138 |
| 10241 | 161 | 78.44 | 9.798 | 1.342 | 19.155 | 0.139 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 72.900 | 70.430 | 9.731 | 13.291 | 7.113 | 19.025 | 11.808 | 28.251 | 1.333 | 15.895 | 6.909 | 11.004 | 96.612 |
| 10241 | 78.440 | 70.715 | 9.798 | 13.184 | 7.158 | 19.155 | 11.837 | 28.719 | 1.342 | 15.999 | 6.971 | 11.354 | 90.151 |

### Phase13 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase13_load_compose32_once_after_phase12_full30_6shape/tri_inverse_mixinverse_phase13_load_compose32_once_after_phase12_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 16.24 | 0.061 | 0.025 | 0.139 | 0.009 |
| 65 | 2 | 17.78 | 0.122 | 0.033 | 0.220 | 0.015 |
| 1024 | 16 | 18.44 | 0.974 | 0.149 | 1.586 | 0.068 |
| 1025 | 17 | 18.60 | 1.034 | 0.157 | 1.724 | 0.069 |
| 10240 | 160 | 71.52 | 9.734 | 1.333 | 15.691 | 0.141 |
| 10241 | 161 | 76.50 | 9.797 | 1.342 | 15.847 | 0.141 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 71.520 | 69.258 | 9.734 | 13.956 | 7.113 | 15.691 | 11.657 | 27.778 | 1.333 | 15.775 | 6.639 | 11.009 | 96.838 |
| 10241 | 76.500 | 69.404 | 9.797 | 13.955 | 7.158 | 15.847 | 11.677 | 28.019 | 1.342 | 15.821 | 6.671 | 11.194 | 90.724 |

Interpretation: Phase13 keeps the Phase12 top-diag reload removal and restores
the Phase11 MTE2 improvement. On shape 10240, `aic_mte2_time(us)` improves
from 19.025 to 15.691 and total `Duration(us)` improves from 72.90 to 71.52.
It is still slower than v0, so more scratch/movement and final-copy reduction
is required.

### Phase15b Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase15b_drop_final_result64_replay_full30_6shape/tri_inverse_mixinverse_phase15b_drop_final_result64_replay_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 14.801 | 0.067 | 0.025 | 0.143 | 0.009 |
| 65 | 2 | 17.900 | 0.135 | 0.033 | 0.273 | 0.016 |
| 1024 | 16 | 18.741 | 1.076 | 0.149 | 1.635 | 0.073 |
| 1025 | 17 | 18.460 | 1.144 | 0.157 | 1.792 | 0.075 |
| 10240 | 160 | 75.541 | 10.769 | 1.333 | 16.064 | 0.149 |
| 10241 | 161 | 76.041 | 10.836 | 1.342 | 16.080 | 0.151 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 75.541 | 72.105 | 10.769 | 14.753 | 7.836 | 16.064 | 12.232 | 27.786 | 1.333 | 15.887 | 6.568 | 11.120 | 95.452 |
| 10241 | 76.041 | 71.980 | 10.836 | 14.931 | 7.885 | 16.080 | 12.266 | 27.693 | 1.342 | 15.995 | 6.622 | 10.906 | 94.660 |

Different-scheme total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase10 | 14.44 | 17.04 | 17.92 | 17.70 | 73.36 | 74.92 |
| phase11 | 14.96 | 17.72 | 17.74 | 17.88 | 73.74 | 77.38 |
| phase12 | 18.32 | 18.10 | 18.78 | 18.88 | 72.90 | 78.44 |
| phase13 | 16.24 | 17.78 | 18.44 | 18.60 | 71.52 | 76.50 |
| phase15b | 14.80 | 17.90 | 18.74 | 18.46 | 75.54 | 76.04 |

Interpretation: Phase15b is a correctness-stable branch that removes the final
verification replay from L1 `result64`. It is not a performance win on all
large shapes, so continue optimization from the saved baseline instead of
treating it as the final performance direction.

### Phase16 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase16_identity32_once_per_tile_full30_6shape/tri_inverse_mixinverse_phase16_identity32_once_per_tile_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 14.620 | 0.067 | 0.024 | 0.124 | 0.010 |
| 65 | 2 | 17.200 | 0.135 | 0.032 | 0.237 | 0.017 |
| 1024 | 16 | 18.040 | 1.077 | 0.137 | 1.364 | 0.077 |
| 1025 | 17 | 17.941 | 1.144 | 0.144 | 1.409 | 0.080 |
| 10240 | 160 | 71.341 | 10.767 | 1.217 | 13.062 | 0.156 |
| 10241 | 161 | 72.601 | 10.837 | 1.224 | 13.174 | 0.157 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 71.341 | 68.813 | 10.767 | 14.338 | 7.826 | 13.062 | 12.157 | 23.896 | 1.217 | 11.813 | 6.622 | 9.641 | 96.457 |
| 10241 | 72.601 | 68.895 | 10.837 | 14.333 | 7.875 | 13.174 | 12.251 | 23.847 | 1.224 | 11.854 | 6.576 | 9.630 | 94.895 |

Different-scheme total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase13 | 16.24 | 17.78 | 18.44 | 18.60 | 71.52 | 76.50 |
| phase15b | 14.80 | 17.90 | 18.74 | 18.46 | 75.54 | 76.04 |
| phase16 | 14.62 | 17.20 | 18.04 | 17.94 | 71.34 | 72.60 |

Interpretation: Phase16 is a real single-dataflow improvement. On shape
10240 it reduces `aic_mte2_time(us)` from 16.064 to 13.062 versus Phase15b
and reduces AIV scalar/MTE3 work by avoiding the second per-tile identity
scratch generation.

### Phase17 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase17_diag32_single_replay_full30_6shape/tri_inverse_mixinverse_phase17_diag32_single_replay_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 14.600 | 0.063 | 0.024 | 0.131 | 0.009 |
| 65 | 2 | 17.341 | 0.126 | 0.032 | 0.204 | 0.015 |
| 1024 | 16 | 18.780 | 1.008 | 0.137 | 1.377 | 0.070 |
| 1025 | 17 | 18.460 | 1.071 | 0.144 | 1.420 | 0.072 |
| 10240 | 160 | 69.981 | 10.080 | 1.217 | 13.052 | 0.149 |
| 10241 | 161 | 75.702 | 10.144 | 1.224 | 13.198 | 0.149 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 69.981 | 67.849 | 10.080 | 14.479 | 7.351 | 13.052 | 12.114 | 24.151 | 1.217 | 12.233 | 7.125 | 9.380 | 96.954 |
| 10241 | 75.702 | 68.209 | 10.144 | 14.569 | 7.397 | 13.198 | 12.225 | 24.452 | 1.224 | 12.274 | 7.117 | 9.711 | 90.102 |

Different-scheme total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase13 | 16.24 | 17.78 | 18.44 | 18.60 | 71.52 | 76.50 |
| phase15b | 14.80 | 17.90 | 18.74 | 18.46 | 75.54 | 76.04 |
| phase16 | 14.62 | 17.20 | 18.04 | 17.94 | 71.34 | 72.60 |
| phase17 | 14.60 | 17.34 | 18.78 | 18.46 | 69.98 | 75.70 |

Interpretation: Phase17 is correctness-stable and removes one diagonal32
final replay. It improves shape 10240 total duration versus Phase16, but shape
10241 regresses, so the next phase should continue with the same narrow replay
class on compose and then re-profile before treating the branch as final.

### Phase18 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase18_compose32_single_replay_full30_6shape/tri_inverse_mixinverse_phase18_compose32_single_replay_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 13.060 | 0.061 | 0.024 | 0.129 | 0.008 |
| 65 | 2 | 17.121 | 0.122 | 0.032 | 0.197 | 0.015 |
| 1024 | 16 | 18.200 | 0.974 | 0.137 | 1.322 | 0.068 |
| 1025 | 17 | 18.120 | 1.035 | 0.144 | 1.430 | 0.070 |
| 10240 | 160 | 69.762 | 9.737 | 1.217 | 13.118 | 0.145 |
| 10241 | 161 | 70.222 | 9.800 | 1.224 | 13.184 | 0.146 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 69.762 | 67.276 | 9.737 | 14.592 | 7.113 | 13.118 | 12.111 | 24.202 | 1.217 | 12.124 | 6.850 | 9.645 | 96.436 |
| 10241 | 70.222 | 67.276 | 9.800 | 14.613 | 7.158 | 13.184 | 12.202 | 24.235 | 1.224 | 12.138 | 6.917 | 9.653 | 95.805 |

### Phase19 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase19_bottom_diag_subblock1_full30_6shape/tri_inverse_mixinverse_phase19_bottom_diag_subblock1_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 17.360 | 0.061 | 0.022 | 0.145 | 0.008 |
| 65 | 2 | 17.580 | 0.122 | 0.027 | 0.258 | 0.015 |
| 1024 | 16 | 17.720 | 0.974 | 0.097 | 1.376 | 0.070 |
| 1025 | 17 | 17.681 | 1.034 | 0.102 | 1.548 | 0.073 |
| 10240 | 160 | 67.642 | 9.734 | 0.817 | 13.033 | 0.149 |
| 10241 | 161 | 70.162 | 9.798 | 0.822 | 13.239 | 0.149 |

Typical kernel details:

| shape | Duration(us) | aicore_time(us) | aic_mac_time(us) | aic_scalar_time(us) | aic_mte1_time(us) | aic_mte2_time(us) | aic_fixpipe_time(us) | aiv_time(us) | aiv_vec_time(us) | aiv_scalar_time(us) | aiv_mte2_time(us) | aiv_mte3_time(us) | cube_utilization(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10240 | 67.642 | 65.465 | 9.734 | 14.098 | 7.113 | 13.033 | 12.191 | 23.626 | 0.817 | 11.909 | 6.265 | 9.745 | 96.782 |
| 10241 | 70.162 | 65.606 | 9.798 | 14.260 | 7.158 | 13.239 | 12.230 | 23.693 | 0.822 | 12.016 | 6.173 | 9.841 | 93.507 |

Different-scheme total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase16 | 14.62 | 17.20 | 18.04 | 17.94 | 71.34 | 72.60 |
| phase17 | 14.60 | 17.34 | 18.78 | 18.46 | 69.98 | 75.70 |
| phase18 | 13.06 | 17.12 | 18.20 | 18.12 | 69.76 | 70.22 |
| phase19 | 17.36 | 17.58 | 17.72 | 17.68 | 67.64 | 70.16 |
| phase23 | 13.98 | 15.64 | 16.56 | 16.66 | 62.40 | 68.16 |

Interpretation: Phase19 is correctness-stable and improves large aligned
shape 10240 by moving bottom diagonal power scratch production to subBlock1.
It reduces `aiv_vec_time(us)` from 1.217 to 0.817 and `aiv_mte2_time(us)` from
6.850 to 6.265 versus Phase18 on shape 10240. It still does not beat v0 on
all required shapes, especially 10241.

### Phase23 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase23_identity32_aiv_batch_reuse_full30_6shape/tri_inverse_mixinverse_phase23_identity32_aiv_batch_reuse_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 13.98 | 0.061 | 0.022 | 0.125 | 0.010 |
| 65 | 2 | 15.64 | 0.122 | 0.027 | 0.217 | 0.017 |
| 1024 | 16 | 16.561 | 0.974 | 0.097 | 1.324 | 0.075 |
| 1025 | 17 | 16.661 | 1.034 | 0.102 | 1.405 | 0.077 |
| 10240 | 160 | 62.401 | 9.734 | 0.817 | 10.788 | 0.162 |
| 10241 | 161 | 68.162 | 9.798 | 0.822 | 10.835 | 0.162 |

Different-scheme total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase18 | 13.06 | 17.12 | 18.20 | 18.12 | 69.76 | 70.22 |
| phase19 | 17.36 | 17.58 | 17.72 | 17.68 | 67.64 | 70.16 |
| phase23 | 13.98 | 15.64 | 16.56 | 16.66 | 62.40 | 68.16 |

Interpretation: Phase23 is the first current branch that beats v0 on the
large aligned shape 10240 and on 65/1024/1025. It reduces AIC-side
`aic_mte2_time(us)` from Phase19 13.033 to 10.788 on shape 10240 by loading
`identity32` once per batch, and keeps AIV identity generation to the batch
head. It still misses the final goal on shape 64 and 10241, so the next phase
should target fixed per-call overhead and odd-tail behavior without merging
L0C-resident accumulation into the same change.

### Phase23 vs Phase27 Rerun

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase23_rerun_20260512_6shape/tri_inverse_mixinverse_phase23_rerun_20260512_6shape.csv
tests/pytest/prof_out/tri_inverse_mixinverse_phase27_rerun_20260512_6shape/tri_inverse_mixinverse_phase27_rerun_20260512_6shape.csv
```

Same-run total duration and MTE2 comparison:

| shape | Phase23 Duration | Phase27 Duration | Duration delta 27-23 | Phase23 aic_mte2 | Phase27 aic_mte2 | MTE2 delta 27-23 | winner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 16.200 | 16.421 | +0.221 | 0.122 | 0.129 | +0.007 | Phase23 |
| 65 | 16.460 | 17.061 | +0.601 | 0.207 | 0.224 | +0.017 | Phase23 |
| 1024 | 16.540 | 17.740 | +1.200 | 1.303 | 1.371 | +0.068 | Phase23 |
| 1025 | 16.380 | 18.080 | +1.700 | 1.393 | 1.454 | +0.061 | Phase23 |
| 10240 | 62.541 | 62.862 | +0.321 | 10.749 | 10.414 | -0.335 | Phase23 |
| 10241 | 68.202 | 68.481 | +0.279 | 10.794 | 10.487 | -0.307 | Phase23 |

Interpretation: under the rerun profile, Phase23 has lower total
`Duration(us)` on all six shapes. Phase27 reduces `aic_mte2_time(us)` on the
large shapes, but that does not translate into total time because Phase27 is
not a pure identity-load change: it also moves raw-MMAD
`InitRawMmadLocalBuffers()/RestoreRawMmadState()/pipe_->Reset()` outside the
batch loop. Therefore Phase27 should be treated as a mixed-attribution branch,
not proof that first-batch-only identity reuse is slower.

### Phase29a Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase29a_identity32_first_batch_l1_reuse_full30_6shape/tri_inverse_mixinverse_phase29a_identity32_first_batch_l1_reuse_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 15.640 | 0.061 | 0.022 | 0.122 | 0.010 |
| 65 | 2 | 16.260 | 0.122 | 0.027 | 0.207 | 0.017 |
| 1024 | 16 | 16.660 | 0.974 | 0.097 | 1.329 | 0.075 |
| 1025 | 17 | 17.480 | 1.034 | 0.102 | 1.407 | 0.078 |
| 10240 | 160 | 61.821 | 9.735 | 0.817 | 10.420 | 0.163 |
| 10241 | 161 | 67.641 | 9.796 | 0.822 | 10.443 | 0.163 |

Same-run total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.82 | 16.12 | 16.90 | 16.98 | 66.26 | 66.04 |
| phase23 | 13.98 | 15.64 | 16.56 | 16.66 | 62.40 | 68.16 |

| phase23_rerun | 16.200 | 16.460 | 16.540 | 16.380 | 62.541 | 68.202 |
| phase27_rerun | 16.421 | 17.061 | 17.740 | 18.080 | 62.862 | 68.481 |
| phase29a | 15.640 | 16.260 | 16.660 | 17.480 | 61.821 | 67.641 |

Phase29a deltas:

| shape | 29a-23 Duration | 29a-27 Duration | 29a-23 MTE2 | 29a-27 MTE2 | duration winner | MTE2 winner |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 64 | -0.560 | -0.781 | +0.000 | -0.007 | Phase29a | Phase23 |
| 65 | -0.200 | -0.801 | +0.000 | -0.017 | Phase29a | Phase23 |
| 1024 | +0.120 | -1.080 | +0.026 | -0.042 | Phase23 | Phase23 |
| 1025 | +1.100 | -0.600 | +0.014 | -0.047 | Phase23 | Phase23 |
| 10240 | -0.720 | -1.041 | -0.329 | +0.006 | Phase29a | Phase27 |
| 10241 | -0.561 | -0.840 | -0.351 | -0.044 | Phase29a | Phase29a |

Interpretation: Phase29a validates the user's expectation for the identity-only
path on most target shapes: first-batch-only `identity32` reuse improves total
duration over Phase23 on 64, 65, 10240, and 10241, and beats Phase27 on all six
shapes. The regression is localized to mid shapes 1024 and 1025, where Phase23
is still faster. For the next optimization, keep Phase29a as the active source
because it is the best large-shape and odd-tail branch, but preserve Phase23 as
the comparison point for 1024/1025 before accepting further movement changes.

### Phase30 Profile

CSV:

```text
tests/pytest/prof_out/tri_inverse_mixinverse_phase30_result64_topright_zero_only_full30_6shape/tri_inverse_mixinverse_phase30_result64_topright_zero_only_full30_6shape.csv
```

| shape | tile_num | Duration(us) | aic_mac_time(us) | aiv_vec_time(us) | aic_mte2_time(us) | aic_mac_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 12.840 | 0.061 | 0.022 | 0.144 | 0.008 |
| 65 | 2 | 16.741 | 0.122 | 0.027 | 0.209 | 0.015 |
| 1024 | 16 | 17.360 | 0.973 | 0.097 | 1.339 | 0.071 |
| 1025 | 17 | 17.541 | 1.034 | 0.102 | 1.441 | 0.074 |
| 10240 | 160 | 62.621 | 9.736 | 0.817 | 10.380 | 0.160 |
| 10241 | 161 | 64.861 | 9.798 | 0.822 | 10.436 | 0.161 |

Same-run total duration comparison, `Duration(us)`:

| scheme | 64 | 65 | 1024 | 1025 | 10240 | 10241 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v0 | 11.820 | 16.120 | 16.900 | 16.980 | 66.260 | 66.040 |
| phase29a | 15.640 | 16.260 | 16.660 | 17.480 | 61.821 | 67.641 |
| phase30 | 12.840 | 16.741 | 17.360 | 17.541 | 62.621 | 64.861 |

Phase30 deltas:

| shape | 30-29a Duration | 30-v0 Duration | 30-29a MTE2 | winner vs 29a |
| --- | ---: | ---: | ---: | --- |
| 64 | -2.800 | +1.020 | +0.022 | Phase30 |
| 65 | +0.481 | +0.621 | +0.002 | Phase29a |
| 1024 | +0.700 | +0.460 | +0.010 | Phase29a |
| 1025 | +0.061 | +0.561 | +0.034 | Phase29a |
| 10240 | +0.800 | -3.639 | -0.040 | Phase29a |
| 10241 | -2.780 | -1.179 | -0.007 | Phase30 |

Interpretation: Phase30 passes precision and is a useful branch, but it is not
a uniform win. It improves 64 and 10241 and makes 10241 faster than v0 for the
first time. It regresses 65, 1024, 1025, and 10240. The large aligned 10240
regression comes from higher scalar/AIV side time despite a small MTE2
improvement, so do not stack broad movement changes on top of Phase30 without
checking whether the top-right-only zero should be conditional or simplified.

## Current Active State

Active source hash:

```text
sha256(op_kernel/tri_inverse_mix.h) = 866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf
```

Phase31a is the current intended active source. It preserves the P7 resident
boundary first: `cInv=aicL0C_` semantically holds packed64 diagonal inverse
lanes `D0/D1/D2/D3` after P7. It additionally prepares resident `zero64` and
packed64 `identity64` once per AIC local-buffer lifecycle for later reuse. Its
downstream compatibility path still uses the older diagonal32 plus compose32
output flow, so P8 first-layer `cTmp + cInv` accumulation is the next
unimplemented phase.

Important correction from 2026-05-12: do not discard a phase only because total
`Duration(us)` is worse if it improves a concrete profile field such as
`aic_mte2_time(us)`. Save and document that phase as a usable branch point.

Identity status: Phase29a reduces `identity32` AIV write and AIC L1 load to
the first batch only, then relies on L1 persistence. Phase31a also adds active
packed64 `identity64` preparation from `TRI_INV_IDENTITY16_NZ4` after
`InitRawMmadLocalBuffers()`. Do not repeat the failed direct AIC `SetValue`
identity attempt as-is; it produced max_abs=1.0-class failures.

Failed Phase14 attempt, 2026-05-12 09:28 CST: replacing the final diagonal
`INV @ I` replay with one helper that Fixpipe-writes the same L0C result first
to L1 and then directly to GM caused the first small-gate case
`shape=64 seed=0` to hang. The experiment was fully reverted, and the active
hash is back to Phase13. Do not retry that exact consecutive L1+GM Fixpipe
chain without a different synchronization design.

Failed Phase14 identity32 cache attempt, 2026-05-12: replacing per-tile AIV
`WriteDiag32IdentityScratch` plus GM reload with an AIC L1-resident
`PrepareIdentity32()` passed the small gate (`64/256/1024 seed=0/21`) but
failed full 30. Failures included aligned large shapes (`4096`, `6144`,
`8192`, `10240`) and odd shapes (`65`, `257 seed=0`, `4097`, `10241`) with
`max_abs=1.0`-class errors. The experiment was reverted to Phase13 and must
not be treated as a saved baseline.

Failed Phase14 static identity attempt, 2026-05-12: replacing the same AIV
identity scratch with per-solve AIC loads from `TRI_INV_IDENTITY16_NZ` also
passed the small gate but failed full 30 with the same `max_abs=1.0`-class
pattern. A same-process spot rerun showed case-order sensitivity (`65 seed=0`
passed when run before larger shapes but failed in the full sequence), so this
identity dataflow is not stable enough to save. Reverted to Phase13.

Phase14 L1 `result64` baseline, 2026-05-12: after restoring the stable
identity scratch path, `result64` assembly passed the small gate and full 30
cases. Save file:
`op_kernel/baseline/tri_inverse_mix_phase14_l1_result64_shadow_full30_pass.h`,
sha256 `2e4a5768a9962362f68d1499dc9d396d199c9b4f3006abde1bd00c81d28b1e89`.
It was revalidated on the healthy physical chip1 by running with
`ASCEND_RT_VISIBLE_DEVICES=1`; the small gate passed and `bash test_run.sh
single` reported `30 passed in 6.04s`.

Invalid Phase15 attempt, 2026-05-12: deleting the diagonal/compose shadow
replay and intermediate BF16 GM writes initially passed the small gate and
full 30, but profiling then ran while another program was profiling/using the
device. The run entered NPU Alarm:
`80C98001 node type=AIC, sensor type=RAS State, event state=module error can
not be fixed`. After that, even restored Phase13/Phase14 large-shape checks
failed or raised AICore exceptions, so the Phase15 result and partial profile
are invalid and must not be used as a saved baseline or performance record.
The temporary Phase15 baseline file was removed. Clear/reset the NPU and
revalidate Phase14 before continuing.

Phase15a retry, 2026-05-12: removing only the intermediate BF16 GM writes
while keeping replay and syncs was tested on the healthy chip1 and hung at
`shape=64 seed=0`. The active source was restored to Phase14. Do not retry
that exact dataflow.

## Next Work Queue

1. Continue from Phase31a on the healthy device only. Current local workaround:
   set `ASCEND_RT_VISIBLE_DEVICES=1`, because physical chip0 still reports
   Alarm while chip1 is OK and the wrapper maps logical device 0.
2. Target one dataflow at a time. Phase32 is explicitly the first-layer P8
   L0C accumulation based on the P7 resident `cInv`; do not skip ahead to
   second-layer accumulation or movement/DataCopy merge.
3. Phase32 formula is `tmp = lowerNeg @ topInv`, then
   `out = bottomInv @ tmp`. `lowerNeg` is already `-L`, so do not add another
   negative sign.
4. Phase33 starts only after Phase32 passes: second-layer L0C accumulation
   into the same resident `cInv`.

2026-05-13 concurrency note: a Phase32 no-Fixpipe isolation candidate was
attempted, but no result is valid. Another Codex session in the same worktree
kept launching `chunk_gated_delta_rule*` build/profiling and repeatedly wrote
`op_kernel/tri_inverse_mix.h` back to an old candidate containing the illegal
`InvPlusPowerScratch` path. The shared `build/` directory and installed OPP
were also overwritten by those non-`tri_inverse` packages. The active source
has been restored to Phase31a:

```text
sha256(op_kernel/tri_inverse_mix.h) = 866a0f6c0d1d3ec8e68f274ae524ddac678ff9f5f740db36d19e0c40b293dabf
```

Before retrying Phase32, ensure there is a single writer and verify
source/package/installed header plus object hashes. If `InvPlusPowerScratch`,
`DataCopyDiag16InvPlus`, `LoadPacked64DiagInvPlus`, or
`SolvePacked64DiagonalToResident` appears in `op_kernel/tri_inverse_mix.h`,
restore Phase31a before continuing.

Additional optimization queue from 2026-05-12:

- `16x16` inverse still has redundancy. The active path solves two `32x32`
  diagonal blocks (`top32` and `bottom32`), not four `16x16` blocks packed into
  one `64x64` inverse. `HalfL1InvAddBlock32()` uses local L0C accumulation
  (`inv @ I` then `inv @ power`), but it is not full L0C residency; intermediate
  `power` and `inv` still Fixpipe back to L1.
- KDA reference boundary, 2026-05-12:
  `/mnt/workspace/gitCode/refresh/KDA/kernel_tri_inv_rec_unroll_annotated.cpp`
  is useful for diagonal `16x16` fractal packing only. For `N=64,F=16`, the
  diagonal L0 offsets are `0/1280/2560/3840` (`i * F * (N + F)`). Do not copy
  the upper-triangular MBH/LX/RX compose formula; this operator still uses
  `outLower = bottomInv @ lowerWork @ topInv`.
- Failed Phase31 full64 update attempt, 2026-05-12: a single packed64
  diagonal-lane smoke passed `shape=64 seed=0` when it only did fixed L1->L0
  lane loads, one `64x64` MMAD, and Fixpipe to L1 without feeding the result
  to output. Replacing each inverse update round with repeated full `64x64`
  packed MMAD then hung/timed out at `shape=64 seed=0` in two implementations
  and was reverted to Phase30. Do not continue by making every 16x16 update a
  full64 MMAD. The next direct path should keep `16x16` as the compute unit and
  optimize 4-lane fixed搬运/identity lifecycle first, or separately do L0C
  residency after a passing correctness baseline.
- Phase31 order correction, 2026-05-12: do not start with P8/compose L0C
  accumulation. First implement P7 `4x16 -> packed64` diagonal inverse so P7
  ends with `cInv=aicL0C_` semantically holding the four diagonal inverse lanes
  `D0/D1/D2/D3`. Only after that PASS should P8 use the `cTmp + cInv`
  accumulation scheme. For later P8, the scratch lower block is already
  `lowerNeg=-L`, so do not add another negative sign.
- DataCopy still has roundtrips: AIV BF16->FP16 goes through GM scratch before
  AIC L1, compose32 lowerWork still goes AIV->GM->AIC, and standalone output
  keeps final BF16 GM writes even though L1 `result64` is the useful fused
  state.
- Identity is now once per batch, not globally once per operator. A full
  operator-lifetime identity path remains open, but previous direct AIC/static
  identity attempts are known unstable and should not be repeated unchanged.
- L1 vs GM split remains intentional for now: L1 is the internal result for
  compose/RHS, GM is required by standalone pytest/golden and by some AIV->AIC
  scratch channels. Future fusion can delete more GM writes after correctness
  is verified through the fused consumer.
   - Inspect remaining GM roundtrips, especially compose32 lowerWork
     AIV->GM->AIC and standalone-only diagonal/compose GM writes.
   - Later only: L0C accumulation for `inv @ I + inv @ power` style updates.
5. After each change run:

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tri_inverse -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet
cd experimental/attention/tri_inverse/tests/pytest
source ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash >/dev/null 2>&1 || true
export ASCEND_RT_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
python3 - <<'PY'
from tri_inverse_operator_single import run_precision_test
for shape in (64, 256, 1024):
    for seed in (0, 21):
        print('RUN', shape, seed, flush=True)
        run_precision_test(shape, seed)
PY
bash test_run.sh single
```

6. Save every full-pass phase under `op_kernel/baseline/` before profiling or
   starting another phase.

Stop and redesign inside `tri_inverse_mix.h` if a step appears to require host,
tiling, golden, wrapper, or interface changes.
