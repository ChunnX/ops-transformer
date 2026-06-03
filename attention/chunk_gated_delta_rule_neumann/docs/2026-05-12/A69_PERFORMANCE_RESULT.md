# A69 Performance Result

## Context

- Phase: A69 P8 second-layer `cInv/cTmp` accumulate.
- Active Stage1 SHA256:
  - `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_baseline.h`
- Package:
  - `/tmp/cgdr_a69_20260512_220105.run`
  - SHA256 `f85b7864e3c757df74d01f2eae656dd88068302c04dc59180989fafdeb28acee`
- Isolated OPP:
  - `/tmp/cgdr_a69_opp_20260512_220105/vendors/custom_transformer`

## Correctness Gate

- `S=64` PASS.
- `S=256` PASS.
- `S=1024` PASS.
- full compare PASS:
  - `16 passed in 6.22s`

## Performance Artifacts

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_20260512/performance_compare.csv`
  - SHA256 `484718014cde93891a68d633eb1f275afcd1d1a5323f346367437b4e7738ec56`
- Raw CSV:
  - `tests/pytest/prof_out/packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_20260512/performance_raw.csv`
  - SHA256 `787211d7a3245f4ff9d27c5c6cd5f1fbc308d43cbaaeca524e219a6a54de836b`

## Ratios

| Case | normal us | neumann us | neumann/normal |
|---|---:|---:|---:|
| `S=64,D=64` | `637.06` | `796.62` | `1.250` |
| `S=512,D=64` | `1669.18` | `2766.20` | `1.657` |
| `S=2048,D=128` | `5033.80` | `12272.70` | `2.438` |
| `S=10240,D=128` | `24244.98` | `59852.76` | `2.469` |

## Conclusion

- A69 is correctness-safe and captures the full first-layer + second-layer `cInv/cTmp` accumulate design.
- A69 is not a performance-preferred baseline:
  - A68 ratios were `1.175 / 1.662 / 2.399 / 2.437`.
  - A67 ratios were `1.181 / 1.635 / 2.425 / 2.456`.
- Keep A69 as the latest correctness surface; use A68/A67 when comparing performance-oriented follow-up phases.
