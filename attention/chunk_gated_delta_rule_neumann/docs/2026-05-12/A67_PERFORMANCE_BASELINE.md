# A67 Performance Baseline

## Source Boundary

- Phase: A67 `identity16_cross_para_keep`.
- Saved baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA67_identity16_cross_para_keep_baseline.h`
- Baseline SHA256:
  - `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`
- Package:
  - `/tmp/packed64_phaseA67_identity16_cross_para_keep_20260512_190504.run`
- Package SHA256:
  - `7d7f48185a28b9038506e7087e262a49d9b0a39f69de1f3cddf05997fd0e4560`
- Isolated OPP:
  - `/tmp/packed64_phaseA67_identity16_cross_para_keep_20260512_190504_opp/vendors/custom_transformer`
- Installed Neumann object SHA256:
  - `edcae896cbcaf76bda7ce5730eac6c58fad9c7619c57e98c4b5d50528a3b87bb`

## Precision Gate

- build PASS.
- isolated install PASS.
- `S=64/256/1024` smoke PASS.
- full compare PASS: `16 passed in 6.18s`.

## Performance Artifacts

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA67_identity16_cross_para_keep_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA67_identity16_cross_para_keep_20260512/performance_compare.csv`
- Compare CSV SHA256:
  - `8de471eab5f6c91e5529f5628c0d61ff46c6e23086a05d16bdd3a02a21e7d08e`
- Raw CSV:
  - `tests/pytest/prof_out/packed64_phaseA67_identity16_cross_para_keep_20260512/performance_raw.csv`
- Raw CSV SHA256:
  - `70ca0b4d650bd8136f8017d38f4043dd9c614cdc98226299da137fb1c5237262`

## Ratios

| Case | neumann/normal |
| --- | ---: |
| `S=64` | `1.181` |
| `S=512` | `1.635` |
| `S=2048` | `2.425` |
| `S=10240` | `2.456` |

## Conclusion

- A67 is the latest correctness-safe baseline and current all-shape preferred performance baseline.
- Compared with A66 `1.244 / 1.778 / 2.528 / 2.561`, A67 improves all four profiled sizes.
- Later changes must rerun build/install, `S=64/256/1024`, full compare, and performance before attributing any new result to the changed source.
