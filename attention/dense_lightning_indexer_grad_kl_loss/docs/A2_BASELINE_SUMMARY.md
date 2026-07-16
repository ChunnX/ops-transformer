# DenseLightningIndexerGradKLLoss A2 Baseline Summary

## 1. Conclusion

Stage 5 passed on Ascend 910B3. All 64 functional cases and all 60 planned
stability repetitions passed the agreed element-wise precision rule. No kernel,
host tiling, or build source was changed from the `master` baseline in stages
1-5; the committed changes are validation code and this report only.

This is the A2 gate for starting A5 migration. No A5 implementation change was
made before this gate passed.

## 2. Baseline

| Item | Value |
|---|---|
| Date | 2026-07-16 |
| Base `master` | `c107eaeb87a9649a8765aa1e6dbd993ae2e4f817` |
| Validated branch head | `67eb71d9e1e57c84b18dd6ca62e992475749d08a` |
| Branch | `feature/dense-li-grad-kl-loss-a5-migration` |
| SoC | Ascend 910B3 |
| Device resources | 20 AIC, 40 AIV, 60.96 GiB |
| CANN | 9.1.0 beta V2 |
| PyTorch | `2.12.0+cu130` |
| torch_npu | `2.12.0.rc1` |
| Remote repository | `/opt/qsc/WorkSpace/DSA_Loss/20260716-feature-dense-li-grad-kl-loss-a5-migration/repo` |
| Artifact root | `/opt/qsc/WorkSpace/DSA_Loss/20260716-feature-dense-li-grad-kl-loss-a5-migration` |

The installed package was built from validation commit `9befdc084`; commits
after it changed test scripts only. The A2 operator implementation remained
bitwise identical to base `master`.

## 3. Build Gate

| Check | Result |
|---|---|
| Package build for `ascend910b` | PASS |
| Package install | PASS |
| Host UT | PASS, 4/4 |
| BSND infer shape/tiling | PASS |
| TND infer shape/tiling | PASS |

Logs: `build.log`, `install.log`, `ophost_ut.log`, and `stage4_status.txt`.

## 4. Functional Results

| Suite | Passed | Total |
|---|---:|---:|
| Smoke | 4 | 4 |
| Typical | 12 | 12 |
| Full | 48 | 48 |
| Total | 64 | 64 |

Coverage includes BSND/TND, FP16/BF16, same-dtype/FP32 weight, rope on/off,
batch 1-16, S1 1024-4096, S2 4096-65536, N1 32/64/128, N1Index
8/16/32/64, partial zero-length TND batches, and repeated cumulative lengths.

Precision rule for every valid element:

- `abs(golden) < 1`: absolute error `<= 1e-3`.
- `abs(golden) >= 1`: relative error `<= 1e-3`.

Worst observed values:

| Output | Max absolute error | Case | Max comparison metric | Case |
|---|---:|---|---:|---|
| `d_query_index` | `6.103515625e-05` | FULL-PW-21 | `6.103515625e-05` | FULL-PW-21 |
| `d_key_index` | `9.765625e-04` | FULL-PW-21 | `9.765625e-04` | FULL-PW-21 |
| `d_weight` | `1.220703125e-04` | FULL-PW-04 | `1.220703125e-04` | FULL-PW-04 |
| `loss` | `5.9814453125e-03` | FULL-PW-21 | `1.6730105926e-05` | FULL-PW-08 |

All outputs were finite. No functional case had a runtime error, hang, or
unexpected NaN/Inf in its accepted run.

## 5. Runtime And Memory

The recorded time is one synchronized operator invocation from `run_case.py`;
it is a functional baseline, not a performance acceptance result.

| Metric | Value |
|---|---:|
| Minimum elapsed time | 311.477 ms |
| Median elapsed time | 552.088 ms |
| Mean elapsed time | 893.997 ms |
| Maximum elapsed time | 11848.948 ms, BND-02 |
| Maximum NPU allocated memory | 42.301 GiB, BND-02 |
| Maximum NPU reserved memory | 42.322 GiB, BND-02 |
| Maximum tiling workspace | 5.599 GiB, BND-02 |

The workspace value is recalculated from the A2 `GetWorkspaceSize()` formula
with 20 AIC and `deterministic=false`. Peak NPU memory also contains input,
output, framework, and allocator memory, so it is intentionally larger.

BND-02 is the maximum shape:
`B=16, S1=4096, S2=65536, N1=128, N1Index=64, D=DIndex=128`.

## 6. Stability

The same manifest and input binaries were reused for each repetition.

| Case | Result | Mean ms | Max `d_key_index` variation | Max `loss` variation |
|---|---|---:|---:|---:|
| SMK-01 | 10/10 PASS | 358.116 | `1.1920928955e-07` | `9.5367431641e-07` |
| SMK-02 | 10/10 PASS | 324.919 | `1.1920928955e-07` | `7.1525573730e-07` |
| TYP-03 | 10/10 PASS | 1596.990 | `1.9073486328e-06` | `9.1552734375e-05` |
| TYP-08 | 10/10 PASS | 485.226 | `4.7683715820e-07` | `1.1444091797e-05` |
| BND-02 | 10/10 PASS | 11849.718 | `2.3841857910e-07` | `4.8828125e-04` |
| BND-04 | 10/10 PASS | 561.543 | `3.8146972656e-06` | `3.8146972656e-06` |

`d_query_index` and `d_weight` were bitwise equal in all 60 accepted runs.
`d_key_index` and `loss` showed small non-bitwise variation, as expected for
the current `deterministic=false` path, while every run passed precision.

An earlier BND-02 attempt was interrupted by unrelated external workloads that
occupied about 44 GiB on the selected device. That attempt is retained under
`stability/` and `stability_retry/` as an environment event. The accepted,
uninterrupted 10/10 result is under `stability_final/`.

## 7. Golden Corrections

`golden_a2_a5.py` is a copy-derived implementation. The original
`D:\Code\SYNC_FILE\DSA_LOSS\golden.py` was not modified.

Changes made in the validation copy:

- Removed the TensorFlow dependency for BF16 binary I/O.
- Fixed multi-batch TND generation and Tensor misuse.
- Supported repeated cumulative lengths for partial zero-length batches.
- Ensured random TND data satisfies every batch's `Q length <= K length`.
- Computed all four max/sum inputs from actual tensors.
- Preserved FP32 `d_weight` for FP32 weight inputs.
- Streamed large Golden calculations to control memory.
- Restored the original 512/256/64 query-slice accumulation order.
- Added `--reuse-existing` to recompute Golden without changing input files.
- Recorded seed, shapes, dtypes, cumulative lengths, and SHA256 checksums.
- Added per-output finite checks, element-wise comparison, memory recording,
  and 10-run stability reports.

During FULL-PW-04, a smaller temporary Golden chunk produced 62 BF16
`d_key_index` mismatches, with maximum absolute error `0.00341796875`.
Recomputing the unchanged input with the original accumulation order reduced
the maximum error to `0.00048828125` with zero failures. This confirmed a
validation-order issue, not an A2 operator defect.

## 8. Reproduction And Artifacts

Important files under the artifact root:

- `environment.txt`
- `build.log`, `install.log`, `ophost_ut.log`
- `a2_smoke.log`, `a2_typical_*.log`, `a2_full_*.log`
- `a2_stage5_aggregate.json`
- `a2_cases/<case-id>/case_manifest.json`
- `a2_cases/<case-id>/run_report.json`
- `a2_cases/<stability-case>/<stability-dir>/stability_report.json`

Artifact facts:

| Item | Value |
|---|---|
| Manifest count | 64 |
| Run report count | 64 |
| Case artifact size | 248651249325 bytes |
| Manifest set SHA256 | `3a34fa1b859655d3dc7e79aa65eb85427361b9dc2ccd21e747172e3c9bc5c936` |
| Aggregate SHA256 | `86ed2d527be44a69d08a4e5a9b1ad4d34590d407161452b36fedcd6e8b50fb3d` |

The accepted A2 manifests and binary inputs are the source for later A5
comparison. They must be reused rather than regenerated.

## 9. Stage 5 Gate

| Acceptance item | Status |
|---|---|
| Smoke, typical, and full suites completed | PASS |
| All four outputs meet precision for all cases | PASS |
| Stability subset completed 10 times each | PASS |
| No accepted-run runtime error, hang, or NaN/Inf | PASS |
| Maximum shape workspace and NPU memory recorded | PASS |

Stage 5 status: **PASS**.
