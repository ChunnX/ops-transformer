# A5 SparseFlashAttention Split-KV validation

## Scope and versions

This PR contains only the Ascend 950 (A5) `sparse_flash_attention` path:
matching FP16 or BF16 inputs/output, TND/PA_BSND, one KV head, G=8,
sparse mode 0/3, TopK 256–2048,
without sinks or returned LSE. Unsupported shapes retain the original path.
It does not change arch22, C8, indexer, or model integration code.

The performance measurements below were collected on 2026-09-30 (single
operator) and 2026-10-06 (AISBench) before the A5/A2 PR split. They are reused
A5 results, not a new performance comparison against current master.
The new branch is based on `d5b874d33cbaafbb31af55e57e296c50933aca0f`;
the arch35 kernel changes are retained from PR !13112.

## Single-operator timing

Ascend950DT_9582, CANN 9.1, PyTorch 2.7.1 / torch_npu 2.7.1.post4.
G=8, BF16, TND/PA_BSND, physical KV length 4096, TopK 2048,
sparse mode 0, no returned LSE. `B` is batch size and `Q` is the number of
queries per batch; total query count is `B × Q`.
Baseline uses the same source with partitioning disabled. Each NPU graph
contains 16 SFA calls; two rounds of 51 event-timed samples are collected.
The table reports the minimum per-call latency (microseconds) across all
102 samples because the device was shared. Speedup is baseline / partitioned.

| B | Q | Baseline (us) | With partitioning (us) | Speedup |
| --- | --- | --- | --- | --- |
| 1 | 1 | 68.793 | 17.795 | 3.866x |
| 1 | 2 | 68.849 | 17.733 | 3.883x |
| 1 | 4 | 69.235 | 19.752 | 3.505x |
| 1 | 8 | 69.550 | 27.246 | 2.553x |
| 2 | 1 | 71.703 | 17.981 | 3.988x |
| 2 | 2 | 71.907 | 20.231 | 3.554x |
| 2 | 4 | 72.150 | 27.932 | 2.583x |
| 2 | 8 | 72.502 | 43.628 | 1.662x |
| 4 | 1 | 71.994 | 20.072 | 3.587x |
| 4 | 2 | 72.319 | 27.875 | 2.594x |
| 4 | 4 | 72.526 | 43.452 | 1.669x |
| 4 | 8 | 72.986 | 73.131 | 0.998x |
| 8 | 1 | 72.572 | 28.086 | 2.584x |
| 8 | 2 | 72.773 | 43.680 | 1.666x |
| 8 | 4 | 73.140 | 73.125 | 1.000x |
| 8 | 8 | 135.508 | 135.741 | 0.998x |

For B=1/Q=1 and B=2/Q=2, full-index outputs were compared against an
independent softmax reference. Relative L2 errors were 0.00190 and 0.00191.

## A5 64K AISBench end-to-end timing

GLM-5.2, A5 TP8 without DCP, DSpark7. C8/indexer-C8, Pivot and DSACP disabled.
`FULL_DECODE_ONLY`, `max_num_seqs=1`, `max_model_len=73728`,
`max_num_batched_tokens=8192`, `gpu_memory_utilization=0.95`.
Each variant uses one warmup request and ten serial measured requests,
with the same model weights and prompt, 1024 output tokens, temperature 0,
ignore EOS, and prefix caching disabled. Decode calls use SFA G=8/mode 3.

| Context | Metric | Baseline (ms) | With partitioning (ms) | Reduction |
| --- | --- | --- | --- | --- |
| 64K | TPOT | 14.08471 | 12.42442 | 11.8% |
| 64K | TTFT | 16811.41 | 16905.24 | -0.6% |
| 64K | PIL | 37.81700 | 34.31507 | 9.3% |

Reduction is `(baseline - partitioned) / baseline`. Values are request means;
TPOT uses `(last_token_time - first_token_time) / 1023`, and PIL is the
AISBench stream-chunk interval.

MTP acceptance recorded during the test was 0.232152 (baseline) and 0.243453
(partitioned). Precision settings were unchanged; acceptance fluctuated slightly.

## Reproducing correctness checks

On 2026-10-08, the independent A5 branch was rebuilt from code commit
`f7a34b774` (FP16 support and redundant selector checks removed):

- Host UT: 12 tests passed, including FP16 mode 0/3 partition selection and
  common-validation rejection of sparse block sizes 0, 2 and 128 for both dtypes.
- Complete `ascend950` SFA package: built successfully with CANN 9.1.
- Device regression: 235 passed, 8 skipped, 1 xfailed on Ascend950DT_9582,
  CANN 9.1, Python 3.11.10, PyTorch 2.10.0 / torch_npu 2.10.0.post4.
  FP16: 116 passed, 4 skipped. BF16: 115 passed, 4 skipped, 1 xfailed.
  Four additional tests verify that mixed Q/KV dtypes are rejected by the
  existing ACLNN OpDef signature check in modes 0 and 3 (error 161002).
  Skips are inapplicable one-query mixed-row cases; the xfail is the existing
  BF16 mode-3 padded-index non-finite output case.
- Repository pre-commit checks: clang-format 18.1.8, Ruff 0.14.14, OAT,
  duplicate-header and basic file checks passed.

The package was installed in a private vendor prefix; system CANN was not
changed. Package SHA256:
`6bc31e130ed4d120d246a7c679f6eb5eb812811d9aa15ba1d9a03fb7111241ff`.

Build an isolated operator package:

```bash
bash build.sh --ophost_test --soc=ascend950 --ops=sparse_flash_attention -j8
bash build.sh --pkg --soc=ascend950 --ops=sparse_flash_attention \
  --vendor_name=sfa_a5_only -j8
```

Install the package in a private prefix, source its vendor `bin/set_env.bash`,
and run:

```bash
python3 -m pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch35.py
```

Coverage includes sparse mode 0/3, single/multi-batch TND, TopK 256–2048,
eager/graph replay, full index rows, padded rows, and mixed split/fallback rows.
Full rows use an independent softmax reference with relative L2 < 0.02 for
BF16 and < 0.005 for FP16. Padded rows are fallback smoke checks; conditional
xfails cover only B=1/Q=4 eager padded-index cases reproduced with partitioning
disabled (BF16 mode 3 and FP16 modes 0/3). Only the BF16 case was non-finite in
this run. These checks do not assert padded-row numerical equivalence.
The historical performance tables above remain BF16 measurements; FP16
coverage added here is correctness validation, not a new performance result.
