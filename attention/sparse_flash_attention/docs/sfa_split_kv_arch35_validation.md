# A5 SparseFlashAttention Split-KV validation

## Scope and versions

This PR contains only the Ascend 950 (A5) `sparse_flash_attention` path:
matching FP16 or BF16 inputs/output, TND/PA_BSND, one KV head, G=1–128,
sparse mode 0/3, TopK 256–2048, LSE on/off, without sinks.
Unsupported shapes retain the original path.
It does not change arch22, C8, indexer, or model integration code.

The performance measurements below were collected on 2026-09-30 (single
operator) and 2026-10-06 (AISBench) before the A5/A2 PR split. They are reused
A5 results, not a new performance comparison against current master.
The new branch is based on `d5b874d33cbaafbb31af55e57e296c50933aca0f`;
the original arch35 optimization was retained from PR !13112. The newer
generalization and template-isolation checks are listed separately below.

## 2026-10-08 head generalization / FD isolation

- Host UT: 12/12 passed. Partition/core-count coverage now includes G=1,3,8,64,65,97,128,
  8/24/56 AICs, TopK boundaries and query counts that disable FD. Tests check
  both the FD template key and workspace lower bounds.
- Final-package device regression: 2067 passed, 16 skipped, 1 xfailed. Every integer G in
  [1,128] is checked in FP16 and BF16 with returned max/sum. A broader matrix
  covers 21 boundary/odd head counts, modes 0/3, eager/graph replay and LSE on/off.
  Full rows are compared independently for attention output and both softmax statistics.
- All 336 dedicated non-FD cases passed: G=1,3,8,64,65,97,128, TopK=128/4096
  or total Q=40, both dtypes, modes 0/3, LSE on/off and eager/graph replay.
- The 16 skips are one-query mixed-row cases. The conditional xfail is a
  pre-existing eager padded-index non-finite case, not a full-row exemption.
  The old package was also run: 1727 passed, 16 skipped, 4 xfailed, and one
  additional BF16 mode-0 eager padded-index non-finite failure. The new package
  passed that case; padded-row numerical equivalence is still not claimed.
- Compile-time template-scope probes confirm one FD selection on arch35 and
  none on arch22. The final A5-only template guard produces byte-identical
  FP16/BF16 kernel objects to the earlier package used for timing. The complete
  final package was also installed in its own vendor prefix and device-tested.

The device uses Ascend950DT_9582, CANN 9.1, PyTorch 2.10.0 and torch_npu
2.10.0.post4. Its old TorchNPU PA/LSE wrapper allocates an incorrect output
shape before kernel launch. The repository's existing ACLNN V2 extension
was used for both arms instead, with no changes to its calculation or output
allocation logic. The opt-in test adapter is described below.

### Current single-operator comparison

Baseline is the previously published A5 code `f7a34b774`, not master:
G8/no-LSE already uses partitioning; other heads and LSE use its original
unsplit path. Candidate adds generic heads, LSE and FD template isolation.
Both arms use identical inputs and the same repository ACLNN adapter.
The runs are interleaved baseline/candidate for three rounds. Each round
uses 32 calls per graph and 101 event-timed replays. Values below are the
median of the three per-round medians, in microseconds. B=1, TopK=2048,
BF16, sparse mode 0, KV length 4096.

| G | Total Q | LSE | Old A5 PR (us) | New (us) | Latency reduction |
| --- | --- | --- | --- | --- | --- |
| 8 | 1 | off | 17.488 | 17.259 | 1.3% |
| 8 | 1 | on | 70.329 | 17.307 | 75.4% |
| 8 | 4 | on | 70.273 | 20.260 | 71.2% |
| 64 | 1 | off | 72.119 | 40.420 | 44.0% |
| 64 | 1 | on | 72.209 | 40.422 | 44.0% |
| 64 | 4 | off | 72.193 | 30.250 | 58.1% |
| 64 | 4 | on | 72.238 | 30.200 | 58.2% |
| 128 | 1 | off | 53.844 | 42.105 | 21.8% |
| 128 | 4 | off | 54.027 | 33.628 | 37.8% |

The complete timing matrix has 96 configurations: Q=1/4, TopK=256/2048,
G=8/64/128, FP16/BF16, modes 0/3 and LSE off/on. Within the candidate,
LSE overhead has a median of 0.18% and a maximum of 5.35%; the largest absolute
increase is 0.662 us. G8/no-LSE changes range from -2.67% to +0.77% latency,
with a median of -1.27%. These are single-operator measurements; AISBench
has not been rerun for this revision.

Final isolated package SHA256:
`e314a8d550ff555e845bd7a4ebe2f8ffecb70060a053e0b165fc3bbeb9c081de`.

## Historical G8 single-operator timing

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

## Historical A5 64K AISBench end-to-end timing

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

## Previous G8/FP16 correctness checks

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

## Running the current regression on older TorchNPU

If the installed TorchNPU already allocates correct PA LSE outputs, no test
adapter is needed. Otherwise, use the repository's unchanged C++ extension:

```bash
mkdir -p /tmp/sfa_arch35_test_adapter
cp attention/sparse_flash_attention/torch_extension/csrc/sparse_flash_attention.cpp \
  torch_extension/cann_ops_transformer/common/aclnn_common.h /tmp/sfa_arch35_test_adapter/
export SFA_TEST_EXTENSION_DIR=/tmp/sfa_arch35_test_adapter
export MAX_JOBS=2
python3 -m pytest -q attention/sparse_flash_attention/tests/pytest/test_sfa_split_kv_arch35.py
```

The same `SFA_TEST_EXTENSION_DIR` is honored by the timing script. Neither
adapter is used by the packaged operator or production kernel.
