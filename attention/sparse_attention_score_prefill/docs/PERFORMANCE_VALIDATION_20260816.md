# SparseAttentionScorePrefill A2 Performance Validation

## Scope

Target: Ascend 910B. The 2026-08-16 matrix used physical device 1; the final
`innerPrecise=0` revalidation below used physical device 7. Every accepted
profile reports `1800 / 1800 MHz`.

All profiles use `msprof op` and select the `SparseAttentionScorePrefill`
kernel. The common prefill shape is:

```
batch=1, qSeqlen=2048, qHeads=16, kvHeads=1,
headDim=128, blockSize=128, topK=16
```

`innerPrecise` has the following implementation meanings on A2:

| innerPrecise | Tiling key | Score S | O_partial workspace |
| --- | --- | --- | --- |
| 0 | 20003 | FP32 | FP32 |
| 1 | 20002 | BF16 | BF16 |
| 4 | 20001 | BF16 | FP32 |

## 128K Baseline (2026-08-16)

The baseline is `SparseAttentionScore` from the separate `quant2/0805`
environment. It uses the same logical shape and was profiled on the same NPU
and frequency.

| Operator / path | Task duration | Reduction vs. score |
| --- | ---: | ---: |
| SparseAttentionScore baseline | 2612.252 us | - |
| Prefill, innerPrecise=0 | 4900.558 us | -87.60% |
| Prefill, innerPrecise=1 | 1947.879 us | 25.43% |
| Prefill, innerPrecise=4 | 2016.980 us | 22.79% |

The default fp32-O_partial path (`innerPrecise=4`) meets the required 20%
reduction. The BF16-O_partial path (`innerPrecise=1`) is also faster than the
baseline and is 3.43% faster than the default path at 128K.

The OPP binary from that validation snapshot was checked against the
`innerPrecise=4` profile dump. Both SHA256 values are
`282ec1a18cba0a99188f493794377c3403fd8600f38015178dccc35bd9927396`, so the
128K profile corresponds to that installed snapshot.

Profile evidence:

- Score baseline:
  `/opt/rrh/code3/sparse_attention_test/score_quant2_0805_128k_baseline_20260816_1934/OPPROF_20260816193128_LCAUHVXFWGMPZBPD/OpBasicInfo.csv`
- Prefill innerPrecise=0:
  `/opt/rrh/code3/sparse_attention_test/prefill_matrix_128k_inner0_20260816/OPPROF_20260816194120_AEMBNWJDOZHYJVDY/OpBasicInfo.csv`
- Prefill innerPrecise=1:
  `/opt/rrh/code3/sparse_attention_test/current_inner1_128k_diag_20260816/OPPROF_20260816185751_WUSXMIZUJAPMLNVG/OpBasicInfo.csv`
- Prefill innerPrecise=4:
  `/opt/rrh/code3/sparse_attention_test/current_inner4_128k_diag_20260816/OPPROF_20260816185834_SIJRJBOSHMNHQZKK/OpBasicInfo.csv`

## InnerPrecise=0 Batch-Group Optimization (2026-08-17)

The old high-precision path forced `batchGroupsMax=1`, although both its L0
tile and FP32 softmax workspace can hold more than one query group. The A2
path now derives the limit with the same capacity model used by the other
precision modes:

```
min(64 / groupSize,
    8192 / (RoundUp(blockSize, 16) * groupSize),
    8)
```

For the production shape (`groupSize=16`, `blockSize=128`) the effective
limit is four query groups, so each Phase1 batch fills the `M=64` Cube tile
instead of running an `M=16` tile. This cuts the number of QK/softmax/PV
pipeline batches by four for a full CSR row.

The larger batch is paired with explicit lifetime protection for the two S
stages, three P stages, and two stats stages. The final FIX transfer is drained
before Phase2 consumes O_partial, and only non-reserved event IDs 0 through 5
are used. Q gather and O_partial scatter remain one group per transfer so CSR
order cannot swap logical groups.

Final same-device profiles:

| Operator / path | Task duration | Reduction vs. score |
| --- | ---: | ---: |
| SparseAttentionScore, device 7 | 2605.392 us | - |
| Prefill, `batchGroupsMax=2`, device 7 | 2971.819 us | -14.06% |
| Prefill, dynamic cap=4, sample 1 | 2032.061 us | 22.01% |
| Prefill, dynamic cap=4, sample 2 | 2044.861 us | 21.51% |
| Prefill, dynamic cap=4, sample 3 | 2005.120 us | 23.04% |
| Prefill, dynamic cap=4, median | **2032.061 us** | **22.01%** |

The 20% target is `<= 2084.314 us`, so the final median passes by 52.253 us.
The previous cap=1 result was 4901.938 us on device 0; it is useful as a
directional reference only because it was not collected on device 7.

All three profile dumps and the installed OPP object have SHA256
`a648c501186af023b356b083be9175e32c3a7b92a497276d0199c5785464d7eb`.

Profile evidence:

- Same-device SparseAttentionScore baseline:
  `/opt/rrh/code3/sparse_attention_test/score_inner0_128k_npu7_0805_20260817_1925/OPPROF_20260817192445_EQLAJSWJMVAQMLZM/OpBasicInfo.csv`
- Final sample 1:
  `/opt/rrh/code3/sparse_attention_test/prefill_inner0_batchgroups_commit_npu7_20260817_2021_t1/OPPROF_20260817202014_PEAPQVWFDVLSOOES/OpBasicInfo.csv`
- Final sample 2:
  `/opt/rrh/code3/sparse_attention_test/prefill_inner0_batchgroups_commit_npu7_20260817_2021_t2/OPPROF_20260817202055_KWYUANIGEXRFKLGE/OpBasicInfo.csv`
- Final sample 3:
  `/opt/rrh/code3/sparse_attention_test/prefill_inner0_batchgroups_commit_npu7_20260817_2021_t3/OPPROF_20260817202129_SBJSRAHKEWEQDJHM/OpBasicInfo.csv`

## KV-Length Matrix

All entries are kernel task durations in microseconds. Each profile reports
`BlockDim=20`, `MixBlockDim=40`, and `Current Freq=Rated Freq=1800 MHz`.

| KV sequence length | innerPrecise=0 | innerPrecise=1 | innerPrecise=4 |
| ---: | ---: | ---: | ---: |
| 64K | 4922.059 | 1948.059 | 1976.900 |
| 128K | 4900.558 | 1947.879 | 2016.980 |
| 256K | 4942.799 | 1990.960 | 2036.521 |

The selected topK remains 16, so KV cache length has little impact on this
fixed-window request. `innerPrecise=1` reduces Phase1/Phase2 O_partial
workspace traffic, but Phase2 converts each partial result back to FP32 for
the weighted combine. The remaining vector/scalar control work limits the
benefit to a few percent over `innerPrecise=4`.

## Precision Coverage

The following A2 tests have passed after the current implementation was
rebuilt and installed:

- `test_bf16_high_precision_softmax` (`innerPrecise=0`): max diff `0.000000`.
- Production GQA precision matrix: `innerPrecise=0/1/4`, Q heads 64, KV heads
  4, group size 16.
- D=256 matrix: `innerPrecise=0/1/4`.
- Paged block-size sweep: every positive multiple of 16 from 16 through 128.
- Axis-0-strided Q/K/V, continuous non-cache KV (`blockSize=0`), zero Q/KV
  lengths, LSE output, 256K KV boundary, and total-Q 1024K boundary.

The representative precision matrix reported max absolute differences of
`0.000244` for `innerPrecise=0/4` and `0.000488` for `innerPrecise=1`.

Final revalidation after the last installation:

- Production GQA inner-precision matrix: `1 passed, 3 subtests passed` in
  59.12 seconds. Observed max differences were `0.000244`, `0.000122`, and
  `0.000244` for modes `0`, `1`, and `4`.
- Short capability regression: `13 passed, 19 subtests passed` in 27.95
  seconds.
- KV=256K and total-Q=1024K boundaries: `2 passed` in 12.03 seconds. The
  former launched in 2.693 ms and the latter in 114.998 ms; both produced the
  expected constant output exactly.

## High-Precision Safety Boundary

`innerPrecise=0` preserves QK score S and O_partial in FP32. It no longer has
a special one-group cap, but it is still bounded by the actual L0 and softmax
workspace capacities. The published A2 regression for this optimization is
`groupSize=16`, `headDim=128`, and a positive block size aligned to 16 through
128. Shapes outside that contract must not bypass host validation or the
capacity calculation.

Final post-install precision checks passed for high-precision softmax, the
medium case, the partial-tail case, and the Q=16 hot-CSR regression (`4 passed,
2 subtests passed` in 29.38 seconds). The target `groupSize=16`,
`innerPrecise=0` path matched its CPU sample golden exactly at Q rows 0, 8,
and 15; the matching `innerPrecise=4` subtest had max difference `0.000061`.
