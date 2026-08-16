# SparseAttentionScorePrefill A2 Performance Validation

## Scope

Target: Ascend 910B, physical device 1, `1800 / 1800 MHz`.

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

## 128K Baseline

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

Profile evidence:

- Score baseline:
  `/opt/rrh/code3/sparse_attention_test/score_quant2_0805_128k_baseline_20260816_1934/OPPROF_20260816193128_LCAUHVXFWGMPZBPD/OpBasicInfo.csv`
- Prefill innerPrecise=0:
  `/opt/rrh/code3/sparse_attention_test/prefill_matrix_128k_inner0_20260816/OPPROF_20260816194120_AEMBNWJDOZHYJVDY/OpBasicInfo.csv`
- Prefill innerPrecise=1:
  `/opt/rrh/code3/sparse_attention_test/current_inner1_128k_diag_20260816/OPPROF_20260816185751_WUSXMIZUJAPMLNVG/OpBasicInfo.csv`
- Prefill innerPrecise=4:
  `/opt/rrh/code3/sparse_attention_test/current_inner4_128k_diag_20260816/OPPROF_20260816185834_SIJRJBOSHMNHQZKK/OpBasicInfo.csv`

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

## High-Precision Softmax Limitation

`innerPrecise=0` is not the low-precision path. It preserves the QK score S
in FP32 through softmax. The A2 kernel intentionally limits this path to one
Q group per batch because the two-group case can reuse an S/P staging lifetime
before the delayed PV consumer completes.

A controlled attempt to raise the cap to two groups passed the small
high-precision test but stalled on the production `groupSize=16` case. The
process was terminated by the required 180-second watchdog and the change was
reverted before it was committed. Therefore the current 4.9 ms result is a
correctness-preserving upper bound, not an accepted performance optimization.

Any future optimization of this path must first separate or add S/P pipeline
stages and prove cross-core flag ownership for the larger batch. Simply
raising `batchGroupsMax` is unsafe.
