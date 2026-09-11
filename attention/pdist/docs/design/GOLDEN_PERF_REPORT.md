# Golden NPU Performance Reference

- **Device ID**: 0
- **Iterations**: 1
- **Warmup**: 3
- **Repeats**: 3
- **Seed**: 42

## Performance Summary

| case | Input Shape | dtype | Median E2E (us) |
|------|-------------|-------|-----------------|
| perf_p0_euclidean | [4096,256] | float32 | 45514.204000 |
| perf_p0_fp16_large | [8192,512] | float16 | 344973.570000 |
| func_p0_hamming_p0 | [1000,200] | float32 | 3351.899000 |
| func_p0_manhattan_p1 | [1000,200] | float32 | 3062.745000 |
| func_p0_euclidean_p2 | [1000,200] | float32 | 3073.319000 |
| func_p0_fractional_p05 | [1000,200] | float32 | 3063.928000 |
| func_p0_chebyshev_pinf | [1000,200] | float32 | 2043.355000 |
| func_p0_min_shape | [2,1] | float32 | 15.807000 |
| func_p0_large_m | [64,8192] | float32 | 696.957000 |
| func_p0_partial_block | [777,300] | float32 | 2840.673000 |

## Raw Repeats

- `perf_p0_euclidean`: [45527.331000, 45514.204000, 45492.542000] us
- `perf_p0_fp16_large`: [344973.570000, 344945.953000, 344981.243000] us
- `func_p0_hamming_p0`: [3351.899000, 3365.285000, 3350.476000] us
- `func_p0_manhattan_p1`: [3062.745000, 3056.538000, 3064.630000] us
- `func_p0_euclidean_p2`: [3073.319000, 3070.598000, 3282.699000] us
- `func_p0_fractional_p05`: [3058.985000, 3065.297000, 3063.928000] us
- `func_p0_chebyshev_pinf`: [2043.355000, 2039.411000, 2044.236000] us
- `func_p0_min_shape`: [15.807000, 16.061000, 15.402000] us
- `func_p0_large_m`: [696.957000, 697.304000, 696.321000] us
- `func_p0_partial_block`: [2840.673000, 2841.381000, 2837.641000] us

All positive NPU kernel durations emitted by the Golden callable are included.
Input preparation and warmup are outside each formal profiler region.
