# LightningIndexer Performance

## Candidate

Hypothesis:
- Many local ST shapes create fewer base blocks than the default host launch count.
- The kernel `SplitCore` already reduces effective work from actual base-block count, but host still launches physical mix blocks, adding scheduling overhead for small and boundary cases.

Change:
- Calculate a static upper bound of base blocks from `B`, `N2`, `S1`, `S2`, layout, `sparse_count`, and `sparse_mode`-compatible base sizes.
- Use `min(physicalBlockDim, maxBaseBlockNum)` for host `blockDim`.

Risk:
- Host only sees shape upper bounds, while actual valid sequence lengths are runtime tensors.
- Therefore the host cap must not be narrower than shape-implied maximum work. Runtime exact split remains inside the kernel.

## Result

| case | baseline us | candidate us | blockDim before | blockDim after | gain |
| --- | ---: | ---: | --- | --- | ---: |
| ST0 | 17.02 | 11.46 | 20/40 | 1/2 | 32.67% |
| ST1 | 17.68 | 9.58 | 20/40 | 2/4 | 45.81% |
| ST2 | 17.44 | 11.92 | 20/40 | 1/2 | 31.65% |
| ST3 | 16.88 | 12.90 | 20/40 | 1/2 | 23.58% |
| ST4 | 16.96 | 9.20 | 20/40 | 2/4 | 45.75% |
| ST5 | 16.44 | 9.30 | 20/40 | 2/4 | 43.43% |
| ST6 | 15.76 | 12.18 | 20/40 | 1/2 | 22.72% |
| ST7 | 18.26 | 8.84 | 20/40 | 2/4 | 51.59% |
| ST8 | 19.96 | 12.60 | 20/40 | 4/8 | 36.87% |
| ST9 | 17.20 | 11.70 | 20/40 | 1/2 | 31.98% |
| README PA large | 25.38 | 25.28 | 20/40 | 16/32 | 0.39% |

Decision:
- Keep the optimization for current matrix.
- Treat README PA large as neutral; do not claim meaningful gain there.

## Next Hypotheses

- Sparse mode 3 PA large may be dominated by TopK/vector merge and workspace traffic rather than launch count.
- Inspect per-task timeline and AIC/AIV utilization for README PA large before modifying kernel loops.
- Any kernel-side optimization must preserve current ST accuracy and use the same matrix for regression checks.
