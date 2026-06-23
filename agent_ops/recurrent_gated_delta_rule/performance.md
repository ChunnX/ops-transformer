# RecurrentGatedDeltaRule Performance

## Candidate

Hypothesis:
- The useful parallel work for this kernel is bounded by batch-head tasks, `B * Nv`.
- Launching more vector cores than `B * Nv` creates idle work and scheduling overhead.

Change:
- After shape validation, compute `batchHeadTaskNum = b * nv`.
- If it is non-zero and below current `vectorCoreNum`, use it as `vectorCoreNum`.

Risk:
- Large shapes with `B * Nv >= platform AIV` should not be narrowed.
- Host UT preserves the existing large-case platform core count and adds a small-case assertion.

## Result

| case | baseline us | candidate us | blockDim before | blockDim after | gain |
| --- | ---: | ---: | --- | --- | ---: |
| ST0 | 11.60 | 8.46 | 40 | 16 | 27.07% |
| ST1 | 10.94 | 7.94 | 40 | 16 | 27.42% |
| ST2 | 11.86 | 10.16 | 40 | 24 | 14.33% |
| ST3 | 11.68 | 8.76 | 40 | 16 | 25.00% |
| ST4 | 12.08 | 9.20 | 40 | 16 | 23.84% |
| ST5 | 12.12 | 7.90 | 40 | 8 | 34.82% |
| ST6 | 11.46 | 8.56 | 40 | 16 | 25.31% |
| ST7 | 15.36 | 11.30 | 40 | 8 | 26.43% |

Decision:
- Keep the optimization for current matrix.

## Next Hypotheses

- ST2 remains the lowest-gain case and may be bound by variable-length/state-index handling or state update memory traffic.
- Inspect kernel timeline and memory movement before changing algorithmic paths.
