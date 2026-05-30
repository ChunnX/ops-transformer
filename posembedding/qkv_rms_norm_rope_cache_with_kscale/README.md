# QkvRmsNormRopeCacheWithKScale

This directory is the implementation workspace for the arch35
`QkvRmsNormRopeCacheWithKScale` task plan.

Current state:

1. T00 host op definition, infer shape, tiling contract, and host UT sources
   are present.
2. Kernel implementation starts from T01 and must consume the T00 contract
   instead of redefining shape or tiling rules.
3. Implementation work must follow `TASKS.md`.
4. After each task is coded and verified, update `TASKS.md` and commit the
   code plus the task status change together.
