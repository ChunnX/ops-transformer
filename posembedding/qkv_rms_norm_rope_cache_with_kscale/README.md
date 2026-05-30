# QkvRmsNormRopeCacheWithKScale

This directory is the implementation workspace for the arch35
`QkvRmsNormRopeCacheWithKScale` task plan.

Current state:

1. T00 host op definition, infer shape, tiling contract, and host UT sources
   are present.
2. T01 kernel controller and basic block skeleton are present. The kernel uses
   a POD tiling layout under `op_kernel` that host tiling writes directly.
3. AIC and paired AIV cores derive the same cube-group tile range; compute,
   slice splitting, buffers, and synchronization are left to later tasks.
4. Implementation work must follow `TASKS.md`.
5. After each task is coded and verified, update `TASKS.md` and commit the
   code plus the task status change together.
