# QkvRmsNormRopeCacheWithKScale

This directory is the implementation workspace for the arch35
`QkvRmsNormRopeCacheWithKScale` task plan.

Current state:

1. T00 host op definition, infer shape, tiling contract, and host UT sources
   are present. `op_host/qkv_rms_norm_rope_cache_with_kscale_contract.cpp`
   contains pure host-side validation and tiling math shared by tiling runtime
   and UT; it is not device kernel code.
2. T01 kernel controller and basic block skeleton are present in
   `op_kernel/qkv_rms_norm_rope_cache_with_k_scale_d128_fast.h`. The kernel
   entry file only selects the tiling key, unpacks tiling, and dispatches the
   controller.
3. The kernel uses a POD tiling layout under `op_kernel` that host tiling writes
   directly. AIC and paired AIV cores derive the same cube-group tile range; compute,
   slice splitting, buffers, and synchronization are left to later tasks.
4. Implementation work must follow `TASKS.md`.
5. After each task is coded and verified, update `TASKS.md` and commit the
   code plus the task status change together.
