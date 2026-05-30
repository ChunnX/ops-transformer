# QkvRmsNormRopeCacheWithKScale

This directory is the implementation workspace for the arch35
`QkvRmsNormRopeCacheWithKScale` task plan.

Current state:

1. T00 host op definition, infer shape, tiling contract, and host UT sources
   are present. `op_host/qkv_rms_norm_rope_cache_with_k_scale_contract.cpp`
   contains pure host-side validation and tiling math shared by tiling runtime
   and UT; it is not device kernel code.
2. T01 kernel controller and basic block skeleton are present under
   `op_kernel`. The kernel entry file only selects the tiling key, unpacks
   tiling, and dispatches the controller through the aggregate
   `qkv_rms_norm_rope_cache_with_k_scale_d128_fast.h` header.
3. The kernel uses a POD tiling layout under `op_kernel` that host tiling writes
   directly. AIC and paired AIV cores derive the same contiguous, balanced
   cube-group token range. T02 keeps the range loop in the controller and adds
   per-tile tail parameters plus AIV0/AIV1 local token/row slices in the D=128
   fast-path basic block.
4. T03 adds static GM/L1/UB tensor binding. `rotation_l1_nz` is AIC-only,
   `a_rot_l1_nz` is the Cube AL1 data slot shared by the later AIV->AIC
   protocol, and `bufferId = tileId & 1` only selects data buffers.
5. T04 adds fixed CrossCore flag ids and wrappers for `A_READY`, `C_READY`, and
   `C_CONSUMED`. CrossCore flag ids are semantic constants and are not derived
   from pingpong `bufferId` or loop generation.
6. After T04 the D=128 fast-path code is split into `common`, `controller`,
   `basic_block`, `cube`, and `vec` headers instead of keeping those larger
   modules inside the aggregate header.
7. Real compute and the concrete synchronization call sites are left to later
   tasks.
8. Implementation work must follow `TASKS.md`.
9. After each task is coded and verified, update `TASKS.md` and commit the
   code plus the task status change together.
