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
   tiling, and dispatches the controller directly.
3. The kernel uses a POD tiling layout under `op_kernel` that host tiling writes
   directly. Host tiling computes `coreTokenTile` with ceil division and writes
   `coreGroupNum`; AIC and paired AIV cores consume those values to derive the
   same contiguous cube-group token range. The controller creates the
   contiguous core token range, expands it into token-size tiles, and derives
   per-tile tail parameters before calling the D=128 fast-path basic block.
4. T03 adds static GM/L1/UB tensor binding. `rotation_l1_nz` is AIC-only,
   `a_rot_l1_nz` is the Cube AL1 data slot shared by the later AIV->AIC
   protocol, and `bufferId = tileId & 1` only selects data buffers. The raw
   `GlobalTensors` bundle is only an entry argument; Cube and Vec bind their
   own typed GM tensors and neither controller nor basic block retains a shared
   bound-GM bundle.
5. T04 adds fixed CrossCore flag ids and wrappers for `A_READY`, `C_READY`, and
   `C_CONSUMED`. CrossCore flag ids are semantic constants and are not derived
   from pingpong `bufferId` or loop generation.
6. T05 adds fixed intra-core event constants and Init/End pairing for reusable
   AIV/AIC buffers. Active event ids stay within `0..5`; `a_rot_l1_nz` remains
   CrossCore-protected AL1 and does not consume AIV intra-core event ids.
7. After T04 the D=128 fast-path code is split into `common`, `controller`,
   `basic_block`, `cube`, and `vec` headers instead of keeping those larger
   modules inside one file. Core-type dispatch is centralized in the
   controller AIC/AIV entry split; lower layers do not inspect `g_coreType`.
8. Real compute and the concrete synchronization call sites are left to later
   tasks.
9. Implementation work must follow `TASKS.md`.
10. After each task is coded and verified, update `TASKS.md` and commit the
   code plus the task status change together.
