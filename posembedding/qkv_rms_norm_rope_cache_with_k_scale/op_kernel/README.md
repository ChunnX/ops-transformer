# op_kernel

This directory contains the AscendC D=128 fast-path implementation for tiling
key `350128` with `KERNEL_TYPE_MIX_AIC_1_2`.

`qkv_rms_norm_rope_cache_with_k_scale.cpp` is only the kernel entry and tiling
key dispatch shell, and it includes the controller directly. The D=128
fast-path implementation is split by module:

- `qkv_rms_norm_rope_cache_with_k_scale_common.h`: common constants, POD
  structs, tensor bundles, scalar helpers, AIV slice helper, and CrossCore
  wrappers.
- `qkv_rms_norm_rope_cache_with_k_scale_controller.h`: controller-level
  private range/tile helpers, contiguous token range loop, and tile scheduling.
- `qkv_rms_norm_rope_cache_with_k_scale_basic_block.h`: resource binding,
  static tensor binding, lifecycle hooks, and AIC/AIV dispatch without
  retaining the raw `GlobalTensors` bundle as member state.
- `qkv_rms_norm_rope_cache_with_k_scale_cube.h`: AIC/Cube branch.
- `qkv_rms_norm_rope_cache_with_k_scale_vec.h`: AIV/Vector branch.
- `qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h`: kernel-side POD tiling
  data.

The controller reads the kernel-side POD tiling data, maps AIC `GetBlockIdx()`
and AIV `GetBlockIdx() / 2` to the same cube group, consumes host-computed
`coreTokenTile`/`coreGroupNum`, and loops over local tiles in controller scope.
It dispatches each tile through `Init`, `ComputeTile`, and `End` lifecycle
hooks.

The host side owns the ceil-div token-per-core calculation. Kernel code only
uses the resulting tiling fields, which keeps the core range policy in tiling
and the per-core tile loop in the controller.

The basic block now derives per-tile tail parameters, including `M_tile` row
size, and computes AIV0/AIV1 local token/row slices using the actual tile token
count.

T03 binds the static tensor layout directly with `LocalTensor<T>(TPosition,
offset, size)` in the D=128 fast-path. No `TPipe`/`TBuf` allocation is used for
these fixed L1/UB regions:

- GM pointers are wrapped as typed `GlobalTensor` members once during `Init`;
  the controller and basic block do not store the raw `GlobalTensors` input
  bundle.
- AIC owns resident `rotation_l1_nz`; `a_rot_l1_nz[2]` exposes two logical
  pingpong AL1 data buffers. The static L1 budget still reserves the 4 * 64 KiB
  A_L1 pool, and AIV0/AIV1 write disjoint row ranges into the same logical
  `bufferId`.
- AIV UB is split into input DB, output DB, and 40 KiB reserve areas. The
  reserve area holds `gamma_ub`, `cos_sin_ub[2]`, `slot_mapping_ub[2]`,
  `v_scale_ub`, and `v_out_ub[2]`.
- Q/K FP8 staging reuses the `qk_after_cube_ub` output DB buffer instead of
  allocating another static region.

T04 adds fixed semantic CrossCore flag ids and wrappers. `bufferId = tileId & 1`
only selects data buffers; it is not used to derive CrossCore flag ids.

| Edge | Direction | Producer set pipe | Consumer wait pipe | Flag |
|------|-----------|-------------------|--------------------|------|
| `A_READY` | AIV -> AIC | `PIPE_MTE3` | `PIPE_MTE1` | `SYNC_A_READY = 0` |
| `C_READY` | AIC -> AIV | `PIPE_FIX` | `PIPE_V` | `SYNC_C_READY = 1` |
| `C_CONSUMED` | AIV -> AIC | `PIPE_MTE3` | `PIPE_FIX` | `SYNC_C_CONSUMED = 2` |

AIV code sets or waits only the base flag id. AIC wrappers handle both AIV0 and
AIV1 using `flag + 16` for AIV1 and the base id for AIV0. The T04 helpers are
defined but not yet called from the stub compute path; T06/T08/T10 will attach
them to the real producer/consumer stages. `a_rot_l1_nz` participates in the
later CrossCore-protected AL1 data handoff, but it does not consume AIV
intra-core event ids.
