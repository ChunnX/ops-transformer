# op_kernel

This directory contains the AscendC D=128 fast-path implementation for tiling
key `350128` with `KERNEL_TYPE_MIX_AIC_1_2`.

`qkv_rms_norm_rope_cache_with_k_scale.cpp` is only the kernel entry and tiling
key dispatch shell, and it includes the controller directly. The D=128
fast-path implementation is split by module:

- `qkv_rms_norm_rope_cache_with_k_scale_common.h`: common constants, POD
  structs, raw GM argument bundle, Cube/Vec typed GM bundles, scalar helpers,
  AIV slice helper, and CrossCore wrappers.
- `qkv_rms_norm_rope_cache_with_k_scale_controller.h`: controller-level
  core token range helper, token-size tile expansion, and explicit AIC/AIV
  entry dispatch.
- `qkv_rms_norm_rope_cache_with_k_scale_basic_block.h`: lifecycle hooks and
  single-tile AIC/AIV compute entry points.
- `qkv_rms_norm_rope_cache_with_k_scale_cube.h`: AIC/Cube branch.
- `qkv_rms_norm_rope_cache_with_k_scale_vec.h`: AIV/Vector branch.
- `qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h`: kernel-side POD tiling
  data.

The controller reads the kernel-side POD tiling data, maps AIC `GetBlockIdx()`
and AIV `GetBlockIdx() / 2` to the same cube group, consumes host-computed
`coreTokenTile`/`coreGroupNum`, expands the resulting contiguous token range
into token-size tiles, and passes each tile to the AIC or AIV basic-block entry.
Lower layers do not inspect `g_coreType`.

The host side owns the ceil-div token-per-core calculation. Kernel code only
uses the resulting tiling fields, which keeps the core range policy in tiling
and the per-core token-size tile loop in the controller.

The controller derives per-tile tail parameters, including `M_tile` row size,
by iterating `tokenSize = min(tokenTile, range.end - tokenOffset)`. The Vec
path computes AIV0/AIV1 local token/row slices using the actual tile token
count.

T03 binds the static tensor layout directly with `LocalTensor<T>(TPosition,
offset, size)` in the D=128 fast-path. No `TPipe`/`TBuf` allocation is used for
these fixed L1/UB regions:

- Cube and Vec each wrap only their own typed `GlobalTensor` subset during
  module `Init`; controller and basic block do not retain a shared bound-GM
  bundle.
- AIC owns resident `rotation_l1_nz`; `a_rot_l1_pool` is one member tensor and
  the current logical pingpong AL1 buffer is derived inside compute with
  `bufferId * aRotL1SingleBufferElements`. The static L1 budget still reserves
  the 4 * 64 KiB A_L1 pool, and AIV0/AIV1 write disjoint row ranges into the
  same logical `bufferId`.
- AIV UB is split into input DB, output DB, and 40 KiB reserve areas. The
  input/output/reserve DB areas are stored as member base tensors; per-buffer
  local tensor views are derived at the use site with offsets. The reserve area
  holds `gamma_ub`, `cos_sin_db_pool`, `slot_mapping_db_pool`, `v_scale_ub`,
  and `v_out_db_pool`.
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

T05 adds fixed intra-core event ids and Init/End pairing. Event ids are local to
their `HardEvent` pipe pair and are statically checked to stay in `0..5`.

| Pipe pair | Buffer | Event ids |
|-----------|--------|-----------|
| `MTE2 -> V` | input/cos-sin/slot-mapping DB pool current buffer | `0/1` |
| `V -> MTE2` | input UB reuse | `0/1` |
| `V -> MTE3` | v-out DB pool current buffer | `0/1` |
| `MTE3 -> V` | v-out DB pool reuse | `0/1` |
| `V -> MTE3` | output DB pool current buffer for Q/K FP8 staging/write-out | `2/3` |
| `MTE3 -> V` | output DB pool Q/K reuse | `2/3` |
| `MTE1 -> MMAD` / `MMAD -> MTE1` | AIC L0 pingpong path | `3/4` |
| `MMAD -> FIX` / `FIX -> MMAD` | AIC L0C pingpong path | `0/1` |

Init currently sets only reusable reverse events (`V_MTE2`, `MTE3_V`,
`M_MTE1`, `FIX_M`); End waits the same events to consume the final reusable
tokens. The optional A temporary UB staging path is not present in the static
layout, so ids `4/5` are not initialized or waited for that purpose.
