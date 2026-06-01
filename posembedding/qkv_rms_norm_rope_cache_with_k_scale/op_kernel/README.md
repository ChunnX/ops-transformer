# op_kernel

This directory contains the AscendC D=128 fast-path implementation for tiling
key `350128` with `KERNEL_TYPE_MIX_AIC_1_2`.

`qkv_rms_norm_rope_cache_with_k_scale.cpp` is only the kernel entry and tiling
key dispatch shell, and it includes the controller directly. The D=128
fast-path implementation is split by module:

- `qkv_rms_norm_rope_cache_with_k_scale_common.h`: common constants, POD
  structs, raw GM argument bundle, scalar helpers, 2D GM-to-UB copy helper
  backed by explicit `DataCopyPad` parameters, AIV slice helper, and CrossCore
  wrappers.
- `qkv_rms_norm_rope_cache_with_k_scale_controller.h`: controller-level
  core token range helper, token-size tile expansion, and explicit AIC/AIV
  entry dispatch.
- `qkv_rms_norm_rope_cache_with_k_scale_basic_block.h`: lifecycle dispatch,
  rolling AIC/AIV tile scheduling, CrossCore tile protocol, and drain.
- `qkv_rms_norm_rope_cache_with_k_scale_cube.h`: AIC/Cube branch.
- `qkv_rms_norm_rope_cache_with_k_scale_vec.h`: AIV/Vector branch.
- `qkv_rms_norm_rope_cache_with_k_scale_vf.h`: D=128 RMSNorm + half-RoPE BF16
  vector function used by the AIV preprocess path, plus V BF16 scale and FP8
  cast helpers.
- `qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h`: kernel-side POD tiling
  data.

The controller reads the kernel-side POD tiling data, maps AIC `GetBlockIdx()`
and AIV `GetBlockIdx() / 2` to the same cube group, consumes host-computed
`coreTokenTile`/`coreGroupNum`, expands the resulting contiguous token range
into token-size tiles, and passes each tile to the AIC or AIV basic-block
scheduler entry. The controller does not split AIV preprocess/postprocess
phases; basic block retains the rolling pipeline state. Lower layers do not
inspect `g_coreType`.

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
AIV1 using `flag + 16` for AIV1 and the base id for AIV0. Basic block owns the
semantic CrossCore protocol: after Vec finishes the AL1 staging boundary for a
tile, basic block notifies `A_READY` through `PIPE_MTE3`; before Cube consumes
`a_rot_l1_nz`, basic block waits `A_READY` through `PIPE_MTE1`. After Cube
Fixpipe completes, basic block sends `C_READY` through `PIPE_FIX`; the AIV side
waits it through `PIPE_V`, runs Q/K postprocess, and sends `C_CONSUMED` after
the output DB pool is safe for reuse.

T05/T06 keep fixed intra-core event ids inside the AIV/AIC implementation
classes. Basic block owns only CrossCore protocol sequencing; it does not expose
or manage intra-core event ids. Event ids are local to their `HardEvent` pipe
pair and are statically checked to stay in `0..5`.

| Pipe pair | Buffer | Event ids |
|-----------|--------|-----------|
| `MTE2 -> V` | input/cos-sin/slot-mapping DB pool current buffer | `0/1` |
| `V -> MTE2` | input UB reuse | `0/1` |
| `MTE3 -> V` | v-out DB pool reuse | `0/1` |
| `MTE3 -> V` | output DB pool Q/K reuse | `2/3` |
| `V -> MTE3` / `MTE3 -> V` | AIV-private AL1 staging pipe boundary | `4` |
| `MTE2 -> MTE1` | resident AIC rotation L1 ready | `5` |
| `MTE1 -> MMAD` / `MMAD -> MTE1` | AIC L0 pingpong path | `3/4` |
| `MMAD -> FIX` / `FIX -> MMAD` | AIC L0C pingpong path | `0/1` |

Init currently sets only reusable reverse events (`V_MTE2`, `MTE3_V`,
`M_MTE1`, `FIX_M`); End waits the same events to consume the final reusable
tokens. T06 uses event id `4` inside `D128FastVec::ComputeTile` for the current
AL1 staging pipe handoff only. It is not initialized in lifecycle hooks and
is not per-AL1-buffer synchronization.

T06 wires the AIV preprocess boundary and the initial numeric Q/K path. Each
AIV derives local tensor views inside `ComputeTile`, waits for input UB reuse,
loads QKV/gamma/slot-mapping/cos-sin through the 2D copy helper, publishes and
consumes the `MTE2 -> V` input-ready event, runs the D=128 VF for
RMSNorm + half-RoPE into a BF16 ND staging area, converts that ND staging into
the logical AL1 NZ buffer, waits for the private MTE3 completion event,
and returns to basic block. Basic block then issues `A_READY`. On the AIC
side, basic block waits `A_READY` before entering Cube compute that touches
the logical AL1 buffer.

T07 adds the AIC resident-rotation and Cube matmul slice. The AIC branch calls
`PrepareRotationBeforeLoop()` after lifecycle init and before the tile loop,
copies the external dense BF16 `[128,128]` `rotation` matrix to
`rotation_l1_nz` with MTE2 ND2NZ, and uses a local `MTE2 -> MTE1` event for
the resident L1 visibility boundary. No CrossCore flag is introduced for
rotation. Per tile, basic block still waits `A_READY` from both AIVs before
entering Cube compute; Cube then loads A and resident rotation into L0 in
up-to-128-row M chunks and issues BF16 x BF16 -> FP32 MMAD into L0C.

T08 adds AIC Fixpipe distribution from L0C to the shared output DB pool in AIV
UB. Cube derives the AIV0/AIV1 split from the actual per-tile token split,
emits the intersecting row ranges with single-target Fixpipe and `subBlockId`
0/1, then releases L0C for reuse. Basic block owns the CrossCore sequencing:
after Cube compute AIC sends `C_READY`, and the AIV side waits it before Q/K
postprocess and `C_CONSUMED`.

T09 adds the V branch early postprocess without moving CrossCore sync into Vec.
The AIV basic-block path now preprocesses Q/K/V, notifies `A_READY`, runs V
postprocess, then waits `C_READY`. Vec delays input UB release until after V
has consumed the staged QKV row data. The V path loads `v_scale` into the
reserve UB, calls `VScaleFp8D128Vf` with raw UB addresses so the token/head
loops, BF16-to-FP32 conversion, scale multiply, and FP8 cast all stay inside
the VF function, then scatters the packed FP8 rows into `v_cache` by
`slot_mapping`, cache block, cache head, and cache token strides. `v_out_ub`
uses local `V_MTE3`/`MTE3_V` event ids `0/1`; CrossCore calls remain only in
basic block through the semantic wrappers.

T10 adds Q/K postprocess after `C_READY` and before `C_CONSUMED`. Basic block
still owns the semantic CrossCore calls; Vec only consumes the already-visible
`qk_after_cube_ub`, quantizes it, and writes results. `QkDynamicQuantD128Vf`
keeps the row loop and numeric work inside VF: D=128 FP32 row load,
`max_abs`, `scale = max_abs / 448`, divide, FP8 RINT/SAT cast, and scale
staging. The Q branch writes `q_out` and `q_scale`; the K branch writes
`k_cache` and `k_scale_cache` using the same slot-to-cache address calculation
as V. Q/K FP8 staging reuses `qk_after_cube_ub`, and the corresponding local
`V_MTE3`/`MTE3_V` handoff uses event ids `2/3`.

T11 keeps the rolling schedule inside basic block, matching the grouped-matmul
basic-block pattern where the block tracks pending state and `End()` drains the
last stage. The controller only expands tiles. For tile `i`, AIV preprocesses
Q/K/V, sends `A_READY`, completes V, then Q/K-postprocesses pending tile
`i - 1`; after the token loop, `End()` drains the final pending tile. AIC keeps
up to the output double-buffer depth in flight: before starting a tile it waits
`C_CONSUMED` only when the output DB window is full, then waits `A_READY`, runs
Cube/Fixpipe, sends `C_READY`, and records one pending output consumption.
`End()` waits the remaining `C_CONSUMED` events. This lets Cube/Fixpipe for
tile `i + 1` overlap with AIV Q/K postprocess for tile `i` when they use
different output DB buffers. Every real tile still has balanced CrossCore
counts: each AIV sets one `A_READY`, waits one `C_READY`, and sets one
`C_CONSUMED`; AIC waits/sets/waits the paired two-AIV events through the fixed
semantic wrappers. Vec waits the Q/K output reusable event during preprocess
before `qk_after_cube_ub` is reused as BF16 staging, and Q/K postprocess
releases that event after FP8 staging MTE3 writeout, including empty local
slices.

T12 makes empty AIV slices an explicit no-data path without changing the
CrossCore protocol. `MakeAivSlice` can return `tokenSize == 0` and
`rowSize == 0`, usually for tail tiles where AIV1 has no token. The AIV still
runs the basic-block phases and therefore still sets `A_READY`, waits
`C_READY`, and sets `C_CONSUMED`. Vec preprocess consumes the local input and
Q/K output reusable events, then skips GM input copy, VF RMSNorm/RoPE, and AL1
ND2NZ when the slice is empty. V and Q/K postprocess skip data writes for empty
slices; Q/K postprocess still releases the Q/K output reusable event that
preprocess consumed. The row ranges remain contiguous because AIV1 starts at
AIV0's actual row count, not a fixed half-tile offset.
