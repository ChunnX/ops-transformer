# QkvRmsNormRopeCacheWithKScale Task Plan

## Rules

1. Work on `master` in this repository.
2. Complete tasks in dependency order.
3. Every task must leave the repository compiling.
4. Host-side tasks require UT coverage.
5. Kernel synchronization tasks currently require compile verification and
   static protocol review; sync-specific tests can be added later.
6. After a task is coded and verified, update this file and commit the task.

## Status

| ID | Task | Status | Verification | Commit |
|----|------|--------|--------------|--------|
| S00 | Create implementation scaffold | Done | Top-level CMake configure reached this no-op directory; configure then failed in pre-existing mc2/ffn targets unrelated to this scaffold | Included in scaffold commit |
| T00 | Host checker/tiling contract + UT | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh` and passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; 9 host UT passed. `--noexec` host UT compile also passed | Included in T00 commit and host UT build.sh fix commit |
| P01 | Empty kernel package precheck | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh` and passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` | e2b91bd90 |
| T01 | Controller and basic block skeleton | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 9 host UT, and passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. Re-ran `--pkg` after splitting the D=128 fast-path kernel template into its own header. | 76d2f307f plus this commit |
| T02 | AIV local slice and tail row range | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT, and passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` | This commit |
| T03 | Static buffer and tensor binding | Done | Direct `LocalTensor<T>(TPosition, offset, size)` binding without `TPipe`/`TBuf`; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| T04 | CrossCore wrappers and flag ids | Done | Added fixed semantic flags `SYNC_A_READY/C_READY/C_CONSUMED` and AIC/AIV wrappers; static review confirmed only `flag + 16` AIV1 handling and no `bufferId`/generation-derived flags; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| R01 | Split D=128 fast-path kernel modules | Done | Split `common`, `controller`, `basic_block`, `cube`, and `vec` into separate headers; `git diff --check` passed; source scan confirmed no `TPipe`/`TBuf`/`InitBuffer` or `BEGIN_TILING_DATA_DEF` usage in the kernel path; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| R02 | Remove empty fast-path aggregate header | Done | Deleted the empty aggregate header and made the kernel entry include `controller` directly; scan found no remaining reference to the removed header; `git diff --check` passed; source scan confirmed no `TPipe`/`TBuf`/`InitBuffer` or `BEGIN_TILING_DATA_DEF` usage in the kernel path; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` and no longer listed the removed header; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| R03 | Move core token split to tiling | Done | Host tiling now computes `coreTokenTile = ceil(totalTokens / aicNum)` and `coreGroupNum`; controller consumes those tiling fields to build the core token range, expands it into token-size tiles, and calls the basic-block single-tile AIC/AIV entries; Cube/Vec bind their own typed GM tensors and controller/basic block do not retain a shared bound-GM bundle; core-type dispatch is centralized in controller AIC/AIV entries with no lower-layer `g_coreType` checks; `git diff --check` passed; source scan confirmed no old `baseTokenSize`/`totalTokens / core` math, no retained shared bound-GM bundle, and no `TPipe`/`TBuf`/`InitBuffer` or `BEGIN_TILING_DATA_DEF`; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT including `CoreTokenRangeUsesHostCeilTokenTile`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` | This commit |
| T05 | Intra-core event constants and Init/End | Done | AIV/AIC intra-core event ids are encapsulated inside `D128FastVec`/`D128FastCube`; reusable-event Init/End pairs AIV input UB, `v_out_ub`, `qk_after_cube_ub`, and AIC L0/L0C reuse paths. `a_rot_l1_nz` still does not allocate per-AL1 AIV event ids. | This commit |
| T06 | AIV Q/K/V preprocess | Done | Added AIV preprocess boundary: input UB reuse and `MTE2 -> V` ready events, QKV/gamma/slot_mapping/cos-sin staging, D=128 VF RMSNorm + half-RoPE + BF16 ND output, ND2NZ AL1 write, and private `V -> MTE3`/`MTE3 -> V` AL1 staging event id `4`. Basic block now owns the `A_READY` CrossCore notify/wait around AIV/Cube tile compute. Follow-up optimized the D=128 VF path to keep low/high halves in registers, moved token/head loops into VF, and kept Vec pingpong storage as one base buffer plus active-buffer offset views. Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. Focused preprocess golden/UT is still pending. | 2459db1b9 plus this commit |
| T07 | AIC rotation resident and Cube matmul | Done | AIC now loads dense BF16 `[D,D]` `rotation` to resident `rotation_l1_nz` before the tile loop, while basic block waits both AIV `A_READY` signals per tile before entering Cube compute. Cube loads A/rotation into L0 and executes BF16 x BF16 -> FP32 matmul. Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. Cube UT is still pending. | aba3656fc |
| T08 | AIC Fixpipe distribution and C_READY | Done | AIC now Fixpipes L0C rows into AIV0/AIV1 `qk_after_cube_ub` by the actual `ceil(tokenSize/2) * qkHeadTile` row boundary, using single-target Fixpipe with `subBlockId` 0/1 instead of fixed half-tile splitting. Basic block owns `C_READY` and placeholder `C_CONSUMED` CrossCore sequencing with fixed per-tile counts: AIC sets `C_READY` twice and waits `C_CONSUMED` twice; each AIV waits `C_READY` once and sets `C_CONSUMED` once. `git diff --check` passed; static scan found CrossCore primitives only in common wrappers; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. | This commit |
| T09 | V branch early postprocess | Done | Basic block now runs V postprocess after `A_READY` and before `C_READY` wait, so CrossCore sequencing stays centralized outside Vec. V postprocess keeps `input_ub` alive through V consumption, scales BF16 V by `v_scale`, casts to FP8 inside `VScaleFp8D128Vf`, and scatters to `v_cache` by `slot_mapping`; `v_out_ub` uses local `V_MTE3`/`MTE3_V` events `0/1`. `git diff --check` passed; static scan found CrossCore primitives only in common wrappers; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. Focused V path UT is still pending. | This commit |
| T10 | AIV Q/K postprocess | Done | Basic block now waits `C_READY`, runs Vec Q/K postprocess, then sends `C_CONSUMED`, keeping CrossCore sequencing centralized. Vec consumes `qk_after_cube_ub`, runs `QkDynamicQuantD128Vf` for D=128 `max_abs / 448`, FP8 RINT/SAT cast, and scale staging, then writes Q rows to `q_out/q_scale` and K rows to `k_cache/k_scale_cache` by `slot_mapping`; Q/K output reuse uses local `V_MTE3`/`MTE3_V` event ids `2/3`. `git diff --check` passed; static scan found CrossCore primitives only in common wrappers; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. Focused Q/K postprocess UT is still pending. | This commit |
| T11 | Rolling pipeline and drain | Done | Basic block now owns the rolling scheduler and final drain: AIV keeps one pending Q/K postprocess tile internally and `End()` drains it, while controller only expands token tiles. AIC delays `C_CONSUMED` waits until the output DB window is full, then drains remaining pending consumptions in `End()`, so Cube/Fixpipe for tile `i + 1` can overlap with AIV Q/K postprocess for tile `i` on different DB buffers. Basic block keeps all CrossCore sequencing centralized through fixed semantic wrappers. Q/K output pool reuse still waits in preprocess before `qk_after_cube_ub` is reused as BF16 staging, and Q/K postprocess releases it after MTE3 writeout, including empty local slices. `git diff --check` passed; static scan found CrossCore primitives only in common wrappers and semantic wrapper calls only in basic block; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. | This commit |
| T12 | Tail and empty-slice synchronization | Done | Empty AIV slices now keep the fixed basic-block CrossCore protocol while Vec preprocess skips GM/UB/L1 data work after consuming input and Q/K output reusable events, so empty slices do not access invalid data paths. Host UT added `TailAndEmptySlicesKeepFixedCrossCoreProtocol`, enumerating tail token sizes `1..tokenTile` to verify contiguous AIV row coverage and fixed per-tile counts: two `A_READY` sets/waits, two `C_READY` sets/waits, and two `C_CONSUMED` sets/waits. `git diff --check` passed; static scan found CrossCore primitives only in common wrappers and semantic wrapper calls only in basic block; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 12 host UT; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. | This commit |
| T13 | Synchronization protocol audit | Not started | Kernel compile + static audit | |
| T14 | Kernel UT/ST closure | Not started | Full UT/ST | |

## Next Task

Start T13 by auditing the CrossCore and intra-core synchronization protocol.

## Kernel Layout

The D=128 fast path keeps the kernel entry as a thin dispatch shell that
includes `controller` directly. Shared constants, raw GM argument bundle, scalar
helpers, AIV slice helper, fixed CrossCore flag ids, and low-level CrossCore
wrappers live in `common`; controller owns core-range derivation, token-size
tile expansion, and explicit AIC/AIV entry dispatch; basic block owns lifecycle
hooks and CrossCore tile sequencing; Cube and Vector code bind their own typed
GM tensor subsets in their module headers. Vec binds each pingpong pool as one
local base tensor and derives active-buffer views by offset inside the tile
compute path.
Host tiling owns the ceil-div core token split and writes the resulting
`coreTokenTile` and `coreGroupNum` into the kernel POD tiling data.

T05/T06 keep intra-core event ids private to the AIV/AIC implementation
classes. AIV reusable-event Init/End covers input UB with `V_MTE2`,
`v_out_ub` with `MTE3_V`, and `qk_after_cube_ub` with `MTE3_V`. The T06 AL1
staging boundary uses AIV-private `V_MTE3`/`MTE3_V` event id `4` inside the
tile compute path only; it protects the pipe handoff around BF16 ND staging and
ND2NZ AL1 write, and is not a per-AL1-buffer CrossCore flag. Basic block owns
the `A_READY` CrossCore notify/wait between that AIV AL1 staging boundary and
Cube AL1 consumption. T07 keeps `rotation_l1_nz` resident on AIC without a
CrossCore flag, and AIC Init/End covers L0 reuse with `M_MTE1` and L0C reuse
with `FIX_M`. All active ids are checked to stay within `0..5`.
T08 binds the AIC-side view of the same output DB pool and emits L0C rows to
the two AIV UB address spaces with `subBlockId` 0/1. The row boundary is
derived from the same actual AIV token split used for AL1 input, so tail tiles
do not use a fixed half-tile C split. CrossCore `C_READY`/`C_CONSUMED` calls
remain centralized in basic block; T10 replaces the original balanced
placeholder with real Q/K postprocess consumption.
T09 inserts the V branch early postprocess in the basic-block AIV sequence
after `A_READY` and before `C_READY` wait. Vec keeps `input_ub` ownership
through this V phase, loads `v_scale`, runs BF16 V scale and FP8 cast inside
`VScaleFp8D128Vf`, stages rows in `v_out_ub`, then writes `v_cache` using
`slot_mapping` and cache block/head/token strides. The local `v_out_ub`
handoff uses `V_MTE3`/`MTE3_V` event ids `0/1`; no CrossCore calls are added
inside Vec.
T10 replaces the balanced placeholder `C_CONSUMED` with real Q/K
postprocess. Basic block waits `C_READY`, calls Vec Q/K postprocess, then
sends `C_CONSUMED`. Vec runs D=128 dynamic quant inside `QkDynamicQuantD128Vf`
using `scale = max_abs / 448`, FP8 RINT/SAT cast, and a temporary scale
staging view over the reserve UB after V has finished using `v_scale_ub`.
Q rows write `q_out` and `q_scale`; K rows write `k_cache` and
`k_scale_cache` by `slot_mapping`. The Q/K output pool uses local
`V_MTE3`/`MTE3_V` event ids `2/3`, and CrossCore calls remain only in basic
block.
T11 keeps the rolling execution state inside basic block. For tile `i`, AIV
first preprocesses Q/K/V, sends `A_READY`, and runs V postprocess, then
consumes `C_READY` and Q/K-postprocesses the internally pending tile `i - 1`
when it exists. `End()` drains the final pending tile so AIC never waits
indefinitely on the last `C_CONSUMED`. AIC waits `C_CONSUMED` only before
reusing a full output DB window, then drains the remaining pending
consumptions in `End()`; it no longer serializes every tile immediately after
`C_READY`. The output DB pool lifecycle is also aligned with rolling reuse:
preprocess waits the Q/K output reusable event before using
`qk_after_cube_ub` as BF16 staging, and Q/K postprocess releases that event
after FP8 staging MTE3 writeout, including empty local slices.
T12 makes the tail/empty-slice contract explicit. `MakeAivSlice` may return an
empty slice, and that AIV still enters the same basic-block preprocess and
postprocess phases so `A_READY/C_READY/C_CONSUMED` counts remain fixed. Vec
preprocess consumes the reusable input and Q/K output local events, then skips
input copy, VF RMSNorm/RoPE, and AL1 ND2NZ when the slice has no rows. V and
Q/K postprocess already skip data work for empty slices; Q/K postprocess still
releases the Q/K output reusable event consumed during preprocess.
