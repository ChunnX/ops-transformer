# Work Log

## 2026-06-23 W4-only baseline convergence

- Converged the current implementation and spec from `W=2/3/4` back to `W=4` only.
- Updated `op_host/causal_conv1d_decode_tiling.cpp`:
  - `weight` shape validation now requires fixed `[4, dim]`.
  - Normal decode requires `state_len >= 3`; with `num_accepted_tokens`, `state_len >= seqLen + 2`.
- Updated `op_kernel/arch35/causal_conv1d_decode_cut_bh.h`:
  - `CopyWeightBias` always copies 4 weight rows.
  - `conv_states` always copies 3 history rows.
  - The single-S path calls only `Conv1dDecodeW4`; the multi-S path calls only `Conv1dDecodeW4MultiS`.
- Updated `op_kernel/arch35/vf/causal_conv1d_decode_width4_vf.h`:
  - Removed W2/W3 VF dead code and wrapper dispatch; kept only W4 single-S and W4 multi-S paths.
- Updated `tests/reference_causal_conv1d_decode.py`:
  - The reference accepts only `weight.shape[0] == 4`; smoke coverage is W4-only.
- Removed the stale `tests/ut/op_host/test_causal_conv1d_decode_tiling.cpp`, which was not wired into a CMake target and referenced obsolete local helper APIs. The current local lightweight validation is the Python reference smoke; CANN/op-host compilation still requires the target CANN environment.
- Synchronized current-scope wording in requirements, specs, dataflow, API docs, and the vLLM alignment audit. Older W2/W3 extension notes remain as history.

## 2026-06-22 Spec/dataflow wording convergence

- Updated section 5 in `reports/baseline_alignment_and_flow.zh.md`:
  - Replaced the old Mermaid overview with a plain-text dataflow: `GM tensors -> UB buffers -> Reg/VF -> GM outputs/side effects`.
  - Clarified that `[T,D]` is only the variable-length input storage form. Inter-core tiling does not use `B*S` or `B*T` as a split axis; the current main path remains `batch x dim`.
  - Clarified that `state_ub` contains only the `W-1` history-row window, while `weight_ub/bias_ub` are copied once per dimLoop and reused across batches.
  - Clarified that VF processes one 64-`d_local` chunk at a time and performs 64-dim vector multiply-adds for each window tap `k`.
- Added PNG:
  - `reports/images/causal_conv1d_decode_overall_flow_text.png`
- Synchronized the same wording into:
  - `reports/spec.zh.md` / `.md`
  - `specs/ascendc_operator_dataflow_spec.zh.md` / `.md`
  - `work/causal_conv1d_decode_bf16/specs/causal_conv1d_decode_bf16.md`
- Cleaned up several misleading `flatten` phrases by describing batch-group x movement as row copy/storage rows rather than an inter-core split axis.

## 2026-06-20 vLLM Ascend PR 8215 decode semantic audit

- Cloned `vllm-project/vllm-ascend` and checked out `pull/8215/head` locally at `external/vllm-ascend-pr8215`.
- Audited only the decode/update path in `csrc/causal_conv1d_v310` and `_310p/ops/fla/gdn_310.py`.
- Added `reports/vllm_ascend_pr8215_decode_audit.zh.md` / `.md`, documenting the three reference decode input modes:
  - `[batch, dim]`: non-MTP single-token decode.
  - `[batch, seqlen, dim]`: fixed-length MTP/speculative decode.
  - `[cuSeqlen, dim] + queryStartLoc`: variable-length flattened decode.
- Corrected `reports/spec.zh.md` / `.md` and `docs/REQUIREMENTS.zh.md` / `.en.md`:
  - Replaced the old `[B,S,D]` / `[T,D]` split with `[B,D]`, `[B,S,D]`, and `[T,D]+query_start_loc`.
  - Changed flattened token-count wording from `0..4` to the decode business range `1..4`, while keeping the note that host tiling cannot statically inspect runtime `query_start_loc` differences.
  - Marked `[B,D]` non-MTP decode as not fully implemented yet; this was fixed on 2026-06-21.
  - Marked the then-current `num_accepted_tokens` semantic gap where accepted count did not offset state-history loading. The history offset and W=4 speculative state writeback were both fixed on 2026-06-21.
- Clarified the risk in the current batch-group flattened-row copy: it can work when `query_start_loc` is monotonic, batch-group rows are continuous, token counts are legal, and compute reconstructs each batch's UB offset correctly; the reference is simpler because each task processes one batch segment.

## 2026-06-21 Decode alignment code fix

- Updated `op_host/causal_conv1d_decode_tiling.cpp`:
  - Optional input detection now treats empty shapes as absent.
  - 2D `x` without valid `query_start_loc` now uses `[B,D]` non-MTP decode with `seqLen=1`.
  - 2D `x` with valid `query_start_loc` still uses flattened `[T,D]` decode.
- Updated `op_kernel/arch35/causal_conv1d_decode_cut_bh.h`:
  - Added `GetHistoryBase(accepted)`.
  - When `num_accepted_tokens` is present, `accepted - 1` is clamped and used as the state-history window base, so accepted count affects `y`.
  - When `num_accepted_tokens` is absent, normal decode uses `conv_states[slot, 0:W-1]` as the persistent history window.
- Updated `tests/reference_causal_conv1d_decode.py`:
  - Added `[B,D]` non-MTP decode reference.
  - Added accepted-count history-window offset semantics.
  - Python smoke passed for width 2/3/4 across `[B,D]`, `[B,S,D]`, and `[T,D]+query_start_loc`.
- Follow-up:
  - Added the vLLM Ascend PR 8215 width=4 speculative state writeback path: keep the two history rows selected by `accepted-1`, then write current tokens from state row 2.
  - CANN compile validation is still pending.
- Tiling-route decision:
  - The current functional-alignment path stays with BH: split `batch x dim`, loop token/S inside the core.
  - BSH is not part of the current main path. It remains a future performance experiment for long-token, prefill/mixed, or shapes where `B x dimBlocks` is too small to feed the cores.

## 2026-06-21 multi-S VF optimization candidate

- Added and then split into `Conv1dDecodeW2/W3/W4MultiSVF` / `Conv1dDecodeMultiS`:
  - `tokenCount == 1` still uses the existing single-S VF path.
  - `tokenCount >= 2` uses one VF call with up to four FP32 accumulators to compute up to four token rows from the same batch.
  - The multi-S VF reuses the same weights and overlapping state/x timeline values, reducing repeated weight casts and repeated VF calls.
- Updated `causal_conv1d_decode_cut_bh.h` to build a `[history..., x0, x1, x2, x3]` timeline inside the batch loop and pass it to the multi-S VF path.
- Split the VF multiply-add path by W into `Conv1dDecodeW2/W3/W4` and `Conv1dDecodeW2/W3/W4MultiS`:
  - The kernel/AICore wrapper dispatches once on `kernelWidth`.
  - The `__simd_vf__` bodies no longer use `kernelWidth >= 3/4` to control active taps, avoiding W-branching inside the VF loop.
  - The S-axis `tokenCount >= 2/3/4` branches are still kept inside multi-S VF; whether to split S-specialized functions should be decided by CANN profiling.
- CANN compile and profiling are still pending.

## 2026-06-21 decode state W-1 movement fix

- Updated `causal_conv1d_decode_cut_bh.h`:
  - Changed the `stateQueueIn` UB buffer from `stateLen` rows to the maximum `W-1=3` history rows.
  - `Compute` now copies only the `W-1` history window for each batch:
    - without `num_accepted_tokens`, from `conv_states[slot, 0:W-1]`;
    - with `num_accepted_tokens`, from `clamp(accepted - 1, 0, stateLen - (W - 1))`.
  - `stateLocal[0..W-2]` is now the compact history window, so compute no longer indexes UB with full-state offsets.
  - Normal decode state update now writes back the newest `W-1` history rows instead of a full `state_len` sliding window.
  - W=4 speculative writeback keeps the vLLM-style layout: `state[0:2]` receives the last two rows from the compact history window, and `state[2:2+token_count]` receives the current token chunk.
- Updated `causal_conv1d_decode_tiling.cpp`:
  - UB estimation now accounts for at most 3 state-history rows.
  - Normal decode requires only `state_len >= W-1`; W=4 speculative layout still requires `state_len >= 2 + seqLen`.
- Updated the CPU reference so the default path writes only `conv_states[slot, 0:W-1]` with the newest `W-1` history rows.
- CANN compilation is still pending.

## 2026-06-21 dimLoop-outer weight/bias reuse

- Updated `causal_conv1d_decode_cut_bh.h`:
  - Changed the intra-core loop order from `batchLoop -> dimLoop` to `dimLoop -> batchLoop`.
  - Split the previous `CopyIn(x + weight + bias)` path:
    - `CopyWeightBias()` copies `weight[W, dimTile]` and optional `bias[dimTile]` once per dimLoop into `weightBiasBuf`.
    - `CopyX()` copies the current batch tile's x into `xQueue` once per batchLoop.
  - `Compute()` receives the current dimLoop's `weightBiasLocal`, reusing it across all batchLoop iterations.
  - State copy-in, y writeback, and conv_states update semantics are unchanged.
- Updated `docs/REQUIREMENTS.zh.md` / `.en.md`, `reports/spec.zh.md` / `.md`, `reports/tiling_strategy.zh.md` / `.md`, and `specs/ascendc_operator_dataflow_spec.zh.md` / `.md`:
  - Recorded the `dimLoop -> batchLoop` loop order and `weightBiasBuf` lifetime.
- Python reference smoke still passes; CANN compile and profiling are still pending.

## 2026-06-21 baseline alignment report

- Added `reports/baseline_alignment_and_flow.zh.md`:
  - Clarifies that functional semantics align with vLLM Ascend PR 8215 decode/update, while interface and A5 dataflow align with ops-transformer.
  - Records intentional differences: BF16, INT32 metadata, decode-only scope, no `initialStateMode/runMode`, and v1 `activation_mode=0` only.
  - Adds flowcharts for overall execution, inter-/intra-core tiling, and GM->UB->VF->GM dataflow.
  - Adds a complete variable-length `[T,D]+query_start_loc`, `W=4`, `state_len=6` case.
- Corrected stale wording in early `reports/summary.zh.md` and the vLLM audit that no longer reflected the current baseline.

## 2026-06-17 Documentation diagrams

- Added tiling, GM/UB movement continuity, `conv_states` slot movement, state writeback, and VF 64-lane + tail-mask diagrams to `docs/REQUIREMENTS.zh.wiki` / `docs/REQUIREMENTS.en.wiki`.
- Added standard Markdown versions `docs/REQUIREMENTS.zh.md` / `docs/REQUIREMENTS.en.md` so Mermaid, Markdown tables, and ASCII layout diagrams render correctly outside wiki syntax.
- Added `ascendc-docs-gen`-structured requirement documents `docs/REQUIREMENTS.ascendc-docs-gen.zh.md` / `.en.md`, with detailed compute semantics, tiling, and memory layout in appendices.
- Rewrote `specs/ascendc_operator_dataflow_spec.zh.md` / `.md` with Mermaid and ASCII memory-layout diagrams for the full GM -> UB -> VF/register -> UB -> GM path.
- Added 2D core split, intra-core UB tile, GM row-contiguous/row-strided movement, VF split, and state-update diagrams to `reports/tiling_strategy.zh.md` / `.md`.
- Added summary diagrams to `reports/spec.zh.md` / `.md` so the VF inner loop is visible even when only the compact spec is read.
- Corrected stale dataflow/tiling wording: current width is 2..4, `conv_states` is an in-place side effect, and SiLU is not a current v1 tuning item.

## 2026-06-14

- User requested an AscendC-style `causal_conv1d` decode operator modeled after `cann/ops-transformer`, especially A5-style operators.
- Retrieved public GitCode pages for `attention/fused_causal_conv1d`.
- Could not clone the repository because approval for `git clone` timed out twice.
- Used public README/docs HTML instead.
- Extracted interface and constraints:
  - upstream supports prefill, mixed, variable decode, fixed-batch decode
  - upstream `run_mode` is legacy and unsupported
  - upstream decode fixed-batch shape is `[batch, m + 1, dim]`
  - upstream variable decode supports token count `[1, 8]`
  - upstream `weight` is `[K, dim]`, with `K=3`
- Chose new op scope:
  - decode only
  - no prefill
  - no mixed path
  - no APC in v1
  - BF16
  - width 4
  - seq_len 1..4 for MTP
- Wrote spec, reference notes, tiling design, host skeleton, and kernel skeleton.
# 2026-06-15 Remote Migration and BF16 vf Update

- Migrated `attention/causal_conv1d_decode` to the new server path `/home/ops-transformer-master`.
- Replaced the early scalar GM-loop kernel with the A5-style BH structure: `causal_conv1d_decode_apt.cpp`, `arch35/causal_conv1d_decode_cut_bh.h`, and `arch35/vf/causal_conv1d_decode_width4_vf.h`.
- Made the vf entry explicitly BF16-only. Inputs and outputs are `bfloat16_t`; FP32 is used only as the accumulator register type inside vf.
- Added the AscendC GM/inter-core/UB/vf dataflow spec in both English and Chinese.
- Remote reference script passed. CANN build was not completed because the GPU image has no Ascend toolkit path configured.

# 2026-06-16 VF/SIMT API and data-movement follow-up

- Added the local `ascendc-vf-simt-api` skill with English and Chinese references for VF, Reg, SIMT, DataCopyPad, TQue, and Duplicate API investigation.
- Clarified the kernel data-movement model: `TQue::EnQue/DeQue` is the queue-managed synchronization boundary for staged UB tensors, while persistent metadata in `TBuf` still uses explicit `SetFlag/WaitFlag`.
- At that time, the combined x/weight UB queue plus `stateQueueIn` and `yQueue` were changed to single-buffer queues. On 2026-06-21 the combined x/weight queue was further split into `xQueue + weightBiasBuf` to support dimLoop-outer weight/bias reuse.
- Added concise code comments for strided `DataCopyPad` use, optional-input default broadcasting through `Duplicate`, bias logical broadcast, state copy-in, y writeback, and state-window update.
- Reference script still passes. CANN compile remains unavailable on the GPU image because no Ascend toolkit path is configured.

# 2026-06-17 vLLM update/decode alignment

- Realigned scope with the user's decision for vLLM Ascend `CausalConv1d` update/decode compatibility:
  - Do not add `initialStateMode` or `runMode`; they are treated as prefill/mode-switching fields outside this decode-only operator.
  - Keep metadata tensors as INT32 for this framework. This is not ABI-identical to vLLM's OpDef, but functional equivalence can be reached through a wrapper/cast layer.
  - Added optional `query_start_loc` to support flattened update input `x: [total_tokens, dim]`.
  - Made `cache_indices` optional; default is `slot = batch_id`.
  - Kept `num_accepted_tokens` optional; default is `accepted = token_count`.
  - Changed `conv_states` to in-place state and exposed only `y` as the public output.
- Extended width from fixed 4 to `weight.shape[0] in [2, 4]`:
  - Host tiling now records `kernelWidth`.
  - Kernel selects the causal window from `kernelWidth`.
  - The VF path is now split into fixed W=2/3/4 functions, so width 2/3 does not read absent weight rows and does not branch on W inside VF.
- Updated and ran the CPU reference:
  - fixed-batch `[B,S,D]`
  - flattened `[T,D] + query_start_loc`
  - width 2, 3, and 4
- Remaining gaps:
  - CANN compilation is still unavailable on the current GPU server because Ascend toolkit is not installed.
  - `activation_mode=1` / SiLU is still not implemented in VF; host tiling still restricts `activation_mode=0`.
