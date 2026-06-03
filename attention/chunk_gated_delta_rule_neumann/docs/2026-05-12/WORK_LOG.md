# 2026-05-12 Work Log

## Revalidation Before Continuing

- Active selector confirmed:
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- Active selector values:
  - suite `p3l1_p8a`
  - variant `C64_1616_FP16`
  - chunk size `64`
- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- Active Stage1 SHA256:
  - `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`
- A41 baseline SHA256:
  - `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`

Conclusion:

- The current active source is byte-identical to the A41 clean correctness baseline.
- It is not the earlier A42 diagnostic candidate described in stale `todo.md` text.
- Continue only from this clean A41 surface.

## A42/A43 Boundary Notes

A42 P7 diagonal-only L0C->L1 Fixpipe copy-out:

- Multiple candidate shapes timed out previously, including exact diagonal, two `32x32` quadrants, sequencing variants, and top-left partial final forms.
- The failure surface is `Fixpipe<..., CFG_NZ>` / L0C->L1 copy-out shape support, not P7/P8 math.
- Keep A42 pending/deferred; do not mark it as PASS.
- Do not stack later phases on any A42 diagnostic source.

A43 P7 L0A no-transpose diagonal load:

- A40 already isolated that L0B transpose one-shot coalesced load is safe.
- A40 also isolated that L0A no-transpose one-shot `repeatTimes=4` load fails at `S=1024`.
- Keep current A41 two paired L0A loads unless a new legal L0A layout/load direction is proven.

## A44 Start

Planned single point:

- Make the `identity64` lifecycle explicit as a raw-MMAD inverse epoch cache.
- Preserve the proven invalidation boundaries:
  - `Init()`
  - `InitRawMmadLocalBuffers()`
  - `RestoreStage1MatmulLocalBuffers()`
  - the current `ParaChunkAIC()` inverse-window boundary
- Do not repeat A30's simple cross-`ParaChunkAIC()` prepared-flag extension.
- Do not modify P7/P8 formulas, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, or final RHS math.

## A44 Validation

Active candidate:

- Active Stage1 SHA256:
  - `d5f12a0737537235f964f9ef8f15515c0fbb5ae2b2bf8b9388f5bc22965249c9`
- Package installed:
  - `../../build_out/cann-ops-transformer-cgdrverify_linux-aarch64.run`
  - package SHA256 `07f1083dd94f763897813d2c51a7a52290797b4146055a2b29752b9ae94052ca`

Implementation boundary:

- `EnsurePacked64IdentityForInverseEpoch()` is called after `PreloadP8LowerLeftBlocksToLowerWork64NoWaitFull64(paraIdx)` and before `P7Packed64DiagonalNeumannOneShotToResult()`.
- `P7Packed64DiagonalNeumannOneShotToResult()` does not call `EnsurePacked64Identity()` internally.
- Invalidation remains at:
  - `Init()`
  - `InitRawMmadLocalBuffers()`
  - `RestoreStage1MatmulLocalBuffers()`
  - current `ParaChunkAIC()` inverse-window boundary
- This is an epoch-safe correctness baseline. It is not yet the final "operator lifecycle generates identity64 only once" design.

Earlier failed A44 candidates not to repeat:

- `db993914...`: pre-generate identity before the inverse loop and remove P7-side ensure; `S=64/256` PASS, `S=1024` FAIL for out/state.
- `19d4ccf...`: pre-generate plus first-use sync; `S=64/256` PASS, `S=1024` FAIL for out while state PASS.

Correctness gate:

- install PASS.
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.21s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA44_identity_epoch_after_preload_baseline.h`
- SHA256:
  - `d5f12a0737537235f964f9ef8f15515c0fbb5ae2b2bf8b9388f5bc22965249c9`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA44_identity_epoch_after_preload_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA44_identity_epoch_after_preload_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.315`
  - `S=512`: `1.971`
  - `S=2048`: `2.824`
  - `S=10240`: `2.860`

## A45 Direct L0C Copy-out Failed Candidate

Candidate boundary:

- Goal: P8 first-layer packed32 output copy-out merge.
- Active Stage1 SHA256:
  - `55422221d4be3edb9722a0e7225cc98724f53961d52fde692825a82dcff863cb`
- Change:
  - added `CopyL0CSubMatrixToPackedHalfL1()`
  - replaced second MMAD full `32x32` writeback plus two `CopyHalfL1PackedBlockToCompact()` calls with two direct compact32 L0C submatrix Fixpipe writes into result64:
    - L0C `(0,0)` -> result64 `(16,0)`
    - L0C `(16,16)` -> result64 `(48,32)`

Gate result:

- build PASS.
- install PASS.
- `S=64` timeout after 90s, exit code `124`.
- `S=256/1024` not run.
- No baseline saved.

Failure judgment:

- High suspicion is `Fixpipe<..., CFG_NZ>` with non-zero compact32 L0C source offset / partial copy-out, consistent with the A42 L0C->L1 partial copy-out timeout surface.
- Do not stack later changes on SHA `55422221...`.

## A45a Start

Candidate boundary:

- Active Stage1 SHA256:
  - `424fa16e0f2b0a99435076fef184bb4781707e73b24972b7bc32854eb32ca6b2`
- Rebuild from the A44 clean surface and avoid non-zero L0C source offset.
- Keep the second MMAD full `32x32` L0C->L1 writeback.
- Replace only the two post-writeback `CopyHalfL1PackedBlockToCompact()` calls with a packed L1 sub-block direct copy:
  - packed32 `(0,0)` -> result64 `(16,0)`
  - packed32 `(16,16)` -> result64 `(48,32)`
- This removes two `X @ I16` layout-copy MMADs while leaving the direct L0C copy-out problem deferred.

Validation status:

- build PASS.
- install PASS.
- `S=64` PASS.
- `S=256` PASS.
- `S=1024` FAIL:
  - out/state actual are near zero.
- No baseline saved.

## A45a2 Failed Candidate

Candidate boundary:

- Active Stage1 SHA256:
  - `43d60696d792c9819cfa3971708ffa9543844f3909f6dad27047fa8011a6d61c`
- Reuses packed64 scratch as the source for the packed L1 sub-block copy.

Gate result:

- build PASS.
- install PASS.
- `S=1024` FAIL with the same near-zero out/state pattern as A45a.
- No baseline saved.

Failure judgment:

- `GetValue` / `SetValue` scalar L1 sub-block copy is not reliable enough for the active path.

## A45a3 Failed Candidate

Candidate boundary:

- Active Stage1 SHA256:
  - `e42e13abbacc8103ddee99fb3e1e24c408e84aa6b6aa3f8abbfacc643963baac`
- Package SHA256:
  - `c5a2f6db6ba50f86dbec3f90dd634c24fac9cffdbd3aa84e63d90470477d632e`
- Change:
  - add `CopyL0CTopLeft16ToPackedHalfL1()`
  - after the P8 first-layer second MMAD, directly Fixpipe the L0C top-left `16x16` to result64 `(16,0)` with a non-final unit flag
  - keep the full `32x32` L0C->L1 writeback
  - keep the old second-lane `CopyHalfL1PackedBlockToCompact()` for `(16,16) -> (48,32)`

Gate result:

- build PASS.
- install PASS.
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare FAIL:
  - `15 passed, 1 failed`
  - failing case `b2_t256_h32_h8_d64`
  - CPU vs ChunkGDRNeumann `out` max_abs `1.160285e-01`
  - worst actual `3.01748514e-07`, expect `1.16028756e-01`
  - `state` PASS
- No baseline saved.

Failure judgment:

- A45a3 avoided the A45a/A45a2 `S=1024` near-zero failure, but still fails full compare under a higher batch/head case.
- The only changed output lane is result64 `(16,0)` from a direct L0C top-left copy-out; the likely issue is Fixpipe visibility/cache ordering for that non-final direct write into result64, not P8 formula order.
- Restore active source to A44 before any next A45 candidate.

## A46b Validation

Candidate boundary:

- Goal: P8 lower-left first-layer GM->lowerWork64 preload merge.
- Active Stage1 SHA256:
  - `c8079697775d384bb5fe7c293b2e9279665c48f3be7be1d2c500574354e8f18d`
- Change:
  - merge only `(16,0)` and `(48,32)` first-layer `16x16` lower-left blocks with one `Nd2NzParams DataCopy`
  - keep `(32,0) 32x32` second-layer input as a separate preload
  - keep P8 formula as `tmp = lower @ topInv`, `out = bottomInv @ tmp`

Initial A46 failure:

- Pair preload followed immediately by `(32,0)` no-wait preload passed `S=64/256` but failed `S=1024`.
- Symptom: `out` near zero while `state` still passed.

A46b fix:

- Add `SyncMte2ToMte1()` after `LoadP8FirstLayerLowerLeftPairToLowerWork64NoWait(paraIdx)` and before the separate `(32,0)` preload.
- This keeps the pair preload and second-layer preload ordered without changing P8 math.

Gate result:

- build PASS.
- install PASS.
- `S=64` PASS.
- `S=256` PASS.
- `S=1024` PASS.
- full compare PASS: `16 passed in 6.56s`.

Artifacts:

- isolated package: `/tmp/cgdr_a46b_20260512_103555.run`
- package SHA256: `fda7b80be692b2510ca3df7dbd7f039877c179a5a6ca3181e9cf98cb3ff18773`
- isolated OPP root: `/tmp/cgdr_a46b_opp_20260512_103555/vendors/custom_transformer`
- installed Neumann object SHA256: `31c71f307df98a44e6df979976c8ebda5cc3fd6a82597b3cab5907ae1df7c137`
- installed sanity: only `aclnnChunkGatedDeltaRule*` / `aclnnChunkGatedDeltaRuleNeumann*`, no `aclnnTriInverse`.

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA46_p8_first_layer_lower_left_pair_preload_sync_baseline.h`
- SHA256:
  - `c8079697775d384bb5fe7c293b2e9279665c48f3be7be1d2c500574354e8f18d`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA46_p8_first_layer_lower_left_pair_preload_sync_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA46_p8_first_layer_lower_left_pair_preload_sync_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.318`
  - `S=512`: `1.976`
  - `S=2048`: `2.778`
  - `S=10240`: `2.840`

## A47 Start

Planned single point:

- Keep `P8Compose64From32Direct32()` as packed `32x32` second-layer compose.
- Keep lower-triangular formula:
  - `tmp = lowerWork64[(32,0)] @ result64[(0,0)]`
  - `out = result64[(32,32)] @ tmp`
- First A47 candidate only specializes the second-layer `32x32` L1->L0 extraction helper to fixed packed offsets.
- Do not change tmp L1 lifetime, final result64 writeback, P7, BF16/resident writeback, Stage2/Stage3/QK, `gBK/vBeta`, or final RHS semantics.

## A47 Validation

Candidate boundary:

- Goal: P8 second-layer packed-MMAD load specialization.
- Active Stage1 SHA256:
  - `46ef5fc3b9a9663a8b4c55f13abf1401934153d368d04c165a279d5070472bd2`
- Change:
  - add fixed `32x32` packed L1->L0 load helpers for the two MMADs inside `P8Compose64From32Direct32()`
  - keep tmp L1 writeback and final result64 writeback unchanged
  - keep formula unchanged:
    - `tmp = lowerWork64[(32,0)] @ result64[(0,0)]`
    - `out = result64[(32,32)] @ tmp`

Artifacts:

- isolated package: `/tmp/cgdr_a47_20260512_105036.run`
- package SHA256: `ebc908b67cd83a3576ce4238d6945d47da381af9e0ac50b1483278d4ad5751bb`
- isolated OPP root: `/tmp/cgdr_a47_opp_20260512_105036/vendors/custom_transformer`
- installed Neumann object SHA256: `9df044696fc5c1fedf1035a5eec9047708b600a45bc96e66569fa68278affe48`
- installed sanity: only `aclnnChunkGatedDeltaRule*` / `aclnnChunkGatedDeltaRuleNeumann*`, no `aclnnTriInverse`.

Gate result:

- build PASS.
- install PASS.
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.18s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA47_p8_second_layer_fixed_load_baseline.h`
- SHA256:
  - `46ef5fc3b9a9663a8b4c55f13abf1401934153d368d04c165a279d5070472bd2`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA47_p8_second_layer_fixed_load_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA47_p8_second_layer_fixed_load_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.368`
  - `S=512`: `1.995`
  - `S=2048`: `2.797`
  - `S=10240`: `2.807`

Comparison with A46:

- A46 ratios were `1.318 / 1.976 / 2.778 / 2.840`.
- A47 slightly worsens small/medium ratios, but improves the largest profiled case from `2.840` to `2.807`.
- Keep A47 as the latest correctness-safe baseline because the phase is target-path PASS and preserves packed `32x32` second-layer semantics; the performance tradeoff is recorded for A48+ prioritization.

## A45b Failed Candidate

Candidate boundary:

- Active Stage1 SHA256:
  - `36e201ac34295257a5a8918982d337bfa5bd3e04128fcb0e386244b88f4fc17e`
- Correct CGDR package SHA256:
  - `a6d8ed0a182cf7336a4fda007facd74bf9029dc84f39a2e9615de9291e7a61c9`
- Installed `libcust_opapi.so` SHA256:
  - `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
- Package sanity:
  - `ASCEND_OP_NAME=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann`
  - `libcust_opapi.so` exports `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`
- Change:
  - same direct L0C top-left `16x16` result64 `(16,0)` copy-out idea as A45a3
  - uses `RAW_MMAD_FINAL_UNIT_FLAG` plus immediate `SyncFixToMte1()`
  - keeps full `32x32` L0C->L1 writeback and second-lane old `CopyHalfL1PackedBlockToCompact()`

Gate result:

- build PASS.
- install PASS.
- `S=1024` timeout after 120s, exit code `124`.
- No residual NPU process after timeout.
- No baseline saved.

Failure judgment:

- A45b confirms the top-left direct L0C->result64 partial Fixpipe path is not a safe active optimization, even with final flag and explicit sync.
- Continue A45 only from A44 clean source, and avoid direct L0C partial copy-out for the next candidate.

## A45c Failed Candidate

Candidate boundary:

- Active Stage1 SHA256:
  - `7bd77182344d14e6787b14079ff257a12c275c46d19bf279483dbbc99f7bcb37`
- Change:
  - add a VECCALC UB copy buffer sized for two NZ fractals.
  - copy packed32 output diagonal lanes from `aicHalfL1TilePower_` through `DataCopy` L1->UB->L1 into result64 `(16,0)` and `(48,32)`.
  - replace only the two post-writeback `CopyHalfL1PackedBlockToCompact()` calls in `P8ComposeFirstLayerPacked32Direct()`.
  - do not touch P7 one-shot, P8 formulas, P8 second-layer, BF16/resident/RHS, Stage2/Stage3/QK, or `gBK/vBeta`.

Build/install:

- build PASS.
- Build output package SHA256:
  - `42bb850654d1a0b21e5fd8ef37a9e1615c7ac0db0521c3a80bf7e7c86266d0eb`
- Before install, a concurrent `tri_inverse` process was found installing/testing against the same `vendors/custom_transformer` directory. Waited for it to finish, then installed the CGDR package.
- Installed `libcust_opapi.so` SHA256:
  - `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
- Installed sanity:
  - exports `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
  - no `aclnnTriInverse` export.
  - installed kernel directory contains `chunk_gated_delta_rule` and `chunk_gated_delta_rule_neumann`.

Gate result:

- `S=64` timeout after 90s, exit code `124`.
- `S=256/1024` not run.
- No baseline saved.

Failure judgment:

- A45c rules out the attempted L1->UB->L1 packed32 two-lane copy-out path.
- Current A45 evidence now covers:
  - direct L0C partial copy-out timeout/failure.
  - scalar L1 sub-block copy `S=1024` near-zero failure.
  - L1->UB->L1 `S=64` timeout.
- A45 is deferred to the later copy-out专项. Restore active source to A44 before continuing A46.

Restore:

- Active Stage1 restored to A44 baseline.
- Active Stage1 SHA256:
  - `d5f12a0737537235f964f9ef8f15515c0fbb5ae2b2bf8b9388f5bc22965249c9`

## tri_inverse vs chunkGDN 求逆差距文档

Added:

- `docs/2026-05-12/TRI_INVERSE_VS_CHUNKGDN_INVERSE_GAP.md`

Purpose:

- Record the pure-inverse comparison boundary between standalone `tri_inverse` and chunkGDN Stage1.
- Separate pure inverse costs from full chunkGDN extra consumers such as resident L1 exposure and later RHS.
- List optimization points already present in `tri_inverse` but not fully landed in chunkGDN:
  - lowerWork one-time load / reuse
  - P8 first-layer output placement
  - P7 L0C copy-out reduction
  - keep chunkGDN P7 as four `16x16` diagonal fractals packed64 together; borrow only the reduced-roundtrip principle, not `tri_inverse` Phase7's two-`32x32` diagonal inverse grain
  - independent persistent identity64 L1 buffer, not another prepared-flag extension
  - P8 final output direct-to-resident
  - BF16 GM inverse materialization row-prefix cleanup
  - remaining AIV/scratch movement reduction
- Record that `FusedRhsAICProcess()` must not be used to explain pure inverse performance gaps; it belongs to full-op latency only.

## identity64 lifecycle correction

User clarification:

- An identity64 constant should not require repeated reconstruction.
- The correct design target is to allocate one L1 buffer that persistently stores identity64 and is only read by P7/P8.

Updated rule:

- Do not treat the next identity64 lifecycle phase as another prepared-flag extension.
- Current `aicHalfL1Identity64_` is a tile inside `aicHalfL1Buff_`, which is initialized under the raw-MMAD local buffer path.
- `InitRawMmadLocalBuffers()` and `RestoreStage1MatmulLocalBuffers()` call `pipe_->Reset()`, and `ParaChunkAIC()` explicitly clears `packed64IdentityPrepared_`; therefore the current implementation is not a true persistent identity64 buffer.
- The next identity64 cleanup must design an independent persistent L1 buffer:
  - not reused as scratch/resident/BF16 materialization workspace
  - not overwritten by `pipe_->Reset()` / `InitBuffer()` rebinding, or explicitly split out from that reset domain
  - generated once per AIC core, since L1 is core-local
  - read-only for later P7/P8
- Replace vague "side effect" wording with concrete synchronization/visibility boundaries in future plans.

## A48 Validation

Candidate boundary:

- Goal: BF16 upper-zero materialization row-suffix merge.
- Active Stage1 SHA256:
  - `5c6eb8a8b8e2431b23b69e37be56959b364fb36103e3a6201a39eb49ca557b73`
- Change:
  - add `CopyZeroHalfL1RectToGmBf16ByMmad()`.
  - keep lower 10 true inverse blocks as per-`16x16` BF16 writeback.
  - merge upper 6 zero blocks into three row-suffix BF16 zero rectangles.
  - keep resident/RHS, P7, P8 formulas, Stage2/Stage3/QK, and `gBK/vBeta` unchanged.
- Important correction:
  - the row-suffix zero helper keeps `PrepareHalfL1IdentityByMmad(INVERSE_SHAPE)` after `EnsureHalfL1ZeroForMte1(...)`.
  - The first A48 candidate without this boundary passed `S=64/256` but failed `S=1024` with near-zero out/state, so this MMAD/Fixpipe visibility boundary is required for the active path.

Artifacts:

- isolated package: `/tmp/cgdr_a48_fix_20260512_114843.run`
- package SHA256: `5a671b529d2a60dc28414f69336e3837d5b86e9c4d3d5f940260baf13c129678`
- isolated OPP root: `/tmp/cgdr_a48_fix_opp_20260512_114843/vendors/custom_transformer`
- installed Neumann object SHA256: `b34f469c5d1de3ae96e4c0b9d6927ab2344af0e3305a56908e18b831fae23d0a`
- installed Stage1 header SHA256:
  - `5c6eb8a8b8e2431b23b69e37be56959b364fb36103e3a6201a39eb49ca557b73`
- installed sanity:
  - exports only `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
  - no `aclnnTriInverse`.

Gate result:

- build PASS.
- install PASS.
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.27s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA48_bf16_upper_zero_rowsuffix_baseline.h`
- SHA256:
  - `5c6eb8a8b8e2431b23b69e37be56959b364fb36103e3a6201a39eb49ca557b73`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA48_bf16_upper_zero_rowsuffix_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA48_bf16_upper_zero_rowsuffix_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.278`
  - `S=512`: `1.960`
  - `S=2048`: `2.745`
  - `S=10240`: `2.778`

Comparison with A47:

- A47 ratios were `1.368 / 1.995 / 2.797 / 2.807`.
- A48 improves all four profiled sizes.
- A48 is the latest correctness-safe baseline.

## A49 BF16 Lower Row-Prefix PASS But Performance Regressed

Candidate boundary:

- Goal: merge BF16 true-inverse lower writeback row-prefix for `cols<=48` only.
- Active Stage1 SHA256:
  - `337b86df5435578536c3b95ce38423aaf4f0592e223456a8f8a9382e2743d149`
- A49 baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA49_bf16_lower_rowprefix48_baseline.h`
  - SHA256 `337b86df5435578536c3b95ce38423aaf4f0592e223456a8f8a9382e2743d149`

Implementation:

- `CopyPackedHalfL1RowPrefixToGmBf16()` added for row-prefix BF16 materialization.
- `StorePackedL1InverseToBf16Attn()` uses row-prefix only for lower columns `16/32/48`.
- `rowBase=48, cols=64` stays on old per-`16x16` writeback.
- P7/P8 formulas, resident/RHS, Stage2/Stage3/QK, and `gBK/vBeta` are unchanged.

Validation:

- build PASS after A49 package creation.
- isolated package: `/tmp/cgdr_a49_prefix48_20260512_122205.run`
- package SHA256: `6c573a086cfd862cf25e83fa630bb03a7d986b17b44d89c0a53bd79f4dc6ca9d`
- isolated OPP root: `/tmp/cgdr_a49_prefix48_opp_20260512_122205/vendors/custom_transformer`
- installed source SHA256: `337b86df5435578536c3b95ce38423aaf4f0592e223456a8f8a9382e2743d149`
- installed object SHA256: `5b43d1e8a1af1d886173a7b4fd279132d81b6f3c9f3dd6cd5f75732fb71ac79d`
- installed op_api SHA256: `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
- installed symbols export only `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`; no `aclnnTriInverse`.
- `S=64/256/1024` smoke PASS.
- full compare PASS: `16 passed in 6.21s`.

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA49_bf16_lower_rowprefix_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA49_bf16_lower_rowprefix_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `2.196`
  - `S=512`: `4.441`
  - `S=2048`: `5.804`
  - `S=10240`: `5.900`

Failure and performance boundary:

- Full lower row-prefix `cols=64` failed at `S=1024` and stays out of A49.
- A49 `cols<=48` is correctness-safe but performance-regressed.
- Root cause from profile/code shape: `CopyPackedHalfL1RowPrefixToGmBf16()` calls `PrepareHalfL1IdentityByMmad(cols)`, so `I32/I48` are rebuilt through scalar GM identity write + MMAD load; scalar time grows sharply.

## A50 BF16 Row-Prefix Identity64 Reuse

Candidate boundary:

- Goal: fix A49's row-prefix identity generation overhead without changing BF16 row-prefix output range.
- Active Stage1 SHA256:
  - `4c995b59523e8e87069da1a38692567d0eee99ac9a8a30ee1cec8f83d6f6208a`
- Change:
  - `CopyPackedHalfL1RowPrefixToGmBf16()` now uses `EnsurePacked64Identity()` and `aicHalfL1Identity64_` with `chunkSize_` RHS packed stride.
  - Removed the A49 `PrepareHalfL1IdentityByMmad(cols)` and post-copy `PrepareHalfL1IdentityByMmad(INVERSE_SHAPE)` calls in this helper only.
  - P7/P8 formulas, resident/RHS, Stage2/Stage3/QK, and `gBK/vBeta` are unchanged.

Build/install artifacts:

- First build attempt failed in FATBIN because generated `kernel_meta/..._mix_aic_0.o` / `_mix_aiv_0.o` files were missing after a temporary rename failure; no C++ compile error was present.
- After clearing the stale generated kernel_meta directory, rebuild PASS.
- isolated package: `/tmp/cgdr_a50_identity64_rowprefix_20260512_124019.run`
- package SHA256: `0b7308bff932c3c2eb5cb932fee76654b7386fcefc9a2a8cff800edd5967dbf1`
- isolated OPP root: `/tmp/cgdr_a50_identity64_rowprefix_opp_20260512_124019/vendors/custom_transformer`
- installed source SHA256: `4c995b59523e8e87069da1a38692567d0eee99ac9a8a30ee1cec8f83d6f6208a`
- installed object SHA256: `b26efc75e743a2f161d260b7abd03701da39814bb5a9ae75ef30d583c9f98d12`
- installed op_api SHA256: `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
- installed symbols export only `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`; no `aclnnTriInverse`.

Correctness gate:

- `S=64` PASS.
- `S=256` PASS.
- `S=1024` PASS.
- full compare PASS: `16 passed in 6.16s`.

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA50_bf16_rowprefix_identity64_baseline.h`
- SHA256:
  - `4c995b59523e8e87069da1a38692567d0eee99ac9a8a30ee1cec8f83d6f6208a`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA50_bf16_rowprefix_identity64_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA50_bf16_rowprefix_identity64_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.275`
  - `S=512`: `1.965`
  - `S=2048`: `2.768`
  - `S=10240`: `2.804`

Conclusion:

- A50 fixes A49's severe performance regression.
- A50 is still slightly worse than A48 at `S=512/2048/10240`; A48 ratios were `1.278 / 1.960 / 2.745 / 2.778`.
- Treat A50 as a correctness-safe baseline and evidence that row-prefix can reuse identity64, but do not use A49/A50 as the default performance branch for the next unrelated optimization. Restore A48 before A51 unless the next phase explicitly targets A50 long-sequence regression.

## A51 Identity64 Block Writeback Attempts

Question answered before implementation:

- A50 fixed A49's row-prefix-specific identity rebuild overhead.
- Applying that fix directly to A48 is not automatically beneficial because A48 still uses per-`16x16` block writeback, not A49/A50 row-prefix writeback.
- A51 was therefore scoped to A48's existing `CopyPackedHalfL1BlockToGmBf16()` block writeback path only.

A51 initial failed candidate:

- Active Stage1 SHA256:
  - `27a94981ec45dce4c3cd12236364a785e59b380e2141cc9ad69b474e65c82b69`
- Change:
  - replace `PrepareHalfL1IdentityByMmad(blockSize)` with `EnsurePacked64Identity()`.
  - use `aicHalfL1Identity64_` and `chunkSize_` RHS packed stride for all block writebacks.
- Gate:
  - build/install PASS.
  - `S=64` PASS.
  - `S=256` PASS.
  - `S=1024` FAIL: out/state both fail, actual near zero.
- No baseline saved.

A51a failed candidate:

- Active Stage1 SHA256:
  - `09e7a24a955263e9464aa6008c5aed92d20b76e1d6cde24f18b1e5e91299586f`
- Change:
  - add `SyncFixToMte1(); SyncMte2ToMte1();` after `EnsurePacked64Identity()` in the block writeback helper.
- Gate:
  - build/install PASS.
  - `S=64` PASS.
  - `S=256` PASS.
  - `S=1024` FAIL: out fails, state passes.
- Conclusion:
  - sync fixes part of the visibility issue, but full block-writeback identity64 reuse is still unsafe.
  - Do not reuse A51/A51a as a baseline.

## A51b BF16 Block Writeback Partial Identity64 PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `50c079ec6ee5dc32deb7f353c038e71ddd131716845d98d85605ac18950fdd24`
- Change:
  - `CopyPackedHalfL1BlockToGmBf16()` uses `aicHalfL1Identity64_` only for rows `0/16/32`.
  - row `48` keeps the A48 old `PrepareHalfL1IdentityByMmad(blockSize)` / `aicHalfL1Identity_` path.
  - P7/P8 formulas, resident/RHS, Stage2/Stage3/QK, and `gBK/vBeta` are unchanged.

Build/install artifacts:

- First build attempt exited with code 2 in a transient third-party/protobuf build stage; retry from the same source passed.
- isolated package:
  - `/tmp/cgdr_a51b_identity64_blockwrite_partial_20260512_131120.run`
- package SHA256:
  - `c958109eedf875798f3fe0624369d690bae3394875f0d595a06161c2e21ac6e9`
- isolated OPP root:
  - `/tmp/cgdr_a51b_identity64_blockwrite_partial_20260512_131120_opp/vendors/custom_transformer`
- installed active Stage1 header SHA256:
  - `50c079ec6ee5dc32deb7f353c038e71ddd131716845d98d85605ac18950fdd24`
- installed `libcust_opapi.so` SHA256:
  - `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
- installed sanity:
  - exports only `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
  - no `aclnnTriInverse`.

Correctness gate:

- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.20s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA51b_bf16_blockwrite_partial_identity64_baseline.h`
- SHA256:
  - `50c079ec6ee5dc32deb7f353c038e71ddd131716845d98d85605ac18950fdd24`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA51b_identity64_blockwrite_partial_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA51b_identity64_blockwrite_partial_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.317`
  - `S=512`: `1.935`
  - `S=2048`: `2.743`
  - `S=10240`: `2.792`

Comparison with A48/A50:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A50 ratios: `1.275 / 1.965 / 2.768 / 2.804`.
- A51b improves `S=512/2048` slightly versus A48, but regresses `S=64/10240`.
- Treat A51b as latest correctness-safe baseline, not as the preferred performance baseline. A48 remains the preferred performance rollback surface.

## A52 RestoreRawMmadState HF32 Single-Field Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `30eda4660a3bb8fcf9475a7dc532211b7552784c0fd5acc6bed14ece727ab3cd`
- Start point:
  - restored from A48 preferred performance baseline SHA256 `5c6eb8a8b8e2431b23b69e37be56959b364fb36103e3a6201a39eb49ca557b73`.
- Change:
  - removed only `SetHF32Mode(HF32Mode::DISABLE);` from `RestoreRawMmadState()`.
  - all other raw-MMAD state restore fields remain unchanged.
  - P7/P8 formulas, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- isolated package:
  - `/tmp/cgdr_a52_restore_hf32_singlefield_20260512_132350.run`
- isolated OPP root:
  - `/tmp/cgdr_a52_restore_hf32_singlefield_20260512_132350_opp/vendors/custom_transformer`
- installed active Stage1 header SHA256:
  - `30eda4660a3bb8fcf9475a7dc532211b7552784c0fd5acc6bed14ece727ab3cd`

Correctness gate:

- build PASS.
- install PASS.
- `S=64` PASS.
- `S=256` PASS.
- `S=1024` PASS.
- full compare PASS:
  - `16 passed in 7.07s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA52_restore_hf32_singlefield_baseline.h`
- SHA256:
  - `30eda4660a3bb8fcf9475a7dc532211b7552784c0fd5acc6bed14ece727ab3cd`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA52_restore_hf32_singlefield_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA52_restore_hf32_singlefield_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.299`
  - `S=512`: `1.945`
  - `S=2048`: `2.749`
  - `S=10240`: `2.793`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A51b ratios: `1.317 / 1.935 / 2.743 / 2.792`.
- A52 is correctness-safe and suitable as a clean continuation surface for raw-MMAD state cleanup.
- A52 is not a clear performance win; A48 remains the preferred performance rollback surface.

## A53 RestoreRawMmadState HF32Trans Single-Field Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `92dfb07d69755d1cbbde3ffbb8ced6489c6ace95714cc4f0975af6e4945ef334`
- Start point:
  - A52 clean source SHA256 `30eda4660a3bb8fcf9475a7dc532211b7552784c0fd5acc6bed14ece727ab3cd`.
- Change:
  - removed only `SetHF32TransMode(HF32TransMode::NEAREST_EVEN);` from `RestoreRawMmadState()`.
  - all other raw-MMAD state restore fields remain unchanged.
  - P7/P8 formulas, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- isolated package:
  - `/tmp/cgdr_a53_restore_hf32trans_singlefield_20260512_133249.run`
- package SHA256:
  - `0a3524996f5992c9fb45f54d772b0ddd611c891654ab9a4e9be572273db630c0`
- isolated OPP root:
  - `/tmp/cgdr_a53_restore_hf32trans_singlefield_20260512_133249_opp/vendors/custom_transformer`
- installed active Stage1 header SHA256:
  - `92dfb07d69755d1cbbde3ffbb8ced6489c6ace95714cc4f0975af6e4945ef334`
- installed Neumann object SHA256:
  - `385528d2d67a980b21168cad83519be4f8dac2f6c8e5656e228f2cafefcaca7c`
- installed sanity:
  - op_api exports only `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.

Correctness gate:

- build PASS.
- install PASS.
- Note: an initial concurrent smoke launch was discarded for gate attribution; formal smoke was rerun sequentially.
- `S=64` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.15s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA53_restore_hf32trans_singlefield_baseline.h`
- SHA256:
  - `92dfb07d69755d1cbbde3ffbb8ced6489c6ace95714cc4f0975af6e4945ef334`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA53_restore_hf32trans_singlefield_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA53_restore_hf32trans_singlefield_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.271`
  - `S=512`: `1.955`
  - `S=2048`: `2.755`
  - `S=10240`: `2.789`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A52 ratios: `1.299 / 1.945 / 2.749 / 2.793`.
- A53 improves `S=64/10240` versus A52, but regresses `S=512/2048`.
- A53 improves only `S=64/512` versus A48 and remains worse at long sequence sizes.
- A53 is correctness-safe and suitable as a clean continuation surface for raw-MMAD state cleanup only; A48 remains the preferred performance rollback surface.

## A54 RestoreRawMmadState MMLayoutTransform Single-Field Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `f22f56f3f2b1251cfb52c70f2d54a5eac21221bbfe5471c9b46024cee2e7c2bf`
- Start point:
  - A53 clean source SHA256 `92dfb07d69755d1cbbde3ffbb8ced6489c6ace95714cc4f0975af6e4945ef334`.
- Change:
  - removed only `SetMMLayoutTransform(false);` from `RestoreRawMmadState()`.
  - `RestoreRawMmadState()` now contains only `SetAtomicNone();` and `SetNdParaImpl(0x1);`.
  - `PrepareRawMmadState()` still sets padding, nd para, HF32 mode, HF32 trans mode, and MM layout transform.
  - P7/P8 formulas, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- isolated package:
  - `/tmp/cgdr_a54_restore_mmlayout_singlefield_20260512_134540.run`
- package SHA256:
  - `05ac1c2cb82891ce4161220a104a1f6811fd30c49eb78f279fc11dfae9e8e1bd`
- isolated OPP root:
  - `/tmp/cgdr_a54_restore_mmlayout_singlefield_20260512_134540_opp/vendors/custom_transformer`
- installed active Stage1 header SHA256:
  - `f22f56f3f2b1251cfb52c70f2d54a5eac21221bbfe5471c9b46024cee2e7c2bf`
- installed Neumann object SHA256:
  - `fd914a7d6249a1298896cf0e421a073e486b8164d3178d94457aef835816350a`

Correctness gate:

- build PASS.
- install PASS.
- `S=64` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.19s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA54_restore_mmlayout_singlefield_baseline.h`
- SHA256:
  - `f22f56f3f2b1251cfb52c70f2d54a5eac21221bbfe5471c9b46024cee2e7c2bf`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA54_restore_mmlayout_singlefield_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA54_restore_mmlayout_singlefield_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.322`
  - `S=512`: `1.915`
  - `S=2048`: `2.746`
  - `S=10240`: `2.778`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A53 ratios: `1.271 / 1.955 / 2.755 / 2.789`.
- A54 improves `S=512/2048/10240` versus A53, but regresses `S=64`.
- A54 improves only `S=512` versus A48; `S=2048/10240` are effectively flat and `S=64` regresses.
- A54 is correctness-safe and suitable as a clean continuation surface for raw-MMAD state cleanup only; A48 remains the preferred performance rollback surface.

## A55 RestoreRawMmadState NdPara Single-Field Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `ca7b044e48fd7c3d1553fe60479e6b9684369847963335eacd8eb14bfc8c4aac`
- Start point:
  - A54 clean source SHA256 `f22f56f3f2b1251cfb52c70f2d54a5eac21221bbfe5471c9b46024cee2e7c2bf`.
- Change:
  - removed only `SetNdParaImpl(0x1);` from `RestoreRawMmadState()`.
  - `RestoreRawMmadState()` now contains only `SetAtomicNone();`.
  - `PrepareRawMmadState()` still sets padding, nd para, HF32 mode, HF32 trans mode, and MM layout transform.
  - P7/P8 formulas, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- isolated package:
  - `/tmp/cgdr_a55_restore_ndpara_singlefield_20260512_135630.run`
- package SHA256:
  - `4d4a92c144f536398cf6fbeb096a3983f32f937ca363ed7f635630ac939805a4`
- isolated OPP root:
  - `/tmp/cgdr_a55_restore_ndpara_singlefield_20260512_135630_opp/vendors/custom_transformer`
- installed Neumann object SHA256:
  - `dec83cd36b255e0dc93bdf99a543f93818a0e7d68a72662786c7db5bdf3c0689`
- note:
  - the isolated OPP directory had to be regenerated from the saved `.run` before completing profiling.

Correctness gate:

- build PASS.
- install PASS.
- `S=64` sequential PASS.
- `S=256` sequential PASS.
- `S=1024` sequential PASS.
- full compare PASS:
  - `16 passed in 6.15s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA55_restore_ndpara_singlefield_baseline.h`
- SHA256:
  - `ca7b044e48fd7c3d1553fe60479e6b9684369847963335eacd8eb14bfc8c4aac`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA55_restore_ndpara_singlefield_retry1_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA55_restore_ndpara_singlefield_retry1_20260512/performance_compare.csv`
- note:
  - first `S=512 neumann` retry failed because the hand-run msprof application did not inherit the custom OPP environment; it produced no valid target operator row and was superseded by the later successful run in the same output tree.
- neumann/normal ratios:
  - `S=64`: `1.291`
  - `S=512`: `1.925`
  - `S=2048`: `2.750`
  - `S=10240`: `2.802`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A54 ratios: `1.322 / 1.915 / 2.746 / 2.778`.
- A55 improves `S=64` versus A54, but regresses `S=512/2048/10240`.
- A55 improves only `S=512` versus A48 and remains worse at `S=64/2048/10240`.
- A55 is correctness-safe and suitable as a clean continuation surface for raw-MMAD state cleanup only; A48 remains the preferred performance rollback surface.

## A56 RestoreRawMmadState AtomicNone Single-Field Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `5646df25a94ee35f030bff0ef756c284738927fb7fe212afe5ae51dbe94c6bfb`
- Start point:
  - A55 clean source SHA256 `ca7b044e48fd7c3d1553fe60479e6b9684369847963335eacd8eb14bfc8c4aac`.
- Change:
  - removed only `SetAtomicNone();` from `RestoreRawMmadState()`.
  - `RestoreRawMmadState()` is now empty.
  - `PrepareRawMmadState()` still sets padding, nd para, HF32 mode, HF32 trans mode, and MM layout transform.
  - P7/P8 formulas, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- isolated package:
  - `/tmp/cgdr_a56_restore_atomicnone_singlefield_20260512_141651.run`
- package SHA256:
  - `61fc3e1f4f27a9cb581290686980e58d4feebd5d8904c91c4ed5714fa8cfce27`
- isolated OPP root:
  - `/tmp/cgdr_a56_restore_atomicnone_singlefield_20260512_141651_opp/vendors/custom_transformer`
- installed Neumann object SHA256:
  - `3ec083c63e026930057e19f40c2dde3e8a697724d621bd12546ee03d6647f87c`

Correctness gate:

- build PASS.
- install PASS.
- `S=64` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.16s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA56_restore_atomicnone_singlefield_baseline.h`
- SHA256:
  - `5646df25a94ee35f030bff0ef756c284738927fb7fe212afe5ae51dbe94c6bfb`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA56_restore_atomicnone_singlefield_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA56_restore_atomicnone_singlefield_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.335`
  - `S=512`: `1.945`
  - `S=2048`: `2.743`
  - `S=10240`: `2.791`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A55 ratios: `1.291 / 1.925 / 2.750 / 2.802`.
- A56 improves `S=2048/10240` versus A55, but regresses `S=64/512`.
- A56 improves `S=512/2048` versus A48, but regresses `S=64/10240`.
- A56 is correctness-safe and closes the current `RestoreRawMmadState()` cleanup line; A48 remains the preferred performance rollback surface.

## A57 P7 Packed64 Diagonal MMAD HF32 Set Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `6a8ad59ee6bfd4cd4edfc8750b7bd6f7ab81e7ffd8182731ad0be3a4ad9d98d9`
- Start point:
  - A56 clean source SHA256 `5646df25a94ee35f030bff0ef756c284738927fb7fe212afe5ae51dbe94c6bfb`.
- Change:
  - removed only the repeated `SetHF32Mode(HF32Mode::DISABLE);` inside `MmadPacked64DiagOnly()` before each packed64 diagonal MMAD.
  - `C64InverseBy16x16Neumann()` still calls `PrepareRawMmadState()`, which keeps the canonical HF32 disable setup.
  - P7/P8 formulas, packed layout, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- isolated package:
  - `/tmp/cgdr_a57_p7_diag_mmad_no_hf32_cgdrpkg_20260512_143719.run`
- package SHA256:
  - `a498e72d04e2547a28c3de28c452a390674f20441604bda1ac588948c4c8accb`
- isolated OPP root:
  - `/tmp/cgdr_a57_p7_diag_mmad_no_hf32_cgdrpkg_20260512_143719_opp/vendors/cgdr_a57_transformer`
- installed active Stage1 header SHA256:
  - `6a8ad59ee6bfd4cd4edfc8750b7bd6f7ab81e7ffd8182731ad0be3a4ad9d98d9`
- installed Neumann object SHA256:
  - `b66226f1ab02bd3c903edf05600a8b08b72a866cf46e83e666ee466545061fea`

Correctness gate:

- build PASS after `bash ../../build.sh --make_clean` cleared a stale `tri_inverse` CMakeCache overwrite in `build_out`.
- install PASS.
- `S=64` sequential PASS.
- `S=256` sequential PASS.
- `S=1024` sequential PASS.
- full compare PASS:
  - `16 passed in 6.21s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA57_p7_diag_mmad_no_hf32_baseline.h`
- SHA256:
  - `6a8ad59ee6bfd4cd4edfc8750b7bd6f7ab81e7ffd8182731ad0be3a4ad9d98d9`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA57_p7_diag_mmad_no_hf32_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA57_p7_diag_mmad_no_hf32_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.293`
  - `S=512`: `1.920`
  - `S=2048`: `2.746`
  - `S=10240`: `2.800`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A56 ratios: `1.335 / 1.945 / 2.743 / 2.791`.
- A57 improves `S=64/512` versus A56, but regresses slightly at `S=2048/10240`.
- A57 improves only `S=512` versus A48 and remains worse at `S=64/2048/10240`.
- A57 is correctness-safe and suitable as a continuation surface for P7 raw-MMAD setup cleanup; A48 remains the preferred performance rollback surface.

## A58 P8 Packed32 MMAD HF32 Set Cleanup PASS

Candidate boundary:

- Active Stage1 SHA256:
  - `4a26eba55d45a58eb8933e01fd9f908744b7a93b099256b1ca4d4fee9b83f76a`
- Start point:
  - A57 clean source SHA256 `6a8ad59ee6bfd4cd4edfc8750b7bd6f7ab81e7ffd8182731ad0be3a4ad9d98d9`.
- Change:
  - removed only the repeated `SetHF32Mode(HF32Mode::DISABLE);` inside `MmadPacked32()` before P8 packed32 MMAD calls.
  - `C64InverseBy16x16Neumann()` still calls `PrepareRawMmadState()`, which keeps the canonical HF32 disable setup.
  - Generic copy/layout helpers still keep their own HF32 setup. P7/P8 formulas, packed layout, BF16 GM writeback, resident L1 RHS exposure, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS math are unchanged.

Build/install artifacts:

- first build failed in FATBIN because `kernel_meta` temporary object rename/output files disappeared; after `bash ../../build.sh --make_clean`, rebuild PASS. This matches the earlier stale build-dir race pattern and was not a C++ compile error.
- isolated package:
  - `/tmp/cgdr_a58_p8_packed32_mmad_no_hf32_20260512_145328.run`
- package SHA256:
  - `fd1a01aeaec2823ad0229b41a7e88f27085466ed01f4a56975bae95985a3694c`
- isolated OPP root:
  - `/tmp/cgdr_a58_p8_packed32_mmad_no_hf32_20260512_145328_opp/vendors/cgdr_a58_transformer`
- installed active Stage1 header SHA256:
  - `4a26eba55d45a58eb8933e01fd9f908744b7a93b099256b1ca4d4fee9b83f76a`
- installed Neumann object SHA256:
  - `482d1ea374283b43900330f48ffc1fe80a8513e260179e162e19869befda48f3`

Correctness gate:

- build PASS.
- install PASS.
- `S=64` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` sequential PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.41s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA58_p8_packed32_mmad_no_hf32_baseline.h`
- SHA256:
  - `4a26eba55d45a58eb8933e01fd9f908744b7a93b099256b1ca4d4fee9b83f76a`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA58_p8_packed32_mmad_no_hf32_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA58_p8_packed32_mmad_no_hf32_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.313`
  - `S=512`: `1.905`
  - `S=2048`: `2.737`
  - `S=10240`: `2.776`

Comparison:

- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A57 ratios: `1.293 / 1.920 / 2.746 / 2.800`.
- A58 improves `S=512/2048/10240` versus A57, but regresses `S=64`.
- A58 improves `S=512/2048/10240` versus A48, but regresses `S=64`.
- A58 is correctness-safe and currently the strongest long-sequence performance surface in this cleanup line; A48 remains the best small-sequence surface.

## A59 Raw Packed Half L1 Accumulate HF32 Cleanup Failed Candidate

Candidate boundary:

- Starting point: A58 clean baseline.
- Candidate SHA256:
  - `4ce3769b1b2ba8849461cfb8a837052f562134d7a61c23e4ea041d3db039cdcf`
- Change:
  - removed only `SetHF32Mode(HF32Mode::DISABLE);` inside `RawMmadPackedHalfL1AccumulateTile()`.
  - P7/P8 formulas, packed layout, BF16/resident/RHS, Stage2/Stage3/QK, and `gBK/vBeta` were unchanged.
- Package:
  - `/tmp/cgdr_a59_packed_block_mmad_no_hf32_20260512_150502.run`
  - package SHA256 `5a2119d3d687463ef0eed786e319327409731b9326ee064a98c1e500c36e60bb`
- Isolated OPP:
  - `/tmp/cgdr_a59_packed_block_mmad_no_hf32_20260512_150502_opp/vendors/cgdr_a59_transformer`
- Installed active header SHA256:
  - `4ce3769b1b2ba8849461cfb8a837052f562134d7a61c23e4ea041d3db039cdcf`
- Installed Neumann object SHA256:
  - `4e4967d00a123fa01e5821ffbc5258d0965e9e0f2b1c9307327a3f2d456469b8`

Gate result:

- build PASS.
- install PASS.
- `S=64` PASS.
- `S=256` PASS.
- `S=1024` FAIL:
  - CPU vs ChunkGDRNeumann out max_abs `9.486666e-02`
  - worst_idx `63396`, expect `9.48596820e-02`, actual `-6.97374344e-06`
  - state PASS.
- No full compare or performance run.
- No baseline saved.

Failure judgment:

- `RawMmadPackedHalfL1AccumulateTile()` is still a generic copy/layout MMAD helper used outside the P7/P8 packed compute calls.
- Unlike `MmadPacked64DiagOnly()` and `MmadPacked32()`, removing its local HF32 setup is not correctness-safe for long sequence output.
- Active source restored to A58 before continuing; do not stack new work on SHA `4ce3769b...`.

## A60 Result64 Zero Scope Candidate Start

Candidate boundary:

- Starting point: A58 clean baseline SHA256 `4a26eba55d45a58eb8933e01fd9f908744b7a93b099256b1ca4d4fee9b83f76a`.
- Candidate SHA256 `5d2cfb3d95a6974f92b4f4d37be3e17c421ea0115ec3f4e1d498d9201f54c773`.
- Change:
  - only `LoadPacked64DiagonalInputsToL1()` strict full64 branch is changed.
  - `aicHalfL1P3Result_` zero scope is reduced from full `64x64` to upper-triangular 6 `16x16` fractals via `ZeroPacked64UpperTriangularBlocks()`.
  - tail / non-full64 path still uses full `64x64` zero.
- Unchanged:
  - BF16 upper-zero row-suffix, resident zero, P7/P8 formulas, L0C copy-out, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS.

Validation status:

- build PASS.
- install PASS.
- package:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`
  - package SHA256 `4622ffd0df85c6780245fd24809700307b870b908a82b78c1b42602982e38160`
- installed Neumann object SHA256:
  - `fabad3def2466c4da32e191d205838c9721327101dcebae81ffacdbc3e49b8c3`
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.12s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA60_result64_upper_zero_only_baseline.h`
- SHA256:
  - `5d2cfb3d95a6974f92b4f4d37be3e17c421ea0115ec3f4e1d498d9201f54c773`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA60_result64_upper_zero_only_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA60_result64_upper_zero_only_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.314`
  - `S=512`: `1.888`
  - `S=2048`: `2.736`
  - `S=10240`: `2.780`

Comparison:

- A58 ratios: `1.313 / 1.905 / 2.737 / 2.776`.
- A48 ratios: `1.278 / 1.960 / 2.745 / 2.778`.
- A60 improves `S=512` versus A58, is effectively flat at `S=2048`, and slightly regresses `S=64/10240`.
- A60 is the latest correctness continuation surface; A48 remains the preferred small-sequence rollback surface.

## A61a P8 First-Layer Direct L0 Zero

Audit:

- Original A61 target was P8 first-layer packed32 tmp/result zero-scope cleanup.
- Current A60 source no longer has the old `tmp32` L1 pre-zero; that was removed in A19.
- Current P8 first-layer repeated L0 clearing was already reduced to one-time clearing in A29.
- Therefore the remaining safe single point is not another L1 tmp/result pre-zero deletion, but replacing the remaining one-time P8 first-layer L0 clear implementation.

Candidate boundary:

- Starting point: A60 clean baseline SHA256 `5d2cfb3d95a6974f92b4f4d37be3e17c421ea0115ec3f4e1d498d9201f54c773`.
- Candidate SHA256 `dfa33e483b0661626f4835a388546b425581ff0455aed3d6cf2efbed7b4b01ac`.
- Single-point change:
  - only `P8ComposeFirstLayerPacked32Direct()` was changed.
  - replaced `EnsureHalfL1Zero(C32*C32) + CopyHalfL1MatrixToL0A/B()` with `ZeroHalfL0(aicHalfL0A_, C32)` and `ZeroHalfL0(aicHalfL0B_, C32)`.
- Unchanged:
  - P8 first-layer output placement.
  - direct resident / resident exposure.
  - partial L0C copy-out.
  - BF16 GM materialization, resident zero, P7/P8 formulas, Stage2/Stage3/QK, `gBK/vBeta`, and final RHS.

Validation:

- build PASS.
- install PASS.
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.16s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA61a_p8_first_layer_direct_l0_zero_baseline.h`
- SHA256:
  - `dfa33e483b0661626f4835a388546b425581ff0455aed3d6cf2efbed7b4b01ac`

Artifacts and caveat:

- Current package path:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`
- Current package SHA256:
  - `b07f852ba5ac7df1dc4a1cd0a42f5aa54bd7241881d7ad70f9453104f47d4da9`
- The default install root now contains nested historical `vendors/custom_transformer` directories.
- The top-level `op_api/lib/libcust_opapi.so` has SHA256 `79ed9356b7bbcc501520dab2ad62fe52e15d59f4e51937cac0ccd4539b3b1c06` and lacks `aclnnChunkGatedDeltaRule*` symbols.
- Nested `op_api/lib/libcust_opapi.so` files have SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752` and do contain the expected symbols.
- Because of this path pollution, both performance attempts produced empty CSVs:
  - `tests/pytest/prof_out/packed64_phaseA61a_p8_first_layer_direct_l0_zero_20260512/performance_compare.csv`
  - `tests/pytest/prof_out/packed64_phaseA61a_p8_first_layer_direct_l0_zero_ld_20260512/performance_compare.csv`
- These performance CSVs are not valid phase data.

Conclusion:

- A61a is correctness-safe and saved as the latest correctness continuation baseline.
- No performance ratio is recorded for A61a until the OPP install path is cleaned or an isolated install root is used again.
- A48 remains the preferred small-sequence performance rollback surface.

## A63a AIV Scratch Rectangular Writeback

Candidate boundary:

- Starting point: A61a clean baseline SHA256 `dfa33e483b0661626f4835a388546b425581ff0455aed3d6cf2efbed7b4b01ac`.
- A63a SHA256 `ca6bc32ca4f80bc093a0b293c0bbd20dcf5b24f3a88a7dc0c594b2804f8a66b5`.
- Single-point change:
  - added rectangular half CopyOut helper for AIV scratch.
  - `L_powHalfWsGm_` writeback is reduced from each AIV half's `32x64` area to the diagonal `32x32` half.
  - `INVInitHalfWsGm_` writeback is reduced to subBlock0 `32x32` and subBlock1 `32x64`.
- Scope guard:
  - no P7/P8 formula change.
  - no BF16 GM materialization, resident L1 exposure, RHS, Stage2/Stage3/QK, or `gBK/vBeta` change.

Important fix:

- `DataCopyExtParams::srcStride` for CopyOut is UB-side stride in 32B units.
- `DataCopyExtParams::dstStride` is GM-side stride in bytes.
- The failed byte-`srcStride` candidate SHA `60e326...` failed at `S=64` with out/state large error and is not a baseline.

Validation:

- build PASS.
- install PASS.
- isolated package:
  - `/tmp/cgdr_a63a_srcstride_20260512_171721.run`
- package SHA256:
  - `0f3a4aa4a99edfb102738feaf6da48cd64fd7c55375bd19cc82fcb344d51d96c`
- isolated OPP root:
  - `/tmp/cgdr_a63a_srcstride_20260512_171721_opp/vendors/custom_transformer`
- installed Neumann object SHA256:
  - `920bacc66168f28101c2e1fedc38fd609c3f4abdf1e8b94f816b34a07f5cc833`
- `S=64/256/1024` smoke PASS.
- full compare PASS:
  - `16 passed in 6.28s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA63a_aiv_scratch_rect_write_baseline.h`
- SHA256:
  - `ca6bc32ca4f80bc093a0b293c0bbd20dcf5b24f3a88a7dc0c594b2804f8a66b5`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA63a_aiv_scratch_rect_write_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA63a_aiv_scratch_rect_write_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.244`
  - `S=512`: `1.948`
  - `S=2048`: `2.710`
  - `S=10240`: `2.775`

Conclusion:

- A63a is the latest correctness continuation baseline.
- A63a is also the current preferred performance baseline because it improves all four measured points versus A48 (`1.278 / 1.960 / 2.745 / 2.778`).

## A64 Resident Upper-Zero Scope

Candidate boundary:

- Starting point: A63a clean baseline SHA256 `ca6bc32ca4f80bc093a0b293c0bbd20dcf5b24f3a88a7dc0c594b2804f8a66b5`.
- A64 SHA256 `9db43aa3868a7e2c9384b58b328dd4e4b8ed19a5b061a761b529d2abd97d0a2c`.
- Single-point change:
  - `StorePackedL1InverseToResidentL1()` now clears only upper triangular 6 `16x16` resident blocks with `ZeroPacked64UpperTriangularBlocks(resident)`.
  - The lower 10 resident blocks are still produced by the existing `CopyHalfL1PackedBlockToCompact()` loop.
- Scope guard:
  - no `FusedRhsAICProcess()` input change.
  - no P8 final output placement change.
  - no BF16 GM materialization, Stage2/Stage3/QK, `gBK/vBeta`, or final RHS math change.

Validation:

- build PASS.
- install PASS.
- isolated package:
  - `/tmp/cgdr_a64_resident_upper_zero_20260512_173829.run`
- package SHA256:
  - `aee16480d5a94585956bb1aa619f91bb4f1287ef4743c299e75b228ff400f0a0`
- isolated OPP root:
  - `/tmp/cgdr_a64_resident_upper_zero_20260512_173829_opp/vendors/custom_transformer`
- installed Neumann object SHA256:
  - `a76ef23e33a9d81859e5ce9c5ddc781413078f20a89a41005190a1b1d43327ec`
- `S=64/256/1024` smoke PASS.
- full compare PASS:
  - `16 passed in 6.20s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA64_resident_upper_zero_only_baseline.h`
- SHA256:
  - `9db43aa3868a7e2c9384b58b328dd4e4b8ed19a5b061a761b529d2abd97d0a2c`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA64_resident_upper_zero_only_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA64_resident_upper_zero_only_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.239`
  - `S=512`: `1.916`
  - `S=2048`: `2.740`
  - `S=10240`: `2.786`

Conclusion:

- A64 is the latest correctness continuation baseline.
- Compared with A63a (`1.244 / 1.948 / 2.710 / 2.775`), A64 improves `S=64/512` but regresses `S=2048/10240`.
- Keep A63a as the all-shape preferred performance baseline; use A64 as the latest correctness surface for the next resident-specific phase.

## A65a Resident Upper-Zero Single Sync

Candidate boundary:

- Starting point: A64 clean baseline SHA256 `9db43aa3868a7e2c9384b58b328dd4e4b8ed19a5b061a761b529d2abd97d0a2c`.
- A65a SHA256 `180e49f446c2df5a92da333e7b895949e1c0b313cec13c2cabd8aacd67aa26d8`.
- Single-point change:
  - split `ZeroHalfL1NoSync()` out of `ZeroHalfL1()`;
  - added `ZeroPacked64UpperTriangularBlocksSingleSync()`;
  - only `StorePackedL1InverseToResidentL1()` uses the single-sync helper for the resident upper 6 blocks.
- Scope guard:
  - no other `ZeroHalfL1()` call behavior changed.
  - no BF16 GM materialization, resident lower 10 block copy, RHS, P7/P8 formula, Stage2/Stage3/QK, or `gBK/vBeta` change.

Validation:

- build PASS.
- isolated install PASS.
- source and installed active header SHA256 both `180e49f446c2df5a92da333e7b895949e1c0b313cec13c2cabd8aacd67aa26d8`.
- package:
  - `/tmp/cgdr_a65a_20260512_175433.run`
- package SHA256:
  - `25f6075f3e5904f539bb5658bdf83fd8821deddf513b97ebf21da61db19da2a3`
- isolated OPP:
  - `/tmp/cgdr_a65a_opp_20260512_175433/vendors/custom_transformer`
- installed Neumann object SHA256:
  - `caefe7792d2b894839b0fad4bc251579495a89956cf6491a3f444cddd8dca7ab`
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.22s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA65a_resident_upper_zero_single_sync_baseline.h`
- SHA256:
  - `180e49f446c2df5a92da333e7b895949e1c0b313cec13c2cabd8aacd67aa26d8`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA65a_resident_upper_zero_single_sync_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA65a_resident_upper_zero_single_sync_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.209`
  - `S=512`: `1.881`
  - `S=2048`: `2.734`
  - `S=10240`: `2.787`

Conclusion:

- A65a is the latest correctness continuation baseline.
- Compared with A64 (`1.239 / 1.916 / 2.740 / 2.786`), A65a improves `S=64/512/2048` and is effectively flat at `S=10240`.
- A63a remains the all-shape preferred performance baseline; use A65a as the latest correctness surface for the next narrow sync-reduction phase.

## A65b Result64 Upper-Zero Single Sync

Candidate boundary:

- Starting point: A65a clean baseline SHA256 `180e49f446c2df5a92da333e7b895949e1c0b313cec13c2cabd8aacd67aa26d8`.
- A65b SHA256 `8883db05bb1a0ce4c642bb3637429b20339e5d1e15c43954a6ffd3619c340c55`.
- Single-point change:
  - `LoadPacked64DiagonalInputsToL1()` strict full64 branch uses `ZeroPacked64UpperTriangularBlocksSingleSync(aicHalfL1P3Result_)`.
  - This applies A65a's same-shape upper-zero single-sync helper to `result64`/`aicHalfL1P3Result_`.
- Scope guard:
  - no resident lower copy, BF16 GM materialization, RHS, P7/P8 formula, Stage2/Stage3/QK, or `gBK/vBeta` change.

Validation:

- build/install PASS.
- `S=64/256/1024` smoke PASS.
- full compare PASS.

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA65b_result64_upper_zero_single_sync_baseline.h`
- SHA256:
  - `8883db05bb1a0ce4c642bb3637429b20339e5d1e15c43954a6ffd3619c340c55`

Performance:

- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA65b_result64_upper_zero_single_sync_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.258`
  - `S=512`: `1.911`
  - `S=2048`: `2.742`
  - `S=10240`: `2.785`

Conclusion:

- A65b is correctness-safe but not the preferred continuation surface.
- Compared with A65a (`1.209 / 1.881 / 2.734 / 2.787`), A65b regresses `S=64/512/2048` and is only trivially better at `S=10240`.
- Active source has been restored to A65a SHA256 `180e49f446c2df5a92da333e7b895949e1c0b313cec13c2cabd8aacd67aa26d8`.
- Do not continue from A65b by default.

## A66 Start: I16 Identity Cache Keep Across Materialization

Candidate boundary:

- Starting point: A65a active source SHA256 `180e49f446c2df5a92da333e7b895949e1c0b313cec13c2cabd8aacd67aa26d8`.
- Candidate SHA256 `db0db360181e0a48f1005469ea2c9b348330419cb9aca19e5e8be0011a1cb2d1`.
- Single-point change:
  - remove the `preparedHalfIdentityBlockSize_ = 0` at the end of `P8ComposeFirstLayerPacked32Direct()`;
  - remove the `preparedHalfIdentityBlockSize_ = 0` immediately before `StorePackedL1InverseToBf16Attn()`;
  - keep the final reset after `StorePackedL1InverseToResidentL1()`.
- Intent:
  - reuse the same `I16` half identity cache inside the current raw-MMAD inverse epoch for P8 packed32 output copy, BF16 materialization, and resident materialization;
  - avoid rebuilding `I16` through GM scratch between these consecutive materialization helpers.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 writeback shape change;
  - no resident writeback shape change;
  - no final RHS math, Stage2/Stage3/QK, or `gBK/vBeta` change.

Validation status:

- build PASS.
- isolated install PASS.
- source and saved baseline SHA256 both `db0db360181e0a48f1005469ea2c9b348330419cb9aca19e5e8be0011a1cb2d1`.
- package:
  - `/tmp/packed64_phaseA66_identity16_cache_keep_20260512_185151.run`
- package SHA256:
  - `7613b0fffd744f4b15c37c3d7d108fdf7c47f2ad49303311f6cd1f98e4e47cae`
- isolated OPP:
  - `/tmp/packed64_phaseA66_identity16_cache_keep_20260512_185151_opp/vendors/custom_transformer`
- installed Neumann object SHA256:
  - `7613edee5a2f3301813a0e8641940ae26400e9d14a90947bf03290f7e1a8bd0a`
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.17s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA66_identity16_cache_keep_baseline.h`
- SHA256:
  - `db0db360181e0a48f1005469ea2c9b348330419cb9aca19e5e8be0011a1cb2d1`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA66_identity16_cache_keep_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA66_identity16_cache_keep_20260512/performance_compare.csv`
- neumann/normal ratios:
  - `S=64`: `1.244`
  - `S=512`: `1.778`
  - `S=2048`: `2.528`
  - `S=10240`: `2.561`

Conclusion:

- A66 is the latest correctness continuation baseline.
- A66 becomes the current all-shape preferred performance baseline.
- Compared with A65a (`1.209 / 1.881 / 2.734 / 2.787`), A66 regresses `S=64` but improves `S=512/2048/10240`.
- Continue from A66 for the next single-point phase.

## A67 Start: I16 Identity Cache Keep Across ParaIdx

Candidate boundary:

- Starting point: A66 clean baseline SHA256 `db0db360181e0a48f1005469ea2c9b348330419cb9aca19e5e8be0011a1cb2d1`.
- Candidate SHA256 `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`.
- Single-point change:
  - remove the final `preparedHalfIdentityBlockSize_ = 0` at the end of `C64InverseBy16x16Neumann()`.
- Intent:
  - allow `I16` half identity cache to survive across multiple `paraIdx` iterations in the same `ParaChunkAIC()` inverse loop.
  - keep existing invalidation at `ParaChunkAIC()` inverse-window start, `InitRawMmadLocalBuffers()`, and `RestoreStage1MatmulLocalBuffers()`.
- Scope guard:
  - no P7/P8 formula change.
  - no BF16/resident writeback shape change.
  - no final RHS, Stage2/Stage3/QK, or `gBK/vBeta` change.

Validation status:

- build PASS.
- isolated install PASS.
- source and saved baseline SHA256 both `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`.
- package:
  - `/tmp/packed64_phaseA67_identity16_cross_para_keep_20260512_190504.run`
- package SHA256:
  - `7d7f48185a28b9038506e7087e262a49d9b0a39f69de1f3cddf05997fd0e4560`
- isolated OPP:
  - `/tmp/packed64_phaseA67_identity16_cross_para_keep_20260512_190504_opp/vendors/custom_transformer`
- installed active Stage1 header SHA256:
  - `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`
- installed Neumann object SHA256:
  - `edcae896cbcaf76bda7ce5730eac6c58fad9c7619c57e98c4b5d50528a3b87bb`
- `S=64` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `5.276948e-04`
- `S=256` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `9.178072e-04`
- `S=1024` PASS:
  - Neumann out max_abs `3.173694e-03`
  - Neumann state max_abs `1.006737e-03`
- full compare PASS:
  - `16 passed in 6.18s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA67_identity16_cross_para_keep_baseline.h`
- SHA256:
  - `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA67_identity16_cross_para_keep_20260512/`
- Compare CSV:
  - `tests/pytest/prof_out/packed64_phaseA67_identity16_cross_para_keep_20260512/performance_compare.csv`
- Compare CSV SHA256:
  - `8de471eab5f6c91e5529f5628c0d61ff46c6e23086a05d16bdd3a02a21e7d08e`
- Raw CSV:
  - `tests/pytest/prof_out/packed64_phaseA67_identity16_cross_para_keep_20260512/performance_raw.csv`
- Raw CSV SHA256:
  - `70ca0b4d650bd8136f8017d38f4043dd9c614cdc98226299da137fb1c5237262`
- Performance baseline note:
  - `docs/2026-05-12/A67_PERFORMANCE_BASELINE.md`
- neumann/normal ratios:
  - `S=64`: `1.181`
  - `S=512`: `1.635`
  - `S=2048`: `2.425`
  - `S=10240`: `2.456`

Conclusion:

- A67 is the latest correctness continuation baseline.
- A67 becomes the current all-shape preferred performance baseline.
- Compared with A66 (`1.244 / 1.778 / 2.528 / 2.561`), A67 improves all four profiled sizes.
- Continue from A67 for the next single-point phase.

## P8 First-Layer cInv/cTmp Accumulate Plan

Plan artifact:

- `docs/2026-05-12/P8_FIRST_LAYER_CINV_CTMP_ACCUMULATE_PLAN.md`

Starting source:

- active Stage1 SHA256 `1d23b34e174e24287b4046054bd6bce35fd68da23f18c031e45a55028b5bc328`
- previous validated A67 baseline SHA256 `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`

Phase 1 scope:

- only replace P8 first-layer organization.
- keep P7 output `cInv = aicL0C_` alive.
- use `cTmp = aicL0C_[C64_L0C_TILE_ELEMS]` for the first-layer tmp.
- use full64 sparse L0 operand loads and full64 Fixpipe copy-out.
- keep existing second-layer unchanged.

Math:

- `INVInit = I - L`, so lower source is already `lowerNeg = -L`.
- compute `tmp = lowerNeg @ topInv`.
- accumulate `cInv += bottomInv @ tmp`.
- do not add another negative sign.

Guard:

- no partial L0C copy-out.
- no BF16/resident/RHS/Stage2/Stage3/QK/`gBK/vBeta` change.
- second-layer cInv/cTmp accumulation remains a later phase.

## A68 P8 First-Layer cInv/cTmp Accumulate Result

- Starting point: A67 clean baseline SHA256 `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`.
- A68 SHA256 `f94c7bbf54186327dcf09d711a30a76152974312b52dc2ce80848fb78ddb8586`.
- Single-point change:
  - P7 keeps `cInv = aicL0C_` after diagonal inverse.
  - P8 first-layer uses `cTmp = aicL0C_[C64_L0C_TILE_ELEMS]`.
  - first MMAD computes `tmp = lowerNeg @ topInv` into `cTmp`.
  - second MMAD uses `RAW_MMAD_ACCUMULATE_C` to accumulate `bottomInv @ tmp` into `cInv`.
  - full64 Fixpipe writes `cInv` back to `aicHalfL1P3Result_` so the old second-layer path can still read L1.
- Formula unchanged; no extra negative sign.
- No partial L0C copy-out; full64 Fixpipe only.
- Gate PASS:
  - build PASS.
  - isolated install PASS.
  - `S=64/256/1024` PASS.
  - full compare PASS `16 passed in 6.16s`.
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA68_p8_first_layer_cinv_ctmp_accum_baseline.h`
  - SHA256 `f94c7bbf54186327dcf09d711a30a76152974312b52dc2ce80848fb78ddb8586`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA68_p8_first_layer_cinv_ctmp_accum_20260512/`
  - ratios `1.175 / 1.662 / 2.399 / 2.437`.

## A69 P8 Second-Layer cInv/cTmp Accumulate Result

- Starting point: A68 clean baseline SHA256 `f94c7bbf54186327dcf09d711a30a76152974312b52dc2ce80848fb78ddb8586`.
- A69 SHA256 `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`.
- Single-point change:
  - strict full64 branch calls `P8Compose64From32AccumFull64(paraIdx)`;
  - `cTmp` computes `tmp64 = lowerNeg32 @ B0`;
  - `cInv` accumulates `B1 @ tmp64`;
  - final full64 Fixpipe writes `cInv -> aicHalfL1P3Result_`;
  - `P8Compose64From32Direct32()` remains for tail/non-full64 fallback.
- Scope guard:
  - no P7/P8 sign change;
  - no BF16/resident/RHS/Stage2/Stage3/QK/`gBK/vBeta` change;
  - no partial L0C copy-out.
- Package and install:
  - package `/tmp/cgdr_a69_20260512_220105.run`
  - package SHA256 `f85b7864e3c757df74d01f2eae656dd88068302c04dc59180989fafdeb28acee`
  - isolated OPP `/tmp/cgdr_a69_opp_20260512_220105/vendors/custom_transformer`
  - installed active Stage1 header SHA256 `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`
  - installed Neumann object SHA256 `2db6ce2a99cc7a55cbc8ed39fd94a6be09dccbbc3648554c61bf84a7ef85b280`
- Gate PASS:
  - build PASS.
  - isolated install PASS.
  - `S=64` PASS.
  - `S=256` PASS.
  - `S=1024` PASS.
  - full compare PASS `16 passed in 6.22s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_baseline.h`
  - SHA256 `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_20260512/performance_compare.csv`
  - compare CSV SHA256 `484718014cde93891a68d633eb1f275afcd1d1a5323f346367437b4e7738ec56`
  - raw CSV SHA256 `787211d7a3245f4ff9d27c5c6cd5f1fbc308d43cbaaeca524e219a6a54de836b`
  - ratios `1.250 / 1.657 / 2.438 / 2.469`.
- Conclusion:
  - A69 is the latest correctness-safe baseline for the full first-layer and second-layer `cInv/cTmp` accumulate implementation.
  - A69 is not the preferred performance baseline; it regresses `S=64/2048/10240` versus A68 and regresses all four profiled sizes versus A67.

## A70 Direct L0 Zero Candidate And Recovery

- Starting point: A69 correctness baseline SHA256 `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`.
- A70 candidate SHA256 before revert: `1237de47447c1530b2068f21677c16d9569d39317ce3f9ba33f1d626a2617985`.
- Single-point candidate:
  - replaced the L1-zero-to-L0 clear path with direct `ZeroHalfL0Pair()`.
- Result:
  - build PASS.
  - `S=64` failed with AIC exception `507015`.
  - error text included `load2d instruction write address greater than max destination address`.
  - no baseline saved.
- Conclusion:
  - direct L0 zero through the current L0 half views is not a safe implementation.
  - do not retry the same `InitConstValue` direct-L0 approach without a new legal L0 layout/lifecycle design.
- Recovery:
  - reverted to the reliable A69-style path:
    - `SyncMToMte1()`;
    - `EnsureHalfL1Zero(...)`;
    - `CopyHalfL1MatrixToL0A/B(aicHalfL1Zero_, ...)`.
  - removed unused `ZeroHalfL0NoSync`, `ZeroHalfL0`, and `ZeroHalfL0Pair` helpers.
  - active source after recovery SHA256 `a39847a6f7b05bc5911cb2bf13cd76bf264a5123ce6842ac55332dac4a248552`.
  - isolated package `/tmp/cgdr_a70_revert_l0_direct_zero_20260512_224528.run`, package SHA256 `4a4ed99568e4408cc43d4b8b114a7e2b017dcf3e0c2effbd5a6572479d627f4f`.
  - isolated OPP `/tmp/cgdr_a70_revert_l0_direct_zero_opp_20260512_224528/vendors/custom_transformer`.
  - installed Neumann object SHA256 `2db6ce2a99cc7a55cbc8ed39fd94a6be09dccbbc3648554c61bf84a7ef85b280`.
  - recovery gate PASS: `S=64/256/1024` PASS and full compare PASS `16 passed in 6.14s`.

## A71 Candidate Plan: BF16 Upper-Zero Identity Prepare Removal

- Starting point: A70 recovery SHA256 `a39847a6f7b05bc5911cb2bf13cd76bf264a5123ce6842ac55332dac4a248552`.
- Scope:
  - only `CopyZeroHalfL1RectToGmBf16ByMmad()`.
  - keep BF16 GM upper-zero writeback, raw-MMAD zero-copy shape, resident L1 writeback, RHS math, P7, and P8 unchanged.
- Rationale:
  - current zero-rect calculation is `aicHalfL1Zero_ @ aicHalfL1Zero_`.
  - `PrepareHalfL1IdentityByMmad(INVERSE_SHAPE)` does not provide an operand for this MMAD.
  - its useful effect is only the historical visibility/sync boundary before the zero-rect raw MMAD.
- Candidate:
  - replace `PrepareHalfL1IdentityByMmad(INVERSE_SHAPE)` with explicit `SyncFixToMte1()` and `SyncMte2ToMte1()`.
  - do not change any final BF16 GM materialization requirement in this phase.

## A71 BF16 Upper-Zero Identity Prepare Removal Result

- Starting point: A70 recovery SHA256 `a39847a6f7b05bc5911cb2bf13cd76bf264a5123ce6842ac55332dac4a248552`.
- A71 SHA256 `3f5e0b61eff7faa38b292312cb1b91f861fcb7db30b6f7fd2a6b00bd835ffa47`.
- Single-point change:
  - in `CopyZeroHalfL1RectToGmBf16ByMmad()`, removed `PrepareHalfL1IdentityByMmad(INVERSE_SHAPE)`.
  - replaced it with `SyncFixToMte1()` and `SyncMte2ToMte1()`.
- Scope guard:
  - BF16 GM upper-zero writeback is still present.
  - no resident/RHS/P7/P8/Stage2/Stage3/QK/`gBK/vBeta` change.
- Package and install:
  - package `/tmp/packed64_phaseA71_bf16_zero_rect_no_identity_20260512_230955.run`.
  - package SHA256 `206f54f91e67badecf0574f4fe663e5e602bc58d4853ede5d8bd612e6a88755b`.
  - isolated OPP `/tmp/packed64_phaseA71_bf16_zero_rect_no_identity_20260512_230955_opp/vendors/custom_transformer`.
  - installed Neumann object SHA256 `669e476bed89f2ea91bfaba812d869ef476926b90b1b871c0d3de74f48edda7c`.
- Gate PASS:
  - build PASS.
  - isolated install PASS.
  - `S=64/256/1024` PASS.
  - full compare PASS `16 passed in 6.19s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA71_bf16_zero_rect_no_identity_baseline.h`.
  - SHA256 `3f5e0b61eff7faa38b292312cb1b91f861fcb7db30b6f7fd2a6b00bd835ffa47`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA71_bf16_zero_rect_no_identity_20260512/performance_compare.csv`.
  - compare CSV SHA256 `c33234973be135ed09f66a90dbfe4df276e3eee9f4cf8d2e65a341520214fa4b`.
  - raw CSV SHA256 `dabe527e6705c72ccb67097ef901745d5d61776da5be92d9615d8c2443b029c1`.
  - ratios `1.260 / 1.680 / 2.424 / 2.444`.
- Conclusion:
  - A71 is correctness-safe.
  - A71 improves long sequence ratios versus A69 and A67, but regresses `S=64/512` versus A69 and does not beat A68 on `S=64/512/2048`.
  - Continue with another isolated writeback redundancy cleanup rather than treating A71 as a performance-preferred endpoint.

## A72 Candidate Plan: Remove Unconsumed BF16 Entry Zero Prepare

- Starting point: A71 SHA256 `3f5e0b61eff7faa38b292312cb1b91f861fcb7db30b6f7fd2a6b00bd835ffa47`.
- Scope:
  - only `StorePackedL1InverseToBf16Attn()`.
  - remove the entry `EnsureHalfL1ZeroForMte1(INVERSE_SHAPE * INVERSE_SHAPE)`.
  - keep each zero-rect writeback's own full64 `EnsureHalfL1ZeroForMte1(C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32)`.
- Rationale:
  - the entry 16x16 zero prepare is not consumed by lower true-inverse BF16 block writes.
  - the first upper-zero rectangle immediately prepares the full64 zero source itself.
  - therefore the entry prepare is redundant in the current 64-aligned path.
- Scope guard:
  - no BF16 deletion.
  - no resident/RHS/P7/P8/Stage2/Stage3/QK/`gBK/vBeta` change.

## A72 BF16 Entry Zero Prepare Drop Result

- Starting point: A71 SHA256 `3f5e0b61eff7faa38b292312cb1b91f861fcb7db30b6f7fd2a6b00bd835ffa47`.
- A72 SHA256 `deb987f598b383bcc73b7d18dc846f78fc3fb120970382b214ee84b7ec006520`.
- Single-point change:
  - removed the entry `EnsureHalfL1ZeroForMte1(INVERSE_SHAPE * INVERSE_SHAPE)` in `StorePackedL1InverseToBf16Attn()`.
  - kept each `CopyZeroHalfL1RectToGmBf16ByMmad()` full64 zero preparation.
- Scope guard:
  - BF16 GM writeback remains present.
  - resident/RHS/P7/P8/Stage2/Stage3/QK/`gBK/vBeta` unchanged.
- Package and install:
  - package `/tmp/packed64_phaseA72_bf16_entry_zero_prepare_drop_20260512_232147.run`.
  - package SHA256 `35113258c7c862c6d461f908c9a15891123d9e5c7b1b142b1651b56ee2135ec6`.
  - isolated OPP `/tmp/packed64_phaseA72_bf16_entry_zero_prepare_drop_20260512_232147_opp/vendors/custom_transformer`.
  - installed Neumann object SHA256 `6926db661d6d448d162643ebd113a58dac25c1706f4789d5bccdad38bab7f9b1`.
- Gate PASS:
  - build PASS.
  - isolated install PASS.
  - `S=64/256/1024` PASS.
  - full compare PASS `16 passed in 6.23s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA72_bf16_entry_zero_prepare_drop_baseline.h`.
  - SHA256 `deb987f598b383bcc73b7d18dc846f78fc3fb120970382b214ee84b7ec006520`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA72_bf16_entry_zero_prepare_drop_20260512/performance_compare.csv`.
  - compare CSV SHA256 `33b4d071bae64d0c5530eb099a5d692e6b49ef50f59df6269295dbce2ca2e253`.
  - raw CSV SHA256 `0ad976c1adfde92c180374c741aa6dd9991059f3bd729751e1f86c3a4b9b6199`.
  - ratios `1.225 / 1.659 / 2.397 / 2.456`.
- Conclusion:
  - A72 is correctness-safe and improves over A71 on all profiled sizes.
  - A72 beats A69 on `S=64/2048/10240` and is close on `S=512`.
  - A72 beats A68 only at `S=2048`; A68 remains the best recorded `S=64/10240` ratio and A67 remains the best recorded `S=512` ratio.

## P7 And RHS Writeback Audit

- P7 currently has three `FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)` writes.
- Audit conclusion:
  - the first two writes make updated `cInv` L1-readable for later P7 MMAD input.
  - the third write is the final result64 writeback for P8/BF16/resident exposure.
  - these are not removable as the same redundancy class; a future P7 optimization must change the legal read-after-write design, not simply delete them.
- Final inverse exposure:
  - `StorePackedL1InverseToBf16Attn()` still writes BF16 GM `attnWsGm_`.
  - `StorePackedL1InverseToResidentL1()` writes compact resident L1.
  - `FusedRhsAICProcess()` consumes `aicHalfL1ResidentInv_`, not `attnWsGm_`.
- Current boundary:
  - BF16 GM writeback is still retained as a lifecycle/sync side effect because prior deletion/delay variants failed at long sequence.
  - do not delete BF16 GM materialization without a new validated lifecycle replacement.

## A73 Candidate Plan: Drop Redundant FIX->MTE1 Sync In Zero-Rect Writeback

- Starting point: A72 SHA256 `deb987f598b383bcc73b7d18dc846f78fc3fb120970382b214ee84b7ec006520`.
- Scope:
  - only `CopyZeroHalfL1RectToGmBf16ByMmad()`.
  - remove `SyncFixToMte1()` after `EnsureHalfL1ZeroForMte1(...)`.
  - keep `SyncMte2ToMte1()` in this phase.
- Rationale:
  - zero source is prepared by M pipe through `EnsureHalfL1ZeroForMte1()` and synchronized by `SyncMToMte1()` when first generated.
  - cached zero source is not produced by FIX.
  - therefore the explicit FIX->MTE1 dependency inherited from the old identity-cache path is not a data dependency for `zero @ zero`.
- Scope guard:
  - no BF16 deletion.
  - no resident/RHS/P7/P8/Stage2/Stage3/QK/`gBK/vBeta` change.

## A73 BF16 Zero-Rect FIX->MTE1 Sync Drop Result

- Starting point: A72 SHA256 `deb987f598b383bcc73b7d18dc846f78fc3fb120970382b214ee84b7ec006520`.
- A73 SHA256 `07d0e7269273d3b997112cef17a77b48923bfe114d6aae3b31384c858c22a073`.
- Single-point change:
  - removed only `SyncFixToMte1()` in `CopyZeroHalfL1RectToGmBf16ByMmad()` after `EnsureHalfL1ZeroForMte1(...)`.
  - kept `SyncMte2ToMte1()`.
- Scope guard:
  - BF16 GM writeback remains present.
  - resident/RHS/P7/P8/Stage2/Stage3/QK/`gBK/vBeta` unchanged.
- Build note:
  - Do not chain `bash ../../build.sh --make_clean && bash ../../build.sh ...`; this repo script's `--make_clean` path does not early-exit and can leave a stale/default package state.
  - For this pass, `../../build` and `../../build_out` were removed manually before rebuilding.
- Package and install:
  - package `/tmp/packed64_phaseA73_bf16_zero_rect_drop_fix_to_mte1_sync_20260512_234820.run`.
  - package SHA256 `957ede6fbfd4ba561000d070eb49b6aec3ed7e06a031b234ad9ca59bed12f32d`.
  - isolated OPP `/tmp/packed64_phaseA73_bf16_zero_rect_drop_fix_to_mte1_sync_20260512_234820_opp/vendors/custom_transformer`.
  - installed `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
  - installed CGDR object SHA256 `70c293ec3247e91eedadd93cf6a101a4d2136d6540726da1eae4320122f1d79c`.
  - installed Neumann object SHA256 `e6d79cb9866a81a7639c591cb4d68ba64fbd87525b94d460f52bf713cd832872`.
- Gate PASS:
  - build PASS.
  - isolated install PASS.
  - `S=64/256/1024` PASS.
  - full compare PASS `16 passed in 6.15s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA73_bf16_zero_rect_drop_fix_to_mte1_sync_baseline.h`.
  - SHA256 `07d0e7269273d3b997112cef17a77b48923bfe114d6aae3b31384c858c22a073`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA73_bf16_zero_rect_drop_fix_to_mte1_sync_20260512/performance_compare.csv`.
  - compare CSV SHA256 `6994d456cf250ea3d4911e7d09e16817e3b7558a8c74d81bceafab2f30714f0d`.
  - raw CSV SHA256 `c3f1af23db1076270352b5a417b37beff0bb8de57b72f340d0fa01b7f105660d`.
  - ratios `1.223 / 1.684 / 2.419 / 2.460`.
- Conclusion:
  - A73 is correctness-safe.
  - A73 improves only `S=64` slightly versus A72 (`1.223` vs `1.225`) and regresses `S=512/2048/10240`.
  - A73 does not beat A68 or A67 as a performance-preferred surface.
  - Continue optimization from A73 only because it is the latest correctness-safe cleanup; compare performance against A67/A68/A72 before claiming improvement.
