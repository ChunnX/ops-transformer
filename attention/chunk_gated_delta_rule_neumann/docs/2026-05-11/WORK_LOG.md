# 2026-05-11 Work Log

## Strict packed64 A11

- Active selector confirmed:
  - `op_kernel/selected/stage1_selected.h`
  - `op_kernel/selected/stage1_selected_config.h`
- Current active Stage1 is:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- A10 baseline retained:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA10_cleanup_baseline.h`
  - SHA256: `35c7b0817a0b81271a5d47f631c4f41bd409fc2c2b38c15433e9e288e157701b`

Change:

- `EnsurePacked64Identity()` now returns immediately when `packed64IdentityPrepared_` is true.
- Removed cache-hit-only syncs:
  - `SyncFixToM()`
  - `SyncFixToMte1()`
  - `SyncMte2ToMte1()`
  - `SyncMte1ToM()`

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.27s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA11_identity_cache_hit_nosync_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.406`
  - `S=512`: `2.136`
  - `S=2048`: `2.950`
  - `S=10240`: `3.003`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA11_identity_cache_hit_nosync_baseline.h`
- SHA256:
  - `b16e8b44c97bc7983ba78f9e535c13b6a136cbf53005cf10429e3d75476f9f1b`

Conclusion:

- A11 is correctness-safe and saved as a baseline.
- A11 improves A10 slightly for `S=64/512/2048`, but `S=10240` is slightly worse than A10 and all shapes remain slower than Phase 10.
- Continue A12+ one optimization at a time.

## Strict packed64 A12 start

Planned single point:

- Remove cache-hit syncs from `PrepareHalfL1IdentityByMmad()` when `preparedHalfIdentityBlockSize_ == blockSize`.

Guardrails:

- Do not modify P7/P8 formulas.
- Do not remove BF16 pre-RHS GM writeback.
- Do not remove RHS operand L1 zero-init.
- Do not remove `RestoreRawMmadState()`.

Result:

- A12 failed and was reverted.
- Change attempted:
  - removed `SyncFixToMte1()` and `SyncMte2ToMte1()` from `PrepareHalfL1IdentityByMmad()` cache-hit branch.
- Validation:
  - build PASS.
  - install PASS.
  - `S=64` smoke PASS.
  - `S=256` smoke PASS.
  - `S=1024` smoke FAIL.
- Failure:
  - `CPU vs ChunkGDRNeumann out max_abs=1.109401e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.557053e-01`
  - outputs were close to zero at worst indices.
- Revert:
  - active Stage1 restored from A11 baseline.
  - restored active SHA256: `b16e8b44c97bc7983ba78f9e535c13b6a136cbf53005cf10429e3d75476f9f1b`
  - rebuild PASS.
  - reinstall PASS.
  - recovery `S=1024` smoke PASS:
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

Conclusion:

- Do not remove the two cache-hit syncs in `PrepareHalfL1IdentityByMmad()` directly.
- Continue from A11 baseline.

## Strict packed64 A13

Single point:

- Remove the dead `ZeroHalfL1(aicHalfL1TilePower_, C64_BLOCK32 * C64_BLOCK32)` in `P8Compose64From32Direct32()`.
- Diff against A11 baseline is exactly this one line.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS.
- `S=256` smoke PASS.
- `S=1024` smoke PASS.
- compare PASS:
  - `16 passed in 6.13s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA13_p8_dead_zero_cleanup_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.413`
  - `S=512`: `2.099`
  - `S=2048`: `2.973`
  - `S=10240`: `2.988`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA13_p8_dead_zero_cleanup_baseline.h`
- SHA256:
  - `57effd24547d6458dd9d3aca74969765a6ed6f4033c04f1c4a279b24ce5d6b80`

Conclusion:

- A13 is correctness-safe and saved as a baseline.
- Performance is mixed versus A11 and still behind Phase 10, so continue A14+.

## Strict packed64 A14

Single point:

- Add `EnsureHalfL1Zero()` and reuse `aicHalfL1Zero_` within the raw-MMAD L1 lifecycle.
- Replace repeated `ZeroHalfL1(aicHalfL1Zero_, ...)` in packed P7/P8 helper paths with size-checked reuse.
- Invalidate the cache in `Init()`, `InitRawMmadLocalBuffers()`, and `RestoreStage1MatmulLocalBuffers()`.

Guardrails:

- Did not modify P7/P8 formula order.
- Did not modify identity cache semantics.
- Did not remove BF16 pre-RHS GM writeback.
- Did not remove RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.37s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA14_l1_zero_cache_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.441`
  - `S=512`: `2.104`
  - `S=2048`: `2.915`
  - `S=10240`: `2.975`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA14_l1_zero_cache_baseline.h`
- SHA256:
  - `7c6d16dec16f54f29bff3eeec69baa1b8b77daba8917a6914dc912e675b66a4f`

Conclusion:

- A14 is correctness-safe and saved as a baseline.
- Large shapes improved versus A13, but small shapes regressed and the path remains slower than Phase 10.
- Continue A15+ one optimization at a time.

## Strict packed64 A15

Single point:

- Reuse the P8 first-layer `16x16` half identity cache into BF16 writeback.
- Removed only the `preparedHalfIdentityBlockSize_ = 0;` immediately before `StorePackedL1InverseToBf16Attn(paraIdx)`.
- Kept the post-writeback `preparedHalfIdentityBlockSize_ = 0;`.
- Did not remove `PrepareHalfL1IdentityByMmad()` cache-hit syncs.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS.
- `S=256` smoke PASS.
- `S=1024` smoke PASS.
- compare PASS:
  - `16 passed in 6.15s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA15_identity_reuse_writeback_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.463`
  - `S=512`: `2.142`
  - `S=2048`: `2.991`
  - `S=10240`: `3.032`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA15_identity_reuse_writeback_baseline.h`
- SHA256:
  - `c570e711c9793149a52269148c511b117d0d2b68c7f4921c68fb9fed90efb65e`

Conclusion:

- A15 is correctness-safe and saved as a baseline.
- Performance regressed versus A14 on all profiled shapes.
- Do not stack more optimization on top of A15; A16 should first restore the pre-writeback identity invalidation, then continue from the A14 behavior.

## Strict packed64 A16

Single point:

- Restore the `preparedHalfIdentityBlockSize_ = 0;` immediately before `StorePackedL1InverseToBf16Attn(paraIdx)`.
- This reverts the A15 performance-negative identity-cache reuse while keeping all A14 behavior.
- Active Stage1 SHA256 equals A14 baseline SHA256.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.18s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA16_restore_writeback_identity_invalidation_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.464`
  - `S=512`: `2.133`
  - `S=2048`: `2.937`
  - `S=10240`: `2.974`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA16_restore_writeback_identity_invalidation_baseline.h`
- SHA256:
  - `7c6d16dec16f54f29bff3eeec69baa1b8b77daba8917a6914dc912e675b66a4f`

Conclusion:

- A16 is correctness-safe and saved as a baseline.
- It intentionally restores the A14 code state after the A15 negative performance result.
- Continue A17+ from A16/A14 behavior; do not retry the A15 writeback identity-cache reuse.

## Strict packed64 A17

Single point:

- Removed the full `aicHalfL1Scratch_` zero-init from `LoadPacked64DiagonalInputsToL1()` on the strict full64 path.
- Moved the conservative scratch zero to `PreloadP8LowerLeftBlocksToLowerWork64Sync()` so fallback/tail paths keep the old behavior.
- Strict full64 path relies on `PreloadP8LowerLeftBlocksToLowerWork64NoWaitFull64()` to cover exactly the P8 lower-left blocks consumed by compose.

Guardrails:

- Did not modify P7/P8 formula order.
- Did not modify BF16 pre-RHS GM writeback.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.15s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA17_strict_full64_skip_lowerwork_zero_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.435`
  - `S=512`: `2.110`
  - `S=2048`: `2.938`
  - `S=10240`: `2.970`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA17_strict_full64_skip_lowerwork_zero_baseline.h`
- SHA256:
  - `8a7d00de0c4d64724b89a96416057ef54bef81ee555d8e609c439d4bb371391e`

Conclusion:

- A17 is correctness-safe and saved as a baseline.
- It improves `S=64/512/10240` slightly versus A16 and is roughly flat at `S=2048`.
- The path is still slower than Phase 10, so continue A18+ one optimization at a time.

## Strict packed64 A18

Single point:

- Removed the `ZeroHalfL1(aicHalfL1TileInv_, C64_BLOCK32 * C64_BLOCK32)` pre-zero in `P8Compose64From32Direct32()`.
- The `aicHalfL1TileInv_` `32x32` tmp tile is immediately overwritten by the first second-layer compose Fixpipe for `tmp = lower @ topInv`.

Guardrails:

- Did not modify P7/P8 formula order.
- Did not modify BF16 pre-RHS GM writeback.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.46s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA18_p8_second_layer_tmp_zero_cleanup_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.442`
  - `S=512`: `2.082`
  - `S=2048`: `2.916`
  - `S=10240`: `2.944`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA18_p8_second_layer_tmp_zero_cleanup_baseline.h`
- SHA256:
  - `18b7b4ace8ba012002210164a39080e8eff603a0a3867161cf90e63d7a5d0e1e`

Conclusion:

- A18 is correctness-safe and saved as a baseline.
- It improves `S=512/2048/10240` versus A17, while `S=64` regresses slightly.
- The path is still slower than Phase 10, so continue A19+ one optimization at a time.

## Strict packed64 A19

Single point:

- Removed the `ZeroHalfL1(tmp32, C32_PACKED_SIZE_U32 * C32_PACKED_SIZE_U32)` pre-zero in `P8ComposeFirstLayerPacked32Direct()`.
- The `tmp32` tile is immediately overwritten by the first-layer compose Fixpipe for `tmp = lower @ topInv`.

Guardrails:

- Did not modify P7/P8 formula order.
- Did not modify BF16 pre-RHS GM writeback.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.20s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA19_p8_first_layer_tmp_zero_cleanup_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.422`
  - `S=512`: `2.099`
  - `S=2048`: `2.901`
  - `S=10240`: `2.953`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA19_p8_first_layer_tmp_zero_cleanup_baseline.h`
- SHA256:
  - `445e8d8af9a1deff28924a7b17e96481297de5d385f2ee7e986462ef21590112`

Note:

- During result collection, active source briefly matched A18 SHA after profiling. The build staged source and package staged source were A19 SHA.
- Re-applied the same one-line A19 diff after profiling and verified active, build staged, and package staged sources all matched SHA `445e8d8af9a1deff28924a7b17e96481297de5d385f2ee7e986462ef21590112` before saving the baseline.

Conclusion:

- A19 is correctness-safe and saved as a baseline.
- It improves `S=64/2048` versus A18, while `S=512/10240` regress slightly.
- The path is still slower than Phase 10, so continue A20+ one optimization at a time.

## Strict packed64 A20

Single point:

- Removed the `ZeroHalfL1(aicHalfL1TileInv_, INVERSE_SHAPE * INVERSE_SHAPE)` pre-zero in `LoadPacked64DiagonalInputsToL1()`.
- In the strict full64 path, `aicHalfL1TileInv_` is later used as `diagScratch64` and overwritten by one-shot Fixpipe; fallback compact solve keeps its own local zero-init.

Guardrails:

- Did not modify P7/P8 formula order.
- Did not modify BF16 pre-RHS GM writeback.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- installed source SHA matched active/package staging SHA: `e65e616756b01b9d8dc08cc9e881885be5c532695a42a6ae68f35eaa7dbecc2b`.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.16s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA20_p7_tileinv_zero_cleanup_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.387`
  - `S=512`: `2.093`
  - `S=2048`: `2.925`
  - `S=10240`: `2.946`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA20_p7_tileinv_zero_cleanup_baseline.h`
- SHA256:
  - `e65e616756b01b9d8dc08cc9e881885be5c532695a42a6ae68f35eaa7dbecc2b`

Conclusion:

- A20 is correctness-safe and saved as a baseline.
- It improves `S=64/512/10240` versus A19 and regresses `S=2048` slightly.
- The path is still slower than Phase 10, so continue A21+ one optimization at a time.

## Strict packed64 A21

Single point:

- Removed the `ZeroHalfL1(aicHalfL1TilePower_, chunkSize_ * chunkSize_)` pre-zero in `LoadPacked64DiagonalInputsToL1()`.
- Strict full64 P7 consumes `diagPower64` through diagonal-only L0 loads, and the four diagonal blocks are overwritten by GM fixed-position loads.
- Kept `aicHalfL1P3Result_` pre-zero because P8 full-block packed loads still depend on zero upper/off-diagonal padding.

Guardrails:

- Did not modify P7/P8 formula order.
- Did not modify BF16 pre-RHS GM writeback.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- active/build staging/package staging SHA matched `18944df04588ad8f18721aa529522eef83afc3f1195265392550316529a65422`.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.38s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA21_p7_diagpower_zero_cleanup_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.380`
  - `S=512`: `2.091`
  - `S=2048`: `2.906`
  - `S=10240`: `2.935`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA21_p7_diagpower_zero_cleanup_baseline.h`
- SHA256:
  - `18944df04588ad8f18721aa529522eef83afc3f1195265392550316529a65422`

Conclusion:

- A21 is correctness-safe and saved as a baseline.
- It improves all four profiled shapes versus A20.
- The path is still slower than Phase 10, so continue A22+ one optimization at a time.

## Strict packed64 A22

Single point:

- Replaced the BF16 upper-triangle zero source in `StorePackedL1InverseToBf16Attn()`.
- Instead of zeroing `aicHalfL1Scratch_` for upper-zero writeback, the path now calls `EnsureHalfL1ZeroForMte1(INVERSE_SHAPE * INVERSE_SHAPE)` and writes zeros from `aicHalfL1Zero_`.
- Cache-hit kept `SyncFixToMte1()` and `SyncMte2ToMte1()` for writeback visibility.

Guardrails:

- Did not remove BF16 pre-RHS GM writeback.
- Did not skip upper-zero BF16 writeback.
- Did not modify P7/P8 formula order.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- active/build staging/package staging SHA matched `5bdec9c9755cec059180f60242a65d0750a3628312de2c58c11ce324a6385479`.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.14s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA22_bf16_upper_zero_source_cache_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.356`
  - `S=512`: `2.114`
  - `S=2048`: `2.966`
  - `S=10240`: `2.976`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA22_bf16_upper_zero_source_cache_baseline.h`
- SHA256:
  - `5bdec9c9755cec059180f60242a65d0750a3628312de2c58c11ce324a6385479`

Conclusion:

- A22 is correctness-safe and saved as a baseline.
- It improves `S=64` versus A21, but regresses `S=512/2048/10240`.
- Continue A23+ with a single-point check of whether the regression comes from the extra cache-hit visibility sync or from the zero-source reuse itself.

## Strict packed64 A23

Single point:

- Removed only the A22 cache-hit visibility sync in `EnsureHalfL1ZeroForMte1()`.
- Cache miss still zeroes `aicHalfL1Zero_`; upper-zero BF16 writeback still writes all upper blocks from `aicHalfL1Zero_`.

Guardrails:

- Did not remove BF16 pre-RHS GM writeback.
- Did not skip upper-zero BF16 writeback.
- Did not modify P7/P8 formula order.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.
- Did not retry A15 writeback identity-cache reuse.

Validation:

- build PASS.
- install PASS.
- active/build staging/package staging SHA matched `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`.
- Diff from A22 baseline was exactly the removal of `SyncFixToMte1()` and `SyncMte2ToMte1()` in `EnsureHalfL1ZeroForMte1()` cache-hit.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.09s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA23_bf16_upper_zero_cache_hit_nosync_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.355`
  - `S=512`: `2.068`
  - `S=2048`: `2.895`
  - `S=10240`: `2.938`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA23_bf16_upper_zero_cache_hit_nosync_baseline.h`
- SHA256:
  - `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`

Conclusion:

- A23 is correctness-safe and saved as a baseline.
- A23 confirms A22's large-shape regression was mainly from the extra cache-hit sync, not from reusing `aicHalfL1Zero_` itself.
- It improves all four shapes versus A22 and improves `S=64/512/2048` versus A21; `S=10240` remains slightly slower than A21.
- The path is still slower than Phase 10, so continue A24+ one optimization at a time.

## Strict packed64 A24

Single point:

- Prepared the `16x16` half identity once at the start of `StorePackedL1InverseToBf16Attn()`.
- Added prepared-identity BF16 writeback helpers so each lower/upper `16x16` block copy no longer calls `PrepareHalfL1IdentityByMmad()` itself.
- This did not retry A15's writeback identity cache reuse across the P8/writeback boundary.

Guardrails:

- Did not remove BF16 pre-RHS GM writeback.
- Did not skip upper-zero BF16 writeback.
- Did not modify P7/P8 formula order.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS.
- `S=256` smoke PASS.
- `S=1024` smoke PASS.
- compare PASS:
  - `16 passed in 6.18s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA24_writeback_identity_once_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.394`
  - `S=512`: `2.076`
  - `S=2048`: `2.929`
  - `S=10240`: `2.967`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA24_writeback_identity_once_baseline.h`
- SHA256:
  - `a11c220b0904d330c638b413485a2904311aa8ea8cb32057ab7bbd78b3e1ee74`

Conclusion:

- A24 is correctness-safe and saved as a baseline.
- Performance is negative versus A23 on most profiled shapes, so it is not the continuing mainline.
- A25 restores A23 behavior before any new optimization.

## Strict packed64 A25

Single point:

- Restored A23 writeback identity behavior by removing the A24 prepared-identity writeback helper path.
- Active source is byte-identical to A23/A25 baseline.

Guardrails:

- Did not remove BF16 pre-RHS GM writeback.
- Did not skip upper-zero BF16 writeback.
- Did not modify P7/P8 formula order.
- Did not modify resident L1 RHS consumption.
- Did not modify RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- active/build staging/package staging SHA matched `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`.
- `S=64` smoke PASS.
- `S=256` smoke PASS.
- `S=1024` smoke PASS.
- compare PASS:
  - `16 passed in 6.14s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA25_restore_writeback_identity_once_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.428`
  - `S=512`: `2.046`
  - `S=2048`: `2.892`
  - `S=10240`: `2.946`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA25_restore_writeback_identity_once_baseline.h`
- SHA256:
  - `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`

Conclusion:

- A25 is a restore phase, not a new performance optimization point.
- The current active path is A25/A23 behavior.
- Continue A26 from SHA `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`, one optimization at a time.

## Strict packed64 A26

Single point attempted:

- Optimize `StorePackedL1InverseToResidentL1()`.
- First candidate tried direct L1-to-L1 full packed tile copy from `aicHalfL1P3Result_` to resident L1.
- Second conservative candidate kept resident zero plus lower-block MMAD copy, but prepared `16x16` identity once before the 10 lower-block loop and reused a prepared-identity helper inside the loop.

Guardrails:

- BF16 pre-RHS GM writeback was restored and kept.
- Upper-zero BF16 writeback was kept.
- RHS operand L1 zero-init was kept.
- `RestoreRawMmadState()` was kept.

Validation:

- direct L1-to-L1 candidate:
  - build PASS.
  - install PASS.
  - `S=64` smoke FAIL:
    - `CPU vs ChunkGDRNeumann out max_abs=1.416709e-01`
    - `CPU vs ChunkGDRNeumann state max_abs=9.476510e-02`
- conservative prepared-identity candidate:
  - build PASS.
  - install PASS.
  - `S=64` smoke PASS.
  - `S=256` smoke PASS.
  - `S=1024` smoke FAIL:
    - `CPU vs ChunkGDRNeumann out max_abs=1.118436e-01`
    - `CPU vs ChunkGDRNeumann state max_abs=1.886734e-01`
    - worst actual was close to zero.

Revert:

- Active Stage1 restored from A25 baseline.
- Restored SHA256:
  - `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`
- recovery build PASS.
- recovery install PASS.
- recovery `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

Conclusion:

- A26 failed and was reverted; no A26 baseline saved.
- Do not continue by bypassing the per-block `PrepareHalfL1IdentityByMmad()` synchronization boundary in resident copy.
- Continue A27 from A25/A23 baseline with a different single point.

## Strict packed64 A27

Single point attempted:

- Fused lower BF16 GM writeback and resident L1 writeback from one L0C.
- The attempted helper wrote the lower block's L0C first to BF16 GM and then to resident L1, while keeping resident zero-init and upper-zero BF16 writeback.

Guardrails:

- Did not remove BF16 pre-RHS GM writeback.
- Did not skip upper-zero BF16 writeback.
- Did not remove resident L1 zero-init.
- Did not modify P7/P8 formula order.
- Did not modify final RHS semantics.
- Did not skip RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke timed out after 90s with no compare output after extension load.

Revert:

- Active Stage1 restored from A25 baseline.
- Restored SHA256:
  - `4f440a78ba2123cfc6d5caffa75b0081cfd7caf9253b2e211b2785ad8de8da1a`
- recovery build PASS.
- recovery install PASS.
- recovery `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- recovery `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

Conclusion:

- A27 failed and was reverted; no A27 baseline saved.
- Do not retry same-L0C dual Fixpipe to GM + resident without a new synchronization design.
- Continue A28 from A25/A23 baseline with a different single point.

## Strict packed64 A28

Single point:

- P7 strict packed64 diag-only MMAD no longer clears full `64x64` L0A/L0B before every diag-only MMAD.
- Added `ZeroPacked64DiagOnlyL0Operands()` and call it once at `P7Packed64DiagonalNeumannOneShotToResult()` entry.
- `RawMmadPacked64DiagOnly()` keeps the old clear behavior by default; only P7 one-shot passes `clearL0=false`.

Guardrails:

- Did not modify P7 Neumann formula.
- Did not modify P8 formula order.
- Did not remove BF16 pre-RHS GM writeback.
- Did not skip upper-zero BF16 writeback.
- Did not modify resident L1 copy.
- Did not skip RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.
- Did not retry A26 resident prepared-identity copy or A27 same-L0C dual Fixpipe.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.17s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA28_p7_l0_zero_once_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.437`
  - `S=512`: `2.065`
  - `S=2048`: `2.889`
  - `S=10240`: `2.939`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA28_p7_l0_zero_once_baseline.h`
- SHA256:
  - `624a74c2f3e7cf026708a358e0cdaf1af2e41537468529e32cd6c048c778a449`

Conclusion:

- A28 is correctness-safe and saved as a baseline.
- It slightly improves `S=2048/10240` versus A25, slightly regresses `S=64/512`, and is still slower than Phase 10.
- Continue A29+ one optimization at a time.

## Strict packed64 A29

Single point:

- P8 first-layer packed32 MMAD no longer clears full `32x32` L0A/L0B before each of its two MMADs.
- Added `ZeroPacked32PrimaryL0Operands()` and call it once at `P8ComposeFirstLayerPacked32Direct()` entry.
- `RawMmadP8FirstLayerLowerTopPacked32()` and `RawMmadP8FirstLayerBottomTmpPacked32()` keep the old clear behavior by default; only the packed first-layer path passes `clearL0=false`.

Guardrails:

- Did not modify P8 formula order: `tmp = lower @ topInv`, then `out = bottomInv @ tmp`.
- Did not modify BF16 pre-RHS GM writeback or upper-zero writeback.
- Did not modify resident L1 copy or RHS consumption.
- Did not skip RHS operand L1 zero-init.
- Did not remove `RestoreRawMmadState()`.
- Did not retry A26/A27 failed resident-writeback paths.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.16s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA29_p8_first_layer_l0_zero_once_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.429`
  - `S=512`: `2.078`
  - `S=2048`: `2.872`
  - `S=10240`: `2.914`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA29_p8_first_layer_l0_zero_once_baseline.h`
- SHA256:
  - `e519d49eb4533bdd80bee3c8955508fb8793ec8f48640cff3ee6aa0c4a4ac638`

Conclusion:

- A29 is correctness-safe and saved as a baseline.
- It improves `S=64/2048/10240` versus A28, regresses `S=512`, and is still slower than Phase 10.
- Continue A30+ one optimization at a time.

## Redundant Operations To Remove

Current active before A30 is A29 baseline. The following operations are not final algorithm requirements; they are temporary correctness-side-effect carriers or copy implementations that must be removed in gated phases:

- `StorePackedL1InverseToResidentL1()` still uses resident zero + per-lower-block `X @ I16` copy. A26 direct full-tile copy failed, and A27 same-L0C dual Fixpipe to BF16 GM + resident L1 timed out at `S=64`; do not keep pushing this writeback fusion path without a new synchronization design.
- `16x16` half identity is still generated by `PrepareHalfL1IdentityByMmad()` for BF16 writeback, resident copy, and pack/compact copy helpers. This is redundant when the intent is copy/layout movement; final code should use direct movement or a dedicated layout-copy path, not generate `I16`. If a transitional MMAD-copy still needs `I16`, source it from the future operator-lifetime `identity64` diagonal fractal instead of generating a separate `16x16` identity.
- `identity64` already has an independent `aicHalfL1Identity64_` tile, but it is still cached only by explicit prepared flags. It gets invalidated by `ParaChunkAIC()` and can be affected by `PrepareHalfL1IdentityByMmad()` clearing `packed64IdentityGmPrepared_`; final target is a L1-resident `identity64` generated once per operator lifecycle and reused by any remaining transitional identity consumer.
- A30 tried only removing the `ParaChunkAIC()` `packed64IdentityPrepared_ = false;` invalidation. It did not remove the `16x16` identity helpers, BF16 GM side effects, resident copy, or `RestoreRawMmadState()`.
- A30 failed at `S=1024` and was reverted to A29, so the para-group identity64 invalidation remains a required lifecycle protection for now.
- BF16 `attnWsGm_` pre-RHS inverse writeback is not the RHS mathematical input because fused RHS reads `aicHalfL1ResidentInv_`. It must be treated as a side-effect carrier and later deleted after the real synchronization/cache/lifecycle dependency is replaced explicitly.
- BF16 upper-zero writeback is not lower-triangular inverse math; it is tied to the current BF16 GM materialization side effect and should be removed after that dependency is isolated.
- `RestoreRawMmadState()` is a raw-MMAD state restoration cost, not inverse math. It should be narrowed or deleted by testing individual state items rather than kept as a permanent algorithm step.

## Strict packed64 A30 failed and reverted

Single point:

- Removed `packed64IdentityPrepared_ = false;` at the start of `ParaChunkAIC()` after `CrossCoreWaitFlag(0x7)`.
- Did not change P7/P8 math, BF16 GM writeback, resident L1 copy, RHS operand zero-init, or `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke FAIL:
  - `CPU vs ChunkGDRNeumann out max_abs=1.109401e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.557053e-01`
  - worst actual values were close to zero, matching a stale/invalid resident identity lifecycle failure.

Recovery:

- Removed an un-gated P8 second-layer packed32 helper candidate that had remained in active but was not part of A29.
- Restored active Stage1 to byte-identical A29 baseline:
  - SHA256 `e519d49eb4533bdd80bee3c8955508fb8793ec8f48640cff3ee6aa0c4a4ac638`
- recovery build PASS.
- recovery install PASS.
- recovery `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

Conclusion:

- A30 is not correctness-safe and no baseline was saved.
- `packed64IdentityPrepared_` cannot simply outlive the para-group boundary because `pipe_->Reset()` / raw-MMAD buffer initialization can invalidate the underlying `aicHalfL1Identity64_` contents. The next identity64 optimization needs a true persistent buffer/lifecycle design, not only a longer prepared flag.

## Strict packed64 A31 candidate failed and reverted

Timestamp:

- 2026-05-11 12:46 CST.

Intent:

- Start the real `identity64` lifecycle redesign after A30 proved that simply extending `packed64IdentityPrepared_` across `ParaChunkAIC()` is unsafe.
- Check whether `identity64` could be persisted through GM and reloaded after raw-MMAD buffer reset instead of relying on stale TSCM contents.

Single point attempted:

- Changed `EnsurePacked64Identity()` to write a full `64x64` FP32 identity into `tmpWsGm_`, then reload it into `aicHalfL1Identity64_`.
- Removed `packed64IdentityGmPrepared_ = false` from `PrepareHalfL1IdentityByMmad()`.
- Did not intentionally change P7/P8 math, BF16 GM writeback, resident L1 copy, RHS operand zero-init, or `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- `S=64` smoke FAIL:
  - `CPU vs ChunkGDRNeumann out max_abs=9.601222e-02`
  - worst actual values were close to zero.

Failure interpretation:

- This is not a correctness-safe A31 baseline.
- `aicHalfL1Identity64_` is allocated from `aicHalfL1Buff_` / TSCM, and `pipe_->Reset()` can invalidate local buffer contents. A true lifecycle design cannot preserve only the prepared flag across that boundary.
- The attempted GM reload path is also invalid as written because `LoadGmFloatBlockToHalfL1ByMmad(tmpWsGm_, aicHalfL1Identity64_, chunkSize_)` uses `tmpWsGm_` as the input data while the helper's copy semantics also depend on the GM identity RHS region. Reusing the same GM region for both roles does not form a valid copy path.

Recovery:

- Reverted active Stage1 to byte-identical A29 baseline.
- During recovery there were concurrent build/compare jobs that temporarily made source, build staged source, and installed source disagree:
  - active source temporarily drifted through A28-like / non-A29 hashes.
  - build staged source temporarily held an A31 failed-candidate hash and then a non-A29 `C64_L0C_TILE_ELEMS` helper variant.
  - installed source temporarily held the A31 failed-candidate hash.
- Waited for concurrent build/compare jobs to finish before the final recovery.
- Rebuilt package from byte-identical A29 active source.
- Reinstalled the rebuilt package.
- Final verification: source, baseline, build staged source, and installed source all match A29:
  - SHA256 `e519d49eb4533bdd80bee3c8955508fb8793ec8f48640cff3ee6aa0c4a4ac638`
- recovery `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`

Conclusion:

- A31 failed at the first smoke and no baseline was saved.
- Do not retry the full64 identity-via-GM reload variant that calls `LoadGmFloatBlockToHalfL1ByMmad(tmpWsGm_, aicHalfL1Identity64_, chunkSize_)` while using the same `tmpWsGm_` area as the identity RHS.
- Current active source/package are restored to A29. The next A31 design must use either separate GM identity source/RHS regions or a direct GM FP16/NZ load path into `aicHalfL1Identity64_`, and must still pass build/install, `S=64 -> 256 -> 1024`, compare, and performance before any baseline is saved.

## Concurrent writer blocker after A31 recovery

Timestamp:

- 2026-05-11 13:01 CST.

Observed state:

- A29 installed package had already passed recovery `S=1024` smoke:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- Repeated hash monitoring showed active source being overwritten seconds after local recovery patches.
- Final observed hashes:
  - active source: `624a74c2f3e7cf026708a358e0cdaf1af2e41537468529e32cd6c048c778a449` (A28)
  - build staged source: `624a74c2f3e7cf026708a358e0cdaf1af2e41537468529e32cd6c048c778a449` (A28)
  - installed source: `e519d49eb4533bdd80bee3c8955508fb8793ec8f48640cff3ee6aa0c4a4ac638` (A29)
- Therefore source/build/installed are not currently aligned.

Likely cause:

- Multiple older Codex sessions are still alive in this worktree.
- The strongest candidate is the older process with rollout log:
  - `/home/developer/.codex/sessions/2026/05/10/rollout-2026-05-10T23-51-37-019e1296-41e7-77a2-9c07-24ace15ef48c.jsonl`
- That session's log contains the same A33/A29 mixed hunk (`C64_L0C_TILE_ELEMS` and P8 zero-once changes) seen in the active-file drift.

Conclusion:

- Stop active kernel edits until the old writer is stopped.
- Do not save a new baseline from the current live state.
- Before continuing, restore the chosen baseline explicitly and verify active, build staged, and installed source hashes are aligned after a short monitoring window.

## Strict packed64 A33 completed from clean A28

Timestamp:

- 2026-05-11 13:10 CST.

Recovery before A33:

- Active Stage1 had drifted to an A29/A33 mixed candidate and then back to A28 during concurrent writer activity.
- Re-established A28 as the clean source point, then re-applied only the intended A33 change.
- Final A33 diff against A28 contains only:
  - `constexpr uint32_t C64_L0C_TILE_ELEMS = C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32;`
  - `cPower = aicL0C_[C64_L0C_TILE_ELEMS]` inside `P7Packed64DiagonalNeumannOneShotToResult()`.
- Explicitly excluded the A29 P8 first-layer L0 zero-once hunk.

Validation:

- build PASS.
- install PASS.
- source and installed Stage1 SHA matched before and after compare:
  - `c138b344294996b07264623270f4c1eaa9e0f4e33d5a19c2b44c451699e15974`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- seed40 boundary PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=2.595484e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.108155e-03`
- full compare PASS:
  - `16 passed in 6.15s`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA33_p7_l0c_const_offset_baseline.h`
- SHA256 `c138b344294996b07264623270f4c1eaa9e0f4e33d5a19c2b44c451699e15974`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA33_p7_l0c_const_offset_20260511/performance_compare.csv`
- neumann/normal ratio:
  - `S=64`: `1.463`
  - `S=512`: `2.079`
  - `S=2048`: `2.877`
  - `S=10240`: `2.935`

Conclusion:

- A33 is correctness-safe and is the current active baseline.
- It is a cleanup/constant-offset phase, not a performance win.
- It remains slower than Phase 10, so continue A34+ one optimization at a time.

## Redundancy audit after A34 handoff

Timestamp:

- 2026-05-11 current handoff discussion.

Direct `I16` to packed64 L1:

- The target is valid in principle: put `16x16 identity` directly into the four packed64 diagonal positions of `aicHalfL1Identity64_`.
- Current code does not do that. It writes/reloads a compact `16x16` identity through GM, clears `identity64`, then uses four `BuildPacked64IdentityDiag16()` calls to Fixpipe it into `(0,0)/(16,16)/(32,32)/(48,48)`.
- A correct direct version must write NZ packed fractals at the exact diagonal offsets, keep all off-diagonal/upper padding zero, and provide the same MTE1 visibility after `pipe_->Reset()` / raw-MMAD local-buffer init.
- A32 direct full64 identity generation failed at `S=64`; do not retry a linear/full64 writer. The next valid attempt would be a single-point direct NZ packed identity64 L1 writer with the normal build/install + `S=64/256/1024` + compare gate.

Detailed redundant items:

- `identity64` construction: GM write/reload plus four `I16 @ I16` placement operations should collapse to one correct packed L1 construction.
- `identity64` lifecycle: `packed64IdentityPrepared_` cannot simply survive para groups; A30 failed at `S=1024`. A future design needs an explicit raw-MMAD buffer epoch or a storage location outside `pipe_->Reset()` invalidation.
- `PrepareHalfL1IdentityByMmad()` and `X @ I16` copy helpers: BF16 writeback, resident copy, and pack/compact movement still use MMAD as copy/layout movement.
- `StorePackedL1InverseToResidentL1()`: resident RHS input is still built by zeroing the full tile and copying each lower `16x16` block through `X @ I16`. A26/A27 failed, so replacement needs new visibility design.
- `StorePackedL1InverseToBf16Attn()`: BF16 GM inverse materialization is not RHS math input, but current correctness depends on its side effects. Deleting complete writeback or upper-zero failed at `S=1024`.
- BF16 upper-zero writeback: not lower-triangular inverse math; keep until the BF16 materialization side effect is explicitly replaced.
- RHS operand zero-init: full tile should be overwritten in theory, but Phase 12 failed when skipping the zero; split content-zero and sync/visibility roles before retrying.
- P8 compose syncs: A35/A36 showed direct sync deletion is unsafe. Future attempts must isolate tmp/result producer-consumer visibility.
- P8 second-layer: already `32x32` packed compose. Remaining optimization is around load/Fixpipe/tmp lifetime, not first-layer-style independent `16x16` lane packing.
- `RestoreRawMmadState()`: not math, but whole-function deletion was performance-negative; validate state items one by one.

## Strict packed64 A38 failed and reverted

Timestamp:

- 2026-05-11 15:25-15:56 CST.

Pre-check:

- Active Stage1, A34 baseline, package staging source, and installed Stage1 were checked by SHA before the edit.
- Active Stage1 and A34 baseline matched:
  - SHA256 `ccec6f1a0e7bfe78b5102ad4ab8cf383441a0de40e1940ad09ca055037be5cdb`
- Selector files matched between source and installed package.

Single point attempted:

- Replaced only `EnsurePacked64Identity()` construction.
- Removed the `16x16` identity GM write/reload and the four `BuildPacked64IdentityDiag16()` calls.
- Added a direct scalar NZ writer that zeroed `aicHalfL1Identity64_`, then wrote `half(1.0f)` to the four diagonal fractals at:
  - `C64_DIAG_NZ_OFFSET0`
  - `C64_DIAG_NZ_OFFSET1`
  - `C64_DIAG_NZ_OFFSET2`
  - `C64_DIAG_NZ_OFFSET3`
- Did not change prepared-flag lifecycle, P7/P8 formulas, BF16 writeback, resident copy, RHS, or `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- source, package staging source, and installed source all matched the A38 candidate:
  - SHA256 `fa0afaef367a9a50741426f9a6ae3e6daa0c57da86031b89bfddba97b00d1b07`
- `S=64` smoke FAIL:
  - `CPU vs ChunkGDRNeumann out max_abs=3.430384e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.232193e-01`
  - worst out: `expect=1.39383465e-01`, `actual=4.82421875e-01`
  - worst state: `expect=1.30177289e-01`, `actual=6.95800781e-03`

Rollback:

- Immediately reverted active Stage1 to byte-identical A34 baseline.
- Rebuilt and reinstalled the A34 package.
- Final source/package staging/installed/A34 baseline SHA all match:
  - SHA256 `ccec6f1a0e7bfe78b5102ad4ab8cf383441a0de40e1940ad09ca055037be5cdb`
- Recovery `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`

Conclusion:

- A38 is not correctness-safe and no baseline was saved.
- Do not retry this exact scalar direct NZ writer variant.
- The failure means direct placement still lacks some property of the current MMAD/Fixpipe construction, likely around the generated NZ fractal content/layout or producer visibility semantics.
- Active/source/installed are restored to A34. Do not proceed to A39 from this failed A38.

## A38-forward and same-stride redundancy audit

Timestamp:

- 2026-05-11 17:12 CST and after.

A38-forward current candidate:

- Implemented a new identity64 candidate using a static GM constant:
  - `CGDR_IDENTITY16_NZ4` contains four consecutive `16x16` NZ identity fractals.
  - `EnsurePacked64Identity()` zeroes `aicHalfL1Identity64_`.
  - It then issues one `DataCopy` from the static GM constant to the four packed64 diagonal NZ offsets with:
    - `blockCount = 4`
    - `blockLen = RAW_FRACTAL_BYTES / BLOCK_SIZE`
    - `srcStride = 0`
    - `dstStride = (C64_DIAG_NZ_OFFSET1 - C64_DIAG_NZ_OFFSET0 - RAW_HALF_FRACTAL_ELEMS) * sizeof(half) / BLOCK_SIZE`
- This still changes only identity64 construction.
- It does not change prepared-flag lifecycle, P7/P8 formulas, BF16 writeback, resident copy, RHS, or `RestoreRawMmadState()`.
- Active/source SHA:
  - `d533cb22580c67d5c1996c97ede5313854b255d61f8412c0d1fa1614f3948fcf`
- Installed source SHA:
  - `d533cb22580c67d5c1996c97ede5313854b255d61f8412c0d1fa1614f3948fcf`
- Validation so far:
  - build PASS
  - install PASS
  - `S=64` smoke PASS:
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- Still required before saving a baseline:
  - `S=256`
  - `S=1024`
  - full compare

Same-stride redundancy audit:

- Rule added to `AGENTS.md`: when several same-shape `16x16` / packed fractals have constant source and destination stride, record them as single `DataCopy` / `LoadData` / `Fixpipe` candidates and modify each as its own gated phase.
- P7 diagonal GM->packed64 L1 input:
  - Current full64 path calls `LoadPacked64DiagonalInputBlock()` for bases `0/16/32/48`.
  - Each base loads both `L_powHalfWsGm_ -> aicHalfL1TilePower_` and `INVInitHalfWsGm_ -> aicHalfL1P3Result_`.
  - Full64 total is 8 `DataCopy` helper calls.
  - The four blocks of each tensor have constant GM source offset gap `MatrixOffset(16,16)` and constant packed64 diagonal destination gap `(C64_PACKED_BLOCK_COUNT + 1) * RAW_HALF_FRACTAL_ELEMS`.
  - Correct target is two `Nd2NzParams DataCopy` calls: one for `L_pow`, one for `INVInit`.
- P8 lower-left GM->lowerWork64 L1 input:
  - Current full64 path loads `(16,0) 16x16`, `(48,32) 16x16`, and `(32,0) 32x32`.
  - The two first-layer `16x16` blocks are a paired-load candidate.
  - The `(32,0) 32x32` second-layer block is a separate shape and should not be mixed into that phase.
- P7 packed64 diagonal L1->L0:
  - `LoadPacked64DiagOnlyToL0()` currently issues four `LoadData()` calls for `C64_DIAG_NZ_OFFSET0/1/2/3`.
  - The offsets are equally spaced in source and destination.
  - Next valid phase should test a single/fewer-load implementation after confirming `LoadData2DParams.dstGap` and repeat semantics.
- P8 first-layer packed32 L1->L0:
  - `LoadP8FirstLayerLowerToPacked32L0A()`, `LoadP8FirstLayerTopInvToPacked32L0B()`, `LoadP8FirstLayerBottomInvToPacked32L0A()`, and `LoadPacked32DiagOnlyToL0B()` each load two independent `16x16` lanes.
  - These are paired-load candidates, independent from P7 diagonal load cleanup.
- Copy-out / exposure:
  - `StorePackedL1InverseToBf16Attn()` writes lower blocks and upper-zero blocks one `16x16` at a time; lower row prefixes are possible future coalescing candidates, but BF16 GM materialization side effects are unresolved.
  - `StorePackedL1InverseToResidentL1()` zeroes resident and copies lower blocks one at a time through `X @ I16`; lower row prefixes are possible future coalescing candidates, but A26/A27 showed the resident visibility design must be fixed before replacement.

Additional copy-out audit after user question:

- P7 diagonal-only Fixpipe copy-out:
  - `HalfL1PowerPacked64DiagOnlyOnC()` and P7 result updates call `FixpipePacked64FullToHalfL1()` even though the next P7 loads only `C64_DIAG_NZ_OFFSET0/1/2/3`.
  - The redundant part is the repeated full `64x64` packed writeback of off-diagonal zero fractals.
  - Planned fix direction: strided/paired diagonal-fractal Fixpipe copy-out as its own phase, after proving `FixpipeParamsV220.ndNum/srcNdStride/dstNdStride` units for NZ source/destination.
- P8 first-layer output copy-out:
  - First-layer tmp writes full packed32 by `CopyL0CMatrixToHalfL1(tmp32, 32, 32)`, then `LoadPacked32DiagOnlyToL0B()` reads only two diagonal lanes.
  - First-layer final writes full packed32 to `aicHalfL1TilePower_`, then copies only two lanes to result64 with two `CopyHalfL1PackedBlockToCompact()` calls.
  - Planned fix direction: packed32 two-lane Fixpipe/output placement, separate from the A41 L1->L0 load coalescing phase.
- BF16/resident row-prefix copy-out:
  - BF16 GM writeback lower rows are row prefixes: `16x16`, `16x32`, `16x48`, `16x64`; they are not one global equal-stride triangular copy.
  - BF16 still needs half->bf16 conversion and the existing GM materialization side effect, so direct same-dtype `DataCopy` is not a valid replacement.
  - Resident L1 exposure has the same row-prefix shape but is RHS-visible state; direct copy/Fixpipe replacement needs a new visibility proof because A26/A27 already failed.

Decision:

- Do not mix any of these redundancy cleanups into A38-forward.
- Finish A38-forward gate first.
- If it passes, save a baseline, then start P7 diagonal GM->packed64 L1 input coalescing as the next single optimization.

## Strict packed64 A38-forward PASS

Timestamp:

- 2026-05-11 17:12-17:28 CST.

Single point:

- Replace identity64 construction with a static-GM source and one strided `DataCopy`.
- `CGDR_IDENTITY16_NZ4` stores four consecutive `16x16` NZ identity fractals.
- `EnsurePacked64Identity()` now:
  - zeros `aicHalfL1Identity64_`
  - binds `GlobalTensor<half>` to `CGDR_IDENTITY16_NZ4`
  - copies the four identity fractals to `C64_DIAG_NZ_OFFSET0/1/2/3` via one `DataCopy`
  - keeps `packed64IdentityPrepared_` lifecycle unchanged

Validation:

- build PASS.
- install PASS.
- source/installed SHA aligned:
  - `d533cb22580c67d5c1996c97ede5313854b255d61f8412c0d1fa1614f3948fcf`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.34s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA38_identity_static_gm_datacopy_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.340`
  - `S=512`: `2.013`
  - `S=2048`: `2.846`
  - `S=10240`: `2.891`

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA38_identity_static_gm_datacopy_baseline.h`
- SHA256:
  - `d533cb22580c67d5c1996c97ede5313854b255d61f8412c0d1fa1614f3948fcf`

Conclusion:

- A38-forward is correctness-safe and saved as a baseline.
- It improves A34 at `S=64/512/2048` and is roughly flat/slightly worse at `S=10240`.
- Next single point is A39: coalesce P7 diagonal GM->packed64 L1 input copies into one `DataCopy` per source tensor.

## Strict packed64 A39 PASS

Timestamp:

- 2026-05-11 17:30-18:29 CST.

Single point:

- Coalesce P7 full64 diagonal GM->packed64 L1 input loads only.
- Added `CleanPacked64DiagonalHalfBlocks()` and `LoadGmHalfPacked64DiagonalBlocksToPackedL1NoWait()`.
- `LoadPacked64DiagonalInputsToL1()` full64 path now:
  - zeroes `aicHalfL1P3Result_`
  - copies four `L_powHalfWsGm_` diagonal `16x16` blocks to `aicHalfL1TilePower_` with one `Nd2NzParams DataCopy`
  - copies four `INVInitHalfWsGm_` diagonal `16x16` blocks to `aicHalfL1P3Result_` with one `Nd2NzParams DataCopy`
  - issues one `SyncMte2ToMte1()` before returning
- Tail/fallback path remains unchanged.
- Did not change prepared-flag lifecycle, P7/P8 formulas, BF16 writeback, resident copy, RHS, or `RestoreRawMmadState()`.

Validation:

- build PASS.
- install PASS.
- active Stage1 and saved A39 baseline SHA:
  - `57ed59cf35ce8c7c9e941361729837d427413d2cf68f001f8c44c0bddb579e9f`
- `S=64` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=5.276948e-04`
- `S=256` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=9.178072e-04`
- `S=1024` smoke PASS:
  - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
  - `CPU vs ChunkGDRNeumann state max_abs=1.006737e-03`
- compare PASS:
  - `16 passed in 6.24s`

Performance:

- Output:
  - `tests/pytest/prof_out/packed64_phaseA39_p7_diag_gm_to_l1_coalesced_20260511/performance_compare.csv`
- ratio neumann/normal:
  - `S=64`: `1.234`
  - `S=512`: `1.979`
  - `S=2048`: `2.872`
  - `S=10240`: not trusted in this run; both normal and neumann warmup hit `libopapi.so` symbol lookup errors.

Baseline:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA39_p7_diag_gm_to_l1_coalesced_baseline.h`
- SHA256:
  - `57ed59cf35ce8c7c9e941361729837d427413d2cf68f001f8c44c0bddb579e9f`

Install/package note:

- This worktree keeps `op_kernel/p8afullocal/` untracked, and the installed package payload does not preserve the active `p8afullocal/...p3l1_p8a.h` path.
- Future installed checks must not assume an installed same-name Stage1 header exists. Use active/baseline SHA, package/build output, kernel object state, and smoke/compare evidence together.
- Before A40, re-run build/install from the A39 baseline because the currently visible top-level custom OPP directory looked like a stale `tri_inverse` install surface while older `chunk_gated_delta_rule_neumann` artifacts remained under nested vendor directories.

Conclusion:

- A39 is correctness-safe and saved as a baseline.
- It reduces the P7 diagonal input MTE2 helper count in the full64 path from 8 per two tensors to 2 per two tensors.
- Next single point is A40: coalesce P7 packed64 diagonal L1->L0 loads.

## Strict packed64 A40 start

Planned single point:

- Replace the four `LoadData()` calls in `LoadPacked64DiagOnlyToL0()` with one diagonal-strided `LoadData()`.
- Local API check:
  - `LoadData2DParams.srcStride` is the source stride between consecutive repeats in 512B fractal units.
  - `LoadData2DParams.dstGap` is the destination gap between consecutive repeats in 512B units.
  - packed64 diagonal offset gap is `C64_DIAG_NZ_OFFSET1 - C64_DIAG_NZ_OFFSET0 = 5 * RAW_HALF_FRACTAL_ELEMS`, so the candidate is `repeatTimes=4`, `srcStride=5`, `dstGap=4`.

Guardrails:

- Do not change P8, BF16, resident, RHS, prepared flags, or formulas.
- Do not mix copy-out/Fixpipe cleanup into A40; those remain A42/A43/A44+.

Current A40 failure state:

- Earlier active/installed Stage1 SHA256:
  - `141e2a20a1fffe8d1267bd006f51f7c8099167eab865331dcdbeebce82ebb051`
- Installed kernel object SHA256:
  - `b6374308d943ce40dac7bea2cf0f13bf2637472ee6080f069264ff53e039f845`
- Change versus A39:
  - added `C64_DIAG_NZ_STRIDE_FRACTALS = 5`
  - added `C64_DIAG_L0_LOAD_DST_GAP_FRACTALS = 4`
  - changed `LoadPacked64DiagOnlyToL0()` to one `LoadData()` with `repeatTimes=4`, `srcStride=5`, `dstGap=4`
  - added an intermediate `SyncMte1ToM(); SyncMToMte1();` between L0A and L0B diagonal loads
- Observed validation:
  - initial coalesced A40 without intermediate sync: `S=64/256` PASS, `S=512/1024` FAIL
  - current candidate with intermediate sync: `S=64/256/512` PASS, `S=1024` FAIL
  - `S=1024` failure has out/state errors with worst actual values close to zero, consistent with a load/visibility/layout issue rather than a P7/P8 formula change.

Forward debug and isolation:

- Stay on A40; do not start A41/A42 while this fails.
- L0A coalesced, L0B restored to four loads:
  - source SHA256 `d2cea029d5f545134044062161a4ba19e727cb110e06f88097c219a102723379`
  - installed object SHA256 `9e20c6a2def7c13eb72ebead28860ac10df6a225d3175b06a693ea6c1f53417a`
  - `S=1024` FAIL: out max_abs `1.017778e-01`, state PASS
  - conclusion: L0A one-shot coalescing is unsafe even when L0B is restored.
- L0A restored to four loads, L0B coalesced:
  - source/installed SHA256 `68992384584c14ba0228b59a67450fb884a3b918570b3d4d565e221bb83c3ab3`
  - installed object SHA256 `82397c7e659adfa635f3f7b025795cc32c5e17d16e486508e00733a9f2d64092`
  - `S=1024` PASS
  - conclusion: `ifTranspose=true` L0B one-shot coalescing is safe; the issue is L0A no-transpose layout/gap.
- Current source state after this audit:
  - active source SHA256 `fcd1ea657d4d1fcacd36979e12557dd04b195234dc968097f6247baece0362eb`
  - installed source SHA256 `68992384584c14ba0228b59a67450fb884a3b918570b3d4d565e221bb83c3ab3`
  - installed object SHA256 `82397c7e659adfa635f3f7b025795cc32c5e17d16e486508e00733a9f2d64092`
  - active source is an unverified A40 paired-load experiment: L0A two paired `repeatTimes=2` loads, L0B one `repeatTimes=4` coalesced load.
  - installed artifacts still correspond to the L0B-only passing isolation, so current source must be rebuilt and reinstalled before any smoke result is attributed to it.

Additional redundancy audit after copy-out question:

- P3/fallback diagonal store-to-GM is also a copy-out redundancy:
  - `StoreP3DiagonalResultsToGm()` calls `StorePackedL1DiagonalBlockToGmHalfFrom()` per diagonal block.
  - Each block performs `CopyHalfL1PackedBlockToCompact()` plus `CopyHalfL1MatrixToGmHalfByMmad()`, so it is another `X @ I16` layout-copy substitute.
  - It is not part of the strict full64 A40 target; optimize only if tail/fallback is restored as a separate phase.
- P7 diagonal-only Fixpipe copy-out:
  - `HalfL1PowerPacked64DiagOnlyOnC()` and P7 result updates still call `FixpipePacked64FullToHalfL1()` and write a full packed `64x64` tile.
  - The consumer path only reloads the four diagonal fractals through `LoadPacked64DiagOnlyToL0A/B()`.
  - Candidate direction is a strided or paired diagonal-fractal Fixpipe copy-out; this must be A42-style, after proving `FixpipeParamsV220` stride units and source layout.
- P8 first-layer output copy-out:
  - `P8ComposeFirstLayerPacked32Direct()` writes full packed32 tmp/result, then copies only two lanes into result64 through two `CopyHalfL1PackedBlockToCompact()` calls.
  - Candidate direction is packed32 two-lane Fixpipe/output placement, separate from A41 L1->L0 load coalescing.
- P8 second-layer copy-out is not the same redundancy:
  - first `32x32 tmp` is fully consumed by the second-layer MMAD.
  - final `(32,0) 32x32` result is a real lower-left block, not two independent diagonal lanes.
- L0 operand clear has a related movement cost:
  - `ZeroPacked64DiagOnlyL0Operands()` and the P8 first-layer zero-once path clear L0 by copying a full L1 zero tile into L0A/L0B.
  - A28/A29 already reduced repeated clears to one-time clears; any direct L0 clear or partial clear experiment must be its own phase.
- BF16/resident row-prefix copy-out:
  - BF16 lower writeback and resident exposure both still walk one `16x16` block at a time through `X @ I16` movement.
  - The lower rows are row prefixes (`16x16`, `16x32`, `16x48`, `16x64`), not one equal-stride triangular copy.
  - BF16 needs half-to-bf16 conversion and unresolved materialization side effects; resident is RHS-visible state. These must be split into later phases, not mixed into A40.

Final A40 result:

- Source/installed/baseline SHA256:
  - `fcd1ea657d4d1fcacd36979e12557dd04b195234dc968097f6247baece0362eb`
- Installed kernel object SHA256:
  - `d7a77f3d7e0fd243ed22727ce3224348a9e9f7fb14297d0053b5dc927d8315b3`
- Final implementation:
  - L0A no-transpose diagonal load uses two paired `repeatTimes=2` loads.
  - L0B transpose diagonal load uses one coalesced `repeatTimes=4` load.
  - The intermediate `SyncMte1ToM(); SyncMToMte1();` between L0A and L0B loads remains.
- Validation:
  - build PASS.
  - install PASS.
  - `S=64` smoke PASS.
  - `S=256` smoke PASS.
  - `S=1024` smoke PASS.
  - compare PASS: `16 passed in 6.17s`.
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA40_p7_diag_l1_to_l0_paired_baseline.h`
  - SHA256 `fcd1ea657d4d1fcacd36979e12557dd04b195234dc968097f6247baece0362eb`
- Performance:
  - output `tests/pytest/prof_out/packed64_phaseA40_p7_diag_l1_to_l0_paired_20260511/performance_compare.csv`
  - trustworthy neumann/normal ratios:
    - `S=64`: `1.271`
    - `S=512`: `1.932`
    - `S=2048`: `2.778`
  - `S=10240` is not trusted because both normal and neumann failed during warmup with package/symbol lookup errors.

Conclusion:

- A40 is correctness-safe and saved as baseline.
- The valid interpretation is not "four diagonal packed offsets are always one load". It is:
  - L0B transpose path can use one coalesced load.
  - L0A no-transpose path needs two paired loads in the current verified implementation.
  - Same NZ stride must still be validated separately for L0A/L0B direction.

## Strict packed64 A41 current failure

Current active/source/installed state:

- Active Stage1 source SHA256:
  - `901499af9d3f1c3a31b7865cc10f59795988e03835f836bfc8bedccf87dacc63`
- Installed Stage1 SHA256:
  - `901499af9d3f1c3a31b7865cc10f59795988e03835f836bfc8bedccf87dacc63`
- Installed kernel object SHA256:
  - `9c422787d69c1bbb18edc53e8c12450ad0efc002b2216ee83fb79b8647ece455`

Change versus A40:

- Added `LoadPacked16PairToL0Offset()`.
- Replaced the two per-lane `LoadPacked16BlockToL0Offset()` calls in these P8 first-layer helpers:
  - `LoadP8FirstLayerLowerToPacked32L0A()`
  - `LoadP8FirstLayerTopInvToPacked32L0B()`
  - `LoadP8FirstLayerBottomInvToPacked32L0A()`
  - `LoadPacked32DiagOnlyToL0B()`

Observed validation:

- `S=64/256/1024` smoke passed before this handoff.
- Full compare fails only at seed40 high-head case:
  - command: `python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 40 --batch-size 2 --seqlen 256 --nk 8 --nv 32 --dk 64 --dv 64`
  - `CPU vs ChunkGDRNeumann out` FAIL, max_abs `1.160285e-01`, worst actual near zero.
  - `CPU vs ChunkGDRNeumann state` FAIL, max_abs `1.723128e-01`, worst actual near zero.

Forward debug boundary:

- Do not save A41 baseline yet.
- Do not start A42/A43 copy-out work on this failing active path.
- Isolate the four paired helpers one at a time against the failing seed40 case.
- A40 already proved equal NZ stride does not make L0A no-transpose and L0B transpose equivalent; A41 must validate/correct those directions separately.

Additional redundant copy-out candidates to keep scheduled:

- Copy-out/writeback candidates are not optional just because A41 failed. They remain required follow-on phases once A41 is corrected:
  - P7 diagonal-only full `64x64` Fixpipe writes.
  - P8 first-layer packed32 tmp/result full Fixpipe writes and lane extraction.
  - P3/fallback diagonal GM writeback through `X @ I16`.
  - BF16 GM and resident row-prefix exposure through per-`16x16` `X @ I16`.
- Each candidate must be modified as its own phase with build/install, `S=64/256/1024` smoke, full compare, and then baseline save.

## Strict packed64 A41 PASS and A42 failure

Correction to the earlier handoff state:

- A41 was fixed after the seed40 failure and is now the latest correctness-safe baseline.
- A41 baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA41_p8_first_layer_l1_to_l0_paired_sync_baseline.h`
  - SHA256 `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`
- A41 root cause:
  - paired first-layer L1->L0 offsets were valid.
  - failure came from MTE1/M visibility/ordering between A/B paired loads.
  - final fix kept all four `LoadPacked16PairToL0Offset()` helpers and retained `SyncMte1ToM(); SyncMToMte1();` between L0A and L0B loads.
- A41 validation:
  - build PASS
  - install PASS
  - seed40 PASS
  - `S=64/256/1024` smoke PASS
  - full compare PASS: `16 passed in 6.17s`

Current A42 state:

- Active/source/installed are A42 failing candidate, not a baseline.
- Active Stage1 source SHA256:
  - `f1eba202336d9ee4582d5fb00df18bf372b4036584f8fa2b41d149b90b4e40da`
- Installed Stage1 SHA256:
  - `f1eba202336d9ee4582d5fb00df18bf372b4036584f8fa2b41d149b90b4e40da`
- Installed kernel object SHA256:
  - `f97bf90c0fd03f5e03439739a8baae3d651640162134ce9a6f39cd34d99284b5`
- A42 candidate changed only P7 diagonal-only copy-out:
  - added `FixpipePacked64DiagSubMatrix16ToHalfL1<base>()`
  - added `FixpipePacked64DiagOnlyToHalfL1()`
  - replaced four full `FixpipePacked64FullToHalfL1()` calls inside P7 with diagonal-only Fixpipe writes
- Validation:
  - build PASS
  - install PASS
  - `S=64` smoke timed out at 90s
  - changing the first three diagonal Fixpipe calls to `RAW_MMAD_NON_FINAL_UNIT_FLAG` and the last to final still timed out

API finding:

- DAV220 L0C->L1 `Fixpipe<..., CFG_NZ>` eventually calls `copy_matrix_cc_to_cbuf(dst, src, ..., nSize, mSize, dstStride, srcStride, unitFlag, ...)`.
- Unlike the L0C->GM row-major path, this L0C->L1 path does not set the `ndNum/srcNdStride/dstNdStride` ND_PARA scatter fields before issuing the copy.
- Therefore four diagonal packed offsets cannot be assumed to be one supported multi-ND L0C->L1 Fixpipe operation.

Conclusion:

- Do not proceed to A43/A44 while A42 is failing.
- Restore A41 only as a clean working surface, not as completion of A42.
- Next A42 work must either find a supported one-instruction or legal grouped L0C->L1 copy-out shape, or explicitly mark exact diagonal L0C->L1 scatter unsupported and choose another safe copy-out target with user-visible record.

## A41 clean restore and copy-out redundancy audit refresh

Current verified state after recheck:

- Active Stage1 source SHA256: `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`.
- A41 baseline SHA256: `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`.
- Installed Stage1 SHA256: `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`.
- Installed kernel object SHA256: `3a5e0da47ca0f34f90f3b932ea7e3e94b2447789a5017f5d1e2b65be31f41772`.
- Therefore the current work surface is A41 clean baseline, not the previous A42 failing candidate.

Docs corrected:

- `todo.md` now records A41 as current active/source/installed state and keeps previous A42 SHA only as the failed candidate record.
- `docs/2026-05-11/PROGRESS_AND_ISSUES.md` now separates current clean baseline from previous A42 failure.

Additional audit after the user's copy-out question:

- Already completed same-stride movement phases:
  - A39: P7 diagonal GM->packed64 L1 inputs are two tensor-level `Nd2NzParams DataCopy` calls for full64, one for `L_pow` and one for `INVInit`.
  - A40: P7 packed64 diagonal L1->L0 load is reduced, but not as one universal load: L0A uses two paired no-transpose loads; L0B uses one coalesced transpose load.
  - A41: P8 first-layer packed32 L1->L0 helpers use paired lane loads; MTE1/M ordering between L0A and L0B is required.
- Current A42 target remains P7 diagonal-only Fixpipe copy-out:
  - full `64x64` Fixpipe writes remain in `HalfL1PowerPacked64DiagOnlyOnC()` and three P7 result updates.
  - Old exact four-diagonal candidate timed out at `S=64`.
  - DAV220 L0C->L1 `Fixpipe<..., CFG_NZ>` and `DataCopyCO12DstParams` both lower to `copy_matrix_cc_to_cbuf(... dstStride, srcStride, unitFlag, ...)`; they do not provide the L0C->GM row-major style multi-ND scatter path.
  - The previous helper also needs L0C source layout scrutiny: L1 packed NZ offsets must not be reused as L0C source offsets without proof; full packed64 C-tile source stride is `64`.
  - A42 forward path should first try only one legal copy-out shape, such as a correct-source-offset rectangular/paired shape. If exact diagonal scatter is unsupported, mark that unsupported and continue within A42 with another supported grouping before A43/A44.
- Follow-on copy-out/writeback queue to keep scheduled:
  - A43: P8 first-layer packed32 output placement, because it still writes full packed32 tmp/result and then extracts two lanes.
  - A44+: BF16 GM writeback and resident exposure row-prefix handling. This must split visibility/side effects before replacing per-`16x16` `X @ I16` copies.
  - Tail/fallback later: `StoreP3DiagonalResultsToGm()` still uses compact scratch plus `CopyHalfL1MatrixToGmHalfByMmad()` for each diagonal block.
  - Generic `X @ I16` helpers remain movement substitutes, but must be replaced by consumer-specific phases rather than one global rewrite.

Rule reminder:

- Copy-out/writeback redundancies are real work items, but they must be modified one phase at a time.
- Precision failure on a phase requires forward diagnosis/fix on that target path; restoring A41 is only a clean workspace action, not completion of A42.

## A42 active-source recheck before validation

Current recheck:

- Active Stage1 source SHA256:
  - `756d8e90aceb04d615a1e0906fe39f3cab495834e32cdf5bc6ce3f70877c33f8`
- A41 baseline SHA256:
  - `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`
- Installed Stage1 SHA256:
  - `170e2e69c0c9993802fac4d792d627e2fd2000f254b5f488aac3987c8013594f`
- Installed kernel object SHA256:
  - `5cd2ebe2576fb4c632dff04f25863ac71584da96bb6714a6470c91568c5091e0`

Diff versus A41:

- Active source is an unverified A42 candidate, not clean A41.
- It adds `FixpipePacked64DiagQuadrant32ToHalfL1<base>()`.
- It replaces P7 full `64x64` Fixpipe writebacks with `FixpipePacked64DiagQuadrants32ToHalfL1()`.
- The current candidate writes `(32,32)` 32x32 quadrant with non-final first, then `(0,0)` 32x32 quadrant with final.

Additional A42 result records from the same copy-out investigation:

- Exact four-diagonal `16x16` L0C->L1 Fixpipe candidate `f1eba202...` timed out at `S=64`.
- two-`32x32` quadrant candidate `4f1a5fc...` timed out at `S=64`.
- two-`32x32` sequencing candidate `f0ccb89...` also timed out at `S=64`.
- Diagnostic candidate `23d7915...` used first `32x32` non-final plus second full64 final and passed `S=64`; this makes non-final/final sequencing less suspicious than the non-zero L0C source offset path.

Copy-out audit conclusion:

- P7 GM->L1 same-stride diagonal input is already A39: one `Nd2NzParams DataCopy` for `L_pow` and one for `INVInit`.
- P7 L1->L0 same-stride diagonal load is already A40/A41-era work: L0A needs paired no-transpose loads, L0B can use one coalesced transpose load.
- P7 L0C->L1 copy-out is a different API surface: `Fixpipe<..., CFG_NZ>` does not expose a multi-ND scatter equivalent to row-major GM output. A42 must use only a legal rectangle/group shape.
- P8 first-layer output copy-out, BF16 GM writeback, resident exposure, and P3/tail fallback copy-out remain scheduled, but must wait until A42 is PASS or explicitly abandoned with a recorded unsupported boundary.

Validation of active A42 candidate:

- build PASS.
- install PASS.
- active Stage1 source SHA256:
  - `756d8e90aceb04d615a1e0906fe39f3cab495834e32cdf5bc6ce3f70877c33f8`
- installed object SHA256 after install:
  - `5cd2ebe2576fb4c632dff04f25863ac71584da96bb6714a6470c91568c5091e0`
- `S=64` smoke:
  - command timed out at 90s.

Conclusion:

- Putting the final `unitFlag` on the L0C source-offset-zero quadrant does not fix A42.
- The remaining likely boundary is still `CFG_NZ` L0C->L1 Fixpipe with non-zero L0C source offset or unsupported partial-source rectangle semantics.
- Continue A42 forward when returning to L0C copy-out. Later 2026-05-11 clarification allows L0C copy-out to be deferred, but no next phase may be stacked on this failing A42 candidate.

## Copy-Out Redundancy Audit Refresh

User question handled:

- P7 GM->L1 diagonal input is already reduced in A39:
  - one `Nd2NzParams DataCopy` for `L_powHalfWsGm_ -> aicHalfL1TilePower_`
  - one `Nd2NzParams DataCopy` for `INVInitHalfWsGm_ -> aicHalfL1P3Result_`
  - these two cannot be merged into one copy because both the GM sources and L1 destinations differ.
- P7 L1->L0 diagonal load is already reduced in A40:
  - L0A no-transpose uses two paired loads.
  - L0B transpose uses one coalesced load.
  - A40 proved same packed diagonal stride does not make no-transpose and transpose loads equivalent.
- P7 L0C->L1 copy-out is a different API surface:
  - it is `Fixpipe<..., CFG_NZ>`, not `DataCopy`.
  - DAV220 L0C->L1 NZ Fixpipe does not expose a row-major GM-style multi-ND scatter path.
  - A42 must stay on legal rectangle/group shapes, not assume four diagonal packed offsets are one scatter copy.

Additional same-class redundancies recorded:

- Generic compact/packed rect L1->L0 helpers still issue row-block `LoadData()` loops:
  - `CopyHalfL1MatrixToL0A()` / `CopyHalfL1MatrixToL0B()`
  - `CopyPackedHalfL1BlockToL0()`
  - `CopyPackedHalfL1RectToL0()`
  - These are candidates only after choosing a concrete consumer; do not replace them globally because A40 showed L0A/L0B direction matters.
- P8 first-layer output still writes full packed32 tmp/result and then extracts two lanes into result64.
- P8 full64 lower-left preload still has two first-layer `16x16` blocks that can be paired; `(32,0) 32x32` is second-layer input and must stay separate.
- BF16 GM materialization and resident exposure still use per-`16x16` `X @ I16` copies; lower rows can later be treated as row prefixes, but side effects and visibility must be split first.
- P3/tail fallback `StoreP3DiagonalResultsToGm()` still uses compact scratch plus `X @ I16` GM copy-out one diagonal block at a time.

Current state recheck:

- Active Stage1 source SHA256: `b9c8df4462b0560f8faedf13a937591551381d64e699f2546affe91b0d7c8535`.
- A41 baseline SHA256: `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`.
- Installed package payload under `chunk_gated_delta_rule_neumann/` currently contains only kernel json/object, not a same-name `p8afullocal` header.
- Build object SHA256: `2ac1e134ad90aeaff67ce2db867422ac9808a8227e22519be2113a61eef44cf3`.
- Installed object SHA256: `2ac1e134ad90aeaff67ce2db867422ac9808a8227e22519be2113a61eef44cf3`.
- Current A42 diagnostic candidate only writes the `base0` `32x32` partial final in `FixpipePacked64DiagQuadrants32ToHalfL1()`.

Next validation:

- Run `S=64` smoke against this exact installed object.
- If it times out, partial final itself is unsafe in this path and A42 should move to a different supported copy-out shape from A41, not proceed to A43/A44.
- If it returns with precision failure, keep debugging A42 source layout / non-zero source offset / sequencing within the same target.

Validation update:

- Command:
  - `timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64`
- Result:
  - timed out at 90s.
- Conclusion:
  - `base0` source-offset-zero `32x32` partial final is also unsafe in this A42 path.
  - The issue is no longer only non-zero L0C source offset.
  - Continue A42 from A41 clean surface with a different supported copy-out shape; do not proceed to A43/A44.

## Plan update after identity/P8/L0C clarification

User clarification:

- `identity64` lifecycle still has redundancy. The intended target is first valid generation and later direct reuse from L1, but this must be implemented with a real raw-MMAD local buffer epoch / `pipe_->Reset()` lifecycle design, not by repeating the failed A30-style longer-lived flag.
- Packed MMAD is not redundancy. P8 should also keep moving in the packed-MMAD direction; optimize redundant movement/copy-out/layout-copy around it instead of decomposing packed compute back into per-`16x16` work.
- L0C copy-out may be solved later if it blocks progress.
- After all planned phases complete, save a baseline first, then use `msprof` real profiling data or simulator analysis to find further optimization points.

Current recheck:

- Active Stage1 SHA256:
  - `767e463b46af157e925839df5000bed658c3210c88bfe8f6521820d0b75f4277`
- A41 baseline SHA256:
  - `23c9fd8d466aa10b711a12028835388dd9b31b5662fa3ee5c26689b4b0edc0aa`
- The expected current build/install object paths were not present in the latest check, so the active source is not installed-aligned.
- Active source is still an A42 diagnostic candidate, not a baseline.

Plan changes recorded:

- `AGENTS.md` now states:
  - precision failure must be solved forward on the target path;
  - packed MMAD is the target, not a redundancy;
  - `identity64` reuse needs a proven lifecycle epoch;
  - a hardware/API-blocked optimization can be moved later only with the boundary recorded;
  - after planned phases pass, save a baseline before looking for new optimization points via `msprof` or simulator analysis.
- `todo.md` now treats A42 as `A42-deferred` if L0C copy-out continues to block.
- Next queue from a clean PASS surface:
  - A43: P7 L0A no-transpose diagonal load continuation.
  - A44: identity64 lifecycle epoch reuse.
  - A45: P8 first-layer packed32 output placement.
  - A46: P8 lower-left first-layer GM->lowerWork64 paired preload.
  - A47: P8 second-layer packed-MMAD load/tmp lifetime continuation.
  - A48+: BF16/resident row-prefix copy-out after side-effect split.
  - A-last: return to deferred L0C copy-out and other pending items.
  - A-final: save baseline, then find new optimization points with `msprof` or simulator analysis.

Execution boundary:

- Do not stack any of the next phases on the current A42 diagnostic source.
- Before implementing A43/A44/A45, restore or otherwise establish a clean PASS active/source/build/install baseline, then run the normal gate for the single changed point.
