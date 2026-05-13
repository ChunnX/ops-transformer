# 2026-05-13 WORK_LOG

## A94 RHS full-tile zero-init drop

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `4eede660d86bcdb61bdf24f4059a9ca484ce1a14532f534183ff1fa4a708f43f`.
- Single-point change:
  - in `FusedRhsAICProcess()`, skip the actual `ZeroHalfL1(aicHalfL1TilePower_, 64 * 64)` when `tileCols == chunkSize_`;
  - keep tail zero-init unchanged;
  - keep one `SyncMToMte1()` in the full-tile branch.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 GM or resident writeback change;
  - no final RHS math change;
  - no Stage2/Stage3/QK or `gBK/vBeta` change.
- Build/package note:
  - one stale shared `../../build_out` package only exported `aclnnTriInverse*` and was rejected before precision attribution;
  - the accepted package was captured immediately after a clean build and verified by `libcust_opapi.so` exported symbols.
- Accepted package:
  - `/tmp/cgdr_a94_build_capture/cgdr_a94.run`
  - SHA256 `e129c84e991180d42d9520281906a7f176c75693839d8b0b8a2f458650950bc7`.
- Isolated OPP:
  - `/tmp/cgdr_a94_repack_opp`
  - exports `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
- Gate PASS:
  - clean build/install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.14s`.
- Performance FAIL:
  - `tests/pytest/prof_out/packed64_phaseA94_rhs_fulltile_sync_20260513/performance_compare.csv`
  - ratios `S=64/512/2048/10240 = 1.228 / 1.603 / 2.317 / 2.344`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A94 regresses all profiled sizes.
- Resolution:
  - no A94 baseline saved.
  - active source restored to A82 and verified SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
  - do not retry this exact RHS full-tile zero-init skip without a materially different lifecycle proof.

## A82 BF16 lower block drop Fix-to-MTE2 sync

- Starting point: A81 correctness-safe baseline SHA256 `7a2ed0e4b1f0653004ef8ce5dec2dc851bb4c8479ef98c5101fc7e4922cf6575`.
- Source/baseline SHA256: `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Single-point change:
  - removed only `SyncFixToMte2()` from `CopyPackedHalfL1BlockToGmBf16()`.
- Scope guard:
  - lower BF16 block Fixpipe GM writeback remains;
  - `SyncFixToM()` and `SyncFixToS()` remain;
  - zero-rect BF16 helper is unchanged in this phase;
  - RHS still consumes `aicHalfL1ResidentInv_`;
  - no P7/P8 formula change;
  - no Stage2/Stage3/QK or `gBK/vBeta` change.
- Package:
  - `/tmp/packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_20260513_031638.run`
  - SHA256 `7543e7163cbaa270ded92d3947e9b370f64ffe7ecefa7135b25ce456af9d0224`.
- Isolated OPP:
  - `/tmp/packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_20260513_031638_opp`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
  - exports both normal and Neumann aclnn symbols.
- Gate PASS:
  - clean build PASS.
  - isolated install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.18s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_20260513/performance_compare.csv`
  - compare CSV SHA256 `08a272e9c3bf34b43e01f24346e08fd67b68ef706fe255d97a36d4a0ef261a4f`
  - raw CSV SHA256 `62dffa9240b7625b962f938b7a4a9dc79d8ce38e843bc4d6230c3c3da2953eaf`
  - ratios `S=64/512/2048/10240 = 1.191 / 1.544 / 2.313 / 2.332`.
  - A82 improves all four profiled sizes versus A81.

## A81 BF16 materialization drop explicit flush

- Starting point: A80 correctness-safe baseline SHA256 `26bee7c03e242f5fa9ebcc9be94a3a4028c623c15b764f281acc7c3388b52708`.
- Source/baseline SHA256: `7a2ed0e4b1f0653004ef8ce5dec2dc851bb4c8479ef98c5101fc7e4922cf6575`.
- Single-point change:
  - removed `FlushBf16MatrixCacheLines()` from `CopyZeroHalfL1RectToGmBf16ByMmad()`;
  - removed `FlushBf16MatrixCacheLines()` from `CopyPackedHalfL1BlockToGmBf16()`.
- Scope guard:
  - BF16 GM Fixpipe materialization remains;
  - `SyncFixToM()`, `SyncFixToMte2()`, and `SyncFixToS()` remain;
  - RHS still consumes `aicHalfL1ResidentInv_`;
  - no P7/P8 formula change;
  - no Stage2/Stage3/QK or `gBK/vBeta` change.
- Build/package notes:
  - first install attempt used a stale/single-op package and failed before precision because normal `aclnnChunkGatedDeltaRule*` symbols were absent;
  - rebuilt after clearing `../../build` and `../../build_out` with both ops selected.
- Package:
  - `/tmp/packed64_phaseA81_bf16_materialization_drop_flush_rebuild_20260513_030246.run`
  - SHA256 `6179a43418b470afce102e1be3cd3f93fc28bd46936ca81a1ffed2b8842c655b`.
- Isolated OPP:
  - `/tmp/packed64_phaseA81_bf16_materialization_drop_flush_rebuild_20260513_030246_opp`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`
  - exports both `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
- Gate PASS:
  - clean build PASS.
  - isolated install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.12s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA81_bf16_materialization_drop_flush_baseline.h`
  - SHA256 `7a2ed0e4b1f0653004ef8ce5dec2dc851bb4c8479ef98c5101fc7e4922cf6575`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA81_bf16_materialization_drop_flush_20260513/performance_compare.csv`
  - compare CSV SHA256 `7ff5b7c567bd799e4354b933c76802c8871162c36ede254b7b09e4c24c018fe4`
  - raw CSV SHA256 `7c69ecff8b510e808973c96481cfb59cdaa958adb7694df8eb33ce081bbbccf0`
  - ratios `S=64/512/2048/10240 = 1.239 / 1.620 / 2.329 / 2.376`.
  - A81 improves medium/large cases versus A80/A78/A67, but regresses `S=64`; keep A77 as small-shape reference.

## A74 Candidate: remove dead packed64 identity GM flag

- Starting point: A73 correctness-safe source SHA256 `07d0e7269273d3b997112cef17a77b48923bfe114d6aae3b31384c858c22a073`.
- Source after edit: `205eb470f3464db0e2f4b175da1ec3283416656c0df517c6391694899c02455e`.
- Single-point change:
  - removed unused `packed64IdentityGmPrepared_`;
  - removed its init/reset writes in `Init()` and `PrepareHalfL1IdentityByMmad()`;
  - removed the member field.
- Static diff versus A73 contains only those three deletions.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 GM writeback change;
  - no resident writeback/RHS math change;
  - no Stage2/Stage3/QK/`gBK/vBeta` change.

## P7/RHS/Resident Audit

- P7 still has three full64 `FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)` writes.
  - first two are L1 visibility points for later P7 MMAD reads from `cInv`;
  - final one exposes `result64` for P8/BF16/resident paths.
  - conclusion: not removable as a simple duplicate-write cleanup; future reduction must redesign legal read-after-write exposure.
- `FusedRhsAICProcess()` consumes `aicHalfL1ResidentInv_`, not `attnWsGm_`.
- Current flow still writes both:
  - BF16 GM inverse to `attnWsGm_`;
  - compact resident inverse to `aicHalfL1ResidentInv_`.
- BF16 GM writeback remains retained as a lifecycle/sync boundary because earlier delete/delay variants failed long sequence.
- Direct L1->L1 resident lower-block copy is not yet supported by local evidence:
  - local API notes cover GM/UB and L1/UB patterns, not a confirmed CBUF->CBUF `DataCopy`;
  - historical `CopyPackedL1SubBlock()` is scalar `GetValue/SetValue`, not acceptable for production optimization.
- The strict-branch `SyncMte2ToMte1()` after P7 must stay for now:
  - `EnsurePacked64Identity()` only synchronizes MTE2 on the first identity generation;
  - cache-hit para iterations still need a later MTE2->MTE1 boundary before P8 reads lowerWork64.

## A74 gate and performance

- Source/baseline SHA256: `205eb470f3464db0e2f4b175da1ec3283416656c0df517c6391694899c02455e`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA74_dead_identity_gm_flag_cleanup_baseline.h`.
- Package:
  - `/tmp/packed64_phaseA74_dead_identity_gm_flag_cleanup_20260513_001603.run`
  - SHA256 `7bfa8f2ddcb74391587034d622e0770bfa9cbe299bb5bb8aa15570d5cbab366a`.
- Isolated OPP:
  - `/tmp/packed64_phaseA74_dead_identity_gm_flag_cleanup_20260513_001603_opp`
  - installed Neumann object SHA256 `751c02da106aea980107ff327870bd39af19f7577c18e62e5234a2fc4c4817ee`.
- Gate PASS:
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.19s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA74_dead_identity_gm_flag_cleanup_20260513/performance_compare.csv`
  - ratios `S=64/512/2048/10240 = 1.221 / 1.682 / 2.421 / 2.464`.
  - A74 is correctness-safe; A67/A68 remained better performance references before A75.

## A75: drop strict P7 input upper zero

- Starting point: A74 correctness-safe baseline.
- Source after edit SHA256: `e0534ccf07b5e0b0abb90e67fb716c711cdf7d167cdbf56831ef071197eeb8e4`.
- Single-point change:
  - removed `ZeroPacked64UpperTriangularBlocks()`;
  - removed its strict `validLen >= 64` call in `LoadPacked64DiagonalInputsToL1()`.
- Rationale:
  - P7 raw MMAD reads only the four diagonal fractals from `aicHalfL1P3Result_`;
  - P7 later full64 Fixpipe overwrites the full `result64` tile before P8/BF16/resident consumers;
  - therefore this pre-P7 upper-triangular clear does not protect a semantic consumer.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 GM or resident writeback change;
  - no RHS math change.
- Package:
  - `/tmp/packed64_phaseA75_p7_input_upper_zero_drop_20260513_003102.run`
  - SHA256 `ca0cd31721c39a1079deb3cd6849933fd1c80bbbdbdf7c950b6dac342b10cebd`.
- Isolated OPP:
  - `/tmp/packed64_phaseA75_p7_input_upper_zero_drop_20260513_003102_opp`
  - installed Neumann object SHA256 `6159b9c09917071a0ff99c5023ab7c945e0d68649087acd1a22b94417a66e376`.
- Gate PASS:
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.23s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA75_p7_input_upper_zero_drop_baseline.h`
  - SHA256 `e0534ccf07b5e0b0abb90e67fb716c711cdf7d167cdbf56831ef071197eeb8e4`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA75_p7_input_upper_zero_drop_20260513/performance_compare.csv`
  - compare CSV SHA256 `bb21904137e1e183dc6bbc02133e7d3b28faf4d8cbcef5237272716ad4fb9429`
  - raw CSV SHA256 `4a88990b79b9427a8befc125e52b92cd1f29444ad22d68acff493ff152027708`
  - ratios `S=64/512/2048/10240 = 1.166 / 1.662 / 2.410 / 2.446`.
  - A75 improves all four ratios versus A74; it is better than A67/A68 at `S=64`, near A68 at `S=512`, but still slower than A68 at `S=2048/10240`.

## Next after A75

- Do not retry A40/A66 L0A no-transpose one-shot coalesced load:
  - old evidence: `repeatTimes=4, srcStride=5, dstGap=4` fails at `S=1024`.
  - current safe P7 L0A shape remains two paired loads.
- Continue from A75 with one bounded point at a time.
- Profiling after A75 still shows Neumann overhead mainly in:
  - AIC scalar ratio: `2.11 / 2.67 / 3.66 / 3.70`;
  - AIC MTE1 ratio: `2.22 / 2.31 / 1.83 / 1.83`;
  - AIC Fixpipe ratio: `1.32 / 1.49 / 1.50 / 1.49`.
- Higher-value next candidates are still BF16/resident materialization and `X @ I16` replay reduction, but only with a proven legal movement/dataflow path.

## A76 P8 First-Layer First MMAD L0 Zero-Copy Drop

- Starting point: A75 correctness-safe baseline SHA256 `e0534ccf07b5e0b0abb90e67fb716c711cdf7d167cdbf56831ef071197eeb8e4`.
- Source after edit SHA256: `7e54195c7419004d92214614d1c387c7203d3ad077bfaf3da42b1dd810b3f038`.
- Single-point change:
  - removed only the `EnsureHalfL1Zero(C64* C64)` plus L1-zero-to-L0A/L0B copy at the start of `RawMmadP8FirstLayerLowerTopFull64()`.
  - kept `SyncMToMte1()` and the two P8 LoadData phases.
- Rationale:
  - P7 starts with `ZeroPacked64DiagOnlyL0Operands()`, so main L0A/L0B off-diagonal lanes are already zero before P8.
  - the first P8 MMAD consumes only newly loaded `(16,0)` / `(48,32)` lower lanes and `(0,0)` / `(32,32)` top-inv lanes; stale P7 diagonal lanes do not feed this output lane set.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 GM writeback or resident L1 change;
  - no RHS, Stage2/Stage3/QK, or `gBK/vBeta` change;
  - did not apply this to the other three P8 full64 helpers.
- Build note:
  - first clean parallel build and then a low-concurrency rerun hit build-generator/object rename noise; a fresh clean serial build succeeded.
- Package:
  - `/tmp/packed64_phaseA76_p8_first_lower_top_l0_zero_drop_20260513_011200.run`
  - SHA256 `0b11155d89eddfba323694c2d1db8beaae231410d4c8955e17a1599dc996d43e`.
- Isolated OPP:
  - `/tmp/packed64_phaseA76_p8_first_lower_top_l0_zero_drop_20260513_011200_opp`
  - installed Neumann object SHA256 `402b10b547368123fa5c4ba252078563f4a7494f0f2cf407d98a35fe932ea6e1`.
- Gate PASS:
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.14s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA76_p8_first_lower_top_l0_zero_drop_baseline.h`
  - SHA256 `7e54195c7419004d92214614d1c387c7203d3ad077bfaf3da42b1dd810b3f038`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA76_p8_first_lower_top_l0_zero_drop_20260513/performance_compare.csv`
  - compare CSV SHA256 `830fb0d2da2fed8ad79d607419a68c3425f1c4ec94c050935c5a2769ef100e04`
  - raw CSV SHA256 `afd963bb0e8cc661e0cbcb5e0fdb2fafa226d4b1d0bef21064eb2d80fdbd4290`
  - ratios `S=64/512/2048/10240 = 1.200 / 1.654 / 2.416 / 2.456`.
  - A76 is correctness-safe but performance-regressed versus A75 at `S=64/2048/10240`; do not continue from A76 as the performance reference.

## A77 Resident Lower Row-Prefix48

- Starting point: A75 behavior restored after A76 performance regression.
- Source/baseline SHA256: `bdd6ecb7dcc55dd37f7e2a128ea3ae0a86750fd80f3bd4df1f58b3aac998d5d1`.
- Single-point change:
  - added `CopyPackedHalfL1RowPrefixToPackedHalfL1()`;
  - `StorePackedL1InverseToResidentL1()` uses row-prefix MMAD-copy for strict full64 resident rows `0/16/32` with cols `16/32/48`;
  - resident row `48` remains the old per-`16x16` copy path to avoid the unproven `cols=64` case.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 GM writeback change;
  - no RHS, Stage2/Stage3/QK, or `gBK/vBeta` change.
- Package:
  - `/tmp/packed64_phaseA77_resident_lower_rowprefix48_20260513_013614.run`
  - SHA256 `f0be37cf1b6a266dbfcc3da55c774275aef5f042fb0e5cf5c5968e197b37cac2`.
- Isolated OPP:
  - `/tmp/packed64_phaseA77_resident_lower_rowprefix48_20260513_013614_opp`
  - installed active header SHA256 `bdd6ecb7dcc55dd37f7e2a128ea3ae0a86750fd80f3bd4df1f58b3aac998d5d1`
  - installed Neumann object SHA256 `91941a1ecb1adfe4f155973a85af5ce2f1987751e2eec6431d0f68e366dcc42e`.
- Gate PASS:
  - fresh clean serial build PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.17s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA77_resident_lower_rowprefix48_baseline.h`
  - SHA256 `bdd6ecb7dcc55dd37f7e2a128ea3ae0a86750fd80f3bd4df1f58b3aac998d5d1`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA77_resident_lower_rowprefix48_20260513/performance_compare.csv`
  - compare CSV SHA256 `fd0657e47a78489c6cade5bd78d06c170852ea1699ea7eccbb098eb763aa1409`
  - raw CSV SHA256 `3dbab9bbfa845b8584fc2affde99f85644573ffc07bd770be98550a0f7379882`
  - ratios `S=64/512/2048/10240 = 1.147 / 1.663 / 2.409 / 2.433`.
  - A77 improves `S=64/2048/10240` versus A75 and improves `S=64/10240` versus A68; use A77 as the current continuation surface.

## A78 Candidate: resident row48 row-prefix64

- Starting point: A77 correctness-safe baseline SHA256 `bdd6ecb7dcc55dd37f7e2a128ea3ae0a86750fd80f3bd4df1f58b3aac998d5d1`.
- Single-point target:
  - extend strict full64 `StorePackedL1InverseToResidentL1()` row-prefix MMAD-copy from rows `0/16/32` to row `48`;
  - delete the remaining four row48 per-`16x16` `CopyHalfL1PackedBlockToCompact()` replays in the resident RHS input path.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 GM writeback change;
  - no RHS math change;
  - no Stage2/Stage3/QK or `gBK/vBeta` change.
- Risk:
  - BF16 row-prefix `cols=64` previously had a long-sequence failure boundary; resident L1 row-prefix64 must be verified independently and cannot be assumed safe.
- Source/baseline SHA256: `dfb99b8fd881dc607b9ac24a37158558c81eb447ddcb5c09a67a748dd693a731`.
- Package:
  - `/tmp/packed64_phaseA78_resident_rowprefix64_20260513_015808.run`
  - SHA256 `508148ca09bf28b99caa922eee3c80f122ca5a999eecd7d410d9dd98f08c4afe`.
- Isolated OPP:
  - `/tmp/packed64_phaseA78_resident_rowprefix64_20260513_015808_opp`
  - installed active header SHA256 `dfb99b8fd881dc607b9ac24a37158558c81eb447ddcb5c09a67a748dd693a731`
  - installed Neumann object SHA256 `b6a6f756c0c03d7b187a3ed294aeb70d4facc0eca609e2a26ab6baa3c0c93b05`.
- Gate PASS:
  - clean serial build PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.30s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA78_resident_rowprefix64_baseline.h`
  - SHA256 `dfb99b8fd881dc607b9ac24a37158558c81eb447ddcb5c09a67a748dd693a731`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA78_resident_rowprefix64_20260513/performance_compare.csv`
  - compare CSV SHA256 `7c6b67152e358d1c6406f4b5dc5985f0c6738f6bc1d80f4f0e2056676d74ca30`
  - raw CSV SHA256 `a2a19f8e5264b2543a666eba1e21cf599cd3f334dec5f23ee79a618966a84daa`
  - ratios `S=64/512/2048/10240 = 1.194 / 1.626 / 2.403 / 2.424`.
  - Compared with A77 `1.147 / 1.663 / 2.409 / 2.433`, A78 improves `S=512/2048/10240` and regresses `S=64`.
  - Interpretation: resident row48 row-prefix64 removes replay for medium/large workloads but adds too much fixed cost for the smallest case.

## A83 Candidate: BF16 Zero-Rect Drop Fix-to-MTE2 Sync

- Starting point: A82 correctness-safe baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256 before build: `f655ef6305a8f000e085e11bd9d416fc7f2adea80e2c3ce629f3a4808ec6a6ba`.
- Single-point change:
  - removed only `SyncFixToMte2()` at the end of `CopyZeroHalfL1RectToGmBf16ByMmad()`;
  - kept BF16 GM zero-rect Fixpipe writeback, `SyncFixToM()`, and `SyncFixToS()`;
  - did not change `CopyPackedHalfL1BlockToGmBf16()`, resident/RHS, P7/P8 formulas, Stage2/Stage3/QK, or `gBK/vBeta`.
- Package:
  - `/tmp/packed64_phaseA83_bf16_zero_rect_drop_fix_mte2_sync_20260513_032803.run`
  - SHA256 `d04c61ef0694a5aebd7824bdc49478e71f9c6f9fb48e314f95e0be5785194890`.
- Isolated OPP:
  - `/tmp/packed64_phaseA83_bf16_zero_rect_drop_fix_mte2_sync_20260513_032803_opp`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
- Gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.17s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA83_bf16_zero_rect_drop_fix_mte2_sync_20260513/performance_compare.csv`
  - compare CSV SHA256 `55b5897b4d2128c460ee2d0e18c334e9e830a0792c403b0fc20b4faa2bf96948`
  - raw CSV SHA256 `d26a2e06c416eec2fdd87441c62d23455219dbf1050ae5616bf1c8e3c416e943`
  - ratios `S=64/512/2048/10240 = 1.141 / 1.589 / 2.340 / 2.342`.
- Conclusion:
  - A83 is correctness-safe but not accepted as the continuation baseline because it improves only `S=64` and regresses `S=512/2048/10240` versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Active source restored to A82 and verified byte-identical to the A82 baseline, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A84 Failed: BF16 Lower Block Drop Fix-to-Scalar Sync

- Starting point: A82 correctness-safe baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `59c935cba6a1f44cf0185c7745a24a2e65867e0467b4d773aaf9921fcd17a145`.
- Single-point change:
  - removed only `SyncFixToS()` at the end of `CopyPackedHalfL1BlockToGmBf16()`;
  - kept lower BF16 block Fixpipe GM writeback and `SyncFixToM()`;
  - did not change zero-rect BF16 helper, resident/RHS, P7/P8 formulas, Stage2/Stage3/QK, or `gBK/vBeta`.
- Package:
  - `/tmp/packed64_phaseA84_bf16_lower_block_drop_fix_to_scalar_sync_20260513_034419.run`
  - SHA256 `5dde9b2aabda5aeaab28bd893c716df7e9b92db8c656d714ef186dc05fb2e348`.
- Isolated OPP:
  - `/tmp/packed64_phaseA84_bf16_lower_block_drop_fix_to_scalar_sync_20260513_034419_opp`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
- Gate result:
  - build/install PASS;
  - `S=64` PASS;
  - `S=256` PASS;
  - `S=1024` FAIL.
- Failure:
  - `CPU vs ChunkGDRNeumann out max_abs=1.576879e-01`, worst actual `-1.10351562e-01` vs expect `4.73363884e-02`;
  - `CPU vs ChunkGDRNeumann state max_abs=1.081070e-01`, worst actual `-5.17578125e-02` vs expect `5.63491546e-02`.
- Conclusion:
  - lower BF16 GM materialization still needs this `SyncFixToS()` for long-sequence visibility/order; do not retry the same deletion.
  - Active source restored to A82 and verified byte-identical to the A82 baseline, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A85 Candidate: BF16 Zero-Rect Drop Fix-to-Scalar Sync

- Starting point: A82 correctness-safe baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `6655407d2f58f8e8c3ec42b685725c87f801211660d68fe94bc9e0933bf54837`.
- Single-point change:
  - removed only `SyncFixToS()` at the end of `CopyZeroHalfL1RectToGmBf16ByMmad()`;
  - kept zero-rect BF16 GM Fixpipe writeback, `SyncFixToM()`, and `SyncFixToMte2()`;
  - did not change lower BF16 block helper, resident/RHS, P7/P8 formulas, Stage2/Stage3/QK, or `gBK/vBeta`.
- Build/package note:
  - first install used a stale tri_inverse-only package and failed before precision because normal CGDR symbols were absent;
  - rebuilt after clearing `../../build` and `../../build_out` with both CGDR ops selected.
- Package:
  - `/tmp/packed64_phaseA85_bf16_zero_rect_drop_fix_to_scalar_sync_rebuild_20260513_035737.run`
  - SHA256 `7f85be22009244d29b51ae98117592c919096fde76ecb0237988f87ef5314577`.
- Isolated OPP:
  - `/tmp/packed64_phaseA85_bf16_zero_rect_drop_fix_to_scalar_sync_rebuild_opp_20260513_035737`
  - exported both `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
- Gate:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.20s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA85_bf16_zero_rect_drop_fix_to_scalar_sync_20260513/performance_compare.csv`
  - compare CSV SHA256 `eeccfeace4ac5c4a381541634492f527bd2dae018db522803a0778d8e0179624`
  - raw CSV SHA256 `c945009aee9e3e3bafe5172bb6b4a7f0d2cdbb5e45bf32f7d1da70fd6814cadb`
  - ratios `S=64/512/2048/10240 = 1.161 / 1.603 / 2.329 / 2.368`.
- Conclusion:
  - A85 is correctness-safe but not accepted as the continuation baseline because it improves only `S=64` and regresses `S=512/2048/10240` versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Active source restored to A82 and verified byte-identical to the A82 baseline, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## Current P7 And RHS Writeback Audit

- P7 still has three full64 `FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)` writes.
  - the first two expose updated `cInv` through L1 for later P7 MMAD reads;
  - the last exposes final `result64` for P8/BF16/resident consumers.
  - these are not simple duplicate writes; reducing them requires a legal alternate read-after-write design.
- `FusedRhsAICProcess()` consumes `aicHalfL1ResidentInv_`, not `attnWsGm_`.
- The current flow still also materializes BF16 inverse to GM `attnWsGm_`:
  - lower true-inverse blocks go through `CopyPackedHalfL1BlockToGmBf16()`;
  - upper zero rectangles go through `CopyZeroHalfL1RectToGmBf16ByMmad()`.
- BF16 GM writeback is still retained as a lifecycle/sync boundary; do not delete it as a whole without a new validated replacement.

## A86 Candidate: BF16 Zero-Rect Drop Fix-to-M Sync

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `fc162cdcaac6114a9c364b7b0527b636f43a7374eb38c06c5488fdbc4b563a20`.
- Single-point change:
  - removed only `SyncFixToM()` at the end of `CopyZeroHalfL1RectToGmBf16ByMmad()`;
  - kept zero-rect BF16 GM Fixpipe writeback, `SyncFixToMte2()`, and `SyncFixToS()`;
  - did not change lower BF16 block helper, resident/RHS, P7/P8 formulas, Stage2/Stage3/QK, or `gBK/vBeta`.
- Package:
  - `/tmp/packed64_phaseA86_bf16_zero_rect_drop_fix_to_m_sync_20260513_040947.run`
  - SHA256 `7d46233cc9bd65b61f1e72179168e3b10f11bb24db76ce44ab74043166fb2ceb`.
- Isolated OPP:
  - `/tmp/packed64_phaseA86_bf16_zero_rect_drop_fix_to_m_sync_opp_20260513_040947/vendors/custom_transformer`
  - exported both `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
- Gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.16s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA86_bf16_zero_rect_drop_fix_to_m_sync_20260513/performance_compare.csv`
  - compare CSV SHA256 `d8e13c1db71abb740555dc933dd663c71924a547b2ca69a6aed301d7ff068eb1`
  - raw CSV SHA256 `7020804e7e820f59403765b78097c4b8cc4c3c58d737c4bf321a090ac19a0aef`
  - ratios `S=64/512/2048/10240 = 1.194 / 1.566 / 2.334 / 2.344`.
- Conclusion:
  - A86 is correctness-safe but not accepted as the continuation baseline because it regresses all four profiled sizes versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Active source restored to A82 and verified by SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A87 Candidate: Rect Raw-MMAD Local HF32 Cleanup

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `9c64d50a80ff85bd27aed169952d03ab14680d565d731e83b11489583552d04b`.
- Single-point change:
  - removed only the local `SetHF32Mode(HF32Mode::DISABLE)` inside `RawMmadPackedHalfL1AccumulateRect()`;
  - kept `PrepareRawMmadState()` canonical HF32 setup;
  - did not change `RawMmadPackedHalfL1AccumulateTile()` because A59 already proved that helper's local HF32 setup is needed for long sequence correctness;
  - did not change P7/P8 formulas, BF16/resident writeback shape, RHS math, Stage2/Stage3/QK, or `gBK/vBeta`.
- Package:
  - `/tmp/packed64_phaseA87_rect_mmad_no_hf32_20260513_042328.run`
  - SHA256 `e0bd99157b0effd9ac3bffed35cf92f9fb78040523bd58ac168fa732b2a0d437`.
- Isolated OPP:
  - `/tmp/packed64_phaseA87_rect_mmad_no_hf32_20260513_042328_opp/vendors/custom_transformer`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
- Gate:
  - clean build PASS;
  - isolated install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.27s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA87_rect_mmad_no_hf32_20260513/performance_compare.csv`
  - compare CSV SHA256 `8303844589804f39385234aa211ed7e42164011f340719f72e6e2dabdf015831`
  - raw CSV SHA256 `291654487de6c62377429ceaecd1a3f54f8d2005c361600b930fe247296a1b51`
  - ratios `S=64/512/2048/10240 = 1.195 / 1.585 / 2.322 / 2.366`.
- Conclusion:
  - A87 is correctness-safe but not accepted as the continuation baseline because it regresses all four profiled sizes versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Active source restored to A82 and verified by SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A88 Candidate: P7 Diag Drop M-to-MTE1 Sync

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `f6241a5a0941b93d73b23e4ce88e1680a272d92dc2f88938861a4ac7b3f632b0`.
- Single-point change:
  - removed only one `SyncMToMte1()` in `RawMmadPacked64DiagOnly()`, between the L0A diagonal load and L0B diagonal load;
  - did not change P7/P8 formulas, BF16/resident writeback shape, RHS math, Stage2/Stage3/QK, or `gBK/vBeta`.
- Gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 7.01s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA88_p7_diag_drop_m_mte1_sync_20260513/performance_compare.csv`
  - ratios `S=64/512/2048/10240 = 1.166 / 1.597 / 2.322 / 2.347`.
- Conclusion:
  - A88 is correctness-safe but not accepted as the continuation baseline because it improves only `S=64` and regresses `S=512/2048/10240` versus A82.
  - Active source restored to A82 and verified by SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A89 Candidate: Defer BF16 GM Materialization After K-RHS

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `524630087629315ac442b1b0486c55380c2b0fc52df2ae90e1ebbdfc5e0740f0`.
- Single-point change:
  - removed `StorePackedL1InverseToBf16Attn(paraIdx)` from the end of `C64InverseBy16x16Neumann()`;
  - inserted BF16 GM materialization after the K-RHS `FusedRhsAICProcess()` loop and before waiting for VBeta;
  - changed `StorePackedL1InverseToBf16Attn()` source from shared `aicHalfL1P3Result_` to per-para `aicHalfL1ResidentInv_`;
  - did not change P7/P8 formulas, resident RHS math, Stage2/Stage3/QK, or `gBK/vBeta`.
- Build/package:
  - first build_out package was rejected before gate because `libcust_opapi.so` did not export CGDR/Neumann symbols;
  - rebuilt after clearing `../../build` and `../../build_out`.
  - trusted package `/tmp/packed64_phaseA89_defer_bf16_after_k_rhs_rebuild_20260513_045536.run`, SHA256 `13503115dd958cebf89f3d58cd855e0f265fdd90ea2e0a5eaa058d645648d272`.
  - isolated OPP `/tmp/packed64_phaseA89_defer_bf16_after_k_rhs_rebuild_20260513_045536_opp/vendors/custom_transformer`.
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`, exports both normal and Neumann aclnn symbols.
  - installed Neumann object SHA256 `874cd6a65ea37f4644601c3c8637abe77965a5c724e3cb227431d42bbfa620c3`.
- Gate:
  - clean build PASS;
  - isolated install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.23s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA89_defer_bf16_after_k_rhs_20260513/performance_compare.csv`, SHA256 `33b794614ec26bb304f093e632d97644a7bbbe287c731ced4b7ec0ff73f7d467`.
  - raw CSV SHA256 `cf89be1075778f9ffe61cad88c9066ecd2c9231bd93185a3a1f2f21aa9c8fc7e`.
  - ratios `S=64/512/2048/10240 = 1.174 / 1.615 / 2.454 / 2.475`.
- Conclusion:
  - A89 is correctness-safe but not accepted as the continuation baseline because it improves only `S=64` and regresses medium/large shapes versus A82 by roughly `+4.6% / +6.1% / +6.1%`.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A90 Candidate: Prepare Raw-MMAD State Once For Inverse Window

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `c2b75f0e6f7086eb165ee7b59976800a772b5cd663bb6d83b7003a0b2a3ae23d`.
- Single-point change:
  - moved inverse-window `PrepareRawMmadState()` to `ParaChunkAIC()` immediately before the `C64InverseBy16x16Neumann()` loop;
  - kept `EnsureAicLocalBuffers()` before the hoisted prepare because first raw entry may call `pipe_->Reset()`;
  - removed only the repeated `PrepareRawMmadState()` inside `C64InverseBy16x16Neumann()`;
  - did not change RHS `FusedRhsAICProcess()`, P7/P8 formulas, BF16/resident writeback shape, Stage2/Stage3/QK, or `gBK/vBeta`.
- Package:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`
  - SHA256 `6180c7d9c363571eca5eb2c99b1a98b4a60e73bf0f6f035b8f88da8436017863`.
- Isolated OPP:
  - `/tmp/packed64_phaseA90_prepare_raw_state_once_20260513_051359_opp/vendors/custom_transformer`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
  - exports both normal and Neumann aclnn symbols.
- Gate:
  - clean build PASS;
  - isolated install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.85s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA90_prepare_raw_state_once_20260513/performance_compare.csv`
  - ratios `S=64/512/2048/10240 = 1.116 / 1.560 / 2.332 / 2.351`.
- Conclusion:
  - A90 is correctness-safe but not accepted as the continuation baseline because it improves only `S=64` and regresses `S=512/2048/10240` versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A91 Candidate: Resident L1 Before BF16 GM Writeback

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `fedb261b7ea0269c495aa38f3c9a79ce73f84a15c8575e48fde207af11b51001`.
- Single-point change:
  - swapped only the two final inverse materialization calls in `C64InverseBy16x16Neumann()`;
  - `StorePackedL1InverseToResidentL1(paraIdx)` runs before `StorePackedL1InverseToBf16Attn(paraIdx)`;
  - BF16 GM writeback remains, RHS still consumes `aicHalfL1ResidentInv_`, and P7/P8 formulas are unchanged.
- Package:
  - `/tmp/packed64_phaseA91_resident_before_bf16_20260513_053115.run`
  - SHA256 `ba571da00e49ccacd88955cdeba6e73d849178d7b29f3986e29ce6d8324d60df`.
- Isolated OPP:
  - `/tmp/packed64_phaseA91_resident_before_bf16_20260513_053115_opp/vendors/custom_transformer`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
  - exports both normal and Neumann aclnn symbols.
- Gate:
  - clean build PASS;
  - isolated install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.20s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA91_resident_before_bf16_20260513/performance_compare.csv`, SHA256 `c1305a5f080d6ec14f14f2ae1f0fc122b56cb7114624d4d2ec35ac55602f2a0c`.
  - raw CSV SHA256 `c22123487582fb8c82a641bf2061f49ed1da5320d48260b8a17732cb23e9fd51`.
  - ratios `S=64/512/2048/10240 = 1.255 / 1.595 / 2.326 / 2.370`.
- Conclusion:
  - A91 is correctness-safe but not accepted as the continuation baseline because it regresses all four profiled sizes versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Resident-first writeback does not hide BF16 GM materialization cost and should not be retried without a materially different dataflow.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A92 Candidate: Resident Row-Prefix Identity Ensure Hoist

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `d04902f14c88ae616385289601cb0a6f976bd9a0495bd38fb094ae980aa96675`.
- Single-point change:
  - removed `EnsurePacked64Identity();` from `CopyPackedHalfL1RowPrefixToPackedHalfL1()`;
  - added one `EnsurePacked64Identity();` before the strict resident row-prefix loop in `StorePackedL1InverseToResidentL1()`;
  - did not change data movement, MMAD shape, Fixpipe, sync, BF16 GM writeback, resident RHS, or P7/P8 formulas.
- Build note:
  - first clean `-j16` build failed in third-party ES/abseil generation, not kernel compile;
  - clean `-j1` rebuild PASS.
- Package:
  - `/tmp/packed64_phaseA92_resident_identity_ensure_hoist_20260513_054603.run`
  - SHA256 `7cf028975075e5401312de43e20e34521a2193df5e7dfdd195a708fda042874b`.
- Isolated OPP:
  - `/tmp/packed64_phaseA92_resident_identity_ensure_hoist_20260513_054603_opp/vendors/custom_transformer`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
- Gate:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.49s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA92_resident_identity_ensure_hoist_20260513/performance_compare.csv`, SHA256 `5c6c66e447c8a596950e3b7bc3ce08475ebde5600310544020fcf823937b95a9`.
  - raw CSV SHA256 `97464d25b9d6a7b78cb40cd07ccdd64d9bb3f2d539cea10fe518f136588be029`.
  - ratios `S=64/512/2048/10240 = 1.248 / 1.592 / 2.334 / 2.356`.
- Conclusion:
  - A92 is correctness-safe but not accepted because it regresses all four profiled sizes versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Hoisting the resident row-prefix identity check is not useful by itself; do not retry this scalar-only move without a materially different resident copy path.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A93 Candidate: Delay Second-Layer Lower Preload Wait Until After First-Layer P8

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `eaf78381d661158be0a1db40e525e17dd46c004d3a7be52b68a9d61d4c16d15a`.
- Single-point change:
  - moved the strict-path `SyncMte2ToMte1()` after `P7Packed64DiagonalNeumannOneShotToResult()` from before `P8ComposeFirstLayerAccumFull64()` to after it;
  - preserved the internal A46b sync between paired first-layer lower preload and separate `(32,0)` preload;
  - did not change any source/destination, MMAD formula, BF16/resident writeback, Stage2/Stage3/QK, or `gBK/vBeta`.
- Build note:
  - first clean `-j1` build failed in FATBIN because opc could not find generated `mix_aic_0.o/mix_aiv_0.o`; retry after cleaning build/tmp succeeded with the same candidate source.
- Package:
  - `/tmp/packed64_phaseA93_p8_second_lower_wait_after_first_layer_20260513_055738.run`
  - SHA256 `130a6682c3a02cef33fe3f4ec213c431ed3db739c11f225d51defad40f0f7afb`.
- Isolated OPP:
  - `/tmp/packed64_phaseA93_p8_second_lower_wait_after_first_layer_20260513_055738_opp/vendors/custom_transformer`
  - `libcust_opapi.so` SHA256 `9721ca309f1a7e65fe3e835088e91037f3f7a36f6431827fa90c8a42e18c0752`.
  - installed Neumann object SHA256 `d16171a90dc5ff8323cfc08113e60bcc8a076e5e47073f3535dbf076b820cb3d`.
- Gate:
  - clean build/install PASS after retry;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.22s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA93_p8_second_lower_wait_after_first_layer_20260513/performance_compare.csv`, SHA256 `66e80b8ccb9b0379c29cf841079432a5713ed4dac0c847f9043214b5a051746f`.
  - raw CSV SHA256 `e8f68b5c3341d7e6afcc68e86aa182882709cb2cbd2870b7e26a00b4bad783df`.
  - ratios `S=64/512/2048/10240 = 1.217 / 1.587 / 2.301 / 2.344`.
- Conclusion:
  - A93 is correctness-safe but not accepted because it improves only `S=2048` and regresses `S=64/512/10240` versus A82.
  - Delaying only the second-layer lower wait does not provide stable performance benefit; do not retry this exact wait placement.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A95 Candidate: Drop AIC Cache Clean Before P7 Diagonal Preload

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `cc9140cecb9621232c880eb8a786c8e85dc4e79734012a95ff5ce3f35be8412d`.
- Single-point change:
  - removed only the AIC-side `CleanHalfMatrixCacheLines()` before strict P7 diagonal GM-to-L1 preload;
  - did not change P7/P8 formulas, BF16 GM writeback, resident RHS, Stage2/Stage3/QK, or `gBK/vBeta`.
- Package:
  - `/tmp/packed64_phaseA95_p7_diag_preload_drop_aic_clean_20260513_0642.run`
  - SHA256 `d4c49fa33938b5e1057b49f0e70008783c6666ed1ea30a3579344aad5f3464c2`.
  - installed Neumann object SHA256 `df55c400c148adc08ccb338f9aadd6e91e631ba20bead50a1e5935038e656e0e`.
- Gate:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.53s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA95_p7_diag_preload_drop_aic_clean_20260513/performance_compare.csv`, SHA256 `0a42fa9d61f6f07c0a27b0386563217bacc3a4d6a0a4a1063f4692e5f465b49d`.
  - raw CSV SHA256 `64062327ac50eff060c9506e5aace215dd77726e5f51ddbbbb44d191b63f9100`.
  - ratios `S=64/512/2048/10240 = 1.244 / 1.610 / 2.298 / 2.335`.
- Conclusion:
  - A95 is correctness-safe but not accepted because it improves only `S=2048` and regresses `S=64/512/10240` versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Do not continue from A95 and do not retry this exact AIC clean deletion without a materially different visibility design.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A96 Candidate: Drop First P7 Intermediate CInv Fix-to-MTE1 Sync

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `426b67b8aad6f326bcb9b19a0ba86cde2960ede208c853fc13913ac3055f4b04`.
- Single-point change:
  - remove only the first `SyncFixToMte1()` immediately after the first intermediate `FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)` inside `P7Packed64DiagonalNeumannOneShotToResult()`;
  - keep all P7 MMAD/Fixpipe math, the second and final P7 `SyncFixToMte1()`, P8, BF16 GM writeback, resident RHS, Stage2/Stage3/QK, and `gBK/vBeta` unchanged.
- Rationale:
  - the next operation after that write is `HalfL1PowerPacked64DiagOnlyOnC(diagScratch64, diagScratch64, ...)`, which does not read `aicHalfL1P3Result_`;
  - before `aicHalfL1P3Result_` is read again, `HalfL1PowerPacked64DiagOnlyOnC()` performs its own Fix-to-MTE1 sync for the newly written power tile, which should also order the earlier `cInv` Fixpipe write.
- Package:
  - `/tmp/packed64_phaseA96_p7_first_cinv_drop_fix_mte1_sync_20260513_0657.run`
  - SHA256 `168721cd43945e94108c3f3de25498c621c98592932517ddc94db586731effa2`.
  - installed Neumann object SHA256 `3fd3835934722a4abb0b3663010ad6e31fb945787150f44b96d8d7ede7b3cdd7`.
- Gate:
  - clean build/install PASS;
  - opapi exports only CGDR/Neumann symbols, no `TriInverse`;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.16s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA96_p7_first_cinv_drop_fix_mte1_sync_20260513/performance_compare.csv`, SHA256 `3e818d84e6678f7cd48d697e14df48fee4e372fb762b05570e1a10dba00a3668`.
  - raw CSV SHA256 `2855a9edf3105954e5fdf52f1b9a0eba2454c50dba7a3d208ae8f3080ad7e05c`.
  - ratios `S=64/512/2048/10240 = 1.140 / 1.582 / 2.322 / 2.363`.
- Conclusion:
  - A96 is correctness-safe but not accepted because it improves only `S=64` and regresses `S=512/2048/10240` versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - The first intermediate P7 Fix-to-MTE1 sync is not a stable medium/large performance deletion point. Do not continue from A96.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A97 Candidate: Drop Second P7 Intermediate CInv Fix-to-MTE1 Sync

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `823fb2ef4bff2816dc67e98d84e1081b49a601ff751742fbd7bf31b27eaa3e3b`.
- Single-point change:
  - removed only the second `SyncFixToMte1()` immediately after the second intermediate `FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)` inside `P7Packed64DiagonalNeumannOneShotToResult()`;
  - kept all P7 MMAD/Fixpipe math, the first and final P7 Fix-to-MTE1 syncs, P8, BF16 GM writeback, resident RHS, Stage2/Stage3/QK, and `gBK/vBeta` unchanged.
- Package:
  - `/tmp/packed64_phaseA97_p7_second_cinv_drop_fix_mte1_sync_20260513_072024.run`.
  - installed Neumann object SHA256 `179afb94331dd7467e97f37195e2a8fd6d95fb2007c79f2e3edf1f09e3848a68`.
- Gate:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.14s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA97_p7_second_cinv_drop_fix_mte1_sync_20260513/performance_compare.csv`.
  - ratios `S=64/512/2048/10240 = 1.204 / 1.564 / 2.327 / 2.336`.
- Conclusion:
  - A97 is correctness-safe but regresses all four profiled sizes versus A82 `1.191 / 1.544 / 2.313 / 2.332`.
  - Do not retry this exact second intermediate sync deletion.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A98 Candidate: Drop Final P7 CInv Fix-to-MTE1 Sync

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `b5ec48ce2b5bf5e3a2a23e3a0d5627d233103baf8b85912f65fa0fa6895dd164`.
- Single-point change:
  - removed only the final `SyncFixToMte1()` immediately after the last `FixpipePacked64FullToHalfL1(aicHalfL1P3Result_, cInv, ...)` in `P7Packed64DiagonalNeumannOneShotToResult()`;
  - kept the full64 Fixpipe writeback itself, P7/P8 formulas, BF16 GM writeback, resident RHS, Stage2/Stage3/QK, and `gBK/vBeta` unchanged.
- Package:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`, SHA256 `da8a331b44a40e6722df0518a6ffc0e399235a441738f22388ad7a48e76352ad`.
- Isolated OPP:
  - `/tmp/packed64_phaseA98_p7_final_cinv_drop_fix_mte1_sync_20260513_0735_opp/vendors/custom_transformer`.
  - `libcust_opapi.so` exports CGDR/Neumann symbols only.
  - installed Neumann object SHA256 `7ab48f6743467d9f41555d7005ebf81b86accad15e27dba85ade521a5819d91c`.
- Gate:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.15s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA98_p7_final_cinv_drop_fix_mte1_sync_20260513/performance_compare.csv`, SHA256 `ea7b74157db18c6b7eff9e1008c19d744452e41c0c1dcb7caceba609c3d3295f`.
  - raw CSV SHA256 `76ce078d513d81fce737b6f783ce8dd5a2060693532e3f9206b5b8d91002f341`.
  - ratios `S=64/512/2048/10240 = 1.166 / 1.572 / 2.326 / 2.355`.
- Conclusion:
  - A98 is correctness-safe but not accepted because it improves only `S=64` and regresses `S=512/2048/10240` versus A82.
  - Together A96/A97/A98 show that dropping any of the three P7 `cInv` Fix-to-MTE1 syncs is not the medium/large performance direction.
  - Active source restored to A82 and verified byte-identical, SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A99 Candidate: BF16 Lower Row-Prefix on A82 Sync Boundary

- Starting point: A82 correctness/performance baseline SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Candidate source SHA256: `4669f1ee6ccd59704173fe0389702cdc067e8762188c232158d74cd7f291a393`.
- Single-point change:
  - add `CopyPackedHalfL1RowPrefixToGmBf16()`;
  - in `StorePackedL1InverseToBf16Attn()`, rows `0/16/32` use lower row-prefix rectangle writeback with cols `16/32/48`;
  - row `48, cols=64` keeps the previous per-`16x16` block writeback;
  - preserve A82 sync boundary exactly: keep `SyncFixToM()` and `SyncFixToS()`, do not restore `SyncFixToMte2()` or flush.
- Package/install:
  - package `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`, SHA256 `b1269e1030235783fdafa96a311f3a2d17cbc9ca903f6e7f6324175a5d3baa09`;
  - isolated OPP `/tmp/packed64_phaseA99_bf16_rowprefix_a82_sync_20260513_0752_opp/vendors/custom_transformer`;
  - installed Neumann object SHA256 `819eebfffbcc475c36f09d6fd6be426ac96b75013028e9e8645aba5afde64e24`;
  - installed opapi exports CGDR/Neumann symbols only, no `TriInverse`.
- Gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.77s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA99_bf16_rowprefix_a82_sync_20260513/performance_compare.csv`, SHA256 `8a8673781684d79818790a3dd749f05fed7eb07d0af1197be38fd2de61097567`;
  - raw CSV SHA256 `c162ad26ab051f9c8a01aef817944d88b2fe1e6092de8c75b5e1fb6579015a44`;
  - ratios `S=64/512/2048/10240 = 1.162 / 1.544 / 2.288 / 2.330`.
- Interpretation:
  - compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A99 improves `S=64/2048/10240`, with `S=512` only noise-level regression;
  - Neumann absolute task time decreases on all four profiled sizes:
    - `788.72 -> 781.92 us`
    - `2598.62 -> 2567.70 us`
    - `11694.64 -> 11529.88 us`
    - `57065.60 -> 56447.20 us`
  - A99 is accepted and becomes the next continuation baseline.

## A100 Candidate: BF16 Lower Row-Prefix64 on A99 Sync Boundary

- Starting point: accepted A99 source SHA256 `4669f1ee6ccd59704173fe0389702cdc067e8762188c232158d74cd7f291a393`.
- Candidate source SHA256: `cde4220b86133c50e19d73fdf6c8b38200d5ff7f068d26078dca536bc6984b6f`.
- Single-point change:
  - in `StorePackedL1InverseToBf16Attn()`, row `48, cols=64` also uses `CopyPackedHalfL1RowPrefixToGmBf16()`;
  - rows `0/16/32` stay on A99 row-prefix rectangles;
  - A82/A99 sync boundary is unchanged.
- Package/install:
  - package `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`, SHA256 `276f84b83d86e8bb25a822b2e543dfd8b3cc3f578fe5a6d0e72eda1565231a47`;
  - isolated OPP `/tmp/packed64_phaseA100_bf16_rowprefix64_a99_sync_20260513_0805_opp/vendors/custom_transformer`;
  - installed Neumann object SHA256 `00cfc9e95568bcd01396291d33e800453590faafffb3777edc0340820ee0408a`;
  - installed opapi exports CGDR/Neumann symbols only, no `TriInverse`.
- Gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.64s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA100_bf16_rowprefix64_a99_sync_20260513/performance_compare.csv`, SHA256 `ca1241af253808be360e93b9251a2eae505b91ac7c8883668b44f0e075476ab0`;
  - raw CSV SHA256 `8a2c82c4c22397bd85f531fdc421ffddb4fc281f8a3800d5bbceba5c29734a2c`;
  - ratios `S=64/512/2048/10240 = 1.173 / 1.448 / 2.190 / 2.225`.
- Interpretation:
  - compared with A99 `1.162 / 1.544 / 2.288 / 2.330`, A100 gives a small `S=64` rollback but clearly improves `S=512/2048/10240`;
  - Neumann absolute task duration decreases versus A99 on all four profiled sizes:
    - `781.92 -> 745.72 us`
    - `2567.70 -> 2423.58 us`
    - `11529.88 -> 11072.26 us`
    - `56447.20 -> 53936.68 us`
  - A100 is accepted and becomes the next continuation baseline.

## A101 Candidate: Remove Dead BF16 Lower Block Fallback

- Starting point: accepted A100 baseline SHA256 `cde4220b86133c50e19d73fdf6c8b38200d5ff7f068d26078dca536bc6984b6f`.
- Single-point code change:
  - `StorePackedL1InverseToBf16Attn()` now always uses row-prefix BF16 writeback for lower rows;
  - remove the now-unreachable per-`16x16` lower block fallback branch;
  - remove `CopyPackedHalfL1BlockToGmBf16()`, which has no remaining active-file call site after A100.
- Scope held fixed:
  - no P7/P8 formula change;
  - no resident/RHS change;
  - no BF16 GM materialization deletion;
  - no upper-zero rectangle change.
- Package/install:
  - package `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`, SHA256 `bdbbacab10392e574e8e53b29ac38d7e8d99e15c76220be48a4f00e3bebd8140`;
  - isolated OPP `/tmp/packed64_phaseA101_bf16_lower_deadbranch_cleanup_20260513_0816_opp/vendors/custom_transformer`;
  - installed Neumann object SHA256 `0663db279808d62d4d28df413cc690af02cc410a927a5bcfc9146b630f29c24f`;
  - installed opapi exports CGDR/Neumann symbols only, no `TriInverse`.
- Gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.17s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA101_bf16_lower_deadbranch_cleanup_20260513/performance_compare.csv`, SHA256 `e95ce451738b7bb2c616f77af9a2141c5953fcb0905c30da15e7e3381fe88516`;
  - raw CSV SHA256 `f97642656967f27cc76750a57082eb0fea70b99be345d47745611711153add93`;
  - ratios `S=64/512/2048/10240 = 1.118 / 1.455 / 2.185 / 2.200`.
- Interpretation:
  - compared with A100 `1.173 / 1.448 / 2.190 / 2.225`, A101 improves `S=64/2048/10240`, with `S=512` only a small rollback;
  - Neumann absolute task duration:
    - `745.72 -> 725.38 us`
    - `2423.58 -> 2431.24 us`
    - `11072.26 -> 11005.92 us`
    - `53936.68 -> 53707.44 us`
  - scalar and Fixpipe both continue to fall on the medium/large rows, so A101 is accepted as the next continuation baseline.

## A102 Candidate: Raw-MMAD Epoch Auxiliary Tiles

- Starting point: accepted A101 source SHA256 `09b97b6ea9fde5086a25918444d2d7d99c20f634f49de8e92a4d120955ccdaf2`.
- Candidate source SHA256: `9cd969fc8edef610266c2dd974b70bb214eeb6c9f3d3ebfd758607736d50ac01`.
- Single-point change:
  - `InitRawMmadLocalBuffers()` now calls `PrepareRawMmadAuxiliaryTiles()`;
  - `PrepareRawMmadAuxiliaryTiles()` prepares `zero64` and `identity64` once per raw-MMAD local buffer epoch;
  - active full64 helpers no longer each lazily regenerate those auxiliary tiles.
- Scope held fixed:
  - no P7/P8 formula change;
  - no BF16 GM writeback deletion;
  - no resident/RHS change;
  - no Stage2/Stage3/QK/`gBK/vBeta` change.
- Package/install:
  - after detecting a misaligned `build_out` and default `custom_transformer` op_api symbol pollution, old `../../build*` outputs were moved aside and the package was clean rebuilt;
  - package `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`, SHA256 `e4b17d8eba153286e381f42310274ab2a5a3f4531ab8aade1a59b1af0a6044c6`;
  - isolated OPP `/tmp/cgdr_a102_opp_20260513_1006/vendors/custom_transformer`;
  - installed active header SHA256 `9cd969fc8edef610266c2dd974b70bb214eeb6c9f3d3ebfd758607736d50ac01`;
  - installed Neumann object SHA256 `4b9f6bf5e43c02df16cf5a52ccacc107d819e5ec65fcb239631d44b520c6a450`;
  - installed opapi exports both ChunkGatedDeltaRule and ChunkGatedDeltaRuleNeumann symbols.
- Gate:
  - isolated build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.20s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA102_raw_epoch_auxiliary_tiles_valid_20260513/performance_compare.csv`, SHA256 `4e3578020edf0d838cc7bf636da08040445144322f3fd0033844b0f08eca6747`;
  - raw CSV SHA256 `0b18e82b6950a50c4e16117ee19bda409be0b698e75a5611a4df978e605d8c45`;
  - ratios `S=64/512/2048/10240 = 1.082 / 1.434 / 2.182 / 2.209`.
- Interpretation:
  - compared with A101 `1.118 / 1.455 / 2.185 / 2.200`, A102 improves `S=64/512/2048`;
  - `S=10240` has a small rollback in this run, with Neumann absolute task duration `53707.44 -> 53842.58 us`;
  - the optimization is accepted because it is correctness-safe, removes repeated local generation paths, and improves three of four profiling points.
- Invalid data note:
  - `tests/pytest/prof_out/packed64_phaseA102_raw_epoch_auxiliary_tiles_20260513/` was produced while the default `custom_transformer` op_api had been overwritten by a no-CGDR-symbol package; do not use it for performance conclusions.

## A103a/A103b Result: Drop BF16 Attn GM Inverse Materialization

- A103a single-point change:
  - removed the active call to `StorePackedL1InverseToBf16Attn(paraIdx)`;
  - retained `StorePackedL1InverseToResidentL1(paraIdx)`;
  - RHS remains `FusedRhsAICProcess()` consuming `aicHalfL1ResidentInv_`.
- A103a gate:
  - isolated build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.29s`.
- A103a performance:
  - `tests/pytest/prof_out/packed64_phaseA103a_drop_bf16_attn_gm_write_valid_20260513/performance_compare.csv`;
  - ratios `S=64/512/2048/10240 = 1.106 / 1.364 / 2.164 / 2.194`.
- A103b single-point cleanup:
  - deleted unreachable BF16 inverse GM helper code after A103a proved no active consumer;
  - removed `attnWsGm_`;
  - retained the legacy workspace slot reservation with the comment `Reserved legacy BF16 inverse slot; no active consumer after resident-L1 RHS.`
- A103b source/baseline:
  - source SHA256 `08c14d7639b4d50d77dd9b5a9aee7e180e6af7a9d7c690566fa7f54a0dab9e62`;
  - baseline `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA103b_bf16_attn_gm_dead_cleanup_baseline.h`;
  - baseline SHA256 `08c14d7639b4d50d77dd9b5a9aee7e180e6af7a9d7c690566fa7f54a0dab9e62`.
- A103b package/install:
  - package SHA256 `b19417bf3fefce51f899a6612c8d4f143f49a08cb126f84a718ac4601a7d56b2`;
  - isolated OPP `/tmp/cgdr_a103b_opp_20260513_103205/vendors/custom_transformer`;
  - installed Neumann object SHA256 `c707508dacf596d71dab77209ada19a2d5c91c9fddad5403e57e792a3c7cce70`.
- A103b gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.52s`.
- A103b performance:
  - `tests/pytest/prof_out/packed64_phaseA103b_bf16_attn_gm_dead_cleanup_valid_20260513/performance_compare.csv`;
  - ratios `S=64/512/2048/10240 = 1.035 / 1.359 / 2.169 / 2.190`.
- Interpretation:
  - compared with A103a `1.106 / 1.364 / 2.164 / 2.194`, A103b improves `S=64/512/10240` and slightly regresses `S=2048`;
  - compared with A102 `1.082 / 1.434 / 2.182 / 2.209`, A103b improves all four profiled ratios;
  - A103b is accepted as the current continuation baseline.

## A104 Plan Update: Full CC Input Preload

- User-directed next target after A103: do not start with diagonal-only preload; load the inverse CC block into L1 by MTE2 as a full `64x64` tile.
- A104 single-point target:
  - preload `INVInitHalfWsGm_` full `64x64` to `aicHalfL1Scratch_`;
  - clean cache over full CC size for `INVInitHalfWsGm_` at the preload point;
  - keep `L_powHalfWsGm_` as packed diagonal preload, because P7 power only consumes diagonal lanes;
  - P7 reads diagonal input from the resident full `INVInit` L1 tile and writes result64 to `aicHalfL1P3Result_`;
  - P8 reads lower blocks from the same resident `aicHalfL1Scratch_`, eliminating strict-path P8 lower GM replay.
- Scope not included in A104:
  - no P7/P8 formula change;
  - no direct resident final `cInv` writeback;
  - no identity64 lifecycle extension;
  - no Stage2/Stage3/QK/`gBK/vBeta`/final RHS changes.

## A104 Result: Full INVInit CC Preload with Per-Line Clean

- Candidate source SHA256: `144c53a63ed6c36eb6d21a58d851513f453602092547c5f397026b060df495f6`.
- Single-point change:
  - strict full64 path keeps `L_powHalfWsGm_` as packed diagonal preload;
  - strict full64 path loads full `INVInitHalfWsGm_` `64x64` into `aicHalfL1Scratch_`;
  - P7 reads diagonal `INVInit` from `aicHalfL1Scratch_`;
  - P8 reads lower blocks from the same `aicHalfL1Scratch_`;
  - strict path no longer calls P8 lower GM preload.
- Package/install:
  - package SHA256 after install check `91dc3139e21cc850c9bf485e2f02104466726c698e9cf1023e932ce1fb1e7e3b`;
  - isolated OPP `/tmp/cgdr_a104_opp_20260513_105420/vendors/custom_transformer`;
  - installed Neumann object SHA256 `8589d10815dd9d99ed4172ae6fcaf1e872103725104d16a3ab973a156a5ff196`;
  - opapi exports CGDR/Neumann only, no `TriInverse`.
- Gate:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.97s`.
- Correctness baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA104_invinit_full_cc_l1_preload_baseline.h`;
  - SHA256 `144c53a63ed6c36eb6d21a58d851513f453602092547c5f397026b060df495f6`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA104_invinit_full_cc_l1_preload_valid_20260513/performance_compare.csv`;
  - compare CSV SHA256 `c4fa400da6a6468ff053d7f777c177b9288a9a9477eab63224a2732bf26ffa7c`;
  - raw CSV SHA256 `992fdc43d9189943f251b91c913f98c4fd34bb5b9dadb8e844e401331ddd991c`;
  - ratios `S=64/512/2048/10240 = 1.078 / 1.457 / 2.168 / 2.212`.
- Interpretation:
  - compared with A103b `1.035 / 1.359 / 2.169 / 2.190`, A104 improves only `S=2048` by a noise-scale amount and regresses `S=64/512/10240`;
  - AIC MTE2 ratio improves (`S=512: 0.882 -> 0.747`, `S=10240: 1.007 -> 0.881`), but AIC scalar ratio regresses (`S=512: 2.208 -> 2.399`, `S=10240: 3.243 -> 3.440`);
  - root hypothesis: full `INVInit` preload is correct, but `CleanHalfMatrixCacheLines(gm, 64, 64, 64)` does 256 single-cacheline DCCI and offsets the reduced MTE2/replay cost.
- Status:
  - A104 is correctness-safe and saved as a baseline, but not accepted as performance continuation surface yet.
  - Continue inside the same target as A104b: keep full `INVInit` preload, replace per-line full CC clean with one entire-data-cache clean.

## A105/A106 Result: Drop Stage1 Explicit DCCI

- User correction:
  - follow origin Stage1: if origin does not add explicit cache clean/flush around same-kernel workspace/output paths, do not keep the extra DCCI in p3l1_p8a;
  - do not change algorithm;
  - do not remove required `SetFlag/WaitFlag` or `Sync*` ordering while deleting DCCI.
- A105:
  - single-point: removed inverse scratch DCCI covered by existing MTE3 flag/wait:
    - `L_powHalfWsGm_` / `INVInitHalfWsGm_` write-after flush;
    - inverse GM->L1 read-before clean;
  - source/baseline SHA256 `36d2969ba9aa9cd3a0c61083c805a84331fd2588322e02d97786fd1591ea63a5`;
  - baseline `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA105_drop_inverse_dcci_baseline.h`;
  - gate PASS: build/install, `S=64/256/1024`, full compare `16 passed in 6.18s`;
  - performance ratios `1.016 / 1.308 / 2.076 / 2.103`.
- A106:
  - single-point after A105: removed all remaining active explicit `DataCacheCleanAndInvalid` wrappers/calls from p3l1_p8a Stage1:
    - `gBKHalfWsGm_` / `vBetaHalfWsGm_` RHS scratch write-after flush and read-before clean;
    - BF16 output/workspace flush calls;
    - old float/half clean/flush calls;
    - dead `Clean*CacheLines()` / `Flush*CacheLines()` helpers and cacheline constants.
  - retained sync/order boundaries:
    - `SyncMte3ToMte2`, `SyncFixToMte2`, `SyncFixToS`, cross-core flags, Fixpipe/MTE waits.
  - failed intermediate note:
    - a broader candidate that deleted sync together with DCCI passed `S=64/256` but failed `S=1024`;
    - root boundary: DCCI cleanup is valid, sync deletion is not part of this optimization.
  - source/baseline SHA256 `a08e61279067653747e67dda533bbbb50e306e051d20acdcb6729df92aefa9d8`;
  - baseline `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA106_drop_all_dcci_baseline.h`;
  - package SHA256 `8c38d903e73b1aced7454e9f74949e48999c5b6afee48710e995f846526cdbc7`;
  - isolated OPP `/tmp/cgdr_a106_drop_all_dcci_sync_kept_20260513_120119/vendors/custom_transformer`;
  - installed Neumann object SHA256 `27afd00cd02fce78a58a7aa83bff1ccef3fbfbe84b78ca850c87dd0d238733e8`.
- A106 gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.17s`.
- A106 performance:
  - `tests/pytest/prof_out/packed64_phaseA106_drop_all_dcci_20260513/performance_compare.csv`, SHA256 `2231e31a64316d61fff07bf5b09c50b481eda1231f34d66810b51fdfa28ae8d1`;
  - raw CSV SHA256 `57b05ccd8d69e931b79b44e7bb1fae54b67488ab197f615018c4f167a5ae6258`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.892 / 0.864 / 0.926 / 0.928`;
  - scalar ratios `0.874 / 0.836 / 0.855 / 0.854`;
  - AIC MTE2 ratios `0.693 / 0.740 / 0.869 / 0.882`.
- Interpretation:
  - compared with A105 `1.016 / 1.308 / 2.076 / 2.103`, A106 improves all four profiled sizes materially;
  - A106 is accepted as the latest correctness/performance continuation baseline.

## A107/A108 Sync Cleanup

- User direction:
  - treat A107, which deletes the QK write-after `SyncFixToMte2()` before `CrossCoreWaitFlag(0x7)`, as the performance baseline despite mixed A106 comparison;
  - continue stacking sync cleanup on top of A107;
  - keep performance status in this work log and emit a compact CSV at `tests/pytest/prof_out/performance_compare.csv`.
- A107 status:
  - source SHA256 was recorded as `05fae483373f81dca8d92fd3d3734483dc6d20cbe11b34b6b2703c5aa61fd267`;
  - gate PASS: build/install, `S=64/256/1024`, full compare `16 passed in 6.09s`;
  - performance ratios recorded in the active run context: `S=64/512/2048/10240 = 0.864 / 0.892 / 0.921 / 0.945`;
  - note: the expected local A107 prof directory was not present under `tests/pytest/prof_out/` during this update, so the compact compare uses the recorded A107 ratio values as the original row.
- A108 stacked candidate:
  - single additional change on top of A107: removed `SyncFixToS()` after final RHS `CopyL0CMatrixToGmBf16()` in `FusedRhsAICProcess()`;
  - still keeps `SyncFixToMte2()` in that RHS path;
  - source SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`;
  - package SHA256 `00fa8b390b3137f212afc746d5360fea119ab5fc5fdda2cf155238d0eda13675`;
  - isolated OPP `/tmp/cgdr_a108_drop_qk_and_rhs_fix_s_20260513_124655/vendors/custom_transformer`;
  - installed Neumann object SHA256 `ead1daf20b419b89a17903a4950a64657df1d248210cf9ae960cf1f98cff8f89`.
- A108 gate:
  - build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.30s`.
- A108 performance:
  - `tests/pytest/prof_out/packed64_phaseA108_drop_qk_and_rhs_fix_s_20260513/performance_compare.csv`, SHA256 `18970dc8809031fd700580f2c253ff51b481a6cfd9442461496896f594031114`;
  - raw CSV SHA256 `30a2b52b2c1c283055ec78894856a92269294edfe1ae0d748ff316f0cb583d02`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.885 / 0.848 / 0.910 / 0.920`;
  - current/A107 ratio `1.025 / 0.950 / 0.988 / 0.973`.
- Interpretation:
  - A108 improves over A107 for `S=512/2048/10240` and regresses `S=64` by about `2.49%`;
  - compact compare CSV written to `tests/pytest/prof_out/performance_compare.csv` with rows: original A107, current A108, current/original.

## Sync Cleanup Performance Standard

- For the current sync-cleanup phase, judge performance mainly by long-sequence benefit.
- Short sequence (`S=64`) may regress or stay unchanged because fixed overhead and launch/sync noise dominate.
- Long sequence (`S=2048/10240`) is the primary acceptance signal because repeated Stage1 work amplifies real reductions in redundant synchronization, movement, or scalar overhead.
- Compact CSV still records all profiled shapes, but interpretation should prioritize `S=2048` and `S=10240` unless correctness or a severe short-sequence regression appears.

## Long-Sequence Performance Standard

- Continue judging new inverse/writeback optimizations mainly by long-sequence performance.
- `S=2048` and `S=10240` are the acceptance-critical rows because redundant inverse, resident writeback, movement, and scalar overhead repeat enough times to expose real gains.
- `S=64` may regress or stay neutral; it is a smoke/overhead signal, not the main performance target, unless the regression is severe enough to indicate a broken dataflow.
- Every accepted continuation baseline must still pass correctness first; performance acceptance requires no clean long-sequence regression against the active baseline, and preferably an improvement in both long-sequence rows.

## A109 Candidate: Drop RHS Fix-to-MTE2 Sync

- Single additional change on top of A108:
  - removed the RHS local `SyncFixToMte2()` after `SyncFixToM()` in `FusedRhsAICProcess()`;
  - kept `SyncMToFix()` before final RHS Fixpipe;
  - kept `SyncFixToM()` after final RHS Fixpipe.
- Candidate source SHA256: `fd4fad1414a4079557f696f8fecc86463b718919296ab4f4e09427a85e437194`.
- Gate status:
  - build/install completed;
  - `S=64` PASS;
  - `S=256` PASS;
  - `S=1024` FAIL.
- Failure boundary:
  - `CPU vs ChunkGDRNeumann out max_abs=3.222431e+01`;
  - `CPU vs ChunkGDRNeumann state max_abs=1.557045e-01`;
  - failure first appeared at the long-sequence smoke, so this sync is required for the current RHS loop visibility/order.
- Result:
  - rejected as a redundant-sync deletion;
  - restored only this candidate by re-adding the RHS `SyncFixToMte2()`;
  - active source returned to A108 SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`.

## A110 Candidate: Drop L_pow Write-After MTE3-to-MTE2 Sync

- Single additional change on top of A108:
  - removed the first `SyncMte3ToMte2()` in `InverseCompute()`, immediately after writing `L_powHalfWsGm_`;
  - rationale tested: the same AIV path continues from UB data, and AIC reads `L_powHalfWsGm_` only after the later cross-core `0x7` flag.
- Candidate source SHA256: `a160530b4e2f8176fa1ef3a0c7c87372fcc97fc591c1a4cf7c07695f85bd78cf`.
- Package/install:
  - package SHA256 `2e18e896d3148458fe68cc96c8b70c9a7be97ecfbabe00fc838d3815fe0cfde5`;
  - isolated OPP `/tmp/cgdr_a110_drop_lpow_mte3_mte2_20260513_130547/vendors/custom_transformer`;
  - installed Neumann object SHA256 `abd6e061a4d35b6dc24644fa287447829787fabacefa0ddd103133b4e6d04af7`.
- Gate:
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.21s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA110_drop_lpow_mte3_mte2_20260513/performance_compare.csv`, SHA256 `759987a6b7d6d223ac46c76744a7137a181a29d7d46291e0417edc0417295169`;
  - raw CSV SHA256 `9a7fe68719443899953cf54a5db97515a1fb5a842046454fb79821d565b247d4`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.953 / 0.841 / 0.916 / 0.923`;
  - current/A108 ratio `1.077 / 0.992 / 1.007 / 1.003`.
- Result:
  - rejected as a continuation baseline despite precision PASS, because the long-sequence standard regresses at `S=2048/10240`;
  - compact compare CSV refreshed at `tests/pytest/prof_out/performance_compare.csv`, SHA256 `d88c17932798c18067161ec317b681b29dfd71321044b6169af53731425d0596`;
  - restored only this candidate by re-adding the first `InverseCompute()` `SyncMte3ToMte2()`;
  - active source returned to A108 SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`.

## A111 Candidate: Drop gBK BF16 Write-After MTE3-to-MTE2 Sync

- Single additional change on top of A108:
  - removed only the `SyncMte3ToMte2()` immediately after BF16 `gBKWsGm_` write in `GBKCompute()`;
  - retained the FP16 `gBKHalfWsGm_` write and its `SyncMte3ToMte2()` because RHS consumes `gBKHalfWsGm_`.
- Candidate source SHA256: `621b736b47f106158833af2d841b1d06a7c985437243cd6989c88a7b41e99765`.
- Package/install:
  - package SHA256 `58ff812949de1d7e1758d5b14d82160ec2938f15e69ab3396fcf7050be4b0ea8`;
  - isolated OPP `/tmp/cgdr_a111_drop_gbk_bf16_mte3_mte2_20260513_131504/vendors/custom_transformer`;
  - installed Neumann object SHA256 `be8aebc4fe147b2701ea2cdb4e1b3ce62906d228d2e761e3bc9691f3639188fd`;
  - installed JSON SHA256 `96f54bab039fb115e40f3cee3d3c5ad996eeed2e261d4b89114606f75471a782`.
- Gate:
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.18s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA111_drop_gbk_bf16_mte3_mte2_20260513/performance_compare.csv`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.898 / 0.858 / 0.934 / 0.938`;
  - current/A108 ratio `1.014 / 1.012 / 1.027 / 1.020`.
- Result:
  - rejected as a continuation baseline despite precision PASS, because the long-sequence standard regresses at `S=2048/10240`;
  - restored only this candidate by re-adding the BF16 `gBKWsGm_` write-after `SyncMte3ToMte2()`;
  - active source returned to A108 SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`;
  - compact compare CSV refreshed at `tests/pytest/prof_out/performance_compare.csv`, SHA256 `472ea3103909d49ede7ab9686b3e7c5650248514f9eb5b72bb1cf77b542ed830`.

## A112 Candidate: Drop vBeta BF16 Write-After MTE3-to-MTE2 Sync

- Single additional change on top of A108:
  - removed only the `SyncMte3ToMte2()` immediately after BF16 `vBetaWsGm_` write in `VBetaCompute()`;
  - retained the FP16 `vBetaHalfWsGm_` write and its `SyncMte3ToMte2()` because RHS consumes `vBetaHalfWsGm_`.
- Candidate source SHA256: `a75e55c1296bba049b389b6d39386211814de2cb193c82310373c9a61657759c`.
- Package/install:
  - package SHA256 `e9a0ea0240c8c299c7bfca6393e3608ef7e273d2d81f58f85dcee13a1241e716`;
  - isolated OPP `/tmp/cgdr_a112_drop_vbeta_bf16_mte3_mte2_20260513_1327/vendors/custom_transformer`;
  - installed Neumann object SHA256 `5ffa466c51065e957f16097754da43135a07b17baa183d3c9b0969bf285e0d6f`;
  - installed JSON SHA256 `862f45bda3eb8a48b942fb88dcf5cd7cb5982f59dc714846d5550c021da73bde`.
- Gate:
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.11s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA112_drop_vbeta_bf16_mte3_mte2_20260513/performance_compare.csv`, SHA256 `699f1558539138112184d97433e3c050868e326ce15b75da9daab18edf66d87f`;
  - raw CSV SHA256 `6b2a4018a5f6e48fb7dd3c4c658a9bb46dcac2dca7be0adad815d709cdbaab52`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.922 / 0.867 / 0.931 / 0.913`;
  - current/A108 ratio `1.042 / 1.023 / 1.023 / 0.993`.
- Result:
  - rejected as a continuation baseline despite precision PASS;
  - long-sequence signal is not clean: `S=2048` regresses by about `2.26%`, while `S=10240` improves only about `0.69%`;
  - restored only this candidate by re-adding the BF16 `vBetaWsGm_` write-after `SyncMte3ToMte2()`;
  - active source returned to A108 SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`;
  - compact compare CSV refreshed at `tests/pytest/prof_out/performance_compare.csv`, SHA256 `58d00f1ff012bcf98f130320c4f70d658afc5daf2365dae0a08042215d9164bd`.

## A113 Candidate: Drop qPrime Stage1-Out MTE3-to-MTE2 Sync

- Single additional change on top of A108:
  - removed only the `SyncMte3ToMte2()` immediately after BF16 `outQPrimeGm_` write in `QPrimeCompute()`;
  - did not change Stage2, qPrime math, or qPrime output layout.
- Boundary checked before edit:
  - `RunStage1()` returns and the top-level kernel executes `SyncAll<false>()` before `RunStage2()` reads `qPrime_`;
  - the candidate tested whether the local `MTE3->MTE2` wait at qPrime write site is redundant with that stage boundary.
- Candidate source SHA256: `4a8aab21efbacdd52a9e5642bd574d709a0914e6a2e8b5bceb084baa4e86f027`.
- Package/install:
  - package SHA256 `7e05eedffeb8f0b450664b3c478a2bb929da85b07fc3834714433b69f3a1edda`;
  - isolated OPP `/tmp/cgdr_a113_drop_qprime_mte3_mte2_20260513_1336/vendors/custom_transformer`;
  - installed Neumann object SHA256 `fd046b14660afb00d13272e800c01484e2fb83bf168573cd846a64d1c75a8042`;
  - installed JSON SHA256 `b0761595efdc2c33b663be009a4ed05c2c6409c5314c292da550db964ca414e9`.
- Gate:
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.12s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA113_drop_qprime_mte3_mte2_20260513/performance_compare.csv`, SHA256 `080a273e20cf329d5239b200c66a17f8e16ceadc2fde0fca3c6551f57e97c71a`;
  - raw CSV SHA256 `1ebad6e6392b1e6be7cf0910aefff36b021d399718751f02d3461dbef3ed4ea8`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.904 / 0.900 / 0.934 / 0.938`;
  - current/A108 ratio `1.021 / 1.061 / 1.026 / 1.020`.
- Result:
  - rejected as a continuation baseline despite precision PASS;
  - long-sequence standard regresses at both `S=2048` and `S=10240`;
  - restored only this candidate by re-adding `QPrimeCompute()` write-after `SyncMte3ToMte2()`;
  - active source returned to A108 SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`;
  - compact compare CSV refreshed at `tests/pytest/prof_out/performance_compare.csv`, SHA256 `47e71c78c066d99d60ef04951b787be54b4d8991af9341e7b554425d3c972ee8`.

## A114 Result: Direct Final cInv Fixpipe to Resident L1

- Starting point: A108 source SHA256 `8dadb22f3363d4b933ac4efd6134672b332290c58fd7da9f3a7c978b66640459`.
- Candidate/source SHA256: `e586f344e24fb297e3dd7b91598a66a45515e4c19452b2b8e6639fb96743e34b`.
- Single-point change:
  - strict full64 path now writes final P8 second-layer `cInv` directly to `aicHalfL1ResidentInv_[paraIdx]` by `FixpipePacked64FullToHalfL1(resident, cInv, ...)`;
  - strict full64 path returns immediately after P8 second-layer, so it skips `StorePackedL1InverseToResidentL1()`;
  - this removes the strict-path resident upper-zero plus four row-prefix `X @ I64` replay copies;
  - tail/non-64 fallback still uses the old resident copy-out.
- Scope guard:
  - no P7/P8 formula change;
  - no BF16 inverse GM materialization reintroduction;
  - no RHS math or RHS operand change;
  - no Stage2/Stage3/QK/`gBK/vBeta` generation change.
- Build/package/install:
  - package SHA256 `735f5d4442fa21c07b8b414f887850c838778787291be88878e3720f98801cdb`;
  - isolated OPP `/tmp/cgdr_a114_direct_cinv_resident_20260513_1353/vendors/custom_transformer`;
  - installed Neumann object SHA256 `220deb8495bacf363b0f5fabe358e143fe72d1bbd83b8ca38b4c05415676f653`;
  - installed JSON SHA256 `53699791164b69df2ecf9435e9fb9cacfb1f9bdd1cbaf9ec981ee9b70fea4673`;
  - installed opapi exports only `aclnnChunkGatedDeltaRule*` and `aclnnChunkGatedDeltaRuleNeumann*`.
- Gate PASS:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.25s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA114_direct_cinv_resident_baseline.h`;
  - SHA256 `e586f344e24fb297e3dd7b91598a66a45515e4c19452b2b8e6639fb96743e34b`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA114_direct_cinv_resident_20260513/performance_compare.csv`, SHA256 `96a223ad55804b42951edcd974f59a6a55472e5ef1416dd722313afaa01c50cd`;
  - raw CSV SHA256 `4b52264a5527a529c90f75b5231f77c1a944ca382cb6c858fea19267f3cce853`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.968 / 0.866 / 0.902 / 0.901`;
  - A114/A108 = `1.093 / 1.021 / 0.991 / 0.980`.
- Interpretation:
  - accepted as the new continuation baseline by the long-sequence standard;
  - `S=2048` improves by about `0.94%` and `S=10240` improves by about `1.97%` versus A108;
  - `S=64/512` regress, but current performance standard prioritizes long-sequence benefit.
- Compact compare CSV refreshed:
  - `tests/pytest/prof_out/performance_compare.csv`, SHA256 `74b62d32bb028a32de742765c7b0e1cf27b1884d73d558f530a9fea099851270`.

## A115 Result: Origin-Style Drop Stage1 Local MTE3-to-MTE2 Syncs

- Starting point: A114 source SHA256 `e586f344e24fb297e3dd7b91598a66a45515e4c19452b2b8e6639fb96743e34b`.
- Candidate/source SHA256: `a48daa7d18c8c1d5a817023e9a814fe73c41f96b92ec13eddc5746fbc553ad80`.
- Single-point change:
  - removed the local write-after `SyncMte3ToMte2()` calls that origin Stage1 does not have;
  - removed both `InverseCompute()` half scratch local waits after `L_powHalfWsGm_` and `INVInitHalfWsGm_` writes;
  - removed both `GBKCompute()` local waits after `gBKWsGm_` BF16 and `gBKHalfWsGm_` half writes;
  - removed both `VBetaCompute()` local waits after `vBetaWsGm_` BF16 and `vBetaHalfWsGm_` half writes;
  - removed `QPrimeCompute()` local wait after `outQPrimeGm_` BF16 write.
- Retained boundaries:
  - outer `CrossCoreSetFlag<0x2, PIPE_MTE3>()` stage boundaries remain, matching the origin Stage1 synchronization style;
  - RHS local `SyncFixToMte2()` remains because A109 proved deleting it fails at `S=1024`;
  - no P7/P8 formula, resident inverse, RHS math, Stage2/Stage3/QK generation, or `gBK/vBeta` math changes.
- Build/package/install:
  - package SHA256 `cea3f6e2370ea4a9dac3d229abcb687aec4aa1616453abbcb58e7c76712cc662`;
  - isolated OPP `/tmp/cgdr_a115_origin_style_drop_stage1_mte3_mte2_20260513_1411/vendors/custom_transformer`;
  - installed Neumann object SHA256 `7f216b7fbbea662cbb67f11b413606a1e265846fbe1cf72ec68256c561cdec8e`;
  - installed JSON SHA256 `cdd9ab431a008486c7953fe2e74d2e0226b47d7d8f4a1e4dae41999352fb1a29`.
- Gate PASS:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.13s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA115_origin_style_drop_stage1_mte3_mte2_baseline.h`;
  - SHA256 `a48daa7d18c8c1d5a817023e9a814fe73c41f96b92ec13eddc5746fbc553ad80`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA115_origin_style_drop_stage1_mte3_mte2_20260513/performance_compare.csv`, SHA256 `b6153ffcca1f02bdb39bb0eea37690274a1e915753f3d8522d38227de748bf78`;
  - raw CSV SHA256 `a1af8e9f5cd4d9529ca99753cc31fadf1a0c137976044d53b899d62f3516cc81`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.927 / 0.831 / 0.899 / 0.895`;
  - A115/A114 = `0.958 / 0.960 / 0.997 / 0.993`.
- Interpretation:
  - accepted as the new continuation baseline by the long-sequence standard;
  - `S=2048` improves by about `0.30%` and `S=10240` improves by about `0.67%` versus A114;
  - short/medium rows also improve in this run, but the acceptance-critical signal remains long sequence.
- Compact compare CSV refreshed:
  - `tests/pytest/prof_out/performance_compare.csv`, SHA256 `7e6253cbd5e079499dceecd0e6411941adce693e1179935f0d7f431e1c671ada`.

## A116 Result: Dead Strict Resident Row-Prefix Cleanup

- Starting point: A115 source SHA256 `a48daa7d18c8c1d5a817023e9a814fe73c41f96b92ec13eddc5746fbc553ad80`.
- Candidate/source SHA256: `4a2fcc34a88ea5f97d4c5d04a632d2ccb154dcb16b5134c09dbb5176c26d8bda`.
- Single-point change:
  - removed unreachable strict branch inside `StorePackedL1InverseToResidentL1()`;
  - removed now-unused `CopyPackedHalfL1RowPrefixToPackedHalfL1()` helper;
  - strict `validLen >= 64` path already returns after direct final `cInv` Fixpipe to `aicHalfL1ResidentInv_`, so this branch had no active consumer after A114.
- Scope guard:
  - fallback/tail resident copy path still uses `CopyHalfL1PackedBlockToCompact()`;
  - no P7/P8 formula, strict resident writeback, RHS math, Stage2/Stage3/QK generation, or `gBK/vBeta` math changes.
- Build/package/install:
  - package SHA256 `caff503e41e027324b6a7412f4217d23f38f16e6e82651aa93d5d548e5c1a681`;
  - isolated OPP `/tmp/cgdr_a116_dead_resident_rowprefix_cleanup_20260513_1422/vendors/custom_transformer`;
  - installed Neumann object SHA256 `a43dc048f9b35e88430a2527a38df2666598c39a0060b3964fb82be98db1aa42`;
  - installed JSON SHA256 `8183acf6fc693624e247152022fd84e803ee80e6ccfa6b2c083316a434045098`.
- Gate PASS:
  - clean build/install PASS;
  - `S=64/256/1024` smoke PASS;
  - full compare PASS: `16 passed in 6.15s`.
- Baseline saved:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA116_dead_resident_rowprefix_cleanup_baseline.h`;
  - SHA256 `4a2fcc34a88ea5f97d4c5d04a632d2ccb154dcb16b5134c09dbb5176c26d8bda`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA116_dead_resident_rowprefix_cleanup_20260513/performance_compare.csv`;
  - neumann/normal task ratios `S=64/512/2048/10240 = 0.903 / 0.832 / 0.903 / 0.897`;
  - A116/A115 = `0.974 / 1.002 / 1.004 / 1.001`.
- Interpretation:
  - accepted as code cleanup, not as a performance continuation improvement;
  - long-sequence deltas are neutral/slightly regressed versus A115, consistent with strict-path dead-code cleanup;
  - stop optimization here and proceed to requested directory cleanup and commit.
- Compact compare CSV refreshed:
  - `tests/pytest/prof_out/performance_compare.csv` now compares A115 to A116.

## Cleanup Commit Scope

- Renamed active Stage1 implementation:
  - from `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
  - to `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1.h`.
- `op_kernel/baseline/` now keeps only the latest A116 baseline.
- Removed inactive `op_kernel/halflocal`, `op_kernel/fulllocal`, `op_kernel/origin`, old p8afullocal helper files, pytest cache/debug dump scripts, variant-matrix scripts, and historical profile output directories.
