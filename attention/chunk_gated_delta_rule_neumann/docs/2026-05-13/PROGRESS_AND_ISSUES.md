# 2026-05-13 PROGRESS_AND_ISSUES

## Current Baseline

- A99 `BF16 lower row-prefix on A82 sync boundary` is the latest correctness-safe continuation baseline.
- Source SHA256: `4669f1ee6ccd59704173fe0389702cdc067e8762188c232158d74cd7f291a393`.
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA99_bf16_rowprefix_a82_sync_baseline.h`.
- Gate PASS:
  - build/install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.77s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA99_bf16_rowprefix_a82_sync_20260513/performance_compare.csv`
  - ratios `S=64/512/2048/10240 = 1.162 / 1.544 / 2.288 / 2.330`.
  - A99 improves Neumann absolute task time on all four profiled sizes versus A82; ratio improves at `S=64/2048/10240`, with `S=512` only noise-level regression.
- Current active source is A99 and matches the saved source hash above.

## Confirmed Boundaries

- P7 full64 Fixpipe writes are not simple duplicates; they currently preserve later L1 reads and final result exposure.
- RHS math consumes resident L1 (`aicHalfL1ResidentInv_`); BF16 GM `attnWsGm_` is not the RHS operand but still cannot be deleted without replacing the lifecycle/sync side effect.
- A81 proves explicit BF16 GM DCCI flush is not needed for the current path when Fixpipe writes and syncs remain; it does not prove the BF16 GM writeback itself is removable.
- A82 proves lower true-inverse BF16 block materialization does not need its local `SyncFixToMte2()` when `SyncFixToM()` and `SyncFixToS()` remain; zero-rect sync is still a separate boundary.
- A83 proves zero-rect BF16 materialization can pass precision without its local `SyncFixToMte2()`, but the performance tradeoff is worse for medium/large shapes; do not continue from A83.
- A84 proves lower true-inverse BF16 block materialization still needs its local `SyncFixToS()` for long-sequence correctness; do not delete it from A82.
- A85 proves zero-rect BF16 materialization can pass precision without its local `SyncFixToS()`, but medium/large performance regresses; do not continue from A85.
- Resident lower true-inverse copy still uses `X @ I16` replay. Direct L1->L1 replacement is deferred until a legal API/dataflow is proven.
- A40/A66 already proved P7 L0A no-transpose one-shot diagonal coalesced load fails at `S=1024`; do not retry that exact direction.
- A94 proves skipping RHS full-tile `aicHalfL1TilePower_` zero-init is correctness-safe but performance-negative; do not continue from A94.
- A99 proves BF16 lower rows `0/16/32` can reuse a row-prefix rectangle writeback on the A82 sync boundary; row `48` still remains on the older per-block writeback shape.

## A94 Result

- A94 `RHS full-tile zero-init drop` is correctness-safe but not performance-accepted.
- Candidate source SHA256: `4eede660d86bcdb61bdf24f4059a9ca484ce1a14532f534183ff1fa4a708f43f`.
- Gate:
  - clean build/install PASS after rejecting one stale `tri_inverse`-only package.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.14s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA94_rhs_fulltile_sync_20260513/performance_compare.csv`
  - ratios `1.228 / 1.603 / 2.317 / 2.344`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A94 regresses all profiled sizes.
- Active source restored to A82 and verified byte-identical to:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A83 Candidate

- A83 `BF16 zero-rect drop Fix-to-MTE2 sync` is correctness-safe but not performance-accepted.
- Candidate source SHA256: `f655ef6305a8f000e085e11bd9d416fc7f2adea80e2c3ce629f3a4808ec6a6ba`.
- Gate:
  - build/install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.17s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA83_bf16_zero_rect_drop_fix_mte2_sync_20260513/performance_compare.csv`
  - ratios `1.141 / 1.589 / 2.340 / 2.342`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A83 improves `S=64` only and regresses `S=512/2048/10240`.
- Active source restored to A82 and verified byte-identical to:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A84 Failed

- A84 `BF16 lower block drop Fix-to-Scalar sync` failed at `S=1024`.
- Candidate source SHA256: `59c935cba6a1f44cf0185c7745a24a2e65867e0467b4d773aaf9921fcd17a145`.
- Gate:
  - build/install PASS.
  - `S=64` PASS.
  - `S=256` PASS.
  - `S=1024` FAIL.
- Failure:
  - `CPU vs ChunkGDRNeumann out max_abs=1.576879e-01`.
  - `CPU vs ChunkGDRNeumann state max_abs=1.081070e-01`.
- Active source restored to A82 and verified byte-identical to:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## A85 Candidate

- A85 `BF16 zero-rect drop Fix-to-Scalar sync` is correctness-safe but not performance-accepted.
- Candidate source SHA256: `6655407d2f58f8e8c3ec42b685725c87f801211660d68fe94bc9e0933bf54837`.
- Gate:
  - clean build/install PASS after rebuilding the correct two-op package.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.20s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA85_bf16_zero_rect_drop_fix_to_scalar_sync_20260513/performance_compare.csv`
  - ratios `1.161 / 1.603 / 2.329 / 2.368`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A85 improves `S=64` only and regresses `S=512/2048/10240`.
- Active source restored to A82 and verified byte-identical to:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.

## Next Optimization Search

- Next higher-value candidates should come from profiling or a bounded design around:
  - resident lower true-inverse replay reduction with a proven movement path;
  - P7/A42 copy-out legality;
  - profiling-guided BF16/resident materialization sync split.
- A75 profiling still shows large Neumann overhead in AIC scalar, MTE1, and Fixpipe, so new work should target materialization/replay instead of already-failed L0A coalescing.

## A76 Update

- A76 `P8 first-layer first MMAD L0 zero-copy drop` is correctness-safe.
- Source/baseline SHA256: `7e54195c7419004d92214614d1c387c7203d3ad077bfaf3da42b1dd810b3f038`.
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA76_p8_first_lower_top_l0_zero_drop_baseline.h`.
- Gate PASS:
  - build/install PASS after fresh clean serial build.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.14s`.
- This only removes the first P8 full64 helper L1-zero-to-L0 clear; other P8 helpers and P7 copy-out remain unchanged.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA76_p8_first_lower_top_l0_zero_drop_20260513/performance_compare.csv`
  - ratios `S=64/512/2048/10240 = 1.200 / 1.654 / 2.416 / 2.456`.
  - A76 performance-regressed versus A75 at `S=64/2048/10240`; keep as correctness-safe but not as preferred performance surface.

## A77 Update

- A77 `resident lower row-prefix48` is correctness-safe.
- Source/baseline SHA256: `bdd6ecb7dcc55dd37f7e2a128ea3ae0a86750fd80f3bd4df1f58b3aac998d5d1`.
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA77_resident_lower_rowprefix48_baseline.h`.
- Gate PASS:
  - build/install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.17s`.
- Implementation:
  - strict full64 resident rows `0/16/32` use row-prefix MMAD-copy with cols `16/32/48`;
  - row `48` stays on the older per-block copy path.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA77_resident_lower_rowprefix48_20260513/performance_compare.csv`
  - ratios `1.147 / 1.663 / 2.409 / 2.433`.

## A78 Candidate

- Target: only extend resident row-prefix MMAD-copy to row `48, cols=64`, replacing the remaining four resident `X @ I16` replay copies.
- Do not change BF16 GM writeback, P7/P8 formulas, RHS math, Stage2/Stage3/QK, or `gBK/vBeta`.
- Risk: BF16 `cols=64` row-prefix had a prior long-sequence failure; resident row-prefix64 must pass `S=64/256/1024` before any performance claim.
- Result: PASS.
- Source/baseline SHA256: `dfb99b8fd881dc607b9ac24a37158558c81eb447ddcb5c09a67a748dd693a731`.
- Baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA78_resident_rowprefix64_baseline.h`.
- Gate:
  - build/install PASS.
  - `S=64/256/1024` smoke PASS.
  - full compare PASS: `16 passed in 6.30s`.
- Performance:
  - `tests/pytest/prof_out/packed64_phaseA78_resident_rowprefix64_20260513/performance_compare.csv`
  - ratios `1.194 / 1.626 / 2.403 / 2.424`.
  - Improves A77 at `S=512/2048/10240`; regresses A77 at `S=64`.

## A86/A87 Results

- A86 `BF16 zero-rect drop Fix-to-M sync` is correctness-safe but not performance-accepted.
  - Gate: build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.16s`.
  - Performance ratios: `1.194 / 1.566 / 2.334 / 2.344`, worse than A82 `1.191 / 1.544 / 2.313 / 2.332` on all profiled sizes.
- A87 `RawMmadPackedHalfL1AccumulateRect local HF32 cleanup` is correctness-safe but not performance-accepted.
  - Gate: build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.27s`.
  - Performance ratios: `1.195 / 1.585 / 2.322 / 2.366`, worse than A82 on all profiled sizes.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not continue from A86/A87. Do not retry zero-rect sync deletion or rect-helper local HF32 deletion without a materially new design.

## A88/A89 Results

- A88 `P7 diag drop M-to-MTE1 sync` is correctness-safe but not performance-accepted.
  - Gate: build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 7.01s`.
  - Performance ratios: `1.166 / 1.597 / 2.322 / 2.347`; improves only `S=64`, regresses `S=512/2048/10240` versus A82.
- A89 `defer BF16 GM materialization after K-RHS` is correctness-safe but not performance-accepted.
  - Gate: clean rebuild/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.23s`.
  - Performance ratios: `1.174 / 1.615 / 2.454 / 2.475`; improves only `S=64`, regresses medium/large shapes versus A82.
  - Rebuild note: the first build_out package was rejected because the installed `libcust_opapi.so` did not export CGDR/Neumann symbols.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not continue from A88/A89. Do not retry BF16 materialization delay unless the dataflow changes materially; the extra GM materialization still is not RHS input, but moving it after K-RHS did not hide the cost.

## A90 Result

- A90 `PrepareRawMmadState()` inverse-window hoist is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.85s`.
  - Performance ratios: `1.116 / 1.560 / 2.332 / 2.351`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A90 improves `S=64` only and regresses `S=512/2048/10240`.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not continue from A90. Hoisting scalar raw-MMAD state setup alone is not the medium/large performance direction.

## A91 Result

- A91 `resident L1 before BF16 GM writeback` is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.20s`.
  - Performance ratios: `1.255 / 1.595 / 2.326 / 2.370`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A91 regresses all four profiled sizes.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not continue from A91. Merely swapping resident/BF16 final materialization order does not improve the path.

## A92/A93 Results

- A92 `resident row-prefix identity ensure hoist` is correctness-safe but not performance-accepted.
  - Gate: clean `-j1` build/install PASS after a third-party generation race in the first `-j16` build, `S=64/256/1024` PASS, full compare `16 passed in 6.49s`.
  - Performance ratios: `1.248 / 1.592 / 2.334 / 2.356`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A92 regresses all profiled sizes.
- A93 `delay second-layer lower preload wait until after first-layer P8` is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS after one transient FATBIN retry, `S=64/256/1024` PASS, full compare `16 passed in 6.22s`.
  - Performance ratios: `1.217 / 1.587 / 2.301 / 2.344`.
  - Compared with A82, A93 improves only `S=2048` and regresses `S=64/512/10240`.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not retry A92 or A93 exact changes. Future wait/preload work needs a materially different dataflow, not just scalar check hoist or wait relocation.

## A95 Result

- A95 `P7 diagonal preload drop AIC cache clean` is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.53s`.
  - Performance ratios: `1.244 / 1.610 / 2.298 / 2.335`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A95 improves only `S=2048` and regresses `S=64/512/10240`.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not retry this exact AIC clean deletion. Profiling still points to AIC scalar/Fixpipe/MTE1 and layout-copy/MMAD replay, not MTE2 preload clean as the main medium/large bottleneck.

## A96 Result

- A96 `first P7 intermediate cInv drop Fix-to-MTE1 sync` is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.16s`.
  - Performance ratios: `1.140 / 1.582 / 2.322 / 2.363`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A96 improves only `S=64` and regresses `S=512/2048/10240`.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not retry this exact P7 intermediate sync deletion. It is another small-shape-only win and not the medium/large direction.

## A97/A98 Results

- A97 `second P7 intermediate cInv drop Fix-to-MTE1 sync` is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.14s`.
  - Performance ratios: `1.204 / 1.564 / 2.327 / 2.336`, worse than A82 on all profiled sizes.
- A98 `final P7 cInv drop Fix-to-MTE1 sync` is correctness-safe but not performance-accepted.
  - Gate: clean build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.15s`.
  - Performance ratios: `1.166 / 1.572 / 2.326 / 2.355`; improves only `S=64`, regresses `S=512/2048/10240`.
- Active source restored to A82:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA82_bf16_lower_block_drop_fix_mte2_sync_baseline.h`
  - SHA256 `ba186210c7ef75bcb7ef3c62a38fdb8e0ac3822906f0e02691d55d5edb2aedca`.
- Do not retry any exact P7 `cInv` Fix-to-MTE1 sync deletion from A96/A97/A98. All three are now disproved as medium/large continuation directions.

## A99 Result

- A99 `BF16 lower row-prefix on A82 sync boundary` is accepted.
  - Single-point change: rows `0/16/32` in `StorePackedL1InverseToBf16Attn()` use `CopyPackedHalfL1RowPrefixToGmBf16()`, while row `48` keeps the prior per-`16x16` block path.
  - Sync boundary remains A82-like: keep `SyncFixToM()` and `SyncFixToS()`, do not restore `SyncFixToMte2()` or flush.
  - Gate: build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.77s`.
  - Performance ratios: `1.162 / 1.544 / 2.288 / 2.330`.
  - Compared with A82 `1.191 / 1.544 / 2.313 / 2.332`, A99 improves `S=64/2048/10240`, leaves `S=512` effectively flat, and lowers Neumann absolute task duration on all four profiled sizes.
- Save A99 as the next continuation baseline; future BF16/P7 work should start from A99, not A82.

## A100 Result

- A100 `BF16 lower row-prefix64 on A99 sync boundary` is accepted.
  - Single-point change: row `48, cols=64` in `StorePackedL1InverseToBf16Attn()` also uses `CopyPackedHalfL1RowPrefixToGmBf16()`.
  - Gate: build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.64s`.
  - Performance ratios: `1.173 / 1.448 / 2.190 / 2.225`.
  - Compared with A99 `1.162 / 1.544 / 2.288 / 2.330`, A100 slightly regresses `S=64` ratio but materially improves `S=512/2048/10240`; Neumann absolute task duration drops on all four profiled sizes.
- Save A100 as the next continuation baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA100_bf16_rowprefix64_a99_sync_baseline.h`
  - SHA256 `cde4220b86133c50e19d73fdf6c8b38200d5ff7f068d26078dca536bc6984b6f`.

## A101 Current Candidate

- A101 removes dead BF16 lower block fallback left behind after A100:
  - lower rows always use row-prefix writeback;
  - delete unreachable per-`16x16` lower block branch and its sole active helper `CopyPackedHalfL1BlockToGmBf16()`.
- This is a code-size/scalar-side cleanup candidate only. It does not change P7/P8 math, resident/RHS semantics, or the requirement to retain BF16 GM materialization.
- A101 is accepted.
  - Gate: build/install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.17s`.
  - Performance ratios: `1.118 / 1.455 / 2.185 / 2.200`.
  - Compared with A100 `1.173 / 1.448 / 2.190 / 2.225`, A101 improves `S=64/2048/10240`, with `S=512` only a small rollback.
- Save A101 as the next continuation baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA101_bf16_lower_deadbranch_cleanup_baseline.h`
  - SHA256 `09b97b6ea9fde5086a25918444d2d7d99c20f634f49de8e92a4d120955ccdaf2`.

## A102 Result

- A102 `raw-MMAD epoch auxiliary tiles` is accepted.
  - Single-point change: `InitRawMmadLocalBuffers()` prepares resident L1 `zero64` and `identity64` once for the raw-MMAD local buffer epoch via `PrepareRawMmadAuxiliaryTiles()`.
  - Local helpers no longer repeat/lazily generate those tiles.
  - This is an epoch lifecycle, not whole-operator lifetime; `pipe_->Reset()` / `RestoreStage1MatmulLocalBuffers()` still invalidates it.
  - Gate: clean rebuild + isolated install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.20s`.
  - Performance ratios: `1.082 / 1.434 / 2.182 / 2.209`.
  - Compared with A101 `1.118 / 1.455 / 2.185 / 2.200`, A102 improves `S=64/512/2048`; `S=10240` is a small rollback in this run.
- Save A102 as the next continuation baseline:
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA102_raw_epoch_auxiliary_tiles_baseline.h`
  - SHA256 `9cd969fc8edef610266c2dd974b70bb214eeb6c9f3d3ebfd758607736d50ac01`.
- Environment issue recorded:
  - default `custom_transformer` op_api can be overwritten by unrelated package builds and lose CGDR symbols;
  - after any suspicious profiling failure, verify `nm -D .../libcust_opapi.so | rg ChunkGatedDeltaRule`;
  - trusted A102 profiling used isolated OPP `/tmp/cgdr_a102_opp_20260513_1006/vendors/custom_transformer`.
- Next active queue:
  - A103: delete BF16 `attnWsGm_` real inverse data write if no hidden consumer/arrival boundary exists.
  - A104: final `cInv` direct resident L1 writeback after A103 PASS.
  - A105: longer identity64 lifecycle only after a stable PASS surface, not mixed with BF16 deletion.

## A103 Result

- A103a removed active BF16 inverse GM materialization call and kept resident L1 RHS path.
  - Gate PASS: isolated build/install, `S=64/256/1024`, full compare `16 passed in 6.29s`.
  - Performance ratios: `1.106 / 1.364 / 2.164 / 2.194`.
- A103b deleted now-dead BF16 inverse GM helper code and removed `attnWsGm_`.
  - The legacy BF16 inverse workspace slot remains reserved intentionally; removing it is a separate host/tiling/workspace layout phase.
  - Gate PASS: isolated build/install, `S=64/256/1024`, full compare `16 passed in 6.52s`.
  - Performance ratios: `1.035 / 1.359 / 2.169 / 2.190`.
  - Accepted baseline:
    - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA103b_bf16_attn_gm_dead_cleanup_baseline.h`
    - SHA256 `08c14d7639b4d50d77dd9b5a9aee7e180e6af7a9d7c690566fa7f54a0dab9e62`.

## A104 Current Plan

- Next phase is user-directed full CC MTE2 preload, not direct resident final writeback.
- Implement as one isolated dataflow change:
  - strict full64 inverse entry preloads `INVInitHalfWsGm_` full `64x64` to `aicHalfL1Scratch_`;
  - clean cache over the full CC region for `INVInitHalfWsGm_`;
  - keep `L_powHalfWsGm_` as diagonal packed preload because off-diagonal `L_pow` is not consumed by P7/P8;
  - P7 reads diagonal `INVInit` from `aicHalfL1Scratch_` and writes result64 to `aicHalfL1P3Result_`;
  - P8 reads lower blocks from the same `aicHalfL1Scratch_`, so strict path no longer replays GM lower-block loads.
- Do not mix A104 with:
  - final `cInv` direct resident L1 writeback;
  - identity64 longer lifecycle;
  - RHS or `gBK/vBeta` changes.
- After A104 PASS, update the direct resident writeback item as the following phase.

## A104 Result and A104b Forward Fix

- A104 full `INVInit` CC preload is correctness-safe:
  - source/baseline SHA256 `144c53a63ed6c36eb6d21a58d851513f453602092547c5f397026b060df495f6`;
  - baseline `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA104_invinit_full_cc_l1_preload_baseline.h`;
  - build/install PASS;
  - `S=64/256/1024` PASS;
  - full compare PASS: `16 passed in 6.97s`.
- A104 performance is not accepted as main continuation:
  - ratios `1.078 / 1.457 / 2.168 / 2.212`;
  - compared with A103b `1.035 / 1.359 / 2.169 / 2.190`, only `S=2048` is effectively flat/slightly better and other sizes regress.
- Root hypothesis:
  - full `INVInit` preload removes P8 lower GM replay and improves MTE2 metrics;
  - per-line full CC cache clean adds scalar cost and offsets the gain.
- A104b next:
  - keep the same full `INVInit` L1 resident dataflow;
  - replace full-CC per-line clean with a single `CacheLine::ENTIRE_DATA_CACHE` DCCI before the DataCopy;
  - validate as a separate phase before deciding whether to accept A104b or restore A103b as performance surface.

## A105/A106 Accepted

- A105 removed inverse scratch DCCI around `L_powHalfWsGm_` / `INVInitHalfWsGm_`.
  - Gate PASS: build/install, `S=64/256/1024`, full compare `16 passed in 6.18s`.
  - Performance ratios: `1.016 / 1.308 / 2.076 / 2.103`.
  - Baseline SHA256 `36d2969ba9aa9cd3a0c61083c805a84331fd2588322e02d97786fd1591ea63a5`.
- A106 removed all remaining active explicit DCCI clean/flush wrappers/calls from p3l1_p8a Stage1.
  - Removed DCCI for RHS half scratch, BF16 output/workspace, old float/half helper paths, and deleted the helper definitions/cacheline constants.
  - Kept `SyncMte3ToMte2`, `SyncFixToMte2`, `SyncFixToS`, cross-core flags, and Fixpipe/MTE ordering.
  - Intermediate warning: deleting sync together with DCCI fails at `S=1024`; sync deletion is not part of the accepted optimization.
  - Gate PASS: build/install, `S=64/256/1024`, full compare `16 passed in 6.17s`.
  - Performance ratios: `0.892 / 0.864 / 0.926 / 0.928`.
  - Baseline:
    - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA106_drop_all_dcci_baseline.h`
    - SHA256 `a08e61279067653747e67dda533bbbb50e306e051d20acdcb6729df92aefa9d8`.
- A106 is now the latest correctness/performance continuation baseline.
- Next optimization should start from A106 and should not reintroduce explicit DCCI unless a new cross-core/cache visibility proof requires it.

## A116 and Cleanup Handoff

- A116 dead strict resident row-prefix cleanup passed build/install, `S=64/256/1024`, and full compare before cleanup.
- A116 performance ratios are `S=64/512/2048/10240 = 0.903 / 0.832 / 0.903 / 0.897`; it is neutral/slightly regressed versus A115 on long sequence, so treat it as cleanup, not a new performance win.
- Active Stage1 was renamed to `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1.h`.
- `op_kernel/baseline/` keeps only latest A116 baseline.
- Cleanup removed inactive Stage1 variant folders/files and pytest debug/cache/profile artifacts. Re-run build/install/full compare after cleanup before committing.
