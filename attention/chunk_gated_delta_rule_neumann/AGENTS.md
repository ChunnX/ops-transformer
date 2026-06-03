# AGENTS.md - chunk_gated_delta_rule_neumann

Chunk Gated Delta Rule Neumann attention operator for Ascend NPU. This file is a durable agent guide only: keep long experiment logs, daily records, and transient results in `docs/YYYY-MM-DD/`, not here.

## Communication

- 每次对话称呼用户为主人。
- 当输出过长时分段输出。
- 用户给出明确实现目标时，按目标正向推进；诊断路径只能用于定位，不能替代目标实现。
- 不要自行 `git push`，除非用户明确要求。
- 不测或暂不启用的 case 不删除，先注释。
- 你目前的性能小shape已经1.x但是大shape只有2.x这证明肯定有一些冗余操作导致在大shape堆积越来越多进而显现，看看有什么能够应该长周期复用的没有复用，应该搬运的却选择了计算加搬运，有没有多余写入写出计算。如果没有想法可以采用仿真分析或者msprof分析流水。你定位出来后必须正向解决
## Current Scope

Current focus is strict packed64 `C64_1616_FP16` merged into the official Stage1 with host `chunkSize=64`.

- Active Stage1 file:
  - `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`
- Current allowed modification area:
  - Stage1 inverse path only: FP16 `L/INV`, diagonal inverse, compose, inverse writeback.
- Do not modify:
  - Stage2, Stage3, QK, `gBK/vBeta` generation, final RHS.
- Final RHS semantics must remain unchanged:
  - active path uses `FusedRhsAICProcess()` with resident inverse from `aicHalfL1ResidentInv_`
  - RHS operands remain `gBKHalfWsGm_` / `vBetaHalfWsGm_`
  - do not change Stage2/Stage3/QK/`gBK/vBeta` generation or final RHS math while optimizing inverse/writeback

Tail/non-64 support is restored on the packed64 path. For tail-related or Stage1 inverse/resident changes, keep both tail and aligned gates active.

## Target Discipline

Do not downgrade the implementation target to get an easier pass.

- If the target is packed/resident/merged MMAD, do not replace it with GM fallback, single-block loops, 32+32 grouping, or conservative writeback unless the user explicitly asks for a temporary fallback.
- Packed MMAD itself is the target, not a redundancy. Treat redundant per-`16x16` movement, copy-out, layout-copy, or identity generation as optimization targets, but keep diagonal inverse and lower compose moving toward packed-MMAD execution.
- `identity64` reuse must be a real lifecycle design. A simple "first time generate, later reuse" flag is valid only after proving the L1 storage survives the raw-MMAD local buffer epoch and `pipe_->Reset()` boundaries; otherwise regenerate once per validated epoch.
- If target implementation fails, debug forward within the same target shape: load/layout/Fixpipe/cache/sync/math order and lifecycle. Precision failure is not a default rollback-complete state.
- Do not move to the next optimization while the current optimization is precision-failing. First identify the concrete cause and fix it on the target path, or explicitly document why that optimization is abandoned.
- Restoring a previous PASS baseline is allowed only to rebuild a clean working surface after abandoning a failing candidate, or when the user asks for recovery; it is not a substitute for solving the active task.
- Always distinguish:
  - `diagnostic path PASS`
  - `target path PASS`
- Only target path PASS means the optimization is complete.

Current target details:

- Diagonal inverse must be `64x64` packed block-diagonal.
  - Pack four `16x16` diagonal blocks into `(0,0)/(16,16)/(32,32)/(48,48)`.
  - Source blocks come from `aicHalfL1TileInv_` / `aicHalfL1TilePower_`.
  - L0 diagonal copy should correspond to `ref/ref.cpp::CopyDiagonalFractalsL1ToL0()`.
  - One packed MMAD direction should cover the four diagonal Neumann subproblems.
- Lower compose must follow lower-triangular math, not the reference upper-triangular MBH formula:
  - `tmp = lower @ topInv`
  - `out = bottomInv @ tmp`
  - Later odd/even packed copy should reference `ref/ref.cpp::CopyOddOrEvenBlocksL1ToL0()` only for packing layout, not for formula direction.

## Verification Gate

One optimization or dataflow change at a time.

When several same-shape `16x16` / packed fractals have a constant source and destination stride, treat them as a single `DataCopy` / `LoadData` / `Fixpipe` candidate instead of repeating per-block helpers. This includes copy-out/writeback paths, not just load paths. Record the candidate first, then modify it as its own phase with the full precision gate.

Required gate before moving to the next optimization:

1. Build package.
2. Install package.
3. Run targeted smoke for the changed path.
4. Run current compare set if smoke passes.

If precision fails, stop on that optimization and solve it forward on the target path. Do not stack the next optimization on a failing path. If a baseline restore is needed to recover a clean workspace, record the failed candidate, root-cause hypothesis, and next forward-fix entry before continuing. If a bounded optimization is proven blocked by a hardware/API boundary, it may be moved later in the queue with that boundary recorded; this is not a correctness pass.

After all planned phases pass, record the exact source hash and gate evidence in dated docs before searching for new optimization points. New candidates may come from `msprof` real profiling or simulator analysis, and each new candidate starts again as one gated phase.

Useful commands from repo root context:

```bash
bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64
CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare
```

## Key Files

- Active C64 FP16 Stage1:
  - `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`
- Reference code:
  - `ref/ref.cpp`
  - `ref/ref.md`
- Current plan / status:
  - `docs/2026-05-15/PACKED64_TAIL_PADDING_NEXT_STEP_PLAN.md`
  - `todo.md`
- Daily status should be recorded in:
  - `docs/YYYY-MM-DD/WORK_LOG.md`
  - `docs/YYYY-MM-DD/PROGRESS_AND_ISSUES.md`

## Stage1 Version Policy

- Active Stage1 implementation lives directly in `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`.
- Retired directories `op_kernel/baseline/`, `op_kernel/packed64local/`, and `op_kernel/selected/` should not be recreated for normal packed64 work.
- Historical phase evidence belongs in `docs/YYYY-MM-DD/` and operator `SKILL/` references, not in duplicate baseline headers.

未通过精度的实验文件不得保存为 baseline；未通过的 helper 可以保留为诊断代码，但不能被描述成 active 目标完成。

## Documentation Policy

- Keep this file short and durable.
- Do not append daily Debug History here.
- Record daily operations, failures, exact results, and temporary decisions in dated docs.
- When a new plan is requested, write it under `docs/YYYY-MM-DD/*_PLAN.md`.

## Skills

Operator-specific skills live under `SKILL/`.

Use `SKILL/precision-debugging/SKILL.md` first for precision failures, fulllocal/compose, resident/L1/TSCM, GM fallback, boundary-shape failures, or baseline restoration.

Other useful skill folders:

- `SKILL/kernel-debugging/`：运行环境、package 加载、hang/timeout、selector/同步问题。
- `SKILL/kernel-optimization-discipline/`：DataCopy/LoadData/Fixpipe/MMAD-copy、resident materialization、cache clean/DCCI、同步边界、tiny-MMAD 性能判断。
- `SKILL/performance-profiling-msprof/`：msprof 性能采集与结果汇总。
- `SKILL/workspace-tiling-checklist/`：workspace、tiling、offset、golden 口径一致性检查。
