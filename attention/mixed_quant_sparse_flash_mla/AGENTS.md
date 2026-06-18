# Repository Guidelines

## Project Structure & Module Organization

This directory contains the `mixed_quant_sparse_flash_mla` Ascend C operator implementation.

- `op_host/`: host-side operator definition, validation, consistency checks, and tiling logic.
- `op_kernel/`: device kernel sources and shared metadata. Architecture-specific code lives under `op_kernel/arch35/`.
- `tests/pytest/`: pytest-based functional, graph-mode, batch, and performance test harnesses.
- `tests/pytest/batch/`: Excel-to-PT batch case generation and PT replay helpers.
- `docs/`: local design and API learning notes for this operator.
- `README.md`: user-facing operator API, constraints, and parameter documentation.
- `/home/panjie/workspace/aichat/mixed_quant_sparse_flash_mla/`: required location for durable documentation-style outputs for this operator, except the progress files listed below. Do not add new investigation logs, design reports, or spreadsheets under a repository-local `aichat/` directory.

The source migration baseline was `/home/panjie/workspace/codes/ops-transformer_qsmla/attention/mixed_quant_sparse_flash_mla/AGENTS.md`, but this repository is newer. Prefer current files over assumptions from that older checkout.

## Build, Test, and Development Commands

- `cmake -S . -B build`: configure this operator directory when building it standalone.
- `cmake --build build`: build configured host/kernel targets when supported by the active CANN toolchain environment.
- `cd tests/pytest && bash test_run.sh single`: run one manually configured test case from `mixed_quant_sparse_flash_mla_paramset.py`.
- `cd tests/pytest && bash test_run.sh batch_save`: read Excel cases, compute golden data, and save `.pt` cases.
- `cd tests/pytest && bash test_run.sh batch_exec`: replay saved `.pt` cases on NPU.
- `cd tests/pytest && bash test_run.sh batch`: run `batch_save` and `batch_exec`; generated `.pt` files are cleaned by default.
- `cd tests/pytest && bash test_run.sh batch --keep-pt`: keep generated `.pt` files for debugging.

Batch commands accept `--excel <path>`, `--sheet <name>`, and `--pt-dir <path>`. Ensure the CANN, torch, and torch_npu environment is initialized before running NPU tests.

## Coding Style & Naming Conventions

Follow the existing C++ style: 4-space indentation, header guards or `#pragma once` consistent with nearby files, and concise helper functions grouped by validation, tiling, and kernel responsibility. Keep operator-specific symbols prefixed with `MixedQuantSparseFlashMla`, `QSMLA`, or `mixed_quant_sparse_flash_mla` as appropriate. Avoid broad refactors when changing one validation path, tiling branch, or arch-specific kernel path.

For kernel-side debug printing, use `printf`.

## Ownership Notes

Treat this directory as the small-team implementation of the `mixed_quant_sparse_flash_mla` operator. Code under sibling shared paths such as `attention/common/` is large-team common infrastructure, even when included by this operator through conditional includes. When analyzing behavior, distinguish operator-owned files (`op_host/`, `op_kernel/`, `tests/pytest/`, `docs/`) from shared common implementations such as `attention/common/op_kernel/matmul.h`.

## Current Repository Notes

This checkout differs from the older `ops-transformer_qsmla` baseline. Current code includes newer handling around:

- `ActualSeqLensParser` and query offset calculation in `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_cube.h`.
- `cmp_residual_kv` assignment and loop metadata in `op_kernel/arch35/mixed_quant_sparse_flash_mla_kvcache.h`.
- `ComputeParamBatch` call signatures in `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h`.
- Expanded pytest result fields in `tests/pytest/utils.py`, including `actlen_mode`, `S1EQS2`, `S2`, and `template_run_mode`.

When porting fixes from the older checkout, inspect these current files first instead of applying old patches mechanically.

## Session Continuity

At the start of a new conversation for this operator, read both progress files before making assumptions or continuing analysis:

- `learning_progress.md` (in current directory): accumulated knowledge points and confirmed technical understanding.
- `work_progress.md` (in current directory): generated documents, completed edits, and current follow-ups.

Keep both progress files in the current directory updated when new durable knowledge is confirmed or documentation/work status changes.

## User Workflow Preferences

- When designing test cases as an Excel table, use the `batch_save` input parameters as the reference schema.
- When designing test cases in `paramset` form, use the `single` mode input parameters as the reference schema.
- When changing parameters for any one of `single` mode, `batch_save`, `batch_exec`, or output result handling, check and update the other paths for consistency.
- When asked to write new rules into this agent file, first polish them into concise, clear, token-efficient instructions that are easy for the agent to follow.

## Sequence Length Terms

- `cu_seqlen_xx` describes the memory layout length in prefix-sum form. Each value is the accumulated S-dimension memory length of all previous batches.
- `usedlen_xx` and `seqused_xx` describe the S length that each batch actually participates in computation.
- Example: for `cu_seqlen_xx = [0, 3]` and `usedlen_xx = [2]`, memory occupation is calculated as `3`, while the compute length is calculated as `2`.
- TND padding means that, when `layout = TND`, the memory layout length inferred from Q or KV `cu_seqlen_xx` differs from the actual compute length specified by `usedlen_xx` or `seqused_xx`. This only appears in TND layout.

## Testing Guidelines

Pytest files follow `test_*.py` naming. Single-case parameters belong in `tests/pytest/mixed_quant_sparse_flash_mla_paramset.py`; golden CPU behavior belongs in `mixed_quant_sparse_flash_mla_golden.py`; comparison logic belongs in `result_compare_method.py`. Use markers defined in `pytest.ini` (`ci`, `graph`) for selected test subsets. Add or update batch cases when changes affect shape handling, layout support, quantization parameters, masks, KV cache behavior, TND padding, actual sequence lengths, or graph-mode execution.

## Commit & Pull Request Guidelines

Recent commits use short, imperative Chinese summaries, for example `新增QSMLA算子` or `还原sinks shapeSize检查逻辑...`. Keep commits focused on one behavior or cleanup. Pull requests should describe the changed host/kernel paths, list tested commands and hardware/toolchain assumptions, and call out any unsupported layouts, dtype limits, or compatibility risks.
