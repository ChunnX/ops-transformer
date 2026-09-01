# Stem Indexer Output Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the arch-920r1 kernel initializes every `sparse_indices` padding element when the runtime launches fewer than the compile-time maximum of 64 logical Vector cores.

**Architecture:** Keep the existing parallel GM initialization algorithm and change only its partition count. The partition count must come from `StemIndexerTilingData::usedCoreNum`, which is populated from the runtime `blockDim`, so the number of partitions exactly matches the logical Vector cores that execute `CleanOutput()`.

**Tech Stack:** AscendC C++, Python source-contract tests, Git

**Spec:** `C:\Users\李东生\.codex\attachments\f723d9fb-0fbf-42be-8b3f-99501ad68a43\pasted-text.txt`

## Global Constraints

- Modify only the arch-920r1 output-initialization path and its regression test.
- Preserve the existing 512-byte output-initialization alignment.
- Do not change TopK selection, metadata scheduling, or arch35 behavior.
- Verification must work without an NPU or remote machine.

---

### Task 1: Reproduce the cleanup coverage gap locally

**Files:**
- Modify: `experimental/attention/stem_indexer/tests/pytest/test_stem_indexer_score_gm_contract.py`

**Interfaces:**
- Consumes: `SIVector<SIT>::InitOutputSingleCore` source and the three failure shapes from the supplied log.
- Produces: `test_output_cleanup_partitions_match_launched_vector_cores`, a regression contract for runtime core-count partitioning.

- [x] **Step 1: Add the failing regression test**

Add a test that extracts `InitOutputSingleCore`, rejects the compile-time `AIV_CORE_NUM` divisor, requires `constInfo_.usedCoreNum`, and arithmetically reproduces the two exact missing padding counts (7680 and 15360) when 54 launched cores execute 64 fixed partitions.

- [x] **Step 2: Run the test to verify RED**

Run the test module directly with the available Python standard library test-function runner.

Expected: the new test fails because `InitOutputSingleCore` currently contains `uint64_t aivCoreNum = AIV_CORE_NUM;`.

### Task 2: Use the launched logical core count for cleanup

**Files:**
- Modify: `experimental/attention/stem_indexer/op_kernel/arch-920r1/stem_indexer_service_vector.h`
- Test: `experimental/attention/stem_indexer/tests/pytest/test_stem_indexer_score_gm_contract.py`

**Interfaces:**
- Consumes: `SICommon::ConstInfo::usedCoreNum`, populated by `SIPreload::InitTilingData` from `StemIndexerTilingData::usedCoreNum`.
- Produces: complete, non-overlapping aligned output initialization over all launched logical Vector cores.

- [x] **Step 1: Implement the minimal fix**

Replace the compile-time core-count divisor in `InitOutputSingleCore` with `constInfo_.usedCoreNum`; retain the existing alignment, offset, bounds, and `InitOutput` calls unchanged.

- [x] **Step 2: Run the regression test to verify GREEN**

Run the standard-library test-function runner over `test_stem_indexer_score_gm_contract.py`.

Expected: all contract tests pass, including both observed missing-count calculations.

- [x] **Step 3: Run static and repository checks**

Run `git diff --check`, inspect the focused diff, and verify that `usedCoreNum` is set from `blockDim` in host tiling and propagated into `ConstInfo` before `SIVector::InitBuffers` invokes `CleanOutput`.

Expected: no whitespace errors; only the plan, regression test, and one kernel line change.
