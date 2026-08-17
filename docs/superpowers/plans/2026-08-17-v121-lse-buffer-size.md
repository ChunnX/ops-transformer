# V121 LSE Buffer Size Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the arch-920r1 LSE temporary-buffer overflow while preserving the existing numerical path.

**Architecture:** Keep the padded LSE representation unchanged and size both task-parity buffers from the 64-row template and eight-float row stride. No kernel scheduling, online-softmax, or output-copy behavior changes.

**Tech Stack:** Ascend C++ kernel code, Python capacity diagnostic, repository build/static checks.

## Global Constraints

- Preserve current numerical precision and computation order.
- Do not change ProcessVec1, ProcessVec2, or pipeline scheduling.
- Size each LSE temporary buffer to 2048 bytes for the fixed 64-row V121 template.

---

### Task 1: Correct the LSE temporary-buffer capacity

**Files:**
- Modify: `experimental/attention/quant_block_sparse_attn/op_kernel/arch-920r1/quant_block_sparse_attn_block_vec.h`

**Interfaces:**
- Consumes: `s1BaseSize == 64U` and the existing padded LSE layout of eight floats per row.
- Produces: two `lseTmpBuff` allocations large enough for all 64 rows.

- [x] **Step 1: Run a failing capacity diagnostic**

  Read the two production allocations and assert that each is 2048B. Expected before the fix: failure reporting `[1024, 1024]`.

- [x] **Step 2: Implement the minimal allocation fix**

  Define `lseTmpBufferSize = s1BaseSize * 8U * sizeof(float)` in `InitLocalBuffer` and use it for both `lseTmpBuff` allocations.

- [x] **Step 3: Re-run the capacity diagnostic**

  Expected: both production allocations resolve to 2048B.

- [x] **Step 4: Run available static/build verification and review the diff**

  Confirm exit status, inspect `git diff --check`, and verify that no unrelated production code changed.
