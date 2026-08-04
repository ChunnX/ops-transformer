# MegaMoe A8W8 Reset Tensor PIPE_ALL Diagnostic Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the A8W8 failure is caused by `resetTensor_` being overwritten by PIPE_V before an earlier PIPE_MTE3 read completes.

**Architecture:** Remove the A8W4-only cumsum diagnostic barrier because it is compile-time unreachable for A8W8. Add one diagnostic `PipeBarrier<PIPE_ALL>()` immediately before `ResetGmm2CombineSyncCounters()` overwrites `resetTensor_`, leaving every other synchronization point unchanged.

**Tech Stack:** Ascend C kernel C++, shell structural assertions, Git diff validation.

---

### Task 1: Move the diagnostic barrier to the A8W8-reachable dependency

**Files:**
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered_sync_audit.md`

- [x] **Step 1: Verify the new structural requirement fails before implementation**

```bash
sed -n '/::ResetGmm2CombineSyncCounters()/,/Duplicate(resetTensor_/p' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h |
  grep -q 'PipeBarrier<PIPE_ALL>'
```

Expected: exit status `1`, because the overwrite currently has no preceding full-pipeline barrier.

- [x] **Step 2: Apply the single-variable diagnostic change**

Remove the cumsum-only diagnostic block:

```cpp
if (localExpertId != 0) {
    PipeBarrier<PIPE_ALL>();
}
```

Add this immediately before the reset-tensor overwrite:

```cpp
if (coreLen > 0) {
    // 诊断性全流水同步：验证前序 MTE3 读 resetTensor_ 与本次 Vector 覆写之间的依赖。
    PipeBarrier<PIPE_ALL>();
    Duplicate(resetTensor_, 0, coreLen);
```

- [x] **Step 3: Update the audit state**

Move the A8W4 cumsum dependency back to the missing-sync list and mark the A8W8-reachable reset dependency as diagnostically covered. State that a successful A8W8 reproduction should replace the new full barrier with `MTE3_V`.

- [x] **Step 4: Verify the structural requirement passes**

```bash
sed -n '/::ResetGmm2CombineSyncCounters()/,/Duplicate(resetTensor_/p' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h |
  grep -q 'PipeBarrier<PIPE_ALL>'
```

Expected: exit status `0`.

- [x] **Step 5: Verify scope and formatting**

```bash
git diff --check
git diff -- mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h
```

Expected: no whitespace errors; the kernel diff shows the diagnostic barrier moved from the A8W4 cumsum branch to the reset-tensor overwrite.
