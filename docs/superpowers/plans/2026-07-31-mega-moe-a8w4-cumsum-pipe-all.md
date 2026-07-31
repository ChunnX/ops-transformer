# MegaMoe A8W4 Cumsum PIPE_ALL Diagnostic Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a diagnostic full-pipeline barrier that verifies whether the A8W4 cumsum corruption is caused by an MTE2-to-MTE3 dependency hazard.

**Architecture:** Keep the change inside `SendCntCal` and only on the `ENABLE_A8W4 && localExpertId != 0` path that restores a cumsum prefix through MTE2. Insert `PipeBarrier<PIPE_ALL>()` immediately before the MTE3 writeback so both the restored prefix and scalar-updated suffix are complete before the whole cumsum range is consumed.

**Tech Stack:** Ascend C kernel C++, shell structural assertions, Git diff validation.

---

### Task 1: Add the diagnostic synchronization

**Files:**
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h:1150-1155`
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered_sync_audit.md`

- [x] **Step 1: Run the failing structural assertion**

```bash
sed -n '/::SendCntCal(/,/DataCopyPad(cumsumInfoGlobalTensor_/p' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h |
  grep -q 'PipeBarrier<PIPE_ALL>'
```

Expected: exit status `1`, because the MTE3 cumsum writeback is not preceded by a full-pipeline barrier.

- [x] **Step 2: Add the minimal diagnostic implementation**

Add the barrier only when the earlier prefix restore was issued:

```cpp
if constexpr (ENABLE_A8W4) {
    // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，更新后写回 GM
    if (localExpertId != 0) {
        // 诊断性全流水同步：验证 GM 恢复 prefix 的 MTE2 与写回 MTE3 之间的依赖。
        PipeBarrier<PIPE_ALL>();
    }
    DataCopyPad(cumsumInfoGlobalTensor_, cumsumInfoTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
}
```

- [x] **Step 3: Update the synchronization audit**

Mark the A8W4 cumsum item as diagnostically fixed by `PIPE_ALL`, reduce the current missing-sync count by one, and state that a successful functional verification should be followed by replacement with a directional `MTE2_MTE3` event.

- [x] **Step 4: Run the structural assertion again**

```bash
sed -n '/::SendCntCal(/,/DataCopyPad(cumsumInfoGlobalTensor_/p' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h |
  grep -q 'PipeBarrier<PIPE_ALL>'
```

Expected: exit status `0`.

- [x] **Step 5: Validate formatting and inspect the exact diff**

```bash
git diff --check
git diff -- \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered_sync_audit.md
```

Expected: `git diff --check` exits `0`; the kernel diff contains only the guarded diagnostic barrier and the report reflects that status.
