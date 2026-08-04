# MegaMoe A8W8 Round 2 PIPE_ALL Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three second-round tagged `PipeBarrier<PIPE_ALL>();` diagnostics while preserving all first-round diagnostics.

**Architecture:** Add Hcomm publication/acquire barriers in `mega_moe_layered.h` and one GMM1 progress-publication barrier in the MX quant epilogue. Keep communication, Drain, AIC, control flow, and existing diagnostic points unchanged.

**Tech Stack:** Ascend C kernel C++, shell structural assertions, Git diff validation.

---

### Task 1: Add and verify R2-H1, R2-H2, and R2-G1

**Files:**
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`
- Modify: `mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h`

- [ ] **Step 1: Verify the second-round tags are absent**

```bash
test "$(
  rg -o 'SYNC_DIAG\[R2-[A-Z][0-9]\]' \
    mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
    mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h |
  wc -l
)" -eq 3
```

Expected: exit status `1`, because the second-round tags have not been added.

- [ ] **Step 2: Add R2-H2 before the Dispatch Hcomm send**

In `DispatchToken`, insert immediately before `WriteWithNotifyNbi`:

```cpp
GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(peerRank, dispatchFlagWinOffset_) +
                         RelayFlagOffset(serverId_, static_cast<uint32_t>(tokenIdx));
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[R2-H2]: READY acquire -> relay Hcomm send
hcomm_.WriteWithNotifyNbi(channel, dstAddr, srcAddr, mxQuantTokenScaleAlignBytes_, remoteFlagAddr, uint64_t(1));
```

- [ ] **Step 3: Add R2-H1 before CrossRankSync uses syncCount as a DMA source**

In `CrossRankSyncInWorldSize`, insert after updating `syncCount` and before the rank loop:

```cpp
int count = ReadGmByPassDCache(syncCount) + 1;
WriteGmByPassDCache(syncCount, count);
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[R2-H1]: syncCount publication -> Hcomm source read
for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
```

- [ ] **Step 4: Add R2-G1 after the GMM1 progress publication**

In `BlockEpilogueSwigluMxQuant::operator()`, insert after the optional AtomicAdd block:

```cpp
if (groupFlagListGmAddr_ != nullptr) {
    AscendC::AtomicAdd(groupFlagListGmAddr_, 1);
}
AscendC::PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[R2-G1]: GMM1 progress publication -> NotifyCube
```

- [ ] **Step 5: Verify all diagnostic tags**

```bash
for tag in R2-H1 R2-H2 R2-G1; do
  test "$(
    rg -o "SYNC_DIAG\\[$tag\\]" \
      mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
      mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h |
    wc -l
  )" -eq 1
done

test "$(
  rg -o 'SYNC_DIAG\[[DC][1-5]\]' \
    mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
    mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h |
  wc -l
)" -eq 9
```

Expected: exit status `0`; the three round-two tags and all nine first-round tags occur exactly once.

- [ ] **Step 6: Validate scope and whitespace**

```bash
git diff --check
git diff -- \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
  mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h
```

Expected: no whitespace errors; the source diff contains exactly three tagged barrier additions and no Drain,
communication, flag, counter, or control-flow changes.

- [ ] **Step 7: Commit the second-round diagnostics**

```bash
git add \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
  mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h
git commit -m "debug: add round2 mega moe pipe all diagnostics"
```

Expected: one source commit containing three insertions in two files. Do not run a local build, per user request.
