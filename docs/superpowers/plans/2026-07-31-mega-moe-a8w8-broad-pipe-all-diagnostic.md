# MegaMoe A8W8 Broad PIPE_ALL Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add nine tagged `PipeBarrier<PIPE_ALL>();` diagnostics around A8W8 Dispatch and Combine boundaries so the original failure can be localized by grouped removal.

**Architecture:** Keep the experiment limited to existing AIV code paths and add five Dispatch barriers plus four Combine barriers. Each barrier receives a stable `SYNC_DIAG` identifier so the full set can be verified structurally and later removed by Dispatch/Combine subgroup without changing communication or Hcomm Drain behavior.

**Tech Stack:** Ascend C kernel C++, shell structural assertions, MegaMoe operator build, Git diff validation.

---

### Task 1: Add the Dispatch diagnostic group

**Files:**
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`

- [ ] **Step 1: Verify the five Dispatch tags are absent**

```bash
test "$(rg -c 'SYNC_DIAG\[D[1-5]\]' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h || true)" -eq 5
```

Expected: exit status `1`, because D1～D5 have not been added.

- [ ] **Step 2: Add D1 before relay READY publication**

In `QuantTokenToLocalRelay`, insert the barrier after the existing MTE3 completion event and before obtaining/writing
`readyFlag`:

```cpp
SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[D1]: relay payload -> READY publication
__gm__ uint64_t *readyFlag =
    reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr + RelayFlagOffset(serverId_, tokenIdx));
```

- [ ] **Step 3: Add D2 between slot metadata and READY**

In `DispatchComputeToken`, replace the commented diagnostic line with:

```cpp
WriteGmByPassDCache(metaPtr, static_cast<int32_t>(tokenIdx));
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[D2]: slot metadata -> READY publication
WriteGmByPassDCache(metaPtr + 1, int32_t(1));
```

- [ ] **Step 4: Add D3 before the dispatch round DONE flag**

In `DispatchComputeKernel`, insert:

```cpp
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[D3]: all slot publications -> round DONE
__gm__ int32_t *donePtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchDonePtr);
WriteGmByPassDCache(donePtr + computeIdx, roundTag);
```

- [ ] **Step 5: Add D4 between relay READY observation and MTE2 read**

In `LoadTokenFromLocalRelay`, insert after the polling loop:

```cpp
while (ReadGmByPassDCache(readyFlag) != uint64_t(1)) {
}
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[D4]: READY observation -> relay payload read

uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, tokenIndex);
```

- [ ] **Step 6: Add D5 before publishing dispatch progress to GMM1**

In `MetaInfoCalAndDispatch`, insert after `waveLo`/`waveHi` are calculated and before `flagBase` is used:

```cpp
int32_t waveLo = roundRowStartLocal / L1_TILE_M_I32;
int32_t waveHi = (roundRowEndLocal - 1) / L1_TILE_M_I32;
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[D5]: dispatched data/meta -> GMM1 progress
__gm__ int32_t *flagBase = gmmAddrInfo.dispatchToGmm1Flag;
```

- [ ] **Step 7: Verify the Dispatch group**

```bash
test "$(rg -c 'SYNC_DIAG\[D[1-5]\]' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h)" -eq 5
rg -n -C 2 'SYNC_DIAG\[D[1-5]\]' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h
```

Expected: exit status `0`; output contains D1, D2, D3, D4, and D5 exactly once.

### Task 2: Add the Combine diagnostic group

**Files:**
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`
- Modify: `mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h`

- [ ] **Step 1: Verify the four Combine tags are absent**

```bash
test "$(
  rg -c 'SYNC_DIAG\[C[1-4]\]' \
    mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
    mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h |
  awk -F: '{sum += $2} END {print sum + 0}'
)" -eq 4
```

Expected: exit status `1`, because C1～C4 have not been added.

- [ ] **Step 2: Add C1 after triple metadata loading**

In `ProcessCombineBatch`, insert after `DataCopy(metaInfoTensor, ...)`:

```cpp
AscendC::DataCopy(metaInfoTensor, tripleGm, batchCount * META_INFO_SIZE);
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[C1]: triple metadata load -> Combine consumption

MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t, true, IsQuant>(
```

- [ ] **Step 3: Add C4 before batch progress and UB reuse**

In the same function, insert after `CombineTokenGroup` returns:

```cpp
MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t, true, IsQuant>(
    currentRow, batchCount, k_, expertIdx, rankId_, gmmAddrInfo.gmm2OutGlobal, params_, metaInfoTensor,
    combineUbTensorSize_, offset, quantTokenSizeBytes);
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[C4]: batch send issue -> progress/UB reuse

rankInfoTensor.SetValue(2 * processRankNum + rankIndex, processedCnt + batchCount);
```

- [ ] **Step 4: Add C2 between quantization and send**

In `CombineTokenGroup` in `mega_moe_combine_send.h`, insert after the quantized path's existing `V_S` event and before
`SendType` is selected:

```cpp
if constexpr (IsQuantized) {
    SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID3>();
    AscendC::DataCopyPad(ubBf16, gmm2OutGm[(tokenStart + i) * n], gm2UbParams, copyPadParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID4>();
    QuantMxFp8<QuantMode, T>(ubQuantData, ubBf16, floatTemp, n);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID5>();
}
PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[C2]: quantization -> staging/send

using SendType = typename std::conditional<IsQuantized, Fp8Type, T>::type;
```

- [ ] **Step 5: Add C3 between staging completion and Hcomm submission**

In `CombineSendTokenToRemote`, insert inside the existing staging branch:

```cpp
if (IsQuantized || toRankId == rankId) {
    uint64_t dstOffset = toRankId == rankId ? dstBaseOffset : 0;
    AscendC::DataCopyPad(gmLocalD[dstOffset], ubQuant, singleCopyParams);
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID4>();
    PipeBarrier<PIPE_ALL>(); // SYNC_DIAG[C3]: staging write -> Hcomm source read
}
```

Do not add or move `hcomm_.Drain()` and do not modify the following `WriteNbi` call.

- [ ] **Step 6: Verify the Combine group**

```bash
test "$(
  rg -c 'SYNC_DIAG\[C[1-4]\]' \
    mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
    mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h |
  awk -F: '{sum += $2} END {print sum + 0}'
)" -eq 4
rg -n -C 2 'SYNC_DIAG\[C[1-4]\]' \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h
```

Expected: exit status `0`; output contains C1, C2, C3, and C4 exactly once.

### Task 3: Validate scope and build the operator kernel

**Files:**
- Verify: `mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h`
- Verify: `mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h`

- [ ] **Step 1: Assert the complete tag set**

```bash
test "$(
  rg -o 'SYNC_DIAG\[[DC][1-5]\]' \
    mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
    mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h |
  wc -l
)" -eq 9

for tag in D1 D2 D3 D4 D5 C1 C2 C3 C4; do
  test "$(
    rg -o "SYNC_DIAG\\[$tag\\]" \
      mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
      mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h |
    wc -l
  )" -eq 1
done
```

Expected: every command exits `0`; all nine tags occur exactly once.

- [ ] **Step 2: Inspect scope and whitespace**

```bash
git diff --check
git diff -- \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h
```

Expected: no whitespace errors; the diff contains only nine tagged `PipeBarrier<PIPE_ALL>();` additions and removes the
old commented D2 placeholder.

- [ ] **Step 3: Build the MegaMoe operator kernel**

```bash
bash build.sh --opkernel --soc=ascend950 --ops=mega_moe -j16
```

Expected: exit status `0` and the MegaMoe operator kernel build completes successfully.

- [ ] **Step 4: Commit the diagnostic implementation**

```bash
git add \
  mc2/mega_moe/op_kernel/arch35/mega_moe_layered.h \
  mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h
git commit -m "debug: add mega moe pipe all diagnostics"
```

Expected: one commit containing only the nine diagnostic barriers.
