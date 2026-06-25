# SMLA FD Work Progress

## 2026-06-25 Phase 1

Status: code written for phase 1. Operator execution and validation were not run in this environment.

### Implemented

- Reused the existing `sparse_flash_mla_metadata` FA/FD metadata ABI; no new metadata region or field was added.
- Added arch35/A5 FD staging workspace reservation in `op_host/sparse_flash_mla_tiling.cpp`.
- Kept the workspace layout aligned with `AttentionCommon::S2SplitFdStagingLayout`:

```text
[partial O slots][max slots][sum slots]
```

Workspace formula now reserved by SMLA host tiling on A5:

```text
maxSplits = 2
broadcastElems = 8
dAlign = headDimAlign_    // currently 512 after host shape checks

stagingM = gSize          // non split-G
stagingM = 128            // split-G, gSize > 64

logicalSlotCount = aicNum       // non split-G
logicalSlotCount = aicNum / 2   // split-G, matching current paired-AIC mapping

slotCount = logicalSlotCount * maxSplits
workspace += slotCount * (
    stagingM * dAlign * sizeof(float)
  + 2 * stagingM * 8 * sizeof(float)
)
```

### Metadata ABI Check

Reviewed `../sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp`:

- FA producer slot base is generated as `firstFdDataWorkspaceIdx`.
- FD task base is generated as `fdWorkspaceIdx = preFdDataNum`.
- FD task split count is generated as `fdS2SplitNum = curKvSplitPart`.
- FD row split for AIV is generated as `fdMStart/fdMNum` in `SplitFD()`.
- Kernel-side writer/reader contract remains:

```text
FA writer slot = firstFdDataWorkspaceIdx + s2SplitIdx
FD reader base = workspaceIdx
FD reader count = workspaceNum
```

### Not Implemented In Phase 1

- Kernel FD metadata parsing.
- S2 range clipping changes in kernel.
- Vec1 max/sum staging.
- Vec2 partial O staging.
- Global FA/FD phase barrier.
- FD reducer and final output copy.
- New runtime trace printing.

### Follow-Up Validation Items

- Verify metadata traces in the requested compact form:

```text
FA: (BN, M, S2Start, S2End, isS2Split, firstFdDataWorkspaceIdx, s2SplitIdx)
FD: (coreEnable, BN, M, workspaceIdx, workspaceNum, mStartIdx, mNum)
```

- Confirm every FD task maps to produced FA writer slots with no duplicate `(slot, M row)` writers.
- Confirm `workspaceNum <= 2` is enforced by metadata generation before enabling kernel Reduce. I did not find an explicit rejection in the reviewed metadata generator.
- Confirm the later kernel staging base offset matches the host append point exactly.
- Confirm final byte access stays within the host workspace bound for normal and split-G cases.

### Verification

Not run. The current environment cannot execute the operator, and validation was intentionally skipped per request.
