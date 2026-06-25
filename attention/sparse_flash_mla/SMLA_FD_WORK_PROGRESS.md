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

### Verification (Code Review, 2026-06-25)

All five Phase 1 validation items were verified by code review. Findings below.

---

#### 验证项1: FA/FD metadata 字段含义

**结论：通过。** FA/FD metadata 字段含义已明确，trace 格式可行。

FA metadata（9 fields/core, `faMetadata[36][9]`）:

| Index | 常量 | 含义 |
|-------|------|------|
| 0 | FA_CORE_ENABLE_INDEX | AIC core 是否启用 |
| 1 | FA_BN2_START_INDEX | BN2 起始索引 |
| 2 | FA_M_START_INDEX | M(S1G row) 起始索引 |
| 3 | FA_S2_START_INDEX | S2 block 起始索引 |
| 4 | FA_BN2_END_INDEX | BN2 结束索引（不含） |
| 5 | FA_M_END_INDEX | M(S1G row) 结束索引（不含） |
| 6 | FA_S2_END_INDEX | S2 block 结束索引（不含） |
| 7 | FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX | 该 core 第一个 FD workspace slot |
| 8 | FA_S2_MAX_NUM | 单核最大 S2 基本块数（全局 max） |

FD metadata（8 fields/core, `fdMetadata[72][8]`）:

| Index | 常量 | 含义 |
|-------|------|------|
| 0 | FD_CORE_ENABLE_INDEX | AIV core 是否启用 |
| 1 | FD_BN2_IDX_INDEX | FD 任务的 BN2 索引 |
| 2 | FD_M_IDX_INDEX | FD 任务的 S1G row 索引 |
| 3 | FD_WORKSPACE_IDX_INDEX | FD 任务的 workspace 起始 slot |
| 4 | FD_WORKSPACE_NUM_INDEX | FD 任务的 S2 split 数（= fdS2SplitNum） |
| 5 | FD_M_START_INDEX | FD 子任务的 M 行起点 |
| 6 | FD_M_NUM_INDEX | FD 子任务的 M 行数 |
| 7 | （预留） | 未使用 |

`isS2Split` 和 `s2SplitIdx` 不在 metadata 中显式存储，kernel 需从 FA metadata 推导：
- `isS2Split`：当前 S1G row 的 S2 范围不是完整行（即 S2Start > 该行起始 或 S2End < 该行结束）
- `s2SplitIdx`：当前 core 在同一 FD group 中的 split 序号，推导方式为 `workspaceIdx - firstFdDataWorkspaceIdx`（需跨行时累加）

---

#### 验证项2: 每个 FD task 能找到对应 FA writer，无重复 (slot, M row)

**结论：通过（有注意事项）。**

核心推导：在 `RecordFDInfo` 中，`fdWorkspaceIdx = preFdDataNum`，`fdS2SplitNum = curKvSplitPart`。FD reader 读取 slots `[workspaceIdx, workspaceIdx + workspaceNum)`。FA writer 写入 slot `firstFdDataWorkspaceIdx[i]`。

追溯 `AssignBlocksToCore` 中 `firstFdDataWorkspaceIdx` 的生成：

```cpp
result.firstFdDataWorkspaceIdx[coreIdx] = preFdDataNum + curKvSplitPart - 1U;
```

对于 k-way split 的 S1G row（k 个 core 共享同一行），core 0..k-1 的 `firstFdDataWorkspaceIdx` 恰好覆盖 slots `[preFdDataNum, preFdDataNum + k - 1]`，与 FD reader 的 `[fdWorkspaceIdx, fdWorkspaceIdx + fdS2SplitNum)` 完全一致。

唯一 writer 保证：每个 core 的 `firstFdDataWorkspaceIdx` 唯一（`preFdDataNum` 单调递增，`curKvSplitPart` 在同一 FD group 内每个 core 递增 1），且 M offset 由 gSize/head 分配唯一确定。

**注意事项**：当一个 core 处理多个 S1G row 且跨越多个 FD group 时（例如 core 1 完成 Row A 的尾部后继续处理 Row B 的首部），`firstFdDataWorkspaceIdx[1]` 仅指向第一个 FD group 的 slot。Kernel 需在 S2 loop 中跟踪 `s2SplitIdx`，跨越 FD group 边界时更新 workspace slot。此逻辑属于阶段2 kernel 适配范畴，不影响阶段1 host/metadata 一致性。

---

#### 验证项3: workspaceNum <= 2 是否被 metadata 生成端显式限制

**结论：未限制。这是一个必须在启用 kernel FD 前修复的问题。**

`curKvSplitPart` 在 `AssignBlocksToCore` 末尾递增，无上限检查：

```cpp
if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
    assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
    assignContext.curKvSplitPart++;
}
```

`RecordFDInfo` 中也无校验：

```cpp
result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
```

`ParamsCheck()` 无相关限制。`result.maxS2SplitNum` 仅记录最大值，不做拒绝。

**影响**：`workspaceNum > FD_MAX_S2_SPLIT_NUM(2)` 时，FD reader 会读取超过 2 个 slot。虽然 workspace 总量仍足够（见验证项5），但 common `Reduce` 的 comment 明确写 `workspaceNum must not exceed maxSplits (currently 2)`，且适配方案要求 `workspaceNum <= 2`。若不限制，3+ split 场景未经测试，且 UB 生命周期和 chunk 逻辑可能有隐含假设。

**修复建议**（阶段1 或阶段2 前必须完成）：

1. 在 `sparse_flash_mla_metadata` 的 `ParamsCheck()` 或 `AssignBlocksToCore()` 中添加：当 `curKvSplitPart > FD_MAX_S2_SPLIT_NUM` 时，拒绝该参数组合或调整分配策略使单行最多被 2 个 core 分割。
2. 或在 `RecordFDInfo` 中断言 `curKvSplitPart <= 2`。
3. Kernel 侧也应有防御性检查。

---

#### 验证项4: kernel staging base offset 与 host append point 一致性

**结论：host 端 workspace 附加点正确；kernel 端尚未实现 FD staging base 计算，存在待补齐的 gap。**

Host 端 `DoOpTiling` workspace 布局（A5/arch35）：

```text
offset 0:    LibApiWorkSpaceSize
offset L:    v0ResGm region (条件: split-G || SCFA)
             = 128 * 512 * 2 * 3 * aicNum       (non split-G)
             = 128 * 512 * 2 * 3 * (aicNum / 2) (split-G)
offset L+V:  s2RealBuf (条件: split-G || SCFA)
             = 3 * 128 * sizeof(int32_t) * aivNum
offset L+V+S: FD staging region (CalcFdStagingWorkspaceSize)
```

Kernel 端 `InitMMResBuf` 当前仅计算了 v0ResGm 的 per-core offset：

```cpp
v0ResTotalOffset = v0ResSize * 3 * (IS_SPLIT_G ? (aicIdx >> 1U) : aicIdx);
v0ResGmBufferManager.Init(workspace + v0ResTotalOffset);
```

**gap**：Kernel 无 `fdStagingBase` 变量，tiling data 中无 FD staging base/size 字段。后续阶段需：

1. 在 kernel 中计算 `fdStagingBase = workspace + v0ResTotalRegionSize + s2RealBufSize`，其中：
   - `v0ResTotalRegionSize = 128 * 512 * sizeof(Q_T) * 3 * (IS_SPLIT_G ? (GetBlockNum() >> 1) : GetBlockNum())`
   - `s2RealBufSize = 3 * 128 * sizeof(int32_t) * aivNum`（条件同上）
2. 或将 FD staging base offset 加入 tiling data（更健壮，推荐）。

**v0ResGm 条件一致性确认**：

- Host: `if (gSize > FD_SPLIT_G_THRESHOLD || perfMode == SCFA_TEMPLATE_MODE)`
- Kernel: `if constexpr (IS_SPLIT_G || TEMPLATE_MODE == SMLATemplateMode::SCFA_TEMPLATE_MODE)`

条件完全匹配。✓

---

#### 验证项5: workspace 字节数覆盖 FD staging 最后访问上界

**结论：通过。** 在 `workspaceNum <= 2` 的前提下，workspace 始终足够。

推导：

1. `preFdDataNum`（最终值）= 所有 FD group 的 `curKvSplitPart` 之和 = 所有 FD group 的 split 总数。
2. 每个 FD group 至少占用 1 个 core boundary（`curKvSplitPart` 从 1 开始，每个 boundary +1），因此 `numOfFdHead <= total_boundaries <= aicNum - 1`。
3. 若 `workspaceNum <= 2`（即每个 group 恰好 2 split），则 `preFdDataNum = 2 * numOfFdHead <= 2 * (aicNum - 1) = 2*aicNum - 2`。
4. `slotCount = logicalSlotCount * maxSplits`：
   - Non split-G: `aicNum * 2 = 2*aicNum`
   - Split-G: `(aicNum/2) * 2 = aicNum`，此时 `aicNum` 指的是 halved 后的值，实际 `preFdDataNum <= 2*(aicNum/2 - 1) = aicNum - 2 < aicNum = slotCount`。
5. 因此 `preFdDataNum < slotCount`，所有 slot 访问在分配范围内。✓

FD reader 的 M 行访问：

- `fdMOffset + fdMNum <= fdMSize <= mBaseSize = gSize`（A5 上 mBaseSize = groupSize）
- Non split-G: `stagingM = gSize`，所以 `fdMOffset + fdMNum <= stagingM` ✓
- Split-G: `stagingM = 128`，`gSize <= 128`（split-G 触发条件 gSize > 64，且 numHeadsQ=128 → gSize=numHeadsQ/numHeadsKv=128），所以 `fdMOffset + fdMNum <= gSize <= 128 = stagingM` ✓

**若 `workspaceNum > 2`**：workspace 总量仍可能足够（`preFdDataNum <= 2*(aicNum-1)` 在多数情况下仍 < `slotCount`），但 3+ split 未被设计覆盖，不应依赖。

---

### 阶段1验证总结

| 验证项 | 结论 | 是否阻塞 |
|--------|------|----------|
| 1. FA/FD 字段含义 | 通过 | 否 |
| 2. FD task → FA writer 一致性 | 通过（注意事项：跨 group core 需跟踪 s2SplitIdx） | 否 |
| 3. workspaceNum <= 2 限制 | **未限制，需修复** | **是** |
| 4. Kernel staging base offset | Host 正确，Kernel 未实现（gap 待补齐） | 否（属阶段2+） |
| 5. Workspace 字节上界 | 通过（前提：workspaceNum <= 2） | 否 |

**阻塞项**：验证项3。在 `sparse_flash_mla_metadata` 中添加 `curKvSplitPart > 2` 的限制或拒绝逻辑后，阶段1方可通过。

---

## 2026-06-25 Phase 2

Status: code written for phase 2. Not yet compiled or run.

### Implemented

#### 1. `util_regbase.h` — New struct and fields

- Added `SmlaFdRunInfo` struct (analogous to QSMLA's `FdRunInfo`, with `uint32_t` types for SMLA):
  ```cpp
  struct SmlaFdRunInfo {
      bool coreEnable = false;
      uint32_t bn2Idx, mIdx, workspaceIdx, workspaceNum, mStartIdx, mNum;
  };
  ```

- Added to `RunParamStr`:
  - `firstFdDataWorkspaceIdx = 0` (from FA metadata index 7)
  - `isS2Split = false`
  - `s2SplitIdx = 0`

- Added to `COMMON_RUN_INFO` (→ `RunInfo`):
  - `isS2Split = false`
  - `s2SplitIdx = 0`

- Added to `RunInfo` struct body:
  - `firstFdDataWorkspaceIdx = 0`

#### 2. `sparse_flash_mla_scfa_kernel.h` — FD metadata parse, S2 clipping, split tracking

**ParseFdRunInfo**: Reads FD metadata from GM using `GetAttrAbsIndex(aivIdx, *, true)`. Only AIV cores parse. Returns immediately if `coreEnable == false`. Fields: `coreEnable`, `bn2Idx`, `mIdx`, `workspaceIdx`, `workspaceNum`, `mStartIdx`, `mNum`.

**ApplyS2MetadataRange**: Clips the S2 token-level range computed by `ComputeS2LoopInfo()` to the metadata range `[s2StartPoint, s2EndPoint)` (in token units, converted from block units via `* s2BaseSize`).

Logic (same as QSMLA reference):
1. Compute `totalLen = oriLen + cmpLen` (full S2 token range from mask/window)
2. For first tile (`isFirstS2RangeTask`): `rangeStart = s2StartPoint`; otherwise `rangeStart = 0`
3. For last tile (`isLastS2RangeTask`): `rangeEnd = effectiveS2EndPoint` (handles terminal sentinel `s2EndPoint == 0` → `totalLen`); otherwise `rangeEnd = totalLen`
4. Clip `rangeStart`/`rangeEnd` to `[0, totalLen]`
5. If `rangeEnd <= rangeStart`: no work → `isS2Split = false`, return true
6. `isS2Split = (rangeStart > 0) || (rangeEnd < totalLen)` — has prev/next core sharing this S1G row
7. Distribute clipped range to ori/cmp line ranges
8. Recompute `oriKvLoopEndIdx`/`cmpKvLoopEndIdx`/`s2LoopEndIdx`

**ProcessMainLoop changes**:
1. `ParseFdRunInfo(fdRunInfo)` called at top (AIV only); `(void)fdRunInfo` suppresses unused warning (FD phase not yet connected)
2. Read `FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX` from metadata → `firstFdDataWorkspaceIdx`
3. Fix `bN2EndIdx++` condition: `if (nextGs1Idx != 0 || s2EndIdx != 0)` (matches QSMLA; handles terminal sentinel case)
4. Track `s2SplitIdxCounter` (initialized to 0, increments when `isS2Split` is true)
5. After `ComputeS2LoopInfo`, if `!s2NoNeedCalc`: call `ApplyS2MetadataRange`; else: `isS2Split = false`
6. `runParam.s2SplitIdx = s2SplitIdxCounter` before the counter increment
7. `SetRunInfo` now copies `firstFdDataWorkspaceIdx`/`isS2Split`/`s2SplitIdx` to `RunInfo`

**No-work core handling**:
- `SmlaFdRunInfo fdRunInfo` is default-initialized (`coreEnable = false`, all fields 0)
- When `hasLoad == 0`, the function returns before entering the FA loop; `fdRunInfo` has no stale state
- `runParam.isS2Split = false` is set explicitly when `s2NoNeedCalc = true` (no stale value from previous tile)

### Not Implemented In Phase 2

- Vec1 max/sum staging (`StageVec1Lse`).
- Vec2 partial O staging (`StageVec2PartialO`).
- Global FA/FD phase barrier (`SyncAll`).
- FD reducer (`AttentionCommon::Reduce`) and final output copy.
- SWA kernel FD adaptation (not in scope for FD; SWA has short sequences unlikely to need S2 split).
- FD staging base offset computation in kernel (needed in Phase 3+).
- `S2SplitFdStagingLayout` construction in kernel (needed in Phase 3+).

### Follow-Up Validation Items

- Verify non-split use cases produce identical results (no functional regression).
- Verify S2 split trace: `[S2Start, S2End)` from metadata matches kernel's clipped range.
- Verify terminal `S2_END == 0` is correctly handled (only in last task).
- Verify empty Q / empty KV / no-work core: `isS2Split` is `false`, `s2SplitIdx` is 0.
- Verify FD metadata parse: `fdRunInfo` fields match `GenMetadata()` output.
- Verify `bN2EndIdx++` with `s2EndIdx != 0` handles terminal sentinel correctly.
