# Flash Attention Scalar Bound 优化 - 逐步修改总结

> 创建时间：2026-06-02
> 目的：逐步验证每个优化步骤的效果，定位劣化项

---

## Step 1: 缓存循环计算结果

**Patch 文件**: `step1_cached_loop_computation.patch`

**修改文件**: `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改内容**:
1. 新增 4 个缓存成员变量（第 118-121 行）：
   - `uint64_t cachedS2LoopTimes`
   - `uint64_t cachedGS1LoopTimes`
   - `int64_t cachedPreTokensLeftUp`
   - `int64_t cachedNextTokensLeftUp`

2. `GetTaskDealMode` 函数（第 474-480 行）：
   - 将 `s2LoopTimes`、`gS1LoopTimes`、`GetPreNextTokenLeftUp` 计算移入 batch 切换分支
   - 仅在 `isFirstTask || prevBIdx != bIdx` 时计算一次

3. `CalcCurS2StartEndWithSparse` 函数（第 548-549 行）：
   - 直接使用缓存的 `cachedPreTokensLeftUp` 和 `cachedNextTokensLeftUp`

4. `CalcParams` 函数（第 707-708 行）：
   - 直接使用缓存的 token 值，不再调用 `GetPreNextTokenLeftUp`

5. `UpdateAxisInfo` 函数（第 749-762 行）：
   - 直接使用 `cachedS2LoopTimes` 和 `cachedGS1LoopTimes`

**预期收益**: 每轮迭代消除 2 次 64-bit 除法 + 1 次函数调用

**风险等级**: 低（纯算法等价变换）

---

## Step 2: CalcParams bIdx/n2Idx 优化

**Patch 文件**: `step2_calcparams_bidx_n2idx.patch`

**修改文件**: `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改内容**:
1. `CalcParams` 函数（第 669-670 行）：
   - `info.bIdx = bN2Cur / constInfo.n2Size` → `info.bIdx = prevBIdx`
   - `info.n2Idx = bN2Cur % constInfo.n2Size` → `info.n2Idx = bN2Cur - prevBIdx * constInfo.n2Size`

**预期收益**: 消除 1 次除法 + 1 次取模运算

**风险等级**: 低（纯算术等价变换）

---

## Step 3: 分支预测提示

**Patch 文件**: `step3_branch_prediction_hints.patch`

**修改文件**: `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改内容**:
1. `FlashAttention` 主循环（第 419-432 行）：
   - `if (shouldDispatchTask)` → `if (likely(shouldDispatchTask))`
   - `if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK)` → `if (likely(taskDealMode == TASK_DEAL_MODE::CREATE_TASK))`
   - `else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO)` → `else if (unlikely(taskDealMode == TASK_DEAL_MODE::DEAL_ZERO))`

**预期收益**: 改善分支预测命中率，减少流水线冲刷

**风险等级**: 极低（仅添加编译器 hint，不改变逻辑）

---

## Step 4: CalcParams 局部变量提升

**Patch 文件**: `step4_calcparams_local_variables.patch`

**修改文件**: `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改内容**:
1. `CalcParams` 函数入口（第 667-670 行）：
   - 添加局部变量：`localN2Size`、`localGSize`、`localIsKvContinuous`、`localSubBlockIdx`
   - 函数内所有对应 `constInfo.xxx` 访问替换为局部变量

2. `CalcCurS2StartEndWithSparse` 函数入口（第 548 行）：
   - 添加局部变量：`localGSize`
   - 函数内所有 `constInfo.gSize` 替换为 `localGSize`

**预期收益**: 减少成员变量访问开销，提升寄存器利用率

**风险等级**: 低（纯局部变量替换）

---

## Step 5: Vec Block 热路径局部变量

**Patch 文件**: `step5_vec_block_local_variables.patch`

**修改文件**: `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改内容**:
1. `ProcessVec1Dn` 函数：
   - 添加 `localScaleValue`、`localSubBlockIdx`
   - 替换 5 处 `constInfo` 访问

2. `ProcessVec1Nd` 函数：
   - 添加 `localScaleValue`、`localSubBlockIdx`
   - 替换 9 处 `constInfo` 访问

3. `ProcessVec2DSplit` 函数：
   - 添加 `localDSizeV`、`localSubBlockIdx`、`localDBasicBlock`
   - 替换 16 处 `constInfo` 访问

4. `ProcessVec2_C2GmUpdateUb` 函数：
   - 添加 `localDSizeV`、`localSubBlockIdx`
   - 替换 8 处 `constInfo` 访问

**预期收益**: 减少 ~38 处 constInfo 成员变量访问

**风险等级**: 低（纯局部变量替换）

---

## Step 6: Cube Block 热路径局部变量

**Patch 文件**: `step6_cube_block_local_variables.patch`

**修改文件**: `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h`

**修改内容**:
1. `IterateBmm1NdL0Split` 函数：
   - 添加 `localDSize`
   - 替换 1 处 `constInfo.dSize`

2. `IterateBmm1NdL1SplitK` 函数：
   - 添加 `localDSize`
   - 替换 2 处 `constInfo.dSize`

3. `IterateBmm1DnSplitK` 函数：
   - 添加 `localDSize`
   - 替换 1 处 `constInfo.dSize`

4. `IterateBmm1Dn` 函数：
   - 添加 `localDSize`
   - 替换 1 处 `constInfo.dSize`

5. `IterateBmm2L1SplitN` 函数：
   - 添加 `localDSizeV`
   - 替换 2 处 `constInfo.dSizeV`

6. `IterateBmm2l0Split` 函数：
   - 添加 `localDSizeV`
   - 替换 2 处 `constInfo.dSizeV`

**预期收益**: 减少 ~15 处 constInfo 成员变量访问

**风险等级**: 低（纯局部变量替换）

---

## Step 7: Softmax Buffer 位掩码索引

**Patch 文件**: `step7_softmax_buffer_bitmask.patch`

**修改文件**: `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改内容**:
1. 添加常量定义（第 173 行）：
   - `static constexpr uint32_t SOFTMAX_BUF_DEPTH = 4`

2. Buffer 声明修改（第 176-178 行）：
   - `TBuf<> softmaxMaxBuf[PRELOAD_N + 1]` → `TBuf<> softmaxMaxBuf[SOFTMAX_BUF_DEPTH]`
   - `TBuf<> softmaxSumBuf[PRELOAD_N + 1]` → `TBuf<> softmaxSumBuf[SOFTMAX_BUF_DEPTH]`
   - `TBuf<> softmaxExpBuf[PRELOAD_N + 1]` → `TBuf<> softmaxExpBuf[SOFTMAX_BUF_DEPTH]`

3. 索引操作替换（16 处）：
   - `% (PRELOAD_N + 1)` → `& (SOFTMAX_BUF_DEPTH - 1)`

4. `SoftmaxInitBuffer` 函数（第 1095-1107 行）：
   - 添加第 4 个 buffer 的初始化

**预期收益**: 消除 16 处取模运算，每次节省 ~5-10 cycle

**风险等级**: 中（多占 768B UB 空间，需确认 UB 预算充足）

---

## Step 8: tilingData 指针引用优化

**Patch 文件**: `step8_tilingdata_pointer_reference.patch`

**修改文件**: `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改内容**:
1. `InitConstInfo` 函数（第 310-352 行）：
   - 添加局部指针变量：
     ```cpp
     const __gm__ FlashAttnBaseParams *baseParams = &(this->tilingData->flashAttnBaseParams);
     const __gm__ FlashAttnAttenMaskParams *maskParams = &(this->tilingData->flashAttnAttenMaskParams);
     const __gm__ FlashAttnWorkspaceParams *wsParams = &(this->tilingData->flashAttnWorkspaceParams);
     ```
   - 所有 `this->tilingData->flashAttnBaseParams.xxx` 替换为 `baseParams->xxx`
   - 所有 `this->tilingData->flashAttnAttenMaskParams.xxx` 替换为 `maskParams->xxx`
   - 所有 `this->tilingData->flashAttnWorkspaceParams.xxx` 替换为 `wsParams->xxx`

**预期收益**: 减少 GM Load 次数，降低指针解引用开销

**风险等级**: 低（仅 Init 阶段，不影响热路径）

---

## Step 9: Event ID 拆标量

**Patch 文件**: `step9_event_id_scalar.patch`

**修改文件**: `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改内容**:
1. 成员变量声明（第 171-172 行）：
   - `TEventID mte3ToVId[2]` → `TEventID mte3ToVId_0`, `TEventID mte3ToVId_1`
   - `TEventID vToMte3Id[2]` → `TEventID vToMte3Id_0`, `TEventID vToMte3Id_1`

2. 所有引用点更新（~26 处）：
   - `mte3ToVId[0]` → `mte3ToVId_0`
   - `mte3ToVId[1]` → `mte3ToVId_1`
   - `vToMte3Id[0]` → `vToMte3Id_0`
   - `vToMte3Id[1]` → `vToMte3Id_1`
   - `vToMte3Id[runInfo.loop & (DB - 1)]` → `((runInfo.loop & (DB - 1)) ? vToMte3Id_1 : vToMte3Id_0)`

**预期收益**: 消除数组动态下标对别名分析的污染

**风险等级**: 中（改动点较多，但逻辑等价）

---

## Step 10: 热路径对象构造外提

**Patch 文件**: `step10_hot_path_object_construction.patch`

**修改文件**: `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改内容**:
1. `ProcessVec2_C2GmUpdateUb` 函数（第 1014-1020 行）：
   - 将 `DataCopyParams` 和 `DataCopyPadParams` 声明移到循环外
   - 循环不变字段在循环前设置：`dstStride`、`srcStride`、`blockLen`
   - 循环内仅设置 `blockCount = subMRealsize`

2. `ProcessVec2DSplit` 函数（第 699-705 行）：
   - 同上，将对象构造移到循环外

**预期收益**: 减少循环内对象构造开销

**风险等级**: 低（仅移动声明位置）

---

## Step 11: RunInfoX 拆分

**Patch 文件**: `step11_runinfo_split.patch`

**修改文件**:
- `op_kernel/utils/flash_attn_common_def.h`
- `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`
- `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`
- `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h`

**修改内容**:
1. `flash_attn_common_def.h`（第 238-270 行）：
   - 拆分为 `HotRunInfoX`（15 个热字段，~64B）和 `ColdRunInfoX`（7 个冷字段，~40B）
   - 删除 4 个未使用字段：`s1Idx`、`gIdx`、`qPaddingBeginOffset`、`kvPaddingBeginOffset`

2. 4 个文件 ~130 处字段引用更新：
   - `info.xxx` → `info.hot.xxx` 或 `info.cold.xxx`
   - `runInfo.xxx` → `runInfo.hot.xxx` 或 `runInfo.cold.xxx`

**预期收益**: 热结构体 64B vs 原 120B，改善缓存命中率

**风险等级**: 高（跨 4 文件、改动量大、引入额外 indirection）

---

## 测试建议

1. **测试顺序**: Step 1 → Step 2 → ... → Step 11
2. **每个 Step 测试流程**:
   - 应用 patch: `git apply attention/flash_attn/patches/stepN_xxx.patch`
   - 编译验证
   - 跑 profiling，记录 Scalar cycle
   - 回退 patch: `git apply -R attention/flash_attn/patches/stepN_xxx.patch`
3. **重点关注**:
   - Step 7（Softmax Buffer 位掩码）：可能因 UB 空间增加导致劣化
   - Step 11（RunInfoX 拆分）：改动量大，最可能引入劣化

---

## 测试结果记录表

| Step | 修改内容 | Scalar Cycle | 效果 | 备注 |
|------|---------|-------------|------|------|
| 1 | 缓存循环计算结果 | | | |
| 2 | CalcParams bIdx/n2Idx 优化 | | | |
| 3 | 分支预测提示 | | | |
| 4 | CalcParams 局部变量提升 | | | |
| 5 | Vec Block 热路径局部变量 | | | |
| 6 | Cube Block 热路径局部变量 | | | |
| 7 | Softmax Buffer 位掩码索引 | | | |
| 8 | tilingData 指针引用优化 | | | |
| 9 | Event ID 拆标量 | | | |
| 10 | 热路径对象构造外提 | | | |
| 11 | RunInfoX 拆分 | | | |
