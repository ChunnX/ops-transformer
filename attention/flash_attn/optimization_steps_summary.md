# Flash Attention Scalar Bound 优化步骤总结

## 背景

**问题**：Flash Attention 算子存在严重的 Scalar Bound 问题，总 8w cycle 中计算仅占 1w cycle，Scalar 占比约 87.5%。

**根因**：Scalar 单元被大量 Load/Store 指令阻塞，无法及时向 Cube/Vector/MTE 分发指令。

**优化原则**：基于 CANN 性能优化最佳实践（P1-P9），通过减少冗余计算、提升寄存器利用率、优化分支预测等手段降低 Scalar 开销。

---

## 优化步骤

### Step 1: 缓存循环计算结果

**主题**：消除热路径中的冗余计算

**修改内容**：
- 新增 4 个缓存成员变量：`cachedS2LoopTimes`、`cachedGS1LoopTimes`、`cachedPreTokensLeftUp`、`cachedNextTokensLeftUp`
- 在 `GetTaskDealMode` 中，将循环次数和 token 计算移入 batch 切换分支
- `CalcCurS2StartEndWithSparse`、`CalcParams`、`UpdateAxisInfo` 直接使用缓存值

**预期收益**：每轮迭代消除 2 次 64-bit 除法 + 1 次函数调用

**风险等级**：低

---

### Step 2: CalcParams bIdx/n2Idx 优化

**主题**：复用已计算结果，避免重复除法

**修改内容**：
- `info.bIdx = prevBIdx`（复用 GetTaskDealMode 已计算的值）
- `info.n2Idx = bN2Cur - prevBIdx * constInfo.n2Size`（用乘法替代取模）

**预期收益**：消除 1 次除法 + 1 次取模运算

**风险等级**：低

---

### Step 3: 分支预测提示

**主题**：通过 likely/unlikely 提示优化分支预测

**修改内容**：
- `FlashAttention` 主循环中添加分支预测提示：
  - `if (likely(shouldDispatchTask))`
  - `if (likely(taskDealMode == TASK_DEAL_MODE::CREATE_TASK))`
  - `else if (unlikely(taskDealMode == TASK_DEAL_MODE::DEAL_ZERO))`

**预期收益**：改善分支预测命中率，减少流水线冲刷

**风险等级**：极低

---

### Step 4: CalcParams 局部变量提升

**主题**：减少成员变量访问开销

**修改内容**：
- `CalcParams` 入口添加局部变量：`localN2Size`、`localGSize`、`localIsKvContinuous`、`localSubBlockIdx`
- `CalcCurS2StartEndWithSparse` 入口添加局部变量：`localGSize`
- 函数内所有对应 `constInfo.xxx` 访问替换为局部变量

**预期收益**：减少成员变量访问开销，提升寄存器利用率

**风险等级**：低

---

### Step 5: Vec Block 热路径局部变量

**主题**：Vector 计算单元热路径优化

**修改内容**：
- `ProcessVec1Dn`：添加 `localScaleValue`、`localSubBlockIdx`，替换 5 处访问
- `ProcessVec1Nd`：添加 `localScaleValue`、`localSubBlockIdx`，替换 9 处访问
- `ProcessVec2DSplit`：添加 `localDSizeV`、`localSubBlockIdx`、`localDBasicBlock`，替换 16 处访问
- `ProcessVec2_C2GmUpdateUb`：添加 `localDSizeV`、`localSubBlockIdx`，替换 8 处访问

**预期收益**：减少 ~38 处 constInfo 成员变量访问

**风险等级**：低

---

### Step 6: Cube Block 热路径局部变量

**主题**：Cube 计算单元热路径优化

**修改内容**：
- `IterateBmm1NdL0Split`、`IterateBmm1NdL1SplitK`、`IterateBmm1DnSplitK`、`IterateBmm1Dn`：各添加 `localDSize`
- `IterateBmm2L1SplitN`、`IterateBmm2l0Split`：各添加 `localDSizeV`
- 总计替换 ~15 处访问

**预期收益**：减少 Cube 单元热路径中的成员变量访问

**风险等级**：低

---

### Step 7: Softmax Buffer 位掩码索引

**主题**：用位运算替代取模运算

**修改内容**：
- 添加常量 `SOFTMAX_BUF_DEPTH = 4`
- Buffer 声明从 `[PRELOAD_N + 1]` 改为 `[SOFTMAX_BUF_DEPTH]`
- 所有索引从 `% (PRELOAD_N + 1)` 改为 `& (SOFTMAX_BUF_DEPTH - 1)`
- `SoftmaxInitBuffer` 添加第 4 个 buffer 初始化

**预期收益**：消除 16 处取模运算，每次节省 ~5-10 cycle

**风险等级**：中（多占 768B UB 空间）

---

### Step 8: tilingData 指针引用优化

**主题**：减少 GM Load 次数

**修改内容**：
- `InitConstInfo` 中添加局部指针变量：
  - `baseParams`、`maskParams`、`wsParams`、`paParams`
- 所有 `this->tilingData->flashAttnXxxParams.xxx` 替换为 `xxxParams->xxx`

**预期收益**：减少 GM Load 次数，降低指针解引用开销

**风险等级**：低

---

### Step 9: Event ID 拆标量

**主题**：消除数组动态下标对别名分析的污染

**修改内容**：
- 成员变量从数组拆为标量：
  - `mte3ToVId[2]` → `mte3ToVId_0`、`mte3ToVId_1`
  - `vToMte3Id[2]` → `vToMte3Id_0`、`vToMte3Id_1`
- 所有引用点更新（~26 处）
- 动态索引改为三目运算符：`((runInfo.loop & (DB - 1)) ? vToMte3Id_1 : vToMte3Id_0)`

**预期收益**：消除数组动态下标对别名分析的污染

**风险等级**：中

---

### Step 10: 热路径对象构造外提

**主题**：减少循环内对象构造开销

**修改内容**：
- `ProcessVec2_C2GmUpdateUb` 和 `ProcessVec2DSplit` 中：
  - 将 `DataCopyParams` 和 `DataCopyPadParams` 声明移到循环外
  - 循环不变字段在循环前设置
  - 循环内仅设置 `blockCount = subMRealsize`

**预期收益**：减少循环内对象构造开销

**风险等级**：低

---

### Step 11: RunInfoX 拆分

**主题**：热冷数据分离，改善缓存命中率

**修改内容**：
- `RunInfoX` 拆分为：
  - `HotRunInfoX`（15 个热字段，~64B）
  - `ColdRunInfoX`（7 个冷字段，~40B）
- 删除 4 个未使用字段
- 4 个文件 ~130 处字段引用更新为 `.hot.xxx` 或 `.cold.xxx`

**预期收益**：热结构体 64B vs 原 120B，改善缓存命中率

**风险等级**：高

---

## 总结表

| Step | 主题 | 修改文件 | 预期收益 | 风险 |
|------|------|---------|---------|------|
| 1 | 缓存循环计算结果 | kernel_base | 消除 2 次除法 + 1 次函数调用/轮 | 低 |
| 2 | bIdx/n2Idx 优化 | kernel_base | 消除 1 次除法 + 1 次取模 | 低 |
| 3 | 分支预测提示 | kernel_base | 改善分支预测命中率 | 极低 |
| 4 | CalcParams 局部变量 | kernel_base | 减少成员变量访问 | 低 |
| 5 | Vec Block 局部变量 | block_vec | 减少 38 处成员访问 | 低 |
| 6 | Cube Block 局部变量 | block_cube | 减少 15 处成员访问 | 低 |
| 7 | Softmax Buffer 位掩码 | block_vec | 消除 16 处取模 | 中 |
| 8 | tilingData 指针引用 | kernel_base | 减少 GM Load | 低 |
| 9 | Event ID 拆标量 | block_vec | 消除数组别名污染 | 中 |
| 10 | 对象构造外提 | block_vec | 减少循环内构造 | 低 |
| 11 | RunInfoX 拆分 | 4 个文件 | 改善缓存命中率 | 高 |

---

## 测试建议

1. **测试顺序**：Step 1 → Step 2 → ... → Step 11
2. **每个 Step 测试流程**：
   - 应用 patch：`git apply attention/flash_attn/patches/stepN_xxx.patch`
   - 编译验证
   - 跑 profiling，记录 Scalar cycle
   - 回退 patch：`git apply -R attention/flash_attn/patches/stepN_xxx.patch`
3. **重点关注**：
   - Step 7（Softmax Buffer 位掩码）：可能因 UB 空间增加导致劣化
   - Step 11（RunInfoX 拆分）：改动量大，最可能引入劣化

---

## Patch 文件列表

```
patches/
├── step1_cached_loop_computation.patch
├── step2_calcparams_bidx_n2idx.patch
├── step3_branch_prediction_hints.patch
├── step4_calcparams_local_variables.patch
├── step5_vec_block_local_variables.patch
├── step6_cube_block_local_variables.patch
├── step7_softmax_buffer_bitmask.patch
├── step8_tilingdata_pointer_reference.patch
├── step9_event_id_scalar.patch
├── step10_hot_path_object_construction.patch
└── step11_runinfo_split.patch
```
