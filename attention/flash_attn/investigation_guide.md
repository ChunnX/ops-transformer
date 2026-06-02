# Flash Attention Scalar 优化逐步排查指南

> **目的**：逐步合入优化项，每步独立编译+跑 profiling，定位每项修改的实际效果（优化/劣化/中性）
> 
> **原则**：低风险/高独立性优先，高耦合/高风险靠后
> 
> **创建日期**：2026-06-02

---

## 排查流程

### 基本步骤

1. **合入当前 Step 的修改**
2. **编译验证**：确保编译通过
3. **跑 profiling**：对比 Scalar cycle 变化
4. **记录结果**：在下方表格中标记效果
5. **决策**：
   - ✅ **优化**：保留，继续下一步
   - ⚠️ **中性**：保留（代码质量改善）
   - ❌ **劣化**：回退该步，记录原因

---

## 排查顺序

### Step 1：冗余计算合并（优化项 10）

**文件**：`op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改点**：
- 新增 4 个缓存成员变量（在 `FlashAttentionNoQuantGqaKernel` 类中）：
  ```cpp
  uint64_t cachedS2LoopTimes = 0;
  uint64_t cachedGS1LoopTimes = 0;
  int64_t cachedPreTokensLeftUp = 0;
  int64_t cachedNextTokensLeftUp = 0;
  ```
- `GetTaskDealMode`：将 `s2LoopTimes`/`gS1LoopTimes`/`GetPreNextTokenLeftUp` 计算移入 batch 切换分支（`isFirstTask || prevBIdx != bIdx`）
- `UpdateAxisInfo`：使用 `cachedS2LoopTimes`/`cachedGS1LoopTimes` 替代重复计算
- `CalcCurS2StartEndWithSparse`：使用缓存 token 值
- `CalcParams`：`info.preTokensLeftUp = cachedPreTokensLeftUp; info.nextTokensLeftUp = cachedNextTokensLeftUp;`

**风险等级**：🟢 低  
**风险说明**：纯算法等价变换，不改数据结构  
**预期收益**：每轮消除 2 次 64-bit 除法 + 1 次函数调用  
**依赖关系**：无

---

### Step 2：CalcParams bIdx/n2Idx 优化（优化项 6 部分）

**文件**：`op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改点**：
- `CalcParams` 中：
  ```cpp
  // 原：info.bIdx = bN2Cur / constInfo.n2Size;
  info.bIdx = prevBIdx;  // 复用已计算值
  
  // 原：info.n2Idx = bN2Cur % constInfo.n2Size;
  info.n2Idx = bN2Cur - prevBIdx * constInfo.n2Size;  // 乘法替代取模
  ```

**风险等级**：🟢 低  
**风险说明**：纯算术等价  
**预期收益**：消除 1 次除法 + 1 次取模  
**依赖关系**：无，但和 Step 1 同文件，建议先合 Step 1 再合此项

---

### Step 3：主循环分支预测提示（优化项 5 部分）

**文件**：`op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改点**：
- `FlashAttention` 主循环中：
  ```cpp
  if (likely(taskDealMode == TASK_DEAL_MODE::CREATE_TASK)) { ... }
  else if (unlikely(taskDealMode == TASK_DEAL_MODE::DEAL_ZERO)) { ... }
  ```

**风险等级**：🟢 极低  
**风险说明**：仅加编译器 hint，不改变逻辑  
**预期收益**：改善分支预测命中率  
**依赖关系**：无

---

### Step 4：CalcParams 局部变量提升（优化项 2 部分）

**文件**：`op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改点**：
- `CalcParams` 入口加局部变量：
  ```cpp
  const uint32_t localN2Size = constInfo.n2Size;
  const uint32_t localGSize = constInfo.gSize;
  const bool localIsKvContinuous = constInfo.isKvContinuous;
  const uint8_t localSubBlockIdx = constInfo.subBlockIdx;
  ```
- 函数内所有对应 `constInfo.xxx` 访问替换为局部变量
- `CalcCurS2StartEndWithSparse` 入口加 `const uint32_t localGSize = constInfo.gSize;`

**风险等级**：🟢 低  
**风险说明**：纯局部变量替换  
**预期收益**：减少成员变量访问开销  
**依赖关系**：无

---

### Step 5：Vec Block 热路径局部变量（优化项 2 部分）

**文件**：`op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改点**：
- `ProcessVec1Dn`：加 `localScaleValue`、`localSubBlockIdx`
- `ProcessVec1Nd`：加 `localScaleValue`、`localSubBlockIdx`
- `ProcessVec2DSplit`：加 `localDSizeV`、`localSubBlockIdx`、`localDBasicBlock`
- `ProcessVec2_C2GmUpdateUb`：加 `localDSizeV`、`localSubBlockIdx`

**风险等级**：🟢 低  
**风险说明**：独立文件，纯局部变量替换  
**预期收益**：减少 ~38 处 constInfo 访问  
**依赖关系**：无

---

### Step 6：Cube Block 热路径局部变量（优化项 2 部分）

**文件**：`op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h`

**修改点**：
- `IterateBmm1NdL0Split`：加 `localDSize`
- `IterateBmm1NdL1SplitK`：加 `localDSize`
- `IterateBmm1DnSplitK`：加 `localDSize`
- `IterateBmm1Dn`：加 `localDSize`
- `IterateBmm2L1SplitN`：加 `localDSizeV`
- `IterateBmm2l0Split`：加 `localDSizeV`

**风险等级**：🟢 低  
**风险说明**：独立文件，纯局部变量替换  
**预期收益**：减少 ~15 处 constInfo 访问  
**依赖关系**：无

---

### Step 7：Softmax Buffer 位掩码（优化项 3）⚠️

**文件**：`op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改点**：
- 添加常量：
  ```cpp
  static constexpr uint32_t SOFTMAX_BUF_DEPTH = 4;
  static_assert((SOFTMAX_BUF_DEPTH & (SOFTMAX_BUF_DEPTH - 1)) == 0, "...");
  ```
- Buffer 声明改为：
  ```cpp
  TBuf<> softmaxMaxBuf[SOFTMAX_BUF_DEPTH];
  TBuf<> softmaxSumBuf[SOFTMAX_BUF_DEPTH];
  TBuf<> softmaxExpBuf[SOFTMAX_BUF_DEPTH];
  ```
- 16 处 `% (PRELOAD_N + 1)` → `& (SOFTMAX_BUF_DEPTH - 1)`
- `SoftmaxInitBuffer` 加第 4 个 buffer 初始化：
  ```cpp
  tPipe->InitBuffer(softmaxSumBuf[3], 256);
  tPipe->InitBuffer(softmaxMaxBuf[3], 256);
  tPipe->InitBuffer(softmaxExpBuf[3], 256);
  ```

**风险等级**：🟡 中  
**风险说明**：多占 3 个 TBuf（768B）UB 空间，需确认 UB 预算充足  
**预期收益**：消除 16 处取模运算  
**依赖关系**：无  
**⚠️ 注意**：如果 UB 不够会编译失败，这是最可能劣化的点之一

---

### Step 8：tilingData 指针引用优化（优化项 7）

**文件**：`op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`

**修改点**：
- `InitConstInfo()` 中用 `const __gm__` 指针替代重复的 `this->tilingData->` 链式解引用：
  ```cpp
  const __gm__ FlashAttnBaseParams *baseParams = &(this->tilingData->flashAttnBaseParams);
  const __gm__ FlashAttnAttenMaskParams *maskParams = &(this->tilingData->flashAttnAttenMaskParams);
  const __gm__ FlashAttnWorkspaceParams *wsParams = &(this->tilingData->flashAttnWorkspaceParams);
  constInfo.bSize = baseParams->bSize;
  // ... 其他字段类似
  ```

**风险等级**：🟢 低  
**风险说明**：仅 Init 阶段，不影响热路径  
**预期收益**：减少 GM Load 次数  
**依赖关系**：无

---

### Step 9：Event ID 数组拆标量（优化项 8）

**文件**：`op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改点**：
- 声明改为标量：
  ```cpp
  TEventID mte3ToVId_0;
  TEventID mte3ToVId_1;
  TEventID vToMte3Id_0;
  TEventID vToMte3Id_1;
  ```
- `AllocEventID`/`FreeEventID` 更新为使用标量
- 所有引用点更新：
  - `mte3ToVId[0]` → `mte3ToVId_0`
  - `vToMte3Id[runInfo.hot.loop & (DB - 1)]` → `((runInfo.hot.loop & (DB - 1)) ? vToMte3Id_1 : vToMte3Id_0)`

**风险等级**：🟡 中  
**风险说明**：改动点较多（~26 处），但逻辑等价  
**预期收益**：消除数组动态下标对别名分析的污染  
**依赖关系**：无

---

### Step 10：热路径对象构造外提（优化项 9）

**文件**：`op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`

**修改点**：
- `ProcessVec2_C2GmUpdateUb`：`DataCopyParams`/`DataCopyPadParams` 提到循环外，循环不变字段在循环前设置
- `ProcessVec2DSplit`：同上

**风险等级**：🟢 低  
**风险说明**：仅移动声明位置  
**预期收益**：减少循环内对象构造开销  
**依赖关系**：无

---

### Step 11：RunInfoX 拆分（优化项 4）🔴

**文件**：
- `op_kernel/utils/flash_attn_common_def.h`
- `op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`
- `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h`
- `op_kernel/arch35/flash_attn_block_cube_noquant_gqa.h`

**修改点**：
- 拆分为 `HotRunInfoX`（15 个热字段）+ `ColdRunInfoX`（7 个冷字段）
- 删除 4 个未使用字段：`s1Idx`、`gIdx`、`qPaddingBeginOffset`、`kvPaddingBeginOffset`
- 4 个文件 ~130 处字段引用改为 `.hot.*`/`.cold.*`

**风险等级**：🔴 最高  
**风险说明**：跨 4 文件、改动量最大、最可能引入劣化或编译问题  
**预期收益**：热结构体 64B vs 原 120B，改善缓存命中率  
**依赖关系**：无  
**⚠️ 建议**：放最后，前面 10 步都验证完再合

---

## 结果记录表

| Step | 优化项 | 文件 | 风险 | Scalar cycle 变化 | 效果 | 备注 |
|------|--------|------|------|------------------|------|------|
| 1 | 冗余计算合并 | kernel_base | 🟢 | | | |
| 2 | bIdx/n2Idx 优化 | kernel_base | 🟢 | | | |
| 3 | 分支预测提示 | kernel_base | 🟢 | | | |
| 4 | CalcParams 局部变量 | kernel_base | 🟢 | | | |
| 5 | Vec Block 局部变量 | block_vec | 🟢 | | | |
| 6 | Cube Block 局部变量 | block_cube | 🟢 | | | |
| 7 | Softmax Buffer 位掩码 | block_vec | 🟡 | | | ⚠️ 可能劣化 |
| 8 | tilingData 指针引用 | kernel_base | 🟢 | | | |
| 9 | Event ID 拆标量 | block_vec | 🟡 | | | |
| 10 | 对象构造外提 | block_vec | 🟢 | | | |
| 11 | RunInfoX 拆分 | 4 个文件 | 🔴 | | | ⚠️ 最可能劣化 |

---

## 重点关注项

### 最可能劣化的项

1. **Step 7（Softmax Buffer 位掩码）**
   - 原因：多占 768B UB 空间
   - 判断：如果编译失败或 Scalar cycle 增加，回退此项

2. **Step 11（RunInfoX 拆分）**
   - 原因：跨 4 文件、改动量大、引入额外 indirection（`.hot.*`）
   - 判断：如果 Scalar cycle 增加或编译问题，回退此项

### 最可能优化的项

1. **Step 1（冗余计算合并）**：每轮消除 2 次 64-bit 除法
2. **Step 5/6（局部变量提升）**：减少 ~53 处成员变量访问
3. **Step 7（位掩码）**：消除 16 处取模运算（如果 UB 够）

---

## 排查策略总结

| 步骤 | 操作 | 判断标准 |
|------|------|---------|
| 每步合入后 | 编译 + 跑 profiling | 对比 Scalar cycle 变化 |
| 如果劣化 | 回退该步，记录原因 | 标记为"劣化项" |
| 如果优化 | 保留，继续下一步 | 标记为"优化项" |
| 如果无变化 | 保留（代码质量改善） | 标记为"中性" |

**建议**：先完成 Step 1-6（低风险），快速验证基础优化效果，再逐步验证 Step 7-11（中高风险）。
