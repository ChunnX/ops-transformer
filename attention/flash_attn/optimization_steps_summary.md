# Flash Attention Scalar Bound 优化步骤总结

## 背景

**问题**：Flash Attention 算子存在严重的 Scalar Bound 问题，总 8w cycle 中计算仅占 1w cycle，Scalar 占比约 87.5%。

**根因**：Scalar 单元被大量 Load/Store 指令阻塞（寄存器 Spill + 别名分析失败 + 常量传播断裂），无法及时向 Cube/Vector/MTE 分发指令。

**优化原则**：基于 CANN 性能优化最佳实践（P1-P9），通过减少冗余计算、提升寄存器利用率、优化分支预测等手段降低 Scalar 开销。

**参考方案**：`scalar_bound_optimization_plan.md`

---

## 第一批：P0 — 结构体瘦身 + 局部变量提升 + 冗余消除（预计 2-3 天）

> 这三类优化互相协同：ConstInfo_t 拆分后热/冷分离，局部变量提升消除 `this->` Load，冗余计算消除减少重复除法。一起做效果最明显。

### Step 1: ConstInfo_t 超大结构体拆分

**对应优化项**：优化项 1 [P7] — 🔴 最高优先级

**主题**：热冷数据分离，解决 >300B 结构体跨 cacheline 问题

**修改内容**：
- 将 `ConstInfo_t<NO_QUANT>` 热路径字段提取为独立 `HotConstInfo` 小结构体（≤48B）：
  - `bSize`, `n2Size`, `gSize`, `dSize`, `dSizeV`, `dBasicBlock`
  - `s1Size`, `s2Size`, `scaleValue`, `coreNum`, `subBlockIdx`, `aicIdx`
- 冷字段（PA、Sink、PSE、PostQuant、LeftPadding、SysPrefix）保留在 `ColdConstInfo`
- `FlashAttentionNoQuantGqaKernel` 中持有两个独立成员：`hotConst` + `coldConst`

**涉及文件**：`op_kernel/utils/flash_attn_common_def.h:371-383`

**预期收益**：热路径 cacheline miss 降低 60%+，编译器可将 HotConstInfo 关键字段驻留寄存器，Scalar cycle 降幅 15-20%

**风险等级**：中

---

### Step 2: FlashAttention 入口局部变量提升

**对应优化项**：优化项 2 [P4] — 🔴 最高优先级（主函数部分）

**主题**：消除主函数热路径中的 `this->constInfo.xxx` Load

**修改内容**：
- `FlashAttention()` 入口处将热路径字段拷贝到局部变量：
  - `const uint32_t n2Size = hotConst.n2Size`
  - `const uint32_t gSize = hotConst.gSize`
  - `const float scaleValue = hotConst.scaleValue`
  - 等
- 后续循环内全部使用局部变量
- `CalcParams()` 改为接收 `const HotConstInfo& hc` 值拷贝

**涉及文件**：`flash_attn_kernel_base_gqa_template.h` 全文件

**预期收益**：消除热路径中 ~40+ 处 `this->` Load，跨函数调用后无需重新加载

**风险等级**：低

---

### Step 3: CalcParams 局部变量 + bIdx/n2Idx 优化

**对应优化项**：优化项 2 [P4] + 优化项 6 [P5/P9]

**主题**：CalcParams 内部成员变量访问消除 + 复用已计算结果避免重复除法

**修改内容**：
- `CalcParams` 入口添加局部变量：`localN2Size`、`localGSize`、`localIsKvContinuous`、`localSubBlockIdx`
- `CalcCurS2StartEndWithSparse` 入口添加局部变量：`localGSize`
- 函数内所有 `constInfo.xxx` 替换为局部变量
- `info.bIdx = prevBIdx`（复用 GetTaskDealMode 已计算的值）
- `info.n2Idx = bN2Cur - prevBIdx * n2Size`（用乘法替代取模）

**涉及文件**：`flash_attn_kernel_base_gqa_template.h:665-742`

**预期收益**：消除 1 次除法 + 1 次取模 + 减少成员变量访问，Scalar cycle 降幅 3-5%

**风险等级**：低

---

### Step 4: 冗余计算合并 — s2LoopTimes/gS1LoopTimes 缓存

**对应优化项**：优化项 10 [P4/P5]

**主题**：消除 GetTaskDealMode 与 UpdateAxisInfo 之间的重复 64-bit 除法

**修改内容**：
- 新增缓存成员变量：`cachedS2LoopTimes`、`cachedGS1LoopTimes`、`cachedPreTokensLeftUp`、`cachedNextTokensLeftUp`
- 在 `GetTaskDealMode` 中，将循环次数和 token 计算移入 batch 切换分支
- `CalcCurS2StartEndWithSparse`、`CalcParams`、`UpdateAxisInfo` 直接使用缓存值
- `GetPreNextTokenLeftUp` 结果缓存，仅在 actSeqLensQ/actSeqLensKv 变化时重算

**涉及文件**：`flash_attn_kernel_base_gqa_template.h:474-476, 744-776`

**预期收益**：每轮迭代消除 2 次 64-bit 除法（~20-40 cycle/次）+ 1 次函数调用，Scalar cycle 降幅 5-10%

**风险等级**：低

---

### Step 5: Vec Block 热路径局部变量

**对应优化项**：优化项 2 [P4]（Vec 部分）

**主题**：Vector 计算单元热路径 constInfo 成员访问消除

**修改内容**：
- `ProcessVec1Dn`：添加 `localScaleValue`、`localSubBlockIdx`，替换 5 处访问
- `ProcessVec1Nd`：添加 `localScaleValue`、`localSubBlockIdx`，替换 9 处访问
- `ProcessVec2DSplit`：添加 `localDSizeV`、`localSubBlockIdx`、`localDBasicBlock`，替换 16 处访问
- `ProcessVec2_C2GmUpdateUb`：添加 `localDSizeV`、`localSubBlockIdx`，替换 8 处访问

**涉及文件**：`flash_attn_block_vec_noquant_gqa.h` 全文件

**预期收益**：减少 ~38 处 constInfo 成员变量访问

**风险等级**：低

---

### Step 6: Cube Block 热路径局部变量

**对应优化项**：优化项 2 [P4]（Cube 部分）

**主题**：Cube 计算单元热路径 constInfo 成员访问消除

**修改内容**：
- `IterateBmm1NdL0Split`、`IterateBmm1NdL1SplitK`、`IterateBmm1DnSplitK`、`IterateBmm1Dn`：各添加 `localDSize`
- `IterateBmm2L1SplitN`、`IterateBmm2l0Split`：各添加 `localDSizeV`
- 总计替换 ~15 处访问

**涉及文件**：`flash_attn_block_cube_noquant_gqa.h` 全文件

**预期收益**：减少 Cube 单元热路径中的成员变量访问

**风险等级**：低

---

## 第二批：P1 — Buffer 索引 + RunInfoX 拆分 + 主循环改造（预计 2 天）

> 在第一批完成结构体瘦身和局部变量提升后，第二批聚焦于索引优化、数据布局改善和主循环控制流重构。

### Step 7: Softmax Buffer 位掩码索引

**对应优化项**：优化项 3 [P1] — 🟠 高优先级

**主题**：用位运算替代取模运算，消除动态下标对常量传播的污染

**修改内容**：
- 添加常量 `SOFTMAX_BUF_DEPTH = 4`
- Buffer 声明从 `[PRELOAD_N + 1]`（深度=3）改为 `[SOFTMAX_BUF_DEPTH]`（深度=4）
- 所有索引从 `% (PRELOAD_N + 1)` 改为 `& (SOFTMAX_BUF_DEPTH - 1)`
- `SoftmaxInitBuffer` 添加第 4 个 buffer 初始化

**涉及文件**：`flash_attn_block_vec_noquant_gqa.h:176-178, 351-354, 512-514, 641, 649, 767, 775, 1064, 1072, 1083`

**预期收益**：消除热路径中 6~9 处取模运算，每次节省 ~5-10 cycle 的 Scalar 除法开销，Scalar cycle 降幅 5-8%

**风险等级**：中（多占 768B UB 空间，需确认 UB 预算）

---

### Step 8: RunInfoX 拆分 + 按值传递

**对应优化项**：优化项 4 [P6/P7] — 🟠 高优先级

**主题**：超大结构体瘦身 + 热函数按值传递消除别名顾虑

**修改内容**：
- `RunInfoX`（>160B）拆分为三个子结构体：
  - `RunInfoCore`（≤32B）：`loop`, `mloop`, `bIdx`, `n2Idx`, `gS1Idx`, `s1Idx`, `s2Idx`, `actMSize`, `actVecMSize`, `vecMbaseIdx`
  - `RunInfoS2`（≤24B）：`actSingleLoopS2Size`, `actSingleLoopS2SizeAlign`, `actS2Size`, `isFirstS2Loop`, `isLastS2Loop`, `isS2SplitCore`
  - `RunInfoMisc`（冷字段）：`actS1Size`, `preTokensLeftUp`, `nextTokensLeftUp`, `faTmpOutWsPos` 等
- 删除未使用字段
- 热路径函数签名改为按值传递小子结构体：`void ProcessVec1Nd(..., RunInfoCore ri, RunInfoS2 rs)`
- 所有引用点更新（~130 处）

**涉及文件**：`op_kernel/utils/flash_attn_common_def.h:238-270`，4 个文件

**预期收益**：热结构体 ≤32B vs 原 >160B，按值传递消除别名顾虑，编译器可做更激进的寄存器分配，Scalar cycle 降幅 8-12%

**风险等级**：高（改动量大）

---

### Step 9: FlashAttention 主循环状态机改造

**对应优化项**：优化项 5 [P2/P3] — 🟠 高优先级

**主题**：隐式状态机改显式循环 + 主尾块分离 + 分支预测优化

**修改内容**：
- 将 `GetTaskDealMode` 中循环无关判断外提：`actSeqLensKv`/`actSeqLensQ` 仅在换 batch 时重算
- `UpdateAxisInfo` 中的 `s2LoopTimes`/`gS1LoopTimes` 与 `GetTaskDealMode` 合并为一次计算（与 Step 4 缓存协同）
- 对 `DEAL_ZERO`/`SKIP`/`S2_END` 等低频路径用 `unlikely()` 标注，确保热路径（`CREATE_TASK`）指令连续排布
- 评估将 `while` 循环改为 `for` 循环的可行性：预计算总任务数 `totalTasks`，用 `for (uint32_t t = 0; t < totalTasks; ++t)` 回跳走专用硬件

**涉及文件**：`flash_attn_kernel_base_gqa_template.h:390-446`

**预期收益**：消除每轮迭代 2 次重复的 64-bit 除法，减少分支预测失败 ~30%，Scalar cycle 降幅 10-15%

**风险等级**：高（控制流重构）

---

## 第三批：P2 — 细节打磨（预计 1-2 天）

> 低风险、低改动量的收尾优化，积少成多。

### Step 10: tilingData 多级指针解引用消除

**对应优化项**：优化项 7 [P6/P8]

**主题**：减少 Init 阶段 GM Load 次数

**修改内容**：
- `InitConstInfo` 中一次性将 `tilingData->flashAttnBaseParams` 整体拷贝到局部结构体：
  ```cpp
  FlashAttnBaseParams baseParams = *(tilingData->flashAttnBaseParams);
  ```
- 添加局部指针变量：`baseParams`、`maskParams`、`wsParams`、`paParams`
- 所有 `this->tilingData->flashAttnXxxParams.xxx` 替换为 `xxxParams.xxx`
- 进一步评估：将编译期可知的字段（如 `dSize=128`）改为模板参数参与常量折叠

**涉及文件**：`flash_attn_kernel_base_gqa_template.h:301-350`

**预期收益**：Init 阶段 GM Load 从 20+ 次降为 1 次批量拷贝，Scalar cycle 降幅 2-3%

**风险等级**：低

---

### Step 11: Event ID 数组拆为标量

**对应优化项**：优化项 8 [P1]

**主题**：消除数组动态下标对别名分析的污染

**修改内容**：
- 成员变量从数组拆为标量：
  - `mte3ToVId[2]` → `mte3ToVId_0`、`mte3ToVId_1`
  - `vToMte3Id[2]` → `vToMte3Id_0`、`vToMte3Id_1`
- 所有引用点更新（~26 处）
- 动态索引改为三目运算符：`((runInfo.loop & (DB - 1)) ? vToMte3Id_1 : vToMte3Id_0)`

**涉及文件**：`flash_attn_block_vec_noquant_gqa.h:171-172`

**预期收益**：消除数组动态下标对同结构体其他成员的别名分析污染，Scalar cycle 降幅 2-3%

**风险等级**：中

---

### Step 12: 热路径对象构造外提

**对应优化项**：优化项 9 [P9]

**主题**：减少循环内对象构造/析构开销

**修改内容**：
- `ProcessVec1Nd` 中 `LocalTensor`（`sumUb`、`maxUb`、`expUb`、`mmRes`、`stage1CastTensor` 等）提到循环外构造，循环内复用
- `ProcessVec2_C2GmUpdateUb` 和 `ProcessVec2DSplit` 中：
  - `DataCopyParams` 和 `DataCopyPadParams` 声明移到循环外
  - 循环不变字段在循环前设置
  - 循环内仅修改变化字段（如 `blockCount = subMRealsize`）

**涉及文件**：`flash_attn_block_vec_noquant_gqa.h:499-623, 625-665`

**预期收益**：减少热路径中每轮 ~10+ 个 Store 指令，Scalar cycle 降幅 3-5%

**风险等级**：低

---

## 总结表

| 批次 | Step | 主题 | 对应优化项 | 涉及文件 | 预期 Scalar 降幅 | 风险 |
|------|------|------|-----------|---------|-----------------|------|
| **第一批 P0** | 1 | ConstInfo_t 结构体拆分 | 项1 [P7] | common_def | 15-20% | 中 |
| | 2 | FlashAttention 入口局部变量 | 项2 [P4] | kernel_base | 15-25% | 低 |
| | 3 | CalcParams 局部变量 + bIdx/n2Idx | 项2+6 [P4/P5] | kernel_base | 3-5% | 低 |
| | 4 | 冗余计算合并（缓存循环结果） | 项10 [P4/P5] | kernel_base | 5-10% | 低 |
| | 5 | Vec Block 热路径局部变量 | 项2 [P4] | block_vec | (含在项2) | 低 |
| | 6 | Cube Block 热路径局部变量 | 项2 [P4] | block_cube | (含在项2) | 低 |
| **第二批 P1** | 7 | Softmax Buffer 位掩码 | 项3 [P1] | block_vec | 5-8% | 中 |
| | 8 | RunInfoX 拆分 + 按值传递 | 项4 [P6/P7] | 4 个文件 | 8-12% | 高 |
| | 9 | 主循环状态机改造 | 项5 [P2/P3] | kernel_base | 10-15% | 高 |
| **第三批 P2** | 10 | tilingData 指针解引用消除 | 项7 [P6/P8] | kernel_base | 2-3% | 低 |
| | 11 | Event ID 拆标量 | 项8 [P1] | block_vec | 2-3% | 中 |
| | 12 | 热路径对象构造外提 | 项9 [P9] | block_vec | 3-5% | 低 |

**综合预期**：Scalar cycle 从 7w 降至 ~3-4w（降幅 40-50%），总 cycle 从 8w 降至 ~4-5w。

---

## 与旧版步骤的变更说明

| 变更 | 说明 |
|------|------|
| **新增 Step 1** | 补充了旧版缺失的 P0 最高优先级项：ConstInfo_t 超大结构体拆分 |
| **新增 Step 9** | 补充了旧版缺失的 P0 项：主循环状态机改显式循环（旧版仅有分支预测提示，未覆盖核心改造） |
| **重组顺序** | 按 P0→P1→P2 三批排列，同批次内按依赖关系排序（先结构体拆分，再局部变量提升，再冗余消除） |
| **合并旧 Step 2→3** | bIdx/n2Idx 优化与 CalcParams 局部变量合并为一步，二者紧密关联 |
| **RunInfoX 升级** | 从旧版简单的 Hot/Cold 二分，升级为按优化计划的三分（RunInfoCore/RunInfoS2/RunInfoMisc）+ 按值传递 |

---

## 测试建议

1. **测试顺序**：按批次执行，每批完成后跑 profiling 对比 Scalar cycle 占比
2. **每个 Step 测试流程**：
   - 应用 patch：`git apply attention/flash_attn/patches/stepN_xxx.patch`
   - 编译验证
   - 跑 profiling，记录 Scalar cycle
   - 回退 patch：`git apply -R attention/flash_attn/patches/stepN_xxx.patch`
3. **重点关注**：
   - Step 1（ConstInfo_t 拆分）：第一批基石，影响后续所有步骤的字段访问方式
   - Step 7（Softmax Buffer 位掩码）：可能因 UB 空间增加导致劣化
   - Step 8（RunInfoX 拆分）：改动量大（~130 处），最可能引入劣化
   - Step 9（主循环改造）：控制流重构，需充分验证功能正确性
4. **第一批完成后必须 profiling**：第一批预期贡献 ~60-70% 的总优化收益，如果效果不达预期需调整后续策略

---

## Patch 文件列表

```
patches/
├── step1_constinfo_split.patch
├── step2_flash_attention_local_vars.patch
├── step3_calcparams_local_vars_bidx.patch
├── step4_redundant_computation_cache.patch
├── step5_vec_block_local_vars.patch
├── step6_cube_block_local_vars.patch
├── step7_softmax_buffer_bitmask.patch
├── step8_runinfox_split_pass_by_value.patch
├── step9_main_loop_state_machine.patch
├── step10_tilingdata_dereference.patch
├── step11_event_id_scalar.patch
└── step12_hot_path_object_construction.patch
```
