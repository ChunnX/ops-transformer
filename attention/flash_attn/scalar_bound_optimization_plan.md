# Flash Attention Scalar Bound 优化方案

> **问题量化**：总 8w cycle，计算仅 1w cycle → Scalar 占比 ~87.5%
> **根因**：Scalar 单元被大量 Load/Store 指令阻塞（寄存器 Spill + 别名分析失败 + 常量传播断裂），无法及时向 Cube/Vector/MTE 分发指令
> **扫描工具**：cann-skills-zyh / ascendc-performance-best-practices / reference/scalar/
> **扫描日期**：2026-06-02

---

## 优化项 1：ConstInfo_t 超大结构体拆分 [P7]

**严重度**：🔴 最高优先级
**涉及文件**：`op_kernel/utils/flash_attn_common_def.h:371-383`

**现状问题**：
`ConstInfo_t<NO_QUANT>` 多重继承 8 个子结构体（CommonConstInfo + PAConstInfo + LseConstInfo + SinkConstInfo + PseConstInfo + TensorListConstInfo + PostQuantConstInfo + LeftPaddingConstInfo + SysPrefixConstInfo），估算 >300B，远超 64B cacheline。热路径每次访问成员都跨 cacheline Load，且编译器无法将整个结构体放入寄存器。

**修改方案**：

1. 将热路径字段提取为独立的 HotConstInfo 小结构体（≤48B）：

```cpp
struct HotConstInfo {
    uint32_t bSize, n2Size, gSize, dSize, dSizeV, dBasicBlock;
    uint64_t s1Size, s2Size;
    float scaleValue;
    uint32_t coreNum;
    uint8_t subBlockIdx;
    uint32_t aicIdx;
};
```

2. 冷字段（PA、Sink、PSE、PostQuant、LeftPadding、SysPrefix）保留在 ColdConstInfo 中，仅在 Init 阶段使用

3. FlashAttentionNoQuantGqaKernel 中持有两个独立成员：

```cpp
HotConstInfo hotConst;   // 热路径频繁访问
ColdConstInfo coldConst; // 仅初始化/特殊路径使用
```

**预期收益**：热路径 cacheline miss 降低 60%+，编译器可将 HotConstInfo 关键字段驻留寄存器

---

## 优化项 2：constInfo 成员变量 → 局部变量 [P4]

**严重度**：🔴 最高优先级
**涉及文件**：

- `flash_attn_kernel_base_gqa_template.h` 全文件
- `flash_attn_block_vec_noquant_gqa.h` 全文件
- `flash_attn_block_cube_noquant_gqa.h` 全文件

**现状问题**：
热路径函数（`FlashAttention`、`CalcParams`、`ProcessVec1Nd`、`IterateBmm1NdL0Split` 等）中大量通过 `constInfo.xxx` / `this->constInfo.xxx` 访问成员变量。每次访问生成 `this+offset` Load，跨函数调用后必须重新 Load（别名分析保守）。

**修改方案**：

1. FlashAttention() 入口处将热路径字段拷贝到局部变量：

```cpp
__aicore__ inline void FlashAttention(uint32_t sectionIdx) {
    const uint32_t n2Size = constInfo.n2Size;
    const uint32_t gSize = constInfo.gSize;
    const uint64_t s2BaseSize_local = s2BaseSize;
    const float scaleValue = constInfo.scaleValue;
    // ... 后续循环内全部使用局部变量
}
```

2. CalcParams() 改为接收 HotConstInfo 值拷贝而非通过 this 访问：

```cpp
__aicore__ inline void CalcParams(..., const HotConstInfo& hc, RunInfoX& info)
```

3. ProcessVec1Nd/ProcessVec1Dn 中将 constInfo.scaleValue、constInfo.subBlockIdx 等提升到函数入口局部变量

**预期收益**：消除热路径中 ~40+ 处 `this->` Load，跨函数调用后无需重新加载

---

## 优化项 3：softmax 状态 Buffer 数组索引优化 [P1]

**严重度**：🟠 高优先级
**涉及文件**：`flash_attn_block_vec_noquant_gqa.h:176-178`, `351-354`, `512-514`, `641`, `649`, `767`, `775`, `1064`, `1072`, `1083`

**现状问题**：

```cpp
TBuf<> softmaxMaxBuf[PRELOAD_N + 1];  // 深度=3
TBuf<> softmaxSumBuf[PRELOAD_N + 1];
TBuf<> softmaxExpBuf[PRELOAD_N + 1];
```

访问方式 `softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)]` 和 `softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)]`：

- `% 3` 取模无法被编译器优化为位运算（3 不是 2 的幂）
- 动态下标访问结构体数组导致编译器放弃整个结构体的常量传播（P1 原则）
- 代码注释已说明："3 非2的幂…若改为位掩码需将深度抬到4，会额外多占 1/3 的 UB"

**修改方案**：

方案 A（推荐）：将深度从 3 抬到 4，用位掩码索引

```cpp
TBuf<> softmaxMaxBuf[4];  // 多 1 个 TBuf（256B），换取消除取模
TBuf<> softmaxSumBuf[4];
TBuf<> softmaxExpBuf[4];
// 访问改为：
softmaxSumBuf[runInfo.mloop & 3]
softmaxExpBuf[runInfo.loop & 3]
```

方案 B：拆为独立标量（如果 UB 预算紧张）

```cpp
TBuf<> softmaxMaxBuf0, softmaxMaxBuf1, softmaxMaxBuf2;
// 用 switch/if 手动分发（适用于深度=3 且循环次数少的场景）
```

**预期收益**：消除热路径中 6~9 处取模运算，每次节省 ~5-10 cycle 的 Scalar 除法开销

---

## 优化项 4：RunInfoX 超大结构体瘦身 [P7] + 按值传递 [P6]

**严重度**：🟠 高优先级
**涉及文件**：`op_kernel/utils/flash_attn_common_def.h:238-270`

**现状问题**：
`RunInfoX` 含 25+ 字段，估算 >160B，远超 64B cacheline。在 `CreateTask`→`CalcParams`→`ExecuteTask`→`ComputeMm1/Vec1/Mm2/Vec2` 的调用链中，RunInfoX 以引用传递，编译器因别名顾虑无法优化。

**修改方案**：

1. 按使用阶段拆分为子结构体：

```cpp
struct RunInfoCore {     // ≤32B，Cube/Vec 计算热路径用
    uint32_t loop, mloop;
    uint32_t bIdx, n2Idx, gS1Idx, s1Idx, s2Idx;
    uint32_t actMSize, actVecMSize, vecMbaseIdx;
};

struct RunInfoS2 {       // ≤24B，S2 循环控制
    uint32_t actSingleLoopS2Size, actSingleLoopS2SizeAlign;
    uint64_t actS2Size;
    bool isFirstS2Loop, isLastS2Loop, isS2SplitCore;
};

struct RunInfoMisc {     // 冷字段，仅特殊路径用
    uint64_t actS1Size;
    int64_t preTokensLeftUp, nextTokensLeftUp;
    uint32_t faTmpOutWsPos;
    // ...
};
```

2. 热路径函数签名改为按值传递小子结构体：

```cpp
void ProcessVec1Nd(..., RunInfoCore ri, RunInfoS2 rs)
// 按值传递 → 无别名顾虑 → 编译器激进优化
```

**预期收益**：热路径函数参数传递开销降低，编译器可做更激进的寄存器分配

---

## 优化项 5：FlashAttention 主循环 — 状态机改显式循环 [P3] + 主尾块分离 [P2]

**严重度**：🟠 高优先级
**涉及文件**：`flash_attn_kernel_base_gqa_template.h:390-446`

**现状问题**：

```cpp
while (shouldDispatchTask || validTaskCount) {
    shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
    if (shouldDispatchTask) {
        TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
        if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) { ... }
        else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) { ... }
        else { ... }
    }
    if (validTaskCount) {
        ExecuteTask(executedTaskCount, taskRunInfo);
        ...
    }
}
```

这是一个隐式状态机驱动的 while 循环，每轮迭代执行 `ShouldDispatchTask` + `GetTaskDealMode`（内部含 5+ 层 if 级联判断），Scalar 分支预测频繁失败。

**修改方案**：

1. 将 GetTaskDealMode 中的循环无关判断外提：actSeqLensKv/actSeqLensQ 仅在换 batch 时重算（已有缓存），但 s2LoopTimes/gS1LoopTimes 每轮都重算 → 改为换行时缓存

2. UpdateAxisInfo 中的 s2LoopTimes/gS1LoopTimes 重复计算（与 GetTaskDealMode 完全相同）→ 合并为一次计算，结果存入局部变量

3. 对 DEAL_ZERO/SKIP/S2_END 等低频路径用 `unlikely()` 标注，确保热路径（CREATE_TASK）的指令连续排布

4. 评估将 while 循环改为 for 循环的可行性：预计算总任务数 totalTasks，用 `for (uint32_t t = 0; t < totalTasks; ++t)` 回跳走专用硬件，不经分支预测器

**预期收益**：消除每轮迭代 2 次重复的 64-bit 除法（`/s2BaseSize`、`/mBaseSize`），减少分支预测失败 ~30%

---

## 优化项 6：CalcParams 热路径内冗余计算消除 [P5][P9]

**严重度**：🟡 中优先级
**涉及文件**：`flash_attn_kernel_base_gqa_template.h:665-742`

**现状问题**：

- `CalcParams` 每轮都计算 `info.bIdx = bN2Cur / constInfo.n2Size` 和 `info.n2Idx = bN2Cur % constInfo.n2Size`（64-bit 除法 + 取模）
- `GetPreNextTokenLeftUp` 每轮都被调用，但结果仅在换 batch 时变化
- `info.actMSizeAlign32`、`info.actVecMSize` 等可通过位运算替代

**修改方案**：

1. bIdx/n2Idx 仅在换 batch 时更新（与 GetTaskDealMode 中的 prevBIdx 判断合并），避免每轮除法

2. GetPreNextTokenLeftUp 结果缓存到成员变量，仅在 actSeqLensQ/actSeqLensKv 变化时重算

3. actMSizeAlign32 = (actMSize + 31) >> 5 << 5 已用位运算 ✓；actVecMSize = actMSize <= 16 ? actMSize : (actMSizeAlign32 >> 1) → 可进一步简化为条件 move 指令

**预期收益**：每轮 task 节省 2 次 64-bit 除法 + 1 次函数调用开销

---

## 优化项 7：tilingData 多级指针解引用消除 [P6][P8]

**严重度**：🟡 中优先级
**涉及文件**：`flash_attn_kernel_base_gqa_template.h:301-350`

**现状问题**：

```cpp
constInfo.bSize = this->tilingData->flashAttnBaseParams.bSize;
constInfo.t1Size = this->tilingData->flashAttnBaseParams.t1Size;
// ... 20+ 行类似的链式访问
```

`tilingData` 是 `__gm__` 指针，每次 `->` 解引用产生 GM Load 延迟（数百 cycle）。虽然仅在 Init 阶段执行，但 Init 中 20+ 次 GM Load 串行累加。

**修改方案**：

1. 一次性将 tilingData->flashAttnBaseParams 整体拷贝到局部结构体：

```cpp
FlashAttnBaseParams baseParams = *(tilingData->flashAttnBaseParams);
// 然后从局部结构体读取各字段
constInfo.bSize = baseParams.bSize;
```

2. 进一步：将 baseParams 中编译期可知的字段（如 dSize=128 对于 config=0）改为模板参数传递，参与常量折叠

**预期收益**：Init 阶段 GM Load 从 20+ 次降为 1 次批量拷贝

---

## 优化项 8：Event ID 数组拆为标量 [P1]

**严重度**：🟡 中优先级
**涉及文件**：`flash_attn_block_vec_noquant_gqa.h:171-172`

**现状问题**：

```cpp
TEventID mte3ToVId[2];
TEventID vToMte3Id[2];
```

通过 `mte3ToVId[runInfo.loop & (DB - 1)]` 动态下标访问。虽然已用位掩码，但数组形式仍让编译器对同结构体其他成员放弃常量传播。

**修改方案**：

```cpp
TEventID mte3ToVId_0, mte3ToVId_1;
TEventID vToMte3Id_0, vToMte3Id_1;
// 使用时：
auto& curMte3ToV = (loop & 1) ? mte3ToVId_1 : mte3ToVId_0;
```

**预期收益**：消除数组动态下标对别名分析的污染

---

## 优化项 9：ProcessVec1Nd/ProcessVec2 热路径内对象构造消除 [P9]

**严重度**：🟡 中优先级
**涉及文件**：`flash_attn_block_vec_noquant_gqa.h:499-623`, `625-665`

**现状问题**：

- `ProcessVec1Nd` 内每轮构造多个 `LocalTensor`（`sumUb`、`maxUb`、`expUb`、`mmRes`、`stage1CastTensor` 等），每个构造展开为初始化 Store
- `ProcessVec2_C2GmUpdateUb` 内循环每轮构造 `DataCopyParams`、`DataCopyPadParams`

**修改方案**：

1. LocalTensor 对象在循环外构造一次，循环内复用：

```cpp
LocalTensor<float> sumUb = softmaxSumBuf[...].Get<float>();
for (...) {
    // 直接使用 sumUb，不再每轮重新构造
}
```

2. DataCopyParams/DataCopyPadParams 提到循环外初始化，循环内仅修改变化字段

**预期收益**：减少热路径中每轮 ~10+ 个 Store 指令

---

## 优化项 10：UpdateAxisInfo 冗余计算合并 [P4][P5]

**严重度**：🟡 中优先级
**涉及文件**：`flash_attn_kernel_base_gqa_template.h:744-776`

**现状问题**：

```cpp
uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
```

这两个值在 `GetTaskDealMode`（:474-476）和 `UpdateAxisInfo`（:747-749）中**完全重复计算**，且每轮迭代都执行 64-bit 除法。

**修改方案**：

1. 在换 batch 时（actSeqLensKv/actSeqLensQ 变化时）计算一次并缓存：

```cpp
cachedS2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
cachedGS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
```

2. GetTaskDealMode 和 UpdateAxisInfo 共用缓存值

**预期收益**：每轮迭代消除 2 次 64-bit 除法（~20-40 cycle/次）

---

## 实施优先级与预期收益汇总

| 优先级 | 优化项 | 对应原则 | 预期 Scalar cycle 降幅 | 实施难度 |
|--------|--------|---------|----------------------|---------|
| **P0** | 1. ConstInfo_t 拆分 | P7 | 15-20% | 中 |
| **P0** | 2. constInfo→局部变量 | P4 | 15-25% | 中 |
| **P0** | 5. 主循环状态机改造 | P2/P3 | 10-15% | 高 |
| **P1** | 3. softmax Buffer 位掩码 | P1 | 5-8% | 低 |
| **P1** | 4. RunInfoX 拆分 | P6/P7 | 8-12% | 中 |
| **P1** | 10. 冗余计算合并 | P4/P5 | 5-10% | 低 |
| **P2** | 6. CalcParams 优化 | P5/P9 | 3-5% | 低 |
| **P2** | 7. tilingData 解引用 | P6/P8 | 2-3%（Init 阶段） | 低 |
| **P2** | 8. Event ID 拆标量 | P1 | 2-3% | 低 |
| **P2** | 9. 热路径对象构造 | P9 | 3-5% | 低 |

**综合预期**：Scalar cycle 从 7w 降至 ~3-4w（降幅 40-50%），总 cycle 从 8w 降至 ~4-5w。

---

## 实施计划

1. **第一批（P0，预计 2-3 天）**：优化项 1 + 2 + 10 → 结构体瘦身 + 局部变量提升 + 冗余消除，这三项互相协同，一起做效果最明显
2. **第二批（P1，预计 2 天）**：优化项 3 + 4 + 5 → Buffer 索引 + RunInfoX 拆分 + 主循环改造
3. **第三批（P2，预计 1-2 天）**：优化项 6~9 → 细节打磨

每批完成后建议跑 profiling 对比 Scalar cycle 占比，验证优化效果。

---

## 附录：参考的 Scalar 优化 9 条编码原则

| 编号 | 原则 | 一句话核心 |
|------|------|-----------|
| P1 | 结构体内慎用数组 | 动态下标的结构体数组会让编译器放弃整个结构体的常量传播 |
| P2 | 循环主尾块分离 | Prologue / Hot Loop（零分支）/ Epilogue 三段式，避免热路径每轮 3 次分支判断 |
| P3 | 显式编写循环代码 | 用嵌套 `for` 替代隐式状态机；`for` 的回跳走专用硬件，不进分支预测 |
| P4 | 尽量使用局部变量 | 成员变量受别名分析压制；跨函数调用后必须从内存重新 Load |
| P5 | 变量定义贴近使用位置 | 缩短活跃范围 = 缩短寄存器占用 = 降低 Spill 概率 |
| P6 | 避免多级指针解引用 | 每级解引用都是带依赖的 Load，用值类型聚合体替代 |
| P7 | 避免使用超大结构体 | > 64B 跨 cacheline；超寄存器容量必 Spill；指针传递引入别名 |
| P8 | 用 constexpr / 模板参数承载编译期常量 | const 成员对编译器仍是 runtime Load；进入常量折叠才能链式带动 P1 类优化 |
| P9 | Hot Loop 内不构造对象、不取地址 | 构造/析构展开后是大量初始化 Store；取地址会让编译器假设变量可被外部修改、强制 Spill |

> 来源：`/mnt/workspace/skill/cann-skills-zyh/ops/ascendc-performance-best-practices/reference/scalar/`
