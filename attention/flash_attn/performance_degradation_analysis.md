# Flash Attention Scalar Bound 优化劣化根因分析

> **分析日期**：2026-06-04
> **分析对象**：`scalar_bound_optimization_plan.md` 中 10 个优化 Step 全部导致性能劣化
> **结论**：优化方向错误 — 做加法（增加结构体/成员/参数）而非做减法（constexpr/去抽象/减指令）

---

## 一、问题总结

| 现象 | 10 个 Step 全部导致性能劣化 |
|------|---------------------------|
| 根因 | 优化方向与 Ascend AICore 架构特性相悖 |
| 核心矛盾 | Ascend 寄存器极少，所有"加法"优化都增加寄存器压力，导致更多 Spill |

---

## 二、逐项劣化原因分析

### Step 1: ConstInfo_t 超大结构体拆分 → 劣化

**修改**：新增 `HotConstInfo` + `ColdConstInfo` 成员，原始 `constInfo` 未删除

**劣化原因**：
1. **数据冗余**：`constInfo` 仍存在，`hotConst` 是额外副本，类总大小增加 ~48B
2. **Init 开销增加**：`InitConstInfo()` 末尾新增 12+ 次 `hotConst.xxx = constInfo.xxx` 赋值
3. **编译器困惑**：两个结构体指向相似数据，别名分析更保守
4. **寄存器压力**：新增成员变量占用本就稀缺的 Scalar 寄存器

**违反原则**：P7（避免超大结构体）的初衷是减小结构体，但实际效果是增加了总数据量

---

### Step 2: FlashAttention 入口局部变量提升 → 劣化

**修改**：`CreateTask`/`CalcParams` 签名增加 5+ 个参数，入口处拷贝热路径字段到局部变量

**劣化原因**：
1. **参数传递开销**：Ascend 寄存器少，多参数传递展开为大量 Store/Load
2. **函数调用膨胀**：每次调用 `CreateTask`/`CalcParams` 都要压栈 5+ 个参数
3. **内联决策改变**：参数增多可能导致编译器放弃内联，产生额外调用开销
4. **局部变量 Spill**：新增的局部变量在寄存器紧张时被 Spill 到栈

**违反原则**：P4（局部变量）在 x86/ARM 上有效，但 Ascend 寄存器太少，局部变量过多反而 Spill

---

### Step 3: Softmax Buffer 位掩码索引 → 劣化

**修改**：Buffer 深度从 3 抬到 4，索引从 `% 3` 改为 `& 3`

**劣化原因**：
1. **UB 预算超限**：多占 768B UB（3 个 TBuf × 256B），挤压其他 buffer 空间
2. **UB 复用开销**：UB 紧张时编译器需要更多地址计算来复用 buffer
3. **取模优化有限**：`% 3` 的开销被高估，bisheng 编译器可能已有优化

**违反原则**：P1（数组索引）的收益被 UB 预算超限的代价抵消

---

### Step 4: RunInfoX 拆分 + 按值传递 → 劣化

**修改**：`RunInfoX` 拆为 `RunInfoCore`(~48B) + `RunInfoS2`(~24B) + `RunInfoMisc`，按值传递

**劣化原因**：
1. **按值拷贝开销巨大**：每次调用拷贝 ~72B 到栈上，展开为 18+ 条 Store 指令
2. **Ascend 无别名优化收益**：bisheng 编译器的别名分析本就保守，按值传递的优化收益有限
3. **结构体切片开销**：`RunInfoX` 继承自三个子结构体，隐式切片传递需要额外拷贝
4. **冷路径重建开销**：冷路径调用 `SoftmaxDataCopyOut` 时需从 `ri`/`rs` 重建 `RunInfoX`

**违反原则**：P6/P7 的按值传递在寄存器充足的架构上有效，Ascend 上拷贝开销 > 别名收益

---

### Step 5: Vec Block 热路径局部变量 → 劣化

**修改**：`ProcessVec1Nd`/`ProcessVec1Dn`/`ProcessVec2DSplit` 等函数入口添加局部变量

**劣化原因**：
1. **寄存器压力叠加**：与 Step 2 类似，新增局部变量在寄存器紧张时被 Spill
2. **函数体膨胀**：每个函数入口增加拷贝代码，I-Cache 压力增加
3. **bisheng 内联失败**：函数体过大可能导致编译器放弃内联

---

### Step 6: Cube Block 热路径局部变量 → 劣化

**修改**：`IterateBmm1`/`IterateBmm2` 系列函数添加 `localDSize`/`localDSizeV`

**劣化原因**：同 Step 5，局部变量过多导致 Spill

---

### Step 7: tilingData 多级指针解引用消除 → 劣化

**修改**：`InitConstInfo` 中批量拷贝 `FlashAttnBaseParams` 等结构体到栈上

**劣化原因**：
1. **Init 阶段非热路径**：Init 只执行一次，优化收益极小
2. **栈空间开销**：~124B 栈空间在 Ascend 上可能导致栈溢出风险
3. **批量拷贝开销**：结构体拷贝展开为大量 Store，可能比原始 GM Load 更慢
4. **编译器已优化**：bisheng 可能对 `tilingData->xxx` 做了缓存优化

---

### Step 8: Event ID 数组拆为标量 → 劣化

**修改**：`mte3ToVId[2]` 拆为 `mte3ToVId_0`/`mte3ToVId_1`，动态索引改为三目运算

**劣化原因**：
1. **寄存器占用增加**：4 个标量变量 vs 2 个数组元素，寄存器占用不变或增加
2. **三目运算开销**：`(cond) ? a : b` 在 Ascend 上可能生成条件分支，比数组索引更慢
3. **别名分析收益有限**：bisheng 的别名分析对数组/标量的处理差异不大

---

### Step 9: 热路径对象构造外提 → 劣化

**修改**：`DataCopyParams`/`DataCopyPadParams` 从循环内移到循环外

**劣化原因**：
1. **bisheng 已优化**：编译器可能已识别循环不变量并外提
2. **条件判断开销**：循环外增加 `if (constInfo.dSizeV != dTemplateAlign64)` 判断
3. **代码膨胀**：循环外初始化代码增加 I-Cache 压力

---

### Step 10: UpdateAxisInfo 冗余计算合并 → 劣化

**修改**：新增 `cachedS2LoopTimes`/`cachedGS1LoopTimes` 成员变量，batch 切换时缓存

**劣化原因**：
1. **新增成员变量**：类大小增加 16B，寄存器压力增加
2. **缓存更新开销**：batch 切换时的缓存写入可能比原始计算更慢
3. **分支预测干扰**：缓存逻辑引入新分支，干扰分支预测器
4. **64-bit 除法开销被高估**：bisheng 可能对除法有硬件加速或特殊优化

---

## 三、根本原因：优化方向错误

### 3.1 核心矛盾

**Ascend AICore 的 Scalar 寄存器数量极其有限**（远少于 x86/ARM），所有"加法"优化都增加寄存器压力：

| 优化类型 | x86/ARM 效果 | Ascend 效果 |
|---------|-------------|-------------|
| 增加成员变量 | 可接受 | 寄存器 Spill 增加 |
| 增加局部变量 | 寄存器充足可驻留 | Spill 到栈 |
| 按值传递 | 拷贝开销小 | 大量 Store 指令 |
| 结构体拆分 | 缓存命中提升 | 总数据量增加 |

### 3.2 Conv 成功案例的启示

参考 `conv_scalar.md`，Conv 的成功优化是**递进式减法**：

```
固定循环轴（删除不必要的循环）
  → 去除 Queue/TBuffer/Tpipe（用底层原语替代高级抽象）
    → TilingData constexpr（编译期常量化，消除所有参数 Load）
      → Scalar Load/Store 从 90%+ 降到 <10%
```

而 FA 的 10 个 Step 是**递进式加法**，每步都增加复杂度，叠加后效果更差。

### 3.3 bisheng 编译器特性

bisheng 针对 Ascend C SDK 的**特定代码模式**做了优化：

1. **模式匹配**：原始 FA 代码是华为工程师精心调优的，编译器已能识别其模式
2. **内联决策**：函数签名变化可能导致内联决策改变
3. **别名分析**：bisheng 的别名分析本就保守，按值传递的收益有限
4. **常量传播**：bisheng 对 `const` 成员的处理可能比预期更好

---

## 四、正确的优化方向

### 4.1 方向 A：模板参数常量化（P8，最高收益潜力）

FA 已有模板参数 `Config`（0 或 1），把热路径中编译期可知的值改为模板参数：

```cpp
// 不要这样（运行时 Load）：
const uint32_t dSize = constInfo.dSize;

// 要这样（编译期常量）：
static constexpr uint32_t dSize = (Config == 0) ? 128 : 128;
static constexpr uint32_t s1BaseSize = (Config == 0) ? 128 : 64;
static constexpr uint32_t s2BaseSize = (Config == 0) ? 128 : 256;
```

**收益**：
- 编译器做常量折叠，链式消除大量地址计算
- 不增加任何成员变量或寄存器压力
- 循环可被完全展开

### 4.2 方向 B：去除热路径中的高级抽象

在 ProcessVec/IterateBmm 的热循环内，把 `TQue`/`TBuf` 替换为 `SetFlag/WaitFlag` + `LocalTensor`：

```cpp
// 原模式（Queue 抽象，展开后大量 Scalar 指令）
auto tensor = que.DeQue<T>();

// 新模式（底层原语，零 Scalar 开销）
SetFlag<HardEvent::V_MTE3>(event_id);
WaitFlag<HardEvent::MTE3_V>(event_id);
LocalTensor<half> tensor(TPosition::VECIN, offset, size);
```

**收益**：
- 消除 Queue 状态查询、缓冲区索引更新、自动同步插入
- Scalar Load/Store 占比可从 >30% 降到 <15%

### 4.3 方向 C：减少热路径总指令数

- 检查 `GetTaskDealMode` 中的分支级联，能否通过查表或位运算替代
- 检查 `CalcParams` 中是否有可以提前计算并硬编码的路径
- 对特定 Config 做函数特化（`if constexpr (Config == 0)` 消除不需要的分支）

### 4.4 方向 D：场景特化

参考 Conv 的"固定循环轴"思路：
- 分析 FA 的典型使用场景（如 GQA 的 head 数量、seq_len 范围）
- 对高频场景做函数特化，消除不必要的分支和循环

---

## 五、验证方法建议

### 5.1 汇编级验证

每个改动后，用 bisheng 的 `-S` 选项输出汇编，对比：
- Scalar Load/Store 指令数量
- 寄存器 Spill 次数
- 热路径总指令数

**不要只看 cycle 数字，要看指令级差异。**

### 5.2 增量验证

每次只做一个方向的改动，编译+profile 验证后再继续。不要叠加多个改动。

### 5.3 对比基线

与原始代码（未优化版本）对比，确保每个改动都有正向收益。

---

## 六、总结

| 问题 | 原因 | 解决 |
|------|------|------|
| 全部劣化 | 做加法（增加 struct/成员/参数），增加了寄存器压力 | 做减法（constexpr、去抽象、减指令） |
| 结构体拆分无效 | 原始 constInfo 未删除，反而多了一份 hotConst | 不拆结构体，改为让编译器更好地利用现有结构体 |
| 缓存变量适得其反 | 新增成员变量增加了类大小和 Spill | 用编译期常量替代运行时缓存 |
| 按值传递劣化 | Ascend 寄存器少，大参数拷贝开销 > 别名消除收益 | 保持引用传递，用 `__restrict__` 或编译器 hint 辅助别名分析 |

**核心建议**：
1. 回退所有修改，回到性能基线
2. 转向 P8（模板参数常量化）+ Conv 式减法优化路径
3. 用汇编级验证替代 cycle 级验证
4. 增量验证，每次只做一个方向的改动

---

## 附录：参考材料

- `/mnt/workspace/skill/cann-skills-zyh/ops/ascendc-performance-best-practices/reference/scalar/coding_principles.md` — 9 条编码原则
- `/mnt/workspace/skill/cann-skills-zyh/ops/ascendc-performance-best-practices/reference/scalar/conv_scalar.md` — Conv 成功案例
- `/mnt/workspace/skill/cann-skills-zyh/ops/ascendc-performance-best-practices/reference/scalar/guide.md` — 策略总览
