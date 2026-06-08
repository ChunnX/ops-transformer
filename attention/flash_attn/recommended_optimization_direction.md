# Flash Attention Scalar Bound 优化 — 推荐方向

> **日期**：2026-06-04
> **基于**：`performance_degradation_analysis.md` 的劣化分析 + Conv 成功案例

---

## 已实施的安全改动

### 显式位移替代编译期除法

**文件**：`flash_attn_kernel_base_gqa_template.h`

将热路径中除以编译期 2 的幂常量的运算改为显式位移：

```cpp
// 新增 constexpr 辅助
static constexpr uint32_t CeilLog2(uint32_t v) {
    return (v <= 1) ? 0 : 1 + CeilLog2(v >> 1);
}
static constexpr uint32_t S2_SHIFT = CeilLog2(s2BaseSize); // 7 or 8
static constexpr uint32_t M_SHIFT = CeilLog2(mBaseSize);   // 6 or 7

// 热路径替换
// 原: (actSeqLensKv + s2BaseSize - 1) / s2BaseSize
// 新: (actSeqLensKv + s2BaseSize - 1) >> S2_SHIFT
```

**影响函数**：`GetTaskDealMode`、`UpdateAxisInfo`、`CalcCurS2StartEndNoSparse`、`CalcCurS2StartEndWithSparse`

**风险**：极低（纯机械替换，bisheng 大概率已做此优化，显式写法确保不被遗漏）

---

## 推荐方向 1：TQue/TBuffer 替换为 SetFlag/WaitFlag + LocalTensor

### 为什么这是最高收益方向

参考 `conv_scalar.md`，Conv 算子通过此方法将 Scalar Load/Store 占比从 >30% 降到 <15%。

FA 的 Vec Block 热路径中大量使用 `TQue`/`TBuf` 高级抽象：

```cpp
// 当前模式（每轮展开为大量 Scalar 指令）
auto sumUb = softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
auto maxUb = softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
auto expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
```

每个 `TBuf::Get<T>()` 展开后包含：
- Queue 状态查询（Load）
- Buffer offset 计算（ALU + Load）
- Buffer 索引更新（Store）

### 替换方案

```cpp
// 新模式（零 Scalar 开销）
constexpr uint32_t SOFTMAX_BUF_OFFSET = 0;  // 预计算偏移
SetFlag<HardEvent::V_MTE3>(vToMte3Id_0);
WaitFlag<HardEvent::MTE3_V>(mte3ToVId_0);
LocalTensor<float> sumUb(TPosition::VECCALC, sumBufOffset, 256 / sizeof(float));
LocalTensor<float> maxUb(TPosition::VECCALC, maxBufOffset, 256 / sizeof(float));
LocalTensor<T> expUb(TPosition::VECCALC, expBufOffset, 256 / sizeof(T));
```

### 实施步骤

1. **分析 buffer 生命周期**：梳理 `softmaxMaxBuf`/`softmaxSumBuf`/`softmaxExpBuf` 的读写时序
2. **计算固定偏移**：将 TBuf 的 TPposition + offset 提取为 constexpr
3. **替换同步原语**：将隐式 Queue 同步改为显式 SetFlag/WaitFlag
4. **逐函数替换**：从 `ProcessVec1Nd` 开始，逐函数替换并 profile 验证

### 风险与注意事项

- **高风险**：需要精确理解 buffer 的读写时序和同步关系
- **需要汇编验证**：每次改动后用 `bisheng -S` 对比 Scalar 指令数
- **渐进式替换**：不要一次性替换所有 buffer，先替换一组（如 softmaxSumBuf）验证效果

---

## 推荐方向 2：运行时除法 → 乘法近似（libdivide 思路）

### 问题

`CalcParams` 中每轮执行：

```cpp
info.bIdx = bN2Cur / constInfo.n2Size;   // 运行时除法
info.n2Idx = bN2Cur % constInfo.n2Size;  // 运行时取模
```

`constInfo.n2Size` 是运行时值（来自 tilingData），编译器无法优化为位移。

### 方案

在 `InitConstInfo` 中预计算倒数乘法因子：

```cpp
// Init 阶段（只执行一次）
struct FastDiv {
    uint64_t magic;
    uint32_t shift;
};

FastDiv n2Div;  // 预计算

void InitConstInfo() {
    // ... existing code ...
    n2Div = ComputeFastDiv(constInfo.n2Size);
}

// 热路径（每轮执行）
info.bIdx = FastDivide(bN2Cur, n2Div);
info.n2Idx = bN2Cur - info.bIdx * constInfo.n2Size;  // 用乘法替代取模
```

### 关键约束

- `FastDiv` 结构体只增加 ~12B 成员变量
- `ComputeFastDiv` 只在 Init 阶段执行一次
- `FastDivide` 展开为 1 次乘法高位 + 1 次移位（无除法指令）
- 需要验证 Ascend 是否支持 `umulhi`（无符号乘法高位）指令

### 风险

- **中风险**：需要验证 Ascend 指令集支持
- **不增加寄存器压力**：FastDivide 的临时变量生命周期极短
- **不改变代码结构**：只替换除法实现，不改变函数签名或数据布局

---

## 推荐方向 3：场景特化 — 消除不需要的分支

### 问题

`ProcessVec1Nd` 中有大量基于 `actSingleLoopS2Size` 的分支：

```cpp
if (runInfo.isFirstS2Loop) {
    if (runInfo.actSingleLoopS2Size == 128) { ... }
    else if (runInfo.actSingleLoopS2Size <= 64) { ... }
    else if (runInfo.actSingleLoopS2Size <= 128) { ... }
    else { ... }
} else {
    if (runInfo.actSingleLoopS2Size == 128) { ... }
    else { ... }
}
```

每个分支展开为不同的 Vector API 调用序列，导致 I-Cache 压力大、分支预测失败率高。

### 方案

对高频场景（如 `actSingleLoopS2Size == s2BaseSize`，即非尾块）做函数特化：

```cpp
// 特化版本：s2Size == s2BaseSize（非尾块，最高频）
__aicore__ inline void ProcessVec1Nd_HotPath(...) {
    // 零分支，直接按 s2BaseSize 调用
    ProcessVec1Vf<s2BaseSize>(...);
}

// 通用版本：处理所有边界情况
__aicore__ inline void ProcessVec1Nd_ColdPath(...) {
    // 原有的多分支逻辑
}

// 分发
if (likely(runInfo.actSingleLoopS2Size == s2BaseSize && !runInfo.isFirstS2Loop)) {
    ProcessVec1Nd_HotPath(...);
} else {
    ProcessVec1Nd_ColdPath(...);
}
```

### 收益

- 热路径零分支 → 分支预测器不介入
- 热路径代码体积小 → I-Cache 命中率高
- 冷路径不影响性能（执行频率低）

### 风险

- **低风险**：不改变数据布局，不增加成员变量
- **代码膨胀**：需要维护两个版本的函数
- **需要 profile 验证**：确认热路径确实是非尾块场景

---

## 实施优先级

| 优先级 | 方向 | 预期收益 | 实施难度 | 风险 |
|--------|------|---------|---------|------|
| **P0** | 已实施：显式位移 | 0-2% | 已完成 | 极低 |
| **P1** | TQue/TBuffer 替换 | 15-30% | 高 | 高 |
| **P2** | 运行时除法→乘法 | 3-5% | 中 | 中 |
| **P3** | 场景特化 | 5-10% | 中 | 低 |

---

## 验证方法

### 必须做的验证

1. **汇编对比**：每次改动前后，用 `bisheng -S` 输出汇编，对比：
   - Scalar Load/Store 指令数量
   - 热路径总指令数
   - 寄存器 Spill 次数

2. **增量验证**：每次只做一个方向的改动，编译 + profile 后再继续

3. **回归测试**：确保功能正确性（数值精度、边界条件）

### 关键指标

- **Scalar cycle 占比**：目标从 87.5% 降到 <50%
- **Scalar Load/Store 占比**：目标从 >30% 降到 <15%
- **总 cycle**：目标从 8w 降到 <5w

---

## 核心原则

1. **做减法不做加法**：减少指令数，而非重排数据
2. **不增加成员变量**：Ascend 寄存器极少，每个新成员都增加 Spill 风险
3. **不改变函数签名**：避免影响 bisheng 的内联决策和模式匹配
4. **编译期常量优先**：能用 constexpr 解决的不用运行时变量
5. **汇编级验证**：不信任 cycle 数字，看指令级差异
