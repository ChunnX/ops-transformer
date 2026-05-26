# 为什么需要 SparseAttnSharedkvMetadata？——拆解设计逻辑

## 一句话概括

SparseAttnSharedkvMetadata 是一个"任务调度器"，它根据每个 batch 的实际序列长度和稀疏 mask 模式，用代价模型算出每个计算核应该处理哪一段 (BN2, gS1, S2) 的注意力计算，输出一张任务分配表 (metadata) 给 SparseAttnSharedkv 的每个核读取执行。

---

## 为什么不能把分核逻辑直接塞进 SparseAttnSharedkv？

核心原因有三个：

### 1. 硬件架构限制——AIC 核无法做复杂调度

SparseAttnSharedkv 跑在 Ascend A3 的 AICore 上（36 个 AIC Cube 核 + 72 个 AIV Vector 核），AIC 核是面向矩阵乘的流式处理器，没有能力运行复杂的分支逻辑和代价估算。而分核算法需要：

- 遍历每个 batch 的实际序列长度（变长）
- 根据 mask 模式（band/causal）计算每个 query 行能看到的 KV token 范围
- 将 win（滑动窗口）和 cmp（压缩）两部分的 KV 长度分别计算
- 用代价模型 cost = 6*ceil(M/16) + 10*ceil(S2/64) 估算每块计算量
- 三级贪心分配（batch → row → block）实现负载均衡
- 如果 S2 被切到多个核，还要规划 FlashDecode 归约任务

这些逻辑只能在 AICPU（CPU 侧）上执行，不可能放在 AICore kernel 里。

### 2. 变长序列 + 稀疏 mask 导致计算量极度不均匀

不同于标准 FlashAttention（每个 batch 的计算量大致正比于 s1*s2），这里：

- 各 batch 的 s1、s2 可能不同（变长推理）
- band mask 让每个 query 行只看 win_left=127 个近邻 KV token
- cmp 部分的长度 = s2 / cmp_ratio（压缩比 4 或 128）
- 不同行的 win/cmp 分界点不同，代价差异巨大

如果不做精细的负载均衡，某些核可能处理 128 个 KV token，另一些核处理 8000+，导致严重的长尾延迟。

### 3. FlashDecode 归约需要提前规划 workspace

当一行 gS1 的 S2 维度被切到多个 AIC 核时（block 级分配），每个核产出的部分 softmax 结果需要 AIV 核做归约（FlashDecode）。这需要：

- 提前知道哪些行被切分了（curKvSplitPart > 1）
- 在 workspace 中为每个切分预留存储位置
- 将归约任务均衡分配给 72 个 AIV 核

这些信息必须在 kernel 启动前就确定，不能运行时动态分配。

---

## 切分逻辑详解——为什么这么切

切分在 BN2 × gS1 × S2 三维空间上进行：

| 维度 | 含义 | 基本块大小 |
|------|------|------------|
| BN2 | batch × kvHeadNum（因为 kvHeadNum=1，即 batch） | 整个 batch |
| gS1 | (s1 × groupSize) / mBaseSize，即 M 轴按 GQA group 切 | mBaseSize = groupSize = 64 |
| S2 | KV 序列长度 | s2BaseSize = 512 |

### 三级贪心分配策略（粗→细）

1. **AssignByBatch**: 整个 batch 分给一个核
   - 零跨核通信，最优，但 batch 可能太大

2. **AssignByRow**: 单个 gS1 行分给一个核
   - 每行内部完整，无需 FlashDecode

3. **AssignByBlock**: 单个 S2=512 块分给一个核
   - 行被拆到多个核，需要 FlashDecode 归约
   - 这是最后手段，因为引入了额外的规约开销

### 代价模型

cost = 6*ceil(M/16) + 10*ceil(S2/64)

这个公式模拟了 Cube 核上 MM1（Q@K^T）和 MM2（Softmax@V）的执行时间，考虑了 16/64 的对齐开销。每核的代价上限 = 剩余总代价 / 剩余核数，容差系数为 2（FA_TOLERANCE_RATIO）。

---

## Metadata 的数据布局

输出固定为 [1024] 的 INT32 张量，结构如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│ FA Metadata: [36 cores × 8 fields] = 288 int32  ← AIC 核读取        │
│   每核: [coreEnable, bn2Start, mStart, s2Start,                    │
│          bn2End, mEnd, s2End, fdWorkspaceIdx]                      │
├─────────────────────────────────────────────────────────────────────┤
│ FD Metadata: [72 cores × 8 fields] = 576 int32  ← AIV 核读取        │
│   每核: [coreEnable, bn2Idx, mIdx,                                  │
│          workspaceIdx, workspaceNum,                               │
│          mStart, mNum, reserved]                                   │
└─────────────────────────────────────────────────────────────────────┘
总计 864 int32 ≤ 1024
```

SparseAttnSharedkv 的每个核启动时，用 GetAttrAbsIndex(coreIdx, metaIdx) 从这个数组中读取自己的任务边界，然后只处理 [bn2Start, bn2End) × [gS1Start, gS1End) × [s2Start, s2End) 范围内的计算。

---

## 总结

| 问题 | 回答 |
|------|------|
| 为什么需要 Metadata 算子？ | AIC 核无法做复杂调度；变长+稀疏导致负载极度不均；FlashDecode 需要提前规划 workspace |
| 为什么这么切分？ | 三级贪心（batch→row→block）在最小化跨核通信和最大化负载均衡之间取平衡 |
| 为什么 S2 固定切 512？ | 匹配 Cube 核 MM1/MM2 的 S2 对齐粒度（64 的倍数），也是 KV cache 的自然分块大小 |
| 为什么 M 切 groupSize？ | GQA 架构下同一 group 的 Q 共享 KV，按 group 切分保证每次 MM1 的 K/V 完整 |
| Metadata 为什么固定 1024？ | 36 AIC + 72 AIV = 108 核 × 8 字段 = 864，预留余量到 1024 |