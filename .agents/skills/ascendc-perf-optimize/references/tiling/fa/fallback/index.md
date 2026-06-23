# FA FlashAttention 类 — 未收录模型

> 当前未收录完整 FA/attention Tiling 模型。命中本类别时返回“暂未收录”，再结合本仓库 attention 算子的实际 host tiling、mask/cache/layout 分支设计实验。

## 适用算子

FlashAttention, GroupNorm

## 未收录范围

- 多核切分策略（FlashAttention: Batch × Head / GroupNorm: Batch 切分 + Group 内归约）
- 单核切分策略
- Buffer 规划
- 分支覆盖

---

> 当前返回：「FA Tiling 建模暂未收录。」
