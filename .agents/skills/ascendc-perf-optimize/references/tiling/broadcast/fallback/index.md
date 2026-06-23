# Broadcast 广播类 — 未收录模型

> 当前未收录完整 Broadcast Tiling 模型。命中本类别时返回“暂未收录”，再按通用 DataCopy、memory 和 pipeline 检查设计实验。

## 适用算子

Add, Mul, Sub 等含广播语义的二元逐元素算子

## 未收录范围

- 多核切分策略（沿最大维度输出 shape 均匀切分）
- 单核切分策略（考虑广播维度的输入复用）
- Buffer 规划（大张量 + 广播张量分别规划）
- 分支覆盖

---

> 当前返回：「Broadcast Tiling 建模暂未收录。」
