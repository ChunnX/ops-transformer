# Elementwise 逐元素类 — 未收录模型

> 当前未收录完整 Elementwise Tiling 模型。命中本类别时返回“暂未收录”，再按通用 DataCopy、memory 和 pipeline 检查设计实验。

## 适用算子

Sin, Cos, Abs, Exp 及逐元素激活函数

## 未收录范围

- 多核切分策略（沿总元素数均匀切分）
- 单核切分策略（UB 容量 / 单元素大小 → 单次处理量）
- Buffer 规划（inBuf / outBuf）
- 分支覆盖

---

> 当前返回：「Elementwise Tiling 建模暂未收录。」
