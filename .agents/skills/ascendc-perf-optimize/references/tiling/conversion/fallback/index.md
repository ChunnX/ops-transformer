# Conversion 数据转换类 — 未收录模型

> 当前未收录完整 Conversion Tiling 模型。命中本类别时返回“暂未收录”，再按通用 DataCopy、memory 和 pipeline 检查设计实验。

## 适用算子

Transpose, Concat, Split

## 未收录范围

- 多核切分策略（Transpose 沿输出维度 / Concat 按输入段 / Split 按输出段）
- 单核切分策略（block 为单位搬移，考虑跨 stride 对齐）
- Buffer 规划
- 分支覆盖

---

> 当前返回：「Conversion Tiling 建模暂未收录。」
