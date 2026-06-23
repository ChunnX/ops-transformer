# Reduction 归约类 — 未收录模型

> 当前未收录完整 Reduction Tiling 模型。命中本类别时返回“暂未收录”，再按通用 reduction、DataCopy、memory 和 pipeline 检查设计实验。

## 适用算子

ReduceSum, Softmax, LayerNorm, ArgMax

## 未收录范围

- 多核切分策略（沿非归约轴切分）
- UB 切分策略（归约轴分 chunk 处理）
- Buffer 规划
- 分支覆盖（dtype、归约轴大小、尾块处理）

---

> 当前返回：「Reduction Tiling 建模暂未收录。」
