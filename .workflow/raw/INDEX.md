# raw/ — 原始文档索引

> 本目录保存所有原始文档，**不可修改**。其他文件是衍生物，这里是 truth source。
> 本索引帮助 agent 按需检索，实现渐进式披露。

## 目录结构

```
raw/
├── INDEX.md        ← 本文件（检索入口）
├── spec/           ← 需求规格说明书（算子接口定义、参数约束等）
├── docs/           ← 代码检查报告、审查文档等
└── design/         ← 设计文档（预留）
```

## spec/ — 需求规格

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [spec/gmmfr_op.md](spec/gmmfr_op.md) | GMMFR V2 算子 aclnn 通路 SRS（MxA8W4 数据流） | 了解算子接口定义、参数约束、计算公式 |

## docs/ — 检查报告与审查文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [docs/GroupedMatmulFinalizeRoutingWeightNzV2_code_report.md](docs/GroupedMatmulFinalizeRoutingWeightNzV2_code_report.md) | V2 算子代码检查报告（拼写/语法/函数名问题） | 执行阶段一修复任务时 |

## design/ — 设计文档

<!-- 预留，待后续补充 -->

暂无文档。

## 检索场景速查

| 我需要... | 读哪个文件 |
|-----------|-----------|
| 了解算子的数学公式和接口定义 | `spec/gmmfr_op.md` |
| 查看代码检查报告中的问题清单 | `docs/GroupedMatmulFinalizeRoutingWeightNzV2_code_report.md` |
| 查看用户的详细任务要求 | `.workflow/reference/next_steps/` 下对应文件（NEXT_STEPS.md 为索引） |
