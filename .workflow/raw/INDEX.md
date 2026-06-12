# raw/ — 原始文档索引

<!-- 更新目录内容时更新索引文档 -->

> 本目录保存所有原始文档，**不可修改**。其他文件是衍生物，这里是 truth source。
> 本索引帮助 agent 按需检索，实现渐进式披露。

## 目录结构

```
raw/
├── INDEX.md        ← 本文件（检索入口）
├── spec/           ← 需求规格说明书（实现无关的原理说明，视项目需求而定）
├── docs/           ← 参考文档（工作中产生的走读、分析、设计文档等）
└── design/         ← 完成本项目任务的设计方案（实现方案、软件架构、编码方案、伪代码等）
```

## spec/ — 需求规格

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [spec/gmmfr_op.md](spec/gmmfr_op.md) | 参考算子原型设计说明书样例（GMMFR MxA8W4 weightNz） | 需要理解参考算子原理、算子规格文档格式时 |
| [spec/gmm_a16w4_950_requirement.md](spec/gmm_a16w4_950_requirement.md) | GMM A16W4(INT4) weight ND 数据流 Ascend950 场景需求 | 需要理解本项目目标、约束和实现方案要点时 |
| [spec/gmm_a16w4nd_op.md](spec/gmm_a16w4nd_op.md) | GMM A16W4 weight ND pergroup 算子原型描述文档 | 需要理解 A16W4 ND 数据流的输入输出规格时 |

## docs/ — 参考文档

<!-- 工作中产生的文档在此登记。包括代码走读、差异分析、设计方案、学习文档等 -->

| 文件 | 内容 | 何时读 |
|------|------|--------|
| （待产出：GMM A16W8 tiling 走读文档） | | 需要了解 A16W8 tiling 实现时 |
| （待产出：GMM A16W8 kernel 走读文档） | | 需要了解 A16W8 kernel 实现时 |
| （待产出：wqbmmv2 ND 反量化走读文档） | | 需要了解 ND int4 pergroup 反量化实现时 |

## development/ — 开发辅助文档

| 文件 | 内容 | 何时读 |
|------|------|--------|
| [development/gmm_a16w4_pergroup_nd_mindmap.md](development/gmm_a16w4_pergroup_nd_mindmap.md) | GMM A16W4 PerGroup ND 脑图（输入/输出/UT/ST） | 需要快速了解算子全貌、测试用例设计时 |

## design/ — 设计文档

<!-- 项目任务实现方案的设计文档，包括实现方案、架构设计、编码方案、伪代码等 -->

| 文件 | 内容 | 何时读 |
|------|------|--------|
| （待产出：GMM A16W4 950 SRS 实现方案） | | 需要了解 A16W4 ND 实现方案时 |
