# GOAL — 总体目标

> 总体目标导向文档，牵引核心方向不偏离，保证跨会话目标一致。

## 项目背景

GroupedMatmulFinalizeRouting（GMMFR）是 ops-transformer 项目中的 MoE 融合算子，将 GroupedMatmul 与 MoeFinalizeRouting 组合，在 MoE 场景中完成分组矩阵乘 + 路由 combine 的端到端计算。

该算子存在多个版本接口（V1/V2/V3 及 WeightNz/WeightNzV2），覆盖 INT8/INT4/MxA8W4 等多种量化数据流。代码审查发现算子实现中存在错误信息不匹配、拼写错误、语法错误等问题，需要系统性整改。

## 总体目标

完成 GroupedMatmulFinalizeRouting 算子的全面整改，包括：
1. 修复代码检查报告中指出的所有问题（错误信息、拼写、语法）
2. 根据后续用户指示完成额外的代码和文档更新
3. 确保整改后代码质量达标，不影响现有功能

## 目标拆解

### 阶段一：代码检查报告问题修复

根据 `raw/docs/GroupedMatmulFinalizeRoutingWeightNzV2_code_report.md` 修复所有已识别问题。

- [x] 任务 1.1: 修复错误信息与函数名不匹配（CheckSupportScene 参数化 + V2 调用链）
- [x] 任务 1.2: 修复 op_api 文件拼写错误（fisrt/paramater/INT_32/viewDimMutltiply）
- [x] 任务 1.3: 修复 infershape 文件拼写/语法错误（FlOAT8_E8M0/is not support/must bigger than 0）
- [x] 任务 1.4: 修复遗漏的语法错误（should less than → should be less than）

### 阶段二：后续代码和文档更新

根据用户后续指示，完成额外的代码和文档整改。

- [ ] 任务 2.1: 待用户指示
- [ ] 任务 2.2: 待用户指示

## 约束与成功标准

- 所有修改仅涉及日志/错误信息字符串和变量名，不改变业务逻辑
- 修改后 grep 验证：所有已知拼写/语法错误模式匹配数为 0
- 编译通过，不影响已有功能
- 每轮修改记录在 CHANGELIST.md 中，可追溯

## 相关链接

- [实现计划](./PLAN.md)
- [用户下一步计划](./NEXT_STEPS.md)
- [检查清单](./CHECKLIST.md)
- [变更记录](./CHANGELIST.md)
- [原始文档索引](./raw/INDEX.md)
