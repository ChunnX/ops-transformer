# NEXT_STEPS — 下一步计划

> 用户任务的**索引**。记录任务概要和状态，链接到详细任务说明。
> 主 agent 通过本文件快速了解任务全貌，具体开发时再读取详细文档。
>
> **详细任务说明**存放在 [reference/next_steps/](reference/next_steps/) 目录下。

## 任务 1：修复 GroupedMatmulFinalizeRoutingWeightNzV2 代码检查报告

**任务状态**：已完成

**简要说明**：根据 `raw/docs/GroupedMatmulFinalizeRoutingWeightNzV2_code_report.md` 中指出的问题，修复错误信息函数名不匹配和多处拼写/语法错误。

**任务详情**：[代码检查报告](raw/docs/GroupedMatmulFinalizeRoutingWeightNzV2_code_report.md)

**完成状态**:
- [x] 修复 V2 调用链中错误信息函数名（CheckSupportScene 参数化 + line 1175）
- [x] 修复 `fisrt` → `first`（7处）
- [x] 修复 `paramater` → `parameter`（1处）
- [x] 修复 `INT_32` → `DT_INT32`（3处）
- [x] 修复 `viewDimMutltiply` → `viewDimMultiply`（3处）
- [x] 修复 infershape 文件拼写/语法错误（FlOAT8_E8M0×2、is not support×2、must bigger than 0×1）
- [x] 修复遗漏的语法错误（`should less than` → `should be less than`）

## 任务 2：后续代码和文档更新

**任务状态**：待开始

**简要说明**：用户提到"后续还有其他代码和文档更新"，等待用户提供具体要求。

**任务详情**：待补充

**完成状态**:
- [ ] 等待用户提供具体更新要求
