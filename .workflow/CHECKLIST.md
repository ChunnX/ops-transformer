# CHECKLIST — 检查清单

> 各阶段的质量检查项。帮助正确完成任务，确保不遗漏关键验证。
> 检查项来源：自我发现的问题、用户提供的要求、历史经验教训。
>
> **审查要求**: 使用 subagent 在独立上下文中客观检查。发现问题必须诚实报告。

## 审查原则

- 基于事实，不基于假设
- 可验证，可复现
- 明确标注：✅ 通过 / ❌ 不通过 / ⬜ 不适用

## 阶段一：代码检查报告问题修复

### 拼写/语法错误清除验证
- [x] grep `fisrt` 匹配数为 0
- [x] grep `paramater` 匹配数为 0
- [x] grep `INT_32`（在日志字符串中）匹配数为 0
- [x] grep `viewDimMutltiply` 匹配数为 0
- [x] grep `FlOAT8_E8M0` 匹配数为 0
- [x] grep `is not support[^e]` 匹配数为 0
- [x] grep `must bigger than` 匹配数为 0
- [x] grep `should less than`（缺少 be）匹配数为 0

### 函数名一致性验证
- [x] V2 函数调用链中无 `GroupedMatmulFinalizeRoutingWeightNz`（无 V2 后缀）的错误信息
- [x] CheckSupportScene 参数化后 V1 调用者行为不变（默认参数值正确）

### 影响范围验证
- [x] 所有修改仅涉及日志字符串和局部变量名
- [x] 无业务逻辑变更
- [ ] 测试用例中无硬编码旧错误信息字符串（需检查 tests/ 目录）

## 阶段二：后续代码和文档更新

### 待定检查项
- [ ] 待用户指示后补充

## 通用检查项

### 代码规范
- [ ] 文件命名与注释匹配
- [ ] Header guard 与文件名匹配
- [ ] 无遗留 TODO/FIXME 未处理

### 功能验证
- [ ] 编译通过
- [ ] 测试通过
- [ ] 不影响已有功能

### 工作流完整性
- [x] CHANGELIST.md 已追加变更记录
- [x] PLAN.md 任务状态已更新
- [x] NEXT_STEPS.md 任务状态已更新
- [x] GOAL.md 目标进度已更新
