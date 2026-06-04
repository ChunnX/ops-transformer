# PLAN — 实现计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段计划。
> 与 TODO 的区别：这是具有全局视角的任务规划/编排方案，TODO 是子任务的执行计划。

## 阶段一：代码检查报告问题修复

### 任务 1.1: 修复错误信息与函数名不匹配
**目标**: `CheckSupportScene` 共享函数中硬编码了 V1 函数名，V2 调用时错误信息不匹配；line 1175 的 V2 函数体内错误信息缺少 V2 后缀
**目标文件**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`
**完成标准**: CheckSupportScene 新增 opName 参数，V1/V2 调用者分别传入正确函数名；line 1175 修正为 V2。grep 验证无遗留。
**状态**: ✅ 已完成

### 任务 1.2: 修复 op_api 文件拼写错误
**目标**: 修复 fisrt→first(7处)、paramater→parameter(1处)、INT_32→DT_INT32(3处)、viewDimMutltiply→viewDimMultiply(3处)
**目标文件**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`
**完成标准**: grep 验证所有错误模式匹配数为 0
**状态**: ✅ 已完成

### 任务 1.3: 修复 infershape 文件拼写/语法错误
**目标**: 修复 FlOAT8_E8M0→FLOAT8_E8M0(2处)、is not support→is not supported(2处)、must bigger than 0→must be greater than 0(1处)
**目标文件**: `gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_infershape.cpp`
**完成标准**: grep 验证所有错误模式匹配数为 0
**状态**: ✅ 已完成

### 任务 1.4: 修复遗漏的语法错误
**目标**: 修复 line 363 `should less than` → `should be less than`（报告中 "should be less or equal to" 指出的问题）
**目标文件**: `gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp`
**完成标准**: grep 验证 `should less than` 模式匹配数为 0
**状态**: ✅ 已完成

## 阶段二：后续代码和文档更新

### 任务 2.1: 待用户指示
**目标**: 根据用户后续提供的代码/文档更新要求执行
**目标文件**: 待定
**完成标准**: 待定
**状态**: ⬜ 待开始

## 当前进度

- [x] 阶段一：代码检查报告问题修复（4/4 任务完成）
- [ ] 阶段二：后续代码和文档更新（等待用户指示）

## 风险与应对

| 风险 | 应对 |
|------|------|
| 修改日志字符串可能影响测试用例中的断言匹配 | 修改后检查 tests/ 目录是否有对应字符串断言，如有需同步更新 |
| 共享函数参数化可能影响其他调用者 | 使用默认参数值，确保未显式传参的调用者行为不变 |
| 后续整改范围不确定 | 等待用户明确指示后再规划具体任务 |
