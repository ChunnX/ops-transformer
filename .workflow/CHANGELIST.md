# CHANGELIST — 变更记录

> 记录项目演进和变更，便于后期回溯和审查。
> **只追加，不删除已有条目。**

---

## 变更记录格式

```
### YYYY-MM-DD | 变更标题

**类型**: 🆕 新功能 | 🔧 修改 | 🐛 修复 | 🗑️ 删除 | 📝 文档
**范围**: 影响范围

#### 变更内容
- 具体变更了什么

#### 关键决策
- 做了什么决策，为什么

#### 影响范围
- 对现有功能的影响
```

---

<!-- 在此追加变更记录 -->

### 2026-06-04 | 修复 GroupedMatmulFinalizeRoutingWeightNzV2 代码检查报告中的问题

**类型**: 🐛 修复
**范围**: gmm/grouped_matmul_finalize_routing/op_api/ 和 op_host/

#### 变更内容
- **错误信息函数名不匹配**: `CheckSupportScene` 函数新增 `opName` 参数（默认值 `"GroupedMatmulFinalizeRoutingWeightNz"`），V1 和 V2 调用者分别传入正确的函数名；line 1175 的错误信息从 `GroupedMatmulFinalizeRoutingWeightNz` 修正为 `GroupedMatmulFinalizeRoutingWeightNzV2`
- **拼写修复 (op_api)**: `fisrt` → `first`（7处）、`paramater` → `parameter`（1处）、`INT_32` → `DT_INT32`（3处）、`viewDimMutltiply` → `viewDimMultiply`（3处）
- **拼写/语法修复 (infershape)**: `FlOAT8_E8M0` → `FLOAT8_E8M0`（2处）、`is not support` → `is not supported`（2处）、`must bigger than 0` → `must be greater than 0`（1处）

#### 关键决策
- `CheckSupportScene` 是 V1/V2 共享函数，采用参数化方案（`const char* opName` 带默认值）而非复制函数，保持向后兼容且最小化改动
- `INT_32` → `DT_INT32` 修正了3处（V1/V2/V3 的 WeightNd 函数），统一使用正确的 DataType 枚举名称

#### 影响范围
- 仅修改日志/错误信息字符串和局部变量名，不影响任何业务逻辑和计算流程

### 2026-06-04 | 补充修复遗漏的语法错误 + 工作流初始化

**类型**: 🐛 修复 + 📝 文档
**范围**: op_api 文件 + .workflow/ 全部工作流文件

#### 变更内容
- **代码修复**: line 363 `should less than` → `should be less than`（上一轮遗漏，报告中 "should be less or equal to" 指出的问题）
- **工作流初始化**: 填充 GOAL.md、PLAN.md、NEXT_STEPS.md、CHECKLIST.md 从模板到实际内容；更新 raw/INDEX.md 匹配实际文件结构

#### 关键决策
- 阶段一所有 4 个任务标记为已完成
- 阶段二预留为"待用户指示"，保持工作流可扩展

#### 影响范围
- 代码：仅 1 处日志字符串修复
- 工作流：所有 .workflow/ 文件从模板状态初始化为实际项目状态
