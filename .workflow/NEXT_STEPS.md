# NEXT_STEPS — 下一步计划

> 用户任务的**索引**。记录任务概要和状态，链接到详细任务说明。

## 任务 1：项目脚手架 + op_host 基础

**任务状态**：✅ 已完成

**简要说明**：创建目录结构、CMakeLists.txt 框架，实现 def.cpp（算子注册）和 infershape.cpp（shape 推导/校验）

**任务详情**：[详细任务说明](../PLAN.md#阶段一项目脚手架--op_host-基础)

**完成状态**:
- [x] 目录结构 + CMakeLists.txt
- [x] def.cpp — 算子注册
- [x] infershape.cpp — shape 推导校验

## 任务 2：op_api ACLNN 接口

**任务状态**：✅ 已完成

**简要说明**：实现 ACLNN L2 公共 API（参数校验、executor 创建）和 L0 内部实现（OP_TYPE_REGISTER、INFER_SHAPE、ADD_TO_LAUNCHER_LIST）

**任务详情**：[详细任务说明](../PLAN.md#阶段二op_api-aclnn-接口)

**完成状态**:
- [x] L2 公共 API (aclnn_*.h/.cpp)
- [x] L0 内部实现 (qkv_*.h/.cpp)

## 任务 3：测试 + 文档 + 验收

**任务状态**：✅ 已完成

**简要说明**：实现 op_api UT、ACLNN 接口文档、README，完成编译验证和 UT 通过

**任务详情**：[详细任务说明](../PLAN.md#阶段三测试--文档--验收)

**完成状态**:
- [x] op_api UT (4/4 测试通过)
- [x] ACLNN 接口文档 + README
- [x] 编译验证 + UT 通过
