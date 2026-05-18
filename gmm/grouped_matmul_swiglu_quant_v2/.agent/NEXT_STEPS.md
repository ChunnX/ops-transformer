# 下一步计划

> 下一步计划是用户后续工作计划，主要用于记录用户下一步需求和原始意图。任务完成后需要记录任务完成状态。

> 完成该项目需要遵循以下定义的agent工作流元文档架构，该工作流将用户提示词或agent关键输出按类别落盘到文件，用于保证跨agent上下文统一，保证完成复杂工作任务时的连贯性，核心文档在[.agent目录](./)下，元文档分类及定义如下：
> - [GOAL.md](./GOAL.md): 总体工作目标和核心任务方向，保证总体方向与用户对齐，不出现偏离。
> - [PLAN.md](./PLAN.md): 每个阶段的任务计划，对当前阶段的任务进行拆解和规划，保证在不同独立上下文之间也能有接续。与TODO的区别是这是具有全局视角的任务计划、规划或编排方案，而TODO是子任务的工作计划。
> - [CHECKLIST.md](./CHECKLIST.md): 记录每个阶段完成任务所需要的检查项，保证任务完成的可靠性，需要在每个阶段完成后按照清单进行检查校验。
> - [CHANGELIST.md](./CHANGELIST.md): 记录项目演进和变更，便于用户后期回溯，对功能实现进展或方案进行审查。
> - [DESIGN.md](./DESIGN.md)：这是当前复杂任务的核心设计方案文档，通常被总结为核心文档的索引入口。
> - [raw目录](./raw/)：其中归档了用户或LLM总结的原始意图或文档，保留没有任何失真的原始信息，需要时进行读取以获得更加准确和详细的信息。

## 任务1：grouped_matmul_swiglu_quant_v2 算子新数据流的框架支持

**任务状态**：已完成

**简要说明**：适配搭建grouped_matmul_swiglu_quant_v2算子的基本代码框架，能够编译验证通过。

**任务详情**：[grouped_matmul_swiglu_quant_v2 算子新数据流的框架支持](./raw/next_steps/1-grouped_matmul_swiglu_quant_v2算子新数据流的框架支持.md)

## 任务2：grouped_matmul_swiglu_quant_v2 MxA8W4算子tiling部分测试验证及实现完善

**任务状态**：已完成（41/42用例通过）

**简要说明**：开发grouped_matmul_swiglu_quant_v2算子的tiling部分代码，基于测试驱动开发完善代码，直到所有host侧tiling UT全部通过。

**任务详情**：[grouped_matmul_swiglu_quant_v2 MxA8W4算子tiling部分测试验证及实现完善](./raw/next_steps/2-grouped_matmul_swiglu_quant_v2的MxA8W4算子tiling部分测试验证及实现完善.md)

**完成状态**:
- [x] 补充MxA8W4 CSV测试用例（8个用例：正常、边界、非法场景）
- [x] 完善tiling校验逻辑：CheckGroupNum()（E<=1024）、CheckScaleShapes()（wscale 4D/xscale 3D校验）
- [x] 编译验证tiling代码（ophost编译通过）
- [x] 编译运行ut程序进行测试验证，41/42用例通过
- [ ] test_mxa8w4_normal_transpose用例因测试框架bool属性检索限制未通过（非代码问题）

## 任务3：grouped_matmul_swiglu_quant_v2 MxA8W4算子kernel部分代码开发

**任务状态**：已完成

**简要说明**：开发grouped_matmul_swiglu_quant_v2算子的kernel部分代码，编写完毕后编译kernel二进制验证是否存在编译错误，直到所有代码完成并编译成功即通过。

**任务详情**：[grouped_matmul_swiglu_quant_v2 MxA8W4算子kernel部分代码开发](./raw/next_steps/3-grouped_matmul_swiglu_quant_v2的MxA8W4算子kernel部分代码开发.md)

**完成状态**:
- [x] 创建gmmsq_weight_quant_cube_compute_tools.h（从GMMFR复制）
- [x] 创建gmmsq_mx_a8w4_vf.h（反量化+SwiGLU+Quant VF函数）
- [x] 创建gmmsq_weight_quant_vec_compute.h（VecCompute层）
- [x] 创建gmmsq_weight_quant_cube_compute.h（CubeCompute层）
- [x] 创建gmmsq_weight_quant_vcv_basic_block.h（BasicBlock层）
- [x] 创建gmmsq_weight_quant_resplit_controller.h（Controller层）
- [x] 更新apt.cpp kernel入口，连接完整Controller调用
- [x] 复制common头文件（basic_block_config.h, basic_block_vf_mx.h, tool.h）
- [x] 修复namespace解析问题（所有kernel文件的using声明）
- [x] 修复模板参数问题（hasBias参数添加）
- [x] 修复MicroAPI调用问题：
  - ShiftRight → ShiftRights（标量移位需使用ShiftRights）
  - 添加CAST_FP32_TO_FP8_TRAIT、CAST_FP16_TO_FP8_TRAIT（FP8 Cast需SatMode::SAT + RoundMode::CAST_RINT）
  - 添加CAST_U32_TO_FP16_TRAIT（uint32→half转换）
- [x] 编译验证opkernel二进制 ✅ 成功
