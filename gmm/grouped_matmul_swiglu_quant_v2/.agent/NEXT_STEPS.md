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

**任务状态**：进行中

**简要说明**：开发grouped_matmul_swiglu_quant_v2算子的tiling部分代码，基于测试驱动开发完善代码，直到所有host侧tiling UT全部通过。

**任务详情**：[grouped_matmul_swiglu_quant_v2 MxA8W4算子tiling部分测试验证及实现完善](./raw/next_steps/2-grouped_matmul_swiglu_quant_v2的MxA8W4算子tiling部分测试验证及实现完善.md)
