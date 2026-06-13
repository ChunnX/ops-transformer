# NEXT_STEPS — 下一步计划

> 用户任务的**索引**。记录任务概要和状态，链接到详细任务说明。
> **详细任务说明**存放在 [reference/next_steps/](reference/next_steps/) 目录下。

## 任务 1：编写算子原型描述文档

**任务状态**：✅ 已完成

**简要说明**：完成 `spec/gmm_a16w4nd_op.md`，参照 `gmmfr_op.md` 格式编写 GMM A16W4 weight ND 的算子规格文档。

**产出文件**：
- `.workflow/raw/spec/gmm_a16w4nd_op.md` — 算子原型描述文档
- `.workflow/raw/development/gmm_a16w4_pergroup_nd_mindmap.md` — 脑图文档

**任务详情**：[详细任务说明](reference/next_steps/1_op_spec.md)

## 任务 2：走读 GMM A16W8 代码实现

**任务状态**：✅ 已完成

**简要说明**：走读 kernel Controller（`GMMWeightQuantBasicController`）、tiling（`grouped_weight_quant_batch_matmul_tiling`）、L1/UB 空间划分和 cube 计算实现，产出完整走读文档。反量化实现不在本任务范围内（参见任务 3）。

**产出文件**：
- `.workflow/raw/docs/gmm_a16w8_tiling_walkthrough.md` — Tiling 走读文档
- `.workflow/raw/docs/gmm_a16w8_kernel_walkthrough.md` — Kernel 走读文档

**任务详情**：[详细任务说明](reference/next_steps/2_walkthrough_gmm_a16w8.md)

## 任务 3：走读 ops-nn wqbmmv2 ND 反量化实现

**任务状态**：✅ 已完成

**简要说明**：走读 ops-nn 仓库中 weight ND int4 的反量化方法（`AntiquantComputeW4NKPerGroup`/`AntiquantComputeKNPerGroup`），了解如何迁移到 GMM。

**产出文件**：
- `.workflow/raw/docs/wqbmmv2_nd_antiquant_walkthrough.md` — WQBMMv2 ND 反量化走读文档

**任务详情**：[详细任务说明](reference/next_steps/3_walkthrough_wqbmmv2.md)

## 任务 4：编写 SRS 算子开发实现方案

**任务状态**：待开始

**简要说明**：基于走读结果，编写详细的 GMM A16W4 实现方案（tiling data、tiling 计算、L1/UB 空间划分、kernel 实现方案、核心代码适配点）。

**任务详情**：[详细任务说明](reference/next_steps/4_srs_design.md)
