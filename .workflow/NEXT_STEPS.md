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

**任务状态**：✅ 已完成

**简要说明**：基于走读结果，按模块分文档编写 GMM A16W4 实现方案，包括交付件清单、tiling 拦截校验与切分方案、kernel 实现方案、与硬件无关的实现方案。

**产出文件**：
- `.workflow/raw/design/01_deliverables.md` — 交付件清单（270 行）
- `.workflow/raw/design/02_platform_independent_design.md` — 与硬件无关的实现方案（313 行）
- `.workflow/raw/design/03_tiling_design.md` — Tiling 拦截校验与切分方案（455 行）
- `.workflow/raw/design/04_kernel_design.md` — Kernel 实现方案（642 行）

**任务详情**：[详细任务说明](reference/next_steps/4_srs_design.md)

## 任务 5：设计 GMM A16W4 PerGroup ND 测试用例

**任务状态**：✅ 已完成

**简要说明**：基于算子约束说明和 01/02/03/04 设计文档，设计具体的测试用例，给出具体参数设置。包括 tiling 异常拦截（黑盒）、精度（黑盒）、功能（黑盒）、性能（黑盒）、泛化（黑盒）、kernel 打桩验证（白盒插桩级）、tiling 白盒（端到端级）、kernel 白盒（端到端级）。产出用例设计文档和输入 Excel 表格（8 个 sheet），相关脚本和用例结果均放在 `.workflow/raw/design/` 目录下。

**产出文件**：
- `.workflow/raw/design/05_test_case_design.md` — 用例设计文档（162 个用例）
- `.workflow/raw/design/test_cases_gmm_a16w4_pergroup_nd.xlsx` — 用例输入 Excel 表格（8 个 sheet）
- `.workflow/raw/design/generate_test_cases.py` — 用例生成脚本

**任务详情**：[详细任务说明](reference/next_steps/5_test_case_design.md)

## 任务 6：总结归纳设计文档

**任务状态**：✅ 已完成

**简要说明**：对方案设计文档、用例文档和脑图进行总结归纳，形成留档查看的设计文档。包括：更新脑图（对齐用例设计文档）、创建 STC 用例设计方案文档、创建 SRS 算子实现方案文档（含 tiling 切分 SVG、buffer 分配 SVG、软件算法映射到硬件的计算流程 SVG、关键修改点）。

**产出文件**（均在 `.workflow/raw/development/` 下）：
- `gmm_a16w4_pergroup_nd_mindmap.md` — 更新后的脑图（对齐用例设计，162 个用例总览）
- `gmm_a16w4_pergroup_nd_stc.md` — STC 用例设计方案文档（方法论 + 数量统计）
- `gmm_a16w4_pergroup_nd_srs.md` — SRS 算子实现方案文档（3 个 SVG 图 + 伪代码 + 关键修改点）

**任务详情**：[详细任务说明](reference/next_steps/6_summary_docs.md)
