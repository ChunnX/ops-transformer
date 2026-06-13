# PLAN — 实现计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段计划。
> 仅关注 tiling + kernel，不涉及 Aclnn 通路（API checker、infershape）。
> **初期方案计划不进行编码，主要用于确认编码方案，保证方案正确有效。**

## 阶段一：规划与设计（当前阶段）

> 本阶段产出完整的算子规格文档、代码走读文档和 SRS 实现方案文档，为后续编码提供充分依据。

### 任务 1：编写算子原型描述文档
**目标**: 完成 `spec/gmm_a16w4nd_op.md`，参照 `gmmfr_op.md` 格式编写 GMM A16W4 weight ND 的算子规格文档
**产出**: `.workflow/raw/spec/gmm_a16w4nd_op.md`
**详细任务说明**: [reference/next_steps/1_op_spec.md](reference/next_steps/1_op_spec.md)

### 任务 2：走读 GMM A16W8 代码实现
**目标**: 走读 kernel 实现（`GMMWeightQuantBasicController`）和 tiling 实现（`grouped_weight_quant_batch_matmul_tiling`），产出完整走读文档
**产出**: `.workflow/raw/docs/` 下的走读文档
**详细任务说明**: [reference/next_steps/2_walkthrough_gmm_a16w8.md](reference/next_steps/2_walkthrough_gmm_a16w8.md)

### 任务 3：走读 ops-nn wqbmmv2 ND 反量化实现
**目标**: 走读 ops-nn 仓库中 weight ND int4 的反量化方法，了解如何迁移到 GMM
**产出**: `.workflow/raw/docs/` 下的走读文档
**详细任务说明**: [reference/next_steps/3_walkthrough_wqbmmv2.md](reference/next_steps/3_walkthrough_wqbmmv2.md)

### 任务 4：编写 SRS 算子开发实现方案
**目标**: 基于走读结果，编写详细的 GMM A16W4 实现方案，包括 tiling data、tiling 计算方案、L1/UB 空间划分、kernel 实现方案、核心代码适配点
**产出**: `.workflow/raw/design/gmm_a16w4_950_design.md`
**详细任务说明**: [reference/next_steps/4_srs_design.md](reference/next_steps/4_srs_design.md)

## 阶段二：编码实现（待启动）

> 在阶段一方案确认后启动编码。

### 任务 5：Tiling 实现
**目标**: 在 arch35 tiling 中为 A16W4 pergroup ND 场景生成正确的 tiling 参数
**目标文件**: `gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_weight_quant_batch_matmul_tiling.cpp`

### 任务 6：Kernel 适配
**目标**: 在 `weight_quant_basic_block` 体系中适配 A16W4 pergroup ND 计算逻辑
**目标文件**:
- `grouped_matmul_weight_quant_basic_controller.h` — Controller 适配
- `basic_block_vf_nd.h` — ND 反量化 VF 函数实现
- `weight_quant_vec_compute.h` — 向量计算适配
- `basic_block_config.h` — WqmmConfig 新增

## 阶段三：验证（待启动）

### 任务 7：单元测试
**目标**: 覆盖 tiling 和 kernel 的正确性

### 任务 8：兼容性回归
**目标**: 确保已有 950 测试不受影响

## 当前进度

- [ ] 阶段一：规划与设计
  - [x] 任务 1：算子原型描述文档 ✅
  - [x] 任务 2：走读 GMM A16W8 代码实现 ✅
  - [x] 任务 3：走读 ops-nn wqbmmv2 ND 反量化实现 ✅
  - [ ] 任务 4：SRS 算子开发实现方案
- [ ] 阶段二：编码实现
- [ ] 阶段三：验证

## 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| `SetGmAddr` 的 pergroup 地址偏移与 perchannel 差异大 | Controller 需要较大修改 | 走读阶段仔细分析偏移公式，设计阶段明确 pergroup 偏移方案 |
| `basic_block_vf_nd.h` 中 pergroup 反量化逻辑缺失 | 需要新增 pergroup ND 向量计算 | 走读 wqbmmv2 的 `AntiquantComputeW4NKPerGroup`/`AntiquantComputeKNPerGroup`，设计迁移方案 |
| WqmmConfig 缺少 A16W4 pergroup ND 配置 | 需要新增配置项 | 参照 `S8S4_NZKN_G` 新增 ND 版本 |
| int4 字节偏移计算容易出错 | 地址偏移需要减半处理 | 设计阶段明确 int4 偏移计算公式，编码阶段重点验证 |
