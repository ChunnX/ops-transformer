# GOAL - GMMSQ算子开发

> 总体目标导向文档，牵引总体方向不偏离

## 项目背景

GroupedMatmulSwigluQuantV2算子，在已有全量化基础上支持MxA8W4的伪量化数据流。

## 总体目标

GroupedMatmulSwigluQuantV2算子支持MxA8W4数据流，tiling拦截校验正确、kernel精度正确，且不影响原有全量化数据流功能。

## 阶段一：代码走读与文档输出（当前阶段）

基于 GMMFR (GroupedMatmulFinalizeRouting) 算子的 MxA8W4 数据流实现，完成全面代码走读并撰写走读文档，为后续 GMMSQ 改写提供完整参考。

### 阶段一目标拆解

1. **走读 GMMFR 核心代码架构**：按 Controller → BasicBlock → VecCompute/CubeCompute 层次全面解读
2. **撰写代码走读文档**：
   - `docs/01_gmmfr_architecture.md`: GMMFR架构总览与数据流
   - `docs/02_gmmfr_controller.md`: Controller层代码走读（分核策略、M/N/K切分）
   - `docs/03_gmmfr_basic_block.md`: BasicBlock层代码走读（AIV/AIC异构协同）
   - `docs/04_gmmfr_vec_compute.md`: VecCompute层代码走读（反量化、Routing）
   - `docs/05_gmmfr_cube_compute.md`: CubeCompute层代码走读（MMA矩阵乘、Scale应用）
   - `docs/06_gmmfr_sync_mechanism.md`: 同步机制分析（AIC↔AIV信号、Pipe Event）
3. **分析 GMMSQ 与 GMMFR 的差异**：
   - `docs/07_gmmsq_diff_analysis.md`: 差异分析与改写点梳理
   - `docs/08_gmmsq_impl_plan.md`: GMMSQ实现详细规划

### 阶段一完成标准

- 所有走读文档完成并审核通过
- GMMSQ 与 GMMFR 的关键差异点清晰明确
- GMMSQ 实现方案经过评审确认

## 阶段二：代码实现（待阶段一完成后启动）

基于走读文档和实现规划，完成 GMMSQ MxA8W4 数据流的代码开发。

## 阶段三：测试验证（待阶段二完成后启动）

完成精度测试、性能测试，确保算子正确性和性能达标。

## 当前目标完成情况

- [x] GOAL.md 更新
- [ ] PLAN.md 制定
- [ ] 代码走读文档输出
- [ ] 差异分析与实现规划

## 相关链接

- GMMFR Kernel入口: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_apt.cpp`
- GMMFR Controller: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h`
- GMMSQ 设计文档: `gmm/grouped_matmul_swiglu_quant_v2/.agent/raw/design/`
