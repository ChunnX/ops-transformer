# GOAL - GMMSQ算子开发

> 总体目标导向文档，牵引总体方向不偏离

## 项目背景

GroupedMatmulSwigluQuantV2算子，在已有全量化基础上支持MxA8W4的伪量化数据流。

## 总体目标

GroupedMatmulSwigluQuantV2算子支持MxA8W4数据流，tiling拦截校验正确、kernel精度正确，且不影响原有全量化数据流功能。

## 阶段一：代码走读与文档输出 ✅ 已完成

基于 GMMFR (GroupedMatmulFinalizeRouting) 算子的 MxA8W4 数据流实现，完成全面代码走读并撰写走读文档，为后续 GMMSQ 改写提供完整参考。

### 阶段一交付物

- [x] `docs/01_gmmfr_architecture.md`: GMMFR架构总览与数据流 (760行)
- [x] `docs/02_gmmfr_controller.md`: Controller层代码走读 (550行)
- [x] `docs/03_gmmfr_basic_block.md`: BasicBlock层代码走读 (662行)
- [x] `docs/04_gmmfr_vec_compute.md`: VecCompute层代码走读
- [x] `docs/05_gmmfr_cube_compute.md`: CubeCompute层代码走读 (894行)
- [x] `docs/06_gmmfr_sync_mechanism.md`: 同步机制分析 (941行)
- [x] `docs/07_gmmsq_diff_analysis.md`: 差异分析与改写点梳理 (483行)
- [x] `docs/08_gmmsq_impl_plan.md`: GMMSQ实现详细规划 (816行)
- [x] `development/srs_chapter8_encoding_plan.md`: SRS第8章节详细编码方案 (1776行)

## 阶段二：框架搭建与代码实现（当前阶段）

### 阶段二-Part1: 框架搭建（当前）
基于新算子开发适配工程指南，完成MxA8W4数据流的框架支持：
- Tiling Key扩展
- Tiling Data定义
- Tiling模板注册
- Kernel入口分支
- Stub kernel保证框架调用正常

### 阶段二-Part2: Kernel实现（后续）
基于走读文档和编码方案，完成GMMSQ MxA8W4数据流的kernel代码开发：
- Controller层改写
- BasicBlock层改写
- VecCompute层改写（SwiGLU+Quant）
- CubeCompute层改写

## 阶段三：测试验证（待阶段二完成后启动）

完成精度测试、性能测试，确保算子正确性和性能达标。

## 相关链接

- GMMFR Kernel入口: `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_apt.cpp`
- GMMFR Controller: `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_resplit_controller.h`
- GMMSQ 设计文档: `gmm/grouped_matmul_swiglu_quant_v2/.agent/raw/design/`
- 适配工程指南: `~/Code/asc-wiki/ops-dev-wiki/raw/docs/`
