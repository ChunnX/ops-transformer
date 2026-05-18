# PLAN - GMMSQ 算子开发计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段

## 阶段一：代码走读与文档输出 ✅ 已完成

### 任务 1.1-1.7 全部完成

- [x] 任务 1.1 GMMFR 架构总览与入口分析 → `docs/01_gmmfr_architecture.md` (760行)
- [x] 任务 1.2 Controller层代码走读 → `docs/02_gmmfr_controller.md` (550行)
- [x] 任务 1.3 BasicBlock层代码走读 → `docs/03_gmmfr_basic_block.md` (662行)
- [x] 任务 1.4 VecCompute层代码走读 → `docs/04_gmmfr_vec_compute.md`
- [x] 任务 1.5 CubeCompute层代码走读 → `docs/05_gmmfr_cube_compute.md` (894行)
- [x] 任务 1.6 同步机制与数据流分析 → `docs/06_gmmfr_sync_mechanism.md` (941行)
- [x] 任务 1.7 GMMSQ差异分析与实现规划 → `docs/07_gmmsq_diff_analysis.md` (483行) + `docs/08_gmmsq_impl_plan.md` (816行)

## 阶段二：框架搭建与代码实现

### 阶段二-Part1: 框架搭建 ✅ 已完成

基于新算子开发适配工程指南，完成MxA8W4数据流的框架支持：

#### 任务 2.1.1: Tiling Key扩展
**目标文件**: `op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h`
- 新增MxA8W4场景的模板参数组合
- 参考GMMFR的`gmm_fr_weight_quant_tiling_key.h`

#### 任务 2.1.2: Tiling Data定义
**目标文件**: `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_tiling_data.h`
- 定义GMMSQ专用的MxA8W4 TilingData结构体
- 参考GMMFR的`gmm_fr_weight_quant_tiling_data.h`

#### 任务 2.1.3: Tiling模板注册
**目标文件**: `op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.h/cpp`
- 新增MxA8W4场景识别（InferScenario）
- 新增TilingKey计算逻辑
- 注册tiling模板（IMPL_OP_OPTILING + REGISTER_TILING_TEMPLATE）
- 参考GMMFR的`grouped_matmul_finalize_routing_weight_quant_tiling.h/cpp`

#### 任务 2.1.4: Kernel入口分支
**目标文件**: `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp`
- 新增`V310_GMM_SQ_ANTI_QUANT`宏定义
- 新增MxA8W4场景的Controller调用分支
- 先使用stub kernel保证框架调用正常

#### 任务 2.1.5: 编译验证
- 编译ophost动态库验证tiling
- 编译kernel二进制验证入口

### 阶段二-Part2: Kernel代码适配 ✅ 已完成

Kernel代码从GMMFR迁移后适配为GMMSQ专用：
- [x] Controller层：命名统一，移除scaleType参数
- [x] BasicBlock层：模板参数对齐Controller，UpdateGlobalAddr签名修正
- [x] CubeCompute层：命名统一，修复MX_SCALE_K_L1_SIZE shadowing bug
- [x] VecCompute层：VF函数调用更新，SwiGLU分离y/gate指针
- [x] VF函数：Fr→Gmmsq前缀重命名，SwiGLU/Quant stub实现
- [x] common/目录：header guard修正
- [x] apt.cpp入口：xScale参数修复，模板参数对齐

## 阶段三：编译验证与测试（待阶段二完成后启动）

完成kernel编译验证、精度测试、性能测试，确保算子正确性和性能达标。

## 当前进度

- [x] 阶段一：代码走读与文档输出
- [x] 阶段二-Part1: 框架搭建
- [x] 阶段二-Part2: Kernel代码适配
- [ ] 阶段三：编译验证与测试
