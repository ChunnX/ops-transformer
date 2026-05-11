# PLAN - GMMSQ 算子开发计划

> 将 GOAL.md 的总体目标拆分为可逐步执行的阶段

## 阶段一：代码走读与文档输出（当前阶段）

### 任务 1.1：GMMFR 架构总览与入口分析
**负责人**: Sisyphus  
**目标**: 理解 GMMFR 算子整体架构，kernel入口到各层的调用关系  
**输入文件**:
- `gmm/grouped_matmul_finalize_routing/op_kernel/grouped_matmul_finalize_routing_apt.cpp`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_key.h`
- `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h`
**输出文档**: `docs/01_gmmfr_architecture.md`
**关键内容**:
- kernel入口函数 `grouped_matmul_finalize_routing` 的模板参数与场景分发
- `V310_GMM_FR_ANTI_QUANT` 宏与 MxA8W4 场景的关联
- Controller/BasicBlock/VecCompute/CubeCompute 四层架构关系
- 数据流：GM → AIV反量化 → L1 → AIC矩阵乘 → L0C → fixpipe → UB → AIV后处理 → GM

### 任务 1.2：Controller层代码走读
**负责人**: Subagent A  
**目标**: 深入解读 `GMMFRWeightQuantResplitController` 的实现原理  
**输入文件**:
- `gmm_fr_weight_quant_resplit_controller.h`
**输出文档**: `docs/02_gmmfr_controller.md`
**关键内容**:
- `Init()`: GM地址初始化、BasicBlock初始化
- `Process()`: 主循环流程，按group遍历处理
- `SplitNByMultiCore()`: N轴多核分块策略，动态K调整
- `GetSplitValueFromGroupList()`: groupList解析（cumsum/count模式）
- `UpdateGmAddr()`: 跨group的GM地址更新
- 动态K选择逻辑：`mxA8W4L1KDynamicConfigMThreshold_`

### 任务 1.3：BasicBlock层代码走读
**负责人**: Subagent B  
**目标**: 深入解读 `GMMFRWeightQuantVcvBasicBlock` 的实现原理  
**输入文件**:
- `gmm_fr_weight_quant_vcv_basic_block.h`
**输出文档**: `docs/03_gmmfr_basic_block.md`
**关键内容**:
- `Init()`: L1/L0/UB buffer分配，AIC/AIV初始化分流
- `InitAtomicGm()`: GM清零与sharedInput初始化
- `ComputeBasicBlock()`: AIV/AIC异构执行分流
- `IterateNzNkWithAiv()`: AIV路径主循环（Pass1 + PhaseB + Pass2）
- `IterateNzNkWithKAic()`: AIC路径主循环
- `End()`: 结束阶段清理
- 同步机制：AIC↔AIV跨核信号（SYNC_AIC_AIV_FLAG等）

### 任务 1.4：VecCompute层代码走读
**负责人**: Subagent C  
**目标**: 深入解读 `GMMFRWeightQuantVecCompute` 的实现原理  
**输入文件**:
- `gmm_fr_weight_quant_vec_compute.h`
- `gmm_fr_mx_a8w4_vf.h` (VF反量化)
- `../common/basic_block_vf_mx.h`
**输出文档**: `docs/04_gmmfr_vec_compute.md`
**关键内容**:
- UB buffer分配（weightLowBit, weightHighBit, bias等）
- `CopyGmToUb()`: GM→UB搬运4bit weight
- `AntiQuantProcessNzMxA8W4()`: 4bit→8bit反量化核心
- `FrAntiQuantMxA8W4NzNkVf()`: VF反量化指令
- `MulLogits()`: logits乘加
- `RoutingYToGm()`: scatter add写出到GM
- 同步Event管理（V→MTE2, MTE3→V等）

### 任务 1.5：CubeCompute层代码走读
**负责人**: Subagent D  
**目标**: 深入解读 `GMMFRWeightQuantCubeCompute` 的实现原理  
**输入文件**:
- `gmm_fr_weight_quant_cube_compute.h`
- `gmm_fr_weight_quant_cube_compute_tools.h`
**输出文档**: `docs/05_gmmfr_cube_compute.md`
**关键内容**:
- L1 buffer分配（Weight, Bias, ScaleA, ScaleB, AL1）
- L0 buffer分配（L0A, L0B, L0C, BiasTable）
- `CopyMxScaleGmToL1()`: MX Scale搬运与Dn2Nz转换
- `CopyAAndBiasGmToL1()`: A矩阵ND2Nz搬运
- `LaunchMatmul()`: MMA启动与L0→L1加载
- `DoMmadCompute()`: MMAD计算与Bias加载
- `GetTensorC()`: L0C→UB fixpipe输出

### 任务 1.6：同步机制与数据流分析
**负责人**: Sisyphus  
**目标**: 梳理 AIC↔AIV 同步机制，分析完整数据流  
**输出文档**: `docs/06_gmmfr_sync_mechanism.md`
**关键内容**:
- SYNC_AIC_AIV_FLAG (9): AIC→AIV Weight ready
- SYNC_AIV_AIC_FLAG (8): AIV→AIC Anti-quantized weight ready
- SYNC_AIV_MTE3_AIC_FIX_FLAG (4): AIV→AIC fixpipe请求
- SYNC_AIC_FIX_AIV_VF_FLAG (3): AIC→AIV UB ready
- Pipe Event定义与双缓冲机制
- 完整数据流时序图

### 任务 1.7：GMMSQ差异分析与实现规划
**负责人**: Sisyphus  
**目标**: 基于走读结果，分析GMMSQ与GMMFR的差异，规划代码改写方案  
**输出文档**:
- `docs/07_gmmsq_diff_analysis.md`
- `docs/08_gmmsq_impl_plan.md`
**关键差异**:
- GMMFR后处理：Routing（scatter add） → GMMSQ后处理：SwiGLU + Quant
- GMMFR输出：FP32 → GMMSQ输出：FP8_E4M3FN + FP8_E8M0 scale
- GMMFR N轴：完整N → GMMSQ N轴：分两半（Swish+Gate）
- GMMSQ需要额外Buffer：yF32, maxB16, scaleE8M0, yFp8

## 阶段二：代码实现（待阶段一完成后启动）

基于走读文档，按模块完成GMMSQ代码开发：
1. Tiling层：拦截校验、tiling切分
2. Kernel层：Controller、BasicBlock、VecCompute、CubeCompute改写
3. API层：接口适配
4. 测试：UT/ST测试用例

## 阶段三：测试验证

1. 精度测试：对比golden，验证MxA8W4数据流精度
2. 性能测试：与GMMFR对比，分析性能瓶颈
3. 回归测试：确保不影响原有全量化数据流

## 当前进度

- [x] PLAN.md 制定
- [x] 任务 1.1 GMMFR 架构总览与入口分析 → `docs/01_gmmfr_architecture.md` (760行)
- [x] 任务 1.2 Controller层代码走读 → `docs/02_gmmfr_controller.md` (550行)
- [x] 任务 1.3 BasicBlock层代码走读 → `docs/03_gmmfr_basic_block.md` (662行)
- [x] 任务 1.4 VecCompute层代码走读 → `docs/04_gmmfr_vec_compute.md`
- [x] 任务 1.5 CubeCompute层代码走读 → `docs/05_gmmfr_cube_compute.md` (894行)
- [x] 任务 1.6 同步机制与数据流分析 → `docs/06_gmmfr_sync_mechanism.md` (941行)
- [x] 任务 1.7 GMMSQ差异分析与实现规划 → `docs/07_gmmsq_diff_analysis.md` (483行) + `docs/08_gmmsq_impl_plan.md` (816行)

**阶段一完成！共输出 8 份文档。**
