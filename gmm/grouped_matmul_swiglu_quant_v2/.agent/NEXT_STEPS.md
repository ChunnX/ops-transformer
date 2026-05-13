# 下一步计划

> 下面是用户后续工作计划，主要用于记录用户下一步需求和原始想法。任务完成后需要记录任务完成状态。
>
> 注意agent工作文档的更新：
> - GOAL.md: 记录总体工作目标和核心任务方向，保证总体方向与用户对齐，不出现偏离。
> - PLAN.md: 更新当前阶段的任务计划，对当前阶段的任务进行拆解和规划，保证在不同独立上下文之间也能有接续。
> - CHECKLIST.md: 记录每个阶段完成任务所需要的检查项，保证任务完成的可靠性。
> - CHANGELIST.md: 记录项目演进和变更，便于后期回溯。

## grouped_matmul_swiglu_quant_v2 算子新数据流的框架支持

### 任务说明

下一步开始第二阶段新算子数据流支持的开发任务，你需要根据[新算子开发适配工程指南](/home/sunduiyang/Code/asc-wiki/ops-dev-wiki/raw/docs)适配grouped_matmul_swiglu_quant_v2算子MxA8W4的数据流，优先查看checklist.md的检查清单，以该清单作为指引完成新算子数据流的支持，详细的适配实现方法可以参考该指南中的其他章节。

为了保证新算子的数据流支持是有效的，请先忽略具体的kernel或tiling的代码实现，只用保证基本框架调用正常、kernel能够正常执行即可，后续代码可以逐步补充。

为了完成这个复杂的任务，请尽可能使用subagent完成拆分后的子任务，保持干净有效的上下文，从而提高任务的可靠性。

### 任务完成状态

- [x] 更新GOAL.md和PLAN.md
- [x] 创建MxA8W4场景的tiling key和tiling data
- [x] 创建MxA8W4场景的tiling模板（arch35）
- [x] 更新apt.cpp添加MxA8W4场景的kernel入口分支
- [ ] 创建stub kernel实现（保证框架调用正常）
- [ ] 编译验证

### 已完成的工作

#### 1. Tiling Key扩展 ✅
文件: `op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h`
- 新增ATRANS/BTRANS/HASBIAS模板参数
- 新增2种MxA8W4组合: (0,0,0)和(0,1,0)

#### 2. Tiling Data定义 ✅
文件: `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_tiling_data.h`
- 定义GMMSQWeightQuantTilingData结构体（40字节，8字节对齐）
- 包含9个字段: groupListType, hasBias, reserve1, groupNum, coreNum, outputBs, groupSize, kSize, nSize

#### 3. Tiling模板注册 ✅
文件: `op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.h` 和 `.cpp`
- GMMSQWeightQuantTiling类，继承自TilingBaseClass
- InferScenario(): 识别MxA8W4场景（x=FP8_E4M3FN, w=FP4_E2M1, wFormat=FRACTAL_NZ, scale=FP8_E8M0）
- DoOpTiling(): 设置tiling data
- GetTilingKey(): 返回GET_TPL_TILING_KEY(ATRANS=0, BTRANS, HASBIAS=0)
- 注册为模板ID=2（REGISTER_OPS_TILING_TEMPLATE）

#### 4. Kernel入口分支 ✅
文件: `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp`
- 新增V310_GMM_SQ_ANTI_QUANT宏定义
- 在FP8_E8M0分支内新增MxA8W4场景的stub kernel调用
- 包含tiling data头文件

#### Phase 1: 框架搭建（当前阶段）

基于GMMFR的MxA8W4实现作为参考，在GMMSQ算子中新增MxA8W4数据流的框架支持：

1. **Tiling Key扩展** (`op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h`)
   - 新增MxA8W4场景的模板参数组合
   - 参考GMMFR的`gmm_fr_weight_quant_tiling_key.h`

2. **Tiling Data扩展** (`op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_tiling_data.h`)
   - 定义GMMSQ专用的MxA8W4 TilingData结构体
   - 参考GMMFR的`gmm_fr_weight_quant_tiling_data.h`

3. **Tiling模板** (`op_host/op_tiling/arch35/gmmsq_weight_quant_tiling.h/cpp`)
   - 新增MxA8W4场景识别（InferScenario）
   - 新增TilingKey计算逻辑
   - 注册tiling模板（IMPL_OP_OPTILING + REGISTER_TILING_TEMPLATE）
   - 参考GMMFR的`grouped_matmul_finalize_routing_weight_quant_tiling.h/cpp`

4. **Kernel入口** (`op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp`)
   - 新增`V310_GMM_SQ_ANTI_QUANT`宏定义
   - 新增MxA8W4场景的Controller调用分支
   - 先使用stub kernel保证框架调用正常

5. **编译验证**
   - 编译ophost动态库验证tiling
   - 编译kernel二进制验证入口

#### Phase 2: Kernel实现（后续阶段）

基于第一阶段框架，逐步补充kernel实现代码（详见srs_chapter8_encoding_plan.md）。

### 关键参考文件

| 参考文件 | 用途 |
|---------|------|
| `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.h` | GMMFR MxA8W4 tiling模板 |
| `gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp` | GMMFR MxA8W4 tiling实现 |
| `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/grouped_matmul_finalize_routing_apt.cpp` | GMMFR kernel入口 |
| `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_key.h` | GMMFR tiling key |
| `gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_tiling_data.h` | GMMFR tiling data |
| `gmm/grouped_matmul_swiglu_quant_v2/.agent/raw/docs/` | 代码走读文档（8份） |
| `gmm/grouped_matmul_swiglu_quant_v2/.agent/raw/development/srs_chapter8_encoding_plan.md` | 详细编码方案 |
