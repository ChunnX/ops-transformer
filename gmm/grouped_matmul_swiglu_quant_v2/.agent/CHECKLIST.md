# CHECKLIST - 检查清单

> 第三方视角的客观审查标准，用于验证项目质量和目标达成。
> 审查要求：需要使用subagent，在独立的上下文进行客观、公正地检查，如果发现问题，必须诚实报告。

## 审查说明

本检查清单用于在项目开发各阶段进行独立审查。审查者应以第三方视角，客观、公正地评估每个检查项。

**审查原则**:
- 基于事实，不基于假设
- 可验证，可复现
- 明确通过/不通过/不适用

## 阶段二-Part1: 框架搭建检查清单（当前阶段）

### Tiling Key
- [x] Tiling Key文件存在: `op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h`
- [x] 复用现有QUANT_B_TRANS/QUANT_A_TRANS模板参数
- [x] MxA8W4场景不新增模板参数（无ATRANS/BTRANS/HASBIAS）
- [x] 保留全量化场景的QUANT_B_TRANS/QUANT_A_TRANS组合

### Tiling Data
- [x] Tiling Data文件存在: `op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h`
- [x] 结构体定义: `GMMSQWeightQuantTilingData`
- [x] 包含8个字段: groupListType, reserve0[3], groupNum, coreNum, outputBs, groupSize, kSize, nSize
- [x] 无hasBias字段（算子不支持bias输入）
- [x] 8字节对齐: `#pragma pack(push, 8)`
- [x] Header guard与文件名匹配: `GROUPED_MATMUL_SWIGLU_QUANT_V2_WEIGHT_QUANT_TILING_DATA_H`

### Tiling模板
- [x] Tiling模板头文件: `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h`
- [x] Tiling模板实现: `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp`
- [x] 类定义: `GroupedMatmulSwigluQuantV2WeightQuantTiling`，继承自`TilingBaseClass`
- [x] InferScenario: 校验MxA8W4 dtype (x=FP8_E4M3FN, w=FP4_E2M1, wFormat=FRACTAL_NZ, scale=FP8_E8M0)
- [x] DoOpTiling: 设置tiling data字段
- [x] GetTilingKey: 返回`GET_TPL_TILING_KEY(ATRANS=0, BTRANS, HASBIAS=0)`
- [x] 注册模板ID=3: `REGISTER_OPS_TILING_TEMPLATE(GroupedMatmulSwigluQuantV2, GroupedMatmulSwigluQuantV2WeightQuantTiling, 3)`

### Tiling分发逻辑
- [x] `GroupedMatmulSwigluQuantV2TilingFunc`中检测MxA8W4 dtype
- [x] MxA8W4场景选择模板3，其他场景保持原有逻辑
- [x] 移除ScenarioType枚举（由TilingFunc统一分发）
- [x] 移除GMMSQArch35WeightQuantTiling命名空间（名称已足够）

### Kernel入口
- [x] apt.cpp中新增`V310_GMM_SQ_ANTI_QUANT`宏定义
- [x] 新增MxA8W4场景的stub kernel调用分支
- [x] 包含tiling data头文件

### 编译验证
- [x] Ophost编译成功: `libophost_transformer.so`
- [ ] Opkernel编译验证（待kernel实现后）

### 代码规范
- [x] 文件名称与\file注释匹配
- [x] Header guard宏与文件名称匹配
- [x] 使用GetInputDesc/GetInputShape的正确API
- [x] 使用GetOriginShape/GetStorageShape获取shape
- [x] compileInfoPtr成员名使用下划线后缀(aicNum_, ubSize_)
