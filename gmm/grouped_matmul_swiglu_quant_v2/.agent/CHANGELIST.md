# CHANGELIST - 变更记录

> 记录项目演进和变更，便于后期回溯

## 2026-05-13: 阶段二-Part1 框架搭建

### 新增文件
- `op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h` — MxA8W4 TilingData结构体定义
- `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h` — MxA8W4 Tiling模板头文件
- `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp` — MxA8W4 Tiling模板实现

### 修改文件
- `op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h` — 新增ATRANS/BTRANS/HASBIAS模板参数
- `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp` — 新增V310_GMM_SQ_ANTI_QUANT宏和MxA8W4 stub kernel分支
- `op_host/op_tiling/grouped_matmul_swiglu_quant_v2_tiling.cpp` — 新增MxA8W4场景分发逻辑，注册模板ID=3
- `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.h` — 移除ScenarioType枚举和GMMSQArch35WeightQuantTiling命名空间
- `op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp` — 简化InferScenario，移除scenarioType_

### 编译错误修复
- Namespace结构错误 → 将class移入正确namespace
- GetAttrPointer API不存在 → 改为GetAttrs()->GetAttrPointer<T>()
- GetShape()不存在 → 改为GetInputShape()->GetOriginShape()/GetStorageShape()
- GetDataType/GetStorageFormat在Shape上不存在 → 使用GetInputDesc()获取descriptor
- compileInfoPtr成员名错误 → aicNum→aicNum_, ubSize→ubSize_
- 索引常量未定义 → 定义GMMSQ_X_INDEX/GMMSQ_W_INDEX局部常量
- Header guard不匹配 → 修正为完整文件名格式

### 架构优化
- 移除ScenarioType枚举，由TilingFunc统一分发
- 移除GMMSQArch35WeightQuantTiling命名空间，简化代码结构
- TilingFunc中增加MxA8W4 dtype检测，只有匹配时才选择模板3
- 移除hasBias相关字段（算子不支持bias输入）：
  - TilingData中移除hasBias和reserve1，改为reserve0[3]
  - GMMSQWeightQuantInputParams中移除hasBias
  - GetTilingKey中移除hasBias参数
  - 设计文档中同步移除所有hasBias引用

### 2026-05-13: 修正输入索引错误

#### 问题
- `PERTOKEN_SCALE_INDEX` 命名错误，GMMSQ算子没有perTokenScale输入
- 根据算子原型定义，该输入应为 `xScale`，索引为1
- 原代码索引值全部错误（W_INDEX=1应为3，SCALE_INDEX=2应为4等）

#### 修正
- 重命名 `PERTOKEN_SCALE_INDEX` → `X_SCALE_INDEX`
- 修正所有输入索引值以匹配算子原型定义：
  - X_INDEX = 0 ✓
  - X_SCALE_INDEX = 1（原PERTOKEN_SCALE_INDEX=3，错误）
  - GROUPLIST_INDEX = 2（原=4，错误）
  - W_INDEX = 3（原=1，错误）
  - SCALE_INDEX = 4（原=2，错误）
  - BIAS_INDEX = 6
- 更新InferScenario中变量名：pertokenScaleDesc → xScaleDesc
- xScale从OptionalInput改为必选Input（GetInputDesc而非GetOptionalInputDesc）

### 2026-05-14: 阶段二-Part2 Tiling测试验证及代码完善

#### Tiling测试验证（🟡 进行中）
- 新增8个MxA8W4 CSV测试用例到`tests/ut/op_host/test_grouped_matmul_swiglu_quant_v2_tiling.csv`：
  - 正常用例：wTrans=false/true各1个
  - 边界用例：min K=32, min N=128, large E=64
  - 非法用例：K未对齐、bias提供、grouplisttype=cumsum
- 完善tiling校验逻辑（`op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp`）：
  - 新增`CheckGroupNum()`：校验groupNum <= 1024且为正数
  - 新增`CheckScaleShapes()`：校验wscale为4D、xscale为3D
  - 注册到validation chain
- 编译运行UT进行用例验证（待进行）

### 2026-05-14: 阶段二-Part3 Kernel代码开发

#### Kernel代码开发（🟡 进行中）
- 新增文件（6个kernel头文件）：
  - `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_cube_compute_tools.h` — Cube工具函数（从GMMFR复制）
  - `op_kernel/arch35/weight_quant_basic_block/gmmsq_mx_a8w4_vf.h` — VF函数（反量化+SwiGLU+Quant）
  - `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vec_compute.h` — VecCompute层
  - `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_cube_compute.h` — CubeCompute层
  - `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_vcv_basic_block.h` — BasicBlock层
  - `op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h` — Controller层
- 新增文件（3个common头文件，从GMMFR复制）：
  - `op_kernel/arch35/common/basic_block_config.h`
  - `op_kernel/arch35/common/basic_block_vf_mx.h`
  - `op_kernel/arch35/common/tool.h`
- 修改文件：
  - `op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp` — 替换stub为完整Controller实例化调用

#### 编译错误修复（多轮迭代）
- Namespace解析问题 → 所有kernel文件添加`using WeightQuantBatchMatmulV2::Arch35::...`声明
- 模板参数缺失 → 添加`hasBias`模板参数到vcv_basic_block和resplit_controller
- `MX_SCALE_K_L1_SIZE` → 修正为正确的常量名
- `CeilDivide`歧义 → 使用`WeightQuantBatchMatmulV2::Arch35::CeilDivide`
- `DataCopyPad2D`未声明 → 添加using声明
- `yScaleGmAddr`不存在 → 从BasicBlockOffsetParam引用中移除
- `GetGMMSQMxA8W4BufferInfo`命名空间错误 → 修正为GMMSQ namespace
- MicroAPI Cast trait → 添加`CAST_B16_TO_B32_TRAIT`参数

#### 待解决问题
- VF函数中MicroAPI调用（ShiftRight with uint32_t, FP8 Cast）与AscendC 9.1.0 API签名不匹配，需要参考官方文档修正
