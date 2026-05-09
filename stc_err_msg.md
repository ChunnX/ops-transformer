# GMM/GMMFR/NPU_FORMAT_CAST 算子错误码规范化整改
# STC软件测试用例与测试报告（System Test Case）

## 1. 测试策略概述

### 1.1 测试目标

**主要目标：**
- 验证错误码规范化整改后，错误信息上报正确性
- 验证 EZ0008-EZ0024 错误码按规范上报
- 验证错误日志格式符合 CANN 错误码规范
- 验证参数名称准确传递到错误信息
- 验证整改后功能不受影响

**测试范围：**
| 算子 | 整改文件数 | 整改错误数 | 测试重点 |
| --- | ---: | ---: | --- |
| GMM | 6个重点文件 | ~341条 | A5伪量化场景错误码上报 |
| GMMFR | 2个重点文件 | ~130条 | A5伪量化场景错误码上报 |
| NPU_FORMAT_CAST | 1个文件 | ~61条 | format转换场景错误码上报 |

### 1.2 测试分层

| 测试层级 | 测试类型 | 测试对象 | 验证内容 | 测试工具 |
| --- | --- | --- | --- | --- |
| **UT** | 单元测试 | op_api/op_host函数 | 错误码上报正确性、参数验证逻辑 | gtest |
| **ST** | 系统测试 | ACLNN API调用 | 错误日志输出格式、端到端错误传递 | Python ACL脚本 |
| **冒烟测试** | 快速验证 | 关键场景 | 基本功能正确性、编译通过性 | build.sh + UT |

### 1.3 测试设计原则

**ErrMsg整改测试设计原则：**
1. **错误场景全覆盖**：覆盖所有整改的错误类型（dtype/format/shape/dim/value等）
2. **错误码精确验证**：验证上报的EZ错误码与错误类型匹配
3. **参数名准确传递**：验证错误日志中的参数名与实际参数名一致
4. **错误信息准确性**：验证错误原因描述准确清晰
5. **反向测试为主**：重点测试异常场景的错误上报，而非正常功能

---

## 2. UT测试用例设计（Unit Test）

### 2.1 UT测试用例脑图

```
UT测试用例脑图
├─ 1. 参数验证错误测试
│  ├─ 1.1 dtype错误（EZ0018/EZ0019/EZ0020/EZ0021）
│  │  ├─ 单参数dtype错误 → EZ0019
│  │  ├─ 多参数dtype不一致 → EZ0021
│  │  ├─ dtype不支持 → EZ0020
│  │  └─ dtype转换错误 → EZ0018
│  ├─ 1.2 format错误（EZ0016/EZ0017/EZ0018）
│  │  ├─ format不匹配 → EZ0017
│  │  ├─ format不支持 → EZ0018
│  │  └─ format转换错误 → EZ0016
│  ├─ 1.3 shape错误（EZ0008/EZ0009/EZ0010）
│  │  ├─ shape不匹配 → EZ0008
│  │  ├─ shape维度错误 → EZ0009
│  │  ├─ 多参数shape错误 → EZ0010
│  │  └─ 空tensor错误 → EZ0009
│  ├─ 1.4 dim错误（EZ0011/EZ0012/EZ0013）
│  │  ├─ 维度数量错误 → EZ0011
│  │  ├─ 维度值错误 → EZ0012
│  │  └─ 多参数维度错误 → EZ0013
│  ├─ 1.5 value错误（EZ0023/EZ0024）
│  │  ├─ 参数值非法 → EZ0023
│  │  ├─ 参数值范围错误 → EZ0024
│  │  ├─ group参数错误 → EZ0023
│  │  ├─ scale参数错误 → EZ0023
│  │  └─ quant参数错误 → EZ0023
│  └─ 1.6 tensor num错误（EZ0022）
│     ├─ tensor数量错误 → EZ0022
│     └─ tensor list长度错误 → EZ0022
├─ 2. 内部错误测试（新增错误码）
│  ├─ 2.1 空指针错误
│  │  ├─ context空指针 → EZ_INNER_NULLPTR（上报新增）
│  │  ├─ tensor空指针 → EZ_PARAM_NULLPTR（评审决定）
│  │  └─ executor空指针 → EZ_INNER_CREATE_EXECUTOR（上报新增）
│  ├─ 2.2 运行时错误
│  │  ├─ 内存分配失败 → EZ_RUNTIME_ERROR（上报新增）
│  │  └─ 执行器创建失败 → EZ_INNER_CREATE_EXECUTOR（上报新增）
│  └─ 2.3 CHECK_RET错误
│     ├─ CHECK_RET失败 → EZ_INNER_NULLPTR（上报新增）
│     └─ 内部断言失败 → EZ_INNER_NULLPTR（上报新增）
├─ 3. OPS_REPORT分类测试
│  ├─ 3.1 参数验证类 → 改为EZ错误码
│  ├─ 3.2 内部错误类 → 上报新增错误码
│  └─ 3.3 业务逻辑类 → 保持不变
└─ 4. 跨算子对比测试
   ├─ 4.1 相同错误场景对比
   ├─ 4.2 不同算子错误码一致性
   └─ 4.3 错误信息格式统一性
```

### 2.2 GMM算子UT测试用例明细

#### 2.2.1 dtype错误测试用例（重点整改文件）

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-DTYPE-001 | dtype不支持错误 | x.dtype=DT_DOUBLE | x=[FLOAT16], weight=[INT8] | EZ0020 | "Parameter x has incorrect dtype DOUBLE" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-DTYPE-002 | dtype不一致错误 | x.dtype ≠ weight.dtype | x=[FLOAT16], weight=[INT8] | EZ0021 | "Parameters x, weight have incorrect dtypes" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-DTYPE-003 | 量化dtype错误 | quant_dtype非法 | x=[INT8], scale=[FLOAT] | EZ0020 | "Parameter scale has incorrect dtype" | grouped_weight_quant_batch_matmul_tiling.cpp |
| UT-GMM-DTYPE-004 | weight dtype错误 | weight.dtype不支持 | weight=[DT_UINT64] | EZ0019 | "Parameter weight has incorrect dtype UINT64" | grouped_matmul_weight_quant_950_checker.cpp |
| UT-GMM-DTYPE-005 | scale dtype不匹配 | scale.dtype与x不匹配 | x=[INT8], scale=[BF16] | EZ0021 | "Parameters x, scale have incorrect dtypes" | grouped_matmul_weight_quant_950_checker.cpp |

#### 2.2.2 shape错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-SHAPE-001 | shape不匹配错误 | x.shape与weight不匹配 | x=[64,256], weight=[128,512] | EZ0008 | "Parameter x has incorrect shape [64,256]" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-SHAPE-002 | shape维度错误 | x.dim=1 | x=[256] | EZ0012 | "Parameter x has incorrect shape dim 1D" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-SHAPE-003 | 空tensor错误 | x.shape包含0 | x=[0,256] | EZ0009 | "Parameter x has incorrect shape [0,256]" | grouped_matmul_tiling.cpp |
| UT-GMM-SHAPE-004 | groupList shape错误 | groupList长度非法 | groupList=[30,60]（应为32,64） | EZ0023 | "Parameter groupList has incorrect value" | grouped_matmul_weight_quant_950_checker.cpp |
| UT-GMM-SHAPE-005 | k维度错误 | k=1（非法） | x=[64,1], weight=[1,128] | EZ0009 | "Parameter x has incorrect shape" | grouped_matmul_infershape.cpp |

#### 2.2.3 format错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-FORMAT-001 | format不支持错误 | x.format=NCHW | x.format=NCHW（应ND） | EZ0018 | "Parameter x has incorrect format NCHW" | grouped_matmul_infershape.cpp |
| UT-GMM-FORMAT-002 | format不匹配错误 | weight.format≠FRACTAL_NZ | weight.format=ND | EZ0017 | "Parameter weight has incorrect format ND" | grouped_weight_quant_batch_matmul_tiling.cpp |
| UT-GMM-FORMAT-003 | NZ格式转换错误 | dstFormat非法 | srcFormat=ND, dstFormat=NCWH | EZ0016 | "Parameters src, dst have incorrect formats" | grouped_weight_quant_batch_matmul_tiling.cpp |

#### 2.2.4 value错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 鄄期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-VALUE-001 | group参数错误 | groupList值非法 | groupList=[30,60] | EZ0023 | "Parameter groupList has incorrect value" | aclnn_grouped_matmul.cpp |
| UT-GMM-VALUE-002 | scale值错误 | scale=0 | scale=[0] | EZ0023 | "Parameter scale has incorrect value 0" | grouped_matmul_950_checker.cpp |
| UT-GMM-VALUE-003 | quantGroupSize错误 | quantGroupSize非法 | quantGroupSize=0 | EZ0023 | "Parameter quantGroupSize has incorrect value" | grouped_matmul_weight_quant_950_checker.cpp |
| UT-GMM-VALUE-004 | splitItem错误 | splitItem非法 | splitItem=5 | EZ0023 | "Parameter splitItem has incorrect value" | grouped_matmul_tiling.cpp |
| UT-GMM-VALUE-005 | groupType错误 | groupType非法 | groupType=3 | EZ0023 | "Parameter groupType has incorrect value" | grouped_matmul_infershape.cpp |

### 2.3 GMMFR算子UT测试用例明细

#### 2.3.1 dtype错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMMFR-DTYPE-001 | x dtype错误 | x.dtype不支持 | x=[DT_UINT64] | EZ0020 | "Parameter x has incorrect dtype UINT64" | grouped_matmul_finalize_routing_weight_quant_950_checker.h |
| UT-GMMFR-DTYPE-002 | weight dtype错误 | weight.dtype不支持 | weight=[DT_UINT64] | EZ0019 | "Parameter weight has incorrect dtype" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |

#### 2.3.2 shape错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMMFR-SHAPE-001 | x shape错误 | x.shape不匹配 | x=[64,128,256] | EZ0008 | "Parameter x has incorrect shape" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |
| UT-GMMFR-SHAPE-002 | weight shape错误 | weight.shape维度错误 | weight.dim=1 | EZ0012 | "Parameter weight has incorrect shape dim" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |
| UT-GMMFR-SHAPE-003 | routing shape错误 | routing.shape不匹配 | routing=[64,32] | EZ0008 | "Parameter routing has incorrect shape" | grouped_matmul_finalize_routing_weight_quant_950_checker.h |

#### 2.3.3 value错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 鄄期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMMFR-VALUE-001 | expertNum错误 | expertNum非法 | expertNum=0 | EZ0023 | "Parameter expertNum has incorrect value" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |
| UT-GMMFR-VALUE-002 | topK错误 | topK非法 | topK=0 | EZ0023 | "Parameter topK has incorrect value" | grouped_matmul_finalize_routing_weight_quant_950_checker.h |

### 2.4 NPU_FORMAT_CAST算子UT测试用例明细

#### 2.4.1 dtype错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-FORMATCAST-DTYPE-001 | src dtype错误 | src.dtype不支持 | src=[DT_DOUBLE] | EZ0020 | "Parameter srcTensor has incorrect dtype DOUBLE" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-DTYPE-002 | dst dtype错误 | dst.dtype不支持 | dst=[DT_DOUBLE] | EZ0019 | "Parameter dstTensor has incorrect dtype" | aclnn_npu_format_cast.cpp |

#### 2.4.2 format错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-FORMATCAST-FORMAT-001 | src format错误 | src.format不支持 | src.format=FRACTAL_Z | EZ0018 | "Parameter srcTensor has incorrect format FRACTAL_Z" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-FORMAT-002 | dst format错误 | dst.format不支持 | dst.format=NCHW | EZ0017 | "Parameter dstTensor has incorrect format NCHW" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-FORMAT-003 | format转换不支持 | src→dst不支持 | src=ND, dst=FRACTAL_Z_3D | EZ0016 | "Parameters srcTensor, dstTensor have incorrect formats" | aclnn_npu_format_cast.cpp |

#### 2.4.3 shape错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-FORMATCAST-SHAPE-001 | src shape错误 | src.dim不在4-8范围 | src.dim=3 | EZ0012 | "Parameter srcTensor has incorrect shape dim 3D" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-SHAPE-002 | shape不匹配错误 | src→dst shape不匹配 | src=[1024,256], dst=[1024] | EZ0008 | "Parameter srcTensor has incorrect shape" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-SHAPE-003 | 空tensor错误 | src.shape包含0 | src=[0,256] | EZ0009 | "Parameter srcTensor has incorrect shape [0,256]" | aclnn_npu_format_cast.cpp |

#### 2.4.4 contiguous错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 鄄期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-FORMATCAST-CONTIG-001 | src不连续 | src.contiguous=false | src非连续 | EZ0023 | "Parameter srcTensor is not contiguous" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-CONTIG-002 | dst不连续 | dst.contiguous=false | dst非连续 | EZ0023 | "Parameter dstTensor is not contiguous" | aclnn_npu_format_cast.cpp |

#### 2.4.5 运行时错误测试用例（上报新增）

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-FORMATCAST-RUNTIME-001 | 内存分配失败 | new失败 | 大shape tensor | EZ_RUNTIME_ERROR（新增） | "Failed to allocate memory" | aclnn_npu_format_cast.cpp |
| UT-FORMATCAST-RUNTIME-002 | executor创建失败 | executor=null | executor创建失败 | EZ_INNER_CREATE_EXECUTOR（新增） | "Failed to create executor" | aclnn_npu_format_cast.cpp |

---

## 3. ST测试用例设计（System Test）

### 3.1 ST测试用例脑图

```
ST测试用例脑图
├─ 1. 端到端错误日志验证
│  ├─ 1.1 错误码输出验证
│  │  ├─ ACL日志中EZ码正确性
│  │  ├─ 错误码与错误类型匹配
│  │  └─ 错误码格式符合规范
```

### 3.2 ST测试用例明细表

#### 3.2.1 端到端错误日志验证测试用例

| 用例ID | 用例名称 | 测试场景 | 测试步骤 | 验证点 | 预期结果 |
| --- | --- | --- | --- | --- | --- |
| ST-ERRLOG-001 | dtype错误日志验证 | GMM dtype错误场景 | 1.调用ACLNN API传入非法dtype<br>2.捕获ACL日志<br>3.解析错误码 | 1.错误码为EZ0020<br>2.参数名正确显示<br>3.错误原因清晰 | 日志包含"EZ0020"、"Parameter x has incorrect dtype"、"Reason" |
| ST-ERRLOG-002 | shape错误日志验证 | GMMFR shape错误场景 | 1.调用ACLNN API传入非法shape<br>2.捕获ACL日志<br>3.解析错误信息 | 1.错误码为EZ0008<br>2.实际shape显示<br>3.期望shape显示 | 日志包含"EZ0008"、"incorrect shape [64,128]"、"correct shape [128,256]" |
| ST-ERRLOG-003 | format错误日志验证 | NPU_FORMAT_CAST format错误 | 1.调用ACLNN API传入非法format<br>2.捕获ACL日志<br>3.验证错误信息 | 1.错误码为EZ0017<br>2.srcFormat显示<br>3.dstFormat显示 | 日志包含"EZ0017"、"incorrect format NHWC" |
| ST-ERRLOG-004 | value错误日志验证 | GMM group参数错误 | 1.调用ACLNN API传入非法groupList<br>2.捕获ACL日志<br>3.验证错误描述 | 1.错误码为EZ0023<br>2.参数名groupList<br>3.错误原因详细 | 日志包含"EZ0023"、"Parameter groupList"、"Reason: groupList sum mismatch" |

#### 3.2.2 多算子一致性验证测试用例

| 用例ID | 用例名称 | 测试场景 | 测试步骤 | 验证点 | 预期结果 |
| --- | --- | --- | --- | --- | --- |
| ST-CONSIST-001 | dtype错误码一致性 | GMM vs GMMFR dtype错误 | 1.触发GMM dtype错误<br>2.触发GMMFR dtype错误<br>3.对比错误码 | 1.错误码相同<br>2.日志格式相同<br>3.参数命名风格相同 | 两者错误码均为EZ0020，日志格式一致 |
| ST-CONSIST-002 | shape错误码一致性 | GMM vs NPU_FORMAT_CAST shape错误 | 1.触发GMM shape错误<br>2.触发FORMAT_CAST shape错误<br>3.对比错误信息 | 1.错误码相同<br>2.错误描述风格相同 | 两者错误码均为EZ0008，描述风格一致 |
| ST-CONSIST-003 | SOC版本兼容性 | A5 vs 910B错误上报 | 1.在A5环境触发错误<br>2.在910B环境触发错误<br>3.对比错误上报 | 1.错误码一致<br>2.错误信息一致 | 不同SOC版本错误上报一致 |

#### 3.2.3 性能影响验证测试用例

| 用例ID | 用例名称 | 测试场景 | 测试步骤 | 验证点 | 预期结果 |
| --- | --- | --- | --- | --- | --- |
| ST-PERF-001 | 正常场景性能对比 | GMM正常推理 | 1.整改前性能测试<br>2.整改后性能测试<br>3.对比性能数据 | 1.性能无退化<br>2.耗时增加<5% | 整改后性能基本不变 |
| ST-PERF-002 | 错误上报性能 | 错误场景响应时间 | 1.触发错误场景<br>2.测量错误上报耗时<br>3.与整改前对比 | 1.错误上报耗时合理<br>2.不超过100ms | 错误上报性能正常 |
| ST-PERF-003 | 内存占用验证 | 长时间运行 | 1.连续触发1000次错误<br>2.监控内存占用<br>3.检查内存泄漏 | 1.内存不泄漏<br>2.内存占用稳定 | 内存占用无异常增长 |

#### 3.2.4 真实场景验证测试用例

| 用例ID | 用例名称 | 测试场景 | 测试步骤 | 验证点 | 预期结果 |
| --- | --- | --- | --- | --- | --- |
| ST-REAL-001 | MoE模型错误定位 | MoE模型中GMM错误 | 1.运行MoE模型<br>2.触发GMM错误<br>3.验证错误信息 | 1.错误信息可定位到具体参数<br>2.错误原因清晰 | 错误日志可快速定位问题根源 |
| ST-REAL-002 | 量化模型format错误 | 量化模型format转换错误 | 1.运行量化模型<br>2.触发format错误<br>3.验证错误信息 | 1.错误码准确<br>2.format转换路径显示 | 错误信息包含src→dst format转换路径 |
| ST-REAL-003 | 多线程并发错误 | 多线程并发触发错误 | 1.10线程并发触发错误<br>2.收集所有错误日志<br>3.验证日志完整性 | 1.日志不丢失<br>2.日志不混淆<br>3.错误码不冲突 | 所有错误日志完整且正确 |

---

## 4. ErrMsg整改专项测试设计

### 4.1 整改前后对比测试矩阵

| 整改点 | 整改前错误码 | 整改后错误码 | 测试验证点 | 测试方法 |
| --- | --- | --- | --- | --- |
| dtype错误 | EZ9999（通用） | EZ0020/EZ0021 | 错误码精确性、参数名准确性 | UT反向测试 + ST日志解析 |
| format错误 | EZ9999（通用） | EZ0017/EZ0018 | 错误码精确性、format路径显示 | UT反向测试 + ST端到端验证 |
| shape错误 | EZ9999（通用） | EZ0008/EZ0009 | 错误码精确性、shape值对比 | UT反向测试 + ST日志验证 |
| dim错误 | EZ9999（通用） | EZ0011/EZ0012 | 错误码精确性、维度显示 | UT反向测试 + ST日志验证 |
| value错误 | EZ9999（通用） | EZ0023 | 错误码精确性、错误原因详细 | UT反向测试 + ST日志验证 |
| CHECK_RET错误 | ACLNN_ERR_INNER_NULLPTR | EZ_INNER_NULLPTR（新增） | 新错误码上报正确性 | UT Mock测试 + ST错误触发 |
| OPS_REPORT | OPS_REPORT | 按内容分类为EZ码 | 分类正确性、错误码映射 | UT分类测试 + ST日志验证 |

### 4.2 重点整改文件专项测试

#### 4.2.1 GMM重点整改文件测试（grouped_matmul_weight_quant_950_checker.cpp）

| 测试ID | 测试场景 | 错误类型 | 整改数量 | 测试用例数 | 优先级 |
| --- | --- | --- | ---: | ---: | --- |
| TEST-GMM-WQ-001 | A5伪量化dtype错误 | dtype | 24条 | 5个 | P0 |
| TEST-GMM-WQ-002 | A5伪量化shape错误 | shape | 30条 | 8个 | P0 |
| TEST-GMM-WQ-003 | A5伪量化scale错误 | value | 47条 | 6个 | P1 |
| TEST-GMM-WQ-004 | A5伪量化group错误 | value | 24条 | 4个 | P1 |

**测试策略：**
- 针对每条整改点设计1-2个反向测试用例
- 重点验证错误码从EZ9999改为精确EZ码
- 验证参数名从"参数X"改为具体参数名（如"x", "weight", "scale"）

#### 4.2.2 GMMFR重点整改文件测试（grouped_matmul_finalize_routing_weight_quant_tiling.cpp）

| 测试ID | 测试场景 | 错误类型 | 整改数量 | 测试用例数 | 优先级 |
| --- | --- | --- | ---: | ---: | --- |
| TEST-GMMFR-WQ-001 | A5伪量化shape错误 | shape | 30条 | 6个 | P0 |
| TEST-GMMFR-WQ-002 | A5伪量化dim错误 | dim | 15条 | 4个 | P1 |
| TEST-GMMFR-WQ-003 | A5伪量化value错误 | value | 39条 | 5个 | P1 |

#### 4.2.3 NPU_FORMAT_CAST整改文件测试（aclnn_npu_format_cast.cpp）

| 测试ID | 测试场景 | 错误类型 | 整改数量 | 测试用例数 | 优先级 |
| --- | --- | --- | ---: | ---: | --- |
| TEST-FORMATCAST-001 | format转换错误 | format | 15条 | 4个 | P0 |
| TEST-FORMATCAST-002 | shape维度错误 | shape | 9条 | 3个 | P0 |
| TEST-FORMATCAST-003 | dtype不支持 | dtype | 8条 | 2个 | P1 |
| TEST-FORMATCAST-004 | contiguous错误 | value | 3条 | 2个 | P2 |
| TEST-FORMATCAST-005 | 运行时错误 | runtime | 13条 | 3个 | P1（上报新增） |

### 4.3 错误信息格式验证测试

**验证规范：**

| 错误码 | 标准格式 | 测试验证点 |
| --- | --- | --- |
| EZ0008 | "Parameter {paramName} of operator {opName} has incorrect shape [{actual}]. The correct one should be [{expected}]." | 1.paramName准确<br>2.opName显示<br>3.shape对比清晰 |
| EZ0011 | "Parameter {paramName} of operator {opName} has incorrect shape dim {actual}. The correct one should be {expected}." | 1.paramName准确<br>2.dim显示清晰 |
| EZ0017 | "Parameter {paramName} of operator {opName} has incorrect format {actual}. The correct one should be {expected}." | 1.paramName准确<br>2.format显示 |
| EZ0020 | "Parameter {paramName} of operator {opName} has incorrect dtype {actual}. Reason: {reason}." | 1.paramName准确<br>2.dtype显示<br>3.reason详细 |
| EZ0023 | "Parameter {paramName} of operator {opName} has incorrect value {actual}. Reason: {reason}." | 1.paramName准确<br>2.value显示<br>3.reason详细 |

---

## 5. 测试实施计划

### 5.1 测试阶段划分

| Phase | 阶段名称 | 时间节点 | 测试内容 | 测试类型 | 输出物 |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | 测试用例设计 | 5.9-5.12 | 设计ErrMsg整改专项测试用例 | - | 测试用例文档、测试数据 |
| Phase 1 | UT测试执行 | 5.18-5.20 | 执行整改文件的UT测试 | pytest/gtest | UT测试报告 |
| Phase 2 | ST测试执行 | 5.22-5.25 | 执行端到端ST测试 | Python ACL脚本 | ST测试报告 |
| Phase 3 | 冒烟测试 | 5.28-5.29 | 整改后冒烟测试验证 | build.sh + UT | 冒烟测试报告 |

### 5.2 测试数据准备

**测试数据分类：**
1. **正常场景数据**：验证整改后功能不受影响
2. **边界场景数据**：shape边界值、dtype边界值等
3. **异常场景数据**：非法dtype、非法shape、非法format等

**数据生成脚本：**
```python
# 生成异常测试数据的示例脚本
def generate_dtype_error_test_data():
    """生成dtype错误测试数据"""
    test_cases = [
        {"x_dtype": "DOUBLE", "weight_dtype": "INT8", "expect_err": "EZ0020"},
        {"x_dtype": "FLOAT16", "weight_dtype": "BF16", "expect_err": "EZ0021"},
    ]
    return test_cases

def generate_shape_error_test_data():
    """生成shape错误测试数据"""
    test_cases = [
        {"x_shape": [0, 256], "weight_shape": [256, 128], "expect_err": "EZ0009"},
        {"x_shape": [64, 1], "weight_shape": [1, 128], "expect_err": "EZ0009"},
    ]
    return test_cases
```

---

## 6. 测试验收标准

### 6.1 UT测试验收标准

| 验收项 | 验收标准 | 验证方法 |
| --- | --- | --- |
| 错误码精确性 | 所有错误场景上报精确EZ码（非EZ9999） | UT反向测试通过率100% |
| 参数名准确性 | 错误信息中参数名与实际参数名一致 | UT日志解析验证 |
| 错误描述准确性 | 错误原因描述清晰、准确 | UT错误信息关键字匹配 |

### 6.2 ST测试验收标准

| 验收项 | 验收标准 | 验证方法 |
| --- | --- | --- |
| 端到端错误传递 | ACL日志中错误码正确传递 | ST日志解析验证 |
| 错误信息完整性 | 错误信息包含所有必要字段 | ST日志字段检查 |
| 多算子一致性 | 相同错误场景错误码一致 | ST对比测试通过 |
| 真实场景可用性 | 错误信息可定位问题根源 | 真实场景测试通过 |

---

### 2. 典型错误日志示例

**示例1：dtype错误**
```
[ERROR] EZ0020: Parameter x of operator grouped_matmul has incorrect dtype DOUBLE. 
Reason: Only support INT8/FLOAT16/BF16.
```

**示例2：shape错误**
```
[ERROR] EZ0008: Parameter x of operator grouped_matmul has incorrect shape [64,128]. 
The correct one should be [128,256].
```
