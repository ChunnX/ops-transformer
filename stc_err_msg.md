# GMM/GMMFR 算子错误码规范化整改
# STC软件测试用例与测试报告（System Test Case）

## 1. 测试策略概述

### 1.1 测试目标

**主要目标：**
- 验证错误码规范化整改后，错误信息上报正确性
- 验证 EZ0008-EZ0017 错误码按规范上报
- 验证错误日志格式符合 CANN 错误码规范
- 验证参数名称准确传递到错误信息
- 验证整改后功能不受影响

**测试范围：**
| 算子 | 整改文件数 | 整改错误数 | 测试重点 |
| --- | ---: | ---: | --- |
| GMM | 3个重点文件 | ~341条 | A5伪量化场景错误码上报 |
| GMMFR | 1个重点文件 | ~130条 | A5伪量化场景错误码上报 |

### 1.2 测试分层

| 测试层级 | 测试类型 | 测试对象 | 验证内容 | 测试工具 |
| --- | --- | --- | --- | --- |
| **UT** | 单元测试 | op_api/op_host函数 | 错误码上报正确性、参数验证逻辑 | gtest/ttk |

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
│  ├─ 1.1 输入shape错误（EZ0008/EZ0009/EZ0010/EZ0011）
│  │  ├─ shape不匹配 → EZ0008
│  │  ├─ shape维度错误 → EZ0009
│  │  ├─ shape错误含原因 → EZ0010
│  │  └─ shape size错误 → EZ0011
│  ├─ 1.2 输出shape错误（EZ0012/EZ0013/EZ0014/EZ0015）
│  │  ├─ 输出shape不匹配 → EZ0012
│  │  ├─ 输出shape维度错误 → EZ0013
│  │  ├─ 输出shape错误含原因 → EZ0014
│  │  └─ 输出shape size错误 → EZ0015
│  ├─ 1.3 输出格式错误（EZ0016/EZ0017）
│  │  ├─ 输出format不匹配 → EZ0016
│  │  └─ 输出dtype不匹配 → EZ0017
│  ├─ 1.4 输入format/dtype错误（旧版宏EZ0006/EZ0007）
│  │  ├─ 输入format不匹配 → EZ0006
│  │  └─ 输入dtype不匹配 → EZ0007
│  └─1.5 其他参数错误（新增错误码）
│     ├─ 参数值非法
│     ├─ 参数值范围错误
│     ├─ group参数错误
│     ├─ scale参数错误
│     └─ quant参数错误
└─ 2. 内部错误测试（新增错误码）
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
```

### 2.2 GMM算子UT测试用例明细

#### 2.2.1 dtype错误测试用例（重点整改文件）

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-DTYPE-001 | dtype不支持错误 | x.dtype=DT_DOUBLE | x=[FLOAT16], weight=[INT8] | EZ0007 | "has incorrect dtype DOUBLE" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-DTYPE-002 | dtype不一致错误 | x.dtype ≠ weight.dtype | x=[FLOAT16], weight=[INT8] | EZ0007 | "Parameters x, weight have incorrect dtypes" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-DTYPE-003 | 量化dtype错误 | quant_dtype非法 | x=[INT8], scale=[FLOAT] | EZ0007 | "Parameter scale has incorrect dtype" | grouped_weight_quant_batch_matmul_tiling.cpp |
| UT-GMM-DTYPE-004 | weight dtype错误 | weight.dtype不支持 | weight=[DT_UINT64] | EZ0007 | "Parameter weight has incorrect dtype UINT64" | grouped_matmul_weight_quant_950_checker.cpp |
| UT-GMM-DTYPE-005 | scale dtype不匹配 | scale.dtype与x不匹配 | x=[INT8], scale=[BF16] | EZ0007 | "Parameters x, scale have incorrect dtypes" | grouped_matmul_weight_quant_950_checker.cpp |

#### 2.2.2 shape错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-SHAPE-001 | shape不匹配错误 | x.shape与weight不匹配 | x=[64,256], weight=[128,512] | EZ0008 | "input x has incorrect shape [64,256]" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-SHAPE-002 | shape维度错误 | x.dim=1 | x=[256] | EZ0009 | "input x has incorrect shape dim [1D]" | grouped_matmul_infershape_weight_quant_checker.cpp |
| UT-GMM-SHAPE-003 | 空tensor错误 | x.shape包含0 | x=[0,256] | EZ0010 | "input x has incorrect shape [0,256]" | grouped_matmul_tiling.cpp |
| UT-GMM-SHAPE-004 | groupList shape错误 | groupList长度非法 | groupList=[30,60]（应为32,64） | EZ9999 | "Parameter groupList has incorrect value" | grouped_matmul_weight_quant_950_checker.cpp |
| UT-GMM-SHAPE-005 | k维度错误 | k=1（非法） | x=[64,1], weight=[1,128] | EZ0009 | "input x has incorrect shape" | grouped_matmul_infershape.cpp |

#### 2.2.3 format错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-FORMAT-001 | format不支持错误 | x.format=NCHW | x.format=NCHW（应ND） | EZ0006 | "has incorrect format NCHW" | grouped_matmul_infershape.cpp |
| UT-GMM-FORMAT-002 | format不匹配错误 | weight.format≠FRACTAL_NZ | weight.format=ND | EZ0006 | "has incorrect format ND" | grouped_weight_quant_batch_matmul_tiling.cpp |
| UT-GMM-FORMAT-003 | NZ格式转换错误 | dstFormat非法 | srcFormat=ND, dstFormat=NCWH | EZ0016 | "output has incorrect format" | grouped_weight_quant_batch_matmul_tiling.cpp |

#### 2.2.4 value错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMM-VALUE-001 | group参数错误 | groupList值非法 | groupList=[30,60] | EZ9999 | "Parameter groupList has incorrect value" | aclnn_grouped_matmul.cpp |
| UT-GMM-VALUE-002 | scale值错误 | scale=0 | scale=[0] | EZ9999 | "Parameter scale has incorrect value 0" | grouped_matmul_950_checker.cpp |
| UT-GMM-VALUE-003 | quantGroupSize错误 | quantGroupSize非法 | quantGroupSize=0 | EZ9999 | "Parameter quantGroupSize has incorrect value" | grouped_matmul_weight_quant_950_checker.cpp |
| UT-GMM-VALUE-004 | splitItem错误 | splitItem非法 | splitItem=5 | EZ9999 | "Parameter splitItem has incorrect value" | grouped_matmul_tiling.cpp |
| UT-GMM-VALUE-005 | groupType错误 | groupType非法 | groupType=3 | EZ9999 | "Parameter groupType has incorrect value" | grouped_matmul_infershape.cpp |

### 2.3 GMMFR算子UT测试用例明细

#### 2.3.1 dtype错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMMFR-DTYPE-001 | x dtype错误 | x.dtype不支持 | x=[DT_UINT64] | EZ0007 | "has incorrect dtype UINT64" | grouped_matmul_finalize_routing_weight_quant_950_checker.h |
| UT-GMMFR-DTYPE-002 | weight dtype错误 | weight.dtype不支持 | weight=[DT_UINT64] | EZ0007 | "has incorrect dtype" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |

#### 2.3.2 shape错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMMFR-SHAPE-001 | x shape错误 | x.shape不匹配 | x=[64,128,256] | EZ0008 | "input x has incorrect shape" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |
| UT-GMMFR-SHAPE-002 | weight shape错误 | weight.shape维度错误 | weight.dim=1 | EZ0009 | "input weight has incorrect shape dim" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |
| UT-GMMFR-SHAPE-003 | routing shape错误 | routing.shape不匹配 | routing=[64,32] | EZ0008 | "input routing has incorrect shape" | grouped_matmul_finalize_routing_weight_quant_950_checker.h |

#### 2.3.3 value错误测试用例

| 用例ID | 用例名称 | 测试场景 | 输入参数 | 预期错误码 | 预期错误信息关键字 | 文件 |
| --- | --- | --- | --- | --- | --- | --- |
| UT-GMMFR-VALUE-001 | expertNum错误 | expertNum非法 | expertNum=0 | EZ9999 | "Parameter expertNum has incorrect value" | grouped_matmul_finalize_routing_weight_quant_tiling.cpp |
| UT-GMMFR-VALUE-002 | topK错误 | topK非法 | topK=0 | EZ9999 | "Parameter topK has incorrect value" | grouped_matmul_finalize_routing_weight_quant_950_checker.h |

## 3. ErrMsg整改专项测试设计

### 3.1 整改前后对比测试矩阵

| 整改点 | 整改前错误码 | 整改后错误码 | 测试验证点 | 测试方法 |
| --- | --- | --- | --- | --- |
| dtype错误 | EZ9999（通用） | EZ0007（输入dtype） | 错误码精确性、参数名准确性 | UT反向测试 + ST日志解析 |
| format错误 | EZ9999（通用） | EZ0006（输入format）/EZ0016（输出format） | 错误码精确性、format路径显示 | UT反向测试 + ST端到端验证 |
| shape错误 | EZ9999（通用） | EZ0008（输入shape）/EZ0012（输出shape） | 错误码精确性、shape值对比 | UT反向测试 + ST日志验证 |
| dim错误 | EZ9999（通用） | EZ0009（输入dim）/EZ0013（输出dim） | 错误码精确性、维度显示 | UT反向测试 + ST日志验证 |
| value错误 | EZ9999（通用） | EZ9999（通用） | 错误原因详细 | UT反向测试 + ST日志验证 |

### 3.2 重点整改文件专项测试

#### 3.2.1 GMM重点整改文件测试（grouped_matmul_weight_quant_950_checker.cpp）

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

#### 3.2.2 GMMFR重点整改文件测试（grouped_matmul_finalize_routing_weight_quant_tiling.cpp）

| 测试ID | 测试场景 | 错误类型 | 整改数量 | 测试用例数 | 优先级 |
| --- | --- | --- | ---: | ---: | --- |
| TEST-GMMFR-WQ-001 | A5伪量化shape错误 | shape | 30条 | 6个 | P0 |
| TEST-GMMFR-WQ-002 | A5伪量化dim错误 | dim | 15条 | 4个 | P1 |
| TEST-GMMFR-WQ-003 | A5伪量化value错误 | value | 39条 | 5个 | P1 |

### 3.3 错误信息格式验证测试

**验证规范：**

| 错误码 | 标准格式 | 测试验证点 |
| --- | --- | --- |
| EZ0008 | "OP {opName} input {inputName} has incorrect shape [{incorrectShape}], it should be [{correctShape}]." | 1.inputName准确<br>2.opName显示<br>3.shape对比清晰 |
| EZ0009 | "OP {opName} input {inputName} has incorrect shape dim [{incorrectDim}], it should be [{correctDim}]." | 1.inputName准确<br>2.dim显示清晰 |
| EZ0011 | "OP {opName} input {inputName} has incorrect shape size {incorrectSize}, it should be {correctSize}." | 1.inputName准确<br>2.size显示清晰 |
| EZ0016 | "OP {opName} output {outputName} has incorrect format {incorrectFormat}, it should be {correctFormat}." | 1.outputName准确<br>2.format显示 |
| EZ0017 | "OP {opName} output {outputName} has incorrect dtype {incorrectDtype}, it should be {correctDtype}." | 1.outputName准确<br>2.dtype显示 |

---

### 5. 错误码说明

**重要说明：**
根据 `/home/developer/Ascend/cann/aarch64-linux/pkg_inc/op_common/log/log.h` 文件，当前 CANN 定义的标准错误码如下：

**输入验证错误码（EZ0008-EZ0011）：**
- EZ0008: 输入shape异常 - "OP {opName} input {inputName} has incorrect shape [{incorrectShape}], it should be [{correctShape}]."
- EZ0009: 输入shape dim异常 - "OP {opName} input {inputName} has incorrect shape dim [{incorrectDim}], it should be [{correctDim}]."
- EZ0010: 输入shape异常（含原因） - "OP {opName} input {inputName} has incorrect shape [{incorrectShape}], reason: {reason}."
- EZ0011: 输入shape size异常 - "OP {opName} input {inputName} has incorrect shape size {incorrectSize}, it should be {correctSize}."

**输出验证错误码（EZ0012-EZ0017）：**
- EZ0012: 输出shape异常
- EZ0013: 输出shape dim异常
- EZ0014: 输出shape异常（含原因）
- EZ0015: 输出shape size异常
- EZ0016: 输出format异常
- EZ0017: 输出dtype异常

**旧版输入验证宏（EZ0001-EZ0007）：**
- EZ0001: 输入shape异常（使用序号index）
- EZ0006: 输入format异常（使用序号index）
- EZ0007: 输入dtype异常（使用序号index）

**通用错误码：**
- EZ9999: 通用内部错误 - 用于其他未明确定义的错误类型

**注意：**
1. dtype/format相关的参数错误应使用 EZ0006/EZ0007
2. value/tensor num等其他参数错误应使用 EZ9999（通用错误码）
3. 如需新增专门的错误码（如value错误、tensor num错误），需要向 CANN 团队提交错误码定义申请

**示例1：输入shape错误**
```
[ERROR] EZ0008: OP grouped_matmul input x has incorrect shape [64,128], it should be [128,256].
```

**示例2：输入shape dim错误**
```
[ERROR] EZ0009: OP grouped_matmul input x has incorrect shape dim [2D], it should be [3D].
```

**示例3：输出dtype错误**
```
[ERROR] EZ0017: OP grouped_matmul output y has incorrect dtype DOUBLE, it should be FLOAT16.
```
