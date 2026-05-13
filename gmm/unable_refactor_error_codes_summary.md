# 无法整改错误码汇总报告

## 一、概述

本报告汇总了 `grouped_weight_quant_batch_matmul` 和 `grouped_matmul_finalize_routing` 两个算子中无法整改的错误码，并根据新的内部错误码规范（EZ0035-EZ0043）进行归类分析。

## 二、内部错误码映射表

| 新错误码 | 错误类型 | 适用场景 |
|---------|---------|---------|
| EZ0035 | Internal_Error_Nullptr | 空指针检查失败 |
| EZ0036 | Internal_Error_Memory_Allocation | 内存分配失败 |
| EZ0037 | Internal_Error_Tiling | Tiling计算/设置失败 |
| EZ0039 | Internal_Error_Memory_Operation | 内存操作失败（memcpy等） |
| EZ0040 | Internal_Error_Registry | 注册表相关错误 |
| EZ0041 | Internal_Error_Context | 上下文相关错误 |
| EZ0042 | Internal_Error_Function_Failed | 函数调用/验证失败 |
| EZ0043 | Internal_Error_Platform | 平台相关错误 |

## 三、grouped_weight_quant_batch_matmul 无法整改错误码分析

### 3.1 EZ0035 (Internal_Error_Nullptr) - 空指针错误

**场景说明：** 各种 Tensor、Desc、Shape、Attr 等关键对象的空指针检查

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| Desc[%zu] is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 179 | TensorDesc空指针检查 |
| Tensor[%zu] is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 180 | Tensor对象空指针检查 |
| Shape[%zu] is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 181 | Shape对象空指针检查 |
| x's tensor or Desc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 188 | 输入x空指针检查 |
| weight's tensor or Desc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 190 | 输入weight空指针检查 |
| antiquantscale's tensor or Desc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 192 | antiquantScale空指针检查 |
| bias's tensor or Desc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 195 | bias空指针检查 |
| antiquantoffset's tensor or Desc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 199 | antiquantOffset空指针检查 |
| %s[%zu] shape is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 210 | 动态参数Shape空指针检查 |
| GroupList is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 437 | GroupList空指针检查 |
| wTensor is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 441 | wTensor空指针检查 |
| compileInfoPtr is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 459 | 编译信息空指针检查 |
| attr is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 464 | 属性空指针检查 |
| xDesc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 514 | xDesc空指针检查 |
| wDesc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 518 | wDesc空指针检查 |
| biasDesc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 534 | biasDesc空指针检查 |
| antiquantScaleDesc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 539 | antiquantScaleDesc空指针检查 |
| antiquantOffsetDesc is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 550 | antiquantOffsetDesc空指针检查 |
| workspaces is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 734 | workspace空指针检查 |
| RawTilingData is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 739 | TilingData空指针检查 |
| perTokenScaleShapePtr is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 828 | perTokenScale Shape空指针检查 |
| xTensor is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 938 | xTensor空指针检查 |
| wTensor is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 942 | wTensor空指针检查 |
| yShapePtr is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 959 | yShapePtr空指针检查 |
| antiquantScale is nullptr. | grouped_weight_quant_batch_matmul_tiling.cpp | 1051 | antiquantScale空指针检查 |

**数量统计：** 25处

**整改建议：** 使用 `OP_LOGE_FOR_NULLPTR(opName, paramName)` 宏进行整改

### 3.2 EZ0037 (Internal_Error_Tiling) - Tiling错误

**场景说明：** Tiling策略计算失败、设置失败、结果无效

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| Unable to calculate resplit-tiling | grouped_weight_quant_batch_matmul_tiling.cpp | 50 | Tiling计算失败 |
| Set Base Tiling Failed | grouped_weight_quant_batch_matmul_tiling.cpp | 53 | Tiling设置失败 |
| Invalid resplit tiling result | grouped_weight_quant_batch_matmul_tiling.cpp | 601 | Tiling结果无效 |

**数量统计：** 3处

**整改建议：** 使用 `OP_LOGE_FOR_TILING_FAIL(opName, reason)` 宏进行整改

### 3.3 EZ0039 (Internal_Error_Memory_Operation) - 内存操作错误

**场景说明：** 内存拷贝操作失败

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| memcpy_s failed, ret = %d | grouped_weight_quant_batch_matmul_tiling.cpp | 743 | memcpy操作失败 |

**数量统计：** 1处

**整改建议：** 使用 `OP_LOGE_FOR_MEMCPY_FAIL(opName, ret)` 宏进行整改

### 3.4 EZ0042 (Internal_Error_Function_Failed) - 函数验证失败

**场景说明：** 各类验证函数失败、参数检查失败、复杂约束检查失败

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| Invalid attr param | grouped_weight_quant_batch_matmul_tiling.cpp | 47 | 属性参数验证失败 |
| Unable to set custom param | grouped_weight_quant_batch_matmul_tiling.cpp | 59 | 自定义参数设置失败 |
| Unable to get shape list | grouped_weight_quant_batch_matmul_tiling.cpp | 68 | Shape列表获取失败 |
| Unable to get MMM shape list | grouped_weight_quant_batch_matmul_tiling.cpp | 71 | MMM Shape列表获取失败 |
| Only support single-single-single or multi-multi-multi mode | grouped_weight_quant_batch_matmul_tiling.cpp | 73 | 模式约束检查失败 |
| When x-weight is bf16/fp16-int4/int32 and grouptype is 0, the... | grouped_weight_quant_batch_matmul_tiling.cpp | 235,240,246,252,258,431 | dtype组合约束检查失败 |
| When x-weight is float8_e8m0-float4_e2m1 and grouptype is 0, the... | grouped_weight_quant_batch_matmul_tiling.cpp | 246 | dtype组合约束检查失败 |
| CheckTensorDimSingleXSingleWeightSingleY failed | grouped_weight_quant_batch_matmul_tiling.cpp | 290 | Tensor维度检查失败 |
| CheckTensorDimMultiXMultiWeightMultiY failed | grouped_weight_quant_batch_matmul_tiling.cpp | 293 | Tensor维度检查失败 |
| CheckTensorNotNull failed | grouped_weight_quant_batch_matmul_tiling.cpp | 394 | Tensor非空检查失败 |
| CheckTensorDim failed | grouped_weight_quant_batch_matmul_tiling.cpp | 396 | Tensor维度检查失败 |
| CheckDimValue failed | grouped_weight_quant_batch_matmul_tiling.cpp | 398 | 维度值检查失败 |
| Check X And Weight Format failed | grouped_weight_quant_batch_matmul_tiling.cpp | 400 | Format检查失败 |
| Check Tensor x Dtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 402 | x dtype检查失败 |
| Check Tensor weight Dtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 404 | weight dtype检查失败 |
| Check Tensor antiquantScale Dtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 406 | antiquantScale dtype检查失败 |
| Check antiquantScale tensor shape failed | grouped_weight_quant_batch_matmul_tiling.cpp | 408 | antiquantScale shape检查失败 |
| Check Tensor antiquantOffset Dtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 411 | antiquantOffset dtype检查失败 |
| Check antiquantOffset tensor shape failed | grouped_weight_quant_batch_matmul_tiling.cpp | 413 | antiquantOffset shape检查失败 |
| Check Tensor bias Dtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 417 | bias dtype检查失败 |
| Check bias tensor shape failed | grouped_weight_quant_batch_matmul_tiling.cpp | 419 | bias shape检查失败 |
| CheckWeightInnerAxisEven failed | grouped_weight_quant_batch_matmul_tiling.cpp | 422 | WeightInnerAxis检查失败 |
| Invalid Input param | grouped_weight_quant_batch_matmul_tiling.cpp | 483 | 输入参数验证失败 |
| CheckTransposeStatus failed | grouped_weight_quant_batch_matmul_tiling.cpp | 489 | Transpose状态检查失败 |
| CheckParam failed | grouped_weight_quant_batch_matmul_tiling.cpp | 491 | 参数检查失败 |
| CheckTensorListSize failed | grouped_weight_quant_batch_matmul_tiling.cpp | 493 | Tensor列表大小检查失败 |
| CheckEmptyTensor failed | grouped_weight_quant_batch_matmul_tiling.cpp | 495 | 空Tensor检查失败 |
| CheckAntiQuantDtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 496 | AntiQuant dtype检查失败 |
| CheckBiasDtype failed | grouped_weight_quant_batch_matmul_tiling.cpp | 498 | Bias dtype检查失败 |
| CheckGroupList failed | grouped_weight_quant_batch_matmul_tiling.cpp | 499 | GroupList检查失败 |
| CheckEveryTensor failed | grouped_weight_quant_batch_matmul_tiling.cpp | 500 | Tensor逐项检查失败 |
| SetShapeList failed | grouped_weight_quant_batch_matmul_tiling.cpp | 503 | ShapeList设置失败 |
| CheckPerTokenScale failed | grouped_weight_quant_batch_matmul_tiling.cpp | 506 | PerTokenScale检查失败 |
| Get C0 size failed | grouped_weight_quant_batch_matmul_tiling.cpp | 579 | C0 size获取失败 |
| In weight quant case, when x-weight is... | grouped_weight_quant_batch_matmul_tiling.cpp | 773,780,788,793,801 | 权重量化场景约束检查失败 |
| when x-weight is bf16/fp16-int32/int4, grouptype only supports -1 or 0 | grouped_weight_quant_batch_matmul_tiling.cpp | 869 | grouptype约束检查失败 |
| When grouptype is -1. splititem can only be 0 or 1 | grouped_weight_quant_batch_matmul_tiling.cpp | 873 | splititem约束检查失败 |
| When grouptype is 0. splititem can only be 2 or 3 | grouped_weight_quant_batch_matmul_tiling.cpp | 877 | splititem约束检查失败 |
| Transposed A is not supported | grouped_weight_quant_batch_matmul_tiling.cpp | 886 | Transpose约束检查失败 |
| CheckGroupSize failed | grouped_weight_quant_batch_matmul_tiling.cpp | 1080 | GroupSize检查失败 |

**数量统计：** 43处

**整改建议：** 使用 `OP_LOGE_FOR_VALIDATION_FAIL(opName, reason)` 宏进行整改

## 四、grouped_matmul_finalize_routing 无法整改错误码分析

### 4.1 EZ0035 (Internal_Error_Nullptr) - 空指针错误

**场景说明：** 各种输入参数、Desc、StorageShape、Attr的空指针检查

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| workspaces is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 137 | workspace空指针检查 |
| RawTilingData is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 150 | TilingData空指针检查 |
| Input xDesc is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 175,216 | xDesc空指针检查（2处） |
| Input wDesc is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 177,472,645 | wDesc空指针检查（3处） |
| Input scaleDesc is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 184,228 | scaleDesc空指针检查（2处） |
| Input xStorageShape is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 218,658 | xStorageShape空指针检查（2处） |
| Input wStorageShape is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 224,676 | wStorageShape空指针检查（2处） |
| Input scaleStorageShape is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 230 | scaleStorageShape空指针检查 |
| Input pertokenScaleDesc is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 235 | pertokenScaleDesc空指针检查 |
| Input pertokenScaleStorageShape is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 239 | pertokenScaleStorageShape空指针检查 |
| Input groupListDesc is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 243 | groupListDesc空指针检查 |
| Input groupListStorageShape is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 247 | groupListStorageShape空指针检查 |
| Input rowIndexDesc is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 251 | rowIndexDesc空指针检查 |
| Input rowIndexStorageShape is nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 255 | rowIndexStorageShape空指针检查 |
| Attrs is nullptr | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 263,544,600 | Attr空指针检查（3处） |
| transpose_w should not be nullptr | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 272 | transpose_w空指针检查 |
| group_list_type should not be nullptr | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 277 | group_list_type空指针检查 |
| dtype should not be nullptr | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 284 | dtype空指针检查 |
| Input sharedInputStorageShape should not be nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 554,730 | sharedInputStorageShape空指针检查（2处） |
| Input shareInputWeightPtr should not be nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 559 | shareInputWeightPtr空指针检查 |
| Input shareInputOffsetPtr should not be nullptr. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 563 | shareInputOffsetPtr空指针检查 |
| Attr shareInputOffset[%lu] should less or equal to outputBS[%ld]. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 568 | 属性值约束检查（标注为空指针错误，实为值约束） |

**数量统计：** 32处

**整改建议：** 使用 `OP_LOGE_FOR_NULLPTR(opName, paramName)` 宏进行整改

### 4.2 EZ0037 (Internal_Error_Tiling) - Tiling错误

**场景说明：** Tiling策略推断失败、设置失败、格式设置失败

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| Failed to infer scenario. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 73 | 场景推断失败 |
| Fail to clear tiling data | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 169 | Tiling数据清理失败 |
| SetWFormat failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 740 | W格式设置失败 |
| ExtractXDimensions failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 741 | X维度提取失败 |
| ExtractWDimensions failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 743 | W维度提取失败 |
| SetSharedInputInfo failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 746 | 共享输入信息设置失败 |

**数量统计：** 6处

**整改建议：** 使用 `OP_LOGE_FOR_TILING_FAIL(opName, reason)` 宏进行整改

### 4.3 EZ0039 (Internal_Error_Memory_Operation) - 内存操作错误

**场景说明：** 内存拷贝操作失败

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| memcpy_s failed, ret = %d | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 155 | memcpy操作失败 |

**数量统计：** 1处

**整改建议：** 使用 `OP_LOGE_FOR_MEMCPY_FAIL(opName, ret)` 宏进行整改

### 4.4 EZ0042 (Internal_Error_Function_Failed) - 函数验证失败

**场景说明：** 输入参数验证失败、各维度验证函数失败

| 原错误消息 | 文件位置 | 行号 | 原因说明 |
|-----------|---------|------|---------|
| GroupedMatmulFinalizeRouting | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 58 | Cube内部错误 |
| Failed to check input params. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 77 | 输入参数检查失败 |
| ValidateXShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 478 | X Shape验证失败 |
| ValidateWShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 486 | W Shape验证失败 |
| ValidateGroupListShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 513 | GroupList Shape验证失败 |
| ValidateRowIndexShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 516 | RowIndex Shape验证失败 |
| ValidateLogitShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 520 | Logit Shape验证失败 |
| ValidateBiasShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 524 | Bias Shape验证失败 |
| ValidateScaleShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 532 | Scale Shape验证失败 |
| ValidatePertokenScaleShape failed. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 536 | PertokenScale Shape验证失败 |
| Failed to check input params in rule[%lu]. | grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 589 | 规则参数检查失败 |

**数量统计：** 11处

**整改建议：** 使用 `OP_LOGE_FOR_VALIDATION_FAIL(opName, reason)` 或 `OP_LOGE_FOR_CUBE_INNER_ERR(opName, reason)` 宏进行整改

## 五、总结统计

| 算子名称 | EZ0035 | EZ0037 | EZ0039 | EZ0042 | 总计 |
|---------|--------|--------|--------|--------|------|
| grouped_weight_quant_batch_matmul | 25 | 3 | 1 | 43 | 72 |
| grouped_matmul_finalize_routing | 32 | 6 | 1 | 11 | 50 |
| **合计** | **57** | **9** | **2** | **54** | **122** |

## 六、无法整改原因分类

### 6.1 空指针检查 (EZ0035)
- **原因：** 这些空指针检查是防御性编程，属于框架内部调用保障，不属于用户参数验证范畴
- **特征：** 错误消息中包含具体对象名称（如 xDesc、wTensor、ShapePtr 等）
- **建议宏：** `OP_LOGE_FOR_NULLPTR(opName, paramName)`

### 6.2 Tiling相关错误 (EZ0037)
- **原因：** Tiling策略是算子内部实现逻辑，失败原因复杂多样，难以标准化为具体参数错误
- **特征：** 消息中包含"tiling"、"scenario"、"format"等关键词
- **建议宏：** `OP_LOGE_FOR_TILING_FAIL(opName, reason)`

### 6.3 内存操作错误 (EZ0039)
- **原因：** 内存拷贝等操作失败属于系统调用错误，非参数验证错误
- **特征：** 消息中包含"memcpy"、"copy"等关键词，并返回错误码
- **建议宏：** `OP_LOGE_FOR_MEMCPY_FAIL(opName, ret)`

### 6.4 函数验证失败 (EZ0042)
- **原因：** 
  - 调用验证函数失败，具体失败原因在函数内部，上层无法直接获取具体参数信息
  - 复杂的dtype组合约束、grouptype/splititem组合约束等，错误消息已经足够清晰，无需拆分为多个参数
  - Cube内部错误属于硬件层面错误
- **特征：** 消息中包含"Check...failed"、"Validate...failed"、"failed"等关键词，或复杂的dtype组合约束描述
- **建议宏：** 
  - `OP_LOGE_FOR_VALIDATION_FAIL(opName, reason)` - 用于验证函数失败
  - `OP_LOGE_FOR_CUBE_INNER_ERR(opName, reason)` - 用于Cube内部错误

## 七、建议整改宏定义

```cpp
// EZ0035: 空指针错误
#define OP_LOGE_FOR_NULLPTR(opName, paramName) \
    OP_LOGE(EZ0035, "Parameter %s of operator %s is null.", paramName, opName);

// EZ0037: Tiling错误
#define OP_LOGE_FOR_TILING_FAIL(opName, reason) \
    OP_LOGE(EZ0037, "Op [%s] tiling failed. Reason: %s.", opName, reason);

// EZ0039: 内存操作错误
#define OP_LOGE_FOR_MEMCPY_FAIL(opName, ret) \
    OP_LOGE(EZ0039, "Memory copy failed in Op [%s]. Return code: %d.", opName, ret);

// EZ0042: 函数验证失败
#define OP_LOGE_FOR_VALIDATION_FAIL(opName, reason) \
    OP_LOGE(EZ0042, "Op [%s] validation failed. Reason: %s.", opName, reason);

// EZ0042: Cube内部错误
#define OP_LOGE_FOR_CUBE_INNER_ERR(opName, reason) \
    OP_LOGE(EZ0042, "Op [%s] cube inner error. Reason: %s.", opName, reason);
```

## 八、备注

- EZ0036 (Internal_Error_Memory_Allocation) 在本次分析的两个算子中未出现
- EZ0040 (Internal_Error_Registry) 在本次分析的两个算子中未出现
- EZ0041 (Internal_Error_Context) 在本次分析的两个算子中未出现
- EZ0043 (Internal_Error_Platform) 在本次分析的两个算子中未出现

这些错误码预留用于其他场景，如内存分配失败、注册失败、上下文错误、平台不支持等。