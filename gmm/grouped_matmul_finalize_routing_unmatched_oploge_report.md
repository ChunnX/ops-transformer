# 无法整改的 OP_LOGE 报告

## 统计信息
- 扫描总数: 84
- 可整改: 31
- 无法整改: 53
- 推荐新增错误码: 5

## 无法整改列表

| 文件路径 | 行号 | 原错误消息 | 分类 | 推荐新增错误码 |
|----------|------|------------|------|----------------|
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 58 | GroupedMatmulFinalizeRouting | Cube内部错误 | EZ0040 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 73 | Failed to infer scenario. | Tiling失败 | EZ0036 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 77 | Failed to check input params. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 137 | workspaces is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 150 | RawTilingData is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 155 | memcpy_s failed, ret = %d | memcpy失败 | EZ0038 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 169 | Fail to clear tiling data | Tiling失败 | EZ0036 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 175 | Input xDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 177 | Input wDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 184 | Input scaleDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 216 | Input xDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 218 | Input xStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 222 | Input wDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 224 | Input wStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 228 | Input scaleDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 230 | Input scaleStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 235 | Input pertokenScaleDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 239 | Input pertokenScaleStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 243 | Input groupListDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 247 | Input groupListStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 251 | Input rowIndexDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 255 | Input rowIndexStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 263 | Attrs is nullptr | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 272 | transpose_w should not be nullptr, but now is nullptr | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 277 | group_list_type should not be nullptr | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 284 | dtype should not be nullptr | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 472 | Input wDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 478 | ValidateXShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 486 | ValidateWShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 513 | ValidateGroupListShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 516 | ValidateRowIndexShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 520 | ValidateLogitShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 524 | ValidateBiasShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 532 | ValidateScaleShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 536 | ValidatePertokenScaleShape failed. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 544 | Attrs is nullptr | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 554 | Input sharedInputStorageShape should not be nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 559 | Input shareInputWeightPtr should not be nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 563 | Input shareInputOffsetPtr should not be nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 589 | Failed to check input params in rule[%lu]. | 验证失败 | EZ0037 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 600 | Attrs is nullptr | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 645 | Input wDesc is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 658 | Input xStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 676 | Input wStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 730 | Input sharedInputStorageShape is nullptr. | 空指针检查 | EZ0035 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 740 | SetWFormat failed. | Tiling失败 | EZ0036 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 741 | ExtractXDimensions failed. | Tiling失败 | EZ0036 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 743 | ExtractWDimensions failed. | Tiling失败 | EZ0036 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 746 | SetSharedInputInfo failed. | Tiling失败 | EZ0036 |

## 可整改列表（EZ0008-EZ0034）

| 文件路径 | 行号 | 原错误消息 | 整改错误码 | 整改状态 |
|----------|------|------------|------------|----------|
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 267 | transpose_x should be false or nullptr, but now is true | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 274 | transpose_w should be true, but now is false | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 280 | Attr groupListType must be 0 or 1, actual is %d. | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 286 | Attr dtype only support 0(float32), actual is %d. | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 292 | offset must be nullptr in MxA8W4 weight Nz scenario. | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 305 | The dimension of x must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 322 | The dimension of w (FRACTAL_NZ) must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 330 | The dimension of w must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 347 | E mismatch: w has E=%ld, group_list has %ld | EZ0010 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 358 | The dimension of row_index must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 363 | row_index shape[0]=%ld must equal M=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 378 | The dimension of logit must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 382 | logit shape[0]=%ld must equal M=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 403 | The dimension of bias must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 407 | bias shape[0]=%ld must equal E=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 410 | bias shape[1]=%ld must equal N=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 423 | The dimension of scale must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 427 | scale shape[0]=%ld must equal E=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 430 | scale shape[1]=%ld must equal N=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 433 | scale shape[2]=%ld must equal CeilDiv(K, 64)=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 437 | scale shape[3]=%ld must equal %d | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 450 | The dimension of pertokenScale must be %u, actual is %zu | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 454 | pertokenScale shape[0]=%ld must equal M=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 458 | pertokenScale shape[1]=%ld must equal CeilDiv(K, 64)=%ld | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 462 | pertokenScale shape[2]=%ld must equal %d | EZ0009 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 496 | K dimension mismatch: x has K=%ld, w has K=%ld | EZ0010 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 498 | K dimension must be aligned to %u, actual K=%ld | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 500 | N dimension must be aligned to %u, actual N=%ld | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 505 | K dimension must be >= %ld, actual K=%ld | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 508 | N dimension must be >= %ld, actual N=%ld | EZ0012 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 548 | Attr outputBS should be >=0. | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 565 | Attr shareInputOffset should be >=0. | EZ0026 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 568 | Attr shareInputOffset[%lu] should less or equal to outputBS[%ld]. | EZ0027 | 可整改 |
| grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 621 | groupNum(E) is 0, cannot compute default outputBs as M/E. | EZ0026 | 可整改 |

## 推荐新增错误码汇总

| 推荐错误码 | 场景 | ErrMessage 建议 | Arglist 建议 | 出现次数 |
|------------|------|-----------------|--------------|----------|
| EZ0035 | 空指针检查 | "Parameter %s of operator %s is null." | op_name, param_name | 31次 |
| EZ0036 | Tiling失败 | "Op [%s] tiling failed. Reason: %s." | op_name, reason | 5次 |
| EZ0037 | 验证函数失败 | "Op [%s] input validation failed. Reason: %s." | op_name, reason | 10次 |
| EZ0038 | memcpy失败 | "Memory copy failed in Op [%s]. Return code: %d." | op_name, ret_code | 1次 |
| EZ0040 | OPS_REPORT_CUBE_INNER_ERR | "Op [%s] cube inner error. Reason: %s." | op_name, reason | 1次 |

## 整改建议

### 1. 新增 EZ0035（空指针检查）宏定义建议

```cpp
// opbase/log.h
#define OP_LOGE_FOR_NULLPTR(opName, paramName) \
    OP_LOGE_EZ0035(opName, paramName)

// 对应错误码定义
// EZ0035: "Parameter %s of operator %s is null."
```

### 2. 新增 EZ0036（Tiling失败）宏定义建议

```cpp
// opbase/log.h
#define OP_LOGE_FOR_TILING_FAIL(opName, reason) \
    OP_LOGE_EZ0036(opName, reason)

// 对应错误码定义
// EZ0036: "Op [%s] tiling failed. Reason: %s."
```

### 3. 新增 EZ0037（验证失败）宏定义建议

```cpp
// opbase/log.h
#define OP_LOGE_FOR_VALIDATION_FAIL(opName, reason) \
    OP_LOGE_EZ0037(opName, reason)

// 对应错误码定义
// EZ0037: "Op [%s] input validation failed. Reason: %s."
```

### 4. 新增 EZ0038（memcpy失败）宏定义建议

```cpp
// opbase/log.h
#define OP_LOGE_FOR_MEMCPY_FAIL(opName, retCode) \
    OP_LOGE_EZ0038(opName, retCode)

// 对应错误码定义
// EZ0038: "Memory copy failed in Op [%s]. Return code: %d."
```

### 5. 新增 EZ0040（Cube内部错误）宏定义建议

```cpp
// opbase/log.h
#define OP_LOGE_FOR_CUBE_INNER_ERR(opName, reason) \
    OP_LOGE_EZ0040(opName, reason)

// 对应错误码定义
// EZ0040: "Op [%s] cube inner error. Reason: %s."
```

## 整改注意事项

1. **CSV文件编码**：UTF-8编码（Excel可直接打开）
2. **MD文件编码**：UTF-8编码
3. **整改优先级**：优先整改可整改部分（EZ0008-EZ0034），再处理新增错误码
4. **宏定义位置**：新增错误码宏定义应添加到 `opbase/log.h`
5. **arglist顺序**：必须严格按照 log.h 中的宏定义参数顺序填写
6. **reason参数**：reason必须是字符串常量，不能是变量表达式