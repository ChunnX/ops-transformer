# 无法整改的 OP_LOGE 报告 - grouped_matmul_finalize_routing

## 统计信息

- 扫描总数: 83
- 已整改（可整改）: 29
- 无法整改: 54
- 整改率: 34.9%
- 推荐新增错误码: 0（均为内部错误，不推荐新增）

## 可整改的EZ错误码分布

| 错误码 | 适用场景 | 出现次数 |
|--------|----------|----------|
| EZ0010 | 参数验证错误 | 11 |
| EZ0011 | 参数验证错误 | 8 |
| EZ0012 | 参数验证错误 | 4 |
| EZ0013 | 参数验证错误 | 1 |
| EZ0024 | 参数验证错误 | 1 |
| EZ0026 | 参数验证错误 | 4 |

## 无法整改原因分类

| 分类 | 原因说明 | 出现次数 |
|------|----------|----------|
| 空指针检查 | 不属于参数验证范畴 | 31 |
| 其他 | 无法分类 | 17 |
| 内部校验函数失败 | 不属于参数验证范畴 | 2 |
| 内部执行错误 | 不属于参数验证范畴 | 2 |
| OPS_REPORT_CUBE_INNER_ERR | 不在EZ整改范围 | 1 |
| 场景推断失败 | 不属于参数验证范畴 | 1 |

## 无法整改详细列表

| 文件路径 | 行号 | 原错误消息 | 分类 | 处理方式 |
|----------|------|------------|------|----------|
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 58 | GroupedMatmulFinalizeRouting... | OPS_REPORT_CUBE_INNER_ERR不在EZ整改范围内 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 73 | Failed to infer scenario.... | 场景推断失败 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 77 | Failed to check input params.... | 内部校验函数失败 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 137 | workspaces is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 150 | RawTilingData is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 155 | memcpy_s failed, ret = %d... | 内部执行错误 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 169 | Fail to clear tiling data... | 内部执行错误 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 175 | Input xDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 177 | Input wDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 184 | Input scaleDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 216 | Input xDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 218 | Input xStorageShape is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 222 | Input wDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 224 | Input wStorageShape is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 228 | Input scaleDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 230 | Input scaleStorageShape is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 235 | Input pertokenScaleDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 239 | Input pertokenScaleStorageShape is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 243 | Input groupListDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 247 | Input groupListStorageShape is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 251 | Input rowIndexDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 255 | Input rowIndexStorageShape is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 263 | Attrs is nullptr... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 272 | transpose_w should not be nullptr, but now is null... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 277 | group_list_type should not be nullptr... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 284 | dtype should not be nullptr... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 292 | offset must be nullptr in MxA8W4 weight Nz scenari... | function失败 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 347 | E mismatch: w has E=%ld, group_list has %ld... | 无法分类 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 472 | Input wDesc is nullptr.... | 空指针检查 | 不整改 |
| op_host/op_tiling/grouped_matmul_finalize_routing_weight_quant_tiling.cpp | 478 | ValidateXShape failed.... | ValidateXShape失败 | 不整改 |

> 注：仅显示前30条，共54条无法整改

## 推荐新增错误码汇总

| 推荐错误码 | 场景 | ErrMessage 建议 | Arglist 建议 | 出现次数 |
|------------|------|-----------------|--------------|----------|
| N/A | 所有无法整改均为内部错误 | 不推荐新增错误码 | N/A | - |

**说明：**
- 所有无法整改的 OP_LOGE 都属于内部执行错误（空指针检查、内部函数失败等），不属于参数验证范畴。
- 根据EZ整改原则，内部错误不在EZ0008-EZ0034覆盖范围内，不推荐新增错误码。
- OPS_REPORT_CUBE_INNER_ERR 不属于 OP_LOGE 类型，不在EZ整改范围内。
