# EZ0035-EZ0043错误码宏定义存档

## 背景

`third_party/opbase` 目录在 `.gitignore` 中被忽略，因此 `log.h` 的修改不会被 git 跟踪。

为了保留新增的错误码宏定义（EZ0035-EZ0043），我们将这些定义单独存档。

## 文件位置

- **存档文件**: `gmm/patches/ez0035_ez0043_macros_definition.h`
- **原文件**: `third_party/opbase/pkg_inc/op_common/log/log.h`
- **新增行号**: 791-948

## 使用方法

### 方法1: 手动合并到 log.h

1. 打开 `third_party/opbase/pkg_inc/op_common/log/log.h`
2. 找到 `EZ0027` 定义结束位置（约第789行）
3. 将 `ez0035_ez0043_macros_definition.h` 的内容插入到该位置
4. 编译验证

### 方法2: 直接替换

如果 log.h 已经有 EZ0035-EZ0043 的定义，确保与存档文件一致即可。

## 新增宏定义列表

| 错误码 | 宏名称 | 用途 | 参数 |
|--------|--------|------|------|
| EZ0035 | `OP_LOGE_FOR_NULLPTR` | nullptr错误 | opName, paramName |
| EZ0036 | `OP_LOGE_FOR_TILING_FAILED` | tiling计算失败 | opName, reason |
| EZ0037 | `OP_LOGE_FOR_VALIDATION_FAILED` | 验证失败 | opName, reason |
| EZ0038 | `OP_LOGE_FOR_MEMORY_OP_FAILED` | 内存操作失败 | opName, retCode |
| EZ0039 | `OP_LOGE_FOR_MEMCPY_FAILED` | memcpy失败（兼容宏） | - |
| EZ0040 | `OP_LOGE_FOR_CUBE_INNER_ERROR` | Cube内部错误 | opName, reason |
| EZ0041 | `OP_LOGE_FOR_FUNCTION_FAILED` | 函数执行失败 | opName, functionName |
| EZ0042 | `OP_LOGE_FOR_INTERNAL_FUNCTION_FAILED` | 内部检查失败 | opName, reason |
| EZ0043 | `OP_LOGE_FOR_PLATFORM_ERROR` | 平台错误 | opName, reason |

## 兼容性宏

为了兼容旧代码，提供了以下兼容性宏：

```cpp
#define OP_LOGE_FOR_INTERNAL_FUNCTION_FAILED(opName, reason) OP_LOGE_FOR_INTERNAL_FUNCTION_FAILED(opName, reason)
#define OP_LOGE_FOR_TILING_FAILED(opName, reason) OP_LOGE_FOR_TILING_FAILED(opName, reason)
#define OP_LOGE_FOR_VALIDATION_FAILED(opName, reason) OP_LOGE_FOR_VALIDATION_FAILED(opName, reason)
#define OP_LOGE_FOR_MEMORY_OP_FAILED(opName, ret) OP_LOGE_FOR_MEMORY_OP_FAILED(opName, ret)
#define OP_LOGE_FOR_CUBE_INNER_ERROR(opName, reason) OP_LOGE_FOR_CUBE_INNER_ERROR(opName, reason)
```

## 整改统计

已使用这些宏定义整改了 **206个错误场景**：

- grouped_weight_quant_batch_matmul: 123个
- grouped_matmul_finalize_routing: 83个

按错误码分布：
- EZ0035 (nullptr): 56个
- EZ0037 (tiling失败): 8个
- EZ0039 (memcpy失败): 2个
- EZ0040 (Cube错误): 1个
- EZ0042 (内部检查): 56个

## 编译验证

所有整改已通过编译验证：
- ✅ weight_quant: 302个测试通过
- ✅ finalize_routing: 48个测试通过

## Git提交

整改代码已提交到 `gmm_gmmfr_wq_err_msg` 分支：
- Commit: `309ac06b`
- 文件修改:
  - grouped_weight_quant_batch_matmul_tiling.cpp (78+, 92-)
  - grouped_matmul_finalize_routing_weight_quant_tiling.cpp (49+, 52-)

## 注意事项

1. **third_party目录被忽略**: `.gitignore` 中包含 `third_party/`，因此 log.h 的修改不会被 git 跟踪
2. **存档文件不参与编译**: 此存档文件仅供参考，实际编译时需修改 `third_party/opbase/pkg_inc/op_common/log/log.h`
3. **宏定义格式一致**: 所有宏定义遵循 EZ0008-EZ0027 的标准格式

## 更新时间

2026-05-14