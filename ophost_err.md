# OPhost Tiling Error Message 整改验证报告

## 测试概述

- **测试套件**: GroupedMatmulFinalizeRoutingWeightQuantTiling
- **测试用例总数**: 27
- **测试结果**: 27/27 PASSED ✅
- **日志文件**: ophost_err.log (119行)
- **目标文件**: gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp

## 验证方法

1. **编译命令**:
   ```bash
   bash build.sh -u --ophost --soc=ascend950 --ops=grouped_matmul_finalize_routing -j8 --noexec
   ```

2. **执行命令**:
   ```bash
   ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=3 BUILD_PATH=$(pwd)/build \
   ./build/tests/ut/framework_normal/op_host/transformer_op_host_ut > ophost_err.log
   ```

3. **验证方式**: 逐用例对照源码中的 `OP_LOGE_FOR_*` 宏调用，检查日志格式、参数名、错误值、Reason 文本是否符合预期。

## 预期错误日志格式

根据 `op_error_check.h` 中的宏定义：

| 宏名称 | 日志格式 |
|--------|----------|
| `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` | `Parameters {params} of operator {op} have incorrect dtypes {dtypes}. Reason: {reason}` |
| `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` | `Parameter {param} of operator {op} has incorrect value {value}. Reason: {reason}` |
| `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON` | `Parameter {param} of operator {op} has incorrect shape dim {dim}. Reason: {reason}` |
| `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON` | `Parameter {param} of operator {op} has incorrect shape {shape}. Reason: {reason}` |
| `OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON` | `Parameters {params} of operator {op} have incorrect shapes {shapes}. Reason: {reason}` |

---

## 逐用例验证结果

### 1. TestMXA8W4WeightNzNormalCase
- **状态**: ✅ PASS
- **预期**: 正常用例，无错误日志
- **实际**: 无错误日志，测试通过

### 2. TestMXA8W4WeightNzCoreNum
- **状态**: ✅ PASS
- **预期**: 正常用例，无错误日志
- **实际**: 无错误日志，测试通过

### 3. TestMXA8W4WeightNzNullSharedInput
- **状态**: ✅ PASS
- **预期**: 正常用例，无错误日志
- **实际**: 无错误日志，测试通过

### 4. TestMXA8W4WeightNzKNotAligned
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON` (line 577)
- **实际日志** (line 13):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:577][OPS_BASE][CheckMxA8W4InputShape]
  Parameter x of operator GroupedMatmulFinalizeRouting has incorrect shape K=33.
  Reason: K of x must be aligned to 32.
  ```
- **验证**: ✅ 完全符合预期

### 5. TestMXA8W4WeightNzNNotAligned
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON` (line 583)
- **实际日志** (line 18):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:583][OPS_BASE][CheckMxA8W4InputShape]
  Parameter weight of operator GroupedMatmulFinalizeRouting has incorrect shape N=33.
  Reason: N of weight must be aligned to 32.
  ```
- **验证**: ✅ 完全符合预期

### 6. TestMXA8W4WeightNzKTooSmall
- **状态**: ⚠️ ANOMALY
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON` (line 591, K < MIN_KN_SIZE)
- **实际日志** (line 23):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:577][OPS_BASE][CheckMxA8W4InputShape]
  Parameter x of operator GroupedMatmulFinalizeRouting has incorrect shape K=30.
  Reason: K of x must be aligned to 32.
  ```
- **问题分析**:
  - 用例设置 k=30，期望触发最小值检查 (line 591: K < 32)
  - 但 30 % 32 != 0，先触发了对齐检查 (line 577)
  - 实际触发的是对齐检查而非最小值检查
- **影响**: 最小值检查路径 (line 591) 未被覆盖

### 7. TestMXA8W4WeightNzNTooSmall
- **状态**: ⚠️ ANOMALY
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON` (line 597, N < MIN_KN_SIZE)
- **实际日志** (line 28):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:583][OPS_BASE][CheckMxA8W4InputShape]
  Parameter weight of operator GroupedMatmulFinalizeRouting has incorrect shape N=16.
  Reason: N of weight must be aligned to 32.
  ```
- **问题分析**:
  - 用例设置 n=16，期望触发最小值检查 (line 597: N < 32)
  - 但 16 % 32 != 0，先触发了对齐检查 (line 583)
  - 实际触发的是对齐检查而非最小值检查
- **影响**: 最小值检查路径 (line 597) 未被覆盖

### 8. TestMXA8W4WeightNzNullScale
- **状态**: ✅ PASS
- **预期**: 触发旧式 `OP_LOGE` (line 211, scaleDesc is nullptr)
- **实际日志** (line 33):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:211][OPS_BASE][InferScenario]
  Input scaleDesc is nullptr.
  ```
- **验证**: ✅ 符合预期（旧式宏，非整改范围）

### 9. TestMXA8W4WeightNzNullPertokenScale
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` (line 237)
- **实际日志** (line 37):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:237][OPS_BASE][InferScenario]
  Parameters x, w, scale, perTokenScale of operator GroupedMatmulFinalizeRouting
  have incorrect dtypes DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT8_E8M0, DT_INT8.
  Reason: Only support MX-A8W4-WEIGHT-NZ mode.
  ```
- **验证**: ✅ 完全符合预期

### 10. TestMXA8W4WeightNzNullRowIndex
- **状态**: ✅ PASS
- **预期**: 触发旧式 `OP_LOGE` (line 280, rowIndexDesc is nullptr)
- **实际日志** (line 41):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:280][OPS_BASE][CheckMxA8W4NzInputPtr]
  Input rowIndexDesc is nullptr.
  ```
- **验证**: ✅ 符合预期（旧式宏，非整改范围）

### 11. TestMXA8W4WeightNzNullGroupList
- **状态**: ✅ PASS
- **预期**: 触发旧式 `OP_LOGE` (line 272, groupListDesc is nullptr)
- **实际日志** (line 46):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:272][OPS_BASE][CheckMxA8W4NzInputPtr]
  Input groupListDesc is nullptr.
  ```
- **验证**: ✅ 符合预期（旧式宏，非整改范围）

### 12. TestMXA8W4WeightNzWrongXDtype
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` (line 237)
- **实际日志** (line 51):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:237][OPS_BASE][InferScenario]
  Parameters x, w, scale, perTokenScale of operator GroupedMatmulFinalizeRouting
  have incorrect dtypes DT_FLOAT16, DT_FLOAT4_E2M1, DT_FLOAT8_E8M0, DT_FLOAT8_E8M0.
  Reason: Only support MX-A8W4-WEIGHT-NZ mode.
  ```
- **验证**: ✅ 完全符合预期

### 13. TestMXA8W4WeightNzWrongWDtype
- **状态**: ⚠️ ANOMALY
- **预期**: 触发 `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` (line 237, weight_quant_tiling.cpp)
- **实际日志** (line 55):
  ```
  [grouped_matmul_finalize_routing_quant_tiling.cpp:201][OPS_BASE][CheckDtype]
  The dtype of bias should be DT_BF16, but now is DT_FLOAT
  ```
- **问题分析**:
  - 错误来自 `grouped_matmul_finalize_routing_quant_tiling.cpp`，而非 `weight_quant_tiling.cpp`
  - 被通用 quant checker 先拦截，未到达 weight quant checker 的 InferScenario
  - 日志格式为旧式 `OP_LOGE`，非整改后的新式宏
- **影响**: weight_quant_tiling.cpp 中的 InferScenario dtype 检查路径未被此用例覆盖

### 14. TestMXA8W4WeightNzWrongScaleDtype
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` (line 237)
- **实际日志** (line 60):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:237][OPS_BASE][InferScenario]
  Parameters x, w, scale, perTokenScale of operator GroupedMatmulFinalizeRouting
  have incorrect dtypes DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_FLOAT, DT_FLOAT8_E8M0.
  Reason: Only support MX-A8W4-WEIGHT-NZ mode.
  ```
- **验证**: ✅ 完全符合预期

### 15. TestMXA8W4WeightNzTransposeWFalse
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` (line 304)
- **实际日志** (line 64):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:304][OPS_BASE][CheckMxA8W4NzAttrPtr]
  Parameter transposeW of operator GroupedMatmulFinalizeRouting has incorrect value false.
  Reason: The value of transposeW must be true.
  ```
- **验证**: ✅ 完全符合预期

### 16. TestMXA8W4WeightNzWrongXShape
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON` (line 344)
- **实际日志** (line 69):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:344][OPS_BASE][ValidateXShape]
  Parameter x of operator GroupedMatmulFinalizeRouting has incorrect shape dim 3.
  Reason: The shape dim of x must be 2.
  ```
- **验证**: ✅ 完全符合预期

### 17. TestMXA8W4WeightNzWrongWShape
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON` (line 364, NZ format)
- **实际日志** (line 75):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:364][OPS_BASE][ValidateWShape]
  Parameter w of operator GroupedMatmulFinalizeRouting has incorrect shape dim 2.
  Reason: The shape dim of w (FRACTAL_NZ) must be 5.
  ```
- **验证**: ✅ 完全符合预期

### 18. TestMXA8W4WeightNzEMismatch
- **状态**: ⚠️ ANOMALY
- **预期**: 触发 `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` (line 394, ValidateGroupListShape E mismatch)
- **实际日志** (line 81):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:491][OPS_BASE][ValidateScaleShape]
  Parameter scale of operator GroupedMatmulFinalizeRouting has incorrect shape [17, 7168, 32, 2].
  Reason: The axis 0 of scale must be equal to E=16.
  ```
- **问题分析**:
  - 用例修改了 scaleShape 的 dim[0] 为 e+1=17
  - 先触发了 ValidateScaleShape (line 491) 的 scale[0]!=E 检查
  - 未到达 ValidateGroupListShape (line 394) 的 E mismatch 检查
- **影响**: ValidateGroupListShape 的 E mismatch 路径未被覆盖

### 19. TestMXA8W4WeightNzKMismatch
- **状态**: ✅ PASS
- **预期**: 触发 `OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON` (line 501, scale[2] != CeilDiv)
- **实际日志** (line 87):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:501][OPS_BASE][ValidateScaleShape]
  Parameter scale of operator GroupedMatmulFinalizeRouting has incorrect shape [16, 7168, 33, 2].
  Reason: The axis 2 of scale must be equal to CeilDiv(K, 64)=32.
  ```
- **验证**: ✅ 完全符合预期

### 20-24. 正常用例 (WithBias, DefaultOutputBs, SmallDimensions, EmptyBias, EZeroWithOutputBs)
- **状态**: ✅ PASS
- **预期**: 正常用例，无错误日志
- **实际**: 无错误日志，测试通过

### 25. TestMXA8W4WeightNzEZeroNullOutputBs
- **状态**: ⚠️ ANOMALY
- **预期**: 触发 E=0 相关检查
- **实际日志** (line 101):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:671][OPS_BASE][CheckMxA8W4AttrWithInput]
  Parameter shareInputOffset of operator GroupedMatmulFinalizeRouting has incorrect value
  shareInputOffset=0, sharedInputStorageShape.GetDim(0)=64, outputBS=0.
  Reason: Attr shareInputOffset + sharedInputStorageShape.GetDim(0) must be <= outputBS.
  ```
- **问题分析**:
  - 用例设置 E=0, outputBS=NULL，期望触发 E=0 相关检查
  - 但实际触发了 shareInputOffset 溢出检查 (line 671)
  - 因为 sharedInputLen=64 > outputBS=0
- **影响**: 覆盖了 shareInputOffset 溢出路径，但非预期的 E=0 路径

### 26. TestCsvFileLoaded
- **状态**: ✅ PASS
- **预期**: 验证 CSV 文件加载
- **实际**: 测试通过

### 27. TestMXA8W4WeightNzInvalidAicAivRatio
- **状态**: ✅ PASS
- **预期**: 触发旧式 `OP_LOGE` (line 66, core ratio check)
- **实际日志** (line 110):
  ```
  [grouped_matmul_finalize_routing_weight_quant_tiling.cpp:66][OPS_BASE][CheckCoreNum]
  Invalid cube/vector core ratio. Expected cube:vector = 1:2, but got cube=24, vector=24
  ```
- **验证**: ✅ 符合预期（旧式宏，非整改范围）

---

## 统计汇总

| 状态 | 数量 | 占比 |
|------|------|------|
| ✅ PASS | 22 | 81.5% |
| ⚠️ ANOMALY | 5 | 18.5% |
| ❌ FAIL | 0 | 0% |
| **总计** | **27** | **100%** |

---

## 异常用例分析

### 1. KTooSmall / NTooSmall
**问题**: 用例设置 k=30/n=16 期望触发最小值检查，但因未对齐先触发了对齐检查。

**根本原因**: 检查顺序问题，对齐检查 (line 577/583) 在最小值检查 (line 591/597) 之前执行。

**影响**: 最小值检查路径 (line 591, 597) 未被覆盖。

**建议**: 修改用例使用已对齐但小于最小值的数值，例如 k=32（已对齐但等于最小值，需确认 MIN_KN_SIZE 定义）或 k=0（如果允许）。

### 2. WrongWDtype
**问题**: 错误来自 `grouped_matmul_finalize_routing_quant_tiling.cpp` 而非 `weight_quant_tiling.cpp`。

**根本原因**: API 调用流程中存在多层检查机制，通用 quant checker 先于 weight quant checker 执行。

**影响**: weight_quant_tiling.cpp 中的 InferScenario dtype 检查路径未被此用例覆盖。

**建议**: 这是 API 设计的正常行为，无需修改代码。如需验证 weight quant checker 的逻辑，需构造能通过通用 checker 的特殊 dtype 组合。

### 3. EMismatch
**问题**: 用例修改 scaleShape 的 dim[0]，先触发了 ValidateScaleShape 而非 ValidateGroupListShape。

**根本原因**: 检查顺序问题，ValidateScaleShape 在 ValidateGroupListShape 之后执行，但因 scaleShape 被修改，先触发了 scale 形状检查。

**影响**: ValidateGroupListShape 的 E mismatch 路径 (line 394) 未被覆盖。

**建议**: 修改用例，保持 scaleShape 正确，但修改 groupListShape 的 dim[0] 使其与 weight 的 E 不匹配。

### 4. EZeroNullOutputBs
**问题**: 用例期望触发 E=0 相关检查，但实际触发了 shareInputOffset 溢出检查。

**根本原因**: 用例设置 E=0, outputBS=NULL (默认为 0)，但 sharedInputLen=64，导致 0+64 > 0 触发溢出检查。

**影响**: 覆盖了 shareInputOffset 溢出路径 (line 671)，但非预期的 E=0 路径。

**建议**: 如需专门测试 E=0 场景，应设置 sharedInputLen=0 或 outputBS 足够大。

---

## 源码错误路径 vs 用例覆盖完整性对照

### 一、InferScenario (lines 199-240)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 237 | dtype combo 不匹配 MxA8W4 | INVALID_DTYPES | WrongXDtype, WrongScaleDtype, NullPertokenScale | ✅ 已覆盖 |

**结论**: 1/1 路径已覆盖。

### 二、CheckMxA8W4NzAttrPtr (lines 289-335)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 296 | transposeX=true | INVALID_VALUE | ❌ 无 | **缺失** |
| 304 | transposeW=false | INVALID_VALUE | TransposeWFalse | ✅ 已覆盖 |
| 313 | groupListType not 0 or 1 | INVALID_VALUE | ❌ 无 | **缺失** |
| 321 | dtype not 0 | INVALID_DTYPE | ❌ 无 | **缺失** |
| 329 | offset not nullptr | INVALID_VALUE | ❌ 无 | **缺失** |

**结论**: 1/5 路径已覆盖，**4 条缺失**。

### 三、ValidateXShape (lines 338-352)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 344 | x dim != 2 | INVALID_SHAPEDIM | WrongXShape | ✅ 已覆盖 |

**结论**: 1/1 路径已覆盖。

### 四、ValidateWShape (lines 355-384)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 364 | w storage dim != 5 (NZ) | INVALID_SHAPEDIM | WrongWShape | ✅ 已覆盖 |
| 374 | w storage dim != 3 (non-NZ) | INVALID_SHAPEDIM | ❌ 无 | **缺失** |

**结论**: 1/2 路径已覆盖，**1 条缺失**。

### 五、ValidateGroupListShape (lines 387-400)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 394 | E mismatch weight vs groupList | INVALID_VALUE | ❌ 无 (EMismatch 触发 scale 检查) | **缺失** |

**结论**: 0/1 路径已覆盖，**1 条缺失**。

### 六、ValidateRowIndexShape (lines 403-419)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 408 | rowIndex dim != 1 | INVALID_SHAPEDIM | ❌ 无 | **缺失** |
| 414 | rowIndex[0] != M | INVALID_SHAPE | ❌ 无 | **缺失** |

**结论**: 0/2 路径已覆盖，**2 条缺失**。

### 七、ValidateLogitShape (lines 422-442)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 431 | logit dim != 1 | INVALID_SHAPEDIM | ❌ 无 | **缺失** |
| 437 | logit[0] != M | INVALID_SHAPE | ❌ 无 | **缺失** |

**结论**: 0/2 路径已覆盖，**2 条缺失**。

### 八、ValidateBiasShape (lines 445-476)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 460 | bias dim != 2 | INVALID_SHAPEDIM | ❌ 无 | **缺失** |
| 466 | bias[0] != E | INVALID_SHAPE | ❌ 无 | **缺失** |
| 471 | bias[1] != N | INVALID_SHAPE | ❌ 无 | **缺失** |

**结论**: 0/3 路径已覆盖，**3 条缺失**。

### 九、ValidateScaleShape (lines 479-512)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 485 | scale dim != 4 | INVALID_SHAPEDIM | ❌ 无 | **缺失** |
| 491 | scale[0] != E | INVALID_SHAPE | EMismatch | ✅ 已覆盖 |
| 496 | scale[1] != N | INVALID_SHAPE | ❌ 无 | **缺失** |
| 501 | scale[2] != CeilDiv(K,64) | INVALID_SHAPE | KMismatch | ✅ 已覆盖 |
| 507 | scale[3] != 2 | INVALID_SHAPE | ❌ 无 | **缺失** |

**结论**: 2/5 路径已覆盖，**3 条缺失**。

### 十、ValidatePertokenScaleShape (lines 515-545)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 521 | pertokenScale dim != 3 | INVALID_SHAPEDIM | ❌ 无 | **缺失** |
| 527 | pertokenScale[0] != M | INVALID_SHAPE | ❌ 无 | **缺失** |
| 533 | pertokenScale[1] != CeilDiv(K,64) | INVALID_SHAPE | ❌ 无 | **缺失** |
| 539 | pertokenScale[2] != 2 | INVALID_SHAPE | ❌ 无 | **缺失** |

**结论**: 0/4 路径已覆盖，**4 条缺失**。

### 十一、CheckMxA8W4InputShape (lines 547-631)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 569 | K mismatch x vs w | INVALID_SHAPES | ❌ 无 | **缺失** |
| 577 | K not aligned | INVALID_SHAPE | KNotAligned | ✅ 已覆盖 |
| 583 | N not aligned | INVALID_SHAPE | NNotAligned | ✅ 已覆盖 |
| 591 | K < MIN_KN_SIZE | INVALID_SHAPEDIM | ❌ 无 (KTooSmall 触发对齐检查) | **缺失** |
| 597 | N < MIN_KN_SIZE | INVALID_SHAPEDIM | ❌ 无 (NTooSmall 触发对齐检查) | **缺失** |

**结论**: 2/5 路径已覆盖，**3 条缺失**。

### 十二、CheckMxA8W4AttrWithInput (lines 633-679)

| 行号 | 检查内容 | 宏类型 | 对应用例 | 状态 |
|------|---------|--------|---------|------|
| 641 | outputBS < 0 | INVALID_VALUE | ❌ 无 | **缺失** |
| 662 | shareInputOffset < 0 | INVALID_VALUE | ❌ 无 | **缺失** |
| 671 | shareInputOffset overflow | INVALID_VALUE | EZeroNullOutputBs | ✅ 已覆盖 |

**结论**: 1/3 路径已覆盖，**2 条缺失**。

---

## 覆盖缺口汇总

| 分类 | 总路径数 | 已覆盖 | 缺失 | 覆盖率 |
|------|---------|--------|------|--------|
| InferScenario | 1 | 1 | 0 | 100% |
| CheckMxA8W4NzAttrPtr | 5 | 1 | 4 | 20% |
| ValidateXShape | 1 | 1 | 0 | 100% |
| ValidateWShape | 2 | 1 | 1 | 50% |
| ValidateGroupListShape | 1 | 0 | 1 | 0% |
| ValidateRowIndexShape | 2 | 0 | 2 | 0% |
| ValidateLogitShape | 2 | 0 | 2 | 0% |
| ValidateBiasShape | 3 | 0 | 3 | 0% |
| ValidateScaleShape | 5 | 2 | 3 | 40% |
| ValidatePertokenScaleShape | 4 | 0 | 4 | 0% |
| CheckMxA8W4InputShape | 5 | 2 | 3 | 40% |
| CheckMxA8W4AttrWithInput | 3 | 1 | 2 | 33% |
| **总计** | **34** | **9** | **25** | **26.5%** |

---

## 缺失路径分析

### 1. CheckMxA8W4NzAttrPtr 的 4 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 296 | transposeX=true | 当前测试框架未支持修改 transposeX 属性 | 在 CSV 中添加 transpose_x 列，或在测试代码中手动设置 |
| 313 | groupListType not 0 or 1 | 当前用例只使用 groupListType=0 | 添加用例设置 groupListType=2 或其他非法值 |
| 321 | dtype not 0 | 当前用例只使用 dtype=0 (float32) | 添加用例设置 dtype=1 或其他非法值 |
| 329 | offset not nullptr | 当前测试框架未支持 offset 输入 | 在测试代码中添加 offset 输入并设置为非空 |

### 2. ValidateWShape 的 1 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 374 | w storage dim != 3 (non-NZ) | 当前用例只使用 FRACTAL_NZ 格式 | 添加用例使用 ND 格式的 weight，并设置错误的 storage dim |

### 3. ValidateGroupListShape 的 1 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 394 | E mismatch weight vs groupList | EMismatch 用例修改了 scaleShape 而非 groupListShape | 修改 EMismatch 用例，保持 scaleShape 正确，修改 groupListShape 的 dim[0] |

### 4. ValidateRowIndexShape 的 2 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 408 | rowIndex dim != 1 | 当前用例 rowIndexShape 始终为 [m] | 添加用例设置 rowIndexShape 为多维或空 |
| 414 | rowIndex[0] != M | 当前用例 rowIndexShape[0] 始终等于 m | 添加用例设置 rowIndexShape[0] != m |

### 5. ValidateLogitShape 的 2 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 431 | logit dim != 1 | 当前用例 logitShape 始终为 [m] | 添加用例设置 logitShape 为多维或空 |
| 437 | logit[0] != M | 当前用例 logitShape[0] 始终等于 m | 添加用例设置 logitShape[0] != m |

### 6. ValidateBiasShape 的 3 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 460 | bias dim != 2 | 当前用例 biasShape 始终为 [e, n] 或空 | 添加用例设置 biasShape 为 1 维或 3 维 |
| 466 | bias[0] != E | 当前用例 biasShape[0] 始终等于 e | 添加用例设置 biasShape[0] != e |
| 471 | bias[1] != N | 当前用例 biasShape[1] 始终等于 n | 添加用例设置 biasShape[1] != n |

### 7. ValidateScaleShape 的 3 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 485 | scale dim != 4 | 当前用例 scaleShape 始终为 4 维 | 添加用例设置 scaleShape 为 3 维或 5 维 |
| 496 | scale[1] != N | 当前用例 scaleShape[1] 始终等于 n | 添加用例设置 scaleShape[1] != n |
| 507 | scale[3] != 2 | 当前用例 scaleShape[3] 始终等于 2 | 添加用例设置 scaleShape[3] != 2 |

### 8. ValidatePertokenScaleShape 的 4 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 521 | pertokenScale dim != 3 | 当前用例 pertokenScaleShape 始终为 3 维 | 添加用例设置 pertokenScaleShape 为 2 维或 4 维 |
| 527 | pertokenScale[0] != M | 当前用例 pertokenScaleShape[0] 始终等于 m | 添加用例设置 pertokenScaleShape[0] != m |
| 533 | pertokenScale[1] != CeilDiv(K,64) | 当前用例 pertokenScaleShape[1] 始终正确 | 添加用例设置 pertokenScaleShape[1] 为错误值 |
| 539 | pertokenScale[2] != 2 | 当前用例 pertokenScaleShape[2] 始终等于 2 | 添加用例设置 pertokenScaleShape[2] != 2 |

### 9. CheckMxA8W4InputShape 的 3 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 569 | K mismatch x vs w | 当前用例 x 和 w 的 K 始终一致 | 添加用例设置 x 的 K 与 w 的 K 不同 |
| 591 | K < MIN_KN_SIZE | KTooSmall 用例触发对齐检查而非最小值检查 | 修改用例使用 k=32（已对齐）并确认 MIN_KN_SIZE > 32，或使用 k=0 |
| 597 | N < MIN_KN_SIZE | NTooSmall 用例触发对齐检查而非最小值检查 | 修改用例使用 n=32（已对齐）并确认 MIN_KN_SIZE > 32，或使用 n=0 |

### 10. CheckMxA8W4AttrWithInput 的 2 条缺失路径

| 行号 | 检查内容 | 缺失原因 | 补充建议 |
|------|---------|---------|---------|
| 641 | outputBS < 0 | 当前用例 outputBS 始终 >= 0 | 添加用例设置 outputBS 为负数 |
| 662 | shareInputOffset < 0 | 当前用例 shareInputOffset 始终 >= 0 | 添加用例设置 shareInputOffset 为负数 |

---

## 结论

### 1. 整改范围覆盖率
- **总 OP_LOGE_FOR_* 路径**: 34 条
- **已覆盖**: 9 条
- **缺失**: 25 条
- **覆盖率**: 26.5%

### 2. 日志格式验证
- 9 条已覆盖路径的日志格式**全部符合预期**
- 日志包含 OpName、参数名、错误值、Reason 文本，格式规范

### 3. 异常用例说明
- 5 个异常用例因检查顺序或 API 多层检查机制导致触发了非预期的错误路径
- 这些用例仍然通过测试（返回 GRAPH_FAILED），但未覆盖预期的错误路径

### 4. 整体评价
- **整改实现正确**: 所有已覆盖的 OP_LOGE_FOR_* 路径日志格式规范，信息清晰
- **覆盖率不足**: 仅覆盖 26.5% 的整改路径，大量 shape 验证、属性验证路径缺失
- **建议优先补充**:
  1. CheckMxA8W4NzAttrPtr 的 4 条属性检查路径
  2. ValidateRowIndexShape / ValidateLogitShape 的 4 条 shape 检查路径
  3. ValidateBiasShape 的 3 条 shape 检查路径
  4. ValidatePertokenScaleShape 的 4 条 shape 检查路径

### 5. 后续工作
建议根据"缺失路径分析"章节的建议，补充 25 条缺失的测试用例，以达到 100% 覆盖率。

---

## 附录：OP_LOGE_FOR_* 宏覆盖情况

| 宏名称 | 代码中调用次数 | 覆盖用例数 | 状态 |
|--------|-------------|-----------|------|
| OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | 1 | 1 | ✅ 已覆盖 |
| OP_LOGE_FOR_INVALID_VALUE_WITH_REASON | 10 | 3 | ⚠️ 部分覆盖 |
| OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON | 9 | 3 | ⚠️ 部分覆盖 |
| OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON | 13 | 2 | ⚠️ 部分覆盖 |
| OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON | 1 | 0 | ❌ 未覆盖 |
| **总计** | **34** | **9** | ⚠️ **26.5%** |
