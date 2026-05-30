# flash_attention_score size 错误日志整改记录

依据 `errmsg/excel/flash_attention_score_errmsg.xlsx` 工作表 **size错误**（共 12 条），在 `errmsg/attention/flash_attention_score` 内将 `OP_LOGE` / `OPS_REPORT_VECTOR_INNER_ERR` 替换为 `opbase/include/op_common/log/log.h` 中职责最小的预定义宏。

说明：

- 原 `OPS_REPORT_VECTOR_INNER_ERR` 上报 **E89999**（内部错误）；本次条目均为用户可控制的 shape/head 维度/size 校验，改为 **EZ0026 / EZ0027**（`OP_LOGE_FOR_INVALID_VALUE*_WITH_REASON`）。
- 本算子目录内 **无** `OPS_REPORT_CUBE_INNER_ERR` 的 size 类条目，故未改动 CUBE 宏。
- 原日志含场景/条件说明时，统一使用 **\_WITH_REASON** 接口，并将原因写入 `reason` 字段。

## 宏选择原则

| 场景 | 选用宏 | 错误码 |
|------|--------|--------|
| 单参数 size/数值非法（含范围、为 0） | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` | EZ0026 |
| 多参数 size 关系（如 head_num 与 n2） | `OP_LOGE_FOR_INVALID_VALUES_WITH_REASON` | EZ0027 |

---

## 1. op_host/flash_attention_score_tiling.cpp:133

| 项目 | 内容 |
|------|------|
| **整改前** | `OP_LOGE(context, "kernelType size is invalid, kernelType size is %u.", kernelTypeSize)` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context, "key", std::to_string(kernelTypeSize).c_str(), "The data type byte size of input key should be greater than 0")` |
| **参数** | input `key`（`KEY_INPUT_INDEX = 1`） |
| **原因** | dtype 字节长度为 0，需说明 input key 的 byte size 应大于 0 |

---

## 2. op_host/flash_attention_score_tiling.cpp:184

| 项目 | 内容 |
|------|------|
| **整改前** | `OP_LOGE(context, "float data size is invalid, size is %u.", floatDataSize)` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context, "softmax_max and softmax_sum", std::to_string(floatDataSize).c_str(), "The data type byte size of output softmax_max and softmax_sum should be greater than 0")` |
| **参数** | output `softmax_max`、`softmax_sum` |
| **原因** | fp32 字节长度为 0 时不满足空 tensor 拷贝约束 |

---

## 3. op_host/arch35/flash_attention_score_tiling_regbase.cpp:255

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query and key", std::to_string(gSize).c_str(), "The group size derived from input query and key shapes should be greater than 0")` |
| **参数** | input `query`、`key`（由 shape 推导 gSize） |

---

## 4. op_host/arch35/flash_attention_score_tiling_regbase.cpp:256

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query and key", std::to_string(n2Size).c_str(), "The n2 size derived from input query and key shapes should be greater than 0")` |
| **参数** | input `query`、`key` |

---

## 5. op_host/arch35/flash_attention_score_tiling_regbase.cpp:258

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not support <= 0")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query and key", std::to_string(dSize).c_str(), "The head dimension size of input query and key should be greater than 0")` |
| **参数** | input `query`、`key` |

---

## 6. op_host/arch35/flash_attention_score_tiling_regbase.cpp:260

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "value's dSize is not support <= 0")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "value", std::to_string(dSizeV).c_str(), "The head dimension size of input value should be greater than 0")` |
| **参数** | input `value` |

---

## 7. op_host/arch35/flash_attention_score_tiling_regbase.cpp:264

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld]", n1Size, n2Size)` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "head_num", (std::to_string(n1Size) + " and " + std::to_string(n2Size)).c_str(), "The value of attr head_num should be a multiple of n2 size derived from input query and key shapes")` |
| **参数** | attr `head_num` 与 shape 推导的 n2 |
| **原因** | 多值关系校验，使用 EZ0027 |

---

## 8. op_host/arch35/flash_attention_score_tiling_regbase.cpp:430

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "h1 [%ld] should be a multiple of n1Size [%ld].", h1, n1Size)` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query", std::to_string(h1).c_str(), "The H dimension size of input query should be a multiple of attr head_num")` |
| **参数** | input `query`、attr `head_num` |

---

## 9. op_host/arch35/flash_attention_score_tiling_regbase.cpp:978

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not in range:(0, 768]")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query and key", std::to_string(dSize).c_str(), "The head dimension size of input query and key should be in range (0, 768]")` |
| **参数** | input `query`、`key` |
| **原因** | 原消息含范围条件 `(0, 768]` |

---

## 10. op_host/arch35/flash_attention_score_tiling_regbase.cpp:980

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "value dSize is not in range:(0, 768]")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "value", std::to_string(dSizeV).c_str(), "The head dimension size of input value should be in range (0, 768]")` |
| **参数** | input `value` |

---

## 11. op_host/arch35/flash_attention_score_tiling_varlen.cpp:77

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "Query or Key dSize is not in range:(0, 768]")` |
| **整改后** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "query and key", std::to_string(dBasicBlock).c_str(), "The aligned head dimension size of input query and key should be 64, 128, 192, or 256 when not greater than 256")` |
| **参数** | input `query`、`key` |
| **原因** | `switch(dBasicBlock)` default 分支：对齐后的 D 块不在支持档位，需说明对齐规则 |

---

## 12. op_host/arch35/flash_attention_score_tiling_basic.cpp:75

| 项目 | 内容 |
|------|------|
| **整改前** | `OPS_REPORT_VECTOR_INNER_ERR(opName, "Query or Key dSize is not in range:(0, 768]")` |
| **整改后** | 与第 11 条相同（`CalcDBasicBlock` default 分支） |

---

## 未纳入本次 size 工作表但相关的说明

Excel 中 **shape错误** / **value错误** 等其它工作表仍含 `shapeSize`、`s1Size` 等日志，不在本次 **size错误** 12 条范围内，未修改。

`OPS_REPORT_CUBE_INNER_ERR`：在 `flash_attention_score` 源码中无 size 类调用，本次无替换。
