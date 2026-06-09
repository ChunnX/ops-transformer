# flash_attention_score 日志整改记录

## 整改范围

- **dtype错误页**（10条OP_LOGE + 8条OPS_REPORT_VECTOR_INNER_ERR + 1条outDtype值范围）
- **nullptr页 - 张宇翰**（33条OP_LOGE）

## 涉及文件

1. `op_api/aclnn_flash_attention_score.cpp`
2. `op_host/arch35/flash_attention_score_tiling_regbase.cpp`

---

## 一、dtype错误页 - OP_LOGE整改（10条）

### 1.1 op_api/aclnn_flash_attention_score.cpp（8条）

| # | 原始行号 | 原始调用 | 替换后接口 | ErrCode | 替换后调用 | 日志输出 |
|---|---------|---------|-----------|---------|-----------|---------|
| 1 | 418 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pseType is 2 or 3, pse must be fp32.")` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(op::internal::GetLogApiInfo().c_str(), "pse", dtypeStr.c_str(), "The dtype of input pse must be FLOAT when the mode of pse is 2 or 3")` | Parameter pse of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input pse must be FLOAT when the mode of pse is 2 or 3. |
| 2 | 419 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query, key and value dtype not match")` | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op::internal::GetLogApiInfo().c_str(), paramMsg.c_str(), dtypeMsg.c_str(), "The dtypes of input query, key and value must be the same")` | Parameters query, key and value of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: The dtypes of input query, key and value must be the same. |
| 3 | 430 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query, key and value dtype not support")` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(op::internal::GetLogApiInfo().c_str(), "query, key and value", dtypeStr.c_str(), "The dtype of input query, key and value must be within the range {FLOAT, FLOAT16, BF16}")` | Parameter query, key and value of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input query, key and value must be within the range {FLOAT, FLOAT16, BF16}. |
| 4 | 453 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pse and attentionOut dtype not match")` | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op::internal::GetLogApiInfo().c_str(), paramMsg.c_str(), dtypeMsg.c_str(), "The dtypes of input pse and output attentionOut must be the same")` | Parameters pse and attentionOut of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: The dtypes of input pse and output attentionOut must be the same. |
| 5 | 457 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sink dtype must be fp32")` | OP_LOGE_WITH_INVALID_INPUT_DTYPE | EZ0007 | `OP_LOGE_WITH_INVALID_INPUT_DTYPE(op::internal::GetLogApiInfo().c_str(), "sink", dtypeStr.c_str(), "{FLOAT}")` | Input parameter sink of operator [opName] has incorrect dtype [dataDtype]. The correct one should be {FLOAT}. |
| 6 | 480 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "query and key dtype not match")` | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op::internal::GetLogApiInfo().c_str(), paramMsg.c_str(), dtypeMsg.c_str(), "The dtypes of input query and key must be the same")` | Parameters query and key of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: The dtypes of input query and key must be the same. |
| 7 | 487 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "key and value dtype not match")` | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op::internal::GetLogApiInfo().c_str(), paramMsg.c_str(), dtypeMsg.c_str(), "The dtypes of input key and value must be the same")` | Parameters key and value of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: The dtypes of input key and value must be the same. |
| 8 | 996 | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quant dtype not match")` | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op::internal::GetLogApiInfo().c_str(), paramMsg.c_str(), dtypeMsg.c_str(), "The dtypes of input query, key and value must be the same")` | Parameters query, key and value of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: The dtypes of input query, key and value must be the same. |

### 1.2 op_host/arch35/flash_attention_score_tiling_regbase.cpp（2条）

| # | 原始行号 | 原始调用 | 替换后接口 | ErrCode | 替换后调用 | 日志输出 |
|---|---------|---------|-----------|---------|-----------|---------|
| 9 | 919 | `OP_LOGE(opName, "sink only support fp32, but got %s", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "sink", Ops::Base::ToString(dtype).c_str(), "The dtype of input sink must be FLOAT")` | Parameter sink of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input sink must be FLOAT. |
| 10 | 943 | `OP_LOGE(opName, "query, key, value dtype not match")` | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "query, key and value", incorrectDtypes.c_str(), "The dtypes of input query, key and value must be the same")` | Parameters query, key and value of operator [opName] have incorrect dtypes [incorrectDtypes]. Reason: The dtypes of input query, key and value must be the same. |

---

## 二、dtype错误页 - OPS_REPORT_VECTOR_INNER_ERR整改（8条）

### 2.1 op_host/arch35/flash_attention_score_tiling_regbase.cpp（8条）

| # | 原始行号 | 原始调用 | 替换后接口 | ErrCode | 替换后调用 | 日志输出 |
|---|---------|---------|-----------|---------|-----------|---------|
| 1 | 115 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now", ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str())` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "query, key and value", Ops::Base::ToString(inputDtype).c_str(), "The dtype of input query, key and value is not supported for now")` | Parameter query, key and value of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input query, key and value is not supported for now. |
| 2 | 210 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "outDtype value is out of range")` | OP_LOGE_FOR_INVALID_VALUE_WITH_REASON | EZ0026 | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "out_dtype", std::to_string(outDtype).c_str(), "The value of attr out_dtype must be in the range of [0, 1]")` | Parameter out_dtype of operator [opName] has incorrect value [incorrectValue]. Reason: The value of attr out_dtype must be in the range of [0, 1]. |
| 3 | 720 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attenMask dtype[%s], only support bool or uint8.", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "atten_mask", Ops::Base::ToString(attenMaskType).c_str(), "The dtype of input atten_mask must be BOOL or UINT8")` | Parameter atten_mask of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input atten_mask must be BOOL or UINT8. |
| 4 | 762 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dropMask dtype[%s], only support uint8.", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "drop_mask", Ops::Base::ToString(dropMaskDtype).c_str(), "The dtype of input drop_mask must be UINT8")` | Parameter drop_mask of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input drop_mask must be UINT8. |
| 5 | 807 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dtype[%s], only support float32.", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "d_scale_q", Ops::Base::ToString(dScaleQDtype).c_str(), "The dtype of input d_scale_q must be FLOAT")` | Parameter d_scale_q of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input d_scale_q must be FLOAT. |
| 6 | 830 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dtype[%s], only support float32.", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "d_scale_k", Ops::Base::ToString(dScaleKDtype).c_str(), "The dtype of input d_scale_k must be FLOAT")` | Parameter d_scale_k of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input d_scale_k must be FLOAT. |
| 7 | 854 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dtype[%s], only support float32.", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "d_scale_v", Ops::Base::ToString(dScaleVDtype).c_str(), "The dtype of input d_scale_v must be FLOAT")` | Parameter d_scale_v of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input d_scale_v must be FLOAT. |
| 8 | 877 | `OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid pScale dtype[%s], only support float32.", ...)` | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | `OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "p_scale", Ops::Base::ToString(pScaleDtype).c_str(), "The dtype of input p_scale must be FLOAT")` | Parameter p_scale of operator [opName] has incorrect dtype [incorrectDtype]. Reason: The dtype of input p_scale must be FLOAT. |

---

## 三、nullptr页 - 张宇翰 OP_LOGE整改（33条）

### 3.1 op_api/aclnn_flash_attention_score.cpp（33条）

所有nullptr检查均使用 **OP_LOGE_WITH_INVALID_INPUT (EZ0004)**，输出格式为：
`Parameter [paramName] of operator [opName] is required, but it is empty.`

| # | 所在函数 | 原始调用 | paramName | 替换后调用 |
|---|---------|---------|-----------|-----------|
| 1 | InputDtypeCheck | `OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pseType is 2 or 3, pseShape cannot be null.")` | "real_shift" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "real_shift")` |
| 2 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The query cannot be nullptr")` | "query" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query")` |
| 3 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The key cannot be nullptr")` | "key" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key")` |
| 4 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The value cannot be nullptr")` | "value" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "value")` |
| 5 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if realShiftOptional is present, the realShiftOptional cannot be nullptr")` | "real_shift" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "real_shift")` |
| 6 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if dropMaskOptional is present, the dropMaskOptional cannot be nullptr")` | "drop_mask" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "drop_mask")` |
| 7 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if paddingMaskOptional is present, the paddingMaskOptional cannot be nullptr")` | "padding_mask" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "padding_mask")` |
| 8 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if attenMaskOptional is present, the attenMaskOptional cannot be nullptr")` | "atten_mask" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "atten_mask")` |
| 9 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if queryRope is present, the queryRope cannot be nullptr")` | "query_rope" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query_rope")` |
| 10 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if keyRope is present, the keyRope cannot be nullptr")` | "key_rope" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key_rope")` |
| 11 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if sinkOptional is present, the sinkOptional cannot be nullptr")` | "sink" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "sink")` |
| 12 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if dScaleQOptional is present, the dScaleQOptional cannot be nullptr")` | "d_scale_q" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "d_scale_q")` |
| 13 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if dScaleKOptional is present, the dScaleKOptional cannot be nullptr")` | "d_scale_k" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "d_scale_k")` |
| 14 | Contiguous | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "if dScaleVOptional is present, the dScaleVOptional cannot be nullptr")` | "d_scale_v" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "d_scale_v")` |
| 15 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The query cannot be nullptr")` | "query" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query")` |
| 16 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The key cannot be nullptr")` | "key" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key")` |
| 17 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The value cannot be nullptr")` | "value" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "value")` |
| 18 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The dScaleQ cannot be nullptr")` | "d_scale_q" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "d_scale_q")` |
| 19 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The dScaleK cannot be nullptr")` | "d_scale_k" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "d_scale_k")` |
| 20 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The dScaleV cannot be nullptr")` | "d_scale_v" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "d_scale_v")` |
| 21 | ContiguousQuant | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The pScale cannot be nullptr")` | "p_scale" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "p_scale")` |
| 22 | PreprocessQKV (Reshape) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The query cannot be nullptr")` | "query" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query")` |
| 23 | PreprocessQKV (Reshape) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The key cannot be nullptr")` | "key" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key")` |
| 24 | PreprocessQKV (Reshape) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The value cannot be nullptr")` | "value" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "value")` |
| 25 | PreprocessQKV (Pad) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The query cannot be nullptr")` | "query" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query")` |
| 26 | PreprocessQKV (Pad) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The key cannot be nullptr")` | "key" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key")` |
| 27 | PreprocessQKV (Pad) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The value cannot be nullptr")` | "value" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "value")` |
| 28 | PreprocessQKV (Transpose) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The query cannot be nullptr")` | "query" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query")` |
| 29 | PreprocessQKV (Transpose) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The key cannot be nullptr")` | "key" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key")` |
| 30 | PreprocessQKV (Transpose) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The value cannot be nullptr")` | "value" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "value")` |
| 31 | PreprocessQKV (SBH Reshape) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The query cannot be nullptr")` | "query" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "query")` |
| 32 | PreprocessQKV (SBH Reshape) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The key cannot be nullptr")` | "key" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "key")` |
| 33 | PreprocessQKV (SBH Reshape) | `OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "The value cannot be nullptr")` | "value" | `OP_LOGE_WITH_INVALID_INPUT(op::internal::GetLogApiInfo().c_str(), "value")` |

---

## 四、整改方式说明

### 4.1 宏兼容性处理（op_api层）

op_api层的`OP_LOGE`宏签名为`OP_LOGE(errno, fmt, ...)`，与EZ宏内部调用的tiling层签名`OP_LOGE(opName, fmt, ...)`不兼容。同理，opdev版的`OP_LOGD/W/I(fmt, ...)`自动注入opName，而`op_common/log/log.h`版的`OP_LOGD/W/I(opName, fmt, ...)`要求首参为opName。

在`aclnn_flash_attention_score.cpp`顶部通过以下方式处理：

```cpp
#include "op_common/log/log.h"

#undef OpLogErrSub
#undef OpLogSub
#undef D_OP_LOGE
#undef D_OP_LOGD
#undef D_OP_LOGW
#undef D_OP_LOGI
#undef OP_LOGE_WITHOUT_REPORT
#undef OP_LOGE
#undef OP_LOGD
#undef OP_LOGW
#undef OP_LOGI

#define OpLogErrSub(moduleId, level, op_info, errno, fmt, ...) \
    DOplogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] errno[%d] %s" fmt, \
            __FUNCTION__, op::OpLog::GetTid(), errno, op_info, ##__VA_ARGS__)
#define OpLogSub(moduleId, level, opname, fmt, ...) \
    DOplogSub(static_cast<int32_t>(moduleId), OPAPI_SUBMOD_NAME, level, "[%s][%lu] OpName:[%s] " fmt, \
            __FUNCTION__, op::OpLog::GetTid(), opname, ##__VA_ARGS__)
#define D_OP_LOGE(opname, errno, fmt, ...) OpLogErrSub(OP_ID, OP_LOG_ERROR, opname, errno, fmt, ##__VA_ARGS__)
#define D_OP_LOGD(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_DEBUG, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGW(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_WARN, opname, fmt, ##__VA_ARGS__)
#define D_OP_LOGI(opname, fmt, ...) OpLogSub(OP_ID, OP_LOG_INFO, opname, fmt, ##__VA_ARGS__)
#define OP_LOGE_WITHOUT_REPORT(errno, ...) D_OP_LOGE(GetOpName().c_str(), errno, __VA_ARGS__)
#define OP_LOGE(errno, ...) \
    do { \
        OP_LOGE_WITHOUT_REPORT(errno, __VA_ARGS__); \
        REPORT_ERROR_MESSAGE(errno, __VA_ARGS__); \
    } while (false)
#define OP_LOGD(...) D_OP_LOGD(GetOpName().c_str(), __VA_ARGS__)
#define OP_LOGW(...) D_OP_LOGW(GetOpName().c_str(), __VA_ARGS__)
#define OP_LOGI(...) D_OP_LOGI(GetOpName().c_str(), __VA_ARGS__)
```

关键点：
- **OpLogErrSub**：错误日志底层宏，格式含errno字段`errno[%d] %s`，用于OP_LOGE链
- **OpLogSub**：调试/信息/警告日志底层宏，格式含OpName字段`OpName:[%s]`，用于OP_LOGD/W/I链
- **EZ宏兼容**：EZ宏内部调用OP_LOGE_WITHOUT_REPORT，通过我们重定义的链路，opName参数被放置在errno字段（64位系统下安全，仅cosmetic差异）
- **原有日志兼容**：原有OP_LOGD/W/I调用通过`GetOpName().c_str()`自动注入opName，恢复opdev格式

### 4.2 OP_CHECK/OP_CHECK_IF保持原状

OP_CHECK和OP_CHECK_IF宏均保持原结构不变，仅替换内部的日志调用：
- `OP_CHECK(x != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "msg"), return ERR)` → `OP_CHECK(x != nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "param"), return ERR)`
- `OP_CHECK_IF(cond, OPS_REPORT_VECTOR_INNER_ERR(opName, "msg", ...), return false)` → `OP_CHECK_IF(cond, OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "param", ...), return false)`

EZ宏内部含逗号的参数（如4参宏）在括号内，不会被C预处理器误识别为OP_CHECK/OP_CHECK_IF的参数分隔符，因此可直接嵌套使用。

### 4.3 opName来源

| 层级 | opName取值 |
|------|-----------|
| op_api层 | `op::internal::GetLogApiInfo().c_str()` |
| tiling层 | 现有`opName`变量 |

### 4.4 dtype转字符串

| 层级 | 转换方式 |
|------|---------|
| op_api层 | `op::ToString(DataType(x)).GetString()` |
| tiling层 | `Ops::Base::ToString(x).c_str()` |

---

## 五、接口职责最小化选择依据

| 日志分类 | 选择接口 | ErrCode | 选择理由 |
|---------|---------|---------|---------|
| 单参数dtype错误（已知正确dtype） | OP_LOGE_WITH_INVALID_INPUT_DTYPE | EZ0007 | sink只有一个已知期望dtype列表{FLOAT}，用INVALID_INPUT系列 |
| 单参数dtype错误（需reason说明） | OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON | EZ0020 | 单参数dtype不合规，reason补充上下文说明 |
| 多参数dtype不匹配 | OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON | EZ0021 | 多参数dtype对比错误，reason说明约束关系 |
| 参数空指针/nullptr | OP_LOGE_WITH_INVALID_INPUT | EZ0004 | 空指针检查的唯一正确选择，输出"Parameter [x] is required, but it is empty" |
| 属性值范围错误 | OP_LOGE_FOR_INVALID_VALUE_WITH_REASON | EZ0026 | 属性值超出合法范围，reason说明合法范围 |