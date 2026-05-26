# OP_LOGE 日志整改详细策略文档

## 一、修改策略概述

### 1.1 核心原则

将原有的 `OP_LOGE(...)` 和 `OPS_REPORT_VECTOR_INNER_ERR(...)` 替换为标准化错误上报宏（EZ0001-EZ0034），使错误信息可被 CANN Error Manager 统一上报和管理。

### 1.2 宏选择黄金法则

| 错误类型 | 宏选择 | 错误码 |
|----------|--------|--------|
| **参数值为空（nullptr）** | `OP_LOGE_WITH_INVALID_INPUT` (2参数) | EZ0004 |
| **单个参数值错误（有期望值）** | `OP_LOGE_FOR_INVALID_VALUE` (4参数) | EZ0024 |
| **单个参数值错误（需原因说明）** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` (4参数) | EZ0026 |
| **多个参数值错误** | `OP_LOGE_FOR_INVALID_VALUES_WITH_REASON` (4参数) | EZ0027 |
| **属性值错误** | `OP_LOGE_WITH_INVALID_ATTR` (4参数) | EZ0002 |

---

## 二、修改前后对比（21条）

### 文件：flash_attention_score_tiling_regbase.cpp（17条）

---

### **修改 #1：head_num 为零检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行159 |
| **错误类型** | 参数值为零 |
| **错误码** | EZ0024 |

**修改前：**
```cpp
OP_CHECK_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "head num is zero."), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(n1Size == 0, OP_LOGE_FOR_INVALID_VALUE(opName, "head_num", "0", "positive"), return false);
```

---

### **修改 #2：keepProb 范围检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行160-161 |
| **错误类型** | 属性值超出范围 |
| **错误码** | EZ0002 |

**修改前：**
```cpp
OP_CHECK_IF(keepProb <= 0.0 || keepProb > 1.0,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "keepProb value must be in range of (0, 1]."), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(keepProb <= 0.0 || keepProb > 1.0,
            OP_LOGE_WITH_INVALID_ATTR(opName, "keep_prob", std::to_string(keepProb).c_str(), "in range of (0, 1]"), return false);
```

---

### **修改 #3：pseType 范围检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行195-197 |
| **错误类型** | 属性值超出范围 |
| **错误码** | EZ0002 |

**修改前：**
```cpp
OP_CHECK_IF(pseType < 0 || pseType >= static_cast<uint8_t>(PseType::PSE_INVALID_TYPE),
            OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType value is out of range"), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(pseType < 0 || pseType >= static_cast<uint8_t>(PseType::PSE_INVALID_TYPE),
            OP_LOGE_WITH_INVALID_ATTR(opName, "pse_type", std::to_string(pseType).c_str(),
                ("in range [0, " + std::to_string(static_cast<uint8_t>(PseType::PSE_INVALID_TYPE) - 1) + "]").c_str()), return false);
```

---

### **修改 #4：actual_seq_len 内存分配失败**

| 项目 | 内容 |
|------|------|
| **位置** | 行294-295 |
| **错误类型** | 参数值导致内存分配失败 |
| **错误码** | EZ0026 |

**修改前：**
```cpp
try {
    res.reserve(seqLen);
} catch (...) {
    OPS_REPORT_VECTOR_INNER_ERR(opName, "Init actual_seq_len failed, array is too long.");
    return false;
}
```

**修改后：**
```cpp
try {
    res.reserve(seqLen);
} catch (...) {
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "actual_seq_len",
        std::to_string(seqLen).c_str(), "array is too long, memory allocation failed");
    return false;
}
```

---

### **修改 #5：N2 为零检查（第一处）**

| 项目 | 内容 |
|------|------|
| **位置** | 行369 |
| **错误类型** | 参数值为零 |
| **错误码** | EZ0024 |

**修改前：**
```cpp
OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(n2Size == 0, OP_LOGE_FOR_INVALID_VALUE(opName, "N2", "0", "positive"), return false);
```

---

### **修改 #6：H 为零检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行430 |
| **错误类型** | 参数值为零（多变量） |
| **错误码** | EZ0024 |

**修改前：**
```cpp
OP_CHECK_IF(h1 == 0 || h2 == 0 || h3 == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "H is zero."), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(h1 == 0 || h2 == 0 || h3 == 0, OP_LOGE_FOR_INVALID_VALUE(opName, "H", "0", "positive"), return false);
```

---

### **修改 #7：hRope 必须是 n1Size 的倍数**

| 项目 | 内容 |
|------|------|
| **位置** | 行435-437 |
| **错误类型** | 参数值约束关系 |
| **错误码** | EZ0026 |

**修改前：**
```cpp
OP_CHECK_IF(hRope % n1Size != 0,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "hRope [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
            return false);
```

**修改后：**
```cpp
OP_CHECK_IF(hRope % n1Size != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "hRope",
                std::to_string(hRope).c_str(),
                ("should be a multiple of n1Size, current n1Size is " + std::to_string(n1Size)).c_str()), return false);
```

---

### **修改 #8：N2 为零检查（第二处）**

| 项目 | 内容 |
|------|------|
| **位置** | 行459 |
| **错误类型** | 参数值为零 |
| **错误码** | EZ0024 |

**修改前：**
```cpp
OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(n2Size == 0, OP_LOGE_FOR_INVALID_VALUE(opName, "N2", "0", "positive"), return false);
```

---

### **修改 #9：N2 为零检查（第三处）**

| 项目 | 内容 |
|------|------|
| **位置** | 行479 |
| **错误类型** | 参数值为零 |
| **错误码** | EZ0024 |

**修改前：**
```cpp
OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
```

**修改后：**
```cpp
OP_CHECK_IF(n2Size == 0, OP_LOGE_FOR_INVALID_VALUE(opName, "N2", "0", "positive"), return false);
```

---

### **修改 #10：context 无效检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行508-510 |
| **错误类型** | context 无效或获取属性失败 |
| **错误码** | EZ0026 |

**修改前：**
```cpp
OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
            return ge::GRAPH_FAILED);
```

**修改后：**
```cpp
OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "context",
            "invalid", "context is null or failed to get attrs from context"),
            return ge::GRAPH_FAILED);
```

---

### **修改 #11：dSize 和 dSizeRope 约束检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行528-530 |
| **错误类型** | 多参数约束 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
if (hasRope && (dSize != 128 || dSizeRope != 64)) {
    OPS_REPORT_VECTOR_INNER_ERR(opName, "MLA concat only support dSize=128, dSizeRope=64.");
    return ge::GRAPH_FAILED;
}
```

**修改后：**
```cpp
if (hasRope && (dSize != 128 || dSizeRope != 64)) {
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "dSize and dSizeRope",
        (std::to_string(dSize) + " and " + std::to_string(dSizeRope)).c_str(),
        "MLA concat only support dSize=128 and dSizeRope=64 when hasRope=true");
    return ge::GRAPH_FAILED;
}
```

---

### **修改 #12：SetPseAlibiParamsRegbase 失败**

| 项目 | 内容 |
|------|------|
| **位置** | 行1009-1010 |
| **错误类型** | pse alibi 参数设置失败 |
| **错误码** | EZ0026 |

**修改前：**
```cpp
OP_CHECK_IF(!SetPseAlibiParamsRegbase(), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to set pse alibi info."),
            return ge::GRAPH_FAILED);
```

**修改后：**
```cpp
OP_CHECK_IF(!SetPseAlibiParamsRegbase(), OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pse_alibi_params",
            "invalid", "fail to set pse alibi info"),
            return ge::GRAPH_FAILED);
```

---

### **修改 #13：preTokens 和 nextTokens 与 S1 配置错误**

| 项目 | 内容 |
|------|------|
| **位置** | 行1292-1294 |
| **错误类型** | 多参数配置错误 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
OP_LOGE(context_,
        "PreTokens[%ld] and nextTokens[%ld] config error with S1[%ld], there is no valid data block.",
        oriPreTokens, oriNextTokens, minS1Val);
return false;
```

**修改后：**
```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "preTokens and nextTokens",
    (std::to_string(oriPreTokens) + " and " + std::to_string(oriNextTokens)).c_str(),
    "config error with S1, there is no valid data block");
return false;
```

---

### **修改 #14：preTokens 和 nextTokens 与 S2 配置错误**

| 项目 | 内容 |
|------|------|
| **位置** | 行1310-1312 |
| **错误类型** | 多参数配置错误 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
OP_LOGE(context_,
        "PreTokens[%ld] and nextTokens[%ld] config error with S2[%ld], there is no valid data block.",
        oriPreTokens, oriNextTokens, minS2Val);
return false;
```

**修改后：**
```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "preTokens and nextTokens",
    (std::to_string(oriPreTokens) + " and " + std::to_string(oriNextTokens)).c_str(),
    "config error with S2, there is no valid data block");
return false;
```

---

### **修改 #15：sparseMode 不支持**

| 项目 | 内容 |
|------|------|
| **位置** | 行1350-1351 |
| **错误类型** | 属性值超出范围 |
| **错误码** | EZ0002 |

**修改前：**
```cpp
if (sparseMode > static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
    OP_LOGE(context_, "Not support sparse mode of %ld.", sparseMode);
    return false;
}
```

**修改后：**
```cpp
if (sparseMode > static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
    OP_LOGE_WITH_INVALID_ATTR(context_->GetNodeName(), "sparse_mode",
        std::to_string(sparseMode).c_str(), "0 to 6");
    return false;
}
```

---

### **修改 #16：prefixN 缺失检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行1646-1647 |
| **错误类型** | 输入参数为空 |
| **错误码** | EZ0026 |

**修改前：**
```cpp
if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
    OP_LOGE(context_, "[%s] prefixN must be provided when sparseMode=6.", templateName);
    failReason << "prefixN must be provided when sparseMode=6";
}
```

**修改后：**
```cpp
if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "prefixN",
        "empty", "prefixN must be provided when sparseMode=6 (PREFIX_COMPRESS)");
}
```

---

### **修改 #17：S1 和 S2 约束检查**

| 项目 | 内容 |
|------|------|
| **位置** | 行1747-1749 |
| **错误类型** | 多参数约束 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
if (s1Size != s2Size) {
    OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
    return false;
}
```

**修改后：**
```cpp
if (s1Size != s2Size) {
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "S1 and S2",
        (std::to_string(s1Size) + " and " + std::to_string(s2Size)).c_str(),
        "Pse alibi only support same S1 and S2 when S1 is larger than 1024");
    return false;
}
```

---

### 文件：flash_attention_score_tiling_varlen.cpp（4条）

---

### **修改 #18：actualSeqQLen/KvLen 与 qStartIdx/kvStartIdx 约束**

| 项目 | 内容 |
|------|------|
| **位置** | 行109-112 |
| **错误类型** | 多参数约束 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
OP_LOGE(context_, "Inner pse sparse mode 8 is only supported when actualSeqLenData[0] %ld + qStartIdx %ld - actualSeqLenKvData[0] %ld - kvStartIdx %ld == 0.",
        actualSeqLenData[0], qStartIdx, actualSeqLenKvData[0], kvStartIdx);
return false;
```

**修改后：**
```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "actualSeqQLen, qStartIdx, actualSeqKvLen and kvStartIdx",
    (std::to_string(actualSeqLenData[0]) + ", " + std::to_string(qStartIdx) + ", " +
     std::to_string(actualSeqLenKvData[0]) + " and " + std::to_string(kvStartIdx)).c_str(),
    "actualSeqQLen[0] - actualSeqKvLen[0] + qStartIdx - kvStartIdx should be 0 for inner pse sparse mode 8");
return false;
```

---

### **修改 #19：actualSeqQLen 必须等于 actualSeqKvLen（sparse mode 8）**

| 项目 | 内容 |
|------|------|
| **位置** | 行117-119 |
| **错误类型** | 两参数必须相等 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
OP_LOGE(context_, "Inner pse sparse mode 8 is only supported when actualSeqQLen[%ld] %ld and actualSeqKvLen[%ld] %ld are equal.", 
        i, actualSeqLenData[i], i, actualSeqLenKvData[i]);
return false;
```

**修改后：**
```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "actualSeqQLen and actualSeqKvLen",
    (std::to_string(actualSeqLenData[i]) + " and " + std::to_string(actualSeqLenKvData[i])).c_str(),
    "actualSeqQLen should equal to actualSeqKvLen for inner pse sparse mode 8");
return false;
```

---

### **修改 #20：actualSeqQLen 必须等于 actualSeqKvLen（inner pse alibi）**

| 项目 | 内容 |
|------|------|
| **位置** | 行143-145 |
| **错误类型** | 两参数必须相等 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
OP_LOGE(context_, "Inner pse alibi is only support when actualSeqQLen and actualSeqKvLen are equal.");
return false;
```

**修改后：**
```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "actualSeqQLen and actualSeqKvLen",
    (std::to_string(actualSeqLenData[i]) + " and " + std::to_string(actualSeqLenKvData[i])).c_str(),
    "actualSeqQLen should equal to actualSeqKvLen for inner pse alibi");
return false;
```

---

### **修改 #21：actualSeqQLen 必须等于 actualSeqKvLen（pse alibi）**

| 项目 | 内容 |
|------|------|
| **位置** | 行160-162 |
| **错误类型** | 两参数必须相等 |
| **错误码** | EZ0027 |

**修改前：**
```cpp
OP_LOGE(context_, "Pse alibi only support when actualSeqQLen and actualSeqKvLen are equal.");
return false;
```

**修改后：**
```cpp
OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "actualSeqQLen and actualSeqKvLen",
    (std::to_string(actualSeqLenData[i]) + " and " + std::to_string(actualSeqLenKvData[i])).c_str(),
    "actualSeqQLen should equal to actualSeqKvLen for pse alibi");
return false;
```

---

## 三、错误码映射汇总

| 错误码 | 宏名称 | 参数数量 | 使用场景 | 本次使用次数 |
|--------|--------|----------|----------|--------------|
| **EZ0002** | `OP_LOGE_WITH_INVALID_ATTR` | 4 | 属性值错误 | 3 |
| **EZ0024** | `OP_LOGE_FOR_INVALID_VALUE` | 4 | 单参数值错误（有期望值） | 5 |
| **EZ0026** | `OP_LOGE_FOR_INVALID_VALUE_WITH_REASON` | 4 | 单参数值错误（需原因说明） | 5 |
| **EZ0027** | `OP_LOGE_FOR_INVALID_VALUES_WITH_REASON` | 4 | 多参数值错误 | 8 |

---

## 四、关键技术要点

### 4.1 数值类型转换

```cpp
// int64_t/uint64_t/size_t 必须用 std::to_string()
std::to_string(n1Size).c_str()

// float 同样需要转换
std::to_string(keepProb).c_str()
```

### 4.2 多参数值拼接

```cpp
// 参数名用 " and " 连接
"actualSeqQLen and actualSeqKvLen"

// 值也用 " and " 或 ", " 连接
(std::to_string(val1) + " and " + std::to_string(val2)).c_str()
```

### 4.3 算子名获取

```cpp
// 在 TilingContext 环境中使用
context_->GetNodeName()

// 若已有 opName 变量，直接使用
opName
```

### 4.4 OP_CHECK_IF 中使用宏

```cpp
// 宏作为第二个参数（log_func），不加分号
OP_CHECK_IF(condition, OP_LOGE_FOR_INVALID_VALUE(...), return false);

// 独立 if 语句中，加分号
if (condition) {
    OP_LOGE_FOR_INVALID_VALUE(...);
    return false;
}
```

---

## 五、修改文件清单

| 文件路径 | 修改数量 |
|----------|----------|
| `op_host/arch35/flash_attention_score_tiling_regbase.cpp` | 17 |
| `op_host/arch35/flash_attention_score_tiling_varlen.cpp` | 4 |
| **合计** | **21** |

---

## 六、责任人筛选信息

**原始Excel文件**: `C:\Users\wangke\Downloads\flash_attention_score_errmsg.xlsx`
**筛选Sheet**: `value错误`
**筛选责任人**: 王科
**筛选结果**: 21条记录