# PR #4260 代码检视报告

> **检视日期**：2026-04-20
> **检视范围**：allto_allv_quant_grouped_mat_mul 算子接口参数顺序调整
> **规范来源**：C++ 安全编码规范（cpp-secure.md）、MC2 领域特定规则（mc2-specific.md）、TOPK 问题清单（ascendc-topk.md）

---

## 一、检视概述

### 变更内容摘要

本次 PR 主要对 `allto_allv_quant_grouped_mat_mul` 算子进行接口参数顺序调整：

1. **输入参数顺序调整**：将 `gmmXScale` 和 `gmmWeightScale` 移动到 `gmmWeight` 之后
2. **属性顺序调整**：将 `gmm_x_quant_mode` 和 `gmm_weight_quant_mode` 移动到 `recv_counts` 之后
3. **删除 InferShape 维度校验**：删除了 MatMul K维度匹配校验逻辑
4. **代码风格优化**：简化 Format 定义，优化日志输出格式

### 检视文件清单

| 文件路径 | 变更类型 | 检视重点 |
|---------|---------|---------|
| `op_api/aclnn_allto_allv_quant_grouped_mat_mul.cpp` | 接口调整 | 参数顺序一致性、指针判空 |
| `op_api/aclnn_allto_allv_quant_grouped_mat_mul.h` | 文档更新 | 参数说明准确性 |
| `op_graph/allto_allv_quant_grouped_mat_mul_proto.h` | IR原型调整 | 输入顺序、属性顺序 |
| `op_graph/fallback_allto_allv_quant_grouped_mat_mul.cpp` | Fallback调整 | 输入/属性索引更新 |
| `op_host/allto_allv_quant_grouped_mat_mul_def.cpp` | 定义调整 | 输入顺序定义 |
| `op_host/allto_allv_quant_grouped_mat_mul_infershape.cpp` | 删除校验 | 维度校验逻辑删除风险 |
| `op_host/op_tiling/*.cpp/.h` | Tiling调整 | 输入/属性索引更新 |
| `op_kernel/allto_allv_quant_grouped_mat_mul_apt.cpp` | Kernel调整 | GM参数顺序 |
| `tests/ut/*` | UT调整 | 测试参数顺序 |

---

## 二、风险点详情

### 风险点 1：InferShape 删除 MatMul 维度匹配校验【高风险】

**规范条款**：`cpp-secure.md` 规则 4.1 外部输入合法性校验、`ascendc-topk.md` 问题 7 融合规则/InferShape/Tiling外部输入校验

**风险代码位置**：`op_host/allto_allv_quant_grouped_mat_mul_infershape.cpp`

**假设检验过程**：

| 步骤 | 分析动作 | 自信值变化 |
|-----|---------|-----------|
| H0: 删除校验是安全的 | - | 0% |
| 证据1：规范违反 | 删除了 gmmX.dim[1] 与 gmmWeight.dim[1]/dim[2] 的匹配校验 | +40% |
| 证据2：上下文防御缺失 | InferShape 中无替代校验，Tiling 中可能存在校验需验证 | +30% |
| 证据3：数据流风险 | MatMul 要求 K轴匹配，删除校验可能导致运行时错误 | +25% |
| **决策** | 自信值 95% > 60%，判定存在风险 | - |

**删除的代码片段**：

```cpp
// CheckDims 函数中删除的校验（原第 69-88 行）
auto k1 = gmmXShape->GetDim(DIM_1);
auto k2 = gmmWeightShape->GetDim(DIM_1);
if (transGmmWeight) {
    k2 = gmmWeightShape->GetDim(DIM_2);
}
if (k1 != k2) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
        "Dim of gmmX and dim of gmmWeight do not match for MatMul");
    result = ge::GRAPH_FAILED;
}

// CheckDimsOptional 函数中删除的校验（原第 97-106 行）
auto k1 = mmXShape->GetDim(DIM_1);
auto k2 = mmWeightShape->GetDim(DIM_0);
if (transMmWeight) {
    k2 = mmWeightShape->GetDim(DIM_1);
}
if (k1 != k2) {
    VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
        "Dim of x and dim of mmWeight do not match for MatMul");
    result = ge::GRAPH_FAILED;
}
```

**建议修复方案**：

1. **方案A（推荐）**：确认 Tiling 层是否有对应校验，若无则恢复 InferShape 校验
2. **方案B**：在 aclnn 接口层（`aclnn_allto_allv_quant_grouped_mat_mul.cpp`）增加 shape 校验

---

### 风险点 2：LOG API 格式化字符串类型不匹配【中风险】

**规范条款**：`cpp-secure.md` 规则 11.3 LOG API 参数类型必须与格式化说明符匹配、`mc2-specific.md` 规则 MC2-09 量化参数类型一致性

**风险代码位置**：`op_api/aclnn_allto_allv_quant_grouped_mat_mul.cpp` 第 105 行

**假设检验过程**：

| 步骤 | 分析动作 | 自信值变化 |
|-----|---------|-----------|
| H0: LOG 格式化正确 | - | 0% |
| 证据1：规范违反 | `gmmXQuantMode` 声明为 `int64_t`，使用 `%lu` 格式化 | +40% |
| 证据2：上下文防御缺失 | 无类型转换或强制匹配 | +30% |
| **决策** | 自信值 70% > 60%，判定存在风险 | - |

**风险代码片段**：

```cpp
// 第 105 行
OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
    "gmmXQuantMode should be 1(pertensor quant) or 6(mx quant), but actual is %lu.", 
    gmmXQuantMode);  // int64_t 应使用 %ld，而非 %lu
```

**建议修复方案**：

```cpp
OP_LOGE(ACLNN_ERR_PARAM_INVALID, 
    "gmmXQuantMode should be 1(pertensor quant) or 6(mx quant), but actual is %ld.", 
    gmmXQuantMode);  // int64_t 使用 %ld
```

---

### 风险点 3：属性索引更新不完整风险【中风险】

**规范条款**：`cpp-secure.md` 规则 3.3 数组索引校验、`ascendc-topk.md` 问题 5 属性获取类型需与ir原型一致

**风险代码位置**：多处文件中的索引常量定义

**假设检验过程**：

| 步骤 | 分析动作 | 自信值变化 |
|-----|---------|-----------|
| H0: 索引更新完整 | - | 0% |
| 证据1：规范违反 | 属性索引从 4/5/6 变为 6/7/8，需验证所有引用点是否同步更新 | +40% |
| 证据2：调用链风险 | 检查 `GetAttrs()->GetAttrPointer<T>(INDEX)` 是否使用正确索引 | +25% |
| **决策** | 自信值 65% > 60%，需人工确认 | - |

**变更的索引常量**：

```cpp
// allto_allv_quant_grouped_mat_mul_tiling_base.h
// 属性索引变更：
constexpr uint32_t ATTR_GMM_X_QUANT_MODE_INDEX = 4;     // 原 7
constexpr uint32_t ATTR_GMM_WEIGHT_QUANT_MODE_INDEX = 5; // 原 8
constexpr uint32_t ATTR_TRANS_GMM_WEIGHT_INDEX = 6;      // 原 4
constexpr uint32_t ATTR_TRANS_MM_WEIGHT_INDEX = 7;       // 原 5
constexpr uint32_t ATTR_PERMUTE_OUT_FLAG_INDEX = 8;      // 原 6

// 输入索引变更：
constexpr uint32_t GMM_X_SCALE_INDEX = 2U;               // 原 6
constexpr uint32_t GMM_WEIGHT_SCALE_INDEX = 3U;          // 原 7
constexpr uint32_t SEND_COUNTS_TENSOR_INDEX = 4U;        // 原 2
constexpr uint32_t RECV_COUNTS_TENSOR_INDEX = 5U;        // 原 3
constexpr uint32_t MM_X_INDEX = 6U;                      // 原 4
constexpr uint32_t MM_WEIGHT_INDEX = 7U;                 // 原 5
```

**建议修复方案**：

1. 确认所有使用这些索引的代码点都已更新
2. 运行 UT 测试验证索引更新正确性

---

### 风险点 4：日志输出变更可能丢失信息【低风险】

**规范条款**：`cpp-secure.md` 规则 11.2 LOG API 参数数量与格式化占位符必须匹配

**风险代码位置**：`op_host/op_tiling/arch35/allto_allv_quant_grouped_mat_mul_tiling_common.cpp`

**假设检验过程**：

| 步骤 | 分析动作 | 自信值变化 |
|-----|---------|-----------|
| H0: 日志变更安全 | - | 0% |
| 证据1：代码变更 | SendCounts/RecvCounts 日志从"仅输出非0值"改为"输出所有值" | +25% |
| 证据2：无规范违反 | 日志格式化参数匹配，无安全风险 | -30% |
| **决策** | 自信值 -5% < 60%，判定安全 | - |

**变更分析**：

变更前（仅输出非0值）：
```cpp
for (int i = 0; i < e_ * epWorldSize_; i++) {
    if (taskTilingInfo.sendCnt[i] != 0) {
        if (i != 0) { ss << " ,"; }
        ss << taskTilingInfo.sendCnt[i];
    }
}
```

变更后（输出所有值）：
```cpp
for (int i = 0; i < e_ * epWorldSize_; i++) {
    if (i != 0) { ss << " ,"; }
    ss << taskTilingInfo.sendCnt[i];
}
```

**分析结论**：此变更增加了日志完整性，是正向改进，无安全风险。

---

## 三、存疑问题清单

### 存疑 1：Kernel 入口参数顺序变更后 GM 地址获取是否正确

**问题描述**：Kernel 入口函数参数顺序变更后，需要确认内部 GM 地址获取逻辑是否同步更新。

**涉及代码**：`op_kernel/allto_allv_quant_grouped_mat_mul_apt.cpp`

**建议**：人工确认 Kernel 内部对 GM 参数的使用是否与新顺序一致。

---

### 存疑 2：UT 测试用例是否覆盖参数顺序变更

**问题描述**：UT 测试文件中的参数顺序已更新，需确认测试用例是否覆盖新顺序的各种场景。

**涉及代码**：`tests/ut/op_host/arch35/test_allto_allv_quant_grouped_mat_mul_tiling.cpp`

**建议**：运行完整 UT 测试套件验证变更正确性。

---

## 四、检视结论

### 风险统计

| 风险等级 | 数量 | 说明 |
|---------|-----|------|
| 高风险 | 1 | InferShape 删除维度校验 |
| 中风险 | 2 | LOG 格式化、索引更新完整性 |
| 低风险 | 0 | - |
| 存疑 | 2 | Kernel GM 地址、UT覆盖 |

### 总体评价

本次 PR 主要是接口参数顺序调整，属于**兼容性变更**。主要风险点如下：

1. **高风险**：删除 InferShape 中的 MatMul K维度匹配校验，可能导致非法 shape 输入未被拦截，建议确认 Tiling 层是否有兜底校验或恢复 InferShape 校验。

2. **中风险**：LOG 格式化字符串类型不匹配（`int64_t` 使用 `%lu`），可能导致日志输出截断或错位。

3. **建议**：运行完整 UT 测试验证参数顺序变更的正确性，特别是 Kernel 侧 GM 地址获取逻辑。

---

## 五、规范条款引用汇总

| 规范来源 | 规范编号 | 规范名称 |
|---------|---------|---------|
| cpp-secure.md | 4.1 | 外部输入合法性校验 |
| cpp-secure.md | 11.2 | LOG API 参数数量与格式化占位符必须匹配 |
| cpp-secure.md | 11.3 | LOG API 参数类型与格式化说明符必须匹配 |
| cpp-secure.md | 3.3 | 数组索引校验 |
| mc2-specific.md | MC2-09 | 量化参数类型一致性 |
| ascendc-topk.md | 5 | 属性获取类型需与ir原型一致 |
| ascendc-topk.md | 7 | 融合规则/InferShape/Tiling外部输入校验 |

---

**检视人**：CANNBot Code Review Agent
**检视完成时间**：2026-04-20