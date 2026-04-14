# GMMFR MxA8W4 aclnn通路代码串讲

## 1. 需求概述

### 1.1 背景

GMMFR（Grouped Matmul Finalize Routing）融合算子需要支持MxA8W4量化数据流：
- **激活（x1）**: float8_e4m3fn，ND格式，shape=(m, k)
- **权重（x2）**: float4_e2m1，NZ_C0_32格式，viewShape=(e, n, k)
- **权重scale**: float8_e8m0，ND格式，shape=(e, n, ceil(k/64), 2)
- **激活scale（pertokenScale）**: float8_e8m0，ND格式，shape=(m, ceil(k/64), 2)

### 1.2 计算公式

```
y_i = (x_i * perTokenScale_i) × (w_i * scale_i) + bias_i
out[rowIndex[i], :] = Σ(y_i[j - start_i] * logit[i]) + sharedInputWeight × sharedInput[j, :]
```

### 1.3 约束条件

| 约束项 | 条件 |
|--------|------|
| k对齐 | k % 32 == 0 |
| n对齐 | n % 32 == 0 |
| 专家数上限 | e <= 1024 |
| sharedInput batch | bsdp <= outputBS |
| offset范围 | sharedInputOffset + bsdp <= outputBS |
| transposeX1 | false（固定） |
| transposeX2 | true（固定） |

---

## 2. 完整调用链

```
aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize
    ↓
aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSizeCommonProcess
    ↓
CheckParams (场景分发)
    ↓
    ├─ isA8W4 && DAV_3510 → AclnnGroupedMatmulFinalizeRoutingWeightQuantChecker (新增)
    │   ├─ CheckNotNull
    │   ├─ CheckDtypeValid
    │   ├─ CheckInputOutDims
    │   ├─ CheckInputOutShape
    │   │   ├─ CheckInputOutShapeConsistency
    │   │   ├─ CheckRequiredShapes
    │   │   └─ CheckOptionalShapes
    │   └─ CheckFormat
    │       ├─ CheckXAndWeightFormat
    │       └─ CheckOtherTensorFormats
    │
    └─ 其他场景 → 原有Checker
```

---

## 3. 代码修改点详解

### 3.1 算子原型新增dtype组合

**文件**: `op_host/grouped_matmul_finalize_routing_def.cpp`

在`config91095`（ascend950配置）的各Input/Output DataType和Format列表末尾追加第14个A8W4数据类型组合：

```cpp
// x: DT_FLOAT8_E4M3FN, FORMAT_ND
// w: DT_FLOAT4_E2M1, FORMAT_FRACTAL_NZ
// scale: DT_FLOAT8_E8M0, FORMAT_ND
// bias: DT_BF16, FORMAT_ND
// pertoken_scale: DT_FLOAT8_E8M0, FORMAT_ND
// ... 其余参数同理
```

**关键点**：
- 仅追加新dtype，不修改原有代码格式
- weight格式为`FORMAT_FRACTAL_NZ`（NZ_C0_32）

### 3.2 新增A8W4专用Checker

**文件**: `op_api/grouped_matmul_finalize_routing_weight_quant_950_checker.h`（新建）

#### 3.2.1 dtype白名单

```cpp
static const std::initializer_list<op::DataType> A8W4_X_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> A8W4_W_TYPE_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> A8W4_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_PERTOKEN_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
```

#### 3.2.2 CheckNotNull

检查必选参数非空，特别验证：
- `pertokenScaleOptional` 在A8W4场景必选（原有场景可选）
- `offset` 在A8W4场景必须为nullptr

#### 3.2.3 CheckInputOutDims

校验各参数维度：
- x1: 2D
- x2: 3D
- scale: 4D
- pertokenScale: 3D
- groupList/logit/rowIndex: 1D
- out: 2D

#### 3.2.4 CheckInputOutShape

**核心校验逻辑**：

```cpp
// k/n对齐约束（A8W4特有）
if (k % 32 != 0) { return false; }
if (n % 32 != 0) { return false; }

// scale shape: (e, n, ceil(k/64), 2) 当transposeX2=true
op::Shape weightScaleTransExpectShape = {e, n, CeilDiv(k, 64), 2};

// pertokenScale shape: (m, ceil(k/64), 2)
op::Shape xScaleExpectShape = {m, CeilDiv(k, 64), 2};

// 专家数上限
if (e > 1024) { return false; }
```

#### 3.2.5 CheckFormat

- x1: 必须ND格式
- x2: 必须NZ格式（FORMAT_FRACTAL_NZ）
- 其他参数: 必须ND格式

### 3.3 CheckParams场景分发

**文件**: `op_api/aclnn_grouped_matmul_finalize_routing.cpp`

```cpp
static aclnnStatus CheckParams(GroupedMatmulParams &params)
{
    bool isA8W4 = params.x1 != nullptr && params.x2 != nullptr &&
                  params.x1->GetDataType() == DataType::DT_FLOAT8_E4M3FN &&
                  params.x2->GetDataType() == DataType::DT_FLOAT4_E2M1;

    if (isA8W4 && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        GmmFinalizeRouting::AclnnGroupedMatmulFinalizeRoutingWeightQuantChecker checker;
        aclnnStatus status = checker.CheckParams(params);
        CHECK_RET(status == ACLNN_SUCCESS, status);
        return ACLNN_SUCCESS;
    }
    // 原有场景逻辑...
}
```

### 3.4 PreMatmulCalcProcess适配

**问题**: MX场景的scale transpose一致性检查要求`transposeX2 == transposescale`，但A8W4场景`transposeX2=true`而`transposescale=false`。

**修复**:

```cpp
if (scale != nullptr && CheckType(x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_MX) &&
    CheckType(scale->GetDataType(), SCALE_TYPE_SUPPORT_LIST_MX)) {
    bool transposescale = false;
    ret = WeightNZCaseProcessForMXScale(scale, transposescale, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // A8W4场景跳过transpose一致性检查
    bool isA8W4 = x1->GetDataType() == DataType::DT_FLOAT8_E4M3FN &&
                  x2->GetDataType() == DataType::DT_FLOAT4_E2M1;
    if (!isA8W4) {
        CHECK_RET(transposeX2 == transposescale, ret);
    }
}
```

### 3.5 V2函数CheckSupportScene适配

**问题**: `aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize`中调用`CheckSupportScene`要求transpose为false，与A8W4的`transposeX2=true`冲突。

**修复**:

```cpp
// A8W4场景：使用专用Checker，跳过CheckSupportScene
bool isA8W4 = x1 != nullptr && tmpWeight != nullptr &&
              x1->GetDataType() == DataType::DT_FLOAT8_E4M3FN &&
              tmpWeight->GetDataType() == DataType::DT_FLOAT4_E2M1;
if (!isA8W4) {
    CheckSupportSceneParams sceneParams{...};
    auto ret0 = CheckSupportScene(sceneParams, transposeX1, transposeX2);
    CHECK_RET(ret0 == ACLNN_SUCCESS, ret0);
}
```

---

## 4. 发现的问题及修复

### 4.1 CheckOptionalShapes中CHECK_COND宏误用

**问题**: `CheckOptionalShapes`函数返回`bool`，但内部使用`CHECK_COND`宏会`return aclnnStatus`，导致错误码被转换为`true`（非零整数→true）。

**修复**: 将`CHECK_COND`改为显式的`if`判断+`return false`：

```cpp
// 修复前
CHECK_COND(bsdp <= outputBS, ACLNN_ERR_PARAM_INVALID, "...");

// 修复后
if (bsdp > outputBS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "...");
    return false;
}
```

### 4.2 Checker缺少k/n对齐约束检查

**问题**: SRS要求`k % 32 == 0`和`n % 32 == 0`，但Checker未校验。

**修复**: 在`CheckInputOutShape`中添加：

```cpp
if (k % 32 != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "k must be a multiple of 32, but got %ld.", k);
    return false;
}
if (n % 32 != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "n must be a multiple of 32, but got %ld.", n);
    return false;
}
```

---

## 5. UT验证结果

| 测试类别 | 用例数 | 通过数 | 状态 |
|----------|--------|--------|------|
| MxA8W4正向用例 | 8 | 8 | ✅ |
| MxA8W4负向用例 | 23 | 23 | ✅ |
| 910B2回归测试 | 26 | 26 | ✅ |
| **总计** | **57** | **57** | **✅** |

---

## 6. 总结

本次修改实现了MxA8W4量化数据流在aclnn通路的完整支持，核心改动包括：
1. 算子原型追加dtype组合
2. 新建专用Checker实现参数校验
3. CheckParams场景分发
4. PreMatmulCalcProcess和V2函数的A8W4适配
5. 修复2个bug（CHECK_COND误用、k/n对齐缺失）

所有修改保持与原有代码解耦，不影响现有MX/Pertoken数据流。
