# V2 算子收编 V1 算子实现架构设计

## 文档信息

- **设计目标**: V2 算子收编 V1 算子实现
- **涉及算子**: 
  - `allto_allv_grouped_mat_mul` (V1) → `allto_allv_grouped_mat_mul_v2` (V2)
  - `grouped_mat_mul_allto_allv` (V1) → `grouped_mat_mul_allto_allv_v2` (V2)
- **设计日期**: 2026-05-28
- **设计者**: Architect

---

## 1. 现状调研

### 1.1 代码结构对比

#### 目录结构

V1 和 V2 算子目录结构相同：

```
算子目录/
├── op_api/           # aclnn 两段式接口
│   ├── aclnn_xxx.h
│   ├── aclnn_xxx.cpp
│   └── aclnnInner_xxx.h (仅 V2)
├── op_graph/         # 图模式执行
│   ├── xxx_gen_task_training.cpp
│   ├── fallback_xxx.cpp
│   └── xxx_proto.h
├── op_kernel/        # Kernel 实现
├── op_host/          # Host 侧代码
├── examples/         # 示例代码
└── tests/            # 测试代码
```

#### 关键差异

| 组件 | V1 | V2 | 说明 |
|------|----|----|------|
| **aclnn 参数** | 无 `commMode` | 新增 `const char* commMode` | V2 通过参数指定通信方式 |
| **输出参数** | (allto_allv) 无 `permuteOutOptional` | 新增 `aclTensor* permuteOutOptional` | V2 支持输出 permute 结果 |
| **inner 接口** | 有 `aclnnInner_xxx` | 有 `aclnnInner_xxx_v2` | 内部实现接口 |
| **通信方式判断** | 环境变量 + 运行时判断 | 参数传递 + 运行时使用 | V2 将判断逻辑外部化 |

---

### 1.2 V1 通信方式判断逻辑

#### 判断机制

**环境变量检查：**

```cpp
// mc2/common/utils/mc2_comm_utils.h
namespace Mc2Comm {
    constexpr char COMM_MODE_ENV[] = "ENV_MC2_COMM_MODE_AICPU";
    constexpr uint8_t COMM_MODE_CCU = 0;    // 默认模式
    constexpr uint8_t COMM_MODE_AICPU = 1;  // AICPU 模式
    
    inline uint8_t GetCommModeFromEnv() {
        auto commModeEnv = getenv(COMM_MODE_ENV);
        if (commModeEnv != nullptr) {
            return COMM_MODE_AICPU;  // 环境变量存在，使用 AICPU
        }
        return COMM_MODE_CCU;  // 默认使用 CCU
    }
}
```

#### 判断位置

**位置 1: op_api/aclnn_xxx.cpp 的 Execute 函数**

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMul(void* workspace, uint64_t workspaceSize, 
                                        aclOpExecutor* executor, aclrtStream stream) {
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        }
    }
    return aclnnInnerAlltoAllvGroupedMatMul(...);
}
```

**判断逻辑：**
- 仅对 arch35 (DAV_3510) 平台进行判断
- 环境变量存在 → AICPU 模式
- 环境变量不存在 → CCU 模式（默认）

**位置 2: op_graph/xxx_gen_task_training.cpp**

```cpp
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context) {
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    
    const char* serverType = nullptr;
    const char* streamType = nullptr;
    
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
    } else if (isArch35) {
        serverType = "ccu server";
        streamType = "ccu_stream";
    } else {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvGroupedMatMulGenTaskFunc(...) {
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        // arch22 始终使用 AICPU GenTask
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);  // AICPU
    }
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);  // CCU
}
```

**判断逻辑：**
- CalcParamFunc 和 GenTaskFunc 都有判断
- arch22 平台：始终使用 AICPU
- arch35 平台：根据环境变量选择 AICPU 或 CCU

---

### 1.3 V2 的 commMode 参数机制

#### 参数定义

```cpp
// V2 aclnn 接口定义
ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight,
    const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional,
    const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    const char* group, int64_t epWorldSize,
    const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    const char* commMode,  // 新增参数
    aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor);
```

**commMode 参数规范：**

| 值 | 说明 | 对应通信方式 |
|----|------|--------------|
| `""` (空字符串) | 默认模式 | arch22: AICPU; arch35: CCU |
| `"ai_cpu"` | AICPU 模式 | 所有架构: AICPU |
| `"ccu"` | CCU 模式 | 仅 arch35: CCU |

#### 参数传递机制

**GetWorkspaceSize 中存储：**

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(..., const char* commMode, ...) {
    // 参数校验
    if (commMode == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    
    // 调用 inner 接口
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulV2GetWorkspaceSize(..., commMode, ...);
    
    // 在 executor 中存储 commMode（供 Execute 使用）
    if (ret == ACLNN_SUCCESS && *executor != nullptr) {
        char* commModePtr = new char[strlen(commMode) + 1];
        strcpy(commModePtr, commMode);
        NnopbaseSetUserHandle(*executor, commModePtr);  // 存储到 executor
    }
    
    return ret;
}
```

**Execute 中获取：**

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulV2(void* workspace, uint64_t workspaceSize, 
                                         aclOpExecutor* executor, aclrtStream stream) {
    const char* commMode = reinterpret_cast<char*>(NnopbaseGetUserHandle(executor));
    if (commMode == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    
    SetNnopbaseHcclServerTypeByArch(executor, commMode);  // 根据架构设置通信方式
    return aclnnInnerAlltoAllvGroupedMatMulV2(...);
}
```

#### GenTask 中的使用

```cpp
ge::Status AlltoAllvGroupedMatMulV2CalcParamFunc(gert::ExeResGenerationContext *context) {
    auto attrs = context->GetAttrs();
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    
    // 根据 commMode 选择 serverType 和 streamType
    if (commMode == nullptr || strlen(commMode) == 0) {
        if (isArch35) {
            serverType = "ccu server";
            streamType = "ccu_stream";
        } else {
            serverType = "aicpu kfc server";
            streamType = "kfc_stream";
        }
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
    } else if (strcmp(commMode, "ccu") == 0) {
        serverType = "ccu server";
        streamType = "ccu_stream";
    }
    
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvGroupedMatMulV2GenTaskFunc(...) {
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);  // arch22: AICPU
    }
    
    auto attrs = context->GetAttrs();
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    
    if (commMode == nullptr || strlen(commMode) == 0 || strcmp(commMode, "ccu") == 0) {
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);  // CCU
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);  // AICPU
    }
}
```

---

### 1.4 fallback 和 gentask 通路分析

#### fallback 通路

**位置**: `op_graph/fallback_xxx.cpp`

**执行流程**:

```cpp
static ge::graphStatus AlltoAllvGroupedMatMulExecuteFunc(gert::OpExecuteContext* host_api_ctx) {
    // 1. 获取输入和属性
    const gert::Tensor* gmmX = host_api_ctx->GetInputTensor(...);
    const gert::Tensor* gmmWeight = host_api_ctx->GetInputTensor(...);
    ...
    const gert::RuntimeAttrs* attrs = host_api_ctx->GetAttrs();
    const char* group = attrs->GetStr(...);
    ...
    
    // 2. 参数校验
    ge::graphStatus ret = CheckInputsAndAttrs(...);
    
    // 3. 解析 sendCounts 和 recvCounts
    std::vector<int64_t> actSendCountsSeqArray;
    std::vector<int64_t> actRecvCountsSeqArray;
    ret = ParseSendRecvCounts(...);
    
    // 4. 获取输出
    auto gmmY = host_api_ctx->GetOutputTensor(...);
    ...
    
    // 5. 调用 aclnn 接口
    const auto apiRet = EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMul,
                                       gmmX, gmmWeight, ..., gmmY, mmY, permuteOut);
    
    return ge::GRAPH_SUCCESS;
}
```

**特点**:
- 直接调用 V1 的 aclnn 接口
- **无通信方式判断逻辑**（依赖 aclnn 内部判断）
- 参数来自 OpExecuteContext 的输入和属性

#### gentask 通路

**位置**: `op_graph/xxx_gen_task_training.cpp`

**执行流程**:
- CalcParamFunc: 计算执行参数（serverType、streamType）
- GenTaskFunc: 生成执行任务（调用对应的 GenTask 回调）

**特点**:
- **有明确的通信方式判断逻辑**
- 根据判断结果选择不同的 GenTask 回调
- 参数来自 ExeResGenerationContext

#### 调用关系图

```
用户调用
  ├── aclnn 接口（单算子调用）
  │   ├── GetWorkspaceSize
  │   │   └── 调用 aclnnInner_xxx
  │   └── Execute
  │       ├── 判断通信方式（V1）
  │       └── 调用 aclnnInner_xxx
  │
  └── 图模式（Graph执行）
      ├── fallback 通路（不支持 gen_task）
      │   └── OpExecuteFunc
      │       └── 调用 aclnn_xxx（间接判断通信方式）
      │
      └── gentask 通路（支持 gen_task）
          ├── CalcParamFunc（判断通信方式）
          └── GenTaskFunc（选择 GenTask 回调）
```

---

### 1.5 V1 和 V2 参数映射

#### allto_allv_grouped_mat_mul 参数对比

| 参数位置 | V1 参数 | V2 参数 | 说明 |
|---------|---------|---------|------|
| 1 | `gmmX` | `gmmX` | 相同 |
| 2 | `gmmWeight` | `gmmWeight` | 相同 |
| 3 | `sendCountsTensorOptional` | `sendCountsTensorOptional` | 相同 |
| 4 | `recvCountsTensorOptional` | `recvCountsTensorOptional` | 相同 |
| 5 | `mmXOptional` | `mmXOptional` | 相同 |
| 6 | `mmWeightOptional` | `mmWeightOptional` | 相同 |
| 7 | `group` | `group` | 相同 |
| 8 | `epWorldSize` | `epWorldSize` | 相同 |
| 9 | `sendCounts` | `sendCounts` | 相同 |
| 10 | `recvCounts` | `recvCounts` | 相同 |
| 11 | `transGmmWeight` | `transGmmWeight` | 相同 |
| 12 | `transMmWeight` | `transMmWeight` | 相同 |
| 13 | `permuteOutFlag` | `permuteOutFlag` | 相同 |
| **新增** | - | **`commMode`** | **V2 新增，V1 需补充** |
| 14 | `gmmY` | `gmmY` | 相同 |
| 15 | `mmYOptional` | `mmYOptional` | 相同 |
| **新增** | - | **`permuteOutOptional`** | **V2 新增输出** |
| 16 | `workspaceSize` | `workspaceSize` | 相同 |
| 17 | `executor` | `executor` | 相同 |

**参数映射策略**:
- V1 → V2: 所有参数直接传递
- commMode: 根据判断逻辑生成
- permuteOutOptional: 根据 permuteOutFlag 传递（为 nullptr 或实际 tensor）

#### grouped_mat_mul_allto_allv 参数对比

| 参数位置 | V1 参数 | V2 参数 | 说明 |
|---------|---------|---------|------|
| 1-12 | (同 allto_allv) | (同 allto_allv) | 相同，除无 permuteOutFlag |
| **新增** | - | **`commMode`** | **V2 新增，V1 需补充** |
| 13 | `y` | `y` | 相同（注意名称差异） |
| 14 | `mmYOptional` | `mmYOptional` | 相同 |
| 15 | `workspaceSize` | `workspaceSize` | 相同 |
| 16 | `executor` | `executor` | 相同 |

---

## 2. 架构设计方案

### 2.1 整体架构

```
V1 算子架构改造
  ├── op_api (aclnn 接口)
  │   ├── 判断通信方式
  │   ├── 转换为 commMode 字符串
  │   └── 调用 V2 aclnn 接口
  │
  ├── op_graph (图模式)
  │   ├── fallback 通路
  │   │   └── 保持原有实现（调用 V1 aclnn）
  │   │
  │   └── gentask 通路
  │       ├── 判断通信方式
  │       ├── 设置 commMode 属性
  │       └── 转发到 V2 GenTask 回调
  │
  └── op_kernel (Kernel)
      └── 保持原有实现（暂不修改）
```

---

### 2.2 aclnn 接口改造方案

#### 方案选择

**推荐方案**: V1 调用 V2 的 aclnn 接口

**理由**:
1. aclnn 接口是对外暴露的标准接口，调用关系清晰
2. V2 的 inner 接口可能不对外暴露，不建议直接调用
3. 通过参数传递 commMode，职责划分清晰

#### GetWorkspaceSize 改造

**改造内容**:

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight,
    const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional,
    const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    const char* group, int64_t epWorldSize,
    const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, bool permuteOutFlag,
    aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 1. 参数校验（保持原有逻辑）
    auto ret_param = CheckParams(...);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(...);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);
    
    // 2. 判断通信方式（新增）
    uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();
    const char* commMode = (commModeInt == Mc2Comm::COMM_MODE_AICPU) ? "ai_cpu" : "";
    
    // 3. 调用 V2 aclnn 接口（新增）
    aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional,
        mmXOptional, mmWeightOptional, group, epWorldSize, sendCounts, recvCounts,
        transGmmWeight, transMmWeight, permuteOutFlag,
        commMode,  // 新增参数
        gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);
    
    return ret;
}
```

**关键改动**:
1. **删除**: 原有的 `aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize` 调用
2. **新增**: 通信方式判断逻辑
3. **新增**: commMode 字符串转换
4. **新增**: 调用 V2 aclnn 接口

#### Execute 改造

**改造内容**:

```cpp
aclnnStatus aclnnAlltoAllvGroupedMatMul(void* workspace, uint64_t workspaceSize,
                                        aclOpExecutor* executor, aclrtStream stream)
{
    // 1. 删除原有的通信方式判断逻辑（简化）
    // 原有代码：
    // if (NnopbaseSetHcclServerType) {
    //     if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
    //         uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    //         ...
    //     }
    // }
    
    // 2. 直接调用 V2 Execute（新增）
    aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2(workspace, workspaceSize, executor, stream);
    
    return ret;
}
```

**关键改动**:
1. **删除**: 原有的通信方式判断和 `NnopbaseSetHcclServerType` 调用
2. **删除**: 原有的 `aclnnInnerAlltoAllvGroupedMatMul` 调用
3. **新增**: 直接调用 V2 Execute

**说明**:
- V2 的 Execute 内部已经处理通信方式设置（从 executor 获取 commMode）
- V1 不需要重复判断

---

### 2.3 fallback 通路方案

#### 方案选择

**推荐方案**: 保持 V1 fallback 不变

**理由**:
1. fallback 调用 V1 aclnn，V1 aclnn 内部已调用 V2，间接使用了 V2 实现
2. 改动最小，风险最低
3. fallback 通路主要用于不支持 gen_task 的场景，改动必要性不大

#### 改造内容

**保持原有实现**:

```cpp
// fallback_allto_allv_grouped_mat_mul.cpp
static ge::graphStatus AlltoAllvGroupedMatMulExecuteFunc(gert::OpExecuteContext* host_api_ctx) {
    // 保持原有逻辑
    ...
    const auto apiRet = EXEC_OPAPI_CMD(aclnnAlltoAllvGroupedMatMul,
                                       gmmX, gmmWeight, ..., gmmY, mmY, permuteOut);
    ...
}
```

**无需修改**:
- V1 aclnn 已经调用 V2，fallback 间接使用 V2 实现
- fallback 无通信方式判断，依赖 aclnn 内部判断

---

### 2.4 gentask 通路方案

#### 方案选择

**推荐方案**: V1 gentask 设置 commMode 属性并调用 V2 GenTask

**理由**:
1. V1 需要在 gentask 中判断通信方式，转换为 commMode 属性
2. V2 GenTask 已有完整的判断逻辑，V1 转发即可
3. 保持 gentask 和 aclnn 的判断逻辑一致性

#### CalcParamFunc 改造

**改造内容**:

```cpp
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context)
{
    // 1. 判断通信方式（保持原有逻辑）
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    
    const char* serverType = nullptr;
    const char* streamType = nullptr;
    
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "ENV_MC2_COMM_MODE_AICPU set, force AICPU GenTask");
    } else if (isArch35) {
        serverType = "ccu server";
        streamType = "ccu_stream";
        OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    } else {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "arch22, use AICPU GenTask");
    }
    
    // 2. 设置 commMode 属性（新增）
    // 方案 A: 通过 context->SetAttr 设置（需要确认 API）
    // 方案 B: 不设置，V2 GenTask 通过环境变量判断（不推荐）
    
    // 当前选择: 不设置额外属性，保持环境变量判断
    // V2 GenTask 会通过 attrs 获取 commMode，如果为空则使用默认逻辑
    
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}
```

**关键改动**:
- **保持**: 原有的通信方式判断逻辑
- **保持**: 原有的 CalcParamFunc 调用

**说明**:
- V1 GenTask 的 CalcParamFunc 已有判断逻辑，生成正确的 serverType/streamType
- V2 GenTask 的 CalcParamFunc 会根据 commMode 属性判断，如果没有 commMode 属性，则使用默认逻辑
- **方案选择**: 保持 V1 CalcParamFunc 不变，依赖 serverType/streamType 的传递

#### GenTaskFunc 改造

**方案 A（推荐）**: 调用 V2 的 GenTask 回调

**改造内容**:

```cpp
ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    // 1. 判断通信方式（保持原有逻辑）
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        OPS_LOG_D(context->GetNodeName(), "arch22, always use AICPU GenTask");
        // 调用 V2 的 AICPU GenTask 回调
        return AlltoAllvGroupedMatMulV2GenTaskFunc(context, tasks);
    }
    
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        OPS_LOG_D(context->GetNodeName(), "arch35 with ENV_MC2_COMM_MODE_AICPU, use AICPU GenTask");
        // 调用 V2 的 AICPU GenTask 回调
        return AlltoAllvGroupedMatMulV2GenTaskFunc(context, tasks);
    }
    
    OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    // 调用 V2 的 GenTask 回调
    return AlltoAllvGroupedMatMulV2GenTaskFunc(context, tasks);
}
```

**关键改动**:
- **删除**: 原有的 `Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback` 和 `Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack` 调用
- **新增**: 调用 V2 的 `AlltoAllvGroupedMatMulV2GenTaskFunc`

**问题**:
- V2 的 `AlltoAllvGroupedMatMulV2GenTaskFunc` 是 V2 的内部函数，是否可以直接调用？
- 需要确认 V2 GenTask 函数的可见性

**方案 B**: 设置 commMode 属性，让 V2 GenTask 自动判断

**改造内容**:

```cpp
ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    // 1. 判断通信方式并设置属性（新增）
    uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    
    // 设置 commMode 属性
    const char* commMode = nullptr;
    if (commModeInt == Mc2Comm::COMM_MODE_AICPU) {
        commMode = "ai_cpu";
    } else if (isArch35) {
        commMode = "ccu";  // 或 ""
    } else {
        commMode = "";  // arch22 默认
    }
    
    // 如何设置属性？需要确认 API
    // context->SetAttr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE), commMode);
    
    // 2. 调用 V2 GenTask 回调（新增）
    return AlltoAllvGroupedMatMulV2GenTaskFunc(context, tasks);
}
```

**关键改动**:
- **新增**: 判断通信方式并转换为字符串
- **新增**: 设置 commMode 属性（需要确认 API）
- **新增**: 调用 V2 GenTask 回调

**问题**:
- `gert::ExeResGenerationContext` 是否有 SetAttr 方法？
- V2 GenTask 函数是否可见？

**方案 C（当前推荐）**: 保持原有实现

**理由**:
1. V1 GenTask 的判断逻辑已完整，生成正确的 GenTask 回调
2. V2 GenTask 通过 attrs 获取 commMode，如果没有则使用默认逻辑或环境变量
3. 避免复杂的属性设置 API 问题

**改造内容**:

```cpp
ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    // 保持原有逻辑
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        OPS_LOG_D(context->GetNodeName(), "arch22, always use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        OPS_LOG_D(context->GetNodeName(), "arch35 with ENV_MC2_COMM_MODE_AICPU, use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    
    OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
}
```

**说明**:
- V1 GenTask 保持原有实现，不调用 V2 GenTask
- V1 和 V2 GenTask 共用相同的 GenTask 回调函数
- 两者判断逻辑相同，无需额外协调

---

### 2.5 实现路线图

#### 第一阶段: aclnn 接口改造

**优先级**: 高

**修改文件**:

| 文件 | 修改内容 |
|------|----------|
| `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul.cpp` | GetWorkspaceSize 调用 V2；Execute 删除判断逻辑 |
| `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul.h` | 无需修改 |
| `mc2/grouped_mat_mul_allto_allv/op_api/aclnn_grouped_mat_mul_allto_allv.cpp` | GetWorkspaceSize 调用 V2；Execute 删除判断逻辑 |
| `mc2/grouped_mat_mul_allto_allv/op_api/aclnn_grouped_mat_mul_allto_allv.h` | 无需修改 |

**修改要点**:

1. **aclnn_allto_allv_grouped_mat_mul.cpp**:
   ```cpp
   // GetWorkspaceSize 改造
   aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(...) {
       // 1. 参数校验（保持）
       ...
       
       // 2. 判断通信方式（新增）
       uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();
       const char* commMode = (commModeInt == Mc2Comm::COMM_MODE_AICPU) ? "ai_cpu" : "";
       
       // 3. 调用 V2 aclnn（替换 inner 调用）
       return aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
           ..., commMode, ...);
   }
   
   // Execute 改造
   aclnnStatus aclnnAlltoAllvGroupedMatMul(...) {
       // 删除原有判断逻辑，直接调用 V2 Execute
       return aclnnAlltoAllvGroupedMatMulV2(workspace, workspaceSize, executor, stream);
   }
   ```

2. **aclnn_grouped_mat_mul_allto_allv.cpp**:
   ```cpp
   // 类似改造
   aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(...) {
       ...
       uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();
       const char* commMode = (commModeInt == Mc2Comm::COMM_MODE_AICPU) ? "ai_cpu" : "";
       
       return aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(
           ..., commMode, ...);
   }
   
   aclnnStatus aclnnGroupedMatMulAlltoAllv(...) {
       return aclnnGroupedMatMulAlltoAllvV2(workspace, workspaceSize, executor, stream);
   }
   ```

**注意事项**:
1. 需要添加 V2 aclnn 头文件引用
2. commMode 参数位置：
   - allto_allv_grouped_mat_mul: 在 permuteOutFlag 之后
   - grouped_mat_mul_allto_allv: 在 transMmWeight 之后
3. permuteOutOptional 参数传递：
   - 根据 permuteOutFlag 判断是否为 nullptr

---

#### 第二阶段: gentask 通路改造（可选）

**优先级**: 低（方案 C 推荐保持不变）

**修改文件**:

| 文件 | 修改内容 |
|------|----------|
| `mc2/allto_allv_grouped_mat_mul/op_graph/allto_allv_grouped_mat_mul_gen_task_training.cpp` | 保持原有实现（方案 C） |
| `mc2/grouped_mat_mul_allto_allv/op_graph/grouped_mat_mul_allto_allv_gen_task_training.cpp` | 保持原有实现（方案 C） |

**说明**:
- V1 GenTask 保持原有判断逻辑
- V1 和 V2 共用相同的 GenTask 回调函数
- 无需额外改造

---

#### 第三阶段: fallback 通路改造（无需）

**优先级**: 无

**修改文件**: 无

**说明**:
- fallback 保持原有实现
- fallback 调用 V1 aclnn，间接使用 V2 实现

---

#### 第四阶段: 测试验证

**优先级**: 高

**测试内容**:

1. **功能测试**:
   - V1 aclnn 接口调用
   - 环境变量设置（ENV_MC2_COMM_MODE_AICPU）
   - 不同架构（arch22、arch35）

2. **兼容性测试**:
   - V1 接口向后兼容
   - 参数传递正确性
   - 输出一致性

3. **性能测试**:
   - V1 vs V2 性能对比
   - 通信方式切换性能

---

### 2.6 关键代码位置说明

#### V1 aclnn 改造位置

**文件**: `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul.cpp`

**修改位置**:

| 行号范围 | 函数 | 改造内容 |
|---------|------|----------|
| 109-130 | `aclnnAlltoAllvGroupedMatMulGetWorkspaceSize` | 判断通信方式；调用 V2 aclnn |
| 132-149 | `aclnnAlltoAllvGroupedMatMul` | 删除判断逻辑；调用 V2 Execute |

**新增依赖**:

```cpp
#include "aclnn_allto_allv_grouped_mat_mul_v2.h"  // V2 aclnn 头文件
```

#### V1 GenTask 改造位置（可选）

**文件**: `mc2/allto_allv_grouped_mat_mul/op_graph/allto_allv_grouped_mat_mul_gen_task_training.cpp`

**修改位置**:

| 行号范围 | 函数 | 改造内容 |
|---------|------|----------|
| 26-48 | `AlltoAllvGroupedMatMulCalcParamFunc` | 保持原有逻辑（方案 C） |
| 50-65 | `AlltoAllvGroupedMatMulGenTaskFunc` | 保持原有逻辑（方案 C） |

---

### 2.7 实现细节

#### V1 如何获取 commMode 参数

**方法**: 环境变量判断 + 字符串转换

```cpp
// 判断通信方式
uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();

// 转换为字符串
const char* commMode = nullptr;
if (commModeInt == Mc2Comm::COMM_MODE_AICPU) {
    commMode = "ai_cpu";  // AICPU 模式
} else {
    commMode = "";  // 默认模式（arch22: AICPU; arch35: CCU）
}
```

**说明**:
- 环境变量 `ENV_MC2_COMM_MODE_AICPU` 存在 → "ai_cpu"
- 环境变量不存在 → ""（空字符串，默认模式）

#### V1 如何调用 V2

**调用方式**: 直接调用 V2 aclnn 接口

```cpp
// GetWorkspaceSize
aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    gmmX, gmmWeight,
    sendCountsTensorOptional, recvCountsTensorOptional,
    mmXOptional, mmWeightOptional,
    group, epWorldSize,
    sendCounts, recvCounts,
    transGmmWeight, transMmWeight, permuteOutFlag,
    commMode,  // 新增参数
    gmmY, mmYOptional, permuteOutOptional,
    workspaceSize, executor);

// Execute
aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2(
    workspace, workspaceSize, executor, stream);
```

**说明**:
- V1 参数与 V2 参数基本一致，直接传递
- 新增 commMode 参数
- 新增 permuteOutOptional 参数（根据 permuteOutFlag）

#### 参数映射实现

**allto_allv_grouped_mat_mul**:

```cpp
// V1 aclnn 调用 V2 aclnn
aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    gmmX,                 // 1: 相同
    gmmWeight,            // 2: 相同
    sendCountsTensorOptional,  // 3: 相同
    recvCountsTensorOptional,  // 4: 相同
    mmXOptional,          // 5: 相同
    mmWeightOptional,     // 6: 相同
    group,                // 7: 相同
    epWorldSize,          // 8: 相同
    sendCounts,           // 9: 相同
    recvCounts,           // 10: 相同
    transGmmWeight,       // 11: 相同
    transMmWeight,        // 12: 相同
    permuteOutFlag,       // 13: 相同
    commMode,             // 14: 新增
    gmmY,                 // 15: 相同
    mmYOptional,          // 16: 相同
    permuteOutOptional,   // 17: 新增（根据 permuteOutFlag）
    workspaceSize,        // 18: 相同
    executor              // 19: 相同
);
```

**grouped_mat_mul_allto_allv**:

```cpp
// V1 aclnn 调用 V2 aclnn
aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(
    gmmX,                 // 1: 相同
    gmmWeight,            // 2: 相同
    sendCountsTensorOptional,  // 3: 相同
    recvCountsTensorOptional,  // 4: 相同
    mmXOptional,          // 5: 相同
    mmWeightOptional,     // 6: 相同
    group,                // 7: 相同
    epWorldSize,          // 8: 相同
    sendCounts,           // 9: 相同
    recvCounts,           // 10: 相同
    transGmmWeight,       // 11: 相同
    transMmWeight,        // 12: 相同
    commMode,             // 13: 新增（注意位置）
    y,                    // 14: 相同
    mmYOptional,          // 15: 相同
    workspaceSize,        // 16: 相同
    executor              // 17: 相同
);
```

#### 错误处理机制

**参数校验**:

```cpp
// V1 GetWorkspaceSize
auto ret_param = CheckParams(...);
CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

auto ret_send_and_recv = CheckSendAndRecv(...);
CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

// 调用 V2 前无需额外校验，V2 内部会校验 commMode
```

**调用失败处理**:

```cpp
// V1 GetWorkspaceSize
aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(...);
if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER, "Call V2 aclnn failed: %d", ret);
    return ret;
}
return ret;
```

---

## 3. 风险和注意事项

### 3.1 可能的问题

#### 问题 1: V2 aclnn 头文件可见性

**问题描述**:
- V1 和 V2 在不同目录，V1 是否能引用 V2 的头文件？
- 编译系统是否支持跨算子引用？

**解决方案**:
- 确认 CMakeLists.txt 或编译配置是否支持跨算子引用
- 如不支持，需要修改编译配置或调整目录结构

#### 问题 2: inner 接口调用

**问题描述**:
- V1 原有调用 `aclnnInner_xxx`，改为调用 V2 aclnn 后，inner 接口是否还需要？
- inner 接口是否对外暴露？

**解决方案**:
- V1 不再调用 inner 接口，改为调用 V2 aclnn
- V1 的 inner 接口保持存在，用于 fallback 或其他场景

#### 问题 3: 参数顺序差异

**问题描述**:
- V2 新增 commMode 参数，位置固定
- V1 调用 V2 时，参数顺序必须正确

**解决方案**:
- 严格按照 V2 参数顺序传递
- commMode 参数位置：
  - allto_allv_grouped_mat_mul: permuteOutFlag 之后
  - grouped_mat_mul_allto_allv: transMmWeight 之后

#### 问题 4: 输出参数差异

**问题描述**:
- allto_allv_grouped_mat_mul V2 新增 permuteOutOptional 输出
- V1 如何处理该参数？

**解决方案**:
- V1 已有 permuteOutOptional 输出参数（在接口中）
- 根据 permuteOutFlag 判断是否为 nullptr
- 直接传递给 V2

---

### 3.2 兼容性考虑

#### 向后兼容

**接口兼容**:
- V1 aclnn 接口签名保持不变
- 用户调用 V1 接口无需修改
- V1 内部调用 V2，用户无感知

**行为兼容**:
- V1 判断逻辑保持不变（环境变量判断）
- V1 将判断结果转换为 commMode 传递给 V2
- V2 执行逻辑与 V1 原有逻辑一致

#### 性能兼容

**性能影响**:
- V1 调用 V2，增加一层函数调用
- 函数调用开销极小（可忽略）
- V2 实现可能更优化，性能可能提升

---

### 3.3 测试建议

#### 单元测试

**测试内容**:

1. **参数传递测试**:
   - V1 参数传递给 V2 是否正确
   - commMode 参数生成是否正确
   - 输出参数传递是否正确

2. **环境变量测试**:
   - 设置 ENV_MC2_COMM_MODE_AICPU，V1 判断是否正确
   - 不设置环境变量，V1 判断是否正确

3. **架构判断测试**:
   - arch22 平台，V1 判断是否正确
   - arch35 平台，V1 判断是否正确

#### 集成测试

**测试内容**:

1. **fallback 通路测试**:
   - fallback 调用 V1 aclnn 是否正常
   - 输出结果是否正确

2. **gentask 通路测试**:
   - GenTask 判断是否正确
   - GenTask 回调是否正常

3. **性能对比测试**:
   - V1 vs V2 性能对比
   - 不同通信方式性能对比

#### ST 测试

**测试内容**:
- 完整算子功能测试
- 不同数据类型测试
- 不同 shape 测试
- 不同场景测试

---

## 4. 附录

### 4.1 相关文件列表

#### V1 算子

**allto_allv_grouped_mat_mul**:
- `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul.h`
- `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul.cpp`
- `mc2/allto_allv_grouped_mat_mul/op_graph/allto_allv_grouped_mat_mul_gen_task_training.cpp`
- `mc2/allto_allv_grouped_mat_mul/op_graph/fallback_allto_allv_grouped_mat_mul.cpp`

**grouped_mat_mul_allto_allv**:
- `mc2/grouped_mat_mul_allto_allv/op_api/aclnn_grouped_mat_mul_allto_allv.h`
- `mc2/grouped_mat_mul_allto_allv/op_api/aclnn_grouped_mat_mul_allto_allv.cpp`
- `mc2/grouped_mat_mul_allto_allv/op_graph/grouped_mat_mul_allto_allv_gen_task_training.cpp`
- `mc2/grouped_mat_mul_allto_allv/op_graph/fallback_grouped_mat_mul_allto_allv.cpp`

#### V2 算子

**allto_allv_grouped_mat_mul_v2**:
- `mc2/allto_allv_grouped_mat_mul_v2/op_api/aclnn_allto_allv_grouped_mat_mul_v2.h`
- `mc2/allto_allv_grouped_mat_mul_v2/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp`
- `mc2/allto_allv_grouped_mat_mul_v2/op_api/aclnnInner_allto_allv_grouped_mat_mul_v2.h`
- `mc2/allto_allv_grouped_mat_mul_v2/op_graph/allto_allv_grouped_mat_mul_v2_gen_task_training.cpp`
- `mc2/allto_allv_grouped_mat_mul_v2/op_graph/fallback_allto_allv_grouped_mat_mul_v2.cpp`

**grouped_mat_mul_allto_allv_v2**:
- `mc2/grouped_mat_mul_allto_allv_v2/op_api/aclnn_grouped_mat_mul_allto_allv_v2.h`
- `mc2/grouped_mat_mul_allto_allv_v2/op_api/aclnn_grouped_mat_mul_allto_allv_v2.cpp`
- `mc2/grouped_mat_mul_allto_allv_v2/op_graph/grouped_mat_mul_allto_allv_v2_gen_task_training.cpp`
- `mc2/grouped_mat_mul_allto_allv_v2/op_graph/fallback_grouped_mat_mul_allto_allv_v2.cpp`

#### 公共文件

- `mc2/common/utils/mc2_comm_utils.h` - 通信方式判断工具

---

### 4.2 参考资料

1. **Ascend C 算子开发指南**
2. **aclnn 接口规范**
3. **GenTask 开发指南**
4. **fallback 开发指南**

---

### 4.3 术语表

| 术语 | 说明 |
|------|------|
| **V1 算子** | 原版算子，无 commMode 参数 |
| **V2 算子** | 新版算子，新增 commMode 参数 |
| **收编** | V1 调用 V2 实现，保持 V1 接口不变 |
| **commMode** | 通信模式参数，指定通信方式 |
| **fallback** | 图模式备选通路，调用 aclnn 接口 |
| **gentask** | 图模式优化通路，生成执行任务 |
| **inner 接口** | 内部实现接口，不对外暴露 |
| **aclnn** | Ascend Compute Library Neural Network 接口 |

---

## 5. 总结

### 5.1 设计要点

1. **aclnn 接口收编**: V1 判断通信方式，转换为 commMode，调用 V2 aclnn
2. **fallback 通路**: 保持原有实现，间接使用 V2
3. **gentask 通路**: 保持原有判断逻辑，共用 GenTask 回调
4. **参数映射**: V1 参数直接传递给 V2，新增 commMode 参数
5. **兼容性**: V1 接口不变，用户无感知

### 5.2 实现优先级

1. **高**: aclnn 接口改造（V1 调用 V2）
2. **低**: gentask 通路改造（保持原有）
3. **无**: fallback 通路改造（无需修改）
4. **高**: 测试验证（功能和兼容性）

### 5.3 风险控制

1. **编译配置**: 确认跨算子引用支持
2. **参数顺序**: 严格按照 V2 参数顺序
3. **输出参数**: 正确处理 permuteOutOptional
4. **测试覆盖**: 单元测试、集成测试、ST 测试

---

## 6. 开发者质疑回应预留

> 本节用于记录 Developer 串讲时的质疑回应，后续补充。

### 6.1 质疑清单

> 待 Developer 串讲后补充

### 6.2 回应记录

> 待串讲后补充