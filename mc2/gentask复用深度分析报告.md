# gentask 复用深度分析报告

## 一、调研对象与目标

### 1.1 调研算子
- **参考算子**：all_gather_matmul (V1) 和 all_gather_matmul_v2 (V2)
- **应用对象**：allto_allv_grouped_mat_mul (V1/V2) 和 grouped_mat_mul_allto_allv (V1/V2)

### 1.2 核心调研内容
重点分析 **gentask** 的复用方式，包括：
1. V1 和 V2 的 gentask 关系（是否调用、共用回调、独立实现）
2. 回调函数的定义和使用方式
3. 参数传递方式（环境变量 vs 属性参数）
4. 架构判断逻辑的实现

---

## 二、V1 和 V2 的 gentask 关系分析

### 2.1 参考算子（all_gather_matmul 系列）

#### V1 的 gentask 实现

**文件**：`mc2/all_gather_matmul/op_graph/all_gather_matmul_gen_task.cpp`

```cpp
// V1: CalcOpParam 函数
static ge::Status AllGatherMatmulCalcOpParam(gert::ExeResGenerationContext *context) {
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

// V1: GenTask 函数
static ge::Status AllGatherMatmulGenTask(const gert::ExeResGenerationContext *context,
                                          std::vector<std::vector<uint8_t>> &tasks) {
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}

// 注册
IMPL_OP(AllGatherMatmul).CalcOpParam(AllGatherMatmulCalcOpParam).GenerateTask(AllGatherMatmulGenTask);
```

**关键特征**：
- ✅ 调用公共回调函数 `CommonKFCMc2CalcParamFunc` 和 `CommonKFCMc2GenTask`
- ❌ **没有架构判断**（没有 `IsTargetPlatformNpuArch`）
- ❌ **没有参数判断**（固定使用 AICPU）
- ❌ **没有环境变量判断**
- **固定使用 AICPU 通信方式**

#### V2 的 gentask 实现

**文件**：`mc2/all_gather_matmul_v2/op_graph/all_gather_matmul_v2_gen_task.cpp`

```cpp
// V2: CalcOpParam 函数
static ge::Status AllGatherMatmulV2CalcOpParam(gert::ExeResGenerationContext *context)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        uint8_t commMode = Mc2Comm::GetCommModeFromEnv();  // 从环境变量读取
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTask CalcOpParam");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
        } else {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask CalcOpParam");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
        }
    }
    OPS_LOG_D(context->GetNodeName(), "AICPU GenTask CalcOpParam");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

// V2: GenTask 函数
static ge::Status AllGatherMatmulV2GenTask(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTaskFunc");
            return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
        } else {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTaskFunc");
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        }
    }
    OPS_LOG_D(context->GetNodeName(), "AICPU GenTaskFunc");
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}

// 注册
IMPL_OP(AllGatherMatmulV2).CalcOpParam(AllGatherMatmulV2CalcOpParam).GenerateTask(AllGatherMatmulV2GenTask);
```

**关键特征**：
- ✅ 调用公共回调函数（`CommonKFCMc2CalcParamFunc`、`CommonKFCMc2GenTask`、`Mc2Arch35GenTaskCallBack`）
- ✅ **有架构判断**（`IsTargetPlatformNpuArch(..., NPUARCH_A5)`）
- ✅ **有参数判断**（从环境变量读取 commMode）
- ✅ **使用环境变量**（`Mc2Comm::GetCommModeFromEnv()`）
- **逻辑流程**：
  - arch35 + AICPU环境变量 → AICPU
  - arch35 + CCU（默认） → CCU
  - arch22 → AICPU

#### V1 和 V2 的关系结论

| 维度 | all_gather_matmul V1 | all_gather_matmul_v2 V2 | 关系 |
|------|---------------------|------------------------|------|
| **调用关系** | 独立实现 | 独立实现 | **没有调用关系** |
| **回调函数** | 使用公共回调 | 使用公共回调 | **共用公共回调** |
| **架构判断** | 无 | 有 | **V2 新增架构判断** |
| **参数传递** | 固定 AICPU | 环境变量 | **V2 新增参数控制** |
| **代码复用** | 无 | 无 | **完全独立** |

**核心发现**：
- ❌ **V1 不调用 V2 的 gentask**
- ❌ **V2 不调用 V1 的 gentask**
- ✅ **V1 和 V2 使用相同的公共回调函数**（定义在 `mc2/common/op_graph/` 目录）
- ✅ **V2 是 V1 的增强版本，新增了架构判断和环境变量判断**

---

### 2.2 当前算子（allto_allv_grouped_mat_mul 系列）

#### V1 的 gentask 实现

**文件**：`mc2/allto_allv_grouped_mat_mul/op_graph/allto_allv_grouped_mat_mul_gen_task_training.cpp`

```cpp
// V1: CalcParam 函数
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context)
{
    // 根据架构判断通信方式，不再依赖环境变量
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);

    const char* serverType = nullptr;
    const char* streamType = nullptr;

    if (isArch35) {
        serverType = "ccu server";
        streamType = "ccu_stream";
        OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    } else {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "arch22, use AICPU GenTask");
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

// V1: GenTask 函数
ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    // 根据架构判断通信方式，不再依赖环境变量
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        OPS_LOG_D(context->GetNodeName(), "arch22, always use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    // arch35 默认使用 CCU
    OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
}

// 注册
IMPL_OP(AlltoAllvGroupedMatMul)
    .CalcOpParam(AlltoAllvGroupedMatMulCalcParamFunc)
    .GenerateTask(AlltoAllvGroupedMatMulGenTaskFunc);
```

**关键特征**：
- ✅ 调用公共回调函数（`CommonKFCMc2CalcParamFunc`、`Mc2MoeGenTaskCallback`、`Mc2Arch35GenTaskCallBack`）
- ✅ **有架构判断**（`IsTargetPlatformNpuArch(..., NPUARCH_A5)`）
- ❌ **没有参数判断**（根据架构固定选择）
- ❌ **不使用环境变量**（注释明确说"不再依赖环境变量"）
- **逻辑流程**：
  - arch35 → CCU（固定）
  - arch22 → AICPU（固定）

#### V2 的 gentask 实现

**文件**：`mc2/allto_allv_grouped_mat_mul_v2/op_graph/allto_allv_grouped_mat_mul_v2_gen_task_training.cpp`

```cpp
// V2: CalcParam 函数
ge::Status AlltoAllvGroupedMatMulV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_D(context->GetNodeName(), "attrs is null, use default AICPU GenTask"),
               return ge::GRAPH_FAILED);
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);

    const char* serverType = nullptr;
    const char* streamType = nullptr;

    if (commMode == nullptr || strlen(commMode) == 0) {
        if (isArch35) {
            serverType = "ccu server";
            streamType = "ccu_stream";
            OPS_LOG_D(context->GetNodeName(), "comm_mode is empty, arch35 use CCU GenTask");
        } else {
            serverType = "aicpu kfc server";
            streamType = "kfc_stream";
            OPS_LOG_D(context->GetNodeName(), "comm_mode is empty, arch22 use AICPU GenTask");
        }
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "comm_mode is ai_cpu, use AICPU GenTask");
    } else if (strcmp(commMode, "ccu") == 0) {
        serverType = "ccu server";
        streamType = "ccu_stream";
        OPS_LOG_D(context->GetNodeName(), "comm_mode is ccu, use CCU GenTask");
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid comm_mode: %s", commMode);
        return ge::GRAPH_FAILED;
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

// V2: GenTask 函数
ge::Status AlltoAllvGroupedMatMulV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        OPS_LOG_D(context->GetNodeName(), "arch22, always use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_D(context->GetNodeName(), "attrs is null, use CCU GenTask"),
               return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks));
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    
    if (commMode == nullptr || strlen(commMode) == 0 || strcmp(commMode, "ccu") == 0) {
        OPS_LOG_D(context->GetNodeName(), "comm_mode is %s, arch35 use CCU GenTask", 
                  commMode == nullptr || strlen(commMode) == 0 ? "empty" : commMode);
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        OPS_LOG_D(context->GetNodeName(), "comm_mode is ai_cpu, arch35 use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid comm_mode: %s", commMode);
        return ge::GRAPH_FAILED;
    }
}

// 注册
IMPL_OP(AlltoAllvGroupedMatMulV2)
    .CalcOpParam(AlltoAllvGroupedMatMulV2CalcParamFunc)
    .GenerateTask(AlltoAllvGroupedMatMulV2GenTaskFunc);
```

**关键特征**：
- ✅ 调用公共回调函数（`CommonKFCMc2CalcParamFunc`、`Mc2MoeGenTaskCallback`、`Mc2Arch35GenTaskCallBack`）
- ✅ **有架构判断**（`IsTargetPlatformNpuArch(..., NPUARCH_A5)`）
- ✅ **有参数判断**（从属性参数读取 commMode）
- ✅ **使用属性参数**（`attrs->GetStr(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE)`）
- **逻辑流程**：
  - arch22 → AICPU（固定）
  - arch35 + "" 或 "ccu" → CCU
  - arch35 + "ai_cpu" → AICPU

#### V1 和 V2 的关系结论

| 维度 | allto_allv_grouped_mat_mul V1 | allto_allv_grouped_mat_mul_v2 V2 | 关系 |
|------|------------------------------|--------------------------------|------|
| **调用关系** | 独立实现 | 独立实现 | **没有调用关系** |
| **回调函数** | 使用公共回调 | 使用公共回调 | **共用公共回调** |
| **架构判断** | 有（固定选择） | 有（参数控制） | **V2 增强判断** |
| **参数传递** | 根据架构固定 | 属性参数 commMode | **V2 新增参数控制** |
| **代码复用** | 无 | 无 | **完全独立** |

**核心发现**：
- ❌ **V1 不调用 V2 的 gentask**
- ❌ **V2 不调用 V1 的 gentask**
- ✅ **V1 和 V2 使用相同的公共回调函数**（定义在 `mc2/common/op_graph/` 目录）
- ✅ **V2 是 V1 的增强版本，新增了属性参数控制**

---

## 三、回调函数的定义和使用分析

### 3.1 公共回调函数的定义位置

**关键文件**：
- `mc2/common/op_graph/mc2_gen_task_ops_utils.h` — 基础 GenTask 工具类
- `mc2/common/op_graph/mc2_gen_task_ops_utils_arch35.h` — arch35 CCU GenTask 工具类
- `mc2/common/op_graph/mc2_moe_gen_task_ops_utils.h` — MoE 系列算子 GenTask 工具类

### 3.2 公共回调函数的定义

#### mc2_gen_task_ops_utils.h

```cpp
namespace ops {
class Mc2GenTaskOpsUtils {
public:
    // 基础的 CalcParam 函数
    static ge::Status CommonKFCMc2CalcParamFunc(const gert::ExeResGenerationContext *context, 
                                                 const ge::AscendString &name, 
                                                 const ge::AscendString &reuse_key);
    
    // 基础的 GenTask 函数（AICPU）
    static ge::Status CommonKFCMc2GenTask(const gert::ExeResGenerationContext *context,
                                           std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops
```

**用途**：
- `CommonKFCMc2CalcParamFunc`: 通用的 CalcParam 实现，接受 serverType 和 streamType 参数
- `CommonKFCMc2GenTask`: 通用的 AICPU GenTask 实现

#### mc2_gen_task_ops_utils_arch35.h

```cpp
namespace ops {
class Mc2Arch35GenTaskOpsUtils {
public:
    // arch35 CCU GenTask 回调函数
    static ge::Status Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context,
                                                std::vector<std::vector<uint8_t>> &tasks)
    {
        if (context == nullptr) {
            OPS_LOG_E("Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack", "Failed to get context.");
            return ge::GRAPH_FAILED;
        }
        
        if (tasks.size() == 0U) {
            OPS_LOG_E(context->GetNodeName(), "Empty task vector when generating task.");
            return ge::GRAPH_FAILED;
        }

        if (CreateCCUFusionTask(context, tasks) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(context->GetNodeName(), "Failed to create CCU fusion task.");
            return ge::GRAPH_FAILED;
        }

        OPS_LOG_I(context->GetNodeName(), "Gen Task for CCU fusion task success.");
        return ge::GRAPH_SUCCESS;
    }

    static ge::Status CreateCCUFusionTask(const gert::ExeResGenerationContext *context,
                                          std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops
```

**用途**：
- `Mc2Arch35GenTaskCallBack`: arch35 平台的 CCU GenTask 实现（内联定义）
- `CreateCCUFusionTask`: 创建 CCU 融合任务的实际实现

#### mc2_moe_gen_task_ops_utils.h

```cpp
namespace ops {
class Mc2MoeGenTaskOpsUtils {
public:
    // MoE 系列算子的 GenTask 回调函数
    static ge::Status Mc2MoeGenTaskCallback(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status Mc2MoeGenTaskCallbackV2(const gert::ExeResGenerationContext *context,
                                               std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops
```

**用途**：
- `Mc2MoeGenTaskCallback`: MoE 系列算子的 AICPU GenTask 实现
- `Mc2MoeGenTaskCallbackV2`: V2 版本的 MoE GenTask 实现

### 3.3 回调函数的使用对比

| 回调函数 | all_gather_matmul V1 | all_gather_matmul_v2 V2 | allto_allv_grouped_mat_mul V1 | allto_allv_grouped_mat_mul_v2 V2 | 定义位置 |
|---------|---------------------|------------------------|------------------------------|--------------------------------|---------|
| `CommonKFCMc2CalcParamFunc` | ✅ | ✅ | ✅ | ✅ | `mc2/common/op_graph/mc2_gen_task_ops_utils.h` |
| `CommonKFCMc2GenTask` | ✅ | ✅（arch35+AICPU） | ❌ | ❌ | `mc2/common/op_graph/mc2_gen_task_ops_utils.h` |
| `Mc2Arch35GenTaskCallBack` | ❌ | ✅（arch35+CCU） | ✅（arch35） | ✅（arch35+CCU） | `mc2/common/op_graph/mc2_gen_task_ops_utils_arch35.h` |
| `Mc2MoeGenTaskCallback` | ❌ | ❌ | ✅（arch22） | ✅（arch22 / arch35+AICPU） | `mc2/common/op_graph/mc2_moe_gen_task_ops_utils.h` |

**核心发现**：
- ✅ **所有回调函数都是跨算子共用的**，定义在公共目录
- ✅ **回调函数的参数签名相同**，可以直接调用
- ✅ **当前算子已经使用了正确的公共回调函数**

---

## 四、参数传递方式对比

### 4.1 环境变量方式（all_gather_matmul_v2）

**定义文件**：`mc2/common/utils/mc2_comm_utils.h`

```cpp
namespace Mc2Comm {
constexpr char COMM_MODE_ENV[] = "ENV_MC2_COMM_MODE_AICPU";

constexpr uint8_t COMM_MODE_CCU = 0;
constexpr uint8_t COMM_MODE_AICPU = 1;

inline uint8_t GetCommModeFromEnv()
{
    auto commModeEnv = getenv(COMM_MODE_ENV);
    if (commModeEnv != nullptr) {
        return COMM_MODE_AICPU;
    }
    return COMM_MODE_CCU;
}
}
```

**使用方式**：

```cpp
// 在 gentask 中使用
uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
if (commMode == Mc2Comm::COMM_MODE_AICPU) {
    // 使用 AICPU
} else {
    // 使用 CCU（默认）
}
```

**特点**：
- ✅ 全局控制（所有算子统一行为）
- ❌ 无法针对单个算子定制
- ❌ 需要设置环境变量（运维负担）
- ✅ 简单易用（只需设置一个环境变量）

### 4.2 属性参数方式（allto_allv_grouped_mat_mul_v2）

**Proto 定义**：`mc2/allto_allv_grouped_mat_mul_v2/op_graph/allto_allv_grouped_mat_mul_v2_proto.h`

```cpp
REG_OP(AlltoAllvGroupedMatMulV2)
    ...
    .ATTR(comm_mode, String, "")  // 默认为空
    .OP_END_FACTORY_REG(AlltoAllvGroupedMatMulV2)
```

**使用方式**：

```cpp
// 在 gentask 中使用
auto attrs = context->GetAttrs();
const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));

if (commMode == nullptr || strlen(commMode) == 0) {
    // 默认行为（根据架构判断）
} else if (strcmp(commMode, "ai_cpu") == 0) {
    // 使用 AICPU
} else if (strcmp(commMode, "ccu") == 0) {
    // 使用 CCU
}
```

**特点**：
- ✅ 针对单个算子定制（精确控制）
- ✅ 不需要设置环境变量
- ✅ 可以在模型层面控制（通过模型参数）
- ❌ 需要修改算子 Proto 定义
- ✅ 更灵活和可控

### 4.3 参数传递方式对比总结

| 维度 | 环境变量方式（all_gather_matmul_v2） | 属性参数方式（allto_allv_grouped_mat_mul_v2） | 推荐度 |
|------|-------------------------------------|------------------------------------------|--------|
| **灵活性** | 低（全局控制） | 高（算子级控制） | **属性参数更好** ✅ |
| **易用性** | 高（一个环境变量） | 中（需要修改 Proto） | **环境变量更简单** |
| **可控性** | 低（无法定制） | 高（精确控制） | **属性参数更好** ✅ |
| **运维负担** | 高（需要设置环境变量） | 低（通过模型参数） | **属性参数更好** ✅ |
| **适用场景** | 统一控制所有算子 | 针对单个算子定制 | **两者互补** |

**核心发现**：
- ✅ **属性参数方式比环境变量方式更先进和灵活**
- ✅ **当前算子已经使用了属性参数方式**，比参考算子更先进
- ❌ **不建议当前算子改用环境变量方式**

---

## 五、架构判断逻辑对比

### 5.1 架构判断的实现

**所有算子统一使用**：

```cpp
bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
```

**判断逻辑**：
- `NPUARCH_A5` 对应 arch35（Atlas A3 / Ascend910B）
- 非 `NPUARCH_A5` 对应 arch22（Atlas A2 / Ascend910）

### 5.2 架构判断的使用位置

| 算子 | CalcOpParam 中架构判断 | GenTask 中架构判断 | 判断逻辑 |
|------|----------------------|------------------|---------|
| all_gather_matmul V1 | ❌ 无 | ❌ 无 | **没有架构判断** |
| all_gather_matmul_v2 V2 | ✅ 有 | ✅ 有 | arch35 → 参数判断；arch22 → AICPU |
| allto_allv_grouped_mat_mul V1 | ✅ 有 | ✅ 有 | arch35 → CCU；arch22 → AICPU |
| allto_allv_grouped_mat_mul_v2 V2 | ✅ 有 | ✅ 有 | arch35 → 参数判断；arch22 → AICPU |

**核心发现**：
- ✅ **当前算子的架构判断逻辑已经完整**
- ✅ **与参考算子的架构判断逻辑一致**

---

## 六、当前算子的 gentask 现状分析

### 6.1 当前 V1 和 V2 的 gentask 关系

| 维度 | allto_allv_grouped_mat_mul V1 | allto_allv_grouped_mat_mul_v2 V2 | grouped_mat_mul_allto_allv V1 | grouped_mat_mul_allto_allv_v2 V2 |
|------|------------------------------|--------------------------------|------------------------------|--------------------------------|
| **调用关系** | ❌ 独立 | ❌ 独立 | ❌ 独立 | ❌ 独立 |
| **回调函数** | ✅ 共用公共回调 | ✅ 共用公共回调 | ✅ 共用公共回调 | ✅ 共用公共回调 |
| **架构判断** | ✅ 有 | ✅ 有 | ✅ 有 | ✅ 有 |
| **参数传递** | ❌ 固定选择 | ✅ 属性参数 | ❌ 固定选择 | ✅ 属性参数 |

### 6.2 与参考算子的对比

| 维度 | all_gather_matmul 系列 | 当前算子系列 | 当前算子的优势 |
|------|----------------------|-------------|--------------|
| **V1 的架构判断** | ❌ 无 | ✅ 有 | **当前算子更先进** ✅ |
| **V2 的参数传递** | ❌ 环境变量 | ✅ 属性参数 | **当前算子更先进** ✅ |
| **回调函数使用** | ✅ 共用 | ✅ 共用 | **一致** |
| **代码复用程度** | ❌ 无 | ❌ 无 | **一致** |

**核心发现**：
- ✅ **当前算子的实现已经比参考算子更先进**
- ✅ **当前算子的 V1 已经有架构判断**（参考算子的 V1 没有）
- ✅ **当前算子的 V2 使用属性参数**（参考算子的 V2 使用环境变量）
- ❌ **不需要参考算子的方案改造当前算子**

---

## 七、具体的应用指导

### 7.1 是否需要改造？

**结论：不需要改造**

**理由**：

1. **当前算子已经使用了属性参数方式**，比参考算子的环境变量方式更先进
2. **当前算子的架构判断逻辑已经完整**，覆盖了所有必要场景
3. **当前算子的回调函数使用正确**，使用了公共回调函数
4. **当前算子的 V2 实现已经满足需求**，没有功能缺陷

### 7.2 当前算子的优势

| 优势点 | 说明 | 代码示例 |
|--------|------|---------|
| **属性参数控制** | 使用 `comm_mode` 属性，不依赖环境变量 | `attrs->GetStr(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE)` |
| **精确控制** | 可以针对单个算子定制通信方式 | arch35 + "ai_cpu" → AICPU；arch35 + "ccu" → CCU |
| **架构判断完整** | V1 和 V2 都有架构判断 | `IsTargetPlatformNpuArch(..., NPUARCH_A5)` |
| **回调函数正确** | 使用了正确的公共回调函数 | `Mc2Arch35GenTaskCallBack`、`Mc2MoeGenTaskCallback` |

### 7.3 参考算子可以借鉴的点（如果有新算子）

**如果开发新的 V2 算子，可以参考当前算子的实现**：

**示例代码（参考当前算子的实现）**：

```cpp
// CalcParam 函数
ge::Status NewOpV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_D(context->GetNodeName(), "attrs is null, use default"),
               return ge::GRAPH_FAILED);
    
    const char* commMode = attrs->GetStr(static_cast<size_t>(NewOpV2AttrIdx::K_COMM_MODE));
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);

    const char* serverType = nullptr;
    const char* streamType = nullptr;

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
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid comm_mode: %s", commMode);
        return ge::GRAPH_FAILED;
    }
    
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

// GenTask 函数
ge::Status NewOpV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                              std::vector<std::vector<uint8_t>> &tasks)
{
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_D(context->GetNodeName(), "attrs is null, use CCU GenTask"),
               return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks));
    
    const char* commMode = attrs->GetStr(static_cast<size_t>(NewOpV2AttrIdx::K_COMM_MODE));
    
    if (commMode == nullptr || strlen(commMode) == 0 || strcmp(commMode, "ccu") == 0) {
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid comm_mode: %s", commMode);
        return ge::GRAPH_FAILED;
    }
}
```

**Proto 定义**：

```cpp
REG_OP(NewOpV2)
    ...
    .ATTR(comm_mode, String, "")  // 新增属性参数
    .OP_END_FACTORY_REG(NewOpV2)
```

---

## 八、风险评估

### 8.1 改造风险（如果改造）

| 风险类型 | 风险描述 | 影响程度 | 发生概率 |
|---------|---------|---------|---------|
| **功能风险** | 改用环境变量方式会丢失算子级控制 | 高 | 高 |
| **兼容性风险** | 已有模型可能依赖属性参数方式 | 高 | 中 |
| **运维风险** | 环境变量方式增加运维负担 | 中 | 高 |
| **测试风险** | 需要重新测试所有场景 | 高 | 高 |

### 8.2 不改造的优势

| 优势类型 | 优势描述 | 效果 |
|---------|---------|------|
| **功能优势** | 保持算子级精确控制 | 灵活性高 |
| **兼容性优势** | 不破坏已有模型 | 无兼容性问题 |
| **运维优势** | 不增加运维负担 | 运维负担低 |
| **测试优势** | 不需要重新测试 | 测试成本低 |

**核心发现**：
- ❌ **不建议改造当前算子**
- ✅ **当前算子的实现已经比参考算子更先进**
- ✅ **不改造可以避免所有风险**

---

## 九、总结表格

### 9.1 gentask 复用方式对比

| 维度 | all_gather_matmul 系列 | 当前算子系列 | 是否需要改造 | 改造建议 |
|------|----------------------|-------------|------------|---------|
| **V1 和 V2 调用关系** | ❌ 没有调用关系 | ❌ 没有调用关系 | ❌ 不需要 | 保持现状 |
| **回调函数使用** | ✅ 共用公共回调 | ✅ 共用公共回调 | ❌ 不需要 | 保持现状 |
| **V1 的架构判断** | ❌ 无架构判断 | ✅ 有架构判断 | ❌ 不需要 | **当前算子更先进** |
| **V2 的参数传递** | ❌ 环境变量方式 | ✅ 属性参数方式 | ❌ 不需要 | **当前算子更先进** |
| **代码复用程度** | ❌ 无代码复用 | ❌ 无代码复用 | ❌ 不需要 | 保持现状 |

### 9.2 回调函数使用对比

| 回调函数 | 参考算子使用 | 当前算子使用 | 是否可共用 | 说明 |
|---------|------------|------------|----------|------|
| `CommonKFCMc2CalcParamFunc` | ✅ 使用 | ✅ 使用 | ✅ 可共用 | 基础 CalcParam 函数 |
| `CommonKFCMc2GenTask` | ✅ 使用 | ❌ 不使用 | ✅ 可共用 | 基础 AICPU GenTask |
| `Mc2Arch35GenTaskCallBack` | ✅ 使用 | ✅ 使用 | ✅ 可共用 | arch35 CCU GenTask |
| `Mc2MoeGenTaskCallback` | ❌ 不使用 | ✅ 使用 | ✅ 可共用 | MoE AICPU GenTask |

### 9.3 参数传递方式对比

| 参数传递方式 | 参考算子使用 | 当前算子使用 | 推荐度 | 说明 |
|------------|------------|------------|--------|------|
| **环境变量方式** | ✅ V2 使用 | ❌ 不使用 | ⭐⭐⭐ | 全局控制，运维负担高 |
| **属性参数方式** | ❌ 不使用 | ✅ V2 使用 | ⭐⭐⭐⭐⭐ | 算子级控制，灵活性高 |

---

## 十、最终结论

### 10.1 核心发现

1. **V1 和 V2 的 gentask 没有调用关系**，是完全独立的实现
2. **V1 和 V2 使用相同的公共回调函数**，定义在公共目录
3. **当前算子的实现已经比参考算子更先进**：
   - V1 有架构判断（参考算子的 V1 没有）
   - V2 使用属性参数（参考算子的 V2 使用环境变量）
4. **不需要改造当前算子**，保持现状即可

### 10.2 应用指导

- ✅ **保持当前算子的实现不变**
- ✅ **继续使用属性参数方式**（比环境变量方式更先进）
- ✅ **继续使用公共回调函数**（已经使用了正确的回调函数）
- ❌ **不建议改用环境变量方式**（会降低灵活性和可控性）

### 10.3 如果开发新算子

- ✅ **参考当前算子的实现**（比参考算子更先进）
- ✅ **使用属性参数方式**（推荐）
- ✅ **使用公共回调函数**（推荐）
- ✅ **完整的架构判断逻辑**（推荐）