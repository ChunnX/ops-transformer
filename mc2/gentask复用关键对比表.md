# gentask 复用关键对比表

## 一、核心问题快速回答

### Q1: V1 是否调用 V2 的 gentask？
**答案**：❌ **否**，V1 和 V2 的 gentask 是完全独立的实现，没有调用关系。

### Q2: 是否使用相同的回调函数？
**答案**：✅ **是**，V1 和 V2 都使用公共回调函数（定义在 `mc2/common/op_graph/` 目录）。

### Q3: 参数传递方式有什么差异？
**答案**：
- 参考算子（all_gather_matmul_v2）：使用**环境变量**（`ENV_MC2_COMM_MODE_AICPU`）
- 当前算子（allto_allv_grouped_mat_mul_v2）：使用**属性参数**（`comm_mode`）
- **当前算子更先进** ✅

### Q4: 当前算子需要改造吗？
**答案**：❌ **不需要**，当前算子的实现已经比参考算子更先进。

---

## 二、gentask 代码对比（参考算子）

### 2.1 all_gather_matmul V1

```cpp
// 固定使用 AICPU，无架构判断，无参数判断
static ge::Status AllGatherMatmulCalcOpParam(gert::ExeResGenerationContext *context) {
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status AllGatherMatmulGenTask(const gert::ExeResGenerationContext *context,
                                          std::vector<std::vector<uint8_t>> &tasks) {
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}
```

### 2.2 all_gather_matmul_v2 V2

```cpp
// 有架构判断，使用环境变量
static ge::Status AllGatherMatmulV2CalcOpParam(gert::ExeResGenerationContext *context) {
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        uint8_t commMode = Mc2Comm::GetCommModeFromEnv();  // 从环境变量读取
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
        } else {
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
        }
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status AllGatherMatmulV2GenTask(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks) {
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
        } else {
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        }
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}
```

---

## 三、gentask 代码对比（当前算子）

### 3.1 allto_allv_grouped_mat_mul V1

```cpp
// 有架构判断，根据架构固定选择
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context) {
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);

    const char* serverType = nullptr;
    const char* streamType = nullptr;

    if (isArch35) {
        serverType = "ccu server";
        streamType = "ccu_stream";
    } else {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks) {
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
}
```

### 3.2 allto_allv_grouped_mat_mul_v2 V2

```cpp
// 有架构判断，使用属性参数
ge::Status AlltoAllvGroupedMatMulV2CalcParamFunc(gert::ExeResGenerationContext *context) {
    auto attrs = context->GetAttrs();
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
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
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvGroupedMatMulV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks) {
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    
    auto attrs = context->GetAttrs();
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    
    if (commMode == nullptr || strlen(commMode) == 0 || strcmp(commMode, "ccu") == 0) {
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
}
```

---

## 四、关键差异对比表

| 维度 | all_gather_matmul V1 | all_gather_matmul_v2 V2 | allto_allv_grouped_mat_mul V1 | allto_allv_grouped_mat_mul_v2 V2 |
|------|---------------------|------------------------|------------------------------|--------------------------------|
| **架构判断** | ❌ 无 | ✅ 有 | ✅ 有 | ✅ 有 |
| **参数传递** | ❌ 固定 AICPU | ✅ 环境变量 | ❌ 根据架构固定 | ✅ 属性参数 |
| **回调函数** | `CommonKFCMc2GenTask` | `CommonKFCMc2GenTask` + `Mc2Arch35GenTaskCallBack` | `Mc2MoeGenTaskCallback` + `Mc2Arch35GenTaskCallBack` | `Mc2MoeGenTaskCallback` + `Mc2Arch35GenTaskCallBack` |
| **代码复用** | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |
| **先进程度** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**核心结论**：
- ✅ **当前算子的 V2 实现最先进**（属性参数 + 完整架构判断）
- ❌ **不需要改造当前算子**

---

## 五、公共回调函数定义位置

| 回调函数 | 定义位置 | 用途 |
|---------|---------|------|
| `CommonKFCMc2CalcParamFunc` | `mc2/common/op_graph/mc2_gen_task_ops_utils.h` | 基础 CalcParam |
| `CommonKFCMc2GenTask` | `mc2/common/op_graph/mc2_gen_task_ops_utils.h` | 基础 AICPU GenTask |
| `Mc2Arch35GenTaskCallBack` | `mc2/common/op_graph/mc2_gen_task_ops_utils_arch35.h` | arch35 CCU GenTask |
| `Mc2MoeGenTaskCallback` | `mc2/common/op_graph/mc2_moe_gen_task_ops_utils.h` | MoE AICPU GenTask |

**所有回调函数都是跨算子共用的** ✅

---

## 六、参数传递方式对比

### 6.1 环境变量方式（all_gather_matmul_v2）

```cpp
// 定义
namespace Mc2Comm {
constexpr char COMM_MODE_ENV[] = "ENV_MC2_COMM_MODE_AICPU";
constexpr uint8_t COMM_MODE_CCU = 0;
constexpr uint8_t COMM_MODE_AICPU = 1;

inline uint8_t GetCommModeFromEnv() {
    auto commModeEnv = getenv(COMM_MODE_ENV);
    if (commModeEnv != nullptr) {
        return COMM_MODE_AICPU;
    }
    return COMM_MODE_CCU;
}
}

// 使用
uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
if (commMode == Mc2Comm::COMM_MODE_AICPU) { ... }
```

**特点**：全局控制，运维负担高 ⭐⭐⭐

### 6.2 属性参数方式（allto_allv_grouped_mat_mul_v2）

```cpp
// Proto 定义
REG_OP(AlltoAllvGroupedMatMulV2)
    ...
    .ATTR(comm_mode, String, "")
    .OP_END_FACTORY_REG(AlltoAllvGroupedMatMulV2)

// 使用
auto attrs = context->GetAttrs();
const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
if (strcmp(commMode, "ai_cpu") == 0) { ... }
```

**特点**：算子级控制，灵活性高 ⭐⭐⭐⭐⭐

---

## 七、架构判断逻辑

**所有算子统一使用**：

```cpp
bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
```

**判断逻辑**：
- `NPUARCH_A5` → arch35（Atlas A3 / Ascend910B）
- 非 `NPUARCH_A5` → arch22（Atlas A2 / Ascend910）

---

## 八、应用指导总结

### 8.1 当前算子应该如何改造？

**结论：不需要改造**

**理由**：
1. 当前算子的 V2 实现已经比参考算子更先进
2. 当前算子使用属性参数方式（比环境变量方式更好）
3. 当前算子的架构判断逻辑已经完整
4. 当前算子使用了正确的公共回调函数

### 8.2 如果开发新算子，应该如何实现？

**建议参考当前算子（allto_allv_grouped_mat_mul_v2）的实现**：

```cpp
// 核心要点：
1. 使用属性参数（comm_mode），不使用环境变量
2. 完整的架构判断逻辑（arch22 固定 AICPU，arch35 参数控制）
3. 使用公共回调函数（Mc2Arch35GenTaskCallBack、Mc2MoeGenTaskCallback）
4. Proto 定义新增 comm_mode 属性
```

---

## 九、风险评估

| 风险类型 | 如果改造的风险 | 不改造的优势 |
|---------|--------------|------------|
| **功能风险** | 高（丢失算子级控制） | 低（保持灵活性） |
| **兼容性风险** | 高（破坏已有模型） | 低（无兼容性问题） |
| **运维风险** | 中（增加运维负担） | 低（运维负担低） |
| **测试风险** | 高（需要重新测试） | 低（测试成本低） |

**推荐：不改造** ✅