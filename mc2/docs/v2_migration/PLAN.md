# V2 算子收编 V1 算子实现开发计划

## 文档信息

- **项目名称**: V2 算子收编 V1 算子实现
- **涉及算子**: 
  - `allto_allv_grouped_mat_mul` (V1) → `allto_allv_grouped_mat_mul_v2` (V2)
  - `grouped_mat_mul_allto_allv` (V1) → `grouped_mat_mul_allto_allv_v2` (V2)
- **计划日期**: 2026-05-28
- **设计文档**: `DESIGN.md`

---

## 1. 需求概述

### 1.1 背景

V2 算子引入 `commMode` 参数，支持通过参数指定通信方式（AICPU/CCU），避免运行时判断。V1 算子需要收编 V2 实现，保持 V1 接口向后兼容。

### 1.2 目标

1. **aclnn 两段式接口**: V1 调用 V2 aclnn，传递 commMode 参数
2. **fallback 通路**: 保持 V1 原有实现（间接使用 V2）
3. **gentask 通路**: 保持 V1 判断逻辑（共用 GenTask 回调）
4. **向后兼容**: V1 接口不变，用户无感知

### 1.3 范围

- **算子范围**:
  - `allto_allv_grouped_mat_mul` (V1)
  - `grouped_mat_mul_allto_allv` (V1)
  
- **代码范围**:
  - op_api: aclnn 接口改造（优先级高）
  - op_graph: gentask/fallback 通路（可选/无需）
  - op_kernel: 保持原有（不修改）
  - op_host: 保持原有（不修改）

---

## 2. 开发阶段规划

### 阶段 1: aclnn 接口改造（高优先级）

**目标**: V1 aclnn 调用 V2 aclnn

**时间**: 2 天

**任务列表**:

| 序号 | 任务 | 文件 | 工作量 | 前置依赖 |
|------|------|------|--------|---------|
| 1.1 | allto_allv_grouped_mat_mul aclnn 改造 | `aclnn_allto_allv_grouped_mat_mul.cpp` | 0.5 天 | 无 |
| 1.2 | grouped_mat_mul_allto_allv aclnn 改造 | `aclnn_grouped_mat_mul_allto_allv.cpp` | 0.5 天 | 无 |
| 1.3 | 编译配置确认 | CMakeLists.txt（跨算子引用） | 0.5 天 | 1.1, 1.2 |
| 1.4 | 单元测试 | test_aclnn_xxx.cpp | 0.5 天 | 1.1, 1.2 |

---

### 阶段 2: gentask 通路改造（低优先级）

**目标**: V1 gentask 保持原有逻辑（方案 C）

**时间**: 0 天（无需修改）

**任务列表**:

| 序号 | 任务 | 文件 | 工作量 | 前置依赖 |
|------|------|------|--------|---------|
| 2.1 | allto_allv_grouped_mat_mul gentask | 无需修改 | 0 天 | 无 |
| 2.2 | grouped_mat_mul_allto_allv gentask | 无需修改 | 0 天 | 无 |

**说明**: 
- V1 gentask 保持原有判断逻辑
- V1 和 V2 共用相同的 GenTask 回调函数
- 无需额外改造

---

### 阶段 3: fallback 通路改造（无需）

**目标**: V1 fallback 保持原有实现

**时间**: 0 天（无需修改）

**任务列表**:

| 序号 | 任务 | 文件 | 工作量 | 前置依赖 |
|------|------|------|--------|---------|
| 3.1 | allto_allv_grouped_mat_mul fallback | 无需修改 | 0 天 | 无 |
| 3.2 | grouped_mat_mul_allto_allv fallback | 无需修改 | 0 天 | 无 |

**说明**: 
- fallback 调用 V1 aclnn，间接使用 V2 实现
- 无需额外改造

---

### 阶段 4: 测试验证（高优先级）

**目标**: 验证功能、兼容性、性能

**时间**: 3 天

**任务列表**:

| 序号 | 任务 | 内容 | 工作量 | 前置依赖 |
|------|------|------|--------|---------|
| 4.1 | 单元测试 | 参数传递、环境变量判断、架构判断 | 1 天 | 1.1, 1.2 |
| 4.2 | 集成测试 | fallback 通路、gentask 通路、性能对比 | 1 天 | 1.1, 1.2 |
| 4.3 | ST 测试 | 完整算子功能、不同 dtype、不同 shape | 1 天 | 1.1, 1.2 |

---

## 3. 详细实现步骤

### 3.1 阶段 1: aclnn 接口改造

#### 步骤 1.1: allto_allv_grouped_mat_mul aclnn 改造

**文件**: `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul.cpp`

**修改内容**:

##### 修改 1: 添加 V2 aclnn 头文件引用

**位置**: 文件头部（约第 10-18 行）

**改动**:
```cpp
#include "aclnn_allto_allv_grouped_mat_mul.h"
#include <algorithm>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnnInner_allto_allv_grouped_mat_mul.h"
#include "mc2_comm_utils.h"
#include "aclnn_allto_allv_grouped_mat_mul_v2.h"  // 新增：V2 aclnn 头文件
```

**说明**: 添加 V2 aclnn 头文件引用

---

##### 修改 2: GetWorkspaceSize 改造

**位置**: 第 109-130 行

**改动**:

**删除代码**:
```cpp
aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(gmmX, gmmWeight, sendCountsTensorOptional,
    recvCountsTensorOptional, mmXOptional, mmWeightOptional, const_cast<char *>(group), epWorldSize, sendCounts,
    recvCounts, transGmmWeight, transMmWeight, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional, workspaceSize,
    executor);
return ret;
```

**新增代码**:
```cpp
// 判断通信方式
uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();
const char* commMode = (commModeInt == Mc2Comm::COMM_MODE_AICPU) ? "ai_cpu" : "";

// 调用 V2 aclnn 接口
aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    gmmX, gmmWeight,
    sendCountsTensorOptional, recvCountsTensorOptional,
    mmXOptional, mmWeightOptional,
    const_cast<char *>(group), epWorldSize,
    sendCounts, recvCounts,
    transGmmWeight, transMmWeight, permuteOutFlag,
    commMode,  // 新增参数
    gmmY, mmYOptional, permuteOutOptional,
    workspaceSize, executor);

return ret;
```

**说明**:
- 判断通信方式（环境变量）
- 转换为 commMode 字符串
- 调用 V2 aclnn 接口

---

##### 修改 3: Execute 改造

**位置**: 第 132-149 行

**改动**:

**删除代码**:
```cpp
if (NnopbaseSetHcclServerType) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            OP_LOGD("arch35 with ENV_MC2_COMM_MODE_AICPU, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            OP_LOGD("arch35, use CCU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
}
aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
return ret;
```

**新增代码**:
```cpp
// 直接调用 V2 Execute
aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2(workspace, workspaceSize, executor, stream);
return ret;
```

**说明**:
- 删除原有通信方式判断逻辑
- 删除 NnopbaseSetHcclServerType 调用
- 删除 aclnnInner_xxx 调用
- 直接调用 V2 Execute

---

#### 步骤 1.2: grouped_mat_mul_allto_allv aclnn 改造

**文件**: `mc2/grouped_mat_mul_allto_allv/op_api/aclnn_grouped_mat_mul_allto_allv.cpp`

**修改内容**:

##### 修改 1: 添加 V2 aclnn 头文件引用

**位置**: 文件头部（约第 10-18 行）

**改动**:
```cpp
#include <algorithm>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnn_grouped_mat_mul_allto_allv.h"
#include "aclnnInner_grouped_mat_mul_allto_allv.h"
#include "mc2_comm_utils.h"
#include "aclnn_grouped_mat_mul_allto_allv_v2.h"  // 新增：V2 aclnn 头文件
```

---

##### 修改 2: GetWorkspaceSize 改造

**位置**: 第 116-135 行

**改动**:

**删除代码**:
```cpp
aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
    gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
    const_cast<char *>(group), epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional,
    workspaceSize, executor);
return ret;
```

**新增代码**:
```cpp
// 判断通信方式
uint8_t commModeInt = Mc2Comm::GetCommModeFromEnv();
const char* commMode = (commModeInt == Mc2Comm::COMM_MODE_AICPU) ? "ai_cpu" : "";

// 调用 V2 aclnn 接口
aclnnStatus ret = aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(
    gmmX, gmmWeight,
    sendCountsTensorOptional, recvCountsTensorOptional,
    mmXOptional, mmWeightOptional,
    const_cast<char *>(group), epWorldSize,
    sendCounts, recvCounts,
    transGmmWeight, transMmWeight,
    commMode,  // 新增参数（注意位置）
    y, mmYOptional,
    workspaceSize, executor);

return ret;
```

---

##### 修改 3: Execute 改造

**位置**: 第 137-154 行

**改动**:

**删除代码**:
```cpp
if (NnopbaseSetHcclServerType) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
        if (commMode == Mc2Comm::COMM_MODE_AICPU) {
            OP_LOGD("Arch35 platform with ENV_MC2_COMM_MODE_AICPU, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else {
            OP_LOGD("Arch35 platform, use CCU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
}
aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
return ret;
```

**新增代码**:
```cpp
// 直接调用 V2 Execute
aclnnStatus ret = aclnnGroupedMatMulAlltoAllvV2(workspace, workspaceSize, executor, stream);
return ret;
```

---

#### 步骤 1.3: 编译配置确认

**任务**: 确认跨算子引用支持

**检查内容**:
1. CMakeLists.txt 是否包含 V2 算子目录
2. 头文件搜索路径是否包含 V2 算子目录
3. 编译顺序是否正确（V2 先编译，V1 后编译）

**修改文件**:
- 可能需要修改 CMakeLists.txt（如果跨算子引用不支持）

**示例修改**:
```cmake
# 添加 V2 算子头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mc2/allto_allv_grouped_mat_mul_v2/op_api)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mc2/grouped_mat_mul_allto_allv_v2/op_api)
```

---

#### 步骤 1.4: 单元测试

**任务**: 编写单元测试验证参数传递和判断逻辑

**测试内容**:

1. **参数传递测试**:
   - V1 参数传递给 V2 是否正确
   - commMode 参数生成是否正确

2. **环境变量测试**:
   - 设置 ENV_MC2_COMM_MODE_AICPU
   - 不设置环境变量

3. **架构判断测试**:
   - arch22 平台
   - arch35 平台

**测试文件**:
- `mc2/allto_allv_grouped_mat_mul/tests/ut/op_api/test_aclnn_allto_allv_grouped_mat_mul.cpp`
- `mc2/grouped_mat_mul_allto_allv/tests/ut/op_api/test_aclnn_grouped_mat_mul_allto_allv.cpp`

---

### 3.2 阶段 4: 测试验证

#### 步骤 4.1: 单元测试

**测试内容**:

| 测试项 | 测试方法 | 预期结果 |
|--------|----------|----------|
| 参数传递 | 调用 V1 aclnn，检查 V2 被调用 | V2 被正确调用 |
| commMode 生成 | 设置环境变量，检查 commMode | "ai_cpu" 或 "" |
| 架构判断 | 不同架构，检查通信方式 | arch22: AICPU; arch35: CCU/AICPU |
| 输出一致性 | V1 和 V2 输出对比 | 输出一致 |

---

#### 步骤 4.2: 集成测试

**测试内容**:

| 测试项 | 测试方法 | 预期结果 |
|--------|----------|----------|
| fallback 通路 | 调用 fallback，检查输出 | 输出正确 |
| gentask 通路 | 调用 gentask，检查任务生成 | 任务生成正确 |
| 性能对比 | V1 vs V2 性能测试 | 性能相近或 V2 更优 |

---

#### 步骤 4.3: ST 测试

**测试内容**:

| 测试项 | 测试方法 | 预期结果 |
|--------|----------|----------|
| 完整功能 | 各种输入组合 | 输出正确 |
| 不同 dtype | FP16/BF16 | 支持正确 |
| 不同 shape | 各种 shape 组合 | 输出正确 |
| 不同场景 | Prefill/Decode | 输出正确 |

---

## 4. 检查点清单

### 4.1 阶段检查点

#### 阶段 1: aclnn 接口改造完成检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| V2 aclnn 头文件引用 | ✅ 完成 | 两个算子都已添加引用 |
| GetWorkspaceSize 改造 | ✅ 完成 | 调用 V2 aclnn，传递 commMode |
| Execute 改造 | ✅ 完成 | 删除判断逻辑，调用 V2 Execute |
| 编译通过 | ⏳ 待验证 | 需执行编译验证 |
| 单元测试通过 | ⏳ 待验证 | 需编写和执行测试 |

---

#### 阶段 4: 测试验证完成检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 单元测试通过 | ✅/❌ | 所有测试项通过 |
| 集成测试通过 | ✅/❌ | fallback 和 gentask 正常 |
| ST 测试通过 | ✅/❌ | 完整功能正确 |
| 性能测试通过 | ✅/❌ | 性能符合预期 |

---

## 5. 风险和应对

### 5.1 技术风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 跨算子引用不支持 | 编译失败 | 修改编译配置，添加头文件路径 |
| 参数顺序错误 | 调用失败 | 严格按照 V2 参数顺序 |
| 输出参数缺失 | 输出错误 | 正确传递 permuteOutOptional |
| 环境变量判断错误 | 通信方式错误 | 单元测试验证判断逻辑 |

---

### 5.2 时间风险

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 编译配置复杂 | 时间延长 | 提前确认编译配置 |
| 测试覆盖不足 | 时间延长 | 提前规划测试内容 |
| 兼容性问题 | 时间延长 | 提前分析兼容性 |

---

## 6. 交付物清单

### 6.1 代码交付物

| 文件 | 类型 | 说明 |
|------|------|------|
| `aclnn_allto_allv_grouped_mat_mul.cpp` | 改造 | V1 aclnn 调用 V2 |
| `aclnn_grouped_mat_mul_allto_allv.cpp` | 改造 | V1 aclnn 调用 V2 |
| CMakeLists.txt | 改造（可能） | 添加跨算子引用支持 |

---

### 6.2 测试交付物

| 文件 | 类型 | 说明 |
|------|------|------|
| 单元测试代码 | 新增/改造 | 验证参数传递和判断逻辑 |
| 集成测试代码 | 新增/改造 | 验证 fallback 和 gentask |
| ST 测试用例 | 新增/改造 | 完整功能验证 |

---

### 6.3 文档交付物

| 文件 | 类型 | 说明 |
|------|------|------|
| DESIGN.md | 设计文档 | 架构设计方案 |
| PLAN.md | 开发计划 | 实现路线图 |
| WALKTHROUGH.md | 串讲记录 | 设计串讲质疑回应 |

---

## 7. 联系方式

**设计者**: Architect

**开发者**: Developer

**测试者**: Tester

**文档路径**: `mc2/docs/v2_migration/`

---

## 8. 附录

### 8.1 算子参数对比表

#### allto_allv_grouped_mat_mul

| 参数位置 | V1 | V2 | 差异 |
|---------|----|----|------|
| 1-13 | (详见 DESIGN.md) | (详见 DESIGN.md) | 无差异 |
| 14 | - | commMode | V2 新增 |
| 15-17 | (详见 DESIGN.md) | (详见 DESIGN.md) | 无差异 |
| 18 | - | permuteOutOptional | V2 新增 |
| 19-21 | (详见 DESIGN.md) | (详见 DESIGN.md) | 无差异 |

#### grouped_mat_mul_allto_allv

| 参数位置 | V1 | V2 | 差异 |
|---------|----|----|------|
| 1-12 | (详见 DESIGN.md) | (详见 DESIGN.md) | 无差异 |
| 13 | - | commMode | V2 新增 |
| 14-17 | (详见 DESIGN.md) | (详见 DESIGN.md) | 无差异 |

---

### 8.2 通信方式判断逻辑表

| 环境变量 | 架构 | commMode | 通信方式 |
|---------|------|----------|----------|
| 设置 | arch22 | "ai_cpu" | AICPU |
| 设置 | arch35 | "ai_cpu" | AICPU |
| 未设置 | arch22 | "" | AICPU（默认） |
| 未设置 | arch35 | "" | CCU（默认） |

---

### 8.3 文件修改清单

| 文件 | 修改类型 | 修改内容 |
|------|----------|----------|
| `aclnn_allto_allv_grouped_mat_mul.cpp` | 改造 | GetWorkspaceSize 调用 V2；Execute 简化 |
| `aclnn_grouped_mat_mul_allto_allv.cpp` | 改造 | GetWorkspaceSize 调用 V2；Execute 简化 |
| CMakeLists.txt | 改造（可能） | 添加 V2 头文件路径 |
| 单元测试文件 | 新增/改造 | 验证改造逻辑 |

---

## 9. 总结

### 9.1 开发要点

1. **优先级**: aclnn 接口改造优先（高），gentask/fallback 改造可选/无需（低）
2. **时间**: 2 天（aclnn 改造）+ 3 天（测试验证）= 5 天
3. **风险**: 编译配置、参数顺序、输出参数
4. **检查**: 编译通过、单元测试通过、集成测试通过、ST 测试通过

### 9.2 成功标准

1. V1 aclnn 调用 V2 aclnn 成功
2. V1 接口向后兼容
3. 输出与 V1 原有实现一致
4. 性能符合预期
5. 所有测试通过

---

## 10. 备注

> 本计划基于 DESIGN.md，实际开发中可能调整。
> 
> Developer 串讲后，需要补充 WALKTHROUGH.md 质疑回应。