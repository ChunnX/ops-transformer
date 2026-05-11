# MC2代码检视报告 - MR#4961

## 基本信息

| 项目 | 内容 |
|------|------|
| MR编号 | #4961 |
| 分支 | aicpu_only |
| Commit | c3c02630a |
| 检视时间 | 2026-05-11 |
| 检视人 | AI Code Reviewer |

## 修改概述

支持`ENV_MC2_COMM_MODE_AICPU`环境变量切换CCU/AICPU通信模式：

1. 新增`Mc2Comm::GetCommModeFromEnv()`统一环境变量读取入口(header-only)
2. 全层级环境变量判断: aclnn + gen_task + tiling + kernel
3. gen_task结合平台判断(A5→CCU, 非A5→AICPU) + 环境变量覆盖
4. Kernel层HcclTypeSelector模板选择HCCL服务器类型
5. tilingKey删除D_T_MM和TILINGKEY_MM维度，模板实例数减少67%

## 检视范围

- `mc2/common/utils/mc2_comm_utils.h`
- `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/hccl_a2av_op.h`
- `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_coarse_grained.h`
- `mc2/allto_allv_quant_grouped_mat_mul/op_host/op_tiling/arch35/allto_allv_quant_grouped_mat_mul_tiling_common.cpp`
- `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp`

---

## MC2领域规则检视

### 规则对照分析

| 规则ID | 规则名称 | 检视结果 | 说明 |
|--------|----------|----------|------|
| MC2-01 | 核间同步必要性 | ✅ 合规 | 本次修改未新增循环结构调用通信API |
| MC2-02 | 流同步正确性 | ✅ 合规 | 未涉及SetFlag/WaitFlag修改 |
| MC2-03 | SyncAll同步生效 | ✅ 合规 | 未涉及SyncAll修改 |
| MC2-04 | 全局操作一致性 | ✅ 合规 | 环境变量全局生效，所有rank参数一致 |
| MC2-16 | 编译规范 | ✅ 合规 | 正确使用`ASC_DEVKIT_MAJOR`宏隔离 |

---

## 潜在风险点分析

### 1. 环境变量线程安全性（低风险）

**代码位置**: `mc2/common/utils/mc2_comm_utils.h`

**代码片段**:
```cpp
inline uint8_t GetCommModeFromEnv()
{
    auto commModeEnv = getenv(COMM_MODE_ENV);  // getenv非线程安全
    if (commModeEnv != nullptr) {
        return COMM_MODE_AICPU;
    }
    return COMM_MODE_CCU;
}
```

**假设检验**:
- 原假设H0：代码线程安全
- 证据：`getenv()`是非线程安全的POSIX函数（C标准）
- 自信值：30%（低风险）

**结论**: 低风险，算子执行前环境变量已稳定，无需修改。

---

### 2. HcclTypeSelector模板默认值一致性（无风险）

**代码位置**: 
- `mc2/allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/communication/hccl_a2av_op.h`
- `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_coarse_grained.h`

**代码片段**:
```cpp
// 量化算子
template<int commMode = TILINGKEY_TPL_CCU>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;  // 默认CCU
};

// 非量化算子
template<int commMode = CCU_COMM_MODE>
struct HcclTypeSelector {
    using type = Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU>;  // 默认CCU
};
```

**假设检验**:
- 原假设H0：两个算子默认值一致
- 证据：两者默认值都为CCU（0），一致
- 自信值：0%（无风险）

**结论**: ✅ 合规，默认行为一致。

---

## 环境变量逻辑验证

### 默认行为（无环境变量）

| 层级 | 平台 | 默认行为 |
|------|------|----------|
| mc2_comm_utils.h | - | 返回CCU(0) |
| gen_task | A5 | CCU server/stream |
| gen_task | 非A5 | AICPU server/stream |
| aclnn | - | NnopbaseSetHcclServerType(CCU) |
| tiling | - | SetCommEngine(CCU_ENGINE) |
| kernel | - | HCCL_SERVER_TYPE_CCU |

### 强制AICPU行为（设置ENV_MC2_COMM_MODE_AICPU）

| 层级 | 行为 |
|------|------|
| mc2_comm_utils.h | 返回AICPU(1) |
| gen_task | 强制aicpu kfc server |
| aclnn | NnopbaseSetHcclServerType(AICPU) |
| tiling | SetCommEngine(AICPU_ENGINE) |
| kernel | HCCL_SERVER_TYPE_AICPU |

---

## 总体结论

**✅ 代码检视通过**

本次修改符合MC2领域规则：
- 无新增同步风险点
- HCCL模板选择正确
- 编译规范合规
- 全局参数一致性保证
- 环境变量逻辑清晰，默认行为与原代码一致

---

## 附录

### MR链接
https://gitcode.com/cann/ops-transformer/merge_requests/4961

### Commit链接
https://gitcode.com/cann/ops-transformer/commit/c3c02630a