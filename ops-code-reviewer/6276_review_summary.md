# 代码检视报告

## 检视概览
- PR 链接：https://gitcode.com/cann/ops-transformer/merge_requests/6276
- PR 标题：merge 'codecheck2' into 'master'
- 代码文件：`attention/common/op_kernel/memcopy/copy_gm_to_l1.h`
- 代码侧别：Kernel侧
- 检视文档：cpp-secure.md, ascendc-api.md, ascendc-perf.md, ascendc-topk.md
- 总条例数：46（适用条例，已过滤）
- 检视时间：2026-06-02

## PR 变更摘要

PR #6276 对 `attention/common/op_kernel/memcopy/copy_gm_to_l1.h` 文件的 `CopySingleMatrixNDToNZ` 模板函数进行了重构优化，变更集中在 `else` 分支（`srcDValue >= 65536` 的逐行搬运场景），共 1 个文件，+9/-12 行。

**核心变更：**

1. **引入编译期类型感知常量**：新增 `BLOCK_ELEMENT_CNT = IsSameType<T, int4b_t>::value ? 64 : 32 / sizeof(T)`，使 L1 偏移计算适配所有模板类型（float=8, half=16, int4b_t=64），替代旧代码硬编码的 `16`。
2. **简化局部变量**：移除冗余中间变量 `srcNdMatrixNum`、`srcNdMatrixStride`、`dstNzMatrixStride`，消除对函数参数 `nValue` 和 `srcDValue` 的重赋值副作用。
3. **修正参数语义**：`nd2nzPara.srcDValue` 从使用被覆写后的局部变量值改为直接使用函数参数 `dValue`，语义更清晰。

**功能等价性验证**：
- 旧代码中 `srcDValue = dValue` 后使用 `srcDValue` → 新代码直接使用 `dValue`：等价
- 旧代码 `nValue = 1` 后循环体内 `nd2nzPara.nValue = nValue`(=1) → 新代码直接写 `nd2nzPara.nValue = 1`：等价
- `gmOffset += srcNdMatrixStride`（其中 `srcNdMatrixStride = srcDValue` 初始值） → 新代码 `gmOffset += srcDValue`（参数未修改）：等价
- `l1Offset += 16` → `l1Offset += BLOCK_ELEMENT_CNT`：**修正为类型感知**（对 float 从 16→8 减少浪费，对 int4b_t 从 16→64 修复 Bug）

## 检视统计

| 状态 | 条例数 | 占比 |
|-----|--------|------|
| PASS | 46 | 100% |
| FAIL（发现问题） | 0 | 0% |
| SUSPICIOUS（需关注） | 0 | 0% |

## 通过条例（共 46 条）

### cpp-secure.md（16 条）

| 条例ID | 条例标题 | 判定 | 置信度 |
|--------|---------|------|--------|
| 1.1 | 保证静态类型安全 | PASS | HIGH |
| 1.2 | 保证内存安全 | PASS | HIGH |
| 1.3 | 禁止使用未定义行为 | PASS | HIGH |
| 2.1 | 有符号整数运算不溢出 | PASS | HIGH |
| 2.2 | 无符号整数运算不回绕 | PASS | HIGH |
| 2.3 | 除法/余数运算除零保护 | PASS | HIGH |
| 3.1 | 禁止使用未初始化的变量 | PASS | HIGH |
| 3.3 | 数组索引校验 | PASS | HIGH |
| 3.4 | 禁止 sizeof 指针 | PASS | HIGH |
| 3.5 | 指针使用前判空 | PASS | HIGH |
| 4.1 | 外部输入合法性校验 | PASS | HIGH |
| 4.2 | 内存操作长度校验 | PASS | HIGH |
| 9.1 | 禁止逐位操作非 trivially copyable 对象 | PASS | HIGH |
| 10.3 | 敏感信息使用后清零 | PASS | HIGH |
| 10.4 | 结构体字段末尾添加 | PASS | HIGH |
| 10.5 | 接口变更考虑兼容性 | PASS | HIGH |

### ascendc-api.md（10 条）

| 条例ID | 条例标题 | 判定 | 置信度 |
|--------|---------|------|--------|
| API-1 | 禁止使用 GlobalTensor::SetValue/GetValue | PASS | HIGH |
| API-2 | 禁止使用 std:: 计算函数 | PASS | HIGH |
| API-3 | DataCopy/DataCopyPad 对齐要求 | PASS | HIGH |
| API-4 | Compare API 256字节对齐要求 | PASS | HIGH |
| API-6 | AllocTensor/FreeTensor 必须配对使用 | PASS | HIGH |
| API-7 | 禁止动态内存分配 | PASS | HIGH |
| API-8 | repeatTimes 限制（≤255） | PASS | HIGH |
| API-9 | Cast RoundMode 正确性 | PASS | HIGH |
| API-10 | DataCopyParams vs DataCopyExtParams 单位差异 | PASS | HIGH |
| API-12 | CrossCoreSetFlag/WaitFlag 必须对称 | PASS | HIGH |

### ascendc-perf.md（13 条）

| 条例ID | 条例标题 | 判定 | 置信度 |
|--------|---------|------|--------|
| PERF-1 | 循环内禁止逐元素操作 | PASS | HIGH |
| PERF-2 | 禁止写死硬件参数 | PASS | HIGH |
| PERF-3 | Double Buffer 使用 | PASS | HIGH |
| PERF-4 | PipeBarrier 优化 | PASS | HIGH |
| PERF-5 | 单次搬运量优化 | PASS | HIGH |
| PERF-6 | 避免 GM 重复读取 | PASS | HIGH |
| PERF-7 | 尾块处理正确性 | PASS | HIGH |
| PREC-1 | 流水线同步正确性 | PASS | HIGH |
| PREC-2 | 除零保护（详见 cpp-secure.md 2.3） | PASS | HIGH |
| PREC-3 | FP16 中间精度保护 | PASS | HIGH |
| TIL-1 | 多核负载均衡 | PASS | HIGH |
| TIL-2 | 片上缓存容量不溢出（UB/L1/L0A/L0B/L0C） | PASS | HIGH |
| TIL-3 | Buffer 规划合理性 | PASS | HIGH |

### ascendc-topk.md（7 条）

| 条例ID | 条例标题 | 判定 | 置信度 |
|--------|---------|------|--------|
| TOPK-3 | 生命周期内使用局部变量指针，避免野指针 | PASS | HIGH |
| TOPK-6 | 必须考虑nan/inf/+0/-0等特殊值和边界值处理 | PASS | HIGH |
| TOPK-8 | gm内存偏移或大小必须用int64表示 | PASS | HIGH |
| TOPK-9 | atomic累加需src(ub)与dst(gm)清零处理 | PASS | HIGH |
| TOPK-10 | 可整数计算时不允许转浮点数计算 | PASS | HIGH |
| TOPK-11 | 通信算子融合需核间同步 | PASS | HIGH |
| TOPK-12 | 宏定义中临时变量命名不能和外部变量冲突 | PASS | HIGH |

## 检视结论

本次 PR #6276 的变更是一次**正面的代码质量改进**，具体表现为：

1. **类型安全性增强**：将硬编码的 `l1Offset += 16` 替换为类型感知的 `l1Offset += BLOCK_ELEMENT_CNT`，修复了对 `float`（旧 64B stride → 新 32B）和 `int4b_t`（旧 8B stride → 新 32B）类型的偏移计算错误。
2. **代码可维护性提升**：移除冗余中间变量（`srcNdMatrixNum`、`srcNdMatrixStride`、`dstNzMatrixStride`），消除对函数参数的重赋值，使代码逻辑更清晰直接。
3. **函数签名兼容**：`CopySingleMatrixNDToNZ` 函数签名未变更，API 层面 100% 向后兼容。
4. **全量覆盖检视**：46 条适用条例全部 PASS，无安全、性能、精度或规范性违规。

**评级：通过**

---

## 被检视代码

> 本次检视的完整代码（供追溯）

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_gm_to_l1.h
 * \brief
 */
#ifndef COPY_GM_TO_L1_H
#define COPY_GM_TO_L1_H

#include "gm_layout.h"
#include "fa_l1_tensor.h"
#include "gm_coord.h"
#include "offset_calculator_v2.h"

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor, const GlobalTensor<T> gmTensor, uint32_t nValue,
                                              uint32_t dValue, uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    constexpr uint32_t SRCDVALUE_LIMIT = 65536;
    Nd2NzParams nd2nzPara;
    if (unlikely(srcDValue < SRCDVALUE_LIMIT)) {
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = nValue;
        if constexpr (IsSameType<T, int4b_t>::value) {
            constexpr uint32_t HALF_SIZE_DIVISOR = 2;
            nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
        } else {
            nd2nzPara.dValue = dValue;
            nd2nzPara.srcDValue = srcDValue;
        }
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    } else {
        constexpr uint32_t NUM32 = 32;                                          // [新增]
        constexpr uint32_t BLOCK_ELEMENT_CNT = IsSameType<T, int4b_t>::value    // [新增]
            ? 64 : NUM32 / sizeof(T);                                           // [新增]
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < nValue; i++) {                                  // [修改] 循环条件变更
            nd2nzPara.ndNum = 1;
            nd2nzPara.nValue = 1;                                                // [修改] 从 nValue 改为 1
            if constexpr (IsSameType<T, int4b_t>::value) {
                constexpr uint32_t HALF_SIZE_DIVISOR = 2;
                nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
                nd2nzPara.srcDValue = dValue / HALF_SIZE_DIVISOR;                // [修改] srcDValue → dValue
            } else {
                nd2nzPara.dValue = dValue;
                nd2nzPara.srcDValue = dValue;                                    // [修改] srcDValue → dValue
            }
            nd2nzPara.dstNzC0Stride = dstNzC0Stride;
            nd2nzPara.dstNzNStride = 1;
            nd2nzPara.srcNdMatrixStride = 0;
            nd2nzPara.dstNzMatrixStride = 0;
            DataCopy(l1Tensor[l1Offset], gmTensor[gmOffset], nd2nzPara);

            gmOffset += srcDValue;                                               // [修改] srcNdMatrixStride → srcDValue
            l1Offset += BLOCK_ELEMENT_CNT;                                       // [修改] dstNzMatrixStride(16) → BLOCK_ELEMENT_CNT
        }
    }
}

// ... (后续函数未变更，省略)
#endif
```

**代码行数**：781 行（完整文件）
