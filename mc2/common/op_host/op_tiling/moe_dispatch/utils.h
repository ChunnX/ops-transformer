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
 * \file utils.h
 * \brief MOE Distribute Dispatch Tiling C 风格工具函数和 Rule 批量校验
 */

#ifndef MOE_DISPATCH_UTILS_H
#define MOE_DISPATCH_UTILS_H

#include <cstdint>
#include "defs.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/storage_shape.h"

namespace moe_dispatch_utils {

// ========== Rule 结构体常量 ==========
constexpr uint32_t MAX_VALID_DTYPES = 4U;

// ========== Rule 结构体定义 ==========

struct TensorCheckRule {
    uint32_t index;                             // tensor 在 context 中的索引
    moe_dispatch_defs::TensorType type;         // INPUT/OUTPUT/OPTIONAL_INPUT
    uint32_t expectedDim;                       // 期望维度（0 = 不检查）
    ge::DataType validDtypes[MAX_VALID_DTYPES]; // 合法 dtype 集合
    uint32_t dtypeCount;                        // validDtypes 有效个数
    const char *name;                           // 日志用名称
};

struct AttrCheckRule {
    uint32_t attrIndex;   // 属性在 context 中的索引
    const char *attrName; // 日志用名称
    uint32_t minVal;      // 下界（含）
    uint32_t maxVal;      // 上界（含）
    bool required;        // 是否必选
};

struct TensorFormatRule {
    uint32_t index;                     // tensor 在 context 中的索引
    moe_dispatch_defs::TensorType type; // INPUT/OUTPUT/OPTIONAL_INPUT
    ge::Format expectedFormat;          // 期望的 format
    bool checkPrimary;                  // true = 使用 GetPrimaryFormat
    const char *name;                   // 日志用名称
};

// ========== 小结果结构体 ==========

struct SharedExpertResult {
    uint32_t localExpertNum;
    bool isSharedRank;
    uint32_t moeExpertPerRankNum;
};

struct PlatformInfo {
    uint32_t aivNum;
    uint64_t ubSize;
    uint32_t blockDim;
};

// ========== 基础计算函数（inline，<=5参数，<=50行） ==========

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0U) ? 0U : (a + b - 1U) / b;
}

inline uint32_t AlignUp(uint32_t val, uint32_t align)
{
    return CeilDiv(val, align) * align;
}

inline int64_t CeilDivInt64(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

inline int64_t AlignUpInt64(int64_t val, int64_t align)
{
    return CeilDivInt64(val, align) * align;
}

// ========== Tensor 基础校验函数 ==========

inline ge::graphStatus CheckTensorDimNum(const gert::StorageShape *shape, uint32_t expectedDim)
{
    if (shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (expectedDim == 0U) {
        return ge::GRAPH_SUCCESS; // 0 表示不检查
    }
    if (shape->GetStorageShape().GetDimNum() != expectedDim) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckTensorDtypeIn(const gert::CompileTimeTensorDesc *desc, const ge::DataType *validDtypes,
                                          uint32_t dtypeCount)
{
    if (desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0U; i < dtypeCount; ++i) {
        if (desc->GetDataType() == validDtypes[i]) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

// ========== 属性范围校验 ==========

inline ge::graphStatus CheckAttrRangeU32(uint32_t val, uint32_t minVal, uint32_t maxVal)
{
    return (val >= minVal && val <= maxVal) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

inline ge::graphStatus CheckAttrRangeInt64(int64_t val, int64_t minVal, int64_t maxVal)
{
    return (val >= minVal && val <= maxVal) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

// ========== Shared Expert 计算 ==========

inline SharedExpertResult CalcLocalExpertNum(uint32_t epRankId, uint32_t sharedRankNum, uint32_t moeExpertNum,
                                             uint32_t epWorldSize)
{
    SharedExpertResult result{};
    result.localExpertNum = 0U;
    result.isSharedRank = false;
    result.moeExpertPerRankNum = 0U;

    if (sharedRankNum == 0U) {
        result.moeExpertPerRankNum = moeExpertNum / epWorldSize;
        result.localExpertNum = result.moeExpertPerRankNum;
        return result;
    }

    if (epRankId < sharedRankNum) {
        result.isSharedRank = true;
        result.localExpertNum = 1U;
        result.moeExpertPerRankNum = 1U;
    } else {
        result.isSharedRank = false;
        uint32_t normalRankNum = epWorldSize - sharedRankNum;
        if (normalRankNum > 0U && moeExpertNum % normalRankNum == 0U) {
            result.moeExpertPerRankNum = moeExpertNum / normalRankNum;
            result.localExpertNum = result.moeExpertPerRankNum;
        }
    }
    return result;
}

// ========== Shared Expert 关联校验 ==========

inline ge::graphStatus CheckSharedExpertRelation(uint32_t sharedExpertNum, uint32_t sharedExpertRankNum,
                                                 uint32_t epWorldSize, uint32_t moeExpertNum)
{
    // 1. sharedExpertRankNum == 0 时 sharedExpertNum 必须为 0
    if (sharedExpertRankNum == 0U && sharedExpertNum != 0U) {
        return ge::GRAPH_FAILED;
    }

    // 2. sharedExpertRankNum > 0 时校验关系
    if (sharedExpertRankNum > 0U) {
        // sharedExpertNum 必须大于 0 且 <= sharedExpertRankNum
        if (sharedExpertNum == 0U || sharedExpertNum > sharedExpertRankNum) {
            return ge::GRAPH_FAILED;
        }
        // sharedExpertRankNum % sharedExpertNum == 0
        if (sharedExpertRankNum % sharedExpertNum != 0U) {
            return ge::GRAPH_FAILED;
        }
        // moeExpertNum % (epWorldSize - sharedExpertRankNum) == 0
        uint32_t normalRankNum = epWorldSize - sharedExpertRankNum;
        if (normalRankNum == 0U || moeExpertNum % normalRankNum != 0U) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ========== EP Rank 关联校验 ==========

inline ge::graphStatus CheckEpRankRelation(uint32_t epRankId, uint32_t epWorldSize)
{
    if (epRankId >= epWorldSize) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ========== 平台信息获取 ==========

inline PlatformInfo GetPlatformInfo(const gert::TilingContext *ctx)
{
    PlatformInfo result{};
    result.aivNum = 0U;
    result.ubSize = 0UL;
    result.blockDim = 0U;

    if (ctx == nullptr) {
        return result;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    result.aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, result.ubSize);
    result.blockDim = ascendcPlatform.CalcTschBlockDim(result.aivNum, 0U, result.aivNum);

    return result;
}

// ========== 批量校验函数（需 cpp 实现） ==========

ge::graphStatus CheckTensorRules(gert::TilingContext *ctx, const TensorCheckRule *rules, uint32_t count);

ge::graphStatus CheckOneTensorRule(gert::TilingContext *ctx, const TensorCheckRule &rule);

ge::graphStatus CheckAttrRules(gert::TilingContext *ctx, const AttrCheckRule *rules, uint32_t count);

ge::graphStatus CheckTensorFormatRules(gert::TilingContext *ctx, const TensorFormatRule *rules, uint32_t count,
                                       const char *nodeName);

} // namespace moe_dispatch_utils

#endif // MOE_DISPATCH_UTILS_H