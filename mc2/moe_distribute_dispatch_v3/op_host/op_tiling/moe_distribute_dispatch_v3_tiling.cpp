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
 * \file moe_distribute_dispatch_v3_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "moe_distribute_dispatch_v3_tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
static ge::graphStatus CheckIndependentSharedDispatch(gert::TilingContext *context, DispatchV2Config &config)
{
    if (!config.enableIndependentSharedExpert) {
        return ge::GRAPH_SUCCESS;
    }
    OP_TILING_CHECK(mc2tiling::GetNpuArch(context) != NpuArch::DAV_3510,
                    OP_LOGE(context->GetNodeName(), "Independent shared experts are only supported on A5."),
                    return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    const auto tp = attrs->GetAttrPointer<int64_t>(4U);
    const auto tpRank = attrs->GetAttrPointer<int64_t>(5U);
    const auto shard = attrs->GetAttrPointer<int64_t>(6U);
    const auto shared = attrs->GetAttrPointer<int64_t>(7U);
    const auto sharedRanks = attrs->GetAttrPointer<int64_t>(8U);
    const auto ep = attrs->GetAttrPointer<int64_t>(0U);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tp);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRank);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shard);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shared);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedRanks);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ep);
    const char *name = context->GetNodeName();
    OP_TILING_CHECK(*tp != 1 || *tpRank != 0 || *shard != 0 || *shared <= 0 || *sharedRanks <= 0 ||
                        *sharedRanks >= *ep || *sharedRanks % *shared != 0,
                    OP_LOGE(name, "Independent shared experts require TP=1, shard=0 and a valid shared rank group."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context->GetOptionalInputDesc(config.elasticInfoIndex) != nullptr,
                    OP_LOGE(name, "Independent shared experts do not support elastic_info."), return ge::GRAPH_FAILED);
    const auto input = context->GetInputShape(config.xIndex);
    const auto inputDesc = context->GetInputDesc(config.xIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, input);
    OPS_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    OP_TILING_CHECK(input->GetStorageShape().GetDimNum() != 2U, OP_LOGE(name, "Token input must be a matrix."),
                    return ge::GRAPH_FAILED);
    auto dtype = inputDesc->GetDataType();
    OP_TILING_CHECK(dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16,
                    OP_LOGE(name, "Independent shared tokens must be BF16 or FP16."), return ge::GRAPH_FAILED);
    int64_t h = input->GetStorageShape().GetDim(1U);
    const auto original = context->GetOptionalInputShape(8U);
    const auto originalDesc = context->GetOptionalInputDesc(8U);
    OPS_CHECK_NULL_WITH_CONTEXT(context, original);
    OPS_CHECK_NULL_WITH_CONTEXT(context, originalDesc);
    OP_TILING_CHECK(original->GetStorageShape().GetDimNum() != 2U ||
                        original->GetStorageShape().GetDim(0U) != input->GetStorageShape().GetDim(0U) ||
                        original->GetStorageShape().GetDim(1U) != h * 2 || originalDesc->GetDataType() != dtype ||
                        originalDesc->GetStorageFormat() != ge::FORMAT_ND,
                    OP_LOGE(name, "shared_expert_input must have shape [BS, 2*H] and the same dtype as x."),
                    return ge::GRAPH_FAILED);
    const auto alg = attrs->GetAttrPointer<char>(config.attrCommAlgIndex);
    OP_TILING_CHECK(alg == nullptr || std::strcmp(alg, "fullmesh_v2") != 0,
                    OP_LOGE(name, "Independent shared dispatch requires fullmesh_v2."), return ge::GRAPH_FAILED);
    const auto quant = attrs->GetAttrPointer<int64_t>(config.attrQuantModeIndex);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quant);
    if (*quant == 0 && config.moeExpertDtype == ge::DT_UNDEFINED) {
        config.moeExpertDtype = static_cast<ge::DataType>(dtype);
    }
    if ((*quant == 4 || *quant == 5) && config.moeExpertDtype == ge::DT_UNDEFINED) {
        config.moeExpertDtype = ge::DT_FLOAT8_E4M3FN;
    }
    OP_TILING_CHECK(config.moeExpertDtype == ge::DT_UNDEFINED || (*quant == 0 && config.moeExpertDtype != dtype) ||
                        ((*quant == 4 || *quant == 5) && config.moeExpertDtype != ge::DT_FLOAT8_E4M3FN),
                    OP_LOGE(name, "Invalid moe_expert_dtype; MX requires E4M3FN."), return ge::GRAPH_FAILED);
    const auto output = context->GetOutputDesc(0U);
    OPS_CHECK_NULL_WITH_CONTEXT(context, output);
    OP_TILING_CHECK(output->GetDataType() != (config.isIndependentSharedRank ? dtype : config.moeExpertDtype),
                    OP_LOGE(name, "expand_x dtype does not match this rank's expert role."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(h < 1024 || h > 4096, OP_LOGE(name, "Independent shared experts require 1024 <= latent H <= 4096."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeDispatchV3TilingFuncBase::MoeDistributeDispatchV3TilingFunc(gert::TilingContext *context)
{
    DispatchV2Config config;
    config.contextIndex = 0;                 // 0: 根据dispatchV3算子原型标志位初始化context索引
    config.xIndex = 1;                       // 1: 根据dispatchV3算子原型标志位初始化groupEp索引
    config.expertIdsIndex = 2;               // 2: 根据dispatchV3算子原型标志位初始化expertIds索引
    config.scalesIndex = 3;                  // 3: 根据dispatchV3算子原型标志位初始化scales索引
    config.xActiveMaskIndex = 4;             // 4: 根据dispatchV3算子原型标志位初始化xActiveMask索引
    config.expertScalesIndex = 5;            // 5: 根据dispatchV3算子原型标志位初始化expertScales索引
    config.elasticInfoIndex = 6;             // 6: 根据dispatchV3算子原型标志位初始化elasticInfo索引
    config.performanceInfoIndex = 7;         // 7: 根据dispatchV3算子原型标志位初始化performanceInfo索引
    config.attrEpWorldSizeIndex = 0;         // 0: 根据dispatchV3算子原型标志位初始化epWorldSize索引
    config.attrEpRankIdIndex = 1;            // 1: 根据dispatchV3算子原型标志位初始化epRankId索引
    config.attrMoeExpertNumIndex = 2;        // 2: 根据dispatchV3算子原型标志位初始化moeExpertNum索引
    config.attrCclBufferSizeIndex = 3;       // 3: 根据dispatchV3算子原型标志位初始化cclBufferSize索引
    config.attrTpWorldSizeIndex = 4;         // 4: 根据dispatchV3算子原型标志位初始化tpWorldSize索引
    config.attrTpRankIdIndex = 5;            // 5: 根据dispatchV3算子原型标志位初始化tpRankId索引
    config.attrExpertSharedTypeIndex = 6;    // 6: 根据dispatchV3算子原型标志位初始化expertSharedType索引
    config.attrSharedExpertNumIndex = 7;     // 7: 根据dispatchV3算子原型标志位初始化sharedExpertNum索引
    config.attrSharedExpertRankNumIndex = 8; // 8: 根据dispatchV3算子原型标志位初始化sharedExpertRankNum索引
    config.attrQuantModeIndex = 9;           // 9: 根据dispatchV3算子原型标志位初始化quantMode索引
    config.attrGlobalBsIndex = 10;           // 10: 根据dispatchV3算子原型标志位初始化globalBs索引
    config.attrExpertTokenNumsTypeIndex = 11; // 11: 根据dispatchV3算子原型标志位初始化expertTokenNumType索引
    config.attrCommAlgIndex = 12;             // 12: 根据dispatchV3算子原型标志位初始化commAlg索引
    config.attrZeroExpertNumIndex = 13;  // 13: 根据dispatchV3算子原型标志位初始化zeroExpertNumIndex索引
    config.attrCopyExpertNumIndex = 14;  // 14: 根据dispatchV3算子原型标志位初始化copyExpertNumIndex索引
    config.attrConstExpertNumIndex = 15; // 15: 根据dispatchV3算子原型标志位初始化constExpertNumIndex索引
    config.isMc2Context = true;
    if (context->GetOptionalInputDesc(8U) != nullptr) {
        const auto attrs = context->GetAttrs();
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
        config.enableIndependentSharedExpert = true;
        // Preserve V3 y_dtype at index 16; V6 supplies the MoE communication dtype.
        const auto moeDtype = attrs->GetAttrPointer<int64_t>(16U);
        OPS_CHECK_NULL_WITH_CONTEXT(context, moeDtype);
        config.moeExpertDtype = static_cast<ge::DataType>(*moeDtype);
        const auto rank = attrs->GetAttrPointer<int64_t>(1U);
        const auto sharedRanks = attrs->GetAttrPointer<int64_t>(8U);
        OPS_CHECK_NULL_WITH_CONTEXT(context, rank);
        OPS_CHECK_NULL_WITH_CONTEXT(context, sharedRanks);
        config.isIndependentSharedRank = config.enableIndependentSharedExpert && *rank < *sharedRanks;
    }

    OP_TILING_CHECK(CheckIndependentSharedDispatch(context, config) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Independent shared expert validation failed."),
                    return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter MoeDistributeDispatchV3 tiling");
    ge::graphStatus ret = MoeDistributeDispatchA3TilingFuncImplPublic(context, config);
    return ret;
}

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
// Register exception func
inline void MoeDistributeDispatchV3ExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    Mc2Exception::Mc2ExceptionImpl(args, userdata, "MoeDistributeDispatchV3");
}

__attribute__((constructor)) void RegisterMoeDistributeDispatchV3ExceptionFunc()
{
    int32_t runtimeVersionNum = 0;
    int32_t metadefVersionNum = 0;

    if (aclsysGetVersionNum("runtime", &runtimeVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeDispatchV3", "Get runtime version failed when register exception func.");
        return;
    }
    if (aclsysGetVersionNum("metadef", &metadefVersionNum) != ACL_SUCCESS) {
        OP_LOGW("MoeDistributeDispatchV3", "Get metadef version failed when register exception func.");
        return;
    }

    if (runtimeVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION || metadefVersionNum < EXCEPTION_DUMP_SUPPORT_VERSION) {
        OP_LOGW("MoeDistributeDispatchV3",
                "The runtime(%d) or metadata(%d) version is lower than the version(%d) supporting exception func.",
                runtimeVersionNum, metadefVersionNum, EXCEPTION_DUMP_SUPPORT_VERSION);
        return;
    }

    IMPL_OP(MoeDistributeDispatchV3).ExceptionDumpParseFunc(MoeDistributeDispatchV3ExceptionImplWrapper);
}
#endif
} // namespace optiling
