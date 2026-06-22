/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_tiling_a2a3.cpp
 * \brief
 */


#include <vector>
#include <map>
#include <algorithm>
#include <cfloat>

#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "mc2_log.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_tiling_utils.h"
#include "../../../op_kernel/arch22/mega_moe_tiling_a2a3.h"
#include "../../../op_kernel/arch22/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch22/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace mc2tiling;

namespace MegaMoeA2A3Tiling {
    const char *K_INNER_DEBUG = "MegaMoeA2A3 Tiling Debug";

    // 算子属性索引
    constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 0;           // moe 专家数
    constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 1;            // EP 并行 world size
    constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 2;          // HCCL 通信缓冲区大小
    constexpr uint32_t ATTR_MAX_RECV_TOKEN_NUM_INDEX = 3;       // 最大接收 token 数（用于预分配 workspace）
    constexpr uint32_t ATTR_DISPATCH_QUANT_MODE_INDEX = 4;      // 分发阶段量化模式
    constexpr uint32_t ATTR_DISPATCH_QUANT_OUT_TYPE_INDEX = 5;  // 分发阶段量化输出数据类型
    constexpr uint32_t ATTR_COMBINE_QUANT_MODE_INDEX = 6;       // 合并阶段量化模式
    constexpr uint32_t ATTR_COMM_ALG_INDEX = 7;                 // 通信算法配置
    constexpr uint32_t ATTR_NUM_MAX_TOKEN_PER_RANK_INDEX = 8;   // 每个 rank 的最大 token 数
    constexpr uint32_t ATTR_ACTIVATION_INDEX = 9;               // 激活函数类型（如 "swiglu"）
    constexpr uint32_t ATTR_ACTIVATION_CLAMP_INDEX = 10;        // 激活函数 clamp 值
    constexpr uint32_t ATTR_ACTIVATION_OUT_DTYPE_INDEX = 11;    // 激活函数输出数据类型
    constexpr uint32_t ATTR_TRANSPOSE_WEIGHT1_INDEX = 12;       // weight1 是否转置
    constexpr uint32_t ATTR_TRANSPOSE_WEIGHT2_INDEX = 13;       // weight2 是否转置
    constexpr uint32_t ATTR_WEIGHT1_INTERLEAVE_INDEX = 14;      // weight1 交错模式

    // 输入 tensor 索引
    constexpr uint32_t CONTEXT_INDEX = 0;        // context，shape = [?]
    constexpr uint32_t X_INDEX = 1;              // 输入 token x，shape = [M, K]
    constexpr uint32_t TOPK_IDS_INDEX = 2;       // topk 专家索引，shape = [bs, topK]
    constexpr uint32_t TOPK_WEIGHTS_INDEX = 3;   // topk 权重，shape = [bs, topK]
    constexpr uint32_t WEIGHT1_INDEX = 4;        // FFN 第一层权重（动态输入，每个专家一组）
    constexpr uint32_t WEIGHT2_INDEX = 5;        // FFN 第二层权重（动态输入，每个专家一组）
    constexpr uint32_t WEIGHT_SCALES1_INDEX = 6; // FFN 第一层权重量化 scale
    constexpr uint32_t WEIGHT_SCALES2_INDEX = 7; // FFN 第二层权重量化 scale
    constexpr uint32_t BIAS1_INDEX = 8;          // GMM1 的 bias（OPTIONAL）
    constexpr uint32_t BIAS2_INDEX = 9;          // GMM2 的 bias（OPTIONAL）
    constexpr uint32_t X_ACTIVE_MASK_INDEX = 10; // token 活跃掩码（OPTIONAL）
    constexpr uint32_t SCALES_INDEX = 11;        // 额外 scale（OPTIONAL）

    // 输出 tensor 索引
    constexpr uint32_t OUTPUT_Y_INDEX = 0;              // 输出 y，shape = [M, K]
    constexpr uint32_t OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 1; // 专家 token 数量，shape = [expert_num]

    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
    constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

    // CCL Buffer 相关常量
    constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;  // （预留）60KB 固定偏移
    constexpr int64_t ALIGN_128 = 128LL;
    constexpr int64_t ALIGN_512 = 512LL;

    // 维度范围限制
    constexpr int64_t MIN_BS = 1;
    constexpr int64_t MAX_BS = 8192;
    constexpr int64_t MIN_HIDDEN_SIZE = 1024;
    constexpr int64_t MAX_HIDDEN_SIZE = 10240;
    constexpr int64_t MIN_TOPK = 1;
    constexpr int64_t MAX_TOPK = 32;
    constexpr int64_t MIN_EXPERT_PER_RANK = 1;
    constexpr int64_t MAX_EXPERT_PER_RANK = 128;
    constexpr int64_t HIDDEN_SIZE_ALIGN = 32;

    // 属性范围限制
    constexpr int64_t MIN_MOE_EXPERT_NUM = 1;
    constexpr int64_t MAX_MOE_EXPERT_NUM = 1024;
    constexpr int64_t VALID_EP_WORLD_SIZE[] = {2, 4, 8, 16, 32, 64, 128};

    constexpr uint32_t TWO_DIMS = 2U;
    constexpr uint32_t ONE_DIM = 1U;
    constexpr uint32_t THREE_DIMS = 3U;

    //
    constexpr int64_t DISPATCH_QUANT_MODE_NO_QUANT = 0;
    constexpr int64_t DISPATCH_QUANT_MODE_PER_TENSOR = 2;

// =========================== Attr 校验分函数 ===========================

static ge::graphStatus CheckMoeExpertNumAttr(const int64_t *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "moeExpertNum");
        return GRAPH_FAILED;
    }
    if (*ptr < MIN_MOE_EXPERT_NUM || *ptr > MAX_MOE_EXPERT_NUM) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "moeExpertNum",
            std::to_string(*ptr).c_str(),
            (std::string("should in [") + std::to_string(MIN_MOE_EXPERT_NUM) + ", " +
             std::to_string(MAX_MOE_EXPERT_NUM) + "]").c_str());
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckEpWorldSizeAttr(const int64_t *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "epWorldSize");
        return GRAPH_FAILED;
    }
    bool isValidEpWorldSize = std::find(std::begin(VALID_EP_WORLD_SIZE),
                                        std::end(VALID_EP_WORLD_SIZE),
                                        *ptr) != std::end(VALID_EP_WORLD_SIZE);
    if (!isValidEpWorldSize) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "epWorldSize",
            std::to_string(*ptr).c_str(),
            "one of {2, 4, 8, 16, 32, 64, 128}");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCclBufferSizeAttr(const int64_t *ptr, MegaMoeA2A3TilingData &info,
                                              gert::TilingContext *context)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "cclBufferSize");
        return GRAPH_FAILED;
    }
    const char *nodeName = context->GetNodeName();
    int64_t cclBufferSize = *ptr;
    OP_LOGD(nodeName, "cclBufferSize = %ld Bytes (%ld MB).", cclBufferSize,
            ops::CeilDiv(cclBufferSize, static_cast<int64_t>(MB_SIZE)));
    if (cclBufferSize == 0) {
        OP_LOGW(nodeName, "cclBufferSize is 0, skip validation.");
        return ge::GRAPH_SUCCESS;
    }

    int64_t bs = info.M;
    int64_t h = info.K;
    int64_t topK = info.topK;
    int64_t epWorldSize = info.epWorldSize;
    int64_t expertPerRank = info.moeExpertNum / epWorldSize;

    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    if (yDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "y");
        return ge::GRAPH_FAILED;
    }
    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    int64_t leastCclBufferSize =
        (epWorldSize * ops::CeilAlign(epWorldSize * expertPerRank, ALIGN_128) * sizeof(int32_t)) + // tokenPerExpert
        (ops::CeilAlign(bs * topK * h, ALIGN_512) * sizeof(int8_t)) + // quantTokenScale
        (ops::CeilAlign(bs * h * topK * yDtypeSize, ALIGN_512)); // combineOut

    if (cclBufferSize < leastCclBufferSize) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize",
            std::to_string(cclBufferSize).c_str(),
            (std::string("should equal or larger than ") + std::to_string(leastCclBufferSize)).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckMaxRecvTokenNumAttr(const int64_t *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "maxRecvTokenNum");
        return GRAPH_FAILED;
    }
    if (*ptr < 0) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "maxRecvTokenNum",
            std::to_string(*ptr).c_str(), ">= 0");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 dispatch_quant_mode 和 dispatch_quant_out_dtype，两者都依赖 weight1 的 dataType
static ge::graphStatus CheckDispatchQuantAttrs(gert::TilingContext *context,
    const int64_t *dispatchQuantModePtr, const int64_t *dispatchQuantOutDtypePtr)
{
    const char *nodeName = context->GetNodeName();
    if (dispatchQuantModePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantMode");
        return GRAPH_FAILED;
    }
    if (dispatchQuantOutDtypePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantOutDtype");
        return GRAPH_FAILED;
    }

    auto w1Desc = context->GetDynamicInputDesc(WEIGHT1_INDEX, 0);
    if (w1Desc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "weight1");
        return GRAPH_FAILED;
    }
    ge::DataType w1DataType = w1Desc->GetDataType();

    // dispatch_quant_mode 校验：与 weight 数据类型对应
    int64_t expectedDispatchQuantMode = 0;
    if (w1DataType == ge::DT_INT4 || w1DataType == ge::DT_INT8) {
        expectedDispatchQuantMode = DISPATCH_QUANT_MODE_PER_TENSOR;
    } else if (w1DataType == ge::DT_BF16 || w1DataType == ge::DT_FLOAT16) {
        expectedDispatchQuantMode = DISPATCH_QUANT_MODE_NO_QUANT;
    }
    if (*dispatchQuantModePtr != expectedDispatchQuantMode) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "dispatchQuantMode",
            std::to_string(*dispatchQuantModePtr).c_str(),
            (std::string("should be ") + std::to_string(expectedDispatchQuantMode) +
             " for weight dataType " + Ops::Base::ToString(w1DataType)).c_str());
        return GRAPH_FAILED;
    }

    // dispatch_quant_out_dtype 校验：与 weight 数据类型对应
    if (*dispatchQuantOutDtypePtr != static_cast<int64_t>(ge::DT_UNDEFINED)) {
        int64_t expectedQuantOutType = 0;
        if (w1DataType == ge::DT_BF16) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_BF16);
        } else if (w1DataType == ge::DT_FLOAT16) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_FLOAT16);
        } else if (w1DataType == ge::DT_INT4 || w1DataType == ge::DT_INT8) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_INT8);
        }
        if (*dispatchQuantOutDtypePtr != expectedQuantOutType) {
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "dispatchQuantOutDtype",
                std::to_string(*dispatchQuantOutDtypePtr).c_str(),
                (std::string("should be ") + std::to_string(expectedQuantOutType) +
                 " for weight dataType " + Ops::Base::ToString(w1DataType)).c_str());
            return GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckCombineQuantModeAttr(const int64_t *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "combineQuantMode");
        return GRAPH_FAILED;
    }
    if (*ptr != 0) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "combineQuantMode",
            std::to_string(*ptr).c_str(), "0");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 num_max_token_per_rank，需要从 x shape 获取 bs
static ge::graphStatus CheckNumMaxTokensPerRankAttr(gert::TilingContext *context, const int64_t *ptr)
{
    const char *nodeName = context->GetNodeName();
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank");
        return GRAPH_FAILED;
    }
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    if (xStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "x");
        return GRAPH_FAILED;
    }
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    if (*ptr > 0) {
        if (*ptr < bs) {
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank",
                std::to_string(*ptr).c_str(),
                (std::string("should be >= bs(") + std::to_string(bs) + ")").c_str());
            return GRAPH_FAILED;
        }
        if (*ptr < MIN_BS || *ptr > MAX_BS) {
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank",
                std::to_string(*ptr).c_str(),
                (std::string("should in [") + std::to_string(MIN_BS) + ", " +
                 std::to_string(MAX_BS) + "]").c_str());
            return GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationAttr(const char *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "activation");
        return GRAPH_FAILED;
    }
    std::string activationStr(ptr);
    if (activationStr != "swiglu") {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "activation",
            ptr, "only support 'swiglu'");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationClampAttr(const float *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "activationClamp");
        return GRAPH_FAILED;
    }
    if (*ptr < 0 || std::isnan(*ptr)) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "activationClamp",
            std::to_string(*ptr).c_str(), ">= 0 and not NAN");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckActivationOutDtypeAttr(const int64_t *ptr)
{
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "activationOutDtype");
        return GRAPH_FAILED;
    }
    if (*ptr != static_cast<int64_t>(ge::DT_UNDEFINED)) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "activationOutDtype",
            std::to_string(*ptr).c_str(),
            (std::string("should be default value ") + std::to_string(static_cast<int64_t>(ge::DT_UNDEFINED))).c_str());
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTransposeWeightAttr(const bool *w1, const bool *w2)
{
    if (w1 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "transposeWeight1");
        return GRAPH_FAILED;
    }
    if (w2 == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("MegaMoe", "transposeWeight2");
        return GRAPH_FAILED;
    }
    if (*w1 != *w2) {
        OP_LOGE_FOR_INVALID_VALUE("MegaMoe", "transposeWeight1/transposeWeight2",
            (std::to_string(static_cast<int>(*w1)) + " and " + std::to_string(static_cast<int>(*w2))).c_str(),
            "must be the same");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 weight1_interleave，需要从 weight1 tensor 获取 N
static ge::graphStatus CheckWeight1InterleaveAttr(gert::TilingContext *context, const int64_t *ptr)
{
    const char *nodeName = context->GetNodeName();
    if (ptr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "weight1Interleave");
        return GRAPH_FAILED;
    }
    if (*ptr < 0) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "weight1Interleave",
            std::to_string(*ptr).c_str(), ">= 0");
        return GRAPH_FAILED;
    }
    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    if (w1Tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "weight1");
        return GRAPH_FAILED;
    }
    uint32_t w1TensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(w1TensorDims - 1);
    uint32_t maxInterleave = N / 2;
    if (static_cast<uint32_t>(*ptr) > maxInterleave) {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "weight1Interleave",
            std::to_string(*ptr).c_str(),
            (std::string("should in [0, ") + std::to_string(maxInterleave) + "]").c_str());
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckAttrAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData& info)
{
    const char *opName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "attrs");
        return ge::GRAPH_FAILED;
    }

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_RECV_TOKEN_NUM_INDEX);
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_DISPATCH_QUANT_MODE_INDEX);
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_DISPATCH_QUANT_OUT_TYPE_INDEX);
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_COMBINE_QUANT_MODE_INDEX);
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TOKEN_PER_RANK_INDEX);
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_ACTIVATION_INDEX));
    auto activationClampPtr = attrs->GetAttrPointer<float>(ATTR_ACTIVATION_CLAMP_INDEX);
    auto activationOutDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_OUT_DTYPE_INDEX);
    auto transposeWeight1Ptr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WEIGHT1_INDEX);
    auto transposeWeight2Ptr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WEIGHT2_INDEX);
    auto weight1InterleavePtr = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT1_INTERLEAVE_INDEX);

    // 1. moe_expert_num
    if (CheckMoeExpertNumAttr(moeExpertNumPtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckMoeExpertNumAttr failed.");
        return GRAPH_FAILED;
    }

    // 2. ep_world_size
    if (CheckEpWorldSizeAttr(epWorldSizePtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckEpWorldSizeAttr failed.");
        return GRAPH_FAILED;
    }

    // 3. ccl_buffer_size
    if (CheckCclBufferSizeAttr(cclBufferSizePtr, info, context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckCclBufferSizeAttr failed.");
        return GRAPH_FAILED;
    }

    // 4. max_recv_token_num
    if (CheckMaxRecvTokenNumAttr(maxRecvTokenNumPtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckMaxRecvTokenNumAttr failed.");
        return GRAPH_FAILED;
    }

    // 5-6. dispatch_quant_mode && dispatch_quant_out_dtype
    if (CheckDispatchQuantAttrs(context, dispatchQuantModePtr,
        dispatchQuantOutDtypePtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckDispatchQuantAttrs failed.");
        return GRAPH_FAILED;
    }

    // 7. combine_quant_mode
    if (CheckCombineQuantModeAttr(combineQuantModePtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckCombineQuantModeAttr failed.");
        return GRAPH_FAILED;
    }

    // 8. num_max_token_per_rank
    if (CheckNumMaxTokensPerRankAttr(context, numMaxTokensPerRankPtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckNumMaxTokensPerRankAttr failed.");
        return GRAPH_FAILED;
    }

    // 9. activation
    if (CheckActivationAttr(activationPtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckActivationAttr failed.");
        return GRAPH_FAILED;
    }

    // 10. activation_clamp
    if (CheckActivationClampAttr(activationClampPtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckActivationClampAttr failed.");
        return GRAPH_FAILED;
    }

    // 11. activation_out_dtype
    if (CheckActivationOutDtypeAttr(activationOutDtypePtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckActivationOutDtypeAttr failed.");
        return GRAPH_FAILED;
    }

    // 12. transpose_weight1/2
    if (CheckTransposeWeightAttr(transposeWeight1Ptr, transposeWeight2Ptr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckTransposeWeightAttr failed.");
        return GRAPH_FAILED;
    }

    // 13. weight1_interleave
    if (CheckWeight1InterleaveAttr(context, weight1InterleavePtr) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckWeight1InterleaveAttr failed.");
        return GRAPH_FAILED;
    }

    info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    info.cclBufferSize = static_cast<uint32_t>(*cclBufferSizePtr);
    info.maxRecvTokenNum = static_cast<uint32_t>(*maxRecvTokenNumPtr);
    info.dispatchQuantMode = static_cast<uint32_t>(*dispatchQuantModePtr);
    info.dispatchQuantOutDtype = static_cast<uint32_t>(*dispatchQuantOutDtypePtr);
    info.combineQuantMode = static_cast<uint32_t>(*combineQuantModePtr);
    info.numMaxTokensPerRank = static_cast<uint32_t>(*numMaxTokensPerRankPtr);
    info.activationOutDtype = activationOutDtypePtr != nullptr ?
        static_cast<uint32_t>(*activationOutDtypePtr) :
        static_cast<uint32_t>(ge::DT_UNDEFINED);
    info.isTransposeW1 = *transposeWeight1Ptr ? 1U : 0U;
    info.isTransposeW2 = *transposeWeight2Ptr ? 1U : 0U;
    info.weight1Interleave = static_cast<uint32_t>(*weight1InterleavePtr);

    info.worldSize = info.epWorldSize;

    OP_LOGD(K_INNER_DEBUG, "moeExpertNum=%u", info.moeExpertNum);
    OP_LOGD(K_INNER_DEBUG, "epWorldSize=%u", info.epWorldSize);
    OP_LOGD(K_INNER_DEBUG, "cclBufferSize=%u", info.cclBufferSize);
    OP_LOGD(K_INNER_DEBUG, "maxRecvTokenNum=%u", info.maxRecvTokenNum);
    OP_LOGD(K_INNER_DEBUG, "dispatchQuantMode=%u", info.dispatchQuantMode);
    OP_LOGD(K_INNER_DEBUG, "dispatchQuantOutDtype=%u", info.dispatchQuantOutDtype);
    OP_LOGD(K_INNER_DEBUG, "combineQuantMode=%u", info.combineQuantMode);
    OP_LOGD(K_INNER_DEBUG, "numMaxTokensPerRank=%u", info.numMaxTokensPerRank);
    OP_LOGD(K_INNER_DEBUG, "activationOutDtype=%u", info.activationOutDtype);
    OP_LOGD(K_INNER_DEBUG, "transposeWeight1=%u", info.isTransposeW1);
    OP_LOGD(K_INNER_DEBUG, "transposeWeight2=%u", info.isTransposeW2);
    OP_LOGD(K_INNER_DEBUG, "weight1Interleave=%u", info.weight1Interleave);
    OP_LOGD(K_INNER_DEBUG, "worldSize=%u", info.worldSize);

    return ge::GRAPH_SUCCESS;
}

// 获取指定 DynamicInput 的 TensorList 长度
static uint32_t GetDynamicInputTensorListLen(gert::TilingContext *context, uint32_t inputIndex)
{
    uint32_t listLen = 0;
    while (context->GetDynamicInputTensor(inputIndex, ++listLen) != nullptr) {}
    return listLen;
}

// 校验 weight_scales1/weight_scales2 输入（可选），quant 权重时必选
static ge::graphStatus CheckWeightScaleInput(gert::TilingContext *context, uint32_t inputIndex,
    const char *inputName, uint32_t listLen,
    uint32_t expertPerRank, int64_t dim1Expected, const char *dim1Name)
{
    const char *opName = context->GetNodeName();
    auto wScaleTensor = context->GetDynamicInputTensor(inputIndex, 0);
    if (wScaleTensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, inputName);
        return GRAPH_FAILED;
    }

    auto wScaleDesc = context->GetDynamicInputDesc(inputIndex, 0);
    if (wScaleDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, inputName);
        return GRAPH_FAILED;
    }
    if (wScaleDesc->GetDataType() != ge::DT_UINT64) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, inputName,
            Ops::Base::ToString(wScaleDesc->GetDataType()).c_str(), "UINT64");
        return GRAPH_FAILED;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(wScaleDesc->GetStorageFormat())) != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, inputName,
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(wScaleDesc->GetStorageFormat()))).c_str(),
            "ND");
        return GRAPH_FAILED;
    }

    uint32_t scaleListLen = GetDynamicInputTensorListLen(context, inputIndex);

    if (scaleListLen != listLen) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(opName, inputName,
            std::to_string(scaleListLen).c_str(), std::to_string(listLen).c_str());
        return GRAPH_FAILED;
    }

    if (scaleListLen == 1) {
        // 单个scale tensor，检查维度
        uint32_t scaleDims = wScaleTensor->GetOriginShape().GetDimNum();
        if (scaleDims != TWO_DIMS && scaleDims != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, inputName,
                (std::to_string(scaleDims) + "D").c_str(), "1D or 2D");
            return GRAPH_FAILED;
        }

        if (scaleDims == TWO_DIMS) {
            int64_t scaleDim0 = wScaleTensor->GetStorageShape().GetDim(0);
            int64_t scaleDim1 = wScaleTensor->GetStorageShape().GetDim(1);
            if (scaleDim0 != expertPerRank) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, inputName,
                    (std::string("dim0=") + std::to_string(scaleDim0)).c_str(),
                    (std::string("The dim0 of ") + inputName + " must equal to expertPerRank(" +
                     std::to_string(expertPerRank) + ").").c_str());
                return GRAPH_FAILED;
            }
            if (scaleDim1 != dim1Expected) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, inputName,
                    (std::string("dim1=") + std::to_string(scaleDim1)).c_str(),
                    (std::string("The dim1 of ") + inputName + " must equal to " + dim1Name + "(" +
                     std::to_string(dim1Expected) + ").").c_str());
                return GRAPH_FAILED;
            }
        }
    } else {
        // 多个scale tensor，逐个检查
        for (uint32_t i = 0; i < scaleListLen; i++) {
            auto sTensorI = context->GetDynamicInputTensor(inputIndex, i);
            if (sTensorI == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(opName, (std::string(inputName) + "[" + std::to_string(i) + "]").c_str());
                return GRAPH_FAILED;
            }
            if (sTensorI->GetOriginShape().GetDimNum() != ONE_DIM) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, (std::string(inputName) + "[" + std::to_string(i) + "]").c_str(),
                    (std::to_string(sTensorI->GetOriginShape().GetDimNum()) + "D").c_str(), "1D");
                return GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 bias1/bias2 输入（可选），INT4 权重时必选
static ge::graphStatus CheckBiasInput(gert::TilingContext *context, uint32_t inputIndex,
    const char *inputName, uint32_t listLen,
    int64_t dim0Expected, const char *dim0Name)
{
    const char *opName = context->GetNodeName();
    const gert::StorageShape *biasStorageShape = context->GetDynamicInputShape(inputIndex, 0);
    if (biasStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, inputName);
        return GRAPH_FAILED;
    }
    auto biasTensor = context->GetDynamicInputTensor(inputIndex, 0);
    if (biasTensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, inputName);
        return GRAPH_FAILED;
    }

    auto biasDesc = context->GetDynamicInputDesc(inputIndex, 0);
    if (biasDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, inputName);
        return GRAPH_FAILED;
    }
    if (biasDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, inputName,
            Ops::Base::ToString(biasDesc->GetDataType()).c_str(), "FP32");
        return GRAPH_FAILED;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat())) != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, inputName,
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(biasDesc->GetStorageFormat()))).c_str(),
            "ND");
        return GRAPH_FAILED;
    }

    uint32_t biasListLen = GetDynamicInputTensorListLen(context, inputIndex);
    if (biasListLen != listLen) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(opName, inputName,
            std::to_string(biasListLen).c_str(), std::to_string(listLen).c_str());
        return GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < biasListLen; i++) {
        auto bTensorI = context->GetDynamicInputTensor(inputIndex, i);
        if (bTensorI == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, (std::string(inputName) + "[" + std::to_string(i) + "]").c_str());
            return GRAPH_FAILED;
        }
        if (bTensorI->GetOriginShape().GetDimNum() != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, (std::string(inputName) + "[" + std::to_string(i) + "]").c_str(),
                (std::to_string(bTensorI->GetOriginShape().GetDimNum()) + "D").c_str(), "1D");
            return GRAPH_FAILED;
        }
        uint32_t bDim0 = bTensorI->GetStorageShape().GetDim(0);
        if (bDim0 != static_cast<uint32_t>(dim0Expected)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName,
                (std::string(inputName) + "[" + std::to_string(i) + "]").c_str(),
                (std::string("dim0=") + std::to_string(bDim0)).c_str(),
                (std::string("The dim0 of ") + inputName + "[" + std::to_string(i) + "] must equal to " +
                 dim0Name + "(" + std::to_string(dim0Expected) + ").").c_str());
            return GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 weight1 动态输入，返回 N / listLen / expertPerRank / w1DataType / w1Format
static ge::graphStatus CheckWeight1Input(gert::TilingContext *context, int64_t hiddenSize,
    uint32_t &outN, uint32_t &outListLen, uint32_t &outExpertPerRank,
    ge::DataType &outW1DataType, ge::Format &outW1Format)
{
    const char *opName = context->GetNodeName();
    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    if (w1Tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "weight1");
        return GRAPH_FAILED;
    }

    auto w1Desc = context->GetDynamicInputDesc(WEIGHT1_INDEX, 0);
    if (w1Desc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "weight1");
        return GRAPH_FAILED;
    }
    ge::DataType w1DataType = w1Desc->GetDataType();
    if (w1DataType != ge::DT_BF16 && w1DataType != ge::DT_FLOAT16 &&
        w1DataType != ge::DT_INT8 && w1DataType != ge::DT_INT4) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "weight1",
            Ops::Base::ToString(w1DataType).c_str(), "BF16, FP16, INT8 or INT4");
        return GRAPH_FAILED;
    }
    ge::Format w1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(w1Desc->GetStorageFormat()));
    if (w1Format != ge::FORMAT_ND && w1Format != ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "weight1",
            Ops::Base::ToString(w1Format).c_str(), "ND or FRACTAL_NZ");
        return GRAPH_FAILED;
    }

    uint32_t w1TensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(w1TensorDims - 1);
    if (N % 2 != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "weight1",
            (std::string("last_dim=") + std::to_string(N)).c_str(), "The last dim should be even.");
        return GRAPH_FAILED;
    }

    uint32_t listLen = GetDynamicInputTensorListLen(context, WEIGHT1_INDEX);

    uint32_t expertPerRank;
    if (listLen == 1) {
        // 场景1：传入一个 3 维 tensor，shape 为 (num_experts_per_rank, hidden_size, N)
        if (w1TensorDims != THREE_DIMS) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "weight1",
                (std::to_string(w1TensorDims) + "D").c_str(), "3D");
            return GRAPH_FAILED;
        }
        expertPerRank = w1Tensor->GetStorageShape().GetDim(0);
        int64_t w1Dim1 = w1Tensor->GetStorageShape().GetDim(1);
        if (w1Dim1 != hiddenSize) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "weight1",
                (std::string("dim1=") + std::to_string(w1Dim1)).c_str(),
                (std::string("The dim1 of weight1 must equal to hidden_size(") +
                 std::to_string(hiddenSize) + ").").c_str());
            return GRAPH_FAILED;
        }
    } else {
        // 场景2：传入多个 2 维 tensor，每个 shape 为 (hidden_size, N)
        expertPerRank = listLen;
        for (uint32_t i = 0; i < listLen; i++) {
            auto wTensorI = context->GetDynamicInputTensor(WEIGHT1_INDEX, i);
            if (wTensorI == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(opName, (std::string("weight1[") + std::to_string(i) + "]").c_str());
                return GRAPH_FAILED;
            }
            if (wTensorI->GetOriginShape().GetDimNum() != TWO_DIMS) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, (std::string("weight1[") + std::to_string(i) + "]").c_str(),
                    (std::to_string(wTensorI->GetOriginShape().GetDimNum()) + "D").c_str(), "2D");
                return GRAPH_FAILED;
            }
            int64_t w1IDim0 = wTensorI->GetStorageShape().GetDim(0);
            if (w1IDim0 != hiddenSize) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName,
                    (std::string("weight1[") + std::to_string(i) + "]").c_str(),
                    (std::string("dim0=") + std::to_string(w1IDim0)).c_str(),
                    (std::string("The dim0 of weight1[") + std::to_string(i) +
                     "] must equal to hidden_size(" + std::to_string(hiddenSize) + ").").c_str());
                return GRAPH_FAILED;
            }
        }
    }

    outN = N;
    outListLen = listLen;
    outExpertPerRank = expertPerRank;
    outW1DataType = w1DataType;
    outW1Format = w1Format;
    return ge::GRAPH_SUCCESS;
}

// 校验 weight2 动态输入，与 weight1 做交叉校验
static ge::graphStatus CheckWeight2Input(gert::TilingContext *context, int64_t hiddenSize,
    uint32_t N, uint32_t listLen, uint32_t expertPerRank,
    ge::DataType w1DataType, ge::Format w1Format,
    ge::DataType &outW2DataType)
{
    const char *opName = context->GetNodeName();
    auto w2Tensor = context->GetDynamicInputTensor(WEIGHT2_INDEX, 0);
    if (w2Tensor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "weight2");
        return GRAPH_FAILED;
    }

    auto w2Desc = context->GetDynamicInputDesc(WEIGHT2_INDEX, 0);
    if (w2Desc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "weight2");
        return GRAPH_FAILED;
    }
    ge::DataType w2DataType = w2Desc->GetDataType();
    if (w2DataType != ge::DT_BF16 && w2DataType != ge::DT_FLOAT16 &&
        w2DataType != ge::DT_INT8 && w2DataType != ge::DT_INT4) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "weight2",
            Ops::Base::ToString(w2DataType).c_str(), "BF16, FP16, INT8 or INT4");
        return GRAPH_FAILED;
    }

    // weight1 和 weight2 数据类型必须一致
    if (w1DataType != w2DataType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, "weight1 and weight2",
            (Ops::Base::ToString(w1DataType) + " and " + Ops::Base::ToString(w2DataType)).c_str(),
            "The dtypes of weight1 and weight2 must be the same.");
        return GRAPH_FAILED;
    }
    ge::Format w2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(w2Desc->GetStorageFormat()));
    if (w2Format != ge::FORMAT_ND && w2Format != ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "weight2",
            Ops::Base::ToString(w2Format).c_str(), "ND or FRACTAL_NZ");
        return GRAPH_FAILED;
    }
    if (w1Format != w2Format) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, "weight1 and weight2",
            (Ops::Base::ToString(w1Format) + " and " + Ops::Base::ToString(w2Format)).c_str(),
            "The formats of weight1 and weight2 must be the same.");
        return GRAPH_FAILED;
    }

    uint32_t w2TensorDims = w2Tensor->GetOriginShape().GetDimNum();

    uint32_t w2ListLen = GetDynamicInputTensorListLen(context, WEIGHT2_INDEX);
    if (w2ListLen != listLen) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(opName, "weight2",
            std::to_string(w2ListLen).c_str(), std::to_string(listLen).c_str());
        return GRAPH_FAILED;
    }

    uint32_t n2 = N / 2;
    if (listLen == 1) {
        if (w2TensorDims != THREE_DIMS) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "weight2",
                (std::to_string(w2TensorDims) + "D").c_str(), "3D");
            return GRAPH_FAILED;
        }
        int64_t w2Dim0 = w2Tensor->GetStorageShape().GetDim(0);
        int64_t w2Dim1 = w2Tensor->GetStorageShape().GetDim(1);
        int64_t w2Dim2 = w2Tensor->GetStorageShape().GetDim(2);
        if (w2Dim0 != expertPerRank) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "weight2",
                (std::string("dim0=") + std::to_string(w2Dim0)).c_str(),
                (std::string("The dim0 of weight2 must equal to expertPerRank(") +
                 std::to_string(expertPerRank) + ").").c_str());
            return GRAPH_FAILED;
        }
        if (w2Dim1 != n2) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "weight2",
                (std::string("dim1=") + std::to_string(w2Dim1)).c_str(),
                (std::string("The dim1 of weight2 must equal to N/2(") + std::to_string(n2) + ").").c_str());
            return GRAPH_FAILED;
        }
        if (w2Dim2 != hiddenSize) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "weight2",
                (std::string("dim2=") + std::to_string(w2Dim2)).c_str(),
                (std::string("The dim2 of weight2 must equal to hidden_size(") +
                 std::to_string(hiddenSize) + ").").c_str());
            return GRAPH_FAILED;
        }
    } else {
        for (uint32_t i = 0; i < listLen; i++) {
            auto wTensorI = context->GetDynamicInputTensor(WEIGHT2_INDEX, i);
            if (wTensorI == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(opName, (std::string("weight2[") + std::to_string(i) + "]").c_str());
                return GRAPH_FAILED;
            }
            if (wTensorI->GetOriginShape().GetDimNum() != TWO_DIMS) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, (std::string("weight2[") + std::to_string(i) + "]").c_str(),
                    (std::to_string(wTensorI->GetOriginShape().GetDimNum()) + "D").c_str(), "2D");
                return GRAPH_FAILED;
            }
            int64_t w2IDim0 = wTensorI->GetStorageShape().GetDim(0);
            int64_t w2IDim1 = wTensorI->GetStorageShape().GetDim(1);
            if (w2IDim0 != n2) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName,
                    (std::string("weight2[") + std::to_string(i) + "]").c_str(),
                    (std::string("dim0=") + std::to_string(w2IDim0)).c_str(),
                    (std::string("The dim0 of weight2[") + std::to_string(i) +
                     "] must equal to N/2(" + std::to_string(n2) + ").").c_str());
                return GRAPH_FAILED;
            }
            if (w2IDim1 != hiddenSize) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName,
                    (std::string("weight2[") + std::to_string(i) + "]").c_str(),
                    (std::string("dim1=") + std::to_string(w2IDim1)).c_str(),
                    (std::string("The dim1 of weight2[") + std::to_string(i) +
                     "] must equal to hidden_size(" + std::to_string(hiddenSize) + ").").c_str());
                return GRAPH_FAILED;
            }
        }
    }

    outW2DataType = w2DataType;
    return ge::GRAPH_SUCCESS;
}

// 校验 context 输入
static ge::graphStatus CheckContextInput(gert::TilingContext *context,
    const gert::StorageShape *contextStorageShape)
{
    const char *opName = context->GetNodeName();
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    if (contextDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "context");
        return GRAPH_FAILED;
    }
    if (contextDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "context",
            Ops::Base::ToString(contextDesc->GetDataType()).c_str(), "INT32");
        return GRAPH_FAILED;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "context",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat()))).c_str(),
            "ND");
        return GRAPH_FAILED;
    }
    if (contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "context",
            (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 x 输入，返回 bs 和 hiddenSize
static ge::graphStatus CheckXInput(gert::TilingContext *context,
    const gert::StorageShape *xStorageShape,
    int64_t &outBs, int64_t &outHiddenSize)
{
    const char *opName = context->GetNodeName();
    if (xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "x",
            (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t hiddenSize = xStorageShape->GetStorageShape().GetDim(1);

    if (bs < MIN_BS || bs > MAX_BS) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "bs",
            std::to_string(bs).c_str(),
            (std::string("should in [") + std::to_string(MIN_BS) + ", " +
             std::to_string(MAX_BS) + "]").c_str());
        return GRAPH_FAILED;
    }
    if (hiddenSize < MIN_HIDDEN_SIZE || hiddenSize > MAX_HIDDEN_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "hiddenSize",
            std::to_string(hiddenSize).c_str(),
            (std::string("should in [") + std::to_string(MIN_HIDDEN_SIZE) + ", " +
             std::to_string(MAX_HIDDEN_SIZE) + "]").c_str());
        return GRAPH_FAILED;
    }
    if (hiddenSize % HIDDEN_SIZE_ALIGN != 0) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "hiddenSize",
            std::to_string(hiddenSize).c_str(),
            (std::string("should be ") + std::to_string(HIDDEN_SIZE_ALIGN) + " aligned").c_str());
        return GRAPH_FAILED;
    }

    auto xDesc = context->GetInputDesc(X_INDEX);
    if (xDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "x");
        return GRAPH_FAILED;
    }
    ge::DataType xDataType = xDesc->GetDataType();
    if (xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "x",
            Ops::Base::ToString(xDataType).c_str(), "FLOAT16 or BF16");
        return GRAPH_FAILED;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "x",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat()))).c_str(),
            "ND");
        return GRAPH_FAILED;
    }

    outBs = bs;
    outHiddenSize = hiddenSize;
    return ge::GRAPH_SUCCESS;
}

// 校验 topk_ids 输入，返回 topK
static ge::graphStatus CheckTopkIdsInput(gert::TilingContext *context,
    const gert::StorageShape *topkIdsStorageShape, int64_t bs,
    int64_t &outTopK)
{
    const char *opName = context->GetNodeName();
    if (topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "topk_ids",
            (std::to_string(topkIdsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }

    int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    if (topkIdsDim0 != bs) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "topk_ids",
            (std::string("dim0=") + std::to_string(topkIdsDim0)).c_str(),
            (std::string("The dim0 of topk_ids must equal to x's dim0(") +
             std::to_string(bs) + ").").c_str());
        return GRAPH_FAILED;
    }
    if (topK < MIN_TOPK || topK > MAX_TOPK) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "topK",
            std::to_string(topK).c_str(),
            (std::string("should in [") + std::to_string(MIN_TOPK) + ", " +
             std::to_string(MAX_TOPK) + "]").c_str());
        return GRAPH_FAILED;
    }

    auto topkIdsDesc = context->GetInputDesc(TOPK_IDS_INDEX);
    if (topkIdsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "topk_ids");
        return GRAPH_FAILED;
    }
    if (topkIdsDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "topk_ids",
            Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str(), "INT32");
        return GRAPH_FAILED;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "topk_ids",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat()))).c_str(),
            "ND");
        return GRAPH_FAILED;
    }

    outTopK = topK;
    return ge::GRAPH_SUCCESS;
}

// 校验 topk_weights 输入
static ge::graphStatus CheckTopkWeightsInput(gert::TilingContext *context,
    const gert::StorageShape *topkWeightsStorageShape, int64_t bs, int64_t topK)
{
    const char *opName = context->GetNodeName();
    if (topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "topk_weights",
            (std::to_string(topkWeightsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }

    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);

    if (topkWeightsDim0 != bs) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "topk_weights",
            (std::string("dim0=") + std::to_string(topkWeightsDim0)).c_str(),
            (std::string("The dim0 of topk_weights must equal to x's dim0(") +
             std::to_string(bs) + ").").c_str());
        return GRAPH_FAILED;
    }
    if (topkWeightsDim1 != topK) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "topk_weights",
            (std::string("dim1=") + std::to_string(topkWeightsDim1)).c_str(),
            (std::string("The dim1 of topk_weights must equal to topk_ids's dim1(") +
             std::to_string(topK) + ").").c_str());
        return GRAPH_FAILED;
    }

    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    if (topkWeightsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "topk_weights");
        return GRAPH_FAILED;
    }
    ge::DataType topkWeightsDataType = topkWeightsDesc->GetDataType();
    if (topkWeightsDataType != ge::DT_FLOAT && topkWeightsDataType != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "topk_weights",
            Ops::Base::ToString(topkWeightsDataType).c_str(), "FLOAT or BF16");
        return GRAPH_FAILED;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "topk_weights",
            Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat()))).c_str(),
            "ND");
        return GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 校验 x_active_mask 输入（可选），返回 isActiveMask
static ge::graphStatus CheckXActiveMaskInput(gert::TilingContext *context, int64_t bs,
    bool &outIsActiveMask)
{
    const char *opName = context->GetNodeName();
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        if (xActiveMaskDimNums != ONE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x_active_mask",
                (std::to_string(xActiveMaskDimNums) + "D").c_str(),
                "The shape dim of x_active_mask must be 1D. 2D is not supported now.");
            return GRAPH_FAILED;
        }
        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        if (xActiveMaskDim0 != bs) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "x_active_mask",
                (std::string("dim0=") + std::to_string(xActiveMaskDim0)).c_str(),
                (std::string("The dim0 of x_active_mask must equal to x's dim0(") +
                 std::to_string(bs) + ").").c_str());
            return GRAPH_FAILED;
        }
        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        if (xActiveMaskDesc == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "x_active_mask");
            return GRAPH_FAILED;
        }
        if (xActiveMaskDesc->GetDataType() != ge::DT_INT8) {
            OP_LOGE_FOR_INVALID_DTYPE(opName, "x_active_mask",
                Ops::Base::ToString(xActiveMaskDesc->GetDataType()).c_str(), "INT8");
            return GRAPH_FAILED;
        }
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT(opName, "x_active_mask",
                Ops::Base::ToString(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat()))).c_str(),
                "ND");
            return GRAPH_FAILED;
        }
    }
    outIsActiveMask = isActiveMask;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckShapeAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData &info)
{
    const char *opName = context->GetNodeName();
    // ==================== 1. 输入张量非空校验 ====================
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(TOPK_IDS_INDEX);
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    if (contextStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "context");
        return GRAPH_FAILED;
    }
    if (xStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "x");
        return GRAPH_FAILED;
    }
    if (topkIdsStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "topk_ids");
        return GRAPH_FAILED;
    }
    if (topkWeightsStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "topk_weights");
        return GRAPH_FAILED;
    }

    // ==================== 2. context 输入校验 ====================
    if (CheckContextInput(context, contextStorageShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckContextInput failed.");
        return GRAPH_FAILED;
    }

    // ==================== 3. x 输入校验 ====================
    int64_t bs = 0;
    int64_t hiddenSize = 0;
    if (CheckXInput(context, xStorageShape, bs, hiddenSize) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckXInput failed.");
        return GRAPH_FAILED;
    }

    // ==================== 4. topk_ids 输入校验 ====================
    int64_t topK = 0;
    if (CheckTopkIdsInput(context, topkIdsStorageShape, bs, topK) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckTopkIdsInput failed.");
        return GRAPH_FAILED;
    }

    // ==================== 5. topk_weights 输入校验 ====================
    if (CheckTopkWeightsInput(context, topkWeightsStorageShape, bs, topK) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckTopkWeightsInput failed.");
        return GRAPH_FAILED;
    }

    // ==================== 6. weight1 输入校验 ====================
    uint32_t N = 0;
    uint32_t listLen = 0;
    uint32_t expertPerRank = 0;
    ge::DataType w1DataType = ge::DT_UNDEFINED;
    ge::Format w1Format = ge::FORMAT_RESERVED;
    if (CheckWeight1Input(context, hiddenSize,
        N, listLen, expertPerRank, w1DataType, w1Format) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckWeight1Input failed.");
        return GRAPH_FAILED;
    }

    info.isQuantRouting = (w1DataType == ge::DT_FLOAT16 || w1DataType == ge::DT_BF16) ? 0U : 1U;
    info.isW4A8 = (w1DataType == ge::DT_INT32 || w1DataType == ge::DT_INT4) ? 1U : 0U;

    // expertPerRank 范围校验
    if (expertPerRank < MIN_EXPERT_PER_RANK || expertPerRank > MAX_EXPERT_PER_RANK) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "expertPerRank",
            std::to_string(expertPerRank).c_str(),
            (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
             std::to_string(MAX_EXPERT_PER_RANK) + "]").c_str());
        return GRAPH_FAILED;
    }

    // ==================== 7. weight2 输入校验 ====================
    ge::DataType w2DataType = ge::DT_UNDEFINED;
    if (CheckWeight2Input(context, hiddenSize,
        N, listLen, expertPerRank, w1DataType, w1Format,
        w2DataType) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckWeight2Input failed.");
        return GRAPH_FAILED;
    }

    // ==================== 量化权重必须提供 scale 校验 ====================
    if (w1DataType == ge::DT_INT8 || w1DataType == ge::DT_INT4) {
        // ==================== 8. weight_scales1 输入校验（可选） ====================
        if (CheckWeightScaleInput(context, WEIGHT_SCALES1_INDEX, "weight_scales1",
            listLen, expertPerRank, N, "N") != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName, "CheckWeightScaleInput for weight_scales1 failed.");
            return GRAPH_FAILED;
        }

        // ==================== 9. weight_scales2 输入校验（可选） ====================
        if (CheckWeightScaleInput(context, WEIGHT_SCALES2_INDEX, "weight_scales2",
            listLen, expertPerRank, hiddenSize, "hidden_size") != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName, "CheckWeightScaleInput for weight_scales2 failed.");
            return GRAPH_FAILED;
        }
    }

    if (w1DataType == ge::DT_INT4) {
        // ==================== 10. bias1 输入校验 ====================
        if (CheckBiasInput(context, BIAS1_INDEX, "bias1",
            listLen, N, "N") != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName, "CheckBiasInput for bias1 failed.");
            return GRAPH_FAILED;
        }

        // ==================== 11. bias2 输入校验 ====================
        if (CheckBiasInput(context, BIAS2_INDEX, "bias2",
            listLen, hiddenSize, "hiddenSize") != ge::GRAPH_SUCCESS) {
            OP_LOGE(opName, "CheckBiasInput for bias2 failed.");
            return GRAPH_FAILED;
        }
    }

    // ==================== 12. x_active_mask 输入校验（可选） ====================
    bool isActiveMask = false;
    if (CheckXActiveMaskInput(context, bs, isActiveMask) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "CheckXActiveMaskInput failed.");
        return GRAPH_FAILED;
    }

    // ==================== 13. scales 输入校验（可选，预留） ====================
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    if (scalesStorageShape != nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "scales");
        return GRAPH_FAILED;
    }

    // ==================== 14. 设置 tiling data ====================
    info.M = static_cast<uint32_t>(bs);
    info.N = N;
    info.K = static_cast<uint32_t>(hiddenSize);
    info.expertPerRank = expertPerRank;
    info.topK = static_cast<uint32_t>(topK);
    info.listLen = listLen;

    OP_LOGD(K_INNER_DEBUG, "bs=%u", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%u", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%u", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%u", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%u", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%u", info.listLen);
    OP_LOGD(K_INNER_DEBUG, "isActiveMask=%d", static_cast<int>(isActiveMask));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3GetPlatformInfoAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

// 表示通信亲和内存布局算法，当前版本仅支持 ""。
static ge::graphStatus MegaMoeA2A3CommAlg(const gert::TilingContext *context)
{
    const char *opName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "attrs");
        return ge::GRAPH_FAILED;
    }
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    if (commAlg == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "commAlg");
        return ge::GRAPH_FAILED;
    }

    if (strlen(commAlg) > 0) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "commAlg", commAlg, "empty string");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckHcclBuffSize(const gert::TilingContext *context,
    MegaMoeA2A3TilingData &info)
{
    const char *opName = context->GetNodeName();
    int64_t cclBufferSize = info.cclBufferSize;
    OP_LOGD(opName, "cclBufferSize = %ld Bytes (%ld MB).",
        cclBufferSize, ops::CeilDiv(cclBufferSize, static_cast<int64_t>(MB_SIZE)));
    if (cclBufferSize == 0) {
        OP_LOGW(opName, "cclBufferSize is 0, skip validation.");
        return ge::GRAPH_SUCCESS;
    }

    int64_t bs = info.M;
    int64_t h = info.K;
    int64_t topK = info.topK;
    int64_t epWorldSize = info.epWorldSize;
    int64_t expertPerRank = info.moeExpertNum / epWorldSize;

    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    if (yDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "y");
        return ge::GRAPH_FAILED;
    }
    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    int64_t leastCclBufferSize =
        (epWorldSize * ops::CeilAlign(epWorldSize * expertPerRank, ALIGN_128) * sizeof(int32_t)) + // tokenPerExpert
        (ops::CeilAlign(bs * topK * h, ALIGN_512) * sizeof(int8_t)) + // quantTokenScale
        (ops::CeilAlign(bs * h * topK * yDtypeSize, ALIGN_512)); // combineOut

    if (cclBufferSize < leastCclBufferSize) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "cclBufferSize",
            std::to_string(cclBufferSize).c_str(),
            (std::string("should equal or larger than ") + std::to_string(leastCclBufferSize)).c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(opName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckOutputTensor(gert::TilingContext *context, const MegaMoeA2A3TilingData &info)
{
    const char *opName = context->GetNodeName();
    // ==================== 1. y 输出校验 ====================
    const gert::StorageShape *yStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    if (yStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "y");
        return GRAPH_FAILED;
    }

    // 维度数校验：必须为 2 维
    if (yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "y",
            (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D");
        return GRAPH_FAILED;
    }

    int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);

    // Shape 一致性校验：y 的 shape 应该与 x 的 shape 一致 [M, K]
    if (yDim0 != static_cast<int64_t>(info.M)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "y",
            (std::string("dim0=") + std::to_string(yDim0)).c_str(),
            (std::string("The dim0 of y must equal to M(") + std::to_string(info.M) + ").").c_str());
        return GRAPH_FAILED;
    }
    if (yDim1 != static_cast<int64_t>(info.K)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "y",
            (std::string("dim1=") + std::to_string(yDim1)).c_str(),
            (std::string("The dim1 of y must equal to K(") + std::to_string(info.K) + ").").c_str());
        return GRAPH_FAILED;
    }

    // 数据类型校验
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    if (yDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "y");
        return GRAPH_FAILED;
    }
    ge::DataType yDataType = yDesc->GetDataType();
    if (yDataType != ge::DT_FLOAT16 && yDataType != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "y",
            Ops::Base::ToString(yDataType).c_str(), "FLOAT16 or BF16");
        return GRAPH_FAILED;
    }

    OP_LOGD(opName, "y dim0 = %ld", yDim0);
    OP_LOGD(opName, "y dim1 = %ld", yDim1);
    OP_LOGD(opName, "y dataType = %d", static_cast<int>(yDataType));

    // ==================== 2. expert_token_nums 输出校验 ====================
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    if (expertTokenNumsStorageShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "expert_token_nums");
        return GRAPH_FAILED;
    }

    // 维度数校验：必须为 1 维
    if (expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "expert_token_nums",
            (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D");
        return GRAPH_FAILED;
    }

    int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    // Shape 校验：expert_token_nums 的 dim0 应该等于 expertPerRank（本卡专家数量）
    if (expertTokenNumsDim0 != static_cast<int64_t>(info.expertPerRank)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "expert_token_nums",
            (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
            (std::string("The dim0 of expert_token_nums must equal to expertPerRank(") +
             std::to_string(info.expertPerRank) + ").").c_str());
        return GRAPH_FAILED;
    }

    // 数据类型校验
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    if (expertTokenNumsDesc == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "expert_token_nums");
        return GRAPH_FAILED;
    }
    if (expertTokenNumsDesc->GetDataType() != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(opName, "expert_token_nums",
            Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(), "INT32");
        return GRAPH_FAILED;
    }

    OP_LOGD(opName, "expert_token_nums dim0 = %ld", expertTokenNumsDim0);
    OP_LOGD(opName, "expert_token_nums dataType = %d", static_cast<int>(expertTokenNumsDesc->GetDataType()));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3TilingFuncImpl(gert::TilingContext *context)
{
    const char *opName = context->GetNodeName();
    // 涉及SyncAll，设置batch mode模式，所有核同时启动
    uint32_t batch_mode = 1U;
    auto ret = context->SetScheduleMode(batch_mode);

    // 1. tilingData
    MegaMoeTilingDataQuant *tilingData = context->GetTilingData<MegaMoeTilingDataQuant>();
    if (tilingData == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "tilingData");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(opName, "MegaMoeA2A3 get tilingData.");
    MegaMoeA2A3TilingData& info = tilingData->common;
    OP_LOGI(opName, "MegaMoeA2A3 get tilingData info.");

    if (MegaMoeA2A3CommAlg(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "MegaMoeA2A3 CheckCommAlg Failed");
        return ge::GRAPH_FAILED;
    }
    if (MegaMoeA2A3CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "MegaMoeA2A3 CheckShapeAndSetTiling Failed");
        return ge::GRAPH_FAILED;
    }
    if (MegaMoeA2A3CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "MegaMoeA2A3 CheckAttrAndSetTiling Failed");
        return ge::GRAPH_FAILED;
    }
    if (MegaMoeA2A3CheckOutputTensor(context, info) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "MegaMoeA2A3 CheckOutputTensor Failed");
        return ge::GRAPH_FAILED;
    }
    if (MegaMoeA2A3GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "MegaMoeA2A3 GetPlatformInfoAndSetTiling Failed");
        return ge::GRAPH_FAILED;
    }
    if (MegaMoeA2A3CheckHcclBuffSize(context, info) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName, "MegaMoeA2A3 CheckHcclBuffSize Failed");
        return ge::GRAPH_FAILED;
    }

    // 2. set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 3. set tiling key
    uint32_t quantMode = info.isQuantRouting != 0U
    ? MEGA_MOE_QUANT_MODE_PER_TENSOR
    : MEGA_MOE_QUANT_MODE_NO_QUANT;
    uint32_t quantOutType = MEGA_MOE_QUANT_OUT_TYPE_UNDEFINED;
    if (info.dispatchQuantOutDtype == static_cast<int32_t>(ge::DT_INT8)) {
        quantOutType = MEGA_MOE_QUANT_OUT_TYPE_INT8;
    }
    if (quantMode == MEGA_MOE_QUANT_MODE_NO_QUANT) {
        if (quantOutType != MEGA_MOE_QUANT_OUT_TYPE_UNDEFINED) {
            OP_LOGE_FOR_INVALID_VALUE(opName, "dispatch_quant_out_type",
                std::to_string(quantOutType).c_str(), "UNDEFINED in non-quant mode");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (quantOutType != MEGA_MOE_QUANT_OUT_TYPE_INT8) {
            OP_LOGE_FOR_INVALID_VALUE(opName, "dispatch_quant_out_type",
                std::to_string(quantOutType).c_str(), "INT8 in quant mode");
            return ge::GRAPH_FAILED;
        }
    }
    std::string socVersion = mc2tiling::GetSocVersion(context);
    uint32_t archCode = socVersion == "Ascend910B" ? SOC_ASCEND910B : SOC_ASCEND910_93;

    uint64_t tilingKey = GET_TPL_TILING_KEY(
        static_cast<int64_t>(info.isTransposeW1),
        static_cast<int64_t>(info.isTransposeW2),
        static_cast<int64_t>(quantMode),
        static_cast<int64_t>(quantOutType),
        static_cast<int64_t>(archCode));
    context->SetTilingKey(tilingKey);

    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = info.expertPerRank * info.worldSize;
    int64_t activeNum = info.M * info.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantModeRouting = 1;
    uint64_t initRoutingQuantTilingKey = 0;
    size_t initRoutingWorkspace = 0;

    if (info.isQuantRouting != 0U) {
        MegaMoeTilingDataQuant *quantTilingData = context->GetTilingData<MegaMoeTilingDataQuant>();
        if (tilingData == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "tilingData");
            return ge::GRAPH_FAILED;
        }
        quantTilingData->common = info;

        MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
        moeInitRoutingQuantV2TilingBase.DoTiling(info.M, info.K, info.topK,
            expertCapacity, expertNum, activeNum, dropPadMode,
            expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
            inuptXDtypeSize, quantModeRouting, scaleDim0, aivNum, ubSize);
        initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
        initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;
        quantTilingData->moeInitRoutingQuantV2TilingData = moeInitRoutingQuantV2TilingBase.quantTilingData;
        quantTilingData->common.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    } else {
        MegaMoeTilingDataNonQuant *nonQuantTilingData = context->GetTilingData<MegaMoeTilingDataNonQuant>();
        if (tilingData == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(opName, "tilingData");
            return ge::GRAPH_FAILED;
        }
        nonQuantTilingData->common = info;

        MoeInitRoutingV2TilingBase moeInitRoutingV2TilingBase;
        moeInitRoutingV2TilingBase.DoTiling(info.M, info.K, info.topK,
            expertCapacity, expertNum, activeNum, dropPadMode,
            expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag,
            inuptXDtypeSize, quantModeRouting, scaleDim0, aivNum, ubSize);
        initRoutingQuantTilingKey = moeInitRoutingV2TilingBase.tilingKey_;
        initRoutingWorkspace = moeInitRoutingV2TilingBase.workspaceSize_;
        nonQuantTilingData->moeInitRoutingV2TilingData = moeInitRoutingV2TilingBase.moeInitRoutingTilingData;
        nonQuantTilingData->common.initRoutingQuantTilingKey = initRoutingQuantTilingKey;
    }

    OP_LOGD(K_INNER_DEBUG, "initRoutingQuantTilingKey=%lu (isQuantRouting=%u)",
            initRoutingQuantTilingKey, info.isQuantRouting);

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    if (workSpaces == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(opName, "workSpaces");
        return ge::GRAPH_FAILED;
    }

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;

    uint64_t megeMoeWorkspace = (info.M + 256 - 1) / 256 * 256 * info.topK * sizeof(int32_t) + // expandedRowIdx
                            // cumsum + preSumBeforeRank + preSumRankInExpert
                            info.worldSize * info.worldSize * info.expertPerRank * sizeof(int32_t) * 3 +
                            info.maxRecvTokenNum * sizeof(float) * 2 + // perTokenScale GMM1&2
                            info.maxRecvTokenNum * info.N * sizeof(int16_t) * 2 + // GMM1 Out
                            info.maxRecvTokenNum * n2 * sizeof(int16_t) * 2 + // GMM2 Out
                            info.maxRecvTokenNum * info.K + // quantizedToken
                            info.maxRecvTokenNum * k2 + // swiglu mid
                            info.worldSize * sizeof(int32_t) * 16; // sync

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(megeMoeWorkspace, initRoutingWorkspace);

    OP_LOGI(K_INNER_DEBUG, "Leave MegaMoeA2A3 tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3TilingFunc(gert::TilingContext* context)
{
    return MegaMoeA2A3TilingFuncImpl(context);
}

struct MegaMoeA2A3CompileInfo {};
ge::graphStatus TilingParseForMegaMoeA2A3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe)
    .Tiling(MegaMoeA2A3TilingFunc)
    .TilingParse<MegaMoeA2A3CompileInfo>(TilingParseForMegaMoeA2A3);
} // namespace MegaMoeA2A3Tiling
