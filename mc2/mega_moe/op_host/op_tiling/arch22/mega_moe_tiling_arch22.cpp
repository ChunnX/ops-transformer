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
#include "../../../op_kernel/arch22/moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
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

    constexpr uint32_t BLOCK_NUM = 20;
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
}

namespace optiling {

static ge::graphStatus MegaMoeA2A3CheckAttrAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData& info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

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
    
    // ==================== 1. moe_expert_num 校验 (REQUIRED) ====================
    OP_TILING_CHECK(moeExpertNumPtr == nullptr || *moeExpertNumPtr <= 0,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(*moeExpertNumPtr < MIN_MOE_EXPERT_NUM || *moeExpertNumPtr > MAX_MOE_EXPERT_NUM,
        OP_LOGE(K_INNER_DEBUG, "moeExpertNum should be in [%ld, %ld], but got %ld.",
            MIN_MOE_EXPERT_NUM, MAX_MOE_EXPERT_NUM, *moeExpertNumPtr), return GRAPH_FAILED);
    
            // ==================== 2. ep_world_size 校验 (REQUIRED) ====================
    OP_TILING_CHECK(epWorldSizePtr == nullptr || *epWorldSizePtr <= 0,
        OP_LOGE(K_INNER_DEBUG, "epWorldSize is invalid."), return GRAPH_FAILED);
    bool isValidEpWorldSize = false;
    for (size_t i = 0; i < sizeof(VALID_EP_WORLD_SIZE) / sizeof(VALID_EP_WORLD_SIZE[0]); i++) {
        if (*epWorldSizePtr == VALID_EP_WORLD_SIZE[i]) {
            isValidEpWorldSize = true;
            break;
        }
    }
    OP_TILING_CHECK(!isValidEpWorldSize,
        OP_LOGE(K_INNER_DEBUG, "epWorldSize should be one of {2, 4, 8, 16, 32, 64, 128}, but got %ld.",
            *epWorldSizePtr), return GRAPH_FAILED);
    
            // ==================== 3. ccl_buffer_size 校验 (REQUIRED) ====================
    OP_TILING_CHECK(cclBufferSizePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "cclBufferSize is null."), return GRAPH_FAILED);
    
        // ==================== 4. max_recv_token_num 校验 (OPTIONAL) ====================
    if (maxRecvTokenNumPtr != nullptr) {
        OP_TILING_CHECK(*maxRecvTokenNumPtr < 0,
            OP_LOGE(K_INNER_DEBUG, "max_recv_token_num is invalid, should be >= 0, but got %ld.",
                *maxRecvTokenNumPtr), return GRAPH_FAILED);
    }
    
    // ==================== 5. dispatch_quant_mode 校验 (OPTIONAL) ====================
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "dispatchQuantMode is null."), return GRAPH_FAILED);
    // dispatch_quant_mode 校验：与 weight 数据类型对应
    // BF16/FP16 -> dispatch_quant_mode=0, INT4/INT8 -> dispatch_quant_mode=2
    // 获取 weight1 数据类型用于后续校验
    auto w1Desc = context->GetDynamicInputDesc(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Desc == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 desc is null."), return GRAPH_FAILED);
    ge::DataType w1DataType = w1Desc->GetDataType();
    int64_t expectedDispatchQuantMode = 0;
    if (w1DataType == ge::DT_INT4 || w1DataType == ge::DT_INT8) {
        expectedDispatchQuantMode = 2;
    } else if (w1DataType == ge::DT_BF16 || w1DataType == ge::DT_FLOAT16) {
        expectedDispatchQuantMode = 0;
    }
    OP_TILING_CHECK(*dispatchQuantModePtr != expectedDispatchQuantMode,
        OP_LOGE(K_INNER_DEBUG, "dispatch_quant_mode is invalid, should be %ld for weight dataType %d, but got %ld.",
            expectedDispatchQuantMode, static_cast<int>(w1DataType), *dispatchQuantModePtr), return GRAPH_FAILED);
    
    // ==================== 6. dispatch_quant_out_dtype 校验 (OPTIONAL) ====================
    OP_TILING_CHECK(dispatchQuantOutDtypePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "dispatchQuantOutDtype is null."), return GRAPH_FAILED);
    // dispatch_quant_out_type 校验：与 weight 数据类型对应
    // 默认值 DT_UNDEFINED 不校验，非默认值时需要匹配
    if (*dispatchQuantOutDtypePtr != static_cast<int64_t>(ge::DT_UNDEFINED)) {
        int64_t expectedQuantOutType = 0;
        if (w1DataType == ge::DT_BF16) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_BF16);
        } else if (w1DataType == ge::DT_FLOAT16) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_FLOAT16);
        } else if (w1DataType == ge::DT_INT4 || w1DataType == ge::DT_INT8) {
            expectedQuantOutType = static_cast<int64_t>(ge::DT_INT8);
        }
        OP_TILING_CHECK(*dispatchQuantOutDtypePtr != expectedQuantOutType,
            OP_LOGE(K_INNER_DEBUG, "dispatch_quant_out_type is invalid, should be %ld for weight dataType %d, but got %ld.",
                expectedQuantOutType, static_cast<int>(w1DataType), *dispatchQuantOutDtypePtr), return GRAPH_FAILED);
    }

    // ==================== 7. combine_quant_mode 校验 (OPTIONAL) ====================
    OP_TILING_CHECK(combineQuantModePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "combineQuantMode is null."), return GRAPH_FAILED);
    // combine_quant_mode 校验：当前仅支持 0
    OP_TILING_CHECK(*combineQuantModePtr != 0,
        OP_LOGE(K_INNER_DEBUG, "combine_quant_mode is invalid, only support 0, but got %ld.",
            *combineQuantModePtr), return GRAPH_FAILED);
    
    // ==================== 8. num_max_token_per_rank 校验 (OPTIONAL) ====================
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "numMaxTokensPerRankPtr is null."), return GRAPH_FAILED);
    // 获取 x 的 bs 值用于后续校验
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "x shape is null."), return GRAPH_FAILED);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);

    // num_max_token_per_rank 校验：必须 >= bs 且在 bs 合法范围内
    if (numMaxTokensPerRankPtr != nullptr && *numMaxTokensPerRankPtr > 0) {
        OP_TILING_CHECK(*numMaxTokensPerRankPtr < bs,
            OP_LOGE(K_INNER_DEBUG, "num_max_token_per_rank is invalid, should be >= bs(%ld), but got %ld.",
                bs, *numMaxTokensPerRankPtr), return GRAPH_FAILED);
        OP_TILING_CHECK(*numMaxTokensPerRankPtr < MIN_BS || *numMaxTokensPerRankPtr > MAX_BS,
            OP_LOGE(K_INNER_DEBUG, "num_max_token_per_rank is invalid, should be in [%ld, %ld], but got %ld.",
                MIN_BS, MAX_BS, *numMaxTokensPerRankPtr), return GRAPH_FAILED);
    }
    // ==================== 9. activation 校验 (OPTIONAL, default="swiglu") ====================
    OP_TILING_CHECK(activationPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "activation is null."), return GRAPH_FAILED);
    // activation 校验：当前仅支持 "swiglu"
    if (activationPtr != nullptr) {
        std::string activationStr(activationPtr);
        OP_TILING_CHECK(activationStr != "swiglu",
            OP_LOGE(K_INNER_DEBUG, "activation is invalid, only support 'swiglu', but got '%s'.",
                activationPtr), return GRAPH_FAILED);
    }

    // ==================== 10. activation_clamp 校验 (OPTIONAL, default=float::max) ====================
    OP_TILING_CHECK(activationClampPtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "activationClamp is null."), return GRAPH_FAILED);
    // activation_clamp 校验：必须 >= 0 且不为 NAN
    if (activationClampPtr != nullptr) {
        OP_TILING_CHECK(*activationClampPtr < 0 || std::isnan(*activationClampPtr),
            OP_LOGE(K_INNER_DEBUG, "activation_clamp is invalid, should be >= 0 and not NAN, but got %f.",
                *activationClampPtr), return GRAPH_FAILED);
    }

    // ==================== 11. activation_out_dtype 校验 (OPTIONAL, default=DT_UNDEFINED) ====================
    OP_TILING_CHECK(activationOutDtypePtr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "activationOutDtypePtr is null."), return GRAPH_FAILED);
    // activation_out_dtype 校验：A2A3当前仅支持 DT_UNDEFINED
    OP_TILING_CHECK(*activationOutDtypePtr != static_cast<int64_t>(ge::DT_UNDEFINED),
        OP_LOGE(K_INNER_DEBUG, "activation_out_dtype is invalid, should be default value %ld, but got %ld.",
            static_cast<int64_t>(ge::DT_UNDEFINED), *activationOutDtypePtr), return GRAPH_FAILED);
    
    // ==================== 12. transpose_weight1/2 校验 (OPTIONAL, default=false) ====================
    OP_TILING_CHECK(transposeWeight1Ptr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "transposeWeight1 is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(transposeWeight2Ptr == nullptr,
        OP_LOGE(K_INNER_DEBUG, "transposeWeight2 is null."), return GRAPH_FAILED);
    // transpose_weight1 和 transpose_weight2 必须一致
    OP_TILING_CHECK(*transposeWeight1Ptr != *transposeWeight2Ptr,
        OP_LOGE(K_INNER_DEBUG, "transpose_weight1 and transpose_weight2 must be the same, transpose_weight1 = %d, transpose_weight2 = %d.",
            static_cast<int>(*transposeWeight1Ptr), static_cast<int>(*transposeWeight2Ptr)), return GRAPH_FAILED);
    
    // ==================== 13. weight1_interleave 校验 (OPTIONAL, default=0) ====================
    if (weight1InterleavePtr != nullptr) {
        OP_TILING_CHECK(*weight1InterleavePtr < 0,
            OP_LOGE(K_INNER_DEBUG, "weight1_interleave is invalid, should be >= 0, but got %ld.",
                *weight1InterleavePtr), return GRAPH_FAILED);
    }

    // 获取 weight1 的 N 值用于 weight1_interleave 范围校验
    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Tensor == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 tensor is null."), return GRAPH_FAILED);
    uint32_t w1TensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(w1TensorDims - 1);

    // weight1_interleave 范围校验：必须在 [0, N/2] 之间
    if (weight1InterleavePtr != nullptr) {
        uint32_t maxInterleave = N / 2;
        OP_TILING_CHECK(static_cast<uint32_t>(*weight1InterleavePtr) > maxInterleave,
            OP_LOGE(K_INNER_DEBUG, "weight1_interleave is invalid, should be in [0, %u], but got %ld.",
                maxInterleave, *weight1InterleavePtr), return GRAPH_FAILED);
    }

    info.moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    info.epWorldSize = static_cast<uint32_t>(*epWorldSizePtr);
    info.cclBufferSize = static_cast<uint32_t>(*cclBufferSizePtr);
    info.maxRecvTokenNum = static_cast<uint32_t>(*maxRecvTokenNumPtr);
    info.dispatchQuantMode = static_cast<uint32_t>(*dispatchQuantModePtr);
    info.dispatchQuantOutDtype = static_cast<uint32_t>(*dispatchQuantOutDtypePtr);
    info.combineQuantMode = static_cast<uint32_t>(*combineQuantModePtr);
    info.numMaxTokensPerRank = static_cast<uint32_t>(*numMaxTokensPerRankPtr);
    info.activationOutDtype = activationOutDtypePtr != nullptr ? static_cast<uint32_t>(*activationOutDtypePtr) : static_cast<uint32_t>(ge::DT_UNDEFINED);
    info.isTransposeW1 = *transposeWeight1Ptr ? 1U : 0U;
    info.isTransposeW2 = *transposeWeight2Ptr ? 1U : 0U;
    info.weight1Interleave = weight1InterleavePtr != nullptr ? static_cast<uint32_t>(*weight1InterleavePtr) : 0U;

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

static ge::graphStatus MegaMoeA2A3CheckShapeAndSetTiling(gert::TilingContext *context, MegaMoeA2A3TilingData &info)
{
    // ==================== 1. 输入张量非空校验 ====================
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(TOPK_IDS_INDEX);
    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "context shape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "x shape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_ids shape is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_weights shape is null."), return GRAPH_FAILED);

    // ==================== 2. context 输入校验 ====================
    auto contextDesc = context->GetInputDesc(CONTEXT_INDEX);
    OP_TILING_CHECK(contextDesc == nullptr,
        OP_LOGE(K_INNER_DEBUG, "context desc is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(K_INNER_DEBUG, "context dataType is invalid, should be INT32, but got %d.",
            static_cast<int>(contextDesc->GetDataType())), return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(K_INNER_DEBUG, "context format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(K_INNER_DEBUG, "contextShape dims must be 1, but current dim num is %zu.",
        contextStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    // ==================== 3. x 输入校验 ====================
    // 维度数校验：必须为 2 维
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "x must be 2-dimension, but got %lu dim.",
            xStorageShape->GetStorageShape().GetDimNum()), return GRAPH_FAILED);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t hiddenSize = xStorageShape->GetStorageShape().GetDim(1);

    // 维度值范围校验
    OP_TILING_CHECK(bs < MIN_BS || bs > MAX_BS,
        OP_LOGE(K_INNER_DEBUG, "x's dim0(bs) is invalid, should be in [%ld, %ld], but got %ld.",
            MIN_BS, MAX_BS, bs), return GRAPH_FAILED);
    OP_TILING_CHECK(hiddenSize < MIN_HIDDEN_SIZE || hiddenSize > MAX_HIDDEN_SIZE,
        OP_LOGE(K_INNER_DEBUG, "x's dim1(hidden_size) is invalid, should be in [%ld, %ld], but got %ld.",
            MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, hiddenSize), return GRAPH_FAILED);
    OP_TILING_CHECK(hiddenSize % HIDDEN_SIZE_ALIGN != 0,
        OP_LOGE(K_INNER_DEBUG, "x's dim1(hidden_size) should be %ld aligned, but got %ld.",
            HIDDEN_SIZE_ALIGN, hiddenSize), return GRAPH_FAILED);

    // 数据类型校验
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "x desc is null."), return GRAPH_FAILED);
    ge::DataType xDataType = xDesc->GetDataType();
    OP_TILING_CHECK(xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_BF16,
        OP_LOGE(K_INNER_DEBUG, "x dataType is invalid, should be FLOAT16 or BF16, but got %d.",
            static_cast<int>(xDataType)), return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(K_INNER_DEBUG, "x format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

    // ==================== 4. topk_ids 输入校验 ====================
    // 维度数校验：必须为 2 维
    OP_TILING_CHECK(topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "topk_ids must be 2-dimension, but got %lu dim.",
            topkIdsStorageShape->GetStorageShape().GetDimNum()), return GRAPH_FAILED);

    int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // Shape 一致性校验：dim0 必须与 x 的 dim0 相等
    OP_TILING_CHECK(topkIdsDim0 != bs,
        OP_LOGE(K_INNER_DEBUG, "topk_ids's dim0 not equal to x's dim0, topk_ids's dim0 = %ld, x's dim0 = %ld.",
            topkIdsDim0, bs), return GRAPH_FAILED);

    // topK 范围校验
    OP_TILING_CHECK(topK < MIN_TOPK || topK > MAX_TOPK,
        OP_LOGE(K_INNER_DEBUG, "topk_ids's dim1(topK) is invalid, should be in [%ld, %ld], but got %ld.",
            MIN_TOPK, MAX_TOPK, topK), return GRAPH_FAILED);

    // 数据类型校验
    auto topkIdsDesc = context->GetInputDesc(TOPK_IDS_INDEX);
    OP_TILING_CHECK(topkIdsDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_ids desc is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(K_INNER_DEBUG, "topk_ids dataType is invalid, should be INT32, but got %d.",
            static_cast<int>(topkIdsDesc->GetDataType())), return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(K_INNER_DEBUG, "topk_ids format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

    // ==================== 5. topk_weights 输入校验 ====================
    // 维度数校验：必须为 2 维
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "topk_weights must be 2-dimension, but got %lu dim.",
            topkWeightsStorageShape->GetStorageShape().GetDimNum()), return GRAPH_FAILED);

    int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);

    // Shape 一致性校验
    OP_TILING_CHECK(topkWeightsDim0 != bs,
        OP_LOGE(K_INNER_DEBUG, "topk_weights's dim0 not equal to x's dim0, topk_weights's dim0 = %ld, x's dim0 = %ld.",
            topkWeightsDim0, bs), return GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsDim1 != topK,
        OP_LOGE(K_INNER_DEBUG, "topk_weights's dim1 not equal to topk_ids's dim1, topk_weights's dim1 = %ld, topk_ids's dim1 = %ld.",
            topkWeightsDim1, topK), return GRAPH_FAILED);

    // 数据类型校验
    auto topkWeightsDesc = context->GetInputDesc(TOPK_WEIGHTS_INDEX);
    OP_TILING_CHECK(topkWeightsDesc == nullptr, OP_LOGE(K_INNER_DEBUG, "topk_weights desc is null."), return GRAPH_FAILED);
    ge::DataType topkWeightsDataType = topkWeightsDesc->GetDataType();
    OP_TILING_CHECK(topkWeightsDataType != ge::DT_FLOAT && topkWeightsDataType != ge::DT_BF16,
        OP_LOGE(K_INNER_DEBUG, "topk_weights dataType is invalid, should be FLOAT or BF16, but got %d.",
            static_cast<int>(topkWeightsDataType)), return GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) != ge::FORMAT_ND,
        OP_LOGE(K_INNER_DEBUG, "topk_weights format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

    // ==================== 6. weight1 输入校验 ====================
    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Tensor == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 tensor is null."), return GRAPH_FAILED);

    auto w1Desc = context->GetDynamicInputDesc(WEIGHT1_INDEX, 0);
    OP_TILING_CHECK(w1Desc == nullptr, OP_LOGE(K_INNER_DEBUG, "weight1 desc is null."), return GRAPH_FAILED);
    ge::DataType w1DataType = w1Desc->GetDataType();
    OP_TILING_CHECK(w1DataType != ge::DT_BF16 && w1DataType != ge::DT_FLOAT16 &&
                    w1DataType != ge::DT_INT8 && w1DataType != ge::DT_INT4,
        OP_LOGE(K_INNER_DEBUG, "weight1 dataType is invalid, should be BF16/FP16/INT8/INT4, but got %d.",
            static_cast<int>(w1DataType)), return GRAPH_FAILED);
    ge::Format w1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(w1Desc->GetStorageFormat()));
    OP_TILING_CHECK(w1Format != ge::FORMAT_ND && w1Format != ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(K_INNER_DEBUG, "weight1 format is invalid, should be FORMAT_ND or FORMAT_FRACTAL_NZ."), return GRAPH_FAILED);

    uint32_t w1TensorDims = w1Tensor->GetOriginShape().GetDimNum();
    uint32_t N = w1Tensor->GetStorageShape().GetDim(w1TensorDims - 1);
    OP_TILING_CHECK(N % 2 != 0,
        OP_LOGE(K_INNER_DEBUG, "weight1 format is invalid, the last dim should be even."), return GRAPH_FAILED);

    // 统计 weight1 的 tensor list 长度
    uint32_t listLen = 0;
    while (true) {
        auto wTensorT = context->GetDynamicInputTensor(WEIGHT1_INDEX, ++listLen);
        if (wTensorT == nullptr) { break; }
    }

    uint32_t expertPerRank;
    if (listLen == 1) {
        // 场景1：传入一个 3 维 tensor，shape 为 (num_experts_per_rank, hidden_size, N)
        OP_TILING_CHECK(w1TensorDims != THREE_DIMS,
            OP_LOGE(K_INNER_DEBUG, "weight1 must be 3-dimension when listLen=1, but got %u dim.", w1TensorDims),
            return GRAPH_FAILED);
        expertPerRank = w1Tensor->GetStorageShape().GetDim(0);
        int64_t w1Dim1 = w1Tensor->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(w1Dim1 != hiddenSize,
            OP_LOGE(K_INNER_DEBUG, "weight1's dim1 not equal to hidden_size, weight1's dim1 = %ld, hidden_size = %ld.",
                w1Dim1, hiddenSize), return GRAPH_FAILED);
    } else {
        // 场景2：传入多个 2 维 tensor，每个 shape 为 (hidden_size, N)
        expertPerRank = listLen;
        for (uint32_t i = 0; i < listLen; i++) {
            auto wTensorI = context->GetDynamicInputTensor(WEIGHT1_INDEX, i);
            OP_TILING_CHECK(wTensorI == nullptr,
                OP_LOGE(K_INNER_DEBUG, "weight1[%u] tensor is null.", i), return GRAPH_FAILED);
            OP_TILING_CHECK(wTensorI->GetOriginShape().GetDimNum() != TWO_DIMS,
                OP_LOGE(K_INNER_DEBUG, "weight1[%u] must be 2-dimension, but got %lu dim.",
                    i, wTensorI->GetOriginShape().GetDimNum()), return GRAPH_FAILED);
            int64_t w1IDim0 = wTensorI->GetStorageShape().GetDim(0);
            OP_TILING_CHECK(w1IDim0 != hiddenSize,
                OP_LOGE(K_INNER_DEBUG, "weight1[%u]'s dim0 not equal to hidden_size, weight1[%u]'s dim0 = %ld, hidden_size = %ld.",
                    i, i, w1IDim0, hiddenSize), return GRAPH_FAILED);
        }
    }

    // expertPerRank 范围校验
    OP_TILING_CHECK(expertPerRank < MIN_EXPERT_PER_RANK || expertPerRank > MAX_EXPERT_PER_RANK,
        OP_LOGE(K_INNER_DEBUG, "expertPerRank is invalid, should be in [%ld, %ld], but got %u.",
            MIN_EXPERT_PER_RANK, MAX_EXPERT_PER_RANK, expertPerRank), return GRAPH_FAILED);

    // ==================== 7. weight2 输入校验 ====================
    auto w2Tensor = context->GetDynamicInputTensor(WEIGHT2_INDEX, 0);
    OP_TILING_CHECK(w2Tensor == nullptr, OP_LOGE(K_INNER_DEBUG, "weight2 tensor is null."), return GRAPH_FAILED);

    auto w2Desc = context->GetDynamicInputDesc(WEIGHT2_INDEX, 0);
    OP_TILING_CHECK(w2Desc == nullptr, OP_LOGE(K_INNER_DEBUG, "weight2 desc is null."), return GRAPH_FAILED);
    ge::DataType w2DataType = w2Desc->GetDataType();
    OP_TILING_CHECK(w2DataType != ge::DT_BF16 && w2DataType != ge::DT_FLOAT16 &&
                    w2DataType != ge::DT_INT8 && w2DataType != ge::DT_INT4,
        OP_LOGE(K_INNER_DEBUG, "weight2 dataType is invalid, should be BF16/FP16/INT8/INT4, but got %d.",
            static_cast<int>(w2DataType)), return GRAPH_FAILED);

    // weight1 和 weight2 数据类型必须一致
    OP_TILING_CHECK(w1DataType != w2DataType,
        OP_LOGE(K_INNER_DEBUG, "weight1 and weight2 must have the same dataType, weight1 dataType = %d, weight2 dataType = %d.",
            static_cast<int>(w1DataType), static_cast<int>(w2DataType)), return GRAPH_FAILED);
    ge::Format w2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(w2Desc->GetStorageFormat()));
    OP_TILING_CHECK(w2Format != ge::FORMAT_ND && w2Format != ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(K_INNER_DEBUG, "weight2 format is invalid, should be FORMAT_ND or FORMAT_FRACTAL_NZ."), return GRAPH_FAILED);
    OP_TILING_CHECK(w1Format != w2Format,
        OP_LOGE(K_INNER_DEBUG, "weight1 and weight2 must have the same format, weight1 format = %d, weight2 format = %d.",
            static_cast<int>(w1Format), static_cast<int>(w2Format)), return GRAPH_FAILED);

    uint32_t w2TensorDims = w2Tensor->GetOriginShape().GetDimNum();
    
    uint32_t w2ListLen = 0;
    while (true) {
        auto wTensorT = context->GetDynamicInputTensor(WEIGHT2_INDEX, ++w2ListLen);
        if (wTensorT == nullptr) { break; }
    }
    OP_TILING_CHECK(w2ListLen != listLen,
        OP_LOGE(K_INNER_DEBUG, "weight2's listLen not equal to weight1's listLen, weight2's listLen = %u, weight1's listLen = %u.",
            w2ListLen, listLen), return GRAPH_FAILED);

    uint32_t n2 = N / 2;
    if (listLen == 1) {
        OP_TILING_CHECK(w2TensorDims != THREE_DIMS,
            OP_LOGE(K_INNER_DEBUG, "weight2 must be 3-dimension when listLen=1, but got %u dim.", w2TensorDims),
            return GRAPH_FAILED);
        int64_t w2Dim0 = w2Tensor->GetStorageShape().GetDim(0);
        int64_t w2Dim1 = w2Tensor->GetStorageShape().GetDim(1);
        int64_t w2Dim2 = w2Tensor->GetStorageShape().GetDim(2);
        OP_TILING_CHECK(w2Dim0 != expertPerRank,
            OP_LOGE(K_INNER_DEBUG, "weight2's dim0 not equal to expertPerRank, weight2's dim0 = %ld, expertPerRank = %u.",
                w2Dim0, expertPerRank), return GRAPH_FAILED);
        OP_TILING_CHECK(w2Dim1 != n2,
            OP_LOGE(K_INNER_DEBUG, "weight2's dim1 not equal to N/2, weight2's dim1 = %ld, N/2 = %u.",
                w2Dim1, n2), return GRAPH_FAILED);
        OP_TILING_CHECK(w2Dim2 != hiddenSize,
            OP_LOGE(K_INNER_DEBUG, "weight2's dim2 not equal to hidden_size, weight2's dim2 = %ld, hidden_size = %ld.",
                w2Dim2, hiddenSize), return GRAPH_FAILED);
    } else {
        for (uint32_t i = 0; i < listLen; i++) {
            auto wTensorI = context->GetDynamicInputTensor(WEIGHT2_INDEX, i);
            OP_TILING_CHECK(wTensorI == nullptr,
                OP_LOGE(K_INNER_DEBUG, "weight2[%u] tensor is null.", i), return GRAPH_FAILED);
            OP_TILING_CHECK(wTensorI->GetOriginShape().GetDimNum() != TWO_DIMS,
                OP_LOGE(K_INNER_DEBUG, "weight2[%u] must be 2-dimension, but got %lu dim.",
                    i, wTensorI->GetOriginShape().GetDimNum()), return GRAPH_FAILED);
            int64_t w2IDim0 = wTensorI->GetStorageShape().GetDim(0);
            int64_t w2IDim1 = wTensorI->GetStorageShape().GetDim(1);
            OP_TILING_CHECK(w2IDim0 != n2,
                OP_LOGE(K_INNER_DEBUG, "weight2[%u]'s dim0 not equal to N/2, weight2[%u]'s dim0 = %ld, N/2 = %u.",
                    i, i, w2IDim0, n2), return GRAPH_FAILED);
            OP_TILING_CHECK(w2IDim1 != hiddenSize,
                OP_LOGE(K_INNER_DEBUG, "weight2[%u]'s dim1 not equal to hidden_size, weight2[%u]'s dim1 = %ld, hidden_size = %ld.",
                    i, i, w2IDim1, hiddenSize), return GRAPH_FAILED);
        }
    }

    // ==================== 量化权重必须提供 scale 校验 ====================
    // 当 weight1 和 weight2 的类型都为 INT8 或 INT4 时，GMM1/GMM2 需要进行反量化
    // 此时 weight_scales1/weight_scales2 必须存在
    bool isQuantWeight = (w1DataType == ge::DT_INT8 || w1DataType == ge::DT_INT4);

    // ==================== 8. weight_scales1 输入校验（可选） ====================
    // weight_scales1：可选输入，weight1/weight2 为 INT8 或 INT4 时需要，GMM1 反量化参数
    // 支持两种场景：
    //   1. 传入一个 TensorList，list 包含一个 2 维 tensor，shape 为 (num_experts_per_rank, N)
    //   2. 传入一个 TensorList，list 包含 num_experts_per_rank 个 tensor，每个 tensor 的 shape 为 (N,)
    auto wScale1Tensor = context->GetDynamicInputTensor(WEIGHT_SCALES1_INDEX, 0);
    if (isQuantWeight) {
        OP_TILING_CHECK(wScale1Tensor == nullptr,
            OP_LOGE(K_INNER_DEBUG, "weight_scales1 is required when weight1/weight2 is INT8 or INT4."),
            return GRAPH_FAILED);
    }
    if (wScale1Tensor != nullptr) {
        // 数据类型校验
        auto wScale1Desc = context->GetDynamicInputDesc(WEIGHT_SCALES1_INDEX, 0);
        OP_TILING_CHECK(wScale1Desc == nullptr,
            OP_LOGE(K_INNER_DEBUG, "weight_scales1 desc is null."), return GRAPH_FAILED);
        OP_TILING_CHECK(wScale1Desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(K_INNER_DEBUG, "weight_scales1 dataType is invalid, should be FLOAT, but got %d.",
                static_cast<int>(wScale1Desc->GetDataType())), return GRAPH_FAILED);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(wScale1Desc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE(K_INNER_DEBUG, "weight_scales1 format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

        // 统计 weight_scales1 的 tensor list 长度
        uint32_t scale1ListLen = 0;
        while (true) {
            auto sTensorT = context->GetDynamicInputTensor(WEIGHT_SCALES1_INDEX, ++scale1ListLen);
            if (sTensorT == nullptr) { break; }
        }

        // weight_scales1 的长度必须与 weight1 的长度一致
        OP_TILING_CHECK(scale1ListLen != listLen,
            OP_LOGE(K_INNER_DEBUG, "weight_scales1's listLen not equal to weight1's listLen, weight_scales1's listLen = %u, weight1's listLen = %u.",
                scale1ListLen, listLen), return GRAPH_FAILED);

        if (scale1ListLen == 1) {
            // 场景1：传入一个 tensor，可以是 2 维或 1 维
            uint32_t scale1Dims = wScale1Tensor->GetOriginShape().GetDimNum();
            OP_TILING_CHECK(scale1Dims != TWO_DIMS && scale1Dims != ONE_DIM,
                OP_LOGE(K_INNER_DEBUG, "weight_scales1 must be 1-dimension or 2-dimension, but got %u dim.", scale1Dims),
                return GRAPH_FAILED);
            if (scale1Dims == TWO_DIMS) {
                // 2 维：shape 为 (num_experts_per_rank, N)
                int64_t scale1Dim0 = wScale1Tensor->GetStorageShape().GetDim(0);
                int64_t scale1Dim1 = wScale1Tensor->GetStorageShape().GetDim(1);
                OP_TILING_CHECK(scale1Dim0 != expertPerRank,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales1's dim0 not equal to expertPerRank, weight_scales1's dim0 = %ld, expertPerRank = %u.",
                        scale1Dim0, expertPerRank), return GRAPH_FAILED);
                OP_TILING_CHECK(scale1Dim1 != N,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales1's dim1 not equal to N, weight_scales1's dim1 = %ld, N = %u.",
                        scale1Dim1, N), return GRAPH_FAILED);
            }
        } else {
            // 场景2：传入多个 tensor，每个 tensor 的 shape 为 (N,)
            for (uint32_t i = 0; i < scale1ListLen; i++) {
                auto sTensorI = context->GetDynamicInputTensor(WEIGHT_SCALES1_INDEX, i);
                OP_TILING_CHECK(sTensorI == nullptr,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales1[%u] tensor is null.", i), return GRAPH_FAILED);
                OP_TILING_CHECK(sTensorI->GetOriginShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales1[%u] must be 1-dimension, but got %lu dim.",
                        i, sTensorI->GetOriginShape().GetDimNum()), return GRAPH_FAILED);
            }
        }
    }

    // ==================== 9. weight_scales2 输入校验（可选） ====================
    // weight_scales2：可选输入，weight1/weight2 为 INT8 或 INT4 时需要，GMM2 反量化参数
    // 支持两种场景：
    //   1. 传入一个 TensorList，list 包含一个 2 维 tensor，shape 为 (num_experts_per_rank, hidden_size)
    //   2. 传入一个 TensorList，list 包含 num_experts_per_rank 个 tensor，每个 tensor 的 shape 为 (hidden_size,)
    auto wScale2Tensor = context->GetDynamicInputTensor(WEIGHT_SCALES2_INDEX, 0);
    if (isQuantWeight) {
        OP_TILING_CHECK(wScale2Tensor == nullptr,
            OP_LOGE(K_INNER_DEBUG, "weight_scales2 is required when weight1/weight2 is INT8 or INT4."),
            return GRAPH_FAILED);
    }
    if (wScale2Tensor != nullptr) {
        // 数据类型校验
        auto wScale2Desc = context->GetDynamicInputDesc(WEIGHT_SCALES2_INDEX, 0);
        OP_TILING_CHECK(wScale2Desc == nullptr,
            OP_LOGE(K_INNER_DEBUG, "weight_scales2 desc is null."), return GRAPH_FAILED);
        OP_TILING_CHECK(wScale2Desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(K_INNER_DEBUG, "weight_scales2 dataType is invalid, should be FLOAT, but got %d.",
                static_cast<int>(wScale2Desc->GetDataType())), return GRAPH_FAILED);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(wScale2Desc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE(K_INNER_DEBUG, "weight_scales2 format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

        // 统计 weight_scales2 的 tensor list 长度
        uint32_t scale2ListLen = 0;
        while (true) {
            auto sTensorT = context->GetDynamicInputTensor(WEIGHT_SCALES2_INDEX, ++scale2ListLen);
            if (sTensorT == nullptr) { break; }
        }

        // weight_scales2 的长度必须与 weight1 的长度一致
        OP_TILING_CHECK(scale2ListLen != listLen,
            OP_LOGE(K_INNER_DEBUG, "weight_scales2's listLen not equal to weight1's listLen, weight_scales2's listLen = %u, weight1's listLen = %u.",
                scale2ListLen, listLen), return GRAPH_FAILED);

        if (scale2ListLen == 1) {
            // 场景1：传入一个 tensor，可以是 2 维或 1 维
            uint32_t scale2Dims = wScale2Tensor->GetOriginShape().GetDimNum();
            OP_TILING_CHECK(scale2Dims != TWO_DIMS && scale2Dims != ONE_DIM,
                OP_LOGE(K_INNER_DEBUG, "weight_scales2 must be 1-dimension or 2-dimension, but got %u dim.", scale2Dims),
                return GRAPH_FAILED);
            if (scale2Dims == TWO_DIMS) {
                // 2 维：shape 为 (num_experts_per_rank, hidden_size)
                int64_t scale2Dim0 = wScale2Tensor->GetStorageShape().GetDim(0);
                int64_t scale2Dim1 = wScale2Tensor->GetStorageShape().GetDim(1);
                OP_TILING_CHECK(scale2Dim0 != expertPerRank,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales2's dim0 not equal to expertPerRank, weight_scales2's dim0 = %ld, expertPerRank = %u.",
                        scale2Dim0, expertPerRank), return GRAPH_FAILED);
                OP_TILING_CHECK(scale2Dim1 != hiddenSize,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales2's dim1 not equal to hidden_size, weight_scales2's dim1 = %ld, hidden_size = %ld.",
                        scale2Dim1, hiddenSize), return GRAPH_FAILED);
            }
        } else {
            // 场景2：传入多个 tensor，每个 tensor 的 shape 为 (hidden_size,)
            for (uint32_t i = 0; i < scale2ListLen; i++) {
                auto sTensorI = context->GetDynamicInputTensor(WEIGHT_SCALES2_INDEX, i);
                OP_TILING_CHECK(sTensorI == nullptr,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales2[%u] tensor is null.", i), return GRAPH_FAILED);
                OP_TILING_CHECK(sTensorI->GetOriginShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(K_INNER_DEBUG, "weight_scales2[%u] must be 1-dimension, but got %lu dim.",
                        i, sTensorI->GetOriginShape().GetDimNum()), return GRAPH_FAILED);
            }
        }
    }

    // ==================== 10. bias1 输入校验 ====================
    // 当 weight1 和 weight2 数据类型都为 INT4 时，bias1 必须存在
    bool isInt4Weight = (w1DataType == ge::DT_INT4 && w2DataType == ge::DT_INT4);
    const gert::StorageShape *bias1StorageShape = context->GetDynamicInputShape(BIAS1_INDEX, 0);
    if (isInt4Weight) {
        OP_TILING_CHECK(bias1StorageShape == nullptr,
            OP_LOGE(K_INNER_DEBUG, "bias1 is required when weight1 and weight2 are INT4."),
            return GRAPH_FAILED);
    }
    
    if (bias1StorageShape != nullptr) {
        auto bias1Tensor = context->GetDynamicInputTensor(BIAS1_INDEX, 0);
        OP_TILING_CHECK(bias1Tensor == nullptr, 
            OP_LOGE(K_INNER_DEBUG, "bias1 tensor is null."), return GRAPH_FAILED);
        
        auto bias1Desc = context->GetDynamicInputDesc(BIAS1_INDEX, 0);
        OP_TILING_CHECK(bias1Desc == nullptr,
            OP_LOGE(K_INNER_DEBUG, "bias1 desc is null."), return GRAPH_FAILED);
        OP_TILING_CHECK(bias1Desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(K_INNER_DEBUG, "bias1 dataType is invalid, should be FP32, but got %d.",
                static_cast<int>(bias1Desc->GetDataType())), return GRAPH_FAILED);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(bias1Desc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE(K_INNER_DEBUG, "bias1 format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

        // 统计 bias1 的 tensor list 长度
        uint32_t bias1ListLen = 0;
        while (true) {
            auto bTensorT = context->GetDynamicInputTensor(BIAS1_INDEX, ++bias1ListLen);
            if (bTensorT == nullptr) { break; }
        }

        // bias1 的长度必须与 weight1 的长度一致
        OP_TILING_CHECK(bias1ListLen != listLen,
            OP_LOGE(K_INNER_DEBUG, "bias1's listLen not equal to weight1's listLen, bias1's listLen = %u, weight1's listLen = %u.",
                bias1ListLen, listLen), return GRAPH_FAILED);
        
        // 每个 bias1[i] 的 shape 必须是 (N,)
        for (uint32_t i = 0; i < bias1ListLen; i++) {
            auto bTensorI = context->GetDynamicInputTensor(BIAS1_INDEX, i);
            OP_TILING_CHECK(bTensorI == nullptr,
                OP_LOGE(K_INNER_DEBUG, "bias1[%u] tensor is null.", i), return GRAPH_FAILED);
            OP_TILING_CHECK(bTensorI->GetOriginShape().GetDimNum() != ONE_DIM,
                OP_LOGE(K_INNER_DEBUG, "bias1[%u] must be 1-dimension, but got %lu dim.",
                    i, bTensorI->GetOriginShape().GetDimNum()), return GRAPH_FAILED);
            uint32_t bias1Dim0 = bTensorI->GetStorageShape().GetDim(0);
            OP_TILING_CHECK(bias1Dim0 != N,
                OP_LOGE(K_INNER_DEBUG, "bias1[%u]'s dim0 not equal to N, bias1[%u]'s dim0 = %u, N = %u.",
                    i, i, bias1Dim0, N), return GRAPH_FAILED);
        }
    }

    // ==================== 11. bias2 输入校验 ====================
    // 当 weight1 和 weight2 数据类型都为 INT4 时，bias2 必须存在
    const gert::StorageShape *bias2StorageShape = context->GetDynamicInputShape(BIAS2_INDEX, 0);
    if (isInt4Weight) {
        OP_TILING_CHECK(bias2StorageShape == nullptr,
            OP_LOGE(K_INNER_DEBUG, "bias2 is required when weight1 and weight2 are INT4."),
            return GRAPH_FAILED);
    }
    
    if (bias2StorageShape != nullptr) {
        auto bias2Tensor = context->GetDynamicInputTensor(BIAS2_INDEX, 0);
        OP_TILING_CHECK(bias2Tensor == nullptr, 
            OP_LOGE(K_INNER_DEBUG, "bias2 tensor is null."), return GRAPH_FAILED);
        
        auto bias2Desc = context->GetDynamicInputDesc(BIAS2_INDEX, 0);
        OP_TILING_CHECK(bias2Desc == nullptr,
            OP_LOGE(K_INNER_DEBUG, "bias2 desc is null."), return GRAPH_FAILED);
        OP_TILING_CHECK(bias2Desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(K_INNER_DEBUG, "bias2 dataType is invalid, should be FP32, but got %d.",
                static_cast<int>(bias2Desc->GetDataType())), return GRAPH_FAILED);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(bias2Desc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE(K_INNER_DEBUG, "bias2 format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);

        // 统计 bias2 的 tensor list 长度
        uint32_t bias2ListLen = 0;
        while (true) {
            auto bTensorT = context->GetDynamicInputTensor(BIAS2_INDEX, ++bias2ListLen);
            if (bTensorT == nullptr) { break; }
        }

        // bias2 的长度必须与 weight1 的长度一致
        OP_TILING_CHECK(bias2ListLen != listLen,
            OP_LOGE(K_INNER_DEBUG, "bias2's listLen not equal to weight1's listLen, bias2's listLen = %u, weight1's listLen = %u.",
                bias2ListLen, listLen), return GRAPH_FAILED);
        
        // 每个 bias2[i] 的 shape 必须是 (hiddenSize,)
        for (uint32_t i = 0; i < bias2ListLen; i++) {
            auto bTensorI = context->GetDynamicInputTensor(BIAS2_INDEX, i);
            OP_TILING_CHECK(bTensorI == nullptr,
                OP_LOGE(K_INNER_DEBUG, "bias2[%u] tensor is null.", i), return GRAPH_FAILED);
            OP_TILING_CHECK(bTensorI->GetOriginShape().GetDimNum() != ONE_DIM,
                OP_LOGE(K_INNER_DEBUG, "bias2[%u] must be 1-dimension, but got %lu dim.",
                    i, bTensorI->GetOriginShape().GetDimNum()), return GRAPH_FAILED);
            uint32_t bias2Dim0 = bTensorI->GetStorageShape().GetDim(0);
            OP_TILING_CHECK(bias2Dim0 != static_cast<uint32_t>(hiddenSize),
                OP_LOGE(K_INNER_DEBUG, "bias2[%u]'s dim0 not equal to hiddenSize, bias2[%u]'s dim0 = %u, hiddenSize = %ld.",
                    i, i, bias2Dim0, hiddenSize), return GRAPH_FAILED);
        }
    }

    // ==================== 12. x_active_mask 输入校验（可选） ====================
    const gert::StorageShape *xActiveMaskStorageShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    bool isActiveMask = (xActiveMaskStorageShape != nullptr);
    if (isActiveMask) {
        int64_t xActiveMaskDimNums = xActiveMaskStorageShape->GetStorageShape().GetDimNum();
        
        OP_TILING_CHECK(xActiveMaskDimNums != ONE_DIM,
            OP_LOGE(K_INNER_DEBUG, "x_active_mask must be 1-dimension, but got %ld dim. 2-dimension is not supported now.",
                xActiveMaskDimNums), return GRAPH_FAILED);

        int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(xActiveMaskDim0 != bs,
            OP_LOGE(K_INNER_DEBUG, "x_active_mask's dim0 not equal to x's dim0, x_active_mask's dim0 = %ld, x's dim0 = %ld.",
                xActiveMaskDim0, bs), return GRAPH_FAILED);

        auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(xActiveMaskDesc == nullptr,
            OP_LOGE(K_INNER_DEBUG, "x_active_mask desc is null."), return GRAPH_FAILED);
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_INT8,
            OP_LOGE(K_INNER_DEBUG, "x_active_mask dataType is invalid, should be INT8, but got %d.",
                static_cast<int>(xActiveMaskDesc->GetDataType())), return GRAPH_FAILED);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xActiveMaskDesc->GetStorageFormat())) != ge::FORMAT_ND,
            OP_LOGE(K_INNER_DEBUG, "x_active_mask format is invalid, should be FORMAT_ND."), return GRAPH_FAILED);
    }

    // ==================== 13. scales 输入校验（可选，预留） ====================
    const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(SCALES_INDEX);
    OP_TILING_CHECK(scalesStorageShape != nullptr,
        OP_LOGE(K_INNER_DEBUG, "scales is not supported yet, please pass nullptr."),
        return GRAPH_FAILED);

    // ==================== 14. 设置 tiling data ====================
    info.bs = static_cast<uint32_t>(bs);
    info.N = N;
    info.K = static_cast<uint32_t>(hiddenSize);
    info.expertPerRank = expertPerRank;
    info.topK = static_cast<uint32_t>(topK);
    info.isWeightTensorList = listLen;

    OP_LOGD(K_INNER_DEBUG, "bs=%u", info.bs);
    OP_LOGD(K_INNER_DEBUG, "K=%u", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%u", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%u", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%u", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%u", info.isWeightTensorList);
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
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);
    auto commAlg = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_COMM_ALG_INDEX));
    OP_TILING_CHECK(commAlg == nullptr, 
        OP_LOGE(K_INNER_DEBUG, "commAlg is null."), return ge::GRAPH_FAILED);
    
    if (strlen(commAlg) > 0) {
        OP_LOGE(K_INNER_DEBUG, "Attr commAlg is invalid, current version only supports \"\", but got \"%s\".", commAlg);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckHcclBuffSize(const gert::TilingContext *context,
    MegaMoeA2A3TilingData &info)
{
    const char *nodeName = K_INNER_DEBUG;
    int64_t cclBufferSize = info.cclBufferSize;
    OP_LOGD(nodeName, "cclBufferSize = %ld Bytes (%ld MB).", cclBufferSize, ops::CeilDiv(cclBufferSize, static_cast<int64_t>(MB_SIZE)));
    OP_TILING_CHECK(cclBufferSize == 0, OP_LOGW(nodeName, "cclBufferSize is 0, skip validation."),
                    return ge::GRAPH_SUCCESS);

    int64_t bs = info.bs;
    int64_t h = info.K;
    int64_t topK = info.topK;
    int64_t epWorldSize = info.epWorldSize;
    int64_t expertPerRank = info.moeExpertNum / epWorldSize;

    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, OP_LOGE(nodeName, "y desc is null."), return ge::GRAPH_FAILED);
    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    int64_t leastCclBufferSize =
        (epWorldSize * ops::CeilAlign(epWorldSize * expertPerRank, ALIGN_128) * sizeof(int32_t)) + // tokenPerExpert
        (ops::CeilAlign(bs * topK * h, ALIGN_512) * sizeof(int8_t)) + // quantTokenScale
        (ops::CeilAlign(bs * h * topK * yDtypeSize, ALIGN_512)); // combineOut

    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
        OP_LOGE(nodeName, "cclBufferSize(%ld) should equal or larger than leastCclBufferSize(%ld). "
                "bs=%ld, h=%ld, topK=%ld, epWorldSize=%ld, expertPerRank=%ld, yDtypeSize=%ld.",
            cclBufferSize, leastCclBufferSize, bs, h, topK, epWorldSize, expertPerRank, yDtypeSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3CheckOutputTensor(gert::TilingContext *context, const MegaMoeA2A3TilingData &info)
{
    // ==================== 1. y 输出校验 ====================
    const gert::StorageShape *yStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, 
        OP_LOGE(K_INNER_DEBUG, "y shape is null."), return GRAPH_FAILED);
    
    // 维度数校验：必须为 2 维
    OP_TILING_CHECK(yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(K_INNER_DEBUG, "y must be 2-dimension, but got %lu dim.",
            yStorageShape->GetStorageShape().GetDimNum()), return GRAPH_FAILED);
    
    int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    
    // Shape 一致性校验：y 的 shape 应该与 x 的 shape 一致 [M, K]
    OP_TILING_CHECK(yDim0 != static_cast<int64_t>(info.bs),
        OP_LOGE(K_INNER_DEBUG, "y's dim0 not equal to M, y's dim0 = %ld, M = %u.",
            yDim0, info.bs), return GRAPH_FAILED);
    OP_TILING_CHECK(yDim1 != static_cast<int64_t>(info.K),
        OP_LOGE(K_INNER_DEBUG, "y's dim1 not equal to K, y's dim1 = %ld, K = %u.",
            yDim1, info.K), return GRAPH_FAILED);
    
    // 数据类型校验
    auto yDesc = context->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yDesc == nullptr, 
        OP_LOGE(K_INNER_DEBUG, "y desc is null."), return GRAPH_FAILED);
    ge::DataType yDataType = yDesc->GetDataType();
    OP_TILING_CHECK(yDataType != ge::DT_FLOAT16 && yDataType != ge::DT_BF16,
        OP_LOGE(K_INNER_DEBUG, "y dataType is invalid, should be FLOAT16 or BF16, but got %d.",
            static_cast<int>(yDataType)), return GRAPH_FAILED);
    
    OP_LOGD(K_INNER_DEBUG, "y dim0 = %ld", yDim0);
    OP_LOGD(K_INNER_DEBUG, "y dim1 = %ld", yDim1);
    OP_LOGD(K_INNER_DEBUG, "y dataType = %d", static_cast<int>(yDataType));
    
    // ==================== 2. expert_token_nums 输出校验 ====================
    const gert::StorageShape *expertTokenNumsStorageShape = context->GetOutputShape(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsStorageShape == nullptr, 
        OP_LOGE(K_INNER_DEBUG, "expert_token_nums shape is null."), return GRAPH_FAILED);
    
    // 维度数校验：必须为 1 维
    OP_TILING_CHECK(expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(K_INNER_DEBUG, "expert_token_nums must be 1-dimension, but got %lu dim.",
            expertTokenNumsStorageShape->GetStorageShape().GetDimNum()), return GRAPH_FAILED);
    
    int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);

    // Shape 校验：expert_token_nums 的 dim0 应该等于 expertPerRank（本卡专家数量）
    OP_TILING_CHECK(expertTokenNumsDim0 != static_cast<int64_t>(info.expertPerRank),
        OP_LOGE(K_INNER_DEBUG, "expert_token_nums's dim0 not equal to expertPerRank, expert_token_nums's dim0 = %ld, expertPerRank = %u.",
            expertTokenNumsDim0, info.expertPerRank), return GRAPH_FAILED);
    
    // 数据类型校验
    auto expertTokenNumsDesc = context->GetOutputDesc(OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsDesc == nullptr, 
        OP_LOGE(K_INNER_DEBUG, "expert_token_nums desc is null."), return GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(K_INNER_DEBUG, "expert_token_nums dataType is invalid, should be INT32, but got %d.",
            static_cast<int>(expertTokenNumsDesc->GetDataType())), return GRAPH_FAILED);
    
    OP_LOGD(K_INNER_DEBUG, "expert_token_nums dim0 = %ld", expertTokenNumsDim0);
    OP_LOGD(K_INNER_DEBUG, "expert_token_nums dataType = %d", static_cast<int>(expertTokenNumsDesc->GetDataType()));
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeA2A3TilingFuncImpl(gert::TilingContext *context)
{
    // 涉及SyncAll，设置batch mode模式，所有核同时启动 
    uint32_t batch_mode = 1U; 
    auto ret = context->SetScheduleMode(batch_mode); 

    // 1. tilingData
    MegaMoeA2A3TilingData *tilingData = context->GetTilingData<MegaMoeA2A3TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(K_INNER_DEBUG, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OP_LOGI(K_INNER_DEBUG, "MegaMoeA2A3 get tilingData.");
    MegaMoeA2A3TilingData& info = *tilingData;
    OP_LOGI(K_INNER_DEBUG, "MegaMoeA2A3 get tilingData info.");
    
    OP_TILING_CHECK(MegaMoeA2A3CommAlg(context) != ge::GRAPH_SUCCESS,
        VECTOR_INNER_ERR_REPORT_TILING(K_INNER_DEBUG, "MegaMoeA2A3 CheckCommAlg Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckOutputTensor(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckOutputTensor Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3GetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(MegaMoeA2A3CheckHcclBuffSize(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "MegaMoeA2A3 CheckHcclBuffSize Failed"),
        return ge::GRAPH_FAILED);

    // 2. set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = info.expertPerRank * info.worldSize;
    int64_t activeNum = info.bs * info.topK;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 2;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t quantMode = 1;
    uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
    moeInitRoutingQuantV2TilingBase.DoTiling(info.bs, info.K, info.topK, expertCapacity, expertNum, activeNum, dropPadMode,
        expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantMode, scaleDim0, aivNumInitRouting, ubSize);
    uint64_t initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(K_INNER_DEBUG, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;

    uint64_t megeMoeWorkspace = (info.bs + 256 - 1) / 256 * 256 * info.topK * sizeof(int32_t) + // expandedRowIdx
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
} // namespace optiling
