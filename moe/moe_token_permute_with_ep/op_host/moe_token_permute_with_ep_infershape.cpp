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
 * \file moe_token_permute_with_ep_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t PERMUTE_WITH_EP_INPUT_TOKENS = 0;
static constexpr int64_t PERMUTE_WITH_EP_INPUT_IDX = 1;
static constexpr int64_t PERMUTE_WITH_EP_INPUT_PROBS = 2;
static constexpr int64_t PERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
static constexpr int64_t PERMUTE_WITH_EP_OUTPUT_IDX = 1;
static constexpr int64_t PERMUTE_WITH_EP_OUTPUT_PROBS = 2;
static constexpr int64_t PERMUTE_WITH_EP_ARRT_RANGE = 0;
static constexpr int64_t PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS = 1;

static ge::graphStatus InferShape4MoeTokenPermuteWithEp(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteWithEpInfershape.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(PERMUTE_WITH_EP_INPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *indicesShape = context->GetInputShape(PERMUTE_WITH_EP_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    gert::Shape *permuteXShape = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteXShape);
    gert::Shape *sortedIndices = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedIndices);
    gert::Shape *permuteProbsShape = context->GetOutputShape(PERMUTE_WITH_EP_OUTPUT_PROBS);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteProbsShape);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto rangePtr = attrs->GetAttrPointer<gert::ContinuousVector>(PERMUTE_WITH_EP_ARRT_RANGE);
    const int64_t *numOutTokensPtr = attrs->GetAttrPointer<int64_t>(PERMUTE_WITH_EP_ARRT_NUM_OUT_TOKENS);

    int64_t sortedIndicesLen = NEG_ONE;
    sortedIndices->SetDimNum(DIM_ONE);
    if (sortedIndices->GetDim(0) < 0) {
        sortedIndices->SetDim(0U, NEG_ONE);
    } else {
        int64_t IndicesDimNnum = indicesShape->GetDimNum();
        if (IndicesDimNnum != DIM_TWO && IndicesDimNnum != DIM_ONE) {
            OP_LOGE(context->GetNodeName(), "The dim of indices should be 1 or 2, but got %ld.", IndicesDimNnum);
            return ge::GRAPH_FAILED;
        }
        int64_t topK = (IndicesDimNnum == 1) ? 1 : indicesShape->GetDim(1);
        int64_t N = indicesShape->GetDim(0);
        // 空输入（topK*N==0，其一为 0 另一非负）时输出长度确定为 0，仅含未知维（topK*N<0）时保持未知 -1。
        // 原实现将 N==0 也判为未知（-1），图模式（GEIR/GE）下 GE 按输出 shape [-1] 计算内存大小失败
        // （plog: CalcAlignedSizeByShape ... shape_size[-1] ... failed，rt v2 model 执行失败）。
        sortedIndicesLen = (topK * N > 0) ? topK * N : ((topK * N == 0) ? 0 : NEG_ONE);
        sortedIndices->SetDim(0U, sortedIndicesLen);
    }

    // 与 tiling 侧 CheckAndGetAttrsInfo 同构的防呆校验: 空 list / size != 2 时稳定报错拒绝。
    // 原版无校验直接 GetData() 解引用 RangeList[0]/[1] 为越界读（实验 17 ASAN 实锤:
    // ContinuousVector capacity=0 分配 64 字节, RangeList[1] 越界读 heap-buffer-overflow,
    // 读值随相邻堆内容漂移），此处校验前置消除 UB；[0,0] 默认值形态按原版语义正常读取。
    int64_t start;
    int64_t end;
    if (rangePtr != nullptr) {
        if (rangePtr->GetSize() != DIM_TWO) {
            OP_LOGE(context->GetNodeName(), "the size of range only support 2");
            return ge::GRAPH_FAILED;
        }
        const int64_t *RangeList = reinterpret_cast<const int64_t *>(rangePtr->GetData());
        start = RangeList[0];
        end = RangeList[1];
    } else {
        start = 0;
        end = *numOutTokensPtr;
    }

    *permuteXShape = *xShape;
    // permute_probs 输出 shape 推导：与 permute_tokens 的 dim0 同公式同值（aclnn 文档: shape 为
    // (rangeOptional[1] - rangeOptional[0],) 即 numOutTokens，与 permute_tokens 行数一致）；
    // 输入含未知 shape 时保持未知 (-1,) 透传。
    // 原实现未设置该输出 shape（gert::Shape 保持默认 [1]），图模式（GEIR/GE）下 tiling 强校验
    // "The size of output permuteProbs should be [N] but got [1]" 失败（probs 连接场景全挂，
    // aclnn 直调路径因 shape 由调用方直接给出而未受影响）——图模式能力与 aclnn 拉齐的补齐点。
    permuteProbsShape->SetDimNum(DIM_ONE);
    permuteProbsShape->SetDim(0U, NEG_ONE);
    if (sortedIndicesLen != NEG_ONE && !Ops::Base::IsUnknownRank(*xShape)) {
        int64_t outTokens = (start >= end) ? sortedIndicesLen + (end - start) : (end - start);
        outTokens = std::min(outTokens, sortedIndicesLen);
        outTokens = std::max(outTokens, (int64_t)0);
        permuteXShape->SetDim(0, outTokens);
        permuteProbsShape->SetDim(0U, outTokens);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenPermuteWithEp(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteWithEpInferDataType.");
    auto xDtype = context->GetInputDataType(PERMUTE_WITH_EP_INPUT_TOKENS);
    auto probsDtype = context->GetInputDataType(PERMUTE_WITH_EP_INPUT_PROBS);
    if (probsDtype == ge::DT_UNDEFINED) {
        // probs 可选输入缺省时，permute_probs 输出 dtype 回退为 tokens dtype（aclnn 文档: 数据类型同 tokens）。
        // 若无此回退：probs 未连接/未传入时 GetInputDataType 返回 DT_UNDEFINED 并被直接设为输出 dtype，
        // 图模式构图（GEIR/GE）将在算子选择阶段报 EZ3002（DT_UNDEFINED ... Failed to select engine）而编译失败。
        probsDtype = xDtype;
    }
    context->SetOutputDataType(PERMUTE_WITH_EP_OUTPUT_TOKENS, xDtype);
    context->SetOutputDataType(PERMUTE_WITH_EP_OUTPUT_IDX, ge::DT_INT32);
    context->SetOutputDataType(PERMUTE_WITH_EP_OUTPUT_PROBS, probsDtype);
    OP_LOGD(context->GetNodeName(), "End to do MoeTokenPermuteWithEpInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermuteWithEp)
    .InferShape(InferShape4MoeTokenPermuteWithEp)
    .InferDataType(InferDataType4MoeTokenPermuteWithEp);
} // namespace ops
