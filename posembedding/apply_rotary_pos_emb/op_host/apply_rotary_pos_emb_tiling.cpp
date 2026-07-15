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
 * \file apply_rotary_pos_emb_tiling.cpp
 * \brief
 */
#include "apply_rotary_pos_emb_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace {
static const int64_t INPUT0 = 0;
static const int64_t INPUT1 = 1;
static const int64_t INPUT2 = 2;
static const int64_t INPUT3 = 3;
static const int64_t DIM_0 = 0;
static const int64_t DIM_1 = 1;
static const int64_t DIM_2 = 2;
static const int64_t DIM_3 = 3;
static const int64_t DIM_4 = 4;
static const int64_t LASTDIM_64 = 64;
static const int64_t LASTDIM_128 = 128;
static const int64_t BLOCK_SIZE = 32;
static const int64_t REPEAT_FP32 = 64;
static const int64_t REPEAT_FP16 = 128;
static const int64_t ONE_BLOCK_NUM = 8;
static const int64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
static const int64_t MIN_DIM = 16;
static const int64_t COMPUTE_BATCH_MIN = 350;

static const std::vector<std::string> inputNames = {"query", "key", "cos", "sin"};

inline int64_t ComputeTimes(const int64_t value, const int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    int64_t loopTimes = value / factor;
    if (value % factor == 0) {
        loopTimes = loopTimes - 1;
    }
    return loopTimes;
}
} // namespace

namespace optiling {
using namespace Ops::Base;
struct ApplyRotaryPosEmbParams {
    int64_t totalCoreNum = 0;
    int64_t totalUbSize = 0;
    int64_t sysWorkspaceSize = 0;
    int64_t preCoreBatch = 0;
    int64_t lastCoreBatch = 0;
    int64_t oneBlockFp32 = 0;
    int64_t qDims = 0;
    int64_t qDim0 = 0;
    int64_t qDim1 = 0;
    int64_t qDim3 = 0;
    int64_t kDim0 = 0;
    int64_t kDim1 = 0;
    int64_t kDim3 = 0;
    int64_t cosDim0 = 0;
    int64_t cosDim1 = 0;
    int64_t cosDim3 = 0;
    bool isCast = false;
    bool isFp32 = false;
    int64_t castDtypeSize = 0;
    int32_t dtypeSize = 0;
    int64_t oneBlock = 0;
    int64_t useCoreNum = 0;
    int64_t lastDim = 0;
    int64_t halfNum = 0;
    int64_t preCBatchB = 0;
    int64_t preCBatchL = 0;
    int64_t lastCBatchL = 0;
    int64_t comBatchBB = 0;
    int64_t comBatchBBL = 0;
    int64_t comBatchBLL = 0;
    int64_t comBatchLBL = 0;
    int64_t comBatchLLL = 0;
    int64_t qPart1Ub = 0;
    int64_t q2q1Part1Ub = 0;
    int64_t cosPart1Ub = 0;
    int64_t sin1UbSize = 0;
    int64_t preCLTimes = 0;
    int64_t lastCLTimes = 0;
    int64_t preCBBTimes = 0;
    int64_t preCBLTimes = 0;
    int64_t preCLLTimes = 0;
    int64_t qCoreOffset = 0;
    int64_t kCoreOffset = 0;
    int64_t cosCoreOffset = 0;
    int64_t qcNum = 0;
    int64_t kcNum = 0;
    int64_t coscNum = 0;
    int64_t qcdNum = 0;
    int64_t kcdNum = 0;
    int64_t coscdNum = 0;
    int64_t qkcNum = 0;
    int64_t mulNum = 0;
    int64_t qcdHalfNum = 0;
    int64_t dstRepSBr = 0;
    int64_t blockLenQ = 0;
    int64_t srcStrideK = 0;
    int64_t blockLenq2q1 = 0;
    int64_t mask = 0;
    int64_t tilingKey = 0;
    int64_t blockMoveQ = 0;
    int64_t singleN = 0;             // max(qcNum, kcNum)，PERF 模型使用
    platform_ascendc::SocVersion  socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

class ApplyRotaryPosEmbTiling {
public:
    explicit ApplyRotaryPosEmbTiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus GetInputParams(gert::TilingContext *context, ApplyRotaryPosEmbParams &params);
    ge::graphStatus CheckParams(gert::TilingContext *context, ApplyRotaryPosEmbParams &params);
    ge::graphStatus ComputeAB(gert::TilingContext *context, ApplyRotaryPosEmbParams &params);
    ge::graphStatus Compute(gert::TilingContext *context, ApplyRotaryPosEmbParams &params);
    void PrintTilingData(gert::TilingContext *context, ApplyRotaryPosEmbTilingData &tiling,
                         ApplyRotaryPosEmbParams &params);
    void SetTilingData(gert::TilingContext *context, ApplyRotaryPosEmbTilingData &tiling,
                       ApplyRotaryPosEmbParams &params);

private:
    gert::TilingContext *context_ = nullptr;
};

ge::graphStatus ApplyRotaryPosEmbTiling::GetInputParams(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    auto q = context->GetInputTensor(INPUT0);
    OP_CHECK_NULL_WITH_CONTEXT(context, q);
    gert::Shape qShape = q->GetStorageShape();
    params.qDims = qShape.GetDimNum();
    if (params.qDims != DIM_4 && params.qDims != DIM_3) {
        std::string dimStr = std::to_string(params.qDims) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "query",
            dimStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    params.qDim0 = params.qDims == DIM_4 ? qShape.GetDim(DIM_0) : 1;//BSND格式下代表B维度的大小，TND格式下为1;
    params.qDim1 = params.qDims == DIM_4 ? qShape.GetDim(DIM_1) : qShape.GetDim(DIM_0);// BSND格式下代表S维度的大小,TND格式下代表T维度的大小;
    params.qcNum = params.qDims == DIM_4 ? qShape.GetDim(DIM_2) : qShape.GetDim(DIM_1);// N;
    params.qDim3 = params.qDims == DIM_4 ? qShape.GetDim(DIM_3) : qShape.GetDim(DIM_2);// D;
    auto k = context->GetInputTensor(INPUT1);
    OP_CHECK_NULL_WITH_CONTEXT(context, k);
    gert::Shape kShape = k->GetStorageShape();
    int64_t kDims = kShape.GetDimNum();
    if (kDims != params.qDims) {
        std::string dimMsg = std::to_string(kDims) + " and " + std::to_string(params.qDims);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "key and query",
            dimMsg.c_str(), "The shape dims of input key and input query should be the same");
        return ge::GRAPH_FAILED;
    }
    params.kDim0 = kDims == DIM_4 ? kShape.GetDim(DIM_0) : 1;
    params.kDim1 = kDims == DIM_4 ? kShape.GetDim(DIM_1) : kShape.GetDim(DIM_0);
    params.kcNum = kDims == DIM_4 ? kShape.GetDim(DIM_2) : kShape.GetDim(DIM_1);
    params.kDim3 = kDims == DIM_4 ? kShape.GetDim(DIM_3) : kShape.GetDim(DIM_2);
    auto cos = context->GetInputTensor(INPUT2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cos);
    gert::Shape cosShape = cos->GetStorageShape();
    int64_t cosDims = cosShape.GetDimNum();
    if (cosDims != params.qDims) {
        std::string dimMsg = std::to_string(cosDims) + " and " + std::to_string(params.qDims);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "cos and query",
            dimMsg.c_str(), "The shape dims of input cos and input query should be the same");
        return ge::GRAPH_FAILED;
    }
    params.cosDim0 = cosDims == DIM_4 ? cosShape.GetDim(DIM_0) : 1;
    params.cosDim1 = cosDims == DIM_4 ? cosShape.GetDim(DIM_1) : cosShape.GetDim(DIM_0);
    params.coscNum = cosDims == DIM_4 ? cosShape.GetDim(DIM_2) : cosShape.GetDim(DIM_1);
    params.cosDim3 = cosDims == DIM_4 ? cosShape.GetDim(DIM_3) : cosShape.GetDim(DIM_2);
    auto sin = context->GetInputTensor(INPUT3);
    OP_CHECK_NULL_WITH_CONTEXT(context, sin);
    gert::Shape sinShape = sin->GetStorageShape();
    int64_t sinDims = sinShape.GetDimNum();
    if (sinDims != DIM_4 && sinDims != DIM_3) {
        std::string dimStr = std::to_string(sinDims) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "sin",
            dimStr.c_str(), "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    if (cosShape != sinShape) {
        std::string shapeMsg = ToString(cosShape) + " and " + ToString(sinShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "cos and sin",
            shapeMsg.c_str(), "The shapes of input cos and input sin should be the same");
        return ge::GRAPH_FAILED;
    }
    if (qShape.GetShapeSize() == 0 || kShape.GetShapeSize() == 0 || cosShape.GetShapeSize() == 0 ||
        sinShape.GetShapeSize() == 0) {
        std::string shapeSizeMsg = 
            std::to_string(qShape.GetShapeSize()) + ", " + std::to_string(kShape.GetShapeSize()) + 
            ", " + std::to_string(cosShape.GetShapeSize()) + " and " + std::to_string(sinShape.GetShapeSize());
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(context->GetNodeName(), "query, key, cos and sin",
            shapeSizeMsg.c_str(), "All inputs must be non-empty tensors");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTiling::CheckParams(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    OP_CHECK_IF(GetInputParams(context, params) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "GetInputParams failed"), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *layoutAttr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutAttr);
    ApplyRotaryPosEmbLayout layout = static_cast<ApplyRotaryPosEmbLayout>(*layoutAttr);
    if (layout != ApplyRotaryPosEmbLayout::BSND && layout != ApplyRotaryPosEmbLayout::SBND && layout != ApplyRotaryPosEmbLayout::TND) {
        std::string layoutStr = std::to_string(static_cast<int64_t>(layout));
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "layout",
            layoutStr.c_str(), "1(BSND), 2(SBND) or 4(TND)");
        return ge::GRAPH_FAILED;
    }
    if ((layout == ApplyRotaryPosEmbLayout::BSND || layout == ApplyRotaryPosEmbLayout::SBND) && params.qDims != DIM_4) {
        std::string dimStr = std::to_string(params.qDims) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query",
            dimStr.c_str(),
            "The shape dims of input query must be 4 when the attr layout is 1 (BSND) or 2 (SBND)");
        return ge::GRAPH_FAILED;
    }
    if (layout == ApplyRotaryPosEmbLayout::TND && params.qDims != DIM_3) {
        std::string dimStr = std::to_string(params.qDims) + "D";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query",
            dimStr.c_str(),
            "The shape dims of input query must be 3 when the attr layout is 4 (TND)");
        return ge::GRAPH_FAILED;
    }
    auto q = context->GetInputTensor(INPUT0);
    OP_CHECK_NULL_WITH_CONTEXT(context, q);
    gert::Shape qShape = q->GetStorageShape();

    auto k = context->GetInputTensor(INPUT1);
    OP_CHECK_NULL_WITH_CONTEXT(context, k);
    gert::Shape kShape = k->GetStorageShape();

    auto cos = context->GetInputTensor(INPUT2);
    OP_CHECK_NULL_WITH_CONTEXT(context, cos);
    gert::Shape cosShape = cos->GetStorageShape();
    
    if ((params.kDim0 != params.qDim0) || (params.cosDim0 != params.kDim0)) {
        std::string shapeMsg = ToString(qShape) + ", " + ToString(kShape) + " and " + ToString(cosShape);
        std::string reasonMsg = "The batches of input query, key and cos should be equal, "
            "where batch is 1 when the attr layout is 4 (TND), otherwise is the 0th dim of its shape";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query, key and cos",
            shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((params.kDim1 != params.qDim1) || (params.cosDim1 != params.kDim1)) {
        std::string shapeMsg = ToString(qShape) + ", " + ToString(kShape) + " and " + ToString(cosShape);
        std::string reasonMsg =
            "The 1st dims of input query, key and cos should be equal when the attr layout is 1 (BSND), "
            "and the 0th dims of input query, key and cos should be equal when the attr layout is 4 (TND)";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query, key and cos",
            shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (params.cosDim3 % MIN_DIM != 0 || params.cosDim3 < MIN_DIM || params.cosDim3 > LASTDIM_128) {
        std::string shapeStr = ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cos",
            shapeStr.c_str(), "The last dim of input cos should be a multiple of 16 in range [16, 128]");
        return ge::GRAPH_FAILED;
    }
    if (params.qDim3 < params.cosDim3) {
        std::string shapeMsg = ToString(qShape) + " and " + ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query and cos",
            shapeMsg.c_str(),
            "The last dim of input query should be greater than or equal to the last dim of input cos");
        return ge::GRAPH_FAILED;
    }
    if (params.qDim3 % MIN_DIM != 0 || params.qDim3 < MIN_DIM || params.qDim3 > LASTDIM_128) {
        std::string shapeStr = ToString(qShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "query",
            shapeStr.c_str(), "The last dim of input query should be a multiple of 16 in range [16, 128]");
        return ge::GRAPH_FAILED;
    }
    if (params.coscNum != 1) {
        std::string reasonMsg =
            "The N axis of input cos should be 1, "
            "where N refers to the 2nd dim when the attr layout is 1 (BSND), "
            "or the 1st dim when the attr layout is 4 (TND)";
        std::string shapeStr = ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cos",
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context->GetInputDesc(INPUT0) == nullptr, OP_LOGE(context->GetNodeName(), "input 0 get desc failed"),
                return ge::GRAPH_FAILED);
    ge::DataType qDtype = context->GetInputDesc(INPUT0)->GetDataType();
    if (qDtype != ge::DT_BF16 && qDtype != ge::DT_FLOAT && qDtype != ge::DT_FLOAT16) {
        std::string dtypeStr = ToString(qDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "query",
            dtypeStr.c_str(), "BF16, FLOAT or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    for (int32_t i = 1; i < DIM_4; i++) {
        auto desc = context->GetInputDesc(i);
        OP_CHECK_IF(desc == nullptr, OP_LOGE(context->GetNodeName(), "get input[%d] Desc is null !", i),
                    return ge::GRAPH_FAILED);
        ge::DataType inputDtype = desc->GetDataType();
        if (inputDtype != qDtype) {
            std::string paramMsg = inputNames[i] + " and query";
            std::string dtypeMsg = ToString(inputDtype) + " and " + ToString(qDtype);
            std::string reasonMsg = "The dtypes of input " + inputNames[i] + " and input query should be the same";
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), paramMsg.c_str(),
                dtypeMsg.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    params.isCast = qDtype != ge::DT_FLOAT;
    params.isFp32 = qDtype == ge::DT_FLOAT;
    params.dtypeSize = ge::GetSizeByDataType(qDtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTiling::ComputeAB(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    auto oneLoopUb = [&](int64_t n, int64_t qPartCoef, int64_t qDtypeSz) -> int64_t {
        int64_t qPart = n * params.lastDim * qDtypeSz;
        int64_t qCalc = n * params.lastDim * params.castDtypeSize;
        return qPart * qPartCoef + params.cosPart1Ub * 4 + qCalc * 2 +
               static_cast<int64_t>(params.isCast) * (params.sin1UbSize * 2);
    };
    auto moveOutUb = [&](int64_t n, int64_t qPartCoef, int64_t qDtypeSz) -> int64_t {
        return (n * params.lastDim * qDtypeSz) * qPartCoef + params.cosPart1Ub * 4;
    };

    int64_t oneLoopCombine = oneLoopUb(params.qkcNum, 3, params.castDtypeSize);
    int64_t oneLoopSep = oneLoopUb(params.singleN, 4, params.dtypeSize);
    OP_LOGD(context->GetNodeName(), "oneLoop Combine %ld, sep %ld, totalUb %ld", oneLoopCombine, oneLoopSep,
            params.totalUbSize);
    OP_CHECK_IF(oneLoopSep > params.totalUbSize || oneLoopSep <= 0,
                OP_LOGE(context->GetNodeName(), "sep oneLoop is too large or small than 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(oneLoopCombine <= 0,
                OP_LOGE(context->GetNodeName(), "Combine oneLoop small than 0"), return ge::GRAPH_FAILED);

    int64_t timesCombine = params.totalUbSize / oneLoopCombine;
    int64_t shengMteCombine = (params.totalUbSize % oneLoopCombine) / moveOutUb(params.qkcNum, 3, params.castDtypeSize);
    shengMteCombine = params.isCast ? 0 : shengMteCombine;
    if (timesCombine > params.preCoreBatch) {
        timesCombine = params.preCoreBatch;
        shengMteCombine = 0;
    }

    int64_t timesSep = params.totalUbSize / oneLoopSep;
    int64_t shengMteSep = (params.totalUbSize % oneLoopSep) / moveOutUb(params.singleN, 4, params.dtypeSize);
    shengMteSep = params.isCast ? 0 : shengMteSep;
    if (timesSep > params.preCoreBatch) {
        timesSep = params.preCoreBatch;
        shengMteSep = 0;
    }
    int64_t n_rate = params.qcNum > params.kcNum ? params.qcNum / params.kcNum : params.kcNum / params.qcNum;
    bool networkPerf = (params.preCoreBatch >= COMPUTE_BATCH_MIN && n_rate < 2);

    int64_t times = 0;
    int64_t shengMte = 0;

    if ((oneLoopSep < params.totalUbSize && oneLoopCombine >= params.totalUbSize) || networkPerf) {
        bool isPerf = (shengMteSep == 0 || params.isCast);
        params.tilingKey = isPerf
            ? static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB_CAST_PERF)
            : static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB_PERF);
        times = timesSep;
        shengMte = shengMteSep;
        params.qPart1Ub = params.singleN * params.lastDim * params.dtypeSize;
        params.q2q1Part1Ub = params.singleN * params.lastDim * params.castDtypeSize;
    } else if(oneLoopCombine < params.totalUbSize){
        bool isPerf = (shengMteCombine == 0 || params.isCast);
        params.tilingKey = isPerf
            ? static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB_CAST)
            : static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_AB);
        times = timesCombine;
        shengMte = shengMteCombine;
        params.qPart1Ub = params.qkcNum * params.lastDim * params.castDtypeSize;
        params.q2q1Part1Ub = params.qkcNum * params.lastDim * params.castDtypeSize;
    }
    OP_LOGD(context->GetNodeName(), "tilingKey %ld, times %ld, shengMte %ld", params.tilingKey, times, shengMte);

    params.preCBatchB = times + shengMte;
    params.preCLTimes = ComputeTimes(params.preCoreBatch, params.preCBatchB);
    params.preCBatchL = params.preCoreBatch - params.preCBatchB * params.preCLTimes;
    params.lastCLTimes = ComputeTimes(params.lastCoreBatch, params.preCBatchB);
    params.lastCBatchL = params.lastCoreBatch - params.preCBatchB * params.lastCLTimes;

    params.qPart1Ub = params.preCBatchB * params.qPart1Ub;
    params.cosPart1Ub = params.preCBatchB * params.cosPart1Ub;
    params.q2q1Part1Ub = times * params.q2q1Part1Ub;
    params.sin1UbSize = times * params.sin1UbSize;

    params.comBatchBB = times;
    params.preCBBTimes = ComputeTimes(params.preCBatchB, params.comBatchBB);
    params.comBatchBBL = params.preCBatchB - params.preCBBTimes * params.comBatchBB;
    params.preCBLTimes = ComputeTimes(params.preCBatchL, params.comBatchBB);
    params.comBatchBLL = params.preCBatchL - params.preCBLTimes * params.comBatchBB;
    params.preCLLTimes = ComputeTimes(params.lastCBatchL, params.comBatchBB);
    params.comBatchLLL = params.lastCBatchL - params.preCLLTimes * params.comBatchBB;

    params.blockLenQ = params.qcdNum / params.oneBlock;
    params.srcStrideK = params.kcdNum / params.oneBlock;

    params.dstRepSBr = params.lastDim / params.oneBlockFp32;
    params.blockLenq2q1 = params.halfNum / params.oneBlockFp32;
    params.mulNum = params.mulNum * DIM_2 / params.oneBlockFp32;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRotaryPosEmbTiling::Compute(gert::TilingContext *context, ApplyRotaryPosEmbParams &params)
{
    OP_LOGD(context->GetNodeName(), "ApplyRotaryPosEmb compute start");
    params.castDtypeSize = params.isCast ? DIM_4 : params.dtypeSize;
    params.oneBlock = BLOCK_SIZE / params.dtypeSize;                       // 搬运，一个block可以存放的输入数据
    params.oneBlockFp32 = params.isCast ? ONE_BLOCK_NUM : params.oneBlock; // 计算，一个block存放的计算数据量
    OP_LOGD(context->GetNodeName(), "isCast is %d, dtypeSize is %d, isFp32 is %d", params.isCast, params.dtypeSize,
            params.isFp32);

    // 当格式为BSND时，a, b, c, d 分别表示为: dim0(B), dim1(S), dim2(N), dim3(D)
    // 当格式为TND时，a, b, c, d 分别表示为: dim0(1), dim1(T), dim2(N), dim3(D)，为了计算逻辑统一，dim0在前面设置为1
    int64_t ab = params.kDim0 * params.kDim1;                                   // 总batch = BS
    params.preCoreBatch = (ab + params.totalCoreNum - 1) / params.totalCoreNum; // front核每核处理batch数据量
    params.useCoreNum = (ab + params.preCoreBatch - 1) / params.preCoreBatch;   // 使用的核数
    params.lastCoreBatch = ab - (params.useCoreNum - 1) * params.preCoreBatch;  // 尾核处理batch数据量
    OP_LOGD(context->GetNodeName(), "preCoreBatch %ld, lastCoreBatch is %ld", params.preCoreBatch,
            params.lastCoreBatch);
    params.mask = (params.isCast || params.isFp32) ? REPEAT_FP32 : REPEAT_FP16;
    params.lastDim = params.cosDim3;
    params.blockMoveQ = params.lastDim / params.oneBlock;  // 部分维度旋转搬运block数
    params.halfNum = params.lastDim / DIM_2;
    params.singleN = (params.qcNum > params.kcNum) ? params.qcNum : params.kcNum;

    params.qcdNum = params.qcNum * params.lastDim;       // Q_n * lastDim ：这里指的是搬运的q的元素数据量
    params.kcdNum = params.kcNum * params.lastDim;       // K_n * lastDim ：这里指的是搬运的k的元素数据量
    params.coscdNum = params.coscNum * params.cosDim3; // coscNum = 1 --> 1 * D
    params.qkcNum = params.qcNum + params.kcNum;       // (Q_n + K_n), Q、K在N轴上进行拼接
    // kernel侧Mul()中repeatTimes参数为uint8_t类型，避免使用qkcNum传参计算时超出范围
    if (params.qkcNum > UINT8_MAX) {
        gert::Shape qShape = context->GetInputTensor(INPUT0)->GetStorageShape();
        gert::Shape kShape = context->GetInputTensor(INPUT1)->GetStorageShape();
        std::string shapeMsg = ToString(qShape) + " and " + ToString(kShape);
        std::string reasonMsg =
            "The sum of the N axis of input query and the N axis of input key must not exceed "
            + std::to_string(UINT8_MAX) + ", where N is the second-to-last dimension";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query and key",
            shapeMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    params.mulNum = params.qkcNum * params.halfNum;    // (Q_n + K_n) * D / 2
    params.qcdHalfNum = params.qcNum * params.halfNum; // Q_n * D / 2
    // 单核处理的数据偏移 batch * ND
    params.qCoreOffset = params.preCoreBatch * params.qcNum * params.qDim3;
    params.kCoreOffset = params.preCoreBatch * params.kcNum * params.kDim3;
    params.cosCoreOffset = params.preCoreBatch * params.coscdNum;
    // ub size
    params.qPart1Ub = params.qkcNum * params.lastDim * params.castDtypeSize; // 搬运 (Q_n + K_n) * D * 4(fp32)/2(fp16/bf16)
    params.cosPart1Ub = params.coscNum * params.lastDim * params.dtypeSize; // 搬运 1 * D * 4(fp32)/2(fp16/bf16)
    params.q2q1Part1Ub = params.qkcNum * params.lastDim * params.castDtypeSize; // 计算 (Q_n + K_n) * D * 4(bf16/fp32)/2(fp16)
    params.sin1UbSize = params.coscNum * params.lastDim * params.castDtypeSize; // 计算 1 * D * 4， 为BF16 cast使用
    int64_t speUb = params.qPart1Ub * 2 + params.cosPart1Ub * 2 + params.q2q1Part1Ub * 2 +
                    static_cast<int64_t>(params.isCast) * (params.sin1UbSize * 2);
    OP_LOGD(context->GetNodeName(), "speUb is %ld, totalUbSize is %ld", speUb, params.totalUbSize);

    // 小shape，每核处理一个batch
    if (params.preCoreBatch == 1 && speUb <= params.totalUbSize) {
        params.tilingKey = static_cast<int64_t>(ApplyRotaryPosEmbTilingKey::TILINGKEY_SMALL);
        params.mulNum = params.qkcNum * params.lastDim;
        params.blockLenQ = params.halfNum / params.oneBlockFp32; // D/2 占用block数，搬运
        params.dstRepSBr = params.lastDim / params.oneBlockFp32; // D 占用block数，计算
        params.qcdHalfNum = params.lastDim / params.mask; // D 计算时的repeat次数（按列计算），至少会执行一次，不能为0
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(ComputeAB(context, params) != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "ComputeAB failed"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ApplyRotaryPosEmb compute end");
    return ge::GRAPH_SUCCESS;
}

void ApplyRotaryPosEmbTiling::PrintTilingData(gert::TilingContext *context, ApplyRotaryPosEmbTilingData &tiling,
                                              ApplyRotaryPosEmbParams &params)
{
    OP_LOGD(context->GetNodeName(),
            "Print ApplyRotaryPosEmb tilingData: useCoreNum is %ld, lastDim %ld,"
            "halfNum is %ld, preCBatchB %ld, preCBatchL is %ld, lastCBatchL is %ld,"
            "comBatchBB is %ld, comBatchBBL %ld, comBatchBLL is %ld, comBatchLLL is %ld,"
            "qPart1Ub is %ld, q2q1Part1Ub is %ld, cosPart1Ub is %ld, sin1UbSize is %ld,"
            "preCLTimes is %ld, lastCLTimes is %ld, preCBBTimes is %ld, preCBLTimes is %ld,"
            "preCLLTimes is %ld, qCoreOffset is %ld, kCoreOffset is %ld, cosCoreOffset is %ld,"
            "qcNum is %ld, kcNum is %ld, coscNum is %ld, qcdNum is %ld, kcdNum is %ld,"
            "coscdNum is %ld, qkcNum is %ld, mulNum is %ld, qcdHalfNum is %ld, dstRepSBr is %ld,"
            "blockLenQ is %ld, srcStrideK is %ld, blockLenq2q1 is %ld,  mask is %ld, tilingKey is %ld, qDim3 is %ld, kDim3 is %ld, blockMoveQ is %ld, singleN is %ld",
            tiling.get_useCoreNum(), tiling.get_lastDim(), tiling.get_halfNum(), tiling.get_preCBatchB(),
            tiling.get_preCBatchL(), tiling.get_lastCBatchL(), tiling.get_comBatchBB(), tiling.get_comBatchBBL(),
            tiling.get_comBatchBLL(), tiling.get_comBatchLLL(), tiling.get_qPart1Ub(), tiling.get_q2q1Part1Ub(),
            tiling.get_cosPart1Ub(), tiling.get_sin1UbSize(), tiling.get_preCLTimes(), tiling.get_lastCLTimes(),
            tiling.get_preCBBTimes(), tiling.get_preCBLTimes(), tiling.get_preCLLTimes(), tiling.get_qCoreOffset(),
            tiling.get_kCoreOffset(), tiling.get_cosCoreOffset(), params.qcNum, params.kcNum, params.coscNum,
            tiling.get_qcdNum(), tiling.get_kcdNum(), tiling.get_coscdNum(), tiling.get_qkcNum(), tiling.get_mulNum(),
            tiling.get_qcdHalfNum(), tiling.get_dstRepSBr(), tiling.get_blockLenQ(), tiling.get_srcStrideK(),
            tiling.get_blockLenq2q1(), tiling.get_mask(), params.tilingKey, tiling.get_qDim3(), tiling.get_kDim3(), tiling.get_blockMoveQ(), tiling.get_singleN());
}

void ApplyRotaryPosEmbTiling::SetTilingData(gert::TilingContext *context, ApplyRotaryPosEmbTilingData &tiling,
                                            ApplyRotaryPosEmbParams &params)
{
    tiling.set_useCoreNum(params.useCoreNum);
    tiling.set_lastDim(params.lastDim);
    tiling.set_halfNum(params.halfNum);
    tiling.set_preCBatchB(params.preCBatchB);
    tiling.set_preCBatchL(params.preCBatchL);
    tiling.set_lastCBatchL(params.lastCBatchL);
    tiling.set_comBatchBB(params.comBatchBB);
    tiling.set_comBatchBBL(params.comBatchBBL);
    tiling.set_comBatchBLL(params.comBatchBLL);
    tiling.set_comBatchLLL(params.comBatchLLL);
    tiling.set_qPart1Ub(params.qPart1Ub);
    tiling.set_q2q1Part1Ub(params.q2q1Part1Ub);
    tiling.set_cosPart1Ub(params.cosPart1Ub);
    tiling.set_sin1UbSize(params.sin1UbSize);
    tiling.set_preCLTimes(params.preCLTimes);
    tiling.set_lastCLTimes(params.lastCLTimes);
    tiling.set_preCBBTimes(params.preCBBTimes);
    tiling.set_preCBLTimes(params.preCBLTimes);
    tiling.set_preCLLTimes(params.preCLLTimes);
    tiling.set_qCoreOffset(params.qCoreOffset);
    tiling.set_kCoreOffset(params.kCoreOffset);
    tiling.set_cosCoreOffset(params.cosCoreOffset);
    tiling.set_qcdNum(params.qcdNum);
    tiling.set_kcdNum(params.kcdNum);
    tiling.set_coscdNum(params.coscdNum);
    tiling.set_qkcNum(params.qkcNum);
    tiling.set_mulNum(params.mulNum);
    tiling.set_qcdHalfNum(params.qcdHalfNum);
    tiling.set_dstRepSBr(params.dstRepSBr);
    tiling.set_blockLenQ(params.blockLenQ);
    tiling.set_srcStrideK(params.srcStrideK);
    tiling.set_blockLenq2q1(params.blockLenq2q1);
    tiling.set_mask(params.mask);
    tiling.set_qcNum(params.qcNum);
    tiling.set_kcNum(params.kcNum);
    tiling.set_qDim3(params.qDim3);
    tiling.set_kDim3(params.kDim3);
    tiling.set_blockMoveQ(params.blockMoveQ);
    tiling.set_singleN(params.singleN);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(params.useCoreNum);
    context->SetTilingKey(params.tilingKey);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = params.sysWorkspaceSize;
}

static std::unique_ptr<ApplyRotaryPosEmbCompileInfo> aropeCompileInfo = nullptr;

class ApplyRotaryPosMembaseEmbTilingClass : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ApplyRotaryPosMembaseEmbTilingClass(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        // get and check platform information
        if (aropeCompileInfo == nullptr) {
            OP_LOGD(context_->GetNodeName(), "get platform information from ascendc interface.");
            auto platformInfo = context_->GetPlatformInfo();
            OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            params.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
            params.socVersion = ascendcPlatform.GetSocVersion();

            uint64_t platformUbSize = 0;
            platformInfo->GetLocalMemSize(fe::LocalMemType::UB, platformUbSize);
            params.totalUbSize = static_cast<int64_t>(platformUbSize);

            int64_t sysWorkspaceSize = static_cast<int64_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
            params.sysWorkspaceSize = sysWorkspaceSize > WORK_SPACE_SIZE ? sysWorkspaceSize : WORK_SPACE_SIZE;

            aropeCompileInfo = std::make_unique<ApplyRotaryPosEmbCompileInfo>(
            ApplyRotaryPosEmbCompileInfo{params.totalCoreNum, platformUbSize, params.sysWorkspaceSize, params.socVersion});
        } else {
            OP_LOGD(context_->GetNodeName(), "get platform information from compile info.");
            params.totalCoreNum = aropeCompileInfo->numBlocks;
            params.totalUbSize = aropeCompileInfo->ubSize;
            params.sysWorkspaceSize = aropeCompileInfo->sysWorkspaceSize;
            params.socVersion = aropeCompileInfo->socVersion;
        }
        OP_LOGD(context_->GetNodeName(), "totalCoreNum is %ld", params.totalCoreNum);
        OP_CHECK_IF(params.totalCoreNum <= 0, OP_LOGE(context_->GetNodeName(), "PrepareTiling fail to get core num."),
                    return ge::GRAPH_FAILED);
        OP_LOGD(context_->GetNodeName(), "totalUbSize is %ld", params.totalUbSize);
        OP_CHECK_IF(params.totalUbSize <= 0, OP_LOGE(context_->GetNodeName(), "PrepareTiling fail to get ub size."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    bool IsCapable() override
    {
        return true;
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override
    {
        ApplyRotaryPosEmbTiling tilingObject(context_);
        ApplyRotaryPosEmbTilingData tilingData;
        if (tilingObject.CheckParams(context_, params) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "CheckParams return failed.");
            return ge::GRAPH_FAILED;
        }

        if (tilingObject.Compute(context_, params) != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "Compute return failed.");
            return ge::GRAPH_FAILED;
        }

        tilingObject.SetTilingData(context_, tilingData, params);
        tilingObject.PrintTilingData(context_, tilingData, params);
        return ge::GRAPH_SUCCESS;
    }
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }
    
private:
    ApplyRotaryPosEmbParams params;
};

static ge::graphStatus Tiling4ApplyRotaryPosEmb(gert::TilingContext *context)
{
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4ApplyRotaryPosEmb(gert::TilingParseContext *context)
{
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<ApplyRotaryPosEmbCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "compileInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->numBlocks = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(ApplyRotaryPosEmb, ApplyRotaryPosMembaseEmbTilingClass, 40000);

IMPL_OP_OPTILING(ApplyRotaryPosEmb)
    .Tiling(Tiling4ApplyRotaryPosEmb)
    .TilingParse<ApplyRotaryPosEmbCompileInfo>(TilingPrepare4ApplyRotaryPosEmb);
} // namespace optiling
