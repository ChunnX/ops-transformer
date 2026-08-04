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
 * \file apply_rotary_pos_emb_grad_tiling.cpp
 * \brief apply_rotary_pos_emb_grad host 侧 tiling 母板类实现
 *
 * 约束:
 *   输入 3D or 4D。所有 tensor dtype 一致。D <= 1024, D % 2 == 0。
 *   grad_query_embed == grad_query, grad_key_embed == grad_key。
 *   grad_query_embed 和 grad_key_embed 除 N 外各维度相同。
 *   cos == sin。cos 的 N == 1，cos 的 B ∈ {1, grad_query_embed.B}，S/D 与 grad_query_embed 一致。
 *   query == grad_query_embed, key == grad_key_embed。grad_cos == cos, grad_sin == sin。
 *   layout attr: {1=BSND, 2=SBND, 3=BNSD, 4=TND}
 *   → internal Layout: BSND(=0, 含 TND) → BAB; SBND(=1) → AB; BNSD(=2, 预留); NO_BROADCAST(=3) → A
 */
#include <cmath>
#include <graph/utils/type_utils.h>
#include "tiling/tiling_api.h"
#include "posembedding/apply_rotary_pos_emb_grad/op_kernel/arch35/apply_rotary_pos_emb_grad_dag.h"
#include "posembedding/apply_rotary_pos_emb_grad/op_kernel/arch35/apply_rotary_pos_emb_grad_tiling_key.h"
#include "apply_rotary_pos_emb_grad_tiling.h"

namespace {
constexpr int64_t GQ_IDX  = 0; // grad_query_embed
constexpr int64_t GK_IDX  = 1; // grad_key_embed
constexpr int64_t COS_IDX = 2;
constexpr int64_t SIN_IDX = 3;
constexpr int64_t Q_IDX   = 4; // query (optional)
constexpr int64_t K_IDX   = 5; // key   (optional)
constexpr int64_t GQO_IDX = 0; // grad_query
constexpr int64_t GKO_IDX = 1; // grad_key
constexpr int64_t GC_IDX  = 2; // grad_cos
constexpr int64_t GS_IDX  = 3; // grad_sin
constexpr int64_t D_4    = 4, D_3 = 3, D_0 = 0, D_1 = 1, D_2 = 2, D_3I = 3;
constexpr int64_t HCOEF   = 2, DMAX = 1024, DBUF = 2;

const std::vector<ge::DataType> SDTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
static const std::vector<std::string> inNames  = {"grad_query_embed", "grad_key_embed", "cos", "sin", "query", "key"};
static const std::vector<std::string> outNames = {"grad_query", "grad_key", "grad_cos", "grad_sin"};
} // namespace

namespace optiling {
using namespace Ops::Base;

// ==========================================================================
// 硬件信息
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.numBlocks = ascendcPlatform.GetCoreNumAiv();
        uint64_t ub;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        aicoreParams_.ubSize = ub;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    vLength_   = Ops::Base::GetVRegSize(context_);
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// nullptr 检查
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckNullptr()
{
    int64_t maxInput = SIN_IDX;
    if (context_->GetInputShape(Q_IDX) != nullptr && context_->GetInputShape(K_IDX) != nullptr) {
        maxInput = K_IDX;
    }
    for (int64_t i = GQ_IDX; i <= maxInput; i++) {
        OP_CHECK_NULL_IF(context_, context_->GetInputDesc(i));
        OP_CHECK_NULL_IF(context_, context_->GetInputShape(i));
    }
    for (int64_t i = GQO_IDX; i <= GS_IDX; i++) {
        OP_CHECK_NULL_IF(context_, context_->GetOutputDesc(i));
        OP_CHECK_NULL_IF(context_, context_->GetOutputShape(i));
    }
    return ge::GRAPH_SUCCESS;
}

#define OP_CHECK_NULL_IF(ctx, p) \
    do { if ((p) == nullptr) { OP_LOGE(ctx, #p " is nullptr"); return ge::GRAPH_FAILED; } } while(0)

// ==========================================================================
// shape 正数检查
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckInPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inNames[idx].c_str(),
                ToString(shape).c_str(), "can not be empty or negative");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckOutPutShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetOutputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), outNames[idx].c_str(),
                ToString(shape).c_str(), "can not be empty or negative");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShapeAllPositive() const
{
    for (int i : {GQ_IDX, GK_IDX, COS_IDX, SIN_IDX})
        CHK(CheckInPutShapeAllPositive(i));
    for (int i : {GQO_IDX, GKO_IDX})
        CHK(CheckOutPutShapeAllPositive(i));
    if (context_->GetInputShape(Q_IDX) != nullptr && context_->GetInputShape(K_IDX) != nullptr) {
        CHK(CheckInPutShapeAllPositive(Q_IDX));
        CHK(CheckInPutShapeAllPositive(K_IDX));
        CHK(CheckOutPutShapeAllPositive(GC_IDX));
        CHK(CheckOutPutShapeAllPositive(GS_IDX));
    }
    return ge::GRAPH_SUCCESS;
}
#define CHK(expr) do { if ((expr) != ge::GRAPH_SUCCESS) return ge::GRAPH_FAILED; } while(0)

// ==========================================================================
// 维度检查: 仅 3D 或 4D
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShapeDim() const
{
    auto s = [&](int idx) -> auto& { return context_->GetInputShape(idx)->GetStorageShape(); };
    auto so = [&](int idx) -> auto& { return context_->GetOutputShape(idx)->GetStorageShape(); };
    for (auto* sh : {&s(GQ_IDX), &s(GK_IDX), &s(COS_IDX), &s(SIN_IDX), &so(GQO_IDX), &so(GKO_IDX)}) {
        auto d = sh->GetDimNum();
        if (d != D_4 && d != D_3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "tensor",
                std::to_string(d).c_str(), "3D or 4D");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// JudgeLayoutByShape
// 新约束:
//   cos/sin shape 与 grad_query_embed、grad_key_embed 完全一致 → NO_BROADCAST → A
//   cos N==1, cos 其余维 == grad_query_embed → 广播模板
//   layout attr 只用于解释物理维度:
//     1=BSND: (B,S,N,D), cos 支持 (1,S,1,D)/(B,S,1,D)
//     2=SBND: (S,B,N,D), cos 支持 (S,1,1,D)/(S,B,1,D)
//     3=BNSD: 预留, 当前不支持
//     4=TND : (T,N,D),   cos 支持 (T,1,D)
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::JudgeLayoutByShape(
    const gert::Shape& dyQShape, const gert::Shape& dyKShape, const gert::Shape& cosShape, int64_t layoutVal)
{
    if (layoutVal < 1 || layoutVal > 4) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout",
            std::to_string(layoutVal).c_str(), "1(BSND), 2(SBND), 3(BNSD) or 4(TND)");
        return ge::GRAPH_FAILED;
    }
    if (layoutVal == 3) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "layout",
            std::to_string(layoutVal).c_str(), "BNSD is reserved and not supported in this implementation");
        return ge::GRAPH_FAILED;
    }

    isTndLayout_ = (layoutVal == 4);
    int64_t expectDimNum = isTndLayout_ ? D_3 : D_4;
    if (dyQShape.GetDimNum() != expectDimNum || dyKShape.GetDimNum() != expectDimNum ||
        cosShape.GetDimNum() != expectDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "tensor",
            std::to_string(dyQShape.GetDimNum()).c_str(), isTndLayout_ ? "3D for TND" : "4D for BSND/SBND");
        return ge::GRAPH_FAILED;
    }

    if (dyQShape == dyKShape && dyQShape == cosShape) {
        layout_ = ApplyRopeLayout::NO_BROADCAST;
        if (isTndLayout_) {
            cosb_ = 1;
        } else if (layoutVal == 2) {
            cosb_ = cosShape.GetDim(D_1);
        } else {
            cosb_ = cosShape.GetDim(D_0);
        }
        return ge::GRAPH_SUCCESS;
    }

    int64_t xB = 1;
    int64_t xS = 1;
    int64_t cosB = 1;
    int64_t cosS = 1;
    int64_t cosN = 1;
    int64_t dyD = isTndLayout_ ? dyQShape.GetDim(D_2) : dyQShape.GetDim(D_3I);
    int64_t cosD = isTndLayout_ ? cosShape.GetDim(D_2) : cosShape.GetDim(D_3I);

    if (layoutVal == 1) { // BSND
        xB = dyQShape.GetDim(D_0);
        xS = dyQShape.GetDim(D_1);
        cosB = cosShape.GetDim(D_0);
        cosS = cosShape.GetDim(D_1);
        cosN = cosShape.GetDim(D_2);
    } else if (layoutVal == 2) { // SBND
        xS = dyQShape.GetDim(D_0);
        xB = dyQShape.GetDim(D_1);
        cosS = cosShape.GetDim(D_0);
        cosB = cosShape.GetDim(D_1);
        cosN = cosShape.GetDim(D_2);
    } else { // TND
        xB = 1;
        xS = dyQShape.GetDim(D_0);
        cosB = 1;
        cosS = cosShape.GetDim(D_0);
        cosN = cosShape.GetDim(D_1);
    }

    if (cosD != dyD) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and grad_query_embed",
            (ToString(cosShape) + " and " + ToString(dyQShape)).c_str(),
            "cos D must equal grad_query_embed D");
        return ge::GRAPH_FAILED;
    }

    if (cosN != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos",
            ToString(cosShape).c_str(), "The N dim of cos must be 1 unless cos shape equals grad shape");
        return ge::GRAPH_FAILED;
    }

    // cos B 可以 =1 或 =dy B，但 cos S 和 cos D 必须和 dy 一致（N 已检查为 1）
    if (cosS != xS) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and grad_query_embed",
            (ToString(cosShape) + " and " + ToString(dyQShape)).c_str(),
            "cos S must equal grad_query_embed S");
        return ge::GRAPH_FAILED;
    }
    if (cosB != 1 && cosB != xB) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "cos and grad_query_embed",
            (ToString(cosShape) + " and " + ToString(dyQShape)).c_str(),
            "cos B must be 1 or equal grad_query_embed B");
        return ge::GRAPH_FAILED;
    }

    // internal layout 只表示 kernel 模板选择，不表示原始物理 layout。
    if (layoutVal == 4 || (layoutVal == 1 && xB == 1 && cosB == 1)) {
        layout_ = ApplyRopeLayout::BSND;
    } else {
        layout_ = ApplyRopeLayout::SBND;
    }
    cosb_ = cosB;
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// D 轴检查
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    if (d > DMAX) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_query_embed",
            ToString(context_->GetInputShape(GQ_IDX)->GetStorageShape()).c_str(),
            ("D must <= " + std::to_string(DMAX)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (d % HCOEF != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "grad_query_embed",
            ToString(context_->GetInputShape(GQ_IDX)->GetStorageShape()).c_str(),
            "D must be divisible by 2 for half mode");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// shape 一致性检查
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShapeLimit()
{
    const auto& s = [&](int idx) -> auto& { return context_->GetInputShape(idx)->GetStorageShape(); };
    const auto& so = [&](int idx) -> auto& { return context_->GetOutputShape(idx)->GetStorageShape(); };

    auto& gq = s(GQ_IDX), &gk = s(GK_IDX), &c = s(COS_IDX), &si = s(SIN_IDX);
    auto& gqO = so(GQO_IDX), &gkO = so(GKO_IDX);

    // cos == sin
    if (c != si) FAIL("cos and sin", "same shape");
    // grad_query_embed == grad_query, grad_key_embed == grad_key
    if (gq != gqO) FAIL("grad_query_embed and grad_query", "same shape");
    if (gk != gkO) FAIL("grad_key_embed and grad_key", "same shape");

    // grad_query_embed 和 grad_key_embed 除 N 外各维相同
    int nd = gq.GetDimNum();
    if (nd != static_cast<int>(gk.GetDimNum())) FAIL("grad_query_embed and grad_key_embed", "same dim count");
    for (int i = 0; i < nd; i++) {
        bool isN = (nd == D_4 && i == D_2) || (nd == D_3 && i == D_1);
        if (!isN && gq.GetDim(i) != gk.GetDim(i))
            FAIL("grad_query_embed and grad_key_embed", "same dims except N");
    }

    // grad_cos/grad_sin 检查 + DcosFlag
    bool hasXQ = (context_->GetInputShape(Q_IDX) != nullptr);
    bool hasXK = (context_->GetInputShape(K_IDX) != nullptr);
    if (hasXQ && hasXK) {
        dCosFlag_ = 1;
        return CheckOptionalInput();
    }
    return ge::GRAPH_SUCCESS;
}

#define FAIL(a,b) do { \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), a, \
        (ToString(s(GQ_IDX))+" and "+ToString(s(GK_IDX))).c_str(), b); \
    return ge::GRAPH_FAILED; } while(0)

// ==========================================================================
// optional input (query, key) 检查
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckOptionalInput() const
{
    auto& c  = context_->GetInputShape(COS_IDX)->GetStorageShape();
    auto& si = context_->GetInputShape(SIN_IDX)->GetStorageShape();
    auto& gq = context_->GetInputShape(GQ_IDX)->GetStorageShape();
    auto& gk = context_->GetInputShape(GK_IDX)->GetStorageShape();
    auto& gqX  = context_->GetInputShape(Q_IDX)->GetStorageShape();
    auto& gkX  = context_->GetInputShape(K_IDX)->GetStorageShape();
    auto& dc = context_->GetOutputShape(GC_IDX)->GetStorageShape();
    auto& ds = context_->GetOutputShape(GS_IDX)->GetStorageShape();

    if (gqX != gq) FAIL_O("query", "grad_query_embed");
    if (gkX != gk) FAIL_O("key", "grad_key_embed");
    if (c != dc)   FAIL_O("cos", "grad_cos");
    if (si != ds)  FAIL_O("sin", "grad_sin");
    return ge::GRAPH_SUCCESS;
}
#define FAIL_O(a,b) do { \
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), a, \
        (ToString(context_->GetInputShape(0)->GetStorageShape())).c_str(), b); \
    return ge::GRAPH_FAILED; } while(0)

// ==========================================================================
// 总 CheckShape
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckShape()
{
    CHK(CheckShapeDim());
    CHK(CheckShapeLimit());
    isTndLayout_ = (context_->GetInputShape(GQ_IDX)->GetStorageShape().GetDimNum() == D_3);
    int64_t dyD = isTndLayout_ ? context_->GetInputShape(GQ_IDX)->GetStorageShape().GetDim(D_2)
                               : context_->GetInputShape(GQ_IDX)->GetStorageShape().GetDim(D_3I);
    int64_t coD = isTndLayout_ ? context_->GetInputShape(COS_IDX)->GetStorageShape().GetDim(D_2)
                               : context_->GetInputShape(COS_IDX)->GetStorageShape().GetDim(D_3I);
    if (coD != dyD) FAIL("grad_query_embed and cos", "same D");
    CHK(CheckRotaryModeShapeRelation(dyD));
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// dtype + attr 检查
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(GQ_IDX)->GetDataType();
    if (std::find(SDTYPE.begin(), SDTYPE.end(), dtype_) == SDTYPE.end()) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_query_embed",
            ToString(dtype_).c_str(), "FLOAT, BF16 or FLOAT16");
        return ge::GRAPH_FAILED;
    }
    for (int i = GQ_IDX; i <= SIN_IDX; i++) {
        if (context_->GetInputDesc(i)->GetDataType() != dtype_)
            DTYPE_FAIL(inNames[i]);
    }
    if (context_->GetInputShape(Q_IDX) != nullptr && context_->GetInputShape(K_IDX) != nullptr) {
        for (int i = Q_IDX; i <= K_IDX; i++) {
            if (context_->GetInputDesc(i)->GetDataType() != dtype_)
                DTYPE_FAIL(inNames[i]);
        }
    }
    return ge::GRAPH_SUCCESS;
}
#define DTYPE_FAIL(n) do { \
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), (n + " and grad_query_embed").c_str(), \
        (ToString(context_->GetInputDesc(GQ_IDX)->GetDataType()) + " and " + ToString(dtype_)).c_str(), \
        ("dtype of " + n + " must equal grad_query_embed").c_str()); \
    return ge::GRAPH_FAILED; } while(0)

ge::graphStatus ApplyRopeGradRegbaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
        OP_LOGE(context_, "platform info is nullptr."), return ge::GRAPH_FAILED);
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_))
        return ge::GRAPH_SUCCESS;
    CHK(CheckNullptr());
    CHK(CheckDtypeAndAttr());
    CHK(CheckShape());
    CHK(CheckShapeAllPositive());
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// GetShapeAttrsInfo — 入口: rotary_mode="half" only
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // attr 0: rotary_mode (string), only "half" supported
    const std::string* modeStr = attrs->GetAttrPointer<std::string>(0);
    if (modeStr != nullptr && *modeStr != "half") {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "rotary_mode",
            modeStr->c_str(), "\"half\" only");
        return ge::GRAPH_FAILED;
    }
    rotaryMode_ = 0; // HALF

    // attr 1: layout (int), {1=BSND, 2=SBND, 3=BNSD, 4=TND}
    const int64_t* layoutAttr = attrs->GetAttrPointer<int64_t>(1);
    int64_t layoutVal = (layoutAttr != nullptr) ? *layoutAttr : 1;

    CHK(CheckParam());

    dtype_ = context_->GetInputDesc(GQ_IDX)->GetDataType();
    auto dyQShape = context_->GetInputShape(GQ_IDX)->GetStorageShape();
    auto dyKShape = context_->GetInputShape(GK_IDX)->GetStorageShape();
    auto cosShape = context_->GetInputShape(COS_IDX)->GetStorageShape();
    CHK(JudgeLayoutByShape(dyQShape, dyKShape, cosShape, layoutVal));

    if (layoutVal == 4) {
        // TND: (T, N, D) -> b=1, s=T, n=N, d=D.
        d_ = dyQShape.GetDim(D_2);
        b_ = 1;
        s_ = dyQShape.GetDim(D_0);
        nQ_ = dyQShape.GetDim(D_1);
        nK_ = dyKShape.GetDim(D_1);
        cosb_ = 1;
    } else if (layoutVal == 2) {
        // SBND: (S, B, N, D) -> internal b=B, s=S, n=N, d=D.
        d_ = dyQShape.GetDim(D_3I);
        s_ = dyQShape.GetDim(D_0);
        b_ = dyQShape.GetDim(D_1);
        nQ_ = dyQShape.GetDim(D_2);
        nK_ = dyKShape.GetDim(D_2);
        cosb_ = cosShape.GetDim(D_1);
    } else {
        // BSND: (B, S, N, D) -> internal b=B, s=S, n=N, d=D.
        d_ = dyQShape.GetDim(D_3I);
        b_ = dyQShape.GetDim(D_0);
        s_ = dyQShape.GetDim(D_1);
        nQ_ = dyQShape.GetDim(D_2);
        nK_ = dyKShape.GetDim(D_2);
        cosb_ = cosShape.GetDim(D_0);
    }
    return ge::GRAPH_SUCCESS;
}

// ==========================================================================
// Tiling data / Reduce / SetKey
// ==========================================================================
ge::graphStatus ApplyRopeGradRegbaseTilingClass::InitTilingData()
{
    if (tilingData_ == nullptr) {
        tilingData_ = context_->GetTilingData<ApplyRopeGradTilingData>();
        OP_CHECK_IF(tilingData_ == nullptr,
            OP_LOGE(context_->GetNodeName(), "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    }
    memset_s(tilingData_, sizeof(ApplyRopeGradTilingData), 0, sizeof(ApplyRopeGradTilingData));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::GetInputParam(
    Ops::Base::ReduceOpInputParam& opInput, uint32_t inputIdx, uint32_t axesIdx)
{
    auto inputDesc = context_->GetInputDesc(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    opInput.inputDtype = inputDesc->GetDataType();
    auto dyInput = context_->GetInputTensor(inputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dyInput);
    auto dyInputShape = Ops::Transformer::OpTiling::EnsureNotScalar(dyInput->GetStorageShape());
    size_t shapeSize = dyInputShape.GetDimNum();
    opInput.shape.resize(shapeSize);
    for (size_t i = 0; i < shapeSize; i++) opInput.shape[i] = dyInputShape.GetDim(i);
    inputDesc = context_->GetInputDesc(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto cosInput = context_->GetInputTensor(axesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosInput);
    auto cosShapeRef = Ops::Transformer::OpTiling::EnsureNotScalar(cosInput->GetStorageShape());
    for (size_t i = 0; i < shapeSize; i++) {
        if (cosShapeRef.GetDim(i) == 1 && opInput.shape[i] != 1)
            opInput.axes.push_back(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::TilingReduce()
{
    Ops::Base::ReduceOpInputParam opInput;
    CHK(GetInputParam(opInput, GQ_IDX, COS_IDX));
    ge::graphStatus st;
    if (ge::GetSizeByDataType(opInput.inputDtype) == sizeof(float))
        st = Tiling4ReduceOp<ApplyRotaryPosEmbGrad::ApplyRopeGradDag<float, float>::OpDag>(
            context_, opInput, key_, &tilingData_->reduceTiling);
    else
        st = Tiling4ReduceOp<ApplyRotaryPosEmbGrad::ApplyRopeGradDag<Ops::Base::half, float>::OpDag>(
            context_, opInput, key_, &tilingData_->reduceTiling);
    if (st != ge::GRAPH_SUCCESS) OP_LOGE(context_->GetNodeName(), "ReduceOp Tiling failed.");
    return st;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::SetTilingKeyBlockDim(uint32_t dxTilingKey, uint32_t layout)
{
    GEN_REDUCE_TILING_KEY(tilingKey_, key_, dxTilingKey, dCosFlag_);
    tilingKey_ |= (static_cast<uint64_t>(layout) << (key_.totalBitCount + 1));
    OP_LOGI(context_->GetNodeName(),
        "patternID:%u dxTilingKey:%u dCosFlag:%u layout:%u tilingKey:%lu",
        key_.patternID, dxTilingKey, dCosFlag_, layout, tilingKey_);
    context_->SetBlockDim(std::max(usedCoreNum_, static_cast<int64_t>(context_->GetBlockDim())));
    context_->SetTilingKey(tilingKey_);
    if (dCosFlag_ == 1 && layout_ != ApplyRopeLayout::NO_BROADCAST) {
        auto workspaces = context_->GetWorkspaceSizes(1);
        int64_t totalN = nQ_ + nK_;
        workspaces[0] += b_ * s_ * totalN * d_ * ge::GetSizeByDataType(dtype_);
        context_->SetScheduleMode(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyRopeGradRegbaseTilingClass::SetRotaryXTilingData()
{
    int64_t totalN = nQ_ + nK_;
    int64_t dSumNum = b_ * s_ * totalN;
    int64_t blockFactorN = CeilDiv(dSumNum, usedCoreNum_);
    int64_t rotaryUsedCoreNum = CeilDiv(dSumNum, blockFactorN);
    int64_t blockTailFactorN = (dSumNum % blockFactorN == 0) ? blockFactorN : (dSumNum % blockFactorN);
    int64_t dSize = CeilAlign(d_ * GetSizeByDataType(dtype_) / dSplitCoef_, blockSize_) * dSplitCoef_;
    int64_t numDAvail = FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), DBUF * dSize);
    int64_t ubFactorN = std::min(blockFactorN, numDAvail / 3);

    tilingData_->rotaryXParams.b = b_;
    tilingData_->rotaryXParams.s = s_;
    tilingData_->rotaryXParams.nQ = nQ_;
    tilingData_->rotaryXParams.nK = nK_;
    tilingData_->rotaryXParams.d = d_;
    tilingData_->rotaryXParams.usedCoreNum = rotaryUsedCoreNum;
    tilingData_->rotaryXParams.blockFactorN = blockFactorN;
    tilingData_->rotaryXParams.blockTailFactorN = blockTailFactorN;
    tilingData_->rotaryXParams.ubFactorN = ubFactorN;
    tilingData_->rotaryXParams.rotaryMode = rotaryMode_;
    return ge::GRAPH_SUCCESS;
}

uint64_t ApplyRopeGradRegbaseTilingClass::GetTilingKey() const
{
    return tilingKey_;
}

} // namespace optiling
