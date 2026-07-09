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
 * \file kv_rms_norm_rope_cache_regbase_dk_split_tiling.cpp
 * \brief Dk放不下UB、Dv可全载场景的Tiling：行方向沿用全载切法，Rope的Dk方向再切dkUbFactor
 */

#include "kv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace optiling {
constexpr int64_t K_SCALE_OFFSET_UB_NUM = 4;
constexpr int64_t COS_SIN_UB_NUM = 4;
constexpr int64_t V_SCALE_OFFSET_UB_NUM = 2;
constexpr static int64_t CONST_TWO = 2;

// Rope的VF尾块以maskAll读满一个VL，各输入buffer的最后一行会越读，与kernel侧的预留保持一致
constexpr static int64_t ROPE_X_UB_HEADROOM_VL_NUM = 2;
constexpr static int64_t ROPE_COS_SIN_UB_HEADROOM_VL_NUM = 1;
constexpr static int64_t ROPE_K_SCALE_UB_HEADROOM_VL_NUM = 1;

constexpr static int64_t CONST_BRCFLAG_ZERO = 0;
constexpr static int64_t CONST_BRCFLAG_ONE = 1;
constexpr static int64_t CONST_BRCFLAG_TWO = 2;

static const std::vector<std::string> inputNames = {
    "kv", "gamma", "cos", "sin", "index", "k_cache", "ckv_cache",
    "k_rope_scale", "c_kv_scale", "k_rope_offset", "c_kv_offset", "v"
};

using namespace Ops::Base;

bool KvRmsNormRopeCacheRegbaseDkSplitTiling::IsCapable()
{
    return isRegbase_;
}

bool KvRmsNormRopeCacheRegbaseDkSplitTiling::CheckScaleOffsetShape(
    const gert::StorageShape* inShape, int64_t lastDim, int64_t& brcFlag)
{
    if (inShape == nullptr) {
        brcFlag = CONST_BRCFLAG_ZERO;
        return true;
    }
    auto shapeSize = inShape->GetStorageShape().GetShapeSize();
    if (shapeSize == lastDim) {
        brcFlag = CONST_BRCFLAG_TWO;
        return true;
    }
    // Regbase模板支持Brc场景
    if (isRegbase_ && shapeSize == 1) {
        brcFlag = CONST_BRCFLAG_ONE;
        return true;
    }
    return false;
}

bool KvRmsNormRopeCacheRegbaseDkSplitTiling::CheckCacheIsQuant(ge::DataType& cacheDtype)
{
    std::vector<ge::DataType> cacheQuantDtypesList = {ge::DataType::DT_INT8, ge::DataType::DT_HIFLOAT8,
                                                  ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2};
    for (const auto& cacheQuantDtype : cacheQuantDtypesList) {
        if (cacheQuantDtype == cacheDtype) {
            return true;
        }
    }
    return false;
}

// 输入shape为非空
ge::graphStatus KvRmsNormRopeCacheRegbaseDkSplitTiling::CheckInputShapeIsEmpty()
{
    for (int i = KV_INDEX; i <= C_KV_OFFSET_IDX; ++i) {
        if (i >= K_ROPE_SCALE_IDX) {
            auto optionalParamShape = context_->GetOptionalInputShape(i);
            if (optionalParamShape != nullptr && optionalParamShape->GetStorageShape().GetShapeSize() == 0) {
                std::string reasonMsg = "The shape of input " + inputNames[i] + " can not be an empty tensor";
                std::string optionalShapeStr = ToString(optionalParamShape->GetStorageShape());
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[i].c_str(),
                    optionalShapeStr.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        else {
            auto inputParam = context_->GetInputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, inputParam);
            gert::Shape inputParamShape = inputParam->GetStorageShape();
            if (inputParamShape.GetShapeSize() == 0) {
                std::string reasonMsg = "The shape of input " + inputNames[i] + " can not be an empty tensor";
                std::string inputShapeStr = ToString(inputParamShape);
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), inputNames[i].c_str(),
                    inputShapeStr.c_str(), reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

bool KvRmsNormRopeCacheRegbaseDkSplitTiling::CheckInputDtype()
{
    // kv dtype
    auto kvDesc = context_->GetInputDesc(KV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kvDesc);
    ge::DataType kvDtype = kvDesc->GetDataType();
    if (kvDtype != ge::DT_FLOAT16 && kvDtype != ge::DT_BF16) {
        std::string kvDtypeStr = ToString(kvDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "kv", kvDtypeStr.c_str(), "FLOAT16, BF16");
        return false;
    }

    // gamma dtype
    auto gammaDesc = context_->GetInputDesc(GAMMA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gammaDesc);
    ge::DataType gammaDtype = gammaDesc->GetDataType();
    if (gammaDtype != kvDtype) {
        std::string dtypeMsg = ToString(gammaDtype) + " and " + ToString(kvDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "gamma and kv", dtypeMsg.c_str(),
            "The dtypes of input gamma and input kv should be the same");
        return false;
    }

    // cos dtype sin dtype
    auto cosDesc = context_->GetInputDesc(COS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosDesc);
    ge::DataType cosDtype = cosDesc->GetDataType();
    auto sinDesc = context_->GetInputDesc(SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sinDesc);
    ge::DataType sinDtype = sinDesc->GetDataType();
    if ((sinDtype != cosDtype) || (sinDtype != kvDtype)) {
        std::string dtypeMsg = ToString(sinDtype) + ", " + ToString(cosDtype) + " and " + ToString(kvDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "sin, cos and kv", dtypeMsg.c_str(),
            "The dtypes of input sin, cos and kv should be the same");
        return false;
    }

    // index dtype
    auto indexDesc = context_->GetInputDesc(INDEX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
    ge::DataType indexDtype = indexDesc->GetDataType();
    if (indexDtype != ge::DT_INT64) {
        std::string indexDtypeStr = ToString(indexDtype);
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "index", indexDtypeStr.c_str(), "INT64");
        return false;
    }

    // k_cache dtype
    auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kcacheDesc);
    ge::DataType kcacheDtype = kcacheDesc->GetDataType();
    if ((kcacheDtype != kvDtype) && (!CheckCacheIsQuant(kcacheDtype))) {
        std::string dtypeMsg = ToString(kcacheDtype);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "k_cache", dtypeMsg.c_str(),
            "The dtype of input k_cache should be INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2, "
            "or the same as the dtype of input kv");
        return false;
    }
    if (CheckCacheIsQuant(kcacheDtype)) {
        // k_rope_scale
        auto kRopeScaleDesc = context_->GetOptionalInputDesc(K_ROPE_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kRopeScaleDesc);
        ge::DataType kRopeScaleDtype = kRopeScaleDesc->GetDataType();
        if (kRopeScaleDtype != ge::DT_FLOAT) {
            std::string kRopeScaleDtypeStr = ToString(kRopeScaleDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "k_rope_scale", kRopeScaleDtypeStr.c_str(),
                "The dtype of input k_rope_scale should be FLOAT "
                "when the dtype of input k_cache is INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2");
            return false;
        }
        // k_rope_offset
        auto kRopeOffsetDesc = context_->GetOptionalInputDesc(K_ROPE_OFFSET_IDX);
        if (kRopeOffsetDesc != nullptr) {
            ge::DataType kRopeOffsetDtype = kRopeOffsetDesc->GetDataType();
            if (kRopeOffsetDtype != ge::DT_FLOAT) {
                std::string kRopeOffsetDtypeStr = ToString(kRopeOffsetDtype);
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "k_rope_offset",
                    kRopeOffsetDtypeStr.c_str(),
                    "The dtype of input k_rope_offset should be FLOAT "
                    "when the dtype of input k_cache is INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2");
                return false;
            }
        }
    }

    // ckv_cache dtype
    auto vcacheDesc = context_->GetInputDesc(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vcacheDesc);
    ge::DataType vcacheDtype = vcacheDesc->GetDataType();
    if ((vcacheDtype != kvDtype) && (!CheckCacheIsQuant(vcacheDtype))) {
        std::string dtypeMsg = ToString(vcacheDtype);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "ckv_cache", dtypeMsg.c_str(),
            "The dtype of input ckv_cache should be INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2, "
            "or the same as the dtype of input kv");
        return false;
    }
    if (CheckCacheIsQuant(vcacheDtype)) {
        // c_kv_scale
        auto ckvScaleDesc = context_->GetOptionalInputDesc(C_KV_SCALE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ckvScaleDesc);
        ge::DataType ckvScaleDtype = ckvScaleDesc->GetDataType();
        if (ckvScaleDtype != ge::DT_FLOAT) {
            std::string ckvScaleDtypeStr = ToString(ckvScaleDtype);
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "c_kv_scale", ckvScaleDtypeStr.c_str(),
                "The dtype of input c_kv_scale should be FLOAT "
                "when the dtype of input ckv_cache is INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2");
            return false;
        }
        // v_kv_offset
        auto vKvOffsetDesc = context_->GetOptionalInputDesc(C_KV_OFFSET_IDX);
        if (vKvOffsetDesc != nullptr) {
            ge::DataType vKvOffsetDtype = vKvOffsetDesc->GetDataType();
            if (vKvOffsetDtype != ge::DT_FLOAT) {
                std::string vKvOffsetDtypeStr = ToString(vKvOffsetDtype);
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "v_kv_offset", vKvOffsetDtypeStr.c_str(),
                    "The dtype of input v_kv_offset should be FLOAT "
                    "when the dtype of input ckv_cache is INT8, HIFLOAT8, FLOAT8_E4M3FN or FLOAT8_E5M2");
                return false;
            }
        }
    }
    return true;
}

/*
NZ排布下kCache的内轴长度dk0 = 32B / sizeof(k_cache dtype)。
Rope的实部与虚部分别落在dk的前后半段，两段都必须落在dk0边界上，否则块内偏移无法表达。
*/
bool KvRmsNormRopeCacheRegbaseDkSplitTiling::CheckNzInnerAxisAligned() const
{
    if (currentCacheMode_ != CacheMode::PA_NZ && currentCacheMode_ != CacheMode::PA_BLK_NZ) {
        return true;
    }
    auto kcacheDesc = context_->GetInputDesc(K_CACHE_INDEX);
    if (kcacheDesc == nullptr) {
        return false;
    }
    ge::DataType kcacheDtype = kcacheDesc->GetDataType();
    std::vector<ge::DataType> quantDtypes = {ge::DataType::DT_INT8, ge::DataType::DT_HIFLOAT8,
                                             ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2};
    int64_t kCacheDtypeSize = kvDtypeSize_;
    for (const auto& quantDtype : quantDtypes) {
        if (quantDtype == kcacheDtype) {
            kCacheDtypeSize = static_cast<int64_t>(sizeof(int8_t));
            break;
        }
    }
    int64_t dk0 = ubBlockSize_ / kCacheDtypeSize;
    return (dk0 > 0) && (dk_ % (CONST_TWO * dk0) == 0);
}

/*
给定dk子块大小，算出UB还能放下多少行。
固定开销：gamma、rope的workspace、k/v的scale与offset、以及三个输入buffer的越读余量；
每行开销：x、cos/sin、输出、rmsnorm的workspace(dv全载)。
*/
int64_t KvRmsNormRopeCacheRegbaseDkSplitTiling::CalcUbFactor(int64_t dkUbFactor) const
{
    int64_t kvAlignNum = ubBlockSize_ / kvDtypeSize_;
    int64_t b8AlignNum = ubBlockSize_ / static_cast<int64_t>(sizeof(int8_t));
    int64_t halfDkUbFactor = dkUbFactor / CONST_TWO;
    int64_t dkUbFactorAlign = CeilAlign(dkUbFactor, kvAlignNum);
    int64_t dkUbFactorB8Align = CeilAlign(dkUbFactor, b8AlignNum);
    int64_t halfDkUbFactorAlign = CeilAlign(halfDkUbFactor, kvAlignNum);
    int64_t dvAlign = CeilAlign(dv_, kvAlignNum);
    int64_t dvB8Align = CeilAlign(dvAlign, b8AlignNum);

    int64_t kScaleOffsetUbSize =
        kScaleType_ > 0 ? K_SCALE_OFFSET_UB_NUM * halfDkUbFactorAlign * static_cast<int64_t>(sizeof(float)) +
                              ROPE_K_SCALE_UB_HEADROOM_VL_NUM * vlFp32_ * static_cast<int64_t>(sizeof(float))
                        : 0;
    int64_t vScaleOffsetUbSize =
        vScaleType_ > 0 ? V_SCALE_OFFSET_UB_NUM * dvAlign * static_cast<int64_t>(sizeof(float)) : 0;
    int64_t gammaUbSize = dvAlign * kvDtypeSize_;
    int64_t ropeWspSize = CONST_TWO * vlFp32_ * static_cast<int64_t>(sizeof(float));
    int64_t xHeadroomSize = ROPE_X_UB_HEADROOM_VL_NUM * vlFp32_ * kvDtypeSize_;
    int64_t cosSinHeadroomSize = ROPE_COS_SIN_UB_HEADROOM_VL_NUM * vlFp32_ * kvDtypeSize_;

    int64_t inCosSinUbSize = COS_SIN_UB_NUM * halfDkUbFactorAlign * kvDtypeSize_;
    int64_t inUbSize = std::max(dkUbFactorAlign, dvAlign) * kvDtypeSize_;
    int64_t kOutUbsize = 0;
    int64_t vOutUbsize = 0;
    // k量化场景
    if (kScaleType_ > 0) {
        if (isOutputKv_) {
            kOutUbsize = dkUbFactorB8Align * kvDtypeSize_ + dkUbFactorB8Align * static_cast<int64_t>(sizeof(int8_t));
        } else {
            kOutUbsize = dkUbFactorB8Align * static_cast<int64_t>(sizeof(int8_t));
        }
    } else {
        kOutUbsize = dkUbFactorAlign * kvDtypeSize_;
    }
    // v量化场景
    if (vScaleType_ > 0) {
        vOutUbsize = dvB8Align * kvDtypeSize_ + dvB8Align * static_cast<int64_t>(sizeof(int8_t));
    } else {
        vOutUbsize = dvAlign * kvDtypeSize_;
    }
    int64_t outUbSize = std::max(kOutUbsize, vOutUbsize);
    int64_t rmsNormWspSize = dvAlign * static_cast<int64_t>(sizeof(float));

    int64_t fixedUbSize = UB_RESERVED_BYTE + kScaleOffsetUbSize + vScaleOffsetUbSize + gammaUbSize + ropeWspSize +
                          DOUBLE_BUFFER * xHeadroomSize + DOUBLE_BUFFER * cosSinHeadroomSize;
    int64_t perRowUbSize = DOUBLE_BUFFER * inUbSize + DOUBLE_BUFFER * inCosSinUbSize + DOUBLE_BUFFER * outUbSize +
                           rmsNormWspSize;
    if (static_cast<int64_t>(ubSize_) <= fixedUbSize || perRowUbSize <= 0) {
        return 0;
    }
    return (static_cast<int64_t>(ubSize_) - fixedUbSize) / perRowUbSize;
}

ge::graphStatus KvRmsNormRopeCacheRegbaseDkSplitTiling::DoOpTiling()
{
    auto kvShapeTuple = GetShapeTuple(context_, KV_INDEX);
    int64_t batchSize = std::get<SHAPE_IDX_B>(kvShapeTuple);
    int64_t seqLen = std::get<SHAPE_IDX_S>(kvShapeTuple);
    int64_t numHead = std::get<SHAPE_IDX_N>(kvShapeTuple);
    auto scale1Shape = context_->GetOptionalInputShape(K_ROPE_SCALE_IDX);
    auto scale2Shape = context_->GetOptionalInputShape(C_KV_SCALE_IDX);
    auto offset1Shape = context_->GetOptionalInputShape(K_ROPE_OFFSET_IDX);
    auto offset2Shape = context_->GetOptionalInputShape(C_KV_OFFSET_IDX);

    OP_CHECK_IF(CheckInputShapeIsEmpty() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "The input param can not be empty"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !CheckScaleOffsetShape(scale1Shape, dk_, kScaleType_),
        OP_LOGE(context_->GetNodeName(), "k_rope_scale shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(scale2Shape, dv_, vScaleType_),
        OP_LOGE(context_->GetNodeName(), "c_kv_scale shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(offset1Shape, dk_, kOffsetType_),
        OP_LOGE(context_->GetNodeName(), "k_rope_offset shape invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckScaleOffsetShape(offset2Shape, dv_, vOffsetType_),
        OP_LOGE(context_->GetNodeName(), "c_kv_offset shape invalid."), return ge::GRAPH_FAILED);
    if ((dk_ % CONST_TWO) != 0) {
        auto cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
        std::string reasonMsg = "The D axis of input cos should be even, where D refers to the " +
            std::to_string(SHAPE_IDX_D) + "th dim";
        std::string cosShapeStr = ToString(cosShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "cos",
            cosShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (numHead != 1) {
        auto kvShape = context_->GetInputShape(KV_INDEX)->GetStorageShape();
        std::string reasonMsg = "The N axis of input kv should be 1, where N refers to the " +
            std::to_string(SHAPE_IDX_N) + "th dim";
        std::string kvShapeStr = ToString(kvShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "kv",
            kvShapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        !CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "kvrmsnormrope dtype is invalid."),
        return ge::GRAPH_FAILED);
    if (currentCacheMode_ == CacheMode::Norm) {
        OP_CHECK_IF(
            !CheckKCacheValid(context_, batchSize, numHead, cacheLength_, dk_),
            OP_LOGE(context_->GetNodeName(), "k_cache shape invalid."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            !CheckVCacheValid(context_, batchSize, numHead, cacheLength_, dv_),
            OP_LOGE(context_->GetNodeName(), "ckv_cache shape invalid."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            !CheckKCacheValidPA(context_, numHead, dk_), OP_LOGE(context_->GetNodeName(), "k_cache shape invalid."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            !CheckVCacheValidPA(context_, numHead, dv_), OP_LOGE(context_->GetNodeName(), "ckv_cache shape invalid."),
            return ge::GRAPH_FAILED);
    }

    // NZ排布对dk的对齐要求不满足时，交给recompute模板
    OP_CHECK_IF(
        !CheckNzInnerAxisAligned(),
        OP_LOGI(context_->GetNodeName(), "Dk split template is not capable. dk %ld is not aligned for NZ cache", dk_),
        return ge::GRAPH_PARAM_INVALID);

    // dk子块以2*VL_FP32为粒度，保证半块(实部/虚部)长度是VL_FP32的整数倍，
    // 从而cos/sin与kCache的NZ内轴偏移都落在对齐边界上。
    // CalcUbFactor对子块大小单调不增，二分取能放下的最大子块(dk可达数十亿，不能线性扫)。
    int64_t dkUbStep = CONST_TWO * vlFp32_;
    int64_t dkUbFactor = 0;
    int64_t ubFactor = 0;
    int64_t low = dkUbStep;
    int64_t high = CeilAlign(dk_, dkUbStep);
    while (low <= high) {
        int64_t mid = FloorDiv(low + (high - low) / CONST_TWO, dkUbStep) * dkUbStep;
        if (mid < dkUbStep) {
            break;
        }
        int64_t midUbFactor = CalcUbFactor(mid);
        if (midUbFactor >= 1) {
            dkUbFactor = mid;
            ubFactor = midUbFactor;
            low = mid + dkUbStep;
        } else {
            high = mid - dkUbStep;
        }
    }
    OP_CHECK_IF(
        (dkUbFactor <= 0),
        OP_LOGI(context_->GetNodeName(), "Dk split template is not capable. dv is %ld, dk is %ld", dv_, dk_),
        return ge::GRAPH_PARAM_INVALID);

    // N = 1
    int64_t bs = batchSize * seqLen * numHead;
    tilingData_.set_batchSize(batchSize);
    tilingData_.set_numHead(numHead);
    tilingData_.set_seqLength(seqLen);
    tilingData_.set_cacheLength(cacheLength_);
    tilingData_.set_blockSize(blockSize_);
    tilingData_.set_reciprocal(reciprocal_);
    tilingData_.set_epsilon(epsilon_);
    tilingData_.set_cosSinNeedBrc(cosSinNeedBrc_);
    tilingData_.set_kScaleType(kScaleType_);
    tilingData_.set_kOffsetType(kOffsetType_);
    tilingData_.set_vScaleType(vScaleType_);
    tilingData_.set_vOffsetType(vOffsetType_);
    tilingData_.set_cacheMode(currentCacheMode_);

    int64_t kvAlignNum = ubBlockSize_ / kvDtypeSize_;
    int64_t b8AlignNum = ubBlockSize_ / static_cast<int64_t>(sizeof(int8_t));
    int64_t halfDkUbFactor = dkUbFactor / CONST_TWO;
    int64_t dkLoop = CeilDiv(dk_, dkUbFactor);
    tilingData_.set_dk(dk_);
    tilingData_.set_halfDk(dk_ / CONST_TWO);
    tilingData_.set_dkUbFactor(dkUbFactor);
    tilingData_.set_dkUbFactorAlign(CeilAlign(dkUbFactor, kvAlignNum));
    tilingData_.set_dkUbFactorB8Align(CeilAlign(dkUbFactor, b8AlignNum));
    tilingData_.set_halfDkUbFactor(halfDkUbFactor);
    tilingData_.set_halfDkUbFactorAlign(CeilAlign(halfDkUbFactor, kvAlignNum));
    tilingData_.set_dkLoop(dkLoop);
    tilingData_.set_dkTail(dk_ - (dkLoop - 1) * dkUbFactor);

    int64_t dvAlign = CeilAlign(dv_, kvAlignNum);
    int64_t dvB8Align = CeilAlign(dvAlign, b8AlignNum);
    tilingData_.set_dv(dv_);
    tilingData_.set_dvAlign(dvAlign);
    tilingData_.set_dvB8Align(dvB8Align);

    int64_t dkUbFactorAlign = tilingData_.get_dkUbFactorAlign();
    int64_t dkUbFactorB8Align = tilingData_.get_dkUbFactorB8Align();
    int64_t inUbSize = std::max(dkUbFactorAlign, dvAlign) * kvDtypeSize_;
    int64_t kOutUbsize = 0;
    int64_t vOutUbsize = 0;
    if (kScaleType_ > 0) {
        if (isOutputKv_) {
            kOutUbsize = dkUbFactorB8Align * kvDtypeSize_ + dkUbFactorB8Align * static_cast<int64_t>(sizeof(int8_t));
        } else {
            kOutUbsize = dkUbFactorB8Align * static_cast<int64_t>(sizeof(int8_t));
        }
    } else {
        kOutUbsize = dkUbFactorAlign * kvDtypeSize_;
    }
    if (vScaleType_ > 0) {
        vOutUbsize = dvB8Align * kvDtypeSize_ + dvB8Align * static_cast<int64_t>(sizeof(int8_t));
    } else {
        vOutUbsize = dvAlign * kvDtypeSize_;
    }
    tilingData_.set_inUbSize(inUbSize);
    tilingData_.set_outUbSize(std::max(kOutUbsize, vOutUbsize));
    tilingData_.set_rmsNormWspSize(dvAlign * static_cast<int64_t>(sizeof(float)));

    int64_t blockFactor = (bs + coreNum_ - 1) / coreNum_;
    ubFactor = ubFactor < blockFactor ? ubFactor : blockFactor;
    usedCoreNum_ = (bs + blockFactor - 1) / blockFactor;
    tilingData_.set_blockFactor(blockFactor);
    tilingData_.set_ubFactor(ubFactor);
    int64_t outputKvKey = isOutputKv_ == true ? 1 : 0;
    tilingData_.set_isOutputKv(outputKvKey);
    tilingKey_ = DK_SPLIT_BASE_TILING_KEY;
    OP_LOGI(context_->GetNodeName(),
        "Dk split template selected. dk %ld dv %ld dkUbFactor %ld dkLoop %ld ubFactor %ld", dk_, dv_, dkUbFactor,
        dkLoop, ubFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvRmsNormRopeCacheRegbaseDkSplitTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(usedCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(KvRmsNormRopeCache, KvRmsNormRopeCacheRegbaseDkSplitTiling, TEMPLATE_D_DK_SPLIT_PRIORITY);
} // namespace optiling
