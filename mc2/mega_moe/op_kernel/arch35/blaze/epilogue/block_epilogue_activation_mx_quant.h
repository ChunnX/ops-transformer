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
 * \file block_epilogue_activation_mx_quant.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
#define BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H

#if defined(__DAV_C310__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../common/mega_moe_constants.h"
#include "../../common/mega_moe_utils.h"
#include "detail/activation_mx_quant_store.h"
#include "detail/activation_mx_quant_ub_layout.h"
#include "gated_activation.h"
#include "quant/mx_quant.h"

namespace ActivationQuantMsg {
using ActMode = MegaMoeImpl::MegaMoeActMode;
using ActSubMode = MegaMoeImpl::MegaMoeActSubMode;

constexpr uint32_t Y_IDX = 0;
constexpr uint32_t Y_SCALE_IDX = 1;
} // namespace ActivationQuantMsg

namespace MegaMoeImpl {

using namespace AscendC;
using namespace ActivationQuantMsg;

#define BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS \
    template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_, \
              bool IsTensorList_, uint32_t TileM, uint32_t TileN, bool TopkWeightsPrefetch, bool IsInterleaved_>
#define BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS \
    DataTypeOut_, DataTypeIn_, DataTypeX2Scale_, DataTypeX1Scale_, IsTensorList_, TileM, TileN, TopkWeightsPrefetch, \
        IsInterleaved_

template <typename DataTypeOut_, typename DataTypeIn_, typename DataTypeX2Scale_, typename DataTypeX1Scale_,
          bool IsTensorList_, uint32_t TileM = 256, uint32_t TileN = 256, bool TopkWeightsPrefetch = false,
          bool IsInterleaved_ = false>
class BlockEpilogueActivationMxQuant {
public:
    __aicore__ inline BlockEpilogueActivationMxQuant() {}

    static constexpr uint32_t MAX_TILE_M = TileM;
    static constexpr uint32_t MAX_SINGLE_MN = TileM * TileN;

    struct Arguments {
        GM_ADDR yGmAddr{nullptr};
        GM_ADDR yScaleGmAddr{nullptr};
        GM_ADDR x2ScaleGmAddr{nullptr};
        GM_ADDR x1ScaleGmAddr{nullptr};
        GM_ADDR biasGmAddr{nullptr};
        float clampLimit{0.0f};
        uint8_t actMode{static_cast<uint8_t>(ActMode::SWIGLU)};
        uint8_t actSubMode{static_cast<uint8_t>(ActSubMode::DEFAULT)};
        float activationAlpha{1.0f};
        float activationBeta{1.0f};
        Arguments() = default;
    };

    // params
    using Params = Arguments;

    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    using DataTypeX1Scale = DataTypeX1Scale_;
    using DataTypeX2Scale = DataTypeX2Scale_;

    // shape
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BaseOffset = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    // y, yScale, x2Scale, x1Scale, bias
    using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;

public:
    __aicore__ inline void Init(Params const &params);
    __aicore__ inline auto GetFirstL0c2UbTensor();
    __aicore__ inline auto GetSecondL0c2UbTensor();
    __aicore__ inline auto GetTopkWeightTensor();
    // 计算并写回一个激活/量化 tile，不包含流水通知。
    __aicore__ inline void operator()(const BlockShape &blockShape, const BlockCoord &blockCoord,
                                      uint16_t pingpongIdx = 0);
    __aicore__ inline void UpdateGlobalAddr(const BlockCoord &baseOffset);
    __aicore__ inline void UpdateNextProblem(const ProblemShape &problemShape);

private:
    using UbPointers = Detail::ActivationMxQuantUbPointers<DataTypeIn>;

    __aicore__ inline void VFDoActivationForMX(uint16_t mSize, const UbPointers &ubPointers);

    __aicore__ inline void VFDoActivationAndQuantForMX(__ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst,
                                                       __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                                       __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
                                                       __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize,
                                                       uint16_t nSize);

    __aicore__ inline void VFDoActivation(__ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc,
                                          __ubuf__ bfloat16_t *gluResAddr, uint16_t validRowCount,
                                          uint16_t validColumnCount, uint32_t outputRowStrideElements);

    // GM ADDR
    AscendC::GlobalTensor<int8_t> quantOutputGlobal_;
    AscendC::GlobalTensor<int8_t> quantScaleGlobal_;
    GM_ADDR yGmAddr_{nullptr};
    GM_ADDR yScaleGmAddr_{nullptr};

    // UB ADDR
    static constexpr uint32_t kUbFirstOffset =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>().firstInputOffsetBytes;
    static constexpr uint32_t kUbInputElementCapacity =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN,
                                                IsInterleaved_>().firstInputElementCapacity;
    static constexpr uint32_t kUbSecondOffset =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>().secondInputOffsetBytes;
    AscendC::LocalTensor<DataTypeIn> l0cOutUbFirst_{AscendC::TPosition::VECIN, kUbFirstOffset,
                                                    kUbInputElementCapacity};
    AscendC::LocalTensor<DataTypeIn> l0cOutUbSecond_{AscendC::TPosition::VECIN, kUbSecondOffset,
                                                     kUbInputElementCapacity};
    AscendC::LocalTensor<int8_t> quantOutput_;
    AscendC::LocalTensor<int8_t> quantScaleOutput_;
    AscendC::LocalTensor<bfloat16_t> gluRes_;
    AscendC::LocalTensor<uint16_t> maxExp_;
    AscendC::LocalTensor<uint16_t> halfScale_;
    AscendC::LocalTensor<float> weightUb_;

    int64_t n_;
    int64_t scaleN_;
    uint32_t singleM_; // cur singleShapeM
    uint32_t singleN_;

    float clampLimit_{0.0f};
    uint8_t actMode_{static_cast<uint8_t>(ActMode::SWIGLU)};
    uint8_t actSubMode_{static_cast<uint8_t>(ActSubMode::DEFAULT)};
    float activationAlpha_{1.0f};
    float activationBeta_{1.0f};
};

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::Init(
    Params const &params)
{
    if constexpr (g_coreType == AscendC::AIC) {
        return;
    }
    // Params 的生命周期可能短于 Epilogue 对象，因此保存输出 GM 地址。
    yGmAddr_ = params.yGmAddr;
    yScaleGmAddr_ = params.yScaleGmAddr;
    clampLimit_ = params.clampLimit;
    actMode_ = params.actMode;
    actSubMode_ = params.actSubMode;
    activationAlpha_ = params.activationAlpha;
    activationBeta_ = params.activationBeta;

    constexpr auto ubOffsets =
        Detail::BuildActivationMxQuantUbOffsets<DataTypeIn, TileM, TileN, IsInterleaved_>();
    gluRes_ = AscendC::LocalTensor<bfloat16_t>(AscendC::TPosition::VECCALC, ubOffsets.activationOutputOffsetBytes,
                                               ubOffsets.activationQuantElementCapacity);
    quantOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, ubOffsets.quantOutputOffsetBytes,
                                                ubOffsets.activationQuantElementCapacity);
    quantScaleOutput_ = AscendC::LocalTensor<int8_t>(AscendC::TPosition::VECOUT, ubOffsets.quantScaleOffsetBytes,
                                                     ubOffsets.scaleElementCapacity);
    maxExp_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, ubOffsets.maxExpOffsetBytes,
                                             ubOffsets.scaleElementCapacity);
    halfScale_ = AscendC::LocalTensor<uint16_t>(AscendC::TPosition::VECCALC, ubOffsets.reciprocalScaleOffsetBytes,
                                                ubOffsets.scaleElementCapacity);

    // weight UB: 放在 VECIN 区之后的安全区域，仅 mode=1 分配
    if constexpr (TopkWeightsPrefetch) {
        weightUb_ = AscendC::LocalTensor<float>(AscendC::TPosition::VECCALC, ubOffsets.topkWeightOffsetBytes,
                                                ubOffsets.topkWeightElementCapacity);
    }
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateGlobalAddr(
    const BlockCoord &baseOffset)
{
    if constexpr (g_coreType == AscendC::AIV) {
        quantOutputGlobal_.SetGlobalBuffer((__gm__ int8_t *)yGmAddr_ + Get<Y_IDX>(baseOffset));
        quantScaleGlobal_.SetGlobalBuffer((__gm__ int8_t *)yScaleGmAddr_ + Get<Y_SCALE_IDX>(baseOffset));
    }
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::UpdateNextProblem(
    const ProblemShape &problemShape)
{
    n_ = Get<N_VALUE>(problemShape); // n/2
    scaleN_ =
        Ops::Base::CeilDiv(static_cast<uint64_t>(n_), static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivation(
    __ubuf__ DataTypeIn *firstSrc, __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr,
    uint16_t validRowCount, uint16_t validColumnCount, uint32_t outputRowStrideElements)
{
    uint32_t inputRowStrideElements;
    if constexpr (IsInterleaved_) {
        // interleaved源布局为[x1, x2]连续存放在同一行，下一行stride是2*validColumnCount。
        // 注意：2*validColumnCount==实际行距(TileN) 仅在满 tile 成立；tiling 已约束 hiddenDim%256==0，
        // 交织调度宽度为完整 hiddenDim，故交织路径不会出现尾 tile；若未来放宽须改用 TileN。
        inputRowStrideElements = static_cast<uint32_t>(validColumnCount) * 2U;
    } else {
        // 非交织源是两块独立 UB tile，生产端按固定行距 TileN 写入（MakeLayoutC(tileM, L1_TILE_N)）；
        // 尾块 validColumnCount < TileN 时行距不随有效宽度收缩，否则第 1 行起整行错位。
        inputRowStrideElements = TileN;
    }

    Activation::GatedActivationArgs<DataTypeIn> gatedArgs;
    gatedArgs.gate = firstSrc;
    gatedArgs.up = secondSrc;
    gatedArgs.output = gluResAddr;
    if constexpr (TopkWeightsPrefetch) {
        gatedArgs.topkWeights = (__ubuf__ float *)weightUb_.GetPhyAddr();
    }
    gatedArgs.validRowCount = validRowCount;
    gatedArgs.validColumnCount = validColumnCount;
    gatedArgs.inputRowStrideElements = inputRowStrideElements;
    gatedArgs.outputRowStrideElements = outputRowStrideElements;

    Activation::ActivationParams activationParams;
    activationParams.clampLimit = clampLimit_;
    activationParams.alpha = activationAlpha_;
    activationParams.beta = activationBeta_;
    activationParams.mode = static_cast<ActMode>(actMode_);
    activationParams.subMode = static_cast<ActSubMode>(actSubMode_);
    Activation::RunGatedActivation<DataTypeIn, TopkWeightsPrefetch>(gatedArgs, activationParams);
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void
BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivationAndQuantForMX(
    __ubuf__ int8_t *outputDst, __ubuf__ uint16_t *scaleDst, __ubuf__ DataTypeIn *firstSrc,
    __ubuf__ DataTypeIn *secondSrc, __ubuf__ bfloat16_t *gluResAddr, __ubuf__ uint16_t *maxExpAddr,
    __ubuf__ uint16_t *halfScaleLocalAddr, uint16_t mSize, uint16_t nSize)
{
    uint32_t nDstUbAligned =
        Ops::Base::CeilAlign(static_cast<uint32_t>(nSize), static_cast<uint32_t>(AscendC::ONE_BLK_SIZE));
    VFDoActivation(firstSrc, secondSrc, gluResAddr, mSize, nSize, nDstUbAligned);

    MxQuant::MxQuantArgs quantArgs{};
    quantArgs.input = gluResAddr;
    quantArgs.maxExp = maxExpAddr;
    quantArgs.outputScale = scaleDst;
    quantArgs.reciprocalScale = halfScaleLocalAddr;
    quantArgs.output = outputDst;
    quantArgs.dataCount = mSize * nDstUbAligned;
    quantArgs.scaleCount = quantArgs.dataCount / AscendC::ONE_BLK_SIZE;
    MxQuant::RunMxQuant<DataTypeOut>(quantArgs);
    return;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::VFDoActivationForMX(
    uint16_t mSize, const UbPointers &ubPointers)
{
    VFDoActivationAndQuantForMX(ubPointers.quantOutput, ubPointers.quantScale, ubPointers.firstInput,
                                ubPointers.secondInput, ubPointers.activationOutput, ubPointers.maxExp,
                                ubPointers.reciprocalScale, mSize, singleN_);
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetFirstL0c2UbTensor()
{
    return l0cOutUbFirst_;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetSecondL0c2UbTensor()
{
    return l0cOutUbSecond_;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline auto BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::GetTopkWeightTensor()
{
    return weightUb_;
}

BLOCK_EPILOGUE_ACTIVATION_QUANT_CLASS_LOCAL_PARAMS
__aicore__ inline void BlockEpilogueActivationMxQuant<BLOCK_EPILOGUE_DEQUANT_FUNC_LOCAL_PARAMS>::operator()(
    const BlockShape &blockShape, const BlockCoord &blockCoord, uint16_t pingpongIdx)
{
    singleM_ = Get<M_VALUE>(blockShape); // 128
    singleN_ = Get<N_VALUE>(blockShape); // 256

    if (singleM_ == 0) {
        return;
    }

    uint64_t yOffset = Get<Y_IDX>(blockCoord);
    uint64_t yScaleOffset = Get<Y_SCALE_IDX>(blockCoord);
    __ubuf__ DataTypeIn *secondInputBase = nullptr;
    if constexpr (!IsInterleaved_) {
        secondInputBase = (__ubuf__ DataTypeIn *)l0cOutUbSecond_.GetPhyAddr();
    }
    auto ubPointers = Detail::ResolveActivationMxQuantUbPointers<DataTypeIn, MAX_SINGLE_MN, IsInterleaved_>(
        (__ubuf__ DataTypeIn *)l0cOutUbFirst_.GetPhyAddr(), secondInputBase,
        (__ubuf__ bfloat16_t *)gluRes_.GetPhyAddr(), (__ubuf__ int8_t *)quantOutput_.GetPhyAddr(),
        (__ubuf__ uint16_t *)quantScaleOutput_.GetPhyAddr(), (__ubuf__ uint16_t *)maxExp_.GetPhyAddr(),
        (__ubuf__ uint16_t *)halfScale_.GetPhyAddr(), singleN_, pingpongIdx);
    VFDoActivationForMX(singleM_, ubPointers); // switch(x)*y 计算quant quantScale
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    // scale已按compact布局生成，直接copy到GM，省掉原先TransMxScaleLayout重排scale。
    if constexpr (IsInterleaved_) {
        if (pingpongIdx == 1U) {
            AscendC::LocalTensor<int8_t> quantOutputPong = quantOutput_[ubPointers.selectedInt8BufferOffsetElements];
            AscendC::LocalTensor<int8_t> quantScalePong =
                quantScaleOutput_[ubPointers.selectedInt8BufferOffsetElements];
            Detail::StoreQuantOutput<DataTypeOut>(quantOutputGlobal_, quantOutputPong, singleM_, yOffset, n_,
                                                  singleN_);
            Detail::StoreQuantScaleCompact(quantScaleGlobal_, quantScalePong, singleM_, yScaleOffset, scaleN_,
                                           singleN_);
        } else {
            Detail::StoreQuantOutput<DataTypeOut>(quantOutputGlobal_, quantOutput_, singleM_, yOffset, n_, singleN_);
            Detail::StoreQuantScaleCompact(quantScaleGlobal_, quantScaleOutput_, singleM_, yScaleOffset, scaleN_,
                                           singleN_);
        }
    } else {
        Detail::StoreQuantOutput<DataTypeOut>(quantOutputGlobal_, quantOutput_, singleM_, yOffset, n_, singleN_);
        Detail::StoreQuantScaleCompact(quantScaleGlobal_, quantScaleOutput_, singleM_, yScaleOffset, scaleN_,
                                       singleN_);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}

} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // BLOCK_EPILOGUE_ACTIVATION_MX_QUANT_H
