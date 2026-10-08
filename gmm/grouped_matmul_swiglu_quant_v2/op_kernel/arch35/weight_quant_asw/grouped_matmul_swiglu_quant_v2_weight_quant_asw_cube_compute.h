/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_asw_cube_compute.h
 * \brief
 */

#pragma once

#include "weight_quant_tool_asw.h"
#include "weight_quant_grouped_matmul_asw_block.h"
#include "../weight_quant_basic_block/gmmsq_weight_quant_cube_compute_tools.h"
#include "grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_data.h"

using AscendC::DataCopy;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::WaitFlag;
using namespace WeightQuantTools;
using namespace WeightQuantGroupedMatmul::Arch35;
using AswParams = GMMSQArch35Tiling::GMMSQWeightQuantAswParams;

namespace GMMSQWeightQuantAsw {
struct GMMSQL1BufferCfg {
    uint32_t bL1SizeElem;
    uint32_t aL1SizeElem;
    uint32_t xScaleL1SizeElem;
    uint32_t wScaleL1SizeElem;
    uint32_t scaleKL1Size;
};

template <typename xType, typename wType, typename xScaleType, typename weightScaleType>
__aicore__ constexpr GMMSQL1BufferCfg GetGMMSQL1BufferCfg()
{
    if constexpr (IsMxA8HiF4<xType, wType, weightScaleType>()) {
        return {.bL1SizeElem = 64 * GetKBUnit<wType>(),
                .aL1SizeElem = 128 * GetKBUnit<xType>(),
                .xScaleL1SizeElem = 16 * GetKBUnit<xScaleType>(),
                .wScaleL1SizeElem = 32 * GetKBUnit<weightScaleType>(),
                .scaleKL1Size = 2048U};
    }
    return {.bL1SizeElem = 64 * GetKBUnit<wType>(),
            .aL1SizeElem = 128 * GetKBUnit<xType>(),
            .xScaleL1SizeElem = 32 * GetKBUnit<xScaleType>(),
            .wScaleL1SizeElem = 32 * GetKBUnit<weightScaleType>(),
            .scaleKL1Size = 4096U};
}

#define GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS                                                                        \
    template <typename xType, typename wType, typename xScaleType, typename weightScaleType>
#define GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS xType, wType, xScaleType, weightScaleType

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
class GMMSQWeightQuantASWCubeCompute {
public:
    __aicore__ inline GMMSQWeightQuantASWCubeCompute()
    {
    }

    __aicore__ inline void Init(LocalTensor<float> &yF32Buffer, uint32_t l0cSizeKB,
                                const TCubeTiling *__restrict mmTilingDataIn);
    __aicore__ inline void UpdateGlobalAddr(__gm__ xType *xGmAddr, __gm__ wType *weightGmAddr,
                                            __gm__ xScaleType *xScaleGmAddr, __gm__ weightScaleType *weightScaleGmAddr);
    __aicore__ inline void MatmulCompute(const ASWTilingParam &params, const ASWOffsetParam &offset,
                                         bool enablePreAlloc);
    __aicore__ inline void GetTensorC(const ASWTilingParam &params);
    __aicore__ inline void EndSync();

protected:
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void CopyXGmToL1(const ASWTilingParam &params, const ASWOffsetParam &offset, uint64_t kL1Size,
                                       uint64_t kOffset, bool enablePreAlloc);
    __aicore__ inline void CopyWeightGmToL1(const ASWTilingParam &params, const ASWOffsetParam &offset,
                                            uint64_t kL1Size, uint64_t kOffset, bool enablePreAlloc);
    __aicore__ inline void CopyXScaleGmToL1(const ASWTilingParam &params, const ASWOffsetParam &offset,
                                            uint64_t aScaleL1GroupNum, uint64_t kOffset, bool enablePreAlloc);
    __aicore__ inline void CopyWeightScaleGmToL1(const ASWTilingParam &params, const ASWOffsetParam &offset,
                                                 uint64_t bScaleL1GroupNum, uint64_t kOffset, bool enablePreAlloc);
    __aicore__ inline void IterateMatmul(const ASWTilingParam &params, uint64_t kL1Size, uint64_t aScaleL1GroupNum,
                                         uint64_t bScaleL1GroupNum, uint64_t kInnerIdx, uint64_t kOffset);
    __aicore__ inline void InitEventIds();

    static constexpr GMMSQL1BufferCfg L1_BUFFER_CFG = GetGMMSQL1BufferCfg<xType, wType, xScaleType, weightScaleType>();

    static constexpr uint32_t B_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<wType>();
    static constexpr uint32_t A_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<xType>();
    static constexpr uint32_t A_SCALE_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<xScaleType>();
    static constexpr uint32_t B_SCALE_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<weightScaleType>();

    static constexpr uint32_t B_L0_OFFSET_ELEM = (L0B_BUFFER_SIZE_KB >> 1) * GetKBUnit<wType>();
    static constexpr uint32_t A_L0_OFFSET_ELEM = (L0A_BUFFER_SIZE_KB >> 1) * GetKBUnit<xType>();

    static constexpr uint64_t MX_A_SCALE_GROUP_SIZE = MX_GROUP_SIZE_32;
    static constexpr uint64_t MX_B_SCALE_GROUP_SIZE =
        IsSameType<weightScaleType, hif4_scale>::value ? MX_GROUP_SIZE_64 : MX_GROUP_SIZE_32;

    using aL0DataType = AscendC::mx_fp8_e4m3_t;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<half> xScaleGlobal_;
    GlobalTensor<half> wScaleGlobal_;

    LocalTensor<xType> aL1Tensor_;
    LocalTensor<wType> bL1Tensor_;
    LocalTensor<xScaleType> aScaleL1Tensor_;
    LocalTensor<weightScaleType> bScaleL1Tensor_;

    LocalTensor<aL0DataType> aL0Tensor_{AscendC::TPosition::A2, 0, L0A_BUFFER_SIZE_KB *GetKBUnit<xType>()};
    LocalTensor<wType> bL0Tensor_{AscendC::TPosition::B2, 0, L0B_BUFFER_SIZE_KB *GetKBUnit<wType>()};
    LocalTensor<float> cL0Tensor_;

    LocalTensor<float> yF32Buffer_;

    uint32_t l0cSizeElem_ = 0;
    const TCubeTiling *__restrict mmTilingData_ = nullptr;

    uint64_t l1DbFlag_ = 0;
    uint64_t scaleL1DbFlag_ = 0;
    uint64_t l0DbFlag_ = 0;
    uint64_t l0cOffset_ = 0;
    uint64_t kBlockSize_ = 0;
    uint64_t totalKBlocks_ = 0;
};

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::Init(
    LocalTensor<float> &yF32Buffer, uint32_t l0cSizeKB, const TCubeTiling *__restrict mmTilingDataIn)
{
    mmTilingData_ = mmTilingDataIn;
    l0cSizeElem_ = l0cSizeKB * GetKBUnit<float>();
    kBlockSize_ = mmTilingData_->stepKa * mmTilingData_->baseK;
    totalKBlocks_ = Ops::Base::CeilDiv(static_cast<uint64_t>(mmTilingData_->Ka), kBlockSize_);

    yF32Buffer_ = yF32Buffer;
    InitCubeBuffers();
    InitEventIds();
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::InitCubeBuffers()
{
    cL0Tensor_ = {AscendC::TPosition::CO1, 0, l0cSizeElem_};

    // L1 Buffer
    //
    // 0 ~ 256   | bL1 ping | aL1 ping | bScaleL1 ping | aScaleL1 ping |  |
    // 256 ~ 512 | bL1 pong | aL1 pong | bScaleL1 pong | aScaleL1 pong |  |
    //
    // 每块buffer的起始地址为ping的起始地址，buffer大小为单块buffer + L1_SIZE/2 (256KB)，覆盖ping的头和pong的尾
    // 通过dbFlag和L1_OFFSET_ELEM (256KB)控制db倒换
    uint64_t l1BufOffset = 0;

    bL1Tensor_ = AscendC::LocalTensor<wType>(AscendC::TPosition::TSCM, l1BufOffset,
                                             L1_BUFFER_CFG.bL1SizeElem + B_L1_OFFSET_ELEM);
    if constexpr (AscendC::IsSupportB4<wType>()) {
        l1BufOffset += L1_BUFFER_CFG.bL1SizeElem >> 1;
    } else {
        l1BufOffset += L1_BUFFER_CFG.bL1SizeElem * sizeof(wType);
    }

    aL1Tensor_ = AscendC::LocalTensor<xType>(AscendC::TPosition::TSCM, l1BufOffset,
                                             L1_BUFFER_CFG.aL1SizeElem + A_L1_OFFSET_ELEM);
    l1BufOffset += L1_BUFFER_CFG.aL1SizeElem * sizeof(xType);

    bScaleL1Tensor_ = AscendC::LocalTensor<weightScaleType>(AscendC::TPosition::TSCM, l1BufOffset,
                                                            L1_BUFFER_CFG.wScaleL1SizeElem + B_SCALE_L1_OFFSET_ELEM);

    l1BufOffset += L1_BUFFER_CFG.wScaleL1SizeElem * sizeof(weightScaleType);
    aScaleL1Tensor_ = AscendC::LocalTensor<xScaleType>(AscendC::TPosition::TSCM, l1BufOffset,
                                                       L1_BUFFER_CFG.xScaleL1SizeElem + A_SCALE_L1_OFFSET_ELEM);
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(
    __gm__ xType *xGmAddr, __gm__ wType *weightGmAddr, __gm__ xScaleType *xScaleGmAddr,
    __gm__ weightScaleType *weightScaleGmAddr)
{
    xGlobal_.SetGlobalBuffer(xGmAddr);
    wGlobal_.SetGlobalBuffer(weightGmAddr);
    xScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(xScaleGmAddr));
    wScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weightScaleGmAddr));
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::MatmulCompute(
    const ASWTilingParam &params, const ASWOffsetParam &offset, bool enablePreAlloc)
{
    uint64_t kOffset = 0;
    uint64_t aScaleL1GroupNum = 0;
    uint64_t bScaleL1GroupNum = 0;

    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks_; kOutIdx++) {
        uint64_t kL1Size = Min<uint64_t>(params.k - kOffset, kBlockSize_);

        if (kOffset % L1_BUFFER_CFG.scaleKL1Size == 0) {
            uint64_t scaleKRemainingAlign = Ops::Base::CeilAlign(params.k - kOffset, MX_SCALE_DIVISOR_SIZE);
            aScaleL1GroupNum = Ops::Base::CeilDiv(Min<uint64_t>(scaleKRemainingAlign, L1_BUFFER_CFG.scaleKL1Size),
                                                  MX_A_SCALE_GROUP_SIZE);
            bScaleL1GroupNum = Ops::Base::CeilDiv(Min<uint64_t>(scaleKRemainingAlign, L1_BUFFER_CFG.scaleKL1Size),
                                                  MX_B_SCALE_GROUP_SIZE);
            WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + static_cast<uint8_t>(scaleL1DbFlag_));
            CopyXScaleGmToL1(params, offset, aScaleL1GroupNum, kOffset, enablePreAlloc);
            CopyWeightScaleGmToL1(params, offset, bScaleL1GroupNum, kOffset, enablePreAlloc);
        }

        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));
        CopyXGmToL1(params, offset, kL1Size, kOffset, enablePreAlloc);
        CopyWeightGmToL1(params, offset, kL1Size, kOffset, enablePreAlloc);

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));

        uint64_t kInnerBlockCount = Ops::Base::CeilDiv(kL1Size, static_cast<uint64_t>(mmTilingData_->baseK));
        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            IterateMatmul(params, kL1Size, aScaleL1GroupNum, bScaleL1GroupNum, kInnerIdx, kOffset);
            kOffset += mmTilingData_->baseK;
        }

        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));
        l1DbFlag_ ^= 1;
        if (kOffset % L1_BUFFER_CFG.scaleKL1Size == 0 || kOffset >= params.k) {
            SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + static_cast<uint8_t>(scaleL1DbFlag_));
            scaleL1DbFlag_ ^= 1;
        }
    }
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::CopyXGmToL1(
    const ASWTilingParam &params, const ASWOffsetParam &offset, uint64_t kL1Size, uint64_t kOffset, bool enablePreAlloc)
{
    uint64_t aGmOffset = offset.offsetX + kOffset;

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = params.singleCoreM;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcDValue = params.k;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = Ops::Base::CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(aL1Tensor_[l1DbFlag_ * A_L1_OFFSET_ELEM], xGlobal_[aGmOffset], nd2nzParams, enablePreAlloc);
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::CopyWeightGmToL1(
    const ASWTilingParam &params, const ASWOffsetParam &offset, uint64_t kL1Size, uint64_t kOffset, bool enablePreAlloc)
{
    uint64_t nAlign = Ops::Base::CeilAlign(params.n, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t nL1HalfSize = params.singleCoreN / SWIGLU_N_HALF_FACTOR;
    uint64_t kBlockCount = Ops::Base::CeilDiv(kL1Size, C0<wType>::value);
    uint64_t weightGmOffset = offset.offsetWeight + kOffset * nAlign;

    // 矩阵左半边 参与swish计算部分
    // DataCopyPad不支持直接往L1搬运fp4，转换成B8搬运，步长和数据长度等需要除2
    DataCopyPad2D(bL1Tensor_[l1DbFlag_ * B_L1_OFFSET_ELEM].template ReinterpretCast<uint8_t>(),
                  wGlobal_[weightGmOffset].template ReinterpretCast<uint8_t>(), kBlockCount,
                  (nL1HalfSize * C0<wType>::value) >> 1, (params.singleCoreN * C0<wType>::value) >> 1,
                  (nAlign * C0<wType>::value) >> 1, enablePreAlloc);

    // 矩阵右半边
    uint64_t weightRightGmExtraOffset = nAlign / SWIGLU_N_HALF_FACTOR * C0<wType>::value;
    uint64_t weightRightL1Offset =
        Ops::Base::CeilAlign(nL1HalfSize, static_cast<uint64_t>(BLOCK_CUBE)) * C0<wType>::value;
    DataCopyPad2D(bL1Tensor_[l1DbFlag_ * B_L1_OFFSET_ELEM + weightRightL1Offset].template ReinterpretCast<uint8_t>(),
                  wGlobal_[weightGmOffset + weightRightGmExtraOffset].template ReinterpretCast<uint8_t>(), kBlockCount,
                  (nL1HalfSize * C0<wType>::value) >> 1, (params.singleCoreN * C0<wType>::value) >> 1,
                  (nAlign * C0<wType>::value) >> 1, enablePreAlloc);
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::CopyXScaleGmToL1(
    const ASWTilingParam &params, const ASWOffsetParam &offset, uint64_t aScaleL1GroupNum, uint64_t kOffset,
    bool enablePreAlloc)
{
    uint64_t kSizeAlign = Ops::Base::CeilAlign(params.k, MX_SCALE_DIVISOR_SIZE);
    uint64_t aScaleKGmGroupNum = Ops::Base::CeilDiv(kSizeAlign, MX_A_SCALE_GROUP_SIZE);
    Dn2NzParams aScaleDn2NzParams;
    ConfigE8M0ScaleDn2NzParams(params.singleCoreM, aScaleKGmGroupNum, aScaleL1GroupNum, aScaleDn2NzParams);
    uint64_t aScaleGmOffset = (offset.offsetPerTokenScale + kOffset / MX_A_SCALE_GROUP_SIZE) / B8_IN_B16_NUM;
    DataCopy(aScaleL1Tensor_[scaleL1DbFlag_ * A_SCALE_L1_OFFSET_ELEM].template ReinterpretCast<half>(),
             xScaleGlobal_[aScaleGmOffset], aScaleDn2NzParams, enablePreAlloc);
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::CopyWeightScaleGmToL1(
    const ASWTilingParam &params, const ASWOffsetParam &offset, uint64_t bScaleL1GroupNum, uint64_t kOffset,
    bool enablePreAlloc)
{
    uint64_t kSizeAlign = Ops::Base::CeilAlign(params.k, MX_SCALE_DIVISOR_SIZE);
    uint64_t bScaleKGmGroupNum = Ops::Base::CeilDiv(kSizeAlign, MX_B_SCALE_GROUP_SIZE);
    Dn2NzParams bScaleDn2NzParams;
    uint64_t bScaleGmOffset = offset.offsetAntiquantScale + kOffset / MX_B_SCALE_GROUP_SIZE;
    uint64_t bScaleRightGmExtraOffset = params.n / SWIGLU_N_HALF_FACTOR * bScaleKGmGroupNum;
    uint64_t bScaleRightL1Offset = params.singleCoreN / SWIGLU_N_HALF_FACTOR * bScaleL1GroupNum;
    if constexpr (IsSameType<weightScaleType, hif4_scale>::value) {
        ConfigHif4ScaleDn2NzParams(params.singleCoreN / SWIGLU_N_HALF_FACTOR, bScaleKGmGroupNum, bScaleL1GroupNum,
                                   bScaleDn2NzParams);
        bScaleGmOffset *= B32_IN_B16_NUM;
        bScaleRightGmExtraOffset *= B32_IN_B16_NUM;
    } else {
        ConfigE8M0ScaleDn2NzParams(params.singleCoreN / SWIGLU_N_HALF_FACTOR, bScaleKGmGroupNum, bScaleL1GroupNum,
                                   bScaleDn2NzParams);
        bScaleGmOffset /= B8_IN_B16_NUM;
        bScaleRightGmExtraOffset /= B8_IN_B16_NUM;
    }

    // 矩阵左半边
    DataCopy(bScaleL1Tensor_[scaleL1DbFlag_ * B_SCALE_L1_OFFSET_ELEM].template ReinterpretCast<half>(),
             wScaleGlobal_[bScaleGmOffset], bScaleDn2NzParams, enablePreAlloc);

    // 矩阵右半边
    DataCopy(
        bScaleL1Tensor_[scaleL1DbFlag_ * B_SCALE_L1_OFFSET_ELEM + bScaleRightL1Offset].template ReinterpretCast<half>(),
        wScaleGlobal_[bScaleGmOffset + bScaleRightGmExtraOffset], bScaleDn2NzParams, enablePreAlloc);
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::IterateMatmul(
    const ASWTilingParam &params, uint64_t kL1Size, uint64_t aScaleL1GroupNum, uint64_t bScaleL1GroupNum,
    uint64_t kInnerIdx, uint64_t kOffset)
{
    GMMSQWeightQuant::L0CopyAndCalcParams l0Params{};
    l0Params.mL1Size = params.singleCoreM;
    l0Params.nL1Size = params.singleCoreN;
    l0Params.mL0Size = l0Params.mL1Size;
    l0Params.nL0Size = l0Params.nL1Size;
    l0Params.kL1Size = kL1Size;
    l0Params.kL0Size = Min<uint64_t>(params.k - kOffset, mmTilingData_->baseK);
    l0Params.aScaleL1GroupNum = aScaleL1GroupNum;
    l0Params.bScaleL1GroupNum = bScaleL1GroupNum;
    l0Params.isFirstKLoop = kOffset == 0;
    l0Params.isLastKLoop = params.k - kOffset <= mmTilingData_->baseK;

    uint64_t kL1Offset = kInnerIdx * mmTilingData_->baseK;
    uint64_t aL1Offset = kL1Offset * Ops::Base::CeilAlign(l0Params.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t bL1Offset = kL1Offset * Ops::Base::CeilAlign(l0Params.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t aScaleL1Offset = (kOffset % L1_BUFFER_CFG.scaleKL1Size) / MX_A_SCALE_GROUP_SIZE * BLOCK_CUBE;
    uint64_t bScaleL1Offset = (kOffset % L1_BUFFER_CFG.scaleKL1Size) / MX_B_SCALE_GROUP_SIZE * BLOCK_CUBE;

    LocalTensor<aL0DataType> aL0 = aL0Tensor_[l0DbFlag_ * A_L0_OFFSET_ELEM];
    LocalTensor<wType> bL0 = bL0Tensor_[l0DbFlag_ * B_L0_OFFSET_ELEM];

    if (l0Params.isFirstKLoop) {
        uint64_t nextL0cBufSize = Ops::Base::CeilAlign(l0Params.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                                  Ops::Base::CeilAlign(l0Params.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
        if (l0cOffset_ + nextL0cBufSize > l0cSizeElem_) {
            l0cOffset_ = 0;
        }
    }

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + static_cast<uint8_t>(l0DbFlag_));

    GMMSQWeightQuant::LoadAAndScaleL1ToL0<aL0DataType, xType, xScaleType, true>(
        aL0, aL1Tensor_[l1DbFlag_ * A_L1_OFFSET_ELEM + aL1Offset],
        aScaleL1Tensor_[scaleL1DbFlag_ * A_SCALE_L1_OFFSET_ELEM + aScaleL1Offset], l0Params);

    GMMSQWeightQuant::LoadBAndScaleL1ToL0<wType, wType, weightScaleType, true,
                                         IsMxA8HiF4<xType, wType, weightScaleType>()>(
        bL0, bL1Tensor_[l1DbFlag_ * B_L1_OFFSET_ELEM + bL1Offset],
        bScaleL1Tensor_[scaleL1DbFlag_ * B_SCALE_L1_OFFSET_ELEM + bScaleL1Offset], l0Params);

    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT_ID);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT_ID);

    GMMSQWeightQuant::MmadCompute<float, aL0DataType, wType, false>(cL0Tensor_[l0cOffset_], aL0, bL0, l0Params);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + static_cast<uint8_t>(l0DbFlag_));
    l0DbFlag_ ^= 1;
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void
GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::GetTensorC(const ASWTilingParam &params)
{
    GMMSQWeightQuant::FixL0CToDstParams fixParams{};
    fixParams.mL0Size = params.singleCoreM;
    fixParams.nL0Size = params.singleCoreN;
    fixParams.outNSize = params.singleCoreN;

    GMMSQWeightQuant::FixL0CToDst<float, float, false>(yF32Buffer_, cL0Tensor_[l0cOffset_], fixParams);

    l0cOffset_ += Ops::Base::CeilAlign(fixParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                  Ops::Base::CeilAlign(fixParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::InitEventIds()
{
    for (uint8_t idx = 0; idx < DOUBLE_BUFFER; ++idx) {
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + idx);
        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + idx);
        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + idx);
    }
}

GMMSQ_WQ_ASW_CUBE_TEMPLATE_CLASS_PARAMS
__aicore__ inline void GMMSQWeightQuantASWCubeCompute<GMMSQ_WQ_ASW_CUBE_TEMPLATE_FUNC_PARAMS>::EndSync()
{
    for (uint8_t idx = 0; idx < DOUBLE_BUFFER; ++idx) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + idx);
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + idx);
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + idx);
    }
}

} // namespace GMMSQWeightQuantAsw
