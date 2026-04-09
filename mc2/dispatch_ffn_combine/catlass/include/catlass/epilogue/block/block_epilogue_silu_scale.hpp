/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SILU_SCALE_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SILU_SCALE_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Epilogue::Block {

template <
    class OutputType_,
    class InputType_>
class BlockEpilogue<
    EpilogueSiluScale,
    OutputType_,
    InputType_>
{
public:
    using DispatchPolicy = EpilogueSiluScale;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;

    static constexpr uint32_t STAGES = 2;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, bool isNTD, uint32_t l1BufAddrStart = 0)
    {
        // Allocate UB space
        siluScaleDealSize = 10240;
        castDealSize = isNTD ? 20480 : 16;
        wsInitSize = isNTD ? 63488 : 16;

        for (uint32_t i = 0; i < STAGES; i++) {
            uint64_t offset = 0;
            lsUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(l1BufAddrStart +
                siluScaleDealSize * sizeof(ElementInput) * i);
            lpTmpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(l1BufAddrStart +
                siluScaleDealSize * sizeof(ElementInput) * STAGES + siluScaleDealSize * sizeof(ElementInput) * i);
            lpUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(l1BufAddrStart +
                siluScaleDealSize * sizeof(ElementInput) * STAGES * 2 + siluScaleDealSize * sizeof(ElementOutput) * i);

            // offset = siluScaleDealSize * sizeof(ElementInput) * STAGES * 2 +
            //          siluScaleDealSize * sizeof(ElementOutput) * STAGES;
            offset = 0;
            loFp32UbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(l1BufAddrStart +
                offset + castDealSize * sizeof(ElementInput) * i);
            loCastUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOutput>(l1BufAddrStart +
                offset + castDealSize * sizeof(ElementInput) * STAGES + castDealSize * sizeof(ElementOutput) * i);
            // offset += castDealSize * sizeof(ElementInput) * STAGES +
            //           castDealSize * sizeof(ElementOutput) * STAGES;
            offset = 0;
            zeroTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(l1BufAddrStart +
                offset + wsInitSize * sizeof(ElementInput) * i);
        }
    }

    CATLASS_DEVICE
    ~BlockEpilogue() {}

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementInput> gS, AscendC::GlobalTensor<ElementOutput> gP,
        uint32_t actualQBlkSize, uint32_t actualKvseqlen, uint32_t tmpBlockIdx, float scale)
    {
        // silu
        siluScale = scale;
        uint32_t totalLen = actualQBlkSize * actualKvseqlen;
        uint32_t dealLoop = CeilDiv(actualQBlkSize * actualKvseqlen, siluScaleDealSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t i = 0; i < dealLoop; i++) {
            if ((i % 2 == 0 && tmpBlockIdx % 2 == 0) || (i % 2 != 0 && tmpBlockIdx % 2 != 0)) {
                uint32_t actualDealLen = i < dealLoop - 1 ? siluScaleDealSize : totalLen - i * siluScaleDealSize;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(pingPongFlag);
                AscendC::DataCopyExtParams dataCopyParams{1,
                    static_cast<uint32_t>(actualDealLen * sizeof(ElementInput)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<ElementInput> padParams{false, 0, 0, 0};
                uint32_t gmOffset = i * siluScaleDealSize;
                AscendC::DataCopyPad(lsUbTensor[pingPongFlag], gS[gmOffset], dataCopyParams, padParams);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingPongFlag);

                uint16_t srcRepeatSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
                uint16_t dstRepeatSize = static_cast<uint16_t>(AscendC::GetVecLen() / sizeof(ElementInput));
                uint16_t repeatTimes = static_cast<uint16_t>(CeilDiv(actualDealLen, srcRepeatSize));
                auto *dst = (__ubuf__ ElementInput*)lpTmpUbTensor[pingPongFlag].GetPhyAddr();
                auto *src = (__ubuf__ ElementInput*)lsUbTensor[pingPongFlag].GetPhyAddr();
                SiluScaleCompute<ElementInput, ElementInput>(dst, src, actualDealLen, srcRepeatSize,
                                                             dstRepeatSize, repeatTimes);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<ElementOutput, ElementInput>(lpUbTensor[pingPongFlag], lpTmpUbTensor[pingPongFlag],
                                                           AscendC::RoundMode::CAST_ROUND, actualDealLen);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);

                // ub -> gm/L1
                AscendC::DataCopyParams dataCopyOutyParams;
                dataCopyOutyParams.blockCount = 1;
                dataCopyOutyParams.blockLen = actualDealLen * sizeof(ElementOutput);
                dataCopyOutyParams.srcStride = 0;
                dataCopyOutyParams.dstStride = 0;
                AscendC::DataCopyPad(gP[gmOffset], lpUbTensor[pingPongFlag], dataCopyOutyParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(pingPongFlag);
                pingPongFlag = 1 - pingPongFlag;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementInput> gOFp32, AscendC::GlobalTensor<ElementOutput> gO,
        uint32_t actualQBlkSize, uint32_t embSize, uint32_t numHeads, uint32_t tmpBlockIdx, bool isNTD)
    {
        uint32_t totalLen = actualQBlkSize * embSize;
        uint32_t dealLoop = CeilDiv(actualQBlkSize * embSize, castDealSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t i = 0; i < dealLoop; i++) {
            if ((i % 2 == 0 && tmpBlockIdx % 2 == 0) || (i % 2 != 0 && tmpBlockIdx % 2 != 0)) {
                uint32_t actualDealLen = i < dealLoop - 1 ? castDealSize : totalLen - i * castDealSize;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(pingPongFlag);
                AscendC::DataCopyExtParams dataCopyParams{1,
                    static_cast<uint32_t>(actualDealLen * sizeof(ElementInput)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<ElementInput> padParams{false, 0, 0, 0};
                uint32_t gmOffset = i * castDealSize;
                AscendC::DataCopyPad(loFp32UbTensor[pingPongFlag], gOFp32[gmOffset], dataCopyParams, padParams);

                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingPongFlag);
                AscendC::Cast<ElementOutput, ElementInput>(loCastUbTensor[pingPongFlag], loFp32UbTensor[pingPongFlag],
                                                           AscendC::RoundMode::CAST_ROUND, actualDealLen);

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);

                AscendC::DataCopyParams dataCopyOutyParams;
                dataCopyOutyParams.blockCount = 1;
                dataCopyOutyParams.blockLen = actualDealLen * sizeof(ElementOutput);
                dataCopyOutyParams.srcStride = 0;
                dataCopyOutyParams.dstStride = 0;
                uint32_t gmOutOffset = isNTD ? gmOffset :
                                               gmOffset / embSize * (embSize * numHeads) + gmOffset % embSize;
                AscendC::DataCopyPad(gO[gmOutOffset], loCastUbTensor[pingPongFlag], dataCopyOutyParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(pingPongFlag);
                pingPongFlag = 1 - pingPongFlag;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementInput> gOFp32, uint32_t initSize, ElementInput initValue,
        uint32_t tmpBlockIdx)
    {
        uint32_t dealLoop = CeilDiv(initSize, wsInitSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        for (uint32_t i = 0; i < dealLoop; i++) {
            if ((i % 2 == 0 && tmpBlockIdx % 2 == 0) || (i % 2 != 0 && tmpBlockIdx % 2 != 0)) {
                uint32_t actualDealLen = i < dealLoop - 1 ? wsInitSize : initSize - i * wsInitSize;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
                AscendC::SetAtomicNone();
                AscendC::Duplicate(zeroTensor[pingPongFlag], initValue, actualDealLen);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingPongFlag);
                AscendC::DataCopy(gOFp32[i * wsInitSize], zeroTensor[pingPongFlag], actualDealLen);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingPongFlag);
                pingPongFlag = 1 - pingPongFlag;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

private:
    float siluScale;
    uint32_t siluScaleDealSize;
    uint32_t castDealSize;
    uint32_t wsInitSize;

    AscendC::LocalTensor<ElementInput> lsUbTensor[STAGES];
    AscendC::LocalTensor<ElementInput> lpTmpUbTensor[STAGES];
    AscendC::LocalTensor<ElementOutput> lpUbTensor[STAGES];
    AscendC::LocalTensor<ElementInput> loFp32UbTensor[STAGES];
    AscendC::LocalTensor<ElementOutput> loCastUbTensor[STAGES];
    AscendC::LocalTensor<ElementInput> zeroTensor[STAGES];

    uint32_t pingPongFlag = 0;

    template <class OutputType, class InputType>
    
    __simd_vf__ inline void SiluScaleCompute(__ubuf__ OutputType *dst, __ubuf__ InputType *src,
        uint32_t count, uint16_t srcRepeatSize, uint16_t dstRepeatSize, uint16_t repeatTimes)
    {
        static constexpr AscendC::MicroAPI::CastTrait castTrait = 
        {AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::MERGING, AscendC::RoundMode::CAST_RINT};
        AscendC::MicroAPI::RegTensor<InputType> srcVreg;
        AscendC::MicroAPI::RegTensor<InputType> tmpReg0;
        AscendC::MicroAPI::RegTensor<OutputType> dstVreg;
        AscendC::MicroAPI::MaskReg maskReg;
        maskReg = AscendC::MicroAPI::UpdateMask<InputType>(count);
        for (uint16_t i = 0; i < repeatTimes; i++) {
            AscendC::MicroAPI::DataCopy(srcVreg, src + i * srcRepeatSize);
            AscendC::MicroAPI::Muls(tmpReg0, srcVreg, -1.0f, maskReg); // tmpReg0 = -x
            AscendC::MicroAPI::Exp(tmpReg0, tmpReg0, maskReg); // tmpReg0 = e^(-x)
            AscendC::MicroAPI::Adds(tmpReg0, tmpReg0, 1.0f, maskReg); // tmpReg0 = 1 + e^(-x)
            AscendC::MicroAPI::Div(tmpReg0, srcVreg, tmpReg0, maskReg); // tmpReg0 = x / (1 + e^(-x))
            AscendC::MicroAPI::Muls(dstVreg, tmpReg0, siluScale, maskReg); // tmpReg0 = x / (1 + e^(-x)) * scale
            AscendC::MicroAPI::DataCopy(dst + i * dstRepeatSize, dstVreg, maskReg);
        }
    }
};
}  // namespace Catlass::Epilogue::Block

#endif  // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SILU_SCALE_HPP