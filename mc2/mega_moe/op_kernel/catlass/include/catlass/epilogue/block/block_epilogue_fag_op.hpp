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

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"

namespace Catlass::Epilogue::Block {

template <typename ElementInput, typename ElementOutput, typename ElementWorkspace>
class BlockEpilogue<EpilogueAscend950FAGOp, ElementInput, ElementOutput, ElementWorkspace>
{
public:
    using DispatchPolicy = EpilogueAscend950FAGOp;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using VecBlockInfo = typename DispatchPolicy::VecBlockInfo;
    using VecAddrInfo = typename DispatchPolicy::VecAddrInfo;

    AscendC::TPipe *pipe;
    AscendC::TBuf<> unifiedBuffer;

    constexpr static uint32_t DTYPE_FACTOR = sizeof(ElementWorkspace) / sizeof(ElementInput);
    constexpr static uint32_t cal_block_num = 32 / sizeof(ElementWorkspace);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(ElementWorkspace);
    constexpr static uint32_t input_block_num = 32 / sizeof(ElementInput);
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static int64_t C0_SIZE = 16;
    constexpr static int64_t VEC_REPEAT = 8;

    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t BoolBegin = 50 * 1024;
    constexpr static uint32_t U8Begin = 58 * 1024;
    constexpr static uint32_t T2BlockBegin = 66 * 1024;

    constexpr static uint32_t DbBegin = 74 * 1024;
    constexpr static int64_t TMP_UB_OFFSET = 148 * 1024;
    constexpr static int64_t SFMG_UB_OFFSET = (148 + 33) * 1024;
    constexpr static int64_t TMP_UB_SIZE = 33 * 1024;
    constexpr static int64_t SFMG_UB_SIZE = 8 * 1024;
    constexpr static int64_t TOTAL_SIZE = 189 * 1024;

    constexpr static uint32_t AttenMaskDimS2 = 2048;

    uint32_t blockIdx;
    uint32_t cubeBlockIdx;
    uint32_t subIdx;

        
    int32_t taskId = 0;
    int32_t pingpongIdx = 0;
    int32_t blockLen = 0;

    int64_t cuQSeqLen;
    int64_t cuKSeqLen;

    ElementWorkspace scaleValue;
    SoftMaxTiling softmaxTilingData;

    int32_t cubeBaseMN;

    int32_t s1VecSize;
    int32_t s2VecSize;
    constexpr static int32_t S1_CUBESIZE = 128;
    constexpr static int32_t S2_CUBESIZE = 128;
    
    int32_t s1Extend;
    int32_t s2Extend;
    int32_t s2ExtendAlign;
    int32_t s1CubeExtend;
    int32_t s2CubeExtend;
    
    int32_t curSeqQIdx;
    int32_t curSeqKIdx;

    // offset 
    int32_t sfmgOffset = 0;

    int64_t copyInOffset = 0;
    int64_t copyOutOffset = 0;
    AscendC::DataCopyParams copyInParam;
    AscendC::DataCopyParams copyOutParam;

    __gm__ uint8_t *cu_seq_qlen_addr;
    __gm__ uint8_t *cu_seq_kvlen_addr;

    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> &resource, AscendC::TPipe *pipe_in,
                  GM_ADDR cu_seq_qlen_addr_in, GM_ADDR cu_seq_kvlen_addr_in,
                  ElementWorkspace scaleValue_in,
                  const SoftMaxTiling &softmaxTiling)
    {
        pipe = pipe_in;
        blockIdx = AscendC::GetBlockIdx();
        cubeBlockIdx = blockIdx / 2;
        subIdx = blockIdx % 2;
        curSeqQIdx = subIdx;
        curSeqKIdx = 0;
        cubeBaseMN = 16 * 128 * 128;

        cu_seq_qlen_addr = cu_seq_qlen_addr_in;
        cu_seq_kvlen_addr = cu_seq_kvlen_addr_in;
        scaleValue = scaleValue_in;
        softmaxTilingData = softmaxTiling;

        pipe->InitBuffer(unifiedBuffer, TOTAL_SIZE);
    }

    CATLASS_DEVICE
    ~BlockEpilogue()
    {
    }

    CATLASS_DEVICE
    void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t &cuSeqQlen, int64_t &cuSeqKvlen)
    {
        if (unlikely(bIdx == 0)) {
            cuSeqQlen = ((__gm__ int64_t *)cu_seq_qlen_addr)[0];
            cuSeqKvlen = ((__gm__ int64_t *)cu_seq_kvlen_addr)[0];
        } else {
            cuSeqQlen =
                ((__gm__ int64_t *)cu_seq_qlen_addr)[bIdx] - ((__gm__ int64_t *)cu_seq_qlen_addr)[bIdx - 1];
            cuSeqKvlen =
                ((__gm__ int64_t *)cu_seq_kvlen_addr)[bIdx] - ((__gm__ int64_t *)cu_seq_kvlen_addr)[bIdx - 1];
        }
        return;
    }

    CATLASS_DEVICE
    void CopyInAttenMaskBool(AscendC::GlobalTensor<uint8_t> &attenMaskU8Gm, AscendC::LocalTensor<uint8_t> &dstTensor,
                            int64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend)
    {
        AscendC::DataCopyExtParams intriParams;
        intriParams.blockCount = s1Extend;
        intriParams.blockLen = s2Extend * sizeof(uint8_t);
        intriParams.srcStride = (AttenMaskDimS2 - s2Extend) * sizeof(uint8_t);
        intriParams.dstStride = 0;
        intriParams.rsv = 0;
        DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset], intriParams, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcAttenMaskBool(
        AscendC::LocalTensor<ElementWorkspace> &dstTensor,
        AscendC::LocalTensor<uint8_t> srcTensor,
        uint32_t s1Extend,
        uint32_t s2SrcExtend,
        uint32_t s2MaskExtend = 128,
        uint8_t maskType = 0)
    {
        AscendC::LocalTensor<uint8_t> tmpUbBuffer = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);

        ElementWorkspace scalar;
        if constexpr (AscendC::IsSameType<ElementWorkspace, float>::value) {
            uint32_t tmp = 0xFF7FFFFF;
            scalar = *reinterpret_cast<ElementWorkspace *>(&tmp);
        } else {
            uint16_t tmp = 0xFBFF;
            scalar = *reinterpret_cast<ElementWorkspace *>(&tmp);
        }

        AscendC::SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Extend;
        info.srcLastAxis = s2SrcExtend;
        info.maskLastAxis = s2MaskExtend;
        dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
        srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
        AscendC::SelectWithBytesMask<ElementWorkspace, uint8_t, false>(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    }

    CATLASS_DEVICE
    void CopyInSoftMax(AscendC::GlobalTensor<ElementWorkspace> &rowMaxGm, AscendC::GlobalTensor<ElementWorkspace> &rowSumGm,
                      AscendC::LocalTensor<ElementWorkspace> &dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
    {
        DataCopyPad(dstTensor, rowSumGm[softMaxOffset], {1, static_cast<uint16_t>(s1Extend * BLOCK_SIZE), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(dstTensor[64 * 8], rowMaxGm[softMaxOffset],
                    {1, static_cast<uint16_t>(s1Extend * BLOCK_SIZE), 0, 0}, {false, 0, 0, 0});
    }

    CATLASS_DEVICE
    void CalcSoftMax(AscendC::LocalTensor<ElementWorkspace>& dstTensor, AscendC::LocalTensor<ElementWorkspace>& src0Tensor, 
                    AscendC::LocalTensor<ElementWorkspace>& src1Tensor, uint32_t s1Extend, uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling& tiling)
    {
        bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

        if (isBasicBlock) {
            AscendC::LocalTensor<uint8_t> sharedTmp = unifiedBuffer.GetWithOffset<uint8_t>(TMP_UB_SIZE / sizeof(uint8_t), TMP_UB_OFFSET);
            uint32_t shapeArray1[2];
            shapeArray1[0] = s1Extend;
            shapeArray1[1] = s2Extend;
            dstTensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            src0Tensor.SetShapeInfo(AscendC::ShapeInfo(2, shapeArray1, AscendC::DataFormat::ND));
            AscendC::SimpleSoftMax<ElementWorkspace, false, true>(dstTensor, src1Tensor, src1Tensor[64 * 8], src0Tensor,
                                        sharedTmp, tiling);
        } else {
            AscendC::LocalTensor<ElementWorkspace> vecOutBuffer = unifiedBuffer.GetWithOffset<ElementWorkspace>(TMP_UB_SIZE / sizeof(ElementWorkspace), TMP_UB_OFFSET);
            uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

            for(uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
                uint32_t subMaskCount = (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
                Sub(dstTensor[subIdx * cal_repeat_num], src0Tensor[subIdx * cal_repeat_num], src1Tensor[64 * 8],
                        subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                AscendC::PipeBarrier<PIPE_V>();
                Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num],
                    subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1),
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8)});
                AscendC::PipeBarrier<PIPE_V>();
                Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], src1Tensor,
                        subMaskCount, s1Extend,
                        {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0,
                        static_cast<uint8_t>(s2ExtendAlign / 8), static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    CATLASS_DEVICE
    void SubGrapA(AscendC::GlobalTensor<ElementWorkspace> &rowMaxGm, AscendC::GlobalTensor<ElementWorkspace> &rowSumGm,
                 AscendC::GlobalTensor<ElementWorkspace> &mm2WorkspaceGm, AscendC::GlobalTensor<ElementOutput> &pWorkSpaceGm,
                 int64_t curIdx, const VecBlockInfo &blockInfo, event_t mte2WaitMte3A)
    {
        uint32_t ubBufferOffset = 0;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }

        AscendC::LocalTensor<ElementWorkspace> vecInBuffer3 =
            unifiedBuffer.GetWithOffset<ElementWorkspace>(8 * 1024 / sizeof(ElementWorkspace), ubBufferOffset + T2BlockBegin);
        
        CopyInSoftMax(rowMaxGm, rowSumGm, vecInBuffer3, s1Extend, sfmgOffset);

        AscendC::LocalTensor<ElementWorkspace> vecClc2Buffer =
            unifiedBuffer.GetWithOffset<ElementWorkspace>(32 * 1024 / sizeof(ElementWorkspace), ubBufferOffset + T2Begin);

        DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        AscendC::PipeBarrier<PIPE_V>();
        Muls(vecClc2Buffer, vecClc2Buffer, scaleValue, s1Extend * s2ExtendAlign);
            
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<uint8_t> attenMaskUbuint8 =
            unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), ubBufferOffset + BoolBegin);
        if (blockInfo.SeqQIdx == blockInfo.SeqKIdx) {
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8[curSeqQIdx * s1VecSize * 128], s1Extend, s2ExtendAlign, S2_CUBESIZE, 0);
        }

        ///////////////////////////////////////////////////////////////
        // simpleSoftMax
        ///////////////////////////////////////////////////////////////
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<ElementWorkspace> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<ElementWorkspace>(33 * 1024 / sizeof(ElementWorkspace), DbBegin);
        CalcSoftMax(simpleSoftmaxResBuf, vecClc2Buffer, vecInBuffer3, s1Extend, s2Extend, s2ExtendAlign, softmaxTilingData);
        AscendC::LocalTensor<ElementWorkspace> vecDropBuffer = simpleSoftmaxResBuf;

        ///////////////////////////////////////////////////////////////
        // cast fp322bf16
        ///////////////////////////////////////////////////////////////
        AscendC::LocalTensor<ElementOutput> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementOutput>(17 * 1024 / sizeof(ElementOutput), ubBufferOffset + T1Begin);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(vecCopyOutBuffer, vecDropBuffer, AscendC::RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);

        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        DataCopyPad(pWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < blockLen - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
        }
    }
    
    CATLASS_DEVICE
    void SubGrapB(AscendC::GlobalTensor<ElementWorkspace> &mm1WorkspaceGm, AscendC::GlobalTensor<ElementOutput> &dsWorkSpaceGm,
                 AscendC::GlobalTensor<ElementWorkspace> &sfmgWorkspaceGm, int64_t curIdx, const VecBlockInfo &blockInfo,
                 event_t mte2WaitMte3B)
    {
        uint32_t ubBufferOffset = DbBegin;

        if (curIdx > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }

        // copyIn sfmg
        AscendC::LocalTensor<ElementWorkspace> sfmgClc3 = unifiedBuffer.GetWithOffset<ElementWorkspace>(SFMG_UB_SIZE / sizeof(ElementWorkspace), SFMG_UB_OFFSET);
        DataCopy(sfmgClc3, sfmgWorkspaceGm[sfmgOffset], s1Extend * 8);

        AscendC::LocalTensor<ElementWorkspace> vecClc1Buffer = unifiedBuffer.GetWithOffset<ElementWorkspace>(33 * 1024 / sizeof(ElementWorkspace), ubBufferOffset + T1Begin);
        
        // copyIn cube result
        DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[copyInOffset], copyInParam, {false, 0, 0, 0});

        event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(vWaitMte2);

        ///////////////////////////////////////////////////////////////
        // sub
        ///////////////////////////////////////////////////////////////
        uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
            uint32_t subMaskCout =
                (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3,
                subMaskCout, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                static_cast<uint8_t>(s2ExtendAlign / 8), 1});
        }

        ///////////////////////////////////////////////////////////////
        // mul
        ///////////////////////////////////////////////////////////////
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<ElementWorkspace> simpleSoftmaxResBuf = unifiedBuffer.GetWithOffset<ElementWorkspace>(32 * 1024 / sizeof(ElementWorkspace), DbBegin);
        Mul(vecClc1Buffer, vecClc1Buffer, simpleSoftmaxResBuf, s1Extend * s2ExtendAlign);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<ElementOutput> vecCopyOutBuffer = unifiedBuffer.GetWithOffset<ElementOutput>(17 * 1024 / sizeof(ElementOutput), ubBufferOffset + T1Begin);
        Cast(vecCopyOutBuffer, vecClc1Buffer, AscendC::RoundMode::CAST_ROUND, s1Extend * s2ExtendAlign);

        event_t mte3WaitV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(mte3WaitV);

        DataCopyPad(dsWorkSpaceGm[copyOutOffset], vecCopyOutBuffer, copyOutParam);

        if (curIdx < blockLen - 1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

    CATLASS_DEVICE
    void operator()(const VecAddrInfo &addrs,
                    AscendC::GlobalTensor<ElementWorkspace> &gRowMax, AscendC::GlobalTensor<ElementWorkspace> &gRowSum,
                    AscendC::GlobalTensor<uint8_t> &gAttenMask, AscendC::GlobalTensor<ElementWorkspace> &gMm1Workspace,
                    AscendC::GlobalTensor<ElementWorkspace> &gMm2Workspace, AscendC::GlobalTensor<ElementOutput> &gDsWorkspace,
                    AscendC::GlobalTensor<ElementOutput> &gPWorkspace, AscendC::GlobalTensor<ElementWorkspace> &gSfmgWorkspace,
                    int64_t nheads_k, int64_t g)
    {
        taskId = addrs.taskId;
        pingpongIdx = taskId % 2;
        blockLen = addrs.blockLength;

        event_t mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3);
        if (taskId == 0) {
            AscendC::LocalTensor<uint8_t> attenMaskUbuint8 =
                    unifiedBuffer.GetWithOffset<uint8_t>(16 * 1024 / sizeof(uint8_t), BoolBegin);
            CopyInAttenMaskBool(gAttenMask, attenMaskUbuint8, 0, 128, 128);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < blockLen; ++i) {
            auto &blockInfo = addrs.VecBlkInfo[i];

            ///////////////////////////////////////////////////////////////
            // do scalar calculate
            ///////////////////////////////////////////////////////////////

            GetSeqQlenKvlenByBidx(blockInfo.batchIdx, cuQSeqLen, cuKSeqLen);

            s1CubeExtend = blockInfo.lengthy;
            s2CubeExtend = 128;

            // split info
            s1VecSize = (s1CubeExtend + 1) / 2;
            s2VecSize = 128;

            s1Extend = subIdx ? s1CubeExtend - s1VecSize : s1VecSize;
            s2Extend = blockInfo.lengthx;
            s2ExtendAlign = (s2Extend + 15) / 16 * 16;

            //offset
            sfmgOffset = 0;
            if (blockInfo.batchIdx > 0) {
                sfmgOffset = ((__gm__ int64_t *)cu_seq_qlen_addr)[blockInfo.batchIdx - 1] * nheads_k * g * 8;
            }
            sfmgOffset += ((blockInfo.nheadsKIdx * g + blockInfo.gIdx) * cuQSeqLen + blockInfo.SeqQIdx * S1_CUBESIZE + curSeqQIdx * s1VecSize) * 8;

            // copyIn cube_workspace params
            copyInOffset = 
                cubeBlockIdx * cubeBaseMN * 2 + pingpongIdx * cubeBaseMN + blockInfo.offset + curSeqQIdx * s1VecSize * s2CubeExtend;
            copyInParam = {
                static_cast<uint16_t>(s1Extend),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(ElementWorkspace)),
                static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(ElementWorkspace)), 
                0
            };

            // copyOut cube_workspace params
            copyOutOffset = 2 * copyInOffset;
            copyOutParam = {
                static_cast<uint16_t>(s1Extend),
                static_cast<uint16_t>(s2ExtendAlign * sizeof(ElementInput)),
                0,
                static_cast<uint16_t>((s2CubeExtend - s2ExtendAlign) * sizeof(ElementInput))
            };

            ///////////////////////////////////////////////////////////////
            // do vector calculate
            ///////////////////////////////////////////////////////////////
            event_t mte2WaitMte3A = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            event_t mte2WaitMte3B = static_cast<event_t>(GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
            SubGrapA(gRowMax, gRowSum, gMm2Workspace, gPWorkspace, i, blockInfo, mte2WaitMte3A);
            SubGrapB(gMm1Workspace, gDsWorkspace, gSfmgWorkspace, i, blockInfo, mte2WaitMte3B);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3A);
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte2WaitMte3B);
        }
    }

};

}

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_FAG_OP_HPP
