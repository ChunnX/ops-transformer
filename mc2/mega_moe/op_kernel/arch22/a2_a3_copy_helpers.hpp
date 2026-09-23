/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_A2_A3_COPY_HELPERS_HPP
#define MEGA_MOE_A2_A3_COPY_HELPERS_HPP
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "utils/const_args.hpp"

namespace Catlass::Gemm::Kernel::Detail {
using namespace AscendC;

template <class ArchTag, typename T>
CATLASS_DEVICE void CopyGMToGMShared(Arch::Resource<ArchTag> &resource, AscendC::GlobalTensor<T> dst,
                                     AscendC::GlobalTensor<T> src, int32_t elemNum, int32_t ubMoveNum)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    using TType = Gemm::GemmType<T, layout::RowMajor>;
    using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
    using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
    constexpr int32_t BufferNum = 2;
    int tmpBufferSize = 32 * 1024 / sizeof(T); // 32 KB
    AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
    tmpBuffer1.SetSize(tmpBufferSize);
    int tmpBufferOffset = 96 * 1024; // half of UB
    AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
    tmpBuffer2.SetSize(tmpBufferSize);

    int pingpongId = 0;
    auto processCount = CeilDiv(elemNum, ubMoveNum);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        uint32_t curProcessNum =
            (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        auto processOffset = processIndex * ubMoveNum;

        auto inputOffset = processOffset;
        auto outputOffset = processOffset;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

CATLASS_DEVICE void CalculateTaskInfoEachCoreShared(uint32_t &curCoreTaskNum, uint32_t &curCoreStartOffset,
                                                    uint32_t totalM, uint32_t coreNum)
{
    int64_t eachCoreTaskNum = (totalM + coreNum - 1) / coreNum;
    int64_t taskNumPertailCore = eachCoreTaskNum - 1;
    int64_t usedCoreNum = totalM >= coreNum ? coreNum : totalM;
    uint32_t tailCoreIdx = totalM - (eachCoreTaskNum - 1) * usedCoreNum;
    uint32_t curCoreId = GetBlockIdx();
    curCoreTaskNum = curCoreId < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    curCoreStartOffset =
        curCoreId < tailCoreIdx ? eachCoreTaskNum * curCoreId : ((eachCoreTaskNum - 1) * curCoreId + tailCoreIdx);
}

template <class ArchTag>
CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4Shared(Arch::Resource<ArchTag> &resource,
                                                       AscendC::GlobalTensor<int8_t> gmA1I4_I8, uint64_t gmOffsetA,
                                                       AscendC::GlobalTensor<float> dstScale,
                                                       AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                       int32_t hiddenSize, uint32_t outputStride, uint32_t coreNum)
{
    uint32_t tmpBufferOffset = 0;

    // double buffer 定义： tmpBuffer 包含int8_t token + float scale, xTensor 只包含int8_t token
    AscendC::LocalTensor<int8_t> xTensor0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    AscendC::LocalTensor<int8_t> tmpBuffer0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    xTensor0.SetSize(hiddenSize);
    tmpBuffer0.SetSize(hiddenSize + ALIGN_512);
    tmpBufferOffset += (hiddenSize + ALIGN_512) * sizeof(int8_t);

    AscendC::LocalTensor<int8_t> xTensor1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    AscendC::LocalTensor<int8_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    xTensor1.SetSize(hiddenSize);
    tmpBuffer1.SetSize(hiddenSize + ALIGN_512);
    tmpBufferOffset += (hiddenSize + ALIGN_512) * sizeof(int8_t);
    // double buffer 定义结束

    AscendC::LocalTensor<int4b_t> xHighI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
    xHighI4Tensor.SetSize(hiddenSize);
    tmpBufferOffset += hiddenSize / INT4_NUM_PER_BYTE;

    AscendC::LocalTensor<int4b_t> xLowI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
    xLowI4Tensor.SetSize(hiddenSize);
    tmpBufferOffset += hiddenSize / INT4_NUM_PER_BYTE;

    AscendC::LocalTensor<half> xHighHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    xHighHalfTensor.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    AscendC::LocalTensor<half> xLowHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    xLowHalfTensor.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    AscendC::LocalTensor<half> xLowHalfTensor2 = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    xLowHalfTensor2.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    AscendC::LocalTensor<int16_t> xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(tmpBufferOffset);
    constexpr int32_t MASK = 128; // int4 解包掩码宽度（int16 单位，bit 为 0x0F0F）
    xLowI16Tensor.SetSize(MASK * sizeof(int16_t));

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

    Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK);
    PipeBarrier<PIPE_V>();
    const size_t LEN_VK = (hiddenSize / INT4_NUM_PER_BYTE) / MASK;
    const size_t LAST_LEN_VK = (hiddenSize % 256) / 2;
    const half ONE_SIXTEENTH = static_cast<half>(0.0625f);

    uint32_t totalM = rows;

    SetFlag<HardEvent::S_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID6);
    uint32_t curCoreTaskNum;
    uint32_t curCoreStartOffset;
    CalculateTaskInfoEachCoreShared(curCoreTaskNum, curCoreStartOffset, totalM, coreNum);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

    constexpr size_t LEN_128 = 128;

    constexpr int32_t BufferNum = 2;
    uint32_t copyInNum = hiddenSize + ALIGN_512;

    int pingpongId = 0;
    for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
        const uint64_t relStartAddr = processIndex * outputStride;
        uint64_t absStartAddr = gmOffsetA + relStartAddr;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID6 : EVENT_ID7;
        AscendC::LocalTensor<int8_t> buf = pingpongId == 0 ? tmpBuffer0 : tmpBuffer1;
        AscendC::LocalTensor<int8_t> xTensor = pingpongId == 0 ? xTensor0 : xTensor1;
        AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
        auto inputOffset = processIndex * copyInNum;
        auto outputOffset = processIndex * hiddenSize;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID);
        AscendC::DataCopy(buf, src[inputOffset], copyInNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        // 远端 scale 拷贝到GM
        AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID);
        // 高四位处理开始
        Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, hiddenSize);
        PipeBarrier<PIPE_V>();
        Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, hiddenSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
        Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, hiddenSize);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
        DataCopy(gmA1I4_I8[absStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / INT4_NUM_PER_BYTE);
        // 高四位处理结束

        // 低四位处理开始
        SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
        And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
            LEN_VK, {1, 1, 1, 8, 8, 0});
        if (LAST_LEN_VK > 0) {
            And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                {1, 1, 1, 8, 8, 0});
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID);
        Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
             AscendC::RoundMode::CAST_NONE, hiddenSize);
        PipeBarrier<PIPE_V>();
        const half MINUS_EIGHT = static_cast<half>(-8);
        Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, hiddenSize);
        PipeBarrier<PIPE_V>();
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
        Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, hiddenSize);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
        DataCopy(gmA1I4_I8[absStartAddr + outputStride / INT4_NUM_PER_BYTE], xLowI4Tensor.ReinterpretCast<int8_t>(),
                 hiddenSize / INT4_NUM_PER_BYTE);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
        // 低四位处理结束
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
}

} // namespace Catlass::Gemm::Kernel::Detail

#endif // MEGA_MOE_A2_A3_COPY_HELPERS_HPP
