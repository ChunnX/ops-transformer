/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline
    void init(Arch::Resource<ArchTag> &resource)
    {
        // Allocate UB space
        constexpr uint32_t TV_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t LO_UB_TENSOR_OFFSET = 6 * UB_UINT8_BLOCK_SIZE;
        constexpr uint32_t GO_UB_TENSOR_OFFSET = 8 * UB_UINT8_BLOCK_SIZE;

        constexpr uint32_t HM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 10 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t GL_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t LSE_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 12 * UB_UINT8_VECTOR_SIZE;
        constexpr uint32_t DM_UB_TENSOR_OFFSET = 10 * UB_UINT8_BLOCK_SIZE + 13 * UB_UINT8_VECTOR_SIZE;

        loUbTensor = resource.ubBuf.template GetBufferByByte<float>(LO_UB_TENSOR_OFFSET);
        dmUbTensor = resource.ubBuf.template GetBufferByByte<float>(DM_UB_TENSOR_OFFSET);
        glUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        tvUbTensor = resource.ubBuf.template GetBufferByByte<float>(TV_UB_TENSOR_OFFSET);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_UB_TENSOR_OFFSET);
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(GO_UB_TENSOR_OFFSET);
        hmUbTensor = resource.ubBuf.template GetBufferByByte<float>(HM_UB_TENSOR_OFFSET);
        gmUbTensor = resource.ubBuf.template GetBufferByByte<float>(GM_UB_TENSOR_OFFSET);
        lse32_ubuf_tensor = resource.ubBuf.template GetBufferByByte<float>(LSE_UB_TENSOR_OFFSET);
    }

    __aicore__ inline
    ~BlockEpilogue() {}

    __aicore__ inline
    void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = static_cast<uint64_t>(len) % static_cast<uint64_t>(FLOAT_VECTOR_SIZE);
        for (uint64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        } else if (len >= FLOAT_VECTOR_SIZE) {
            AscendC::SetVectorMask<int8_t>(mask, (uint64_t)-1);
        } else {
            AscendC::SetVectorMask<int8_t>(0x0, mask);
        }
    }
    __aicore__ inline
    void InvalidLineLSEProcess(
        uint32_t qNThisSubBlock, int32_t delStartRow, uint32_t qSBlockIdx, uint32_t inRowOffsetThisSubBlock,
        uint32_t totalRowNum, int32_t delEndRow, uint32_t qSeqlen, uint32_t qSThisSubBlock)
    {
        uint32_t qNSubBlockStartOffset = qNThisSubBlock == 0U
            ? qSBlockIdx * VECTOR_SIZE + inRowOffsetThisSubBlock
            : qSBlockIdx * VECTOR_SIZE;
        uint32_t qNSubBlockEnbdOffset = totalRowNum + qNSubBlockStartOffset;
        if (qNThisSubBlock == 0U && delStartRow != 0 && qNSubBlockEnbdOffset >= delStartRow) {
            uint32_t start = qNSubBlockStartOffset > delStartRow ? 0 : (delStartRow - qNSubBlockStartOffset);
            uint32_t end = totalRowNum;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                tvUbTensor[start * FLOAT_BLOCK_SIZE],
                LSE_OUT_INI,
                (end - start) * FLOAT_BLOCK_SIZE
            );
        }
        if (qNThisSubBlock == 0U && delEndRow != qSeqlen && qNSubBlockStartOffset < delEndRow) {
            uint32_t rowStart = qNSubBlockStartOffset;
            uint32_t start = 0;
            uint32_t end = rowStart + totalRowNum >= delEndRow ? (delEndRow - rowStart) : totalRowNum;
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(
                tvUbTensor[start * FLOAT_BLOCK_SIZE],
                LSE_OUT_INI,
                (end - start) * FLOAT_BLOCK_SIZE
            );
        }
        if (qNThisSubBlock != 0U && delStartRow != 0 && qNSubBlockEnbdOffset >= delStartRow) {
            uint32_t start = delStartRow - qNSubBlockStartOffset;
            uint32_t end = qSThisSubBlock;
            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(
                    tvUbTensor[(qNIdx * qSThisSubBlock + start) * FLOAT_BLOCK_SIZE],
                    LSE_OUT_INI,
                    (end - start) * FLOAT_BLOCK_SIZE
                );
            }
        }
        if (qNThisSubBlock != 0U && delEndRow != qSeqlen && qNSubBlockStartOffset < delEndRow) {
            uint32_t start = 0;
            uint32_t end = qNSubBlockEnbdOffset >= delEndRow ? (delEndRow - qNSubBlockStartOffset) : totalRowNum;
            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Duplicate(
                    tvUbTensor[(qNIdx * qSThisSubBlock + start) * FLOAT_BLOCK_SIZE],
                    LSE_OUT_INI,
                    (end - start) * FLOAT_BLOCK_SIZE
                );
            }
        }
    }
    __aicore__ inline
    void CopyOToGm(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        uint32_t qSThisSubBlock, uint32_t embedValue, uint32_t embedRoundV, uint32_t oHiddenSize)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, embedValue * SIZE_OF_16BIT, 0, (oHiddenSize - embedValue) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embedValue;
            innerGOUbOffset += proTokenNum * embedRoundV;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, embedValue * SIZE_OF_16BIT, 0, (oHiddenSize - embedValue) * SIZE_OF_16BIT, 0));
            innerOGmOffset += embedValue;
            innerGOUbOffset += qSThisSubBlock * embedRoundV;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor16[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, embedValue * SIZE_OF_16BIT, 0, (oHiddenSize - embedValue) * SIZE_OF_16BIT, 0));
        }
    }

    __aicore__ inline
    void CopyOToGmFp32(
        AscendC::GlobalTensor<float> gOutput,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        uint32_t qSThisSubBlock, uint32_t embedV, uint32_t embedRoundV, uint32_t oHiddenSize,
        uint32_t oHiddenSize_gmlo)
    {
        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;
        uint32_t blockLen = embedV * sizeof(float);
        uint32_t blockLenAligned = (blockLen + 31) / 32 * 32;
        uint32_t srcStride = (embedRoundV * sizeof(float) - blockLenAligned) / 32;
        if (proTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize],
                goUbTensor32[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    proTokenNum, blockLen, srcStride, (oHiddenSize_gmlo - embedV) * sizeof(float), 0));
            innerOGmOffset += embedV;
            innerGOUbOffset += proTokenNum * embedRoundV;
        }
        for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor32[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    qSThisSubBlock, blockLen, srcStride, (oHiddenSize_gmlo - embedV) * sizeof(float), 0));
            innerOGmOffset += embedV;
            innerGOUbOffset += qSThisSubBlock * embedRoundV;
        }
        if (epiTokenNum != 0U) {
            AscendC::DataCopyPad(
                gOutput[innerOGmOffset],
                goUbTensor32[innerGOUbOffset],
                AscendC::DataCopyExtParams(
                    epiTokenNum, blockLen, srcStride, (oHiddenSize_gmlo - embedV) * sizeof(float), 0));
        }
    }
