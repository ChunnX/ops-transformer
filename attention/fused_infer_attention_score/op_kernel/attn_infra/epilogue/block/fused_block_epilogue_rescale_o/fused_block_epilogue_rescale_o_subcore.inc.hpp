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
    void SubCoreCompute(
        AscendC::GlobalTensor<ElementOutput> gOutput,
        AscendC::GlobalTensor<ElementInput> gInput,
        AscendC::GlobalTensor<ElementUpdate> gUpdate,
        AscendC::GlobalTensor<ElementLse> gLse,
        const LayoutOutput &layoutOutput,
        const LayoutInput &layoutInput,
        const LayoutUpdate &layoutUpdate,
        const LayoutLse &layoutLse,
        uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum,
        uint32_t isFirstStackTile, uint32_t LastStackTileFlag, uint32_t curStackTileMod,
        uint32_t needRowLoopFlag, uint32_t isLastRowLoop, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = NpuArch::Detail::Alignment::RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        if (!isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)(-1), (uint64_t)(-1));
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoopFlag) {
                AscendC::DataCopy(
                    goUbTensor32, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<float, false>(
                goUbTensor32,
                goUbTensor32,
                loUbTensor,
                (uint64_t)0,
                (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // *** go = lo
            AscendC::DataCopy(
                goUbTensor32, gInput, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (LastStackTileFlag) {
            // *** gl_block = expand_to_block(gl), 存放于 tv
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint32_t>(),
                glUbTensor.ReinterpretCast<uint32_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
            }
            if (embed % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embed % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embed / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRound / FLOAT_BLOCK_SIZE, embedRound / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)(-1), (uint64_t)(-1));
            }
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = castfp32to16(go)
            if (std::is_same<ElementOutput, bfloat16_t>::value) {
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            } else {
                AscendC::Cast<ElementOutput, float, false>(
                    goUbTensor16, goUbTensor32,
                    AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                    (curRowNum * embedRound + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    AscendC::UnaryRepeatParams(1, 1, 4, 8));
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            // ***move O to GM
            CopyOToGm(
                gOutput,
                proTokenIdx,
                proTokenNum,
                epiTokenNum,
                integralHeadNum,
                qSThisSubBlock,
                embed,
                embedRound,
                oHiddenSize);
            if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
                if (isLastRowLoop) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(
                        lse32_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0, NpuArch::Detail::Alignment::CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 8, 8));

                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<float, false>(
                        lse32_ubuf_tensor,
                        lse32_ubuf_tensor,
                        gmUbTensor,
                        (uint64_t)0, NpuArch::Detail::Alignment::CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
                        AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
                    AscendC::PipeBarrier<PIPE_V>();

                    // *** lse_block = expand_to_block(lse), 存放于 tv
                    AscendC::Brcb(
                        tvUbTensor.ReinterpretCast<uint32_t>(),
                        lse32_ubuf_tensor.ReinterpretCast<uint32_t>(),
                        NpuArch::Detail::Alignment::CeilDiv(totalRowNum, FLOAT_BLOCK_SIZE),
                        AscendC::BrcbRepeatParams(1, 8));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                    if (qNThisSubBlock == 0U) {
                        AscendC::DataCopyPad(
                            gLse, tvUbTensor,
                            AscendC::DataCopyExtParams(
                                totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            AscendC::DataCopyPad(
                                gLse[qNIdx],
                                tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                AscendC::DataCopyExtParams(
                                    qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoopFlag) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor32, AscendC::DataCopyParams(1, curRowNum * embedRound / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }
