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
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t needRowLoop, uint32_t LastRowLoopFlag, uint32_t rowOffsetLoop,
        uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum,
        uint32_t rowOffsetCurLoop, int32_t delStartRow, int32_t delEndRow, uint32_t qSeqlen,
        uint32_t qSBlockIdx, uint32_t rowNum, uint32_t inRowOffsetThisSubBlock,
        const SplitKVParams& splitParams, uint32_t curQNBlockTile)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embedV = layoutInput.shape(1);
        uint32_t embedRoundV = layoutInput.stride(0);
        uint32_t curRowNumRound = NpuArch::Detail::Alignment::RoundUp(curRowNum, FLOAT_BLOCK_SIZE);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        uint32_t oHiddenSize_gmlo = 0;
        uint32_t qHeads_gmlse = 0;
        if (splitParams.isSplitkv) {
            oHiddenSize_gmlo = splitParams.layoutgmLo->shape(1);
            qHeads_gmlse = splitParams.layoutgmLse->shape(1);
        }

        if (!isFirstStackTile) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::DataCopy(
                loUbTensor, gInput, AscendC::DataCopyParams(1, curRowNum * embedRoundV / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            AscendC::Brcb(tvUbTensor.ReinterpretCast<uint32_t>(),
                dmUbTensor[dmUbOffsetCurStackTile].ReinterpretCast<uint32_t>(),
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            if (needRowLoop) {
                AscendC::DataCopy(
                    goUbTensor32, gUpdate,
                    AscendC::DataCopyParams(1, curRowNum * embedRoundV / FLOAT_BLOCK_SIZE, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            }
            // *** go = go * dm_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vmul_idx = 0; vmul_idx < embedV / FLOAT_VECTOR_SIZE; ++vmul_idx) {
                AscendC::Mul<float, false>(
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vmul_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRoundV / FLOAT_BLOCK_SIZE, embedRoundV / FLOAT_BLOCK_SIZE, 1));
            }
            if (embedV % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embedV % FLOAT_VECTOR_SIZE);
                AscendC::Mul<float, false>(
                    goUbTensor32[embedV / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embedV / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(1, 1, 0,
                        embedRoundV / FLOAT_BLOCK_SIZE, embedRoundV / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)(-1), (uint64_t)(-1));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // *** go = lo + go
            AscendC::Add<float, false>(
                goUbTensor32,
                goUbTensor32,
                loUbTensor,
                (uint64_t)0,
                NpuArch::Detail::Alignment::CeilDiv(curRowNum * embedRoundV, FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // *** go = lo
            AscendC::DataCopy(
                goUbTensor32, gInput, AscendC::DataCopyParams(1, curRowNum * embedRoundV / FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // *** gl_block = expand_to_block(gl), 存放于 tv
            AscendC::Brcb(
                tvUbTensor.ReinterpretCast<uint32_t>(),
                glUbTensor.ReinterpretCast<uint32_t>()[rowOffsetLoop],
                curRowNumRound / FLOAT_BLOCK_SIZE,
                AscendC::BrcbRepeatParams(1, 8));
            AscendC::PipeBarrier<PIPE_V>();
            // *** go = go / gl_block
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            for (uint32_t vdiv_idx = 0; vdiv_idx < embedV / FLOAT_VECTOR_SIZE; ++vdiv_idx) {
                AscendC::Div<float, false>(
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    goUbTensor32[vdiv_idx * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRoundV / FLOAT_BLOCK_SIZE, embedRoundV / FLOAT_BLOCK_SIZE, 1));
            }
            if (embedV % FLOAT_VECTOR_SIZE > 0) {
                SetMask(embedV % FLOAT_VECTOR_SIZE);
                AscendC::Div<float, false>(
                    goUbTensor32[embedV / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    goUbTensor32[embedV / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE],
                    tvUbTensor,
                    (uint64_t)0,
                    curRowNum,
                    AscendC::BinaryRepeatParams(
                        1, 1, 0, embedRoundV / FLOAT_BLOCK_SIZE, embedRoundV / FLOAT_BLOCK_SIZE, 1));
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // *** go = castfp32to16(go)
            if (!splitParams.isSplitkv) {
                if (std::is_same<ElementOutput, bfloat16_t>::value) {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16, goUbTensor32,
                        AscendC::RoundMode::CAST_RINT, (uint64_t)0,
                        NpuArch::Detail::Alignment::CeilDiv(curRowNum * embedRoundV, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                } else {
                    AscendC::Cast<ElementOutput, float, false>(
                        goUbTensor16, goUbTensor32,
                        AscendC::RoundMode::CAST_NONE, (uint64_t)0,
                        NpuArch::Detail::Alignment::CeilDiv(curRowNum * embedRoundV, FLOAT_VECTOR_SIZE),
                        AscendC::UnaryRepeatParams(1, 1, 4, 8));
                }
            }
            uint32_t rowStart = qSBlockIdx * VECTOR_SIZE + rowOffsetCurLoop;
            uint32_t innerGOUbOffset = 0;
            uint32_t subBlockStart = (curQNBlockTile == 1U)
                ? rowStart
                : (rowStart >= qSeqlen ? rowStart - rowStart / qSeqlen * qSeqlen : rowStart);
            if (delStartRow != 0) {
                if (proTokenNum != 0U && subBlockStart + proTokenNum >= delStartRow) {
                    uint32_t start = subBlockStart >= delStartRow ? 0 : delStartRow - subBlockStart;
                    uint32_t end = proTokenNum;
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset + start * embedRoundV],
                        static_cast<ElementOutput>(0),
                        (end - start) * embedRoundV
                    );
                    innerGOUbOffset += proTokenNum * embedRoundV;
                }
                if (subBlockStart + qSThisSubBlock >= delStartRow) {
                    for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
                        uint32_t start = subBlockStart >= delStartRow ? 0 : delStartRow - subBlockStart;
                        uint32_t end = qSThisSubBlock;
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Duplicate<ElementOutput>(
                            goUbTensor16[innerGOUbOffset + start * embedRoundV],
                            static_cast<ElementOutput>(0),
                            (end - start) * embedRoundV
                        );
                        innerGOUbOffset += qSThisSubBlock * embedRoundV;
                    }
                }
                if (epiTokenNum != 0U && subBlockStart + epiTokenNum >= delStartRow) {
                    uint32_t start = subBlockStart >= delStartRow ? 0 : delStartRow - subBlockStart;
                    uint32_t end = epiTokenNum;
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset + start * embedRoundV],
                        static_cast<ElementOutput>(0),
                        (end - start) * embedRoundV
                    );
                }
            }
            if (delEndRow != qSeqlen) {
                if (proTokenNum != 0U && subBlockStart < delEndRow) {
                    uint32_t start = curQNBlockTile == 1U ? rowStart : 0;
                    uint32_t end = (subBlockStart + proTokenNum >= delEndRow)
                        ? (curQNBlockTile == 1U ? delEndRow : delEndRow - subBlockStart)
                        : subBlockStart + proTokenNum;
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset],
                        static_cast<ElementOutput>(0),
                        (end - start) * embedRoundV
                    );
                    innerGOUbOffset += proTokenNum * embedRoundV;
                }
                if (subBlockStart < delEndRow) {
                    for (uint32_t qN_idx = 0; qN_idx < integralHeadNum; qN_idx++) {
                        uint32_t start = curQNBlockTile == 1U ? subBlockStart : proTokenNum;
                        uint32_t end = (subBlockStart + qSThisSubBlock >= delEndRow)
                            ? (curQNBlockTile == 1U ? delEndRow : delEndRow - subBlockStart)
                            : start + qSThisSubBlock;
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Duplicate<ElementOutput>(
                            goUbTensor16[innerGOUbOffset],
                            static_cast<ElementOutput>(0),
                            (end - start) * embedRoundV
                        );
                        innerGOUbOffset += qSThisSubBlock * embedRoundV;
                    }
                }
                if (epiTokenNum != 0U && subBlockStart < delEndRow) {
                    uint32_t start = curQNBlockTile == 1U
                        ? subBlockStart
                        : proTokenNum + integralHeadNum * qSThisSubBlock + subBlockStart;
                    uint32_t end = curQNBlockTile == 1U
                        ? (subBlockStart + epiTokenNum >= delEndRow ? delEndRow : subBlockStart + epiTokenNum)
                        : (epiTokenNum >= delEndRow ? start + delEndRow : start + epiTokenNum);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Duplicate<ElementOutput>(
                        goUbTensor16[innerGOUbOffset],
                        static_cast<ElementOutput>(0),
                        (end - start) * embedRoundV
                    );
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

            if (splitParams.isSplitkv) {
                CopyOToGmFp32(
                    splitParams.gCombineo,
                    proTokenIdx,
                    proTokenNum,
                    epiTokenNum,
                    integralHeadNum,
                    qSThisSubBlock,
                    embedV,
                    embedRoundV,
                    oHiddenSize, oHiddenSize_gmlo);
            } else {
                CopyOToGm(
                    gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum,
                    qSThisSubBlock, embedV, embedRoundV, oHiddenSize);
            }

            if constexpr (LSE_MODE_ == LseMode::OUT_ONLY) {
                if (LastRowLoopFlag) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Ln<float, false>(
                        lse32_ubuf_tensor,
                        glUbTensor,
                        (uint64_t)0,
                        NpuArch::Detail::Alignment::CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
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
                    if (!splitParams.isSplitkv) {
                        InvalidLineLSEProcess(
                            qNThisSubBlock,
                            delStartRow,
                            qSBlockIdx,
                            inRowOffsetThisSubBlock,
                            totalRowNum,
                            delEndRow,
                            qSeqlen,
                            qSThisSubBlock);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);

                    if (qNThisSubBlock == 0U) {
                        if (splitParams.isSplitkv) {
                            AscendC::DataCopyPad(
                                splitParams.gCombineLse, tvUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                        } else {
                            AscendC::DataCopyPad(
                                gLse, tvUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            if (splitParams.isSplitkv) {
                                AscendC::DataCopyPad(
                                    splitParams.gCombineLse[qNIdx],
                                    tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                            } else {
                                AscendC::DataCopyPad(
                                    gLse[qNIdx],
                                    tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                            }
                        }
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                }
            } else {
                if (splitParams.isSplitkv) {
                    if (LastRowLoopFlag) {
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
                            (uint64_t)0,
                            NpuArch::Detail::Alignment::CeilDiv(totalRowNum, FLOAT_VECTOR_SIZE),
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
                                splitParams.gCombineLse, tvUbTensor,
                                AscendC::DataCopyExtParams(
                                    totalRowNum, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                        } else {
                            for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                                AscendC::DataCopyPad(
                                    splitParams.gCombineLse[qNIdx],
                                    tvUbTensor[qNIdx * qSBlockSize * FLOAT_BLOCK_SIZE],
                                    AscendC::DataCopyExtParams(
                                        qSBlockSize, sizeof(float), 0, (qHeads_gmlse - 1) * sizeof(float), 0));
                            }
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
                    }
                }
            }
        } else if (needRowLoop) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
            AscendC::DataCopy(
                gUpdate, goUbTensor32, AscendC::DataCopyParams(1, curRowNum * embedRoundV / FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }
