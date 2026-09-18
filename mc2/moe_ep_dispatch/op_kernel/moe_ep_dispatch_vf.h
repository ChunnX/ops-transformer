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
 * \file moe_ep_dispatch_vf.h
 * \brief Native VF helpers used only by the Direct moe_ep_dispatch template.
 */

#ifndef MOE_EP_DISPATCH_VF_H
#define MOE_EP_DISPATCH_VF_H

namespace MoeEpDispatchVf {

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;

/**
 * Convert top-k ids, build the expert histogram, derive/deduplicate destination ranks and build the rank histogram.
 *
 * Buffer lifetime (all addresses are UB):
 *   topkS32       : input int32; reused as deduplicated int16 scratch after PrepareExpert.
 *   expertScratch : converted int16 expert ids.
 *   dstRankS16    : raw int16 rank ids; overwritten with final deduplicated rank ids before return.
 *   expertHistU32 : uint32 local expert histogram.
 *   rankHistU32   : uint32 local rank histogram, placed after expertHistU32 in dstExpBuf_.
 *   expertCountS32/rankCountS32: persistent Direct counters accumulated before return.
 *
 * The histogram layout follows dispatch.hpp::deduplicate: every 256-bin block is stored as uint32 counters by
 * interleaving the uint16 histogram result with zero. calCnt is at most 16K in the Direct template, so uint16 local
 * accumulation cannot overflow.
 */
__aicore__ inline void DeduplicateVf(__ubuf__ int32_t *topkS32, __ubuf__ int16_t *expertScratch,
                                     __ubuf__ int16_t *dstRankS16, __ubuf__ uint32_t *expertHistU32,
                                     __ubuf__ uint32_t *rankHistU32, __ubuf__ int32_t *expertCountS32,
                                     __ubuf__ int32_t *rankCountS32, uint32_t calCnt, uint32_t tokenCnt,
                                     uint32_t topK, uint32_t expertNum, uint32_t rankNum, uint32_t expertsPerRank)
{
    __VEC_SCOPE__
    {
        // PrepareExpert: keep the same DINTLV/LOWER narrowing layout as dispatch.hpp::deduplicate.
        {
            vector_s32 expertS32Low;
            vector_s32 expertS32High;
            vector_s32 negOneS32;
            vector_s16 expertS16;
            vector_s16 rankS16;
            vector_s16 negOne;
            vector_s16 expertsPerRankS16;
            vector_align loadState;
            vector_align expertStoreState;
            vector_align rankStoreState;
            vector_bool all = pge_b8(PAT_ALL);
            vector_bool geS32Low;
            vector_bool geS32High;
            vector_bool validS32Low;
            vector_bool validS32High;
            vector_bool validExpert;

            uint16_t convertLoops = static_cast<uint16_t>((calCnt + 127U) / 128U);
            __ubuf__ int32_t *topkRead = topkS32;
            __ubuf__ int16_t *expertWrite = expertScratch;
            __ubuf__ int16_t *rankWrite = dstRankS16;
            vdup(negOneS32, static_cast<int32_t>(-1), all, MODE_ZEROING);
            vdup(negOne, static_cast<int16_t>(-1), all, MODE_ZEROING);
            vdup(expertsPerRankS16, static_cast<int16_t>(expertsPerRank), all, MODE_ZEROING);
            vldas(loadState, topkRead);
            for (uint16_t loop = 0; loop < convertLoops; ++loop) {
                vldus(expertS32Low, loadState, ((__ubuf__ int32_t *&)topkRead), 64, POST_UPDATE);
                vldus(expertS32High, loadState, ((__ubuf__ int32_t *&)topkRead), 64, POST_UPDATE);
                vcmps_ge(geS32Low, expertS32Low, static_cast<int32_t>(0), all);
                vcmps_ge(geS32High, expertS32High, static_cast<int32_t>(0), all);
                vcmps_le(validS32Low, expertS32Low, static_cast<int32_t>(expertNum - 1U), geS32Low);
                vcmps_le(validS32High, expertS32High, static_cast<int32_t>(expertNum - 1U), geS32High);
                vsel(expertS32Low, expertS32Low, negOneS32, validS32Low);
                vsel(expertS32High, expertS32High, negOneS32, validS32High);
                vdintlvv2(expertS16, (vector_s16)expertS32Low, (vector_s16)expertS32High, LOWER);
                vcmps_ge(validExpert, expertS16, static_cast<int16_t>(0), all);
                vdiv(rankS16, expertS16, expertsPerRankS16, all);
                vsel(rankS16, rankS16, negOne, validExpert);
                vstus(expertStoreState, 128, expertS16, ((__ubuf__ int16_t *&)expertWrite), POST_UPDATE);
                vstus(rankStoreState, 128, rankS16, ((__ubuf__ int16_t *&)rankWrite), POST_UPDATE);
            }
            vstas(expertStoreState, ((__ubuf__ half *&)expertWrite), 0, POST_UPDATE);
            vstas(rankStoreState, ((__ubuf__ half *&)rankWrite), 0, POST_UPDATE);
            mem_bar(VST_VLD);
        }

        // SetExpertHistogram. This is the validated 256-bin organization from dispatch.hpp::deduplicate.
        {
            vector_s16 expertLow;
            vector_s16 expertHigh;
            vector_s16 binStart;
            vector_u8 expertU8;
            vector_u16 countLow;
            vector_u16 countHigh;
            vector_u16 zeros;
            vector_bool all = pge_b8(PAT_ALL);
            vector_bool tailMask;
            vector_bool geLow;
            vector_bool geHigh;
            vector_bool leLow;
            vector_bool leHigh;
            vector_bool histMask;

            uint16_t binLoops = static_cast<uint16_t>((expertNum + 255U) / 256U);
            uint16_t dataLoops = static_cast<uint16_t>((calCnt + 255U) / 256U);
            vdup(zeros, static_cast<uint16_t>(0), all, MODE_ZEROING);
            vdup(binStart, static_cast<int16_t>(0), all, MODE_ZEROING);

            for (uint16_t bin = 0; bin < binLoops; ++bin) {
                uint32_t remaining = calCnt;
                vdup(countLow, static_cast<uint16_t>(0), all, MODE_ZEROING);
                vdup(countHigh, static_cast<uint16_t>(0), all, MODE_ZEROING);
                for (uint16_t dataLoop = 0; dataLoop < dataLoops; ++dataLoop) {
                    __ubuf__ int16_t *inputLow = expertScratch + dataLoop * 256U;
                    __ubuf__ int16_t *inputHigh = inputLow + 128U;
                    vector_address loadAddr = vag_b16(0, 256);
                    tailMask = plt_b8(remaining, POST_UPDATE);
                    vld(expertLow, inputLow, loadAddr, NORM);
                    vld(expertHigh, inputHigh, loadAddr, NORM);
                    vsub(expertLow, expertLow, binStart, all);
                    vsub(expertHigh, expertHigh, binStart, all);
                    vcmps_ge(geLow, expertLow, 0, all);
                    vcmps_ge(geHigh, expertHigh, 0, all);
                    vcmps_le(leLow, expertLow, 255, geLow);
                    vcmps_le(leHigh, expertHigh, 255, geHigh);
                    pdintlvv2_b8(histMask, leLow, leHigh, LOWER);
                    pand(histMask, histMask, tailMask, all);
                    vdintlvv2((vector_s8 &)expertU8, (vector_s8)expertLow, (vector_s8)expertHigh, LOWER);
                    dhistv2(countLow, expertU8, histMask, Bin_N0);
                    dhistv2(countHigh, expertU8, histMask, Bin_N1);
                }
                __ubuf__ uint16_t *outputLow =
                    reinterpret_cast<__ubuf__ uint16_t *>(expertHistU32 + bin * 256U);
                __ubuf__ uint16_t *outputHigh = outputLow + 256U;
                vector_address storeAddr = vag_b16(512);
                vst(countLow, zeros, outputLow, storeAddr, INTLV_B16, all);
                vst(countHigh, zeros, outputHigh, storeAddr, INTLV_B16, all);
                vadds(binStart, binStart, 256, all);
            }
        }

        // DedupAndSendDirect: one VF vector block contains an integral number of token rows.
        {
            vector_align loadState;
            vector_align storeState;
            vector_s16 linear;
            vector_s16 tokenIndex;
            vector_s16 laneIndex;
            vector_s16 gatherBase;
            vector_s16 gatherIndex;
            vector_s16 rankData;
            vector_s16 previousRank;
            vector_s16 negOne;
            vector_s16 topKReg;
            vector_s16 dedupData;
            vector_bool all = pge_b8(PAT_ALL);
            vector_bool duplicate;
            vector_bool earlier;
            vector_bool equalRank;

            uint16_t tokenPerReg = static_cast<uint16_t>(128U / topK);
            uint32_t mainSize = static_cast<uint32_t>(tokenPerReg) * topK;
            uint16_t dedupLoops = static_cast<uint16_t>((tokenCnt + tokenPerReg - 1U) / tokenPerReg);
            __ubuf__ int16_t *rankRead = dstRankS16;
            __ubuf__ int16_t *dedupWrite = reinterpret_cast<__ubuf__ int16_t *>(topkS32);

            vdup(topKReg, static_cast<int16_t>(topK), all, MODE_ZEROING);
            vci(linear, 0);
            vdiv(tokenIndex, linear, topKReg, all, MODE_ZEROING);
            vmuls(gatherBase, tokenIndex, static_cast<int16_t>(topK), all, MODE_ZEROING);
            vsub(laneIndex, linear, gatherBase, all, MODE_ZEROING);
            vdup(negOne, static_cast<int16_t>(-1), all, MODE_ZEROING);

            vldas(loadState, rankRead);
            for (uint16_t loop = 0; loop < dedupLoops; ++loop) {
                vldus(rankData, loadState, ((__ubuf__ int16_t *&)rankRead), mainSize, POST_UPDATE);
                duplicate = pge_b16(PAT_ALLF);
                for (uint16_t k = 0; k < static_cast<uint16_t>(topK); ++k) {
                    vadds(gatherIndex, gatherBase, static_cast<int16_t>(k), all, MODE_ZEROING);
                    vselr(previousRank, rankData, (vector_u16)gatherIndex);
                    vcmps_gt(earlier, laneIndex, static_cast<int16_t>(k), all);
                    vcmp_eq(equalRank, rankData, previousRank, earlier);
                    por(duplicate, duplicate, equalRank, all);
                }
                vsel(dedupData, negOne, rankData, duplicate);
                vstus(storeState, mainSize, dedupData, ((__ubuf__ int16_t *&)dedupWrite), POST_UPDATE);
            }
            vstas(storeState, ((__ubuf__ half *&)dedupWrite), 0, POST_UPDATE);
            mem_bar(VST_VLD);
        }

        // SetRankHistogram. Use a region independent from expertScratch to avoid a VLD-to-VST reuse hazard.
        {
            vector_s16 rankLow;
            vector_s16 rankHigh;
            vector_s16 binStart;
            vector_u8 rankU8;
            vector_u16 countLow;
            vector_u16 countHigh;
            vector_u16 zeros;
            vector_bool all = pge_b8(PAT_ALL);
            vector_bool tailMask;
            vector_bool geLow;
            vector_bool geHigh;
            vector_bool leLow;
            vector_bool leHigh;
            vector_bool histMask;

            __ubuf__ int16_t *dedupRank = reinterpret_cast<__ubuf__ int16_t *>(topkS32);
            uint16_t binLoops = static_cast<uint16_t>((rankNum + 255U) / 256U);
            uint16_t dataLoops = static_cast<uint16_t>((calCnt + 255U) / 256U);
            vdup(zeros, static_cast<uint16_t>(0), all, MODE_ZEROING);
            vdup(binStart, static_cast<int16_t>(0), all, MODE_ZEROING);

            for (uint16_t bin = 0; bin < binLoops; ++bin) {
                uint32_t remaining = calCnt;
                vdup(countLow, static_cast<uint16_t>(0), all, MODE_ZEROING);
                vdup(countHigh, static_cast<uint16_t>(0), all, MODE_ZEROING);
                for (uint16_t dataLoop = 0; dataLoop < dataLoops; ++dataLoop) {
                    __ubuf__ int16_t *inputLow = dedupRank + dataLoop * 256U;
                    __ubuf__ int16_t *inputHigh = inputLow + 128U;
                    vector_address loadAddr = vag_b16(0, 256);
                    tailMask = plt_b8(remaining, POST_UPDATE);
                    vld(rankLow, inputLow, loadAddr, NORM);
                    vld(rankHigh, inputHigh, loadAddr, NORM);
                    vsub(rankLow, rankLow, binStart, all);
                    vsub(rankHigh, rankHigh, binStart, all);
                    vcmps_ge(geLow, rankLow, 0, all);
                    vcmps_ge(geHigh, rankHigh, 0, all);
                    vcmps_le(leLow, rankLow, 255, geLow);
                    vcmps_le(leHigh, rankHigh, 255, geHigh);
                    pdintlvv2_b8(histMask, leLow, leHigh, LOWER);
                    pand(histMask, histMask, tailMask, all);
                    vdintlvv2((vector_s8 &)rankU8, (vector_s8)rankLow, (vector_s8)rankHigh, LOWER);
                    dhistv2(countLow, rankU8, histMask, Bin_N0);
                    dhistv2(countHigh, rankU8, histMask, Bin_N1);
                }
                __ubuf__ uint16_t *outputLow =
                    reinterpret_cast<__ubuf__ uint16_t *>(rankHistU32 + bin * 256U);
                __ubuf__ uint16_t *outputHigh = outputLow + 256U;
                vector_address storeAddr = vag_b16(512);
                vst(countLow, zeros, outputLow, storeAddr, INTLV_B16, all);
                vst(countHigh, zeros, outputHigh, storeAddr, INTLV_B16, all);
                vadds(binStart, binStart, 256, all);
            }
        }

        // Materialize the final rank entity at the original Direct output address.
        {
            __ubuf__ int16_t *dedupRank = reinterpret_cast<__ubuf__ int16_t *>(topkS32);
            __ubuf__ int16_t *rankWrite = dstRankS16;
            vector_s16 rankData;
            vector_align loadState;
            vector_align storeState;
            uint16_t copyLoops = static_cast<uint16_t>((calCnt + 127U) / 128U);
            vldas(loadState, dedupRank);
            for (uint16_t loop = 0; loop < copyLoops; ++loop) {
                vldus(rankData, loadState, ((__ubuf__ int16_t *&)dedupRank), 128, POST_UPDATE);
                vstus(storeState, 128, rankData, ((__ubuf__ int16_t *&)rankWrite), POST_UPDATE);
            }
            vstas(storeState, ((__ubuf__ half *&)rankWrite), 0, POST_UPDATE);
        }

        // Accumulate both local histograms into the persistent Direct counters in the same VF invocation.
        {
            vector_s32 localCount;
            vector_s32 totalCount;
            vector_align histLoadState;
            vector_align countLoadState;
            vector_align expertCountStoreState;
            vector_align rankCountStoreState;
            vector_bool activeMask;
            mem_bar(VST_VLD);
            uint32_t remaining = expertNum;
            uint16_t expertLoops = static_cast<uint16_t>((expertNum + 63U) / 64U);
            __ubuf__ int32_t *histRead = reinterpret_cast<__ubuf__ int32_t *>(expertHistU32);
            __ubuf__ int32_t *countRead = expertCountS32;
            __ubuf__ int32_t *countWrite = expertCountS32;
            vldas(histLoadState, histRead);
            vldas(countLoadState, countRead);
            for (uint16_t loop = 0; loop < expertLoops; ++loop) {
                activeMask = plt_b32(remaining, POST_UPDATE);
                vldus(localCount, histLoadState, ((__ubuf__ int32_t *&)histRead), 64, POST_UPDATE);
                vldus(totalCount, countLoadState, ((__ubuf__ int32_t *&)countRead), 64, POST_UPDATE);
                vadd(totalCount, totalCount, localCount, activeMask, MODE_ZEROING);
                vstus(expertCountStoreState, 64, totalCount, ((__ubuf__ int32_t *&)countWrite), POST_UPDATE);
            }
            vstas(expertCountStoreState, ((__ubuf__ int32_t *&)countWrite), 0, POST_UPDATE);
            remaining = rankNum;
            uint16_t rankLoops = static_cast<uint16_t>((rankNum + 63U) / 64U);
            histRead = reinterpret_cast<__ubuf__ int32_t *>(rankHistU32);
            countRead = rankCountS32;
            countWrite = rankCountS32;
            vldas(histLoadState, histRead);
            vldas(countLoadState, countRead);
            for (uint16_t loop = 0; loop < rankLoops; ++loop) {
                activeMask = plt_b32(remaining, POST_UPDATE);
                vldus(localCount, histLoadState, ((__ubuf__ int32_t *&)histRead), 64, POST_UPDATE);
                vldus(totalCount, countLoadState, ((__ubuf__ int32_t *&)countRead), 64, POST_UPDATE);
                vadd(totalCount, totalCount, localCount, activeMask, MODE_ZEROING);
                vstus(rankCountStoreState, 64, totalCount, ((__ubuf__ int32_t *&)countWrite), POST_UPDATE);
            }
            vstas(rankCountStoreState, ((__ubuf__ int32_t *&)countWrite), 0, POST_UPDATE);
            mem_bar(VST_VLD);
        }
    }
}

/** Compress token indices for one destination rank. The outer Direct loop retains ordering, GM offsets and MTE3. */
__aicore__ inline void SqueezeTokenIdxVf(__ubuf__ int32_t *tokenIdxOut, __ubuf__ uint32_t *countOut,
                                         __ubuf__ int16_t *rankEntity, int32_t dstRankId, int32_t tokenOffset,
                                         uint32_t tokenNum, uint32_t topK)
{
    __VEC_SCOPE__
    {
        vector_s32 tokenIndex;
        vector_s32 topKReg;
        vector_s32 rankData;
        vector_s32 squeezed;
        vector_align loadState;
        vector_align storeState;
        vector_bool all = pge_b8(PAT_ALL);
        vector_bool mainMask;
        vector_bool tailMask;
        vector_bool equalMask;

        uint16_t tokenPerReg = static_cast<uint16_t>(64U / topK);
        uint16_t loopNum = static_cast<uint16_t>(tokenNum / tokenPerReg);
        uint32_t mainSize = static_cast<uint32_t>(tokenPerReg) * topK;
        uint32_t tailSize = (tokenNum - static_cast<uint32_t>(loopNum) * tokenPerReg) * topK;
        uint32_t mainMaskSize = mainSize;
        uint32_t tailMaskSize = tailSize;
        mainMask = plt_b32(mainMaskSize, POST_UPDATE);
        tailMask = plt_b32(tailMaskSize, POST_UPDATE);

        vdup(topKReg, static_cast<int32_t>(topK), all, MODE_ZEROING);
        vci(tokenIndex, 0);
        vdiv(tokenIndex, tokenIndex, topKReg, all);
        vadds(tokenIndex, tokenIndex, tokenOffset, all, MODE_ZEROING);
        sprclr(SPR_AR);
        vldas(loadState, rankEntity);
        for (uint16_t loop = 0; loop < loopNum; ++loop) {
            vldus((vector_s16 &)rankData, loadState, ((__ubuf__ int16_t *&)rankEntity), mainSize, POST_UPDATE);
            vunpack(rankData, (vector_s16)rankData, LOWER);
            vcmps_eq(equalMask, rankData, dstRankId, mainMask);
            vsqz(squeezed, tokenIndex, equalMask, MODE_STORED);
            vstur(storeState, squeezed, tokenIdxOut, POST_UPDATE);
            vadds(tokenIndex, tokenIndex, static_cast<int32_t>(tokenPerReg), all, MODE_ZEROING);
        }
        if (tailSize > 0U) {
            vldus((vector_s16 &)rankData, loadState, ((__ubuf__ int16_t *&)rankEntity), tailSize, POST_UPDATE);
            vunpack(rankData, (vector_s16)rankData, LOWER);
            vcmps_eq(equalMask, rankData, dstRankId, tailMask);
            vsqz(squeezed, tokenIndex, equalMask, MODE_STORED);
            vstur(storeState, squeezed, tokenIdxOut, POST_UPDATE);
        }
        vstar(storeState, tokenIdxOut);
        sprsts(SPR_AR, countOut, 0);
    }
}

/** Sum counter rows column-wise and accumulate them into slotStart. */
__aicore__ inline void PrefixSlotVf(__ubuf__ int32_t *counterRows, __ubuf__ int32_t *slotStart,
                                    uint32_t rowCount, uint32_t rowStride, uint32_t validColumns)
{
    __VEC_SCOPE__
    {
        vector_s32 sum;
        vector_s32 row;
        vector_s32 slot;
        vector_align rowLoadState;
        vector_align slotLoadState;
        vector_align slotStoreState;
        vector_bool all = pge_b8(PAT_ALL);
        vector_bool activeMask;
        uint32_t remaining = validColumns;
        uint16_t columnLoops = static_cast<uint16_t>((validColumns + 63U) / 64U);
        __ubuf__ int32_t *slotRead = slotStart;
        __ubuf__ int32_t *slotWrite = slotStart;
        vldas(slotLoadState, slotRead);

        for (uint16_t loop = 0; loop < columnLoops; ++loop) {
            uint32_t column = static_cast<uint32_t>(loop) * 64U;
            activeMask = plt_b32(remaining, POST_UPDATE);
            vdup(sum, static_cast<int32_t>(0), all, MODE_ZEROING);
            for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
                __ubuf__ int32_t *rowRead = counterRows + rowIndex * rowStride + column;
                vldas(rowLoadState, rowRead);
                vldus(row, rowLoadState, ((__ubuf__ int32_t *&)rowRead), 64, POST_UPDATE);
                vadd(sum, sum, row, activeMask, MODE_ZEROING);
            }
            vldus(slot, slotLoadState, ((__ubuf__ int32_t *&)slotRead), 64, POST_UPDATE);
            vadd(slot, slot, sum, activeMask, MODE_ZEROING);
            vstus(slotStoreState, 64, slot, ((__ubuf__ int32_t *&)slotWrite), POST_UPDATE);
        }
        vstas(slotStoreState, ((__ubuf__ int32_t *&)slotWrite), 0, POST_UPDATE);
        mem_bar(VST_VLD);
    }
}

#endif // ENABLE_MOE_EP_KERNEL

} // namespace MoeEpDispatchVf

#endif // MOE_EP_DISPATCH_VF_H
