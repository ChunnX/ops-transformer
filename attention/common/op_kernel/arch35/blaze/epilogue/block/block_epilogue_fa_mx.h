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
 * \file block_epilogue_dequant.h
 * \brief MIX template epilogue (AIV side): out = x1@x2(L0C in UB) * x2Scale * x1Scale + bias.
 *        The x1@x2 accumulator is read from UB (L0C->UB by AIC fixpipe, raw NoQuant copy);
 *        x2Scale / x1Scale / bias are read from GM. All scale multiplies + bias add are done
 *        on the vector using the VF (vector-function) AscendC::Reg style (RegTensor / MaskReg /
 *        __VEC_SCOPE__), mirroring VFDoDequant in qbmm_mix_online_dynamic.h.
 *        Dual AIV sub-block split, 4-way M partition, ping-pong output.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/epilogue/tile/tile_fa_vf.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Epilogue {
namespace Block {

using AscendC::IsSameType;
using AscendC::RoundMode;
using Blaze::Gemm::CeilAlign;
using Blaze::Gemm::CeilDiv;


template <bool HAS_MASK = true>
class BlockEpilogueFaMx {
public:
    using INPUT_T = fp8_e4m3fn_t;
    using OUTPUT_T = float16_t;
    using T = float;

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR attenMaskGmAddr{nullptr};
        GM_ADDR pScaleGm{nullptr};
        GM_ADDR actualSeqLengths{nullptr};
    };

    __aicore__ inline BlockEpilogueFaMx() {
        if ASCEND_IS_AIV {
            CrossCoreSetFlag<Gemm::SYNC_MODE_4, PIPE_V>(7);
            CrossCoreSetFlag<Gemm::SYNC_MODE_4, PIPE_V>(9);
            CrossCoreSetFlag<Gemm::SYNC_MODE_4, PIPE_V>(5);
        }
    }
    __aicore__ inline ~BlockEpilogueFaMx() {}

    __aicore__ inline void Init(const Params& params)
    {
        attentionOutGm = reinterpret_cast<__gm__ OUTPUT_T*>(params.outGmAddr);
        attenMaskGmInt = reinterpret_cast<__gm__ uint8_t*>(params.attenMaskGmAddr);
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)params.actualSeqLengths);
        GlobalTensor<float> pScale;
        pScale.SetGlobalBuffer((__gm__ float *)params.pScaleGm);
        pScaleValue = pScale.GetValue(0);
        subBlockIdx = AscendC::GetSubBlockIdx();
        InitBuffers();
    }

    // ================= 流水化拆分（while 软件流水使用） =================
    // 原来 vec1(softmax) 和 vec2(输出) 焊在 operator() 里同步执行，无法错开 2 拍。
    // 现在拆成三个：
    //   - Prepare      : 把 blockShape/coord 解析成成员变量缓存（vec1/vec2 都要用）
    //   - vec1Process  : 做 softmax（读 AIC MM1 的 QK 结果 mm1Res，把 P/Scale 写回 L1 p1）
    //   - vec2Process  : 做最终输出（读 AIC MM2 的 mm2Res，做 FlashUpdate + Cast 写到 GM）
    //   - operator()   : 兼容入口，等价 Prepare + vec1Process + vec2Process（供 for 版本用）
    __aicore__ inline void operator()(const BlockShape& blockShape, const BlockCoord& coord, uint64_t mLoop)
    {
        Prepare(blockShape, coord, mLoop);
        vec1Process();
        vec2Process();
    }

    // 公共状态准备：从 blockShape/coord 取出本次任务的尺寸/索引，存成成员，供 vec1/vec2 复用。
    // 因为 vec1 和 vec2 是分别被调用的，它们必须共享同一份"本次任务"的状态，所以要先 Prepare。
    __aicore__ inline void Prepare(const BlockShape& blockShape, const BlockCoord& coord, uint64_t mLoop)
    {
        actS1Size = AscendC::Te::Get<0>(blockShape); // 当前head的s1大小
        actS2Size = AscendC::Te::Get<1>(blockShape); // 当前head的s2大小
        curS1gBlockSize = AscendC::Te::Get<2>(blockShape); // 当前循环实际处理的s1g大小
        curS2BlockSize = AscendC::Te::Get<3>(blockShape);
        curBIdx = AscendC::Te::Get<0>(coord);
        curN2Idx = AscendC::Te::Get<1>(coord);
        curS1gIdx = AscendC::Te::Get<2>(coord);
        curS2Idx = AscendC::Te::Get<3>(coord);
        curMLoop = mLoop;
        mSizeAlign16 = Gemm::Align32(curS1gBlockSize) >> 1;  // vec 一次处理的行数（32对齐后/2，双 subblock 分配用）
        actVecMSize = (curS1gBlockSize + 1) >> 1;            // 本 subblock 负责的行数
        vecMbaseIdx = subBlockIdx * mSizeAlign16;            // 本 subblock 在总行里的起始偏移
        nextTokensLeftUp = actS2Size - actS1Size;            // 因果掩码偏移量
    }

    // ---- vec1：softmax ----
    // 读 AIC MM1 写在 UB(mm1Res) 的 QK 分数，逐 256 列做 softmax，得到 P，
    // 并把 P 及其 scale 写回 L1(p1) 供 AIC 的 MM2 使用。结束时 set 跨核标志通知 AIC。
    __aicore__ inline void vec1Process()
    {
        auto s1gBlockSize = curS1gBlockSize;
        auto s2BlockSize = curS2BlockSize;
        auto s1gIdx = curS1gIdx;
        auto s2Idx = curS2Idx;
        auto mLoop = curMLoop;
        auto mSizeAlign16 = this->mSizeAlign16;
        constexpr uint32_t s2TileSize = 256;
        uint32_t nLoopNum = Gemm::CeilDiv(static_cast<uint32_t>(s2BlockSize), s2TileSize);
        auto mm2Index = mm2Index_ % 1;
        // s2Size按256切分，分2轮循环计算
        for (uint32_t loop = 0; loop < nLoopNum; ++loop) {
            auto s2CurSize = (loop + 1 == nLoopNum) ? (s2BlockSize - loop * s2TileSize) : s2TileSize;
            auto syncId = 6 + (qkOutUbIdx_ & 0x1) * 2;
            // 跨核同步：等 AIC 的 MM1 把这一段 QK 结果写好（放在 syncId 对应的 UB 槽）
            AscendC::CrossCoreWaitFlag<Gemm::SYNC_MODE_4, PIPE_V>(syncId); // 6, 8

            if constexpr (HAS_MASK) {
                // 把注意力 mask 拷进 UB（当前 s1g 行 × 当前 s2 列）
                AttenMaskCopyInMX(s1gBlockSize, s2CurSize, actS1Size, actS2Size, loop, s1gIdx, s2Idx);                
            }

            // 加锁保护这段 PIPE_V 计算（缓存互斥，避免与其它任务冲突）
            asc_lock(pipe_t::PIPE_V, (mm2Index_ % 2 + 3));
            asc_lock(pipe_t::PIPE_V, (mLoop % 3 + 5));
            asc_lock(pipe_t::PIPE_V, 2);
            asc_lock(pipe_t::PIPE_V, 0);
           
            uint32_t tmp1 = 0xFF7FFFFF;
            float negativeFloatScalar = *((float *)&tmp1);   // -FLT_MAX，用于上个 row 无效时把分数压到极小
            // 本 s1g 组第一个 KV 块（s2Idx==0）用"初始化"版本，否则用"累加更新"版本（在线 softmax）
            if (s2Idx == 0) {
                process_vec1_noupdate(mSizeAlign16, loop, negativeFloatScalar, mLoop);
            } else{
                process_vec1_update(mSizeAlign16, loop, negativeFloatScalar, mLoop);
            }

            asc_unlock(pipe_t::PIPE_V, (mm2Index_ % 2 + 3));
            asc_unlock(pipe_t::PIPE_V, (mLoop % 3 + 5));
            asc_unlock(pipe_t::PIPE_V, 2);
            asc_unlock(pipe_t::PIPE_V, 0);

            // 通知 AIC：这段 softmax 算完了，可以继续
            AscendC::CrossCoreSetFlag<Gemm::SYNC_MODE_4, PIPE_V>(syncId + 1); // 7, 9
            // 把 P(和 scale)从 UB 拷到 L1(p1)，供 AIC MM2 使用
            asc_lock(pipe_t::PIPE_MTE3, 1);
            asc_lock(pipe_t::PIPE_MTE3, 2);
            if ((loop > 0)) {
                ub_l1_vec1_pscale((__cbuf__ fp8_e8m0_t*)p1[mm2Index_ % 3] + mBaseSize * s2BaseSize, mSizeAlign16);
            } else if (s2BlockSize <= s2TileSize) {
                ub_l1_vec1_pscale2((__cbuf__ fp8_e8m0_t*)p1[mm2Index_ % 3] + mBaseSize * s2BaseSize, mSizeAlign16);
            }
            asc_unlock(pipe_t::PIPE_MTE3, 2);

            asc_lock(pipe_t::PIPE_MTE3, (mm2Index_ % 2 + 3));

            ub_l1_vec1(p1[mm2Index_%3], mSizeAlign16, loop, mm2Index_ % 2);  // P: UB -> L1

            asc_unlock(pipe_t::PIPE_MTE3, (mm2Index_ % 2 + 3));
            asc_unlock(pipe_t::PIPE_MTE3, 1);
            

            qkOutUbIdx_++;   // UB 双缓冲轮替
        }

        // 通知 AIC：整个 softmax(P) 已写完 L1，MM2 可以取了
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(mm2Index); // 0, 1, 2
    }

    // ---- vec2：最终输出 ----
    // 读 AIC MM2 写在 UB(mm2Res) 的 PV 结果，把不同 KV 块的累加结果做 FlashUpdate 合并，
    // 最后反量化(除以 exp 和)并 Cast 成 fp16 写回 GM。结束时 set 跨核标志（配对完成）。
    __aicore__ inline void vec2Process()
    {
        auto s2Idx = curS2Idx;
        auto bIdx = curBIdx;
        auto n2Idx = curN2Idx;
        auto s1gIdx = curS1gIdx;
        auto mLoop = curMLoop;
        auto s1gBlockSize = curS1gBlockSize;

        // V2：跨核同步，等 AIC 的 MM2 把 PV 结果写好
        AscendC::CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(4); // 4
        if (unlikely(actVecMSize == 0)) {
            // 本 subblock 没有有效行，直接返回（但仍要 set 标志告诉 AIC 完成）
            AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CROSS_CORE_SYNC_V2_C2); // 5
            return;
        }

        int64_t vec2CalcSize = mSizeAlign16 * dTemplateAlign64;
        float deSCaleVValue = 1.0f;
 
        __ubuf__ float * dstUb = (__ubuf__ T*)stage2OutBuf;  // 输出累加缓冲
        __ubuf__ float * curUb = (__ubuf__ T*)mm2Res;        // AIC MM2 的原始结果

        asc_lock(pipe_t::PIPE_V, 8);
        if (unlikely(s2Idx == 0)) {
            // 第一个 KV 块：直接把 MM2 结果当初始值拷到输出缓冲
            uint16_t blockLen = vec2CalcSize / (32 / 4);
            copy_ubuf_to_ubuf((__ubuf__ void*)stage2OutBuf, (__ubuf__ void*)mm2Res, 0, 1U, blockLen, 0U, 0U);
        } else {
            // 后续 KV 块：把这次的 MM2 结果和之前的累加按 exp 更新在线合并
            __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[mm2Index_ % 3];
            if ((s2Idx + 1) * s2BaseSize < actS2Size) {
                // 还不是最后一个 KV 块：普通 FlashUpdate（只更新 max 和累加和）
                __ubuf__ float * preUb = (__ubuf__ T*)stage2OutBuf;
                FlashUpdateBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, mSizeAlign16, dTemplateAlign64, 1.0, 1.0);
            } else {
                // 最后一个 KV 块：FlashUpdate 同时带上 exp 和（最后要除的）
                __ubuf__ float * preUb = (__ubuf__ T*)stage2OutBuf;
                __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mLoop % 3];
                FlashUpdateLastBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, expSumUb, mSizeAlign16, dTemplateAlign64, 1.0,
                    1.0);
            }
        }
        asc_unlock(pipe_t::PIPE_V, 8);

        // 通知 AIC：vec2 做完了
        AscendC::CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(5); // 5
        if ((s2Idx + 1) * s2BaseSize >= actS2Size) {
            // 这是该 s1g 组的最后一个 KV 块：此时输出累积完成，做除 exp 和 + Cast + 写 GM
            if (unlikely(s2Idx == 0)) {
                __ubuf__ float * srcUb = (__ubuf__ T*)stage2OutBuf;
                __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mLoop % 3];
                LastDivNewVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    dstUb, srcUb, expSumUb, mSizeAlign16, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            // 计算本 subblock 实际要写回 GM 的行数（分成 32 一组，按 subblock 分配上/下半）
            uint32_t gmDealRowCount;
            uint32_t groupsOf32 = (s1gBlockSize + 31) / 32;
            if (subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > s1gBlockSize ? s1gBlockSize : groupsOf32 * 16;
            } else {
                int32_t vec1RemainRows = s1gBlockSize - 16 * groupsOf32;
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
            
            if (gmDealRowCount == 0) {
                return;
            }
            
            // Cast 成 fp16 并写回 GM 的 attentionOut（按 bIdx/n2Idx 定位）
            Bmm2ResCastAndCopyOut(mLoop, bIdx, n2Idx, s1gIdx, dstUb, 0, mSizeAlign16, gmDealRowCount);
        }

        mm2Index_++;   // 独立自推进，保证下次 vec2 用下一组输出缓冲
    }

private:
    __aicore__ inline void ub_l1_vec1(__cbuf__ void *p1, uint32_t actMSizeAlign16, uint32_t subLoop, uint32_t mm2Index)
    {
        // block_count = 256 / 32;
        // block_len = (uint16_t)16;//(uint16_t)actMSizeAlign16
        // srcGap = (uint16_t)(65 - 16);//(uint16_t)(vec1Srcstride - actMSizeAlign16)
        // dstGap = (uint16_t)(128 - 16);//(uint16_t)(128 - actMSizeAlign16)
        int64_t dstOffset =
                subBlockIdx * 32 * actMSizeAlign16 +
                s2SplitSize * mBaseSize * subLoop;
        copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ float8_e4m3_t *)p1 + dstOffset), 
                          (__ubuf__ void*)((__ubuf__ float8_e4m3_t *)stage1OutQue[mm2Index]), 
                           0, s2SplitSize / 32, (uint16_t)actMSizeAlign16 , (uint16_t)(vec1Srcstride - actMSizeAlign16), (uint16_t)(128 - actMSizeAlign16));

    }

    __aicore__ inline void ub_l1_vec1_pscale(__cbuf__ void *pscale, uint32_t actMSizeAlign16)
    { 
        uint16_t copyCount = actMSizeAlign16 / 16; // 1
        uint64_t pScaleDataLen = actMSizeAlign16 * s2SplitSize / MXFP_GROUP_SIZE; // 4*16*2
        uint64_t vecOffset = subBlockIdx * pScaleDataLen;
        uint64_t subLoopOffset = pScaleDataLen * 2; // 8*16*2
        uint16_t dstStride = s2SplitSize / MXFP_GROUP_SIZE / 2 - 1;
        for (uint16_t i = 0; i < 4; i++) { // PScale在s2方向的block块大小为32，所以一共有256/32=8个，而L1上需要满足16x2的分形，所以重复拷贝4次
            copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ fp8_e8m0_t *)pscale + vecOffset + i * 32), (__ubuf__ void*)pScaleSubLoop0UB, 0, copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)dstStride);
            copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ fp8_e8m0_t *)pscale + subLoopOffset + vecOffset + i * 32), (__ubuf__ void*)((__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB + 128), 0,  copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)dstStride);
        }
    }

    __aicore__ inline void ub_l1_vec1_pscale2(__cbuf__ void *pscale, uint32_t actMSizeAlign16)
    {
        uint16_t copyCount = actMSizeAlign16 / 16;
        uint64_t pScaleDataLen = actMSizeAlign16 * s2SplitSize / MXFP_GROUP_SIZE;
        uint64_t vecOffset = subBlockIdx * pScaleDataLen;
         constexpr uint16_t pScaleDstStride = s2SplitSize / MXFP_GROUP_SIZE / 2 - 1;
        for (uint16_t i = 0; i < 4; i++) {  
            copy_ubuf_to_cbuf((__cbuf__ void*)((__cbuf__ fp8_e8m0_t *)pscale + vecOffset + i * 32), (__ubuf__ void*)pScaleSubLoop0UB, 0, copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)pScaleDstStride);
        }
    }

    __aicore__ inline void process_vec1_update(uint32_t m,uint32_t subLoop, float minValue, uint32_t mLoop, const float pScale = 1.0f)
    {
        const uint32_t blockStride = mBaseSize >> 1 | 0x1;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[mm2Index_ % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[mm2Index_ % 2] + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mLoop % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mLoop % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mLoop % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mLoop % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[mm2Index_ % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mLoop % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res[qkOutUbIdx_ & 0x1];//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];

        __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB;
        __ubuf__ float *preLoopMaxUb = (__ubuf__ float *)preLoopMaxUB;
        __ubuf__ float *preLoopSumUb = (__ubuf__ float *)preLoopSumUB;;
        __ubuf__ float *firstLoopSumUb = (__ubuf__ float *)firstLoopSumUB;

        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloopCount, loop);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloopCount % 3, loop % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 3);
        // __ubuf__ uint32_t *dropMaskUb1 = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
        // __ubuf__ uint32_t *dropMaskUb2 = (__ubuf__ uint32_t *)(dropTensor.GetPhyAddr() + 256 / 16);

        const uint32_t nPadding = (256 + 32 - 1) / 32 * 32;
        const uint32_t oriTailN1 = s2SplitSize - 64 * 2 < 64 ? s2SplitSize - 64 * 2 : 64;
        const uint32_t oriTailN2 = static_cast<int32_t>(s2SplitSize - 64 * 3) <= 0 ? 0 : s2SplitSize - 64 * 3;
        const uint32_t tailN1 = 256 - 64 * 2;
        const uint32_t tailN2 = 256 - 64 * 3;
        uint32_t pltOriTailN1 = oriTailN1;
        uint32_t pltOriTailN2 = oriTailN2;
        uint32_t pltTailN1 = tailN1;
        uint32_t pltTailN2 = tailN2;
        // float divValue = 1.0f / keepProb;
        // uint32_t pltN = 256;
        if (subLoop == 0) {
            process_vec1_update_subloop0<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, m, static_cast<float>(scaleValue), minValue, pScaleValue);
        } else {
            process_vec1_update_subloop1<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, m, static_cast<float>(scaleValue), minValue, pScaleValue);
        }
    }

    __aicore__ inline void process_vec1_noupdate(uint32_t m, uint32_t subLoop, float minValue, uint32_t mLoop, const float pScale = 1.0f)
    {
        const uint32_t blockStride = mBaseSize >> 1 | 0x1;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[mm2Index_ % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[mm2Index_ % 2] + 65 * 128;
        // __ubuf__ pseShiftType *pseUb = (__ubuf__ pseShiftType *)pseTensor.GetPhyAddr();
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mLoop % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mLoop % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mLoop % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mLoop % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[mm2Index_ % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mLoop % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res[qkOutUbIdx_ & 0x1];//这个看mm1_id需要改吗
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB;
        ////printf("vec1: runInfo.mloop = %d, runInfo.loop = %d\n", mloop, loop);
        ////printf("vec1: runInfo.mloop % (PRELOAD_N + 1)= %d, runInfo.loop % (PRELOAD_N + 1) = %d\n", mloop % 3, loop % 3);

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 3);
        // __ubuf__ uint32_t *dropMaskUb1 = (__ubuf__ uint32_t *)dropTensor.GetPhyAddr();
        // __ubuf__ uint32_t *dropMaskUb2 = (__ubuf__ uint32_t *)(dropTensor.GetPhyAddr() + 256 / 16);

        const uint32_t nPadding = (256 + 32 - 1) / 32 * 32;
        const uint32_t oriTailN1 = s2SplitSize - 64 * 2 < 64 ? s2SplitSize - 64 * 2 : 64;
        const uint32_t oriTailN2 = static_cast<int32_t>(s2SplitSize - 64 * 3) <= 0 ? 0 : s2SplitSize - 64 * 3;
        const uint32_t tailN1 = 256 - 64 * 2;
        const uint32_t tailN2 = 256 - 64 * 3;
        uint32_t pltOriTailN1 = oriTailN1;
        uint32_t pltOriTailN2 = oriTailN2;
        uint32_t pltTailN1 = tailN1;
        uint32_t pltTailN2 = tailN2;
        // float divValue = 1.0f / keepProb;
        // uint32_t pltN = 256;

        if (subLoop == 0) {
            process_vec1_noupdate_subloop0<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expSumUb, indexesUb, 
                maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, m, static_cast<float>(scaleValue), minValue, pScaleValue);
        } else {
            process_vec1_noupdate_subloop1<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb,
                inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, 
                maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub, nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2, m, static_cast<float>(scaleValue), minValue, pScaleValue);
        }
    }

    __aicore__ inline void AttenMaskCopyInMX(uint32_t s1gSize, uint32_t s2RealSize, uint32_t actS1Size, uint32_t actS2Size, 
        uint32_t subLoop, uint32_t s1gIdx, uint32_t s2Idx)
    {
        uint32_t mSizeAlign16 = Gemm::Align32(s1gSize) >> 1;
        uint32_t gs1StartIdx = s1gIdx * mBaseSize + AscendC::GetSubBlockIdx() * mSizeAlign16;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2StartIdx = s2Idx * s2BaseSize + s2BaseSizeCur * subLoop;
        bool IsSkipMask = IsSkipAttentionmaskMX(gs1StartIdx, s2StartIdx);

        if (IsSkipMask) {
            asc_lock(pipe_t::PIPE_V, 0);
            DuplicateFromScalarImpl<uint8_t, Reg::RegTensor<uint8_t, Reg::RegTraitNumOne>>(
                (__ubuf__ uint8_t*)attenMaskUB[0], static_cast<uint8_t>(0U), mSizeAlign16 * s2BaseSizeCur);
            asc_unlock(pipe_t::PIPE_V, 0);
            return;
        } else {    
            AttentionmaskCopyInForSgLayoutMX(s2StartIdx, gs1StartIdx, mSizeAlign16, s2RealSize, actS1Size, actS2Size);//0,16,16,1
        }
    }

    __aicore__ inline void AttentionmaskCopyInForSgLayoutMX(int32_t s2StartIdx, uint32_t gs1StartIdx, uint32_t mDealSize, uint32_t s2RealSize, uint32_t actS1Size, uint32_t actS2Size)
    {
        uint32_t s1StartIdx = gs1StartIdx / gSize;//0/8=0
        uint32_t s1EndIdx = (gs1StartIdx + mDealSize - 1) / gSize;//(0+16-1)/8 = 1
        uint32_t s1Count = s1EndIdx - s1StartIdx + 1;//1
        uint32_t headGSize = mDealSize;//16
        if (s1Count > 1) {
            headGSize = (gs1StartIdx % gSize == 0) ? 0 : (gSize - gs1StartIdx % gSize);
        }
        uint32_t remainRowCount = mDealSize - headGSize;//16
        uint32_t midS1Count = remainRowCount / gSize;//2
        uint32_t tailGSize = remainRowCount % gSize;//0
        uint32_t attenMaskSizeAlign = Gemm::Align32(s2RealSize);//32
        uint32_t attenMaskDstStride = (256 - Gemm::Align32(s2RealSize)) / 32;//(256-32)/32=7
        uint32_t attenMaskS2Stride = attenMaskSizeAlign + 32 * attenMaskDstStride;//32+ 256 -32=256
    
        // // ub-head
        if (headGSize > 0) {
            asc_lock(pipe_t::PIPE_MTE2, 0);
            AttentionmaskDataCopyMX(mDealSize, s2RealSize, s2StartIdx,  s1StartIdx, s1StartIdx + 1);
            asc_unlock(pipe_t::PIPE_MTE2, 0);
            s1StartIdx++;
        }
    
        // ub-remain
        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffsetMX(mDealSize, s2RealSize, s2StartIdx, s1StartIdx);//252*2048 ，2
            uint16_t blockCount = midS1Count + (tailGSize > 0);//2每次搬一个G大小
            uint32_t blockLen = 256;//2
            int64_t srcStride = attenMaskS2Size - 256;//2048-2
            int64_t dstStride = (gSize - 1) * attenMaskS2Stride / 32 + attenMaskDstStride;//(8-1)*256 /32 + （256-32）/32=63
            
            const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)((__gm__ uint8_t*)attenMaskGmInt + maskOffset)) >> 60);
            // DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - info.s2dealNum), 1U};//isPad,leftPadding,rightPadding,paddingvalue
            set_mov_pad_val(GetScalarBitcodeValue(1U));
            uint32_t unitOfBytes = 1;
            uint32_t padLength = 0 * 1 + (attenMaskSizeAlign - 256) * 1;
            uint64_t srcStride310 = srcStride * unitOfBytes + blockLen; // GM  DataCopy:32Bytes, DataCopyPad:1Byte
            uint32_t  dstStride310 =
                AlignUp(static_cast<uint32_t>(dstStride) * 32 + blockLen + padLength, 32);//(8-1)*256 + （256-32） + 2 +（32-2）   g*256
            uint8_t leftPaddingCnt = 0;
            uint8_t rigntPaddingCnt = 0;

            using PrimType = PrimT<uint8_t>;
            asc_lock(pipe_t::PIPE_MTE2, 0);
            // mask  2048*2048中挑适合这个块的mask值
            copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0] + headGSize * s2SplitSize,
                                     (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 
                                        blockCount, blockLen,
                                        0, attenMaskSizeAlign - 256, true, cacheMode, srcStride310,
                                        dstStride310);
            asc_unlock(pipe_t::PIPE_MTE2, 0);
        }
        // S1*S2行的值   扩充到S1G * S2行
        asc_lock(pipe_t::PIPE_V, 0);
        MaskUbCopyS1GMX((__ubuf__ uint8_t*)attenMaskUB[0], headGSize, gSize, tailGSize, midS1Count);
        asc_unlock(pipe_t::PIPE_V, 0);
    }

    __aicore__ inline void AttentionmaskDataCopyMX(uint32_t mDealSize, uint32_t s2RealSize, 
                                                int32_t s2StartIdx, uint32_t s1StartIdx, uint32_t s1EndIdx, bool isPre = false)
    {
        uint32_t attenMaskSizeAlign = Gemm::CeilAlign(s2RealSize, 32U);
        uint64_t maskOffset = ComputeAttenMaskOffsetMX(mDealSize, s2RealSize, s2StartIdx, s1StartIdx);
        // DataCopyExtParams dataCopyParams;
        // dataCopyParams.blockCount = s1EndIdx - s1StartIdx;
        // dataCopyParams.blockLen = s2SplitSize / 32;
        // dataCopyParams.srcStride = (info.attenMaskS1Stride - s2SplitSize) / 32;
        // dataCopyParams.dstStride = 0;
        // DataCopyPadExtParams<T> padParams{true, 0, 0, 1U};
        const uint8_t cacheMode = ExtractCacheMode((__gm__ uint8_t*)attenMaskGmInt + maskOffset);
        // DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(attenMaskSizeAlign - s2RealSize), 1U};
        // set_mov_pad_val(GetScalarBitcodeValue(1U));
        // DataCopyPad(attenMaskUb, srcGmAddr[maskOffset], dataCopyParams, padParams);

        uint16_t blockCount = s1EndIdx - s1StartIdx;
        uint32_t blockLen = s2SplitSize / 32;
        int64_t srcStride = (attenMaskS2Size - s2SplitSize) / 32;
        int64_t dstStride = 0;

        uint32_t unitOfBytes = 32;
        uint32_t burstLength = blockLen * unitOfBytes;
        uint64_t srcStride310 = srcStride * unitOfBytes + burstLength; // GM  DataCopy:32Bytes, DataCopyPad:1Byte
        uint32_t dstStride310 = AlignUp(static_cast<uint32_t>(dstStride) * 32 + burstLength, 32);

        copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)attenMaskUB[0],
                                     (__gm__ uint8_t*)attenMaskGmInt + maskOffset, 0, 
                                        blockCount, burstLength,
                                        0, 0, false, cacheMode, srcStride310,
                                        dstStride310);
    }

     template <typename T1, typename RegType>
    __simd_vf__ inline void DuplicateFromScalarImpl(__ubuf__ T1 * dstLocal, const T1 scalarValue, const int32_t calCount)
    {
        RegType dstReg;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        Reg::MaskReg mask;
        constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() * RegType::trait.REG_NUM);
        uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
        Reg::MaskReg maskFull = Reg::CreateMask<T1, Reg::MaskPattern::ALL, RegType::trait>();
        Reg::Duplicate(dstReg, scalarValue, maskFull);
        for (uint16_t i = 0; i < repeatTime; ++i) {
            mask = Reg::UpdateMask<T1, RegType::trait>(sreg);
            Reg::StoreAlign(dstLocal + i * repeatStride, dstReg, mask);
        }
    }

    __aicore__ inline uint64_t ComputeAttenMaskOffsetMX(uint32_t mDealSize, uint32_t s2RealSize, int32_t s2StartIdx, uint32_t s1StartIdx)
    {
        uint64_t offset = 0;
        int64_t delta = nextTokensLeftUp + s1StartIdx - s2StartIdx;//512-4-256，   -2+0-0
        uint32_t attenMaskSizeAlign = Gemm::Align32(s2RealSize);//32
        if (delta < 0) {
            offset = (-delta) < static_cast<int64_t>(mDealSize) ? (-delta) : mDealSize; // min (-delta, s1Size)
        } else {
            offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) * attenMaskS2Size; // min(delta, s2inner)
        }

        return offset;
    }

    __aicore__ inline bool IsSkipAttentionmaskMX(uint32_t gs1StartIdx, uint32_t s2StartIdx)
    {
        int32_t s1StartIdx = gs1StartIdx / gSize;
        if (static_cast<int64_t>(s2StartIdx + s2SplitSize) <= static_cast<int64_t>(s1StartIdx) + nextTokensLeftUp) {
            return true;
        }
        return false;
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(uint32_t s1gIdx, int64_t s1FirstValidToken, int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = s1gIdx * mBaseSize + vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        // S1G layout
        s1StartTdx = vecMStartIdx / gSize;
        s1EndTdx = vecMEndIdx / gSize;
        ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
       
        return ret;
    }
 
    __aicore__ inline void RowInvalid(__ubuf__ float *vec2ResUb, int64_t mStartVec, uint32_t mDealSize, int64_t dSizeAligned64, uint32_t mloop, uint32_t s1gIdx)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken = Gemm::Min(Gemm::Max(static_cast<int64_t>(-nextTokensLeftUp), 0L), static_cast<int64_t>(actS1Size));
            int64_t s1LastValidToken = Gemm::Min(Gemm::Max(static_cast<int64_t>(preTokens + actS2Size), 0L), static_cast<int64_t>(actS1Size));
            s1LastValidToken = Gemm::Max(s1LastValidToken - 1, 0L);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < actS1Size);
            bool batchNeedRowInvalid = isRowInvalidOpen || // 手动开启行无效
                                       ((sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }
 
            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(s1gIdx, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || isRowInvalidOpen;
 
            if (blockNeedRowInvalid) {
                uint32_t min = 0xFF7FFFFF; // min value of float
                __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3]+ mStartVec;
                RowInvalidUpdateVF_MX<float>(vec2ResUb, maxUb, mDealSize, dVBaseSize,
                                          static_cast<uint32_t>(dSizeAligned64), min);//其实是constInfo.dSizeV
            }
        }
    }
 
    __aicore__ inline void Bmm2DataCopyOutTrans(uint32_t bIdx, uint32_t n2Idx, uint32_t s1gIdx, __ubuf__ OUTPUT_T * attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        auto gs1StartIdx = s1gIdx * mBaseSize;
        uint32_t s1IdxStart = (gs1StartIdx + vecMbaseIdx) / gSize;//0,16
        uint32_t gIdxStart = (gs1StartIdx + vecMbaseIdx) % gSize;
        uint32_t s1IdxEnd = (gs1StartIdx + vecMbaseIdx + gmDealRowCount) / gSize;
        uint32_t gIdxEnd = (gs1StartIdx + vecMbaseIdx + gmDealRowCount) % gSize;
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = gSize - gIdxStart;
        }
        uint32_t blockCount = headSize;// (gS1Idx + vecMbaseIdx)/8
        uint32_t blockLen = dVBaseSize * sizeof(OUTPUT_T);
        uint32_t srcStride = 0;//(dTemplateAlign64 - dSizeV)/(32/2);
        uint32_t dstStride = 0;//(dSizeV - dSizeV) * sizeof(OUT_T);
        
        uint64_t tIdx = 0;
        if (bIdx == 0) {
            tIdx = 0;
        } else {
            tIdx = actualSeqLengthsGmQ.GetValue(bIdx - 1);
        }
        tIdx += (gs1StartIdx + vecMbaseIdx + vecMIdx) / gSize;
        uint64_t attenOutGmbaseOffset = tIdx * n2Size * gSize * dVBaseSize + n2Idx * gSize * dVBaseSize;
        uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * dVBaseSize;
        uint64_t ubOffset = 0;
 
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)((__gm__ OUTPUT_T*)attentionOutGm + gmOffset)) >> 60);
        uint32_t unitOfBytes = 1;
        uint32_t burstLength = blockLen;
        uint32_t srcStride310 = 0;
        srcStride310 = static_cast<uint32_t>(srcStride) * 32
                       + burstLength; // UB   DataCopy:32Bytes, DataCopyPad:32Bytes
        // srcstride should be 32B aligned
        srcStride310 = DivCeil(srcStride310, 32) * 32;
        
        uint64_t dstStride310 = dstStride + burstLength; // GM   DataCopy:32Bytes, DataCopyPad:1Byte
 
        // set_mark(0x4511223);
        // set_mark(gmOffset);//0
        // set_mark(ubOffset);//0
        // set_mark(blockCount);//4
        // set_mark(burstLength);//100
        // set_mark(dstStride310);//100
        // set_mark(srcStride310);//100
        copy_ubuf_to_gm_align_v2((__gm__ void*)((__gm__ OUTPUT_T*)attentionOutGm + gmOffset), 
                                 (__ubuf__ void*)((__ubuf__ OUTPUT_T *)attenOutUb + ubOffset), 0, blockCount, burstLength, cacheMode, dstStride310, srcStride310);
        if (s1IdxEnd - s1IdxStart >= 1) {
            if (s1IdxEnd > 1 + s1IdxStart) {
                gmOffset = attenOutGmbaseOffset + n2Size * gSize * dVBaseSize;
                ubOffset = ((uint64_t)headSize) * ((uint64_t)dVBaseSize);
                uint32_t blockCountTmp = s1IdxEnd - s1IdxStart -1;
                uint32_t blockLenTmp = dVBaseSize * sizeof(OUTPUT_T) * gSize;//dVBaseSize * sizeof(OUT_T) * gSize
                uint32_t srcStrideTmp = 0;
                uint64_t dstStrideTmp = n2Size * gSize * dVBaseSize * 2; // constInfo.n2Size * gSize * dVBaseSize * sizeof(OUT_T) - blockLenTmp
                uint32_t srcStride310Tmp = DivCeil(blockLenTmp, 32) * 32;
                const uint8_t cacheMode2 = static_cast<uint8_t>(((uint64_t)((__gm__ OUTPUT_T*)attentionOutGm + gmOffset)) >> 60);
                copy_ubuf_to_gm_align_v2((__gm__ void*)(attentionOutGm + gmOffset), (__ubuf__ void*)(attenOutUb + ubOffset), 0, blockCountTmp, blockLenTmp,
                                        cacheMode2, dstStrideTmp, srcStride310Tmp);
                gmOffset += n2Size * gSize * dVBaseSize * blockCountTmp;
                ubOffset += gSize * dTemplateAlign64 * blockCountTmp;
            }
            // 处理尾块
            if (gIdxEnd > 0) {
                blockCount = gIdxEnd;
                const uint8_t cacheMode3 = static_cast<uint8_t>(((uint64_t)((__gm__ OUTPUT_T*)attentionOutGm + gmOffset)) >> 60);
                copy_ubuf_to_gm_align_v2((__gm__ void*)(attentionOutGm + gmOffset), (__ubuf__ void*)(attenOutUb + ubOffset), 0, blockCount, blockLen,
                                        cacheMode3, dstStride310, srcStride310);
 
            }
        }
    }
    
    __aicore__ inline void Bmm2ResCastAndCopyOut(uint32_t mloop, uint32_t bIdx, uint32_t n2Idx, uint32_t s1gIdx, __ubuf__ float *vec2ResUb, uint32_t mStartVec, uint32_t mDealSize, uint32_t gmDealRowCount)
    {
        __ubuf__ OUTPUT_T *attenOut;
        attenOut = (__ubuf__ OUTPUT_T *)vec2ResUb;
        RowInvalid(vec2ResUb, mStartVec, mDealSize, dTemplateAlign64, mloop, s1gIdx);
        using DstPrimType = PrimT<OUTPUT_T>;
        using SrcPrimType = PrimT<float>;
        using DST_TYPE = typename CastParam::CastTypeTrait<DstPrimType>::RealType;
        using SRC_TYPE = typename CastParam::CastTypeTrait<SrcPrimType>::RealType;
        __ubuf__ DST_TYPE* dst = reinterpret_cast<__ubuf__ DST_TYPE*>(attenOut);
        __ubuf__ SRC_TYPE* src = reinterpret_cast<__ubuf__ SRC_TYPE*>(vec2ResUb);
        CastIntrinsicsImplVF_MX<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND>(dst, src, mDealSize * dTemplateAlign64);
           
        set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        asc_lock(pipe_t::PIPE_MTE3, 8);
        Bmm2DataCopyOutTrans(bIdx, n2Idx, s1gIdx, attenOut, mStartVec, mDealSize, gmDealRowCount);
        asc_unlock(pipe_t::PIPE_MTE3, 8);
    }

    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t stage1OutSize = (64 + 1) * 256;
        constexpr uint32_t maskSize = 64  * 256;
        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        constexpr uint32_t mm2LeftSize = 128 * 512 + 128 * 512 / 32;
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        p1[0] = (__cbuf__ void *)get_imm(0);   
        p1[1] = (__cbuf__ void *)get_imm(mm2LeftSize);
        p1[2] = (__cbuf__ void *)get_imm(mm2LeftSize * 2);

        mm1Res[0] = (__ubuf__ void *)get_imm(0); 
        mm1Res[1] = (__ubuf__ void *)get_imm(mm1ResultSize);
        mm2Res = (__ubuf__ void *)get_imm(mm1ResultSize * 2);
        stage1OutQue[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize);
        stage1OutQue[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize);

        attenMaskUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2);//16384
        softmaxSumUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize);
        softmaxSumUB[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256);
        softmaxSumUB[2] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 2);
        softmaxMaxUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 3);
        softmaxMaxUB[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 4);
        softmaxMaxUB[2] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 5);
        softmaxExpUB[0] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 6);
        softmaxExpUB[1] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 7);
        softmaxExpUB[2] = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 8);
        preLoopMaxUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 9);
        preLoopSumUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 10);
        firstLoopSumUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 11);
        commonUB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12);

        stage2OutBuf = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512);
        pScaleSubLoop0UB = (__ubuf__ void *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4);
        vselrIndexesUB[0] = (__ubuf__ uint8_t *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4 + 512);
        vselrIndexesUB[1] = (__ubuf__ uint8_t *)get_imm(mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4+ 512 + 128);
        AscendC::LocalTensor<uint8_t> vselrIndexTensor{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE / sizeof(uint8_t)};
        for (int i = 0; i < 128; i++) {
            vselrIndexTensor[mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4 + 512].SetValue(i, i * 2);
        }
        for (int i = 0; i < 64; i++) {
            vselrIndexTensor[mm1ResultSize * 2 + mm2ResultSize + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4+ 512 + 128].SetValue(i, i * 4);
        }
    }

private:
    static constexpr uint32_t mBaseSize = 128;
    static constexpr uint32_t s2BaseSize = 512;
    static constexpr uint32_t dBaseSize = 128;
    static constexpr uint32_t dVBaseSize = 128;
    static constexpr uint32_t dTemplateAlign64 = 128;
    static constexpr uint32_t s2SplitSize = 256;
    static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr int64_t attenMaskS2Size = 2048;
    static constexpr int64_t preTokens = 2147483647;
    uint8_t subBlockIdx = 0;
    float pScaleValue { 1.0f };
    float scaleValue {0.0883883476483184f};
    

    uint32_t actVecMSize = 0;
    
    int64_t nextTokensLeftUp = 0;//修改 actSeqLenKV - actSeqLensQ
    uint32_t actS1Size = 0;//修改
    uint32_t actS2Size = 0;//修改

    // 流水拆分后由 Prepare 缓存、vec1Process/vec2Process 共用的状态
    uint32_t curS1gBlockSize = 0;
    uint32_t curS2BlockSize = 0;
    uint32_t curBIdx = 0;
    uint32_t curN2Idx = 0;
    uint32_t curS1gIdx = 0;
    uint32_t curS2Idx = 0;
    uint64_t curMLoop = 0;
    uint32_t mSizeAlign16 = 0;
    static constexpr bool isRowInvalidOpen = 0;
    static constexpr uint32_t sparseMode = 3;
    uint32_t vecMbaseIdx = 0U;

    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 4;
    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    __gm__ float *pScaleGm;
    __ubuf__ void *stage1OutQue[2]; // bufferId = 3,4
    __ubuf__ void *stage2OutBuf;    // bufferId = 8
    __ubuf__ void *attenMaskUB[2];  // bufferId = 0
    __ubuf__ void *softmaxSumUB[3]; // bufferId = 5,6,7
    __ubuf__ void *softmaxMaxUB[3];
    __ubuf__ void *softmaxExpUB[3];
    __ubuf__ void *preLoopMaxUB;
    __ubuf__ void *preLoopSumUB;
    __ubuf__ void *firstLoopSumUB;
    __ubuf__ void *commonUB;
    __ubuf__ void *pScaleSubLoop0UB; // bufferId = 2
    __ubuf__ uint8_t *vselrIndexesUB[4];
    __ubuf__ float * pScaleUb;
    __ubuf__ void *mm1Res[2];
    __ubuf__ void *mm2Res;
    __cbuf__ void *p1[3];           // bufferId = 1

    using pseShiftType = half;
    __ubuf__ pseShiftType *pseUb;
    __ubuf__ uint32_t *dropMaskUb1;
    __ubuf__ uint32_t *dropMaskUb2;
    __gm__ uint8_t* attenMaskGmInt;
    __gm__ OUTPUT_T* attentionOutGm; 
    uint64_t qkOutUbIdx_{0};
    uint64_t mm2Index_{0};
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze