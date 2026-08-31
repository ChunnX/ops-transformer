/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_fa_mx.h
 * \brief AIV 侧 FA-MX epilogue，完全按 fia_block_vec_fullquant_mx.h 的框架重写：
 *        while 软件流水驱动、ProcessVec1/ProcessVec2 接口、set_intra_block/wait_intra_block 跨核同步。
 *        底层 VF 调用复用 tile_fa_vf.h（Blaze::Epilogue 命名空间，5 模板参数版本）。
 *        原始 for 循环版本备份见同目录 block_epilogue_fa_mx_backup.h。
 *
 * 关键变化（相对原始 for 版本）：
 *   1. operator()/Prepare/vec1Process/vec2Process  →  ProcessVec1/ProcessVec2（状态全部由参数传入）
 *   2. CrossCoreSetFlag/CrossCoreWaitFlag           →  set_intra_block/wait_intra_block
 *   3. asc_lock/asc_unlock                          →  set_flag/wait_flag
 *   4. 形状参数由运行时成员改为 static constexpr（与 FIA 一致，标 //修改 处需按实际 shape 调整）
 *   5. mm1Res/mm2Res/p1 不再是本类成员，由 kernel 分配并通过参数传入
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

using namespace AscendC;

template <bool HAS_MASK = true>
class BlockEpilogueFaMx {
public:
    using INPUT_T = fp8_e4m3fn_t;
    using OUTPUT_T = float16_t;
    using T = float;
    using OUT_T = OUTPUT_T;

    static constexpr bool HAS_ATTEN = HAS_MASK;

    // ===================== static constexpr 形状参数（与 FIA 对齐，//修改 处需按实际调整）=====================
    static constexpr uint32_t actS1Size = (uint32_t)4;          //修改
    static constexpr uint32_t actS2Size = (uint32_t)1024;       //修改
    static constexpr uint32_t mBaseSize = (uint32_t)128;
    static constexpr uint32_t s2BaseSize = (uint32_t)512;
    static constexpr uint32_t dBaseSize = (uint32_t)128;
    static constexpr uint32_t dVBaseSize = (uint32_t)128;
    static constexpr uint32_t dTemplateAlign64 = (uint32_t)128;
    static constexpr uint32_t s2SplitSize = (uint32_t)256;
    static constexpr uint32_t bSize = (uint32_t)1;              //修改
    static constexpr uint32_t n2Size = (uint32_t)8;
    static constexpr uint32_t gSize = (uint32_t)8;
    static constexpr uint32_t realGSize = (uint32_t)8;
    static constexpr uint32_t s1Size = (uint32_t)4;
    static constexpr uint32_t s2Size = (uint32_t)512;
    static constexpr uint32_t maxBlockNumPerBatch = (uint32_t)2;//修改
    static constexpr uint32_t attenMaskS1Size = 2048;
    static constexpr uint32_t attenMaskS2Size = 2048;
    static constexpr uint32_t FirstS2Loop = (uint32_t)0;
    static constexpr uint32_t LastS2Loop = (uint32_t)2;         //修改
    static constexpr uint32_t actMSize = (uint32_t)32;
    static constexpr uint32_t actVecMSize = (uint32_t)16;
    static constexpr uint32_t actMSizeAlign32 = (uint32_t)32;
    static constexpr uint32_t actMSizeAlign16 = (uint32_t)16;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;
    static constexpr int64_t preTokens = 2147483647;
    static constexpr int64_t nextTokensLeftUp = 1024 - 4;       //修改 actSeqLenKV - actSeqLensQ
    static constexpr bool isRowInvalidOpen = 0;
    static constexpr uint32_t sparseMode = 3;
    static constexpr float scaleValue = 0.08838834764831843f;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    static constexpr uint32_t PRELOAD_N = 2;

    struct Params {
        GM_ADDR outGmAddr{nullptr};
        GM_ADDR attenMaskGmAddr{nullptr};
        GM_ADDR pScaleGm{nullptr};
        GM_ADDR actualSeqLengths{nullptr};
    };

    // ---- AIV 专用 UB 缓冲（mm1Res/mm2Res/p1 由 kernel 分配，通过参数传入）----
    __ubuf__ void *stage1OutQue[2];
    __ubuf__ void *stage2OutBuf;
    __ubuf__ void *attenMaskUB[2];
    __ubuf__ void *softmaxSumUB[3];
    __ubuf__ void *softmaxMaxUB[3];
    __ubuf__ void *softmaxExpUB[3];
    __ubuf__ void *preLoopMaxUB;
    __ubuf__ void *preLoopSumUB;
    __ubuf__ void *firstLoopSumUB;
    __ubuf__ void *commonUB;
    __ubuf__ void *pScaleSubLoop0UB;
    __ubuf__ uint8_t *vselrIndexesUB[4];

    __gm__ uint64_t *actualSeqLengthsGmQ;
    __gm__ float *pScaleGm;
    __gm__ uint8_t *attenMaskGmInt;
    __gm__ OUTPUT_T *attentionOutGm;
    uint8_t subBlockIdx = GetSubBlockIdx();
    float pScaleValue{1.0f};
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);
    float negativeFloatScalar = 0.0f;

    __aicore__ inline BlockEpilogueFaMx() {}
    __aicore__ inline ~BlockEpilogueFaMx() {}

    // ===================== Init（对应 FIA InitVecBlock）=====================
    __aicore__ inline void Init(const Params &params)
    {
        uint32_t tmp1 = 0xFF7FFFFF;
        this->negativeFloatScalar = *((float *)&tmp1);

        attentionOutGm = reinterpret_cast<__gm__ OUTPUT_T *>(params.outGmAddr);
        attenMaskGmInt = reinterpret_cast<__gm__ uint8_t *>(params.attenMaskGmAddr);
        actualSeqLengthsGmQ = reinterpret_cast<__gm__ uint64_t *>(params.actualSeqLengths);
        pScaleGm = reinterpret_cast<__gm__ float *>(params.pScaleGm);
        pScaleValue = pScaleGm[0];
        subBlockIdx = GetSubBlockIdx();
    }

    // ===================== InitBuffers（对应 FIA InitBuffers）=====================
    // 只分配 AIV 专用 UB；mm1Res/mm2Res/p1 由 kernel 的 InitMMResBuf 分配。
    __aicore__ inline void InitBuffers()
    {
        constexpr uint32_t stage1OutSize = (64 + 1) * 256;
        constexpr uint32_t maskSize = 64 * 256;
        constexpr uint32_t mm1ResultSize = 128 / 2 * 512 * 4 / 2;
        constexpr uint32_t mm2ResultSize = 128 / 2 * 128 * 4;
        // mm1Res*2 + mm2Res 已由 kernel 分配，本类从其后开始
        uint32_t base = mm1ResultSize * 2 + mm2ResultSize;
        stage1OutQue[0] = (__ubuf__ void *)get_imm(base);
        stage1OutQue[1] = (__ubuf__ void *)get_imm(base + stage1OutSize);

        attenMaskUB[0] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2);
        softmaxSumUB[0] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize);
        softmaxSumUB[1] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256);
        softmaxSumUB[2] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 2);
        softmaxMaxUB[0] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 3);
        softmaxMaxUB[1] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 4);
        softmaxMaxUB[2] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 5);
        softmaxExpUB[0] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 6);
        softmaxExpUB[1] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 7);
        softmaxExpUB[2] = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 8);
        preLoopMaxUB = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 9);
        preLoopSumUB = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 10);
        firstLoopSumUB = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 11);
        commonUB = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 12);

        stage2OutBuf = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 12 + 512);
        pScaleSubLoop0UB = (__ubuf__ void *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4);
        vselrIndexesUB[0] = (__ubuf__ uint8_t *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4 + 512);
        vselrIndexesUB[1] = (__ubuf__ uint8_t *)get_imm(base + stage1OutSize * 2 + maskSize + 256 * 12 + 512 + 64 * 128 * 4 + 512 + 128);
        for (int i = 0; i < 128; i++) {
            *(vselrIndexesUB[0] + i) = i * 2;
        }
        for (int i = 0; i < 64; i++) {
            *(vselrIndexesUB[1] + i) = i * 4;
        }
    }

    __aicore__ inline void AllocEventID()
    {
        set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)1);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)3);
        set_flag(PIPE_MTE3, PIPE_V, (event_t)4);
        set_flag(PIPE_V, PIPE_MTE2, (event_t)4);
    }

    __aicore__ inline void FreeEventID()
    {
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)1);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)2);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)3);
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)4);
        wait_flag(PIPE_V, PIPE_MTE2, (event_t)4);
    }

    // =================================================================================
    //  ProcessVec1：softmax（读 AIC MM1 的 QK 结果，写 P + scale 到 L1 p1）
    //  参数与 FIA ProcessVec1 完全一致：
    //    p1        : L1 中 P 缓冲（3 槽环形之一）
    //    mm1Res    : UB 中 MM1 结果（2 槽之一）
    //    bmm1_id   : MM1 编号（用于跨核同步 id 计算）
    //    ipuse     : L1 P 槽编号（subLoop==1 时用于 set_intra_block 通知 AIC）
    //    bIdx/n2Idx/gS1Idx/s2Idx : 任务坐标
    //    loop      : 全局登记序号（用于 softmax UB 3 槽轮替）
    //    mloop     : (bN2,s1g) 组编号（用于 softmax state 3 槽轮替）
    //    vecMbaseIdx : 本 subblock 在总行里的起始偏移
    //    curS2Loop : s2 下标（FirstS2Loop 用 noupdate，否则 update）
    //    subLoop   : s2Split 内 0/1（256 列一轮，共 2 轮）
    // =================================================================================
    __aicore__ inline void ProcessVec1(__cbuf__ void *p1, __ubuf__ void *mm1Res, int bmm1_id, int ipuse,
                                       uint32_t bIdx, uint32_t gS1Idx, uint32_t s2Idx,
                                       uint32_t loop, uint32_t mloop, uint32_t vecMbaseIdx,
                                       uint32_t curS2Loop, uint32_t subLoop)
    {
        int bmm1_idused = 6 + bmm1_id * 2;
        wait_intra_block(PIPE_V, bmm1_idused + 0);

        wait_flag(PIPE_V, PIPE_MTE2, (event_t)4);  // attenMaskUb AllocTensor
        AttenMaskCopyIn_MX(bIdx, s2Idx, gS1Idx, vecMbaseIdx, actMSizeAlign16, subLoop);
        wait_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);  // stage1OutQue AllocTensor

        wait_flag(PIPE_MTE3, PIPE_V, (event_t)4);  // pscale AllocTensor
        if (curS2Loop == FirstS2Loop) {
            process_vec1_noupdate(mm1Res, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        } else {
            process_vec1_update(mm1Res, bmm1_id, subLoop, loop, mloop, negativeFloatScalar);
        }
        set_intra_block(PIPE_V, bmm1_idused + 1);

        set_flag(PIPE_V, PIPE_MTE2, (event_t)4);  // attenMaskUb FreeTensor

        set_flag(PIPE_V, PIPE_MTE3, (event_t)3);  // pscale EnQue
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)3); // pscale DeQue
        if (subLoop % 2 == 1) {
            ub_l1_vec1_pscale((__cbuf__ fp8_e8m0_t *)p1 + mBaseSize * s2BaseSize, subLoop, loop % 2);
        }
        set_flag(PIPE_MTE3, PIPE_V, (event_t)4);  // pscale FreeTensor

        set_flag(PIPE_V, PIPE_MTE3, (event_t)2);  // stage1OutQue EnQue
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)2); // stage1OutQue DeQue
        ub_l1_vec1(p1, subLoop, loop % 2);
        set_flag(PIPE_MTE3, PIPE_V, 2 + loop % 2);  // stage1OutQue FreeTensor

        if (subLoop % 2 == 1) {
            set_intra_block(PIPE_MTE3, ipuse);
        }
    }

    // =================================================================================
    //  ProcessVec2：最终输出（读 AIC MM2 的 PV 结果，FlashUpdate 合并 + Cast 写 GM）
    //  参数与 FIA ProcessVec2 完全一致
    // =================================================================================
    __aicore__ inline void ProcessVec2(__ubuf__ void *mm2Res, int bmm2_id, uint32_t loop, uint32_t mloop,
                                       uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx,
                                       uint32_t vecMbaseIdx, uint32_t curS2Loop)
    {
        wait_intra_block(PIPE_V, bmm2_id);

        uint32_t vecMSize = actMSizeAlign32 / 2;
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;
        float deSCaleVValue = 1.0f;

        __ubuf__ float *dstUb = (__ubuf__ T *)stage2OutBuf;
        __ubuf__ float *curUb = (__ubuf__ T *)mm2Res;
        wait_flag(PIPE_MTE3, PIPE_V, (event_t)0);

        if (curS2Loop == FirstS2Loop) {
            uint16_t blockLen = vec2CalcSize / (32 / 4);
            copy_ubuf_to_ubuf((__ubuf__ void *)stage2OutBuf, (__ubuf__ void *)mm2Res, 0, 1U, blockLen, 0U, 0U);
        } else {
            __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[loop % 3];
            if ((curS2Loop + 1) != LastS2Loop) {
                __ubuf__ float *preUb = (__ubuf__ T *)stage2OutBuf;
                FlashUpdateBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, vecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                __ubuf__ float *preUb = (__ubuf__ T *)stage2OutBuf;
                __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
                FlashUpdateLastBasicVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, 1, false, false>(
                    dstUb, curUb, preUb, expMaxUb, expSumUb, vecMSize, dTemplateAlign64, 1.0, 1.0);
            }
        }
        set_intra_block(PIPE_V, bmm2_id + 1);

        if ((curS2Loop + 1) == LastS2Loop) {
            if (curS2Loop == FirstS2Loop) {
                __ubuf__ float *srcUb = (__ubuf__ T *)stage2OutBuf;
                __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
                LastDivNewVF_MX<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    dstUb, srcUb, expSumUb, vecMSize, (uint16_t)dTemplateAlign64, deSCaleVValue);
            }
            uint32_t gmDealRowCount;
            uint32_t groupsOf32 = (actMSize + 31) / 32;
            if (subBlockIdx == 0) {
                gmDealRowCount = groupsOf32 * 16 > actMSize ? actMSize : groupsOf32 * 16;
            } else {
                int32_t vec1RemainRows = actMSize - 16 * groupsOf32;
                gmDealRowCount = 0 > vec1RemainRows ? 0 : vec1RemainRows;
            }
            if (gmDealRowCount == 0) {
                set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
                return;
            }
            Bmm2ResCastAndCopyOut(dstUb, 0, bIdx, n2Idx, gS1Idx, vecMbaseIdx, vecMSize, gmDealRowCount, mloop);
        }
        set_flag(PIPE_MTE3, PIPE_V, (event_t)0);
    }

private:
    // ===================== vec1 softmax 包装（调用 tile_fa_vf.h）=====================
    __aicore__ inline void process_vec1_noupdate(__ubuf__ void *mm1Res, int bmm1_id, uint32_t subLoop,
                                                  uint32_t loop, uint32_t mloop, float minValue,
                                                  const float pScale = 1.0f)
    {
        const uint32_t blockStride = mBaseSize >> 1 | 0x1;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2] + 65 * 128;
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[loop % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res;
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB;

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 3);

        const uint32_t nPadding = (256 + 32 - 1) / 32 * 32;
        const uint32_t oriTailN1 = s2SplitSize - 64 * 2 < 64 ? s2SplitSize - 64 * 2 : 64;
        const uint32_t oriTailN2 = static_cast<int32_t>(s2SplitSize - 64 * 3) <= 0 ? 0 : s2SplitSize - 64 * 3;
        const uint32_t tailN1 = 256 - 64 * 2;
        const uint32_t tailN2 = 256 - 64 * 3;
        uint32_t pltOriTailN1 = oriTailN1;
        uint32_t pltOriTailN2 = oriTailN2;
        uint32_t pltTailN1 = tailN1;
        uint32_t pltTailN2 = tailN2;

        if (subLoop == 0) {
            process_vec1_noupdate_subloop0<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expSumUb, indexesUb,
                maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2,
                actMSizeAlign16, static_cast<float>(scaleValue), minValue, pScaleValue);
        } else {
            process_vec1_noupdate_subloop1<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb,
                tmpExpSumUb, tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb,
                maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2,
                actMSizeAlign16, static_cast<float>(scaleValue), minValue, pScaleValue);
        }
    }

    __aicore__ inline void process_vec1_update(__ubuf__ void *mm1Res, int bmm1_id, uint32_t subLoop,
                                                uint32_t loop, uint32_t mloop, float minValue,
                                                const float pScale = 1.0f)
    {
        const uint32_t blockStride = mBaseSize >> 1 | 0x1;
        const uint32_t repeatStride = 1;
        __ubuf__ fp8_e4m3fn_t *expUb1 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2];
        __ubuf__ fp8_e4m3fn_t *expUb2 = (__ubuf__ fp8_e4m3fn_t *)stage1OutQue[loop % 2] + 65 * 128;
        __ubuf__ float *expSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *inExpSumUb = (__ubuf__ float *)softmaxSumUB[mloop % 3];
        __ubuf__ float *tmpExpSumUb = (__ubuf__ float *)commonUB;
        __ubuf__ float *tmpExpSumUb2 = (__ubuf__ float *)commonUB;
        __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *maxUbStart = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *expMaxUb = (__ubuf__ float *)softmaxExpUB[loop % 3];
        __ubuf__ float *inMaxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3];
        __ubuf__ float *tmpMaxUb = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *tmpMaxUb2 = (__ubuf__ float *)commonUB + 64;
        __ubuf__ float *srcUb = (__ubuf__ float *)mm1Res;
        __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)vselrIndexesUB[0];
        __ubuf__ fp8_e8m0_t *pScaleSubLoop0Ub = (__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB;
        __ubuf__ float *preLoopMaxUb = (__ubuf__ float *)preLoopMaxUB;
        __ubuf__ float *preLoopSumUb = (__ubuf__ float *)preLoopSumUB;
        __ubuf__ float *firstLoopSumUb = (__ubuf__ float *)firstLoopSumUB;

        __ubuf__ uint32_t *maskUb1 = (__ubuf__ uint32_t *)attenMaskUB[0];
        __ubuf__ uint32_t *maskUb2 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64);
        __ubuf__ uint32_t *maskUb3 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 2);
        __ubuf__ uint32_t *maskUb4 = (__ubuf__ uint32_t *)((__ubuf__ uint8_t *)attenMaskUB[0] + 64 * 3);

        const uint32_t nPadding = (256 + 32 - 1) / 32 * 32;
        const uint32_t oriTailN1 = s2SplitSize - 64 * 2 < 64 ? s2SplitSize - 64 * 2 : 64;
        const uint32_t oriTailN2 = static_cast<int32_t>(s2SplitSize - 64 * 3) <= 0 ? 0 : s2SplitSize - 64 * 3;
        const uint32_t tailN1 = 256 - 64 * 2;
        const uint32_t tailN2 = 256 - 64 * 3;
        uint32_t pltOriTailN1 = oriTailN1;
        uint32_t pltOriTailN2 = oriTailN2;
        uint32_t pltTailN1 = tailN1;
        uint32_t pltTailN2 = tailN2;

        if (subLoop == 0) {
            process_vec1_update_subloop0<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2,
                actMSizeAlign16, static_cast<float>(scaleValue), minValue, pScaleValue);
        } else {
            process_vec1_update_subloop1<float, fp8_e4m3fn_t, mBaseSize, s2SplitSize, HAS_MASK>(
                expUb1, expUb2, maxUb, maxUbStart, srcUb, expMaxUb, inMaxUb, expSumUb, inExpSumUb, tmpExpSumUb,
                tmpExpSumUb2, tmpMaxUb, tmpMaxUb2, indexesUb, maskUb1, maskUb2, maskUb3, maskUb4, pScaleSubLoop0Ub,
                preLoopMaxUb, preLoopSumUb, firstLoopSumUb,
                nPadding, blockStride, repeatStride,
                pltOriTailN1, pltOriTailN2, pltTailN1, pltTailN2,
                actMSizeAlign16, static_cast<float>(scaleValue), minValue, pScaleValue);
        }
    }

    // ===================== Attention Mask 拷入（FIA 风格）=====================
    __aicore__ inline bool IsSkipAttentionmask_MX(uint32_t gs1StartIdx, uint32_t s2StartIdx)
    {
        int32_t s1StartIdx = gs1StartIdx / gSize;
        int64_t nextToken = static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size);
        if (static_cast<int64_t>(s2StartIdx + s2SplitSize) <= static_cast<int64_t>(s1StartIdx) + nextToken) {
            return true;
        }
        return false;
    }

    __aicore__ inline uint64_t ComputeAttenMaskOffset_MX(int32_t s2StartIdx, uint32_t s1StartIdx)
    {
        int64_t nextToken = static_cast<int64_t>(actS2Size) - static_cast<int64_t>(actS1Size);
        uint64_t offset = 0;
        int64_t delta = nextToken + s1StartIdx - s2StartIdx;
        uint32_t attenMaskSizeAlign = s2SplitSize;
        if (delta < 0) {
            offset = (-delta) < static_cast<int64_t>(actVecMSize) ? (-delta) : actVecMSize;
        } else {
            offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                     attenMaskS2Size;
        }
        return offset;
    }

    __simd_vf__ void MaskUbCopyS1G_MX(__ubuf__ uint8_t *maskUb, int32_t headGLoop, int32_t midGLoop,
                                      int32_t tailGLoop, int32_t midS1Count)
    {
        if (midGLoop <= 1) {
            return;
        }
        constexpr uint32_t repeatStride = 8;
        RegTensor<uint8_t> vreg_g_head;
        RegTensor<uint8_t> vreg_g_mid;
        RegTensor<uint8_t> vreg_g_tail;
        MaskReg preg_all = CreateMask<bool, MaskPattern::ALL>();

        for (uint16_t x = headGLoop; x > 0; x = 0) {
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
            vsldb((RegTensor<uint8_t> &)vreg_g_head, (__ubuf__ uint8_t *&)maskUb,
                  (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                constexpr auto pv = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsstb((RegTensor<uint8_t> &)vreg_g_head, (__ubuf__ uint8_t *&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, pv);
            }
        }
        for (uint16_t x = midGLoop; x > 0; x = 0) {
            for (uint16_t i = 0; i < midS1Count; ++i) {
                constexpr auto pv = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsldb((RegTensor<uint8_t> &)vreg_g_mid, (__ubuf__ uint8_t *&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, pv);
                for (uint16_t j = 1; j < midGLoop; ++j) {
                    constexpr auto pv2 = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                    vsstb((RegTensor<uint8_t> &)vreg_g_mid, (__ubuf__ uint8_t *&)maskUb,
                          (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, pv2);
                }
                mem_bar(VST_VLD);
            }
        }
        for (uint16_t x = tailGLoop; x > 1; x = 0) {
            constexpr auto pv = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
            vsldb((RegTensor<uint8_t> &)vreg_g_tail, (__ubuf__ uint8_t *&)maskUb,
                  (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, pv);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                constexpr auto pv2 = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsstb((RegTensor<uint8_t> &)vreg_g_tail, (__ubuf__ uint8_t *&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, pv2);
            }
        }
    }

    __aicore__ inline void AttentionmaskCopyInForSgLayout_MX(int32_t s2StartIdx, uint32_t gs1StartIdx)
    {
        uint32_t s1StartIdx = gs1StartIdx / gSize;
        uint32_t s1EndIdx = (gs1StartIdx + actVecMSize - 1) / gSize;
        uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
        uint32_t headGSize = actVecMSize;
        if (s1Count > 1) {
            headGSize = (gs1StartIdx % gSize == 0) ? 0 : (gSize - gs1StartIdx % gSize);
        }
        uint32_t remainRowCount = actVecMSize - headGSize;
        uint32_t midS1Count = remainRowCount / gSize;
        uint32_t tailGSize = remainRowCount % gSize;

        if (remainRowCount > 0) {
            uint64_t maskOffset = ComputeAttenMaskOffset_MX(s2StartIdx, s1StartIdx);
            uint16_t blockCount = midS1Count + (tailGSize > 0);
            int64_t srcStride = attenMaskS2Size - s2SplitSize;
            int64_t dstStride = (gSize - 1) * s2SplitSize / 32;
            const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)((__gm__ uint8_t *)attenMaskGmInt + maskOffset)) >> 60);
            set_mov_pad_val(GetScalarBitcodeValue(1U));
            uint32_t unitOfBytes = 1;
            uint32_t padLength = 0;
            uint64_t srcStride310 = srcStride * unitOfBytes + s2SplitSize;
            uint32_t dstStride310 = AlignUp(static_cast<uint32_t>(dstStride) * 32 + s2SplitSize + padLength, 32);
            copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t *)attenMaskUB[0] + headGSize * s2SplitSize,
                                     (__gm__ uint8_t *)attenMaskGmInt + maskOffset, 0,
                                     blockCount, s2SplitSize,
                                     0, 0, true, cacheMode, srcStride310, dstStride310);
        }
        set_flag(PIPE_MTE2, PIPE_V, event_t(0));
        wait_flag(PIPE_MTE2, PIPE_V, event_t(0));
        MaskUbCopyS1G_MX((__ubuf__ uint8_t *)attenMaskUB[0], headGSize, gSize, tailGSize, midS1Count);
    }

    __aicore__ inline void AttenMaskCopyIn_MX(uint32_t bIdx, uint32_t s2Idx, uint32_t gS1Idx,
                                               uint32_t vecMbaseIdx, uint32_t mDealSize, uint32_t subLoop)
    {
        uint32_t gs1StartIdx = gS1Idx + vecMbaseIdx + 0;
        uint32_t s2StartIdx = (subLoop == 0) ? s2Idx : (s2Idx + s2SplitSize);
        bool IsSkipMask = IsSkipAttentionmask_MX(gs1StartIdx, s2StartIdx);
        if (IsSkipMask) {
            DuplicateFromScalarImpl<uint8_t, Reg::RegTensor<uint8_t, Reg::RegTraitNumOne>>(
                (__ubuf__ uint8_t *)attenMaskUB[0], static_cast<uint8_t>(0U), mDealSize * s2SplitSize);
            return;
        } else {
            AttentionmaskCopyInForSgLayout_MX(s2StartIdx, gs1StartIdx);
        }
    }

    // ===================== UB→L1 拷贝（P 和 PScale）=====================
    __aicore__ inline void ub_l1_vec1_pscale(__cbuf__ void *pscale, uint32_t subLoop, uint32_t createdTaskCount)
    {
        uint16_t copyCount = actMSizeAlign16 / 16;
        uint64_t pScaleDataLen = actMSizeAlign16 * 256 / MXFP_GROUP_SIZE;
        uint64_t vecOffset = subBlockIdx * pScaleDataLen;
        uint64_t subLoopOffset = pScaleDataLen * 2;
        uint16_t dstStride = 256 / MXFP_GROUP_SIZE / 2 - 1;
        for (uint16_t i = 0; i < 4; i++) {
            copy_ubuf_to_cbuf((__cbuf__ void *)((__cbuf__ fp8_e8m0_t *)pscale + vecOffset + i * 32),
                              (__ubuf__ void *)pScaleSubLoop0UB, 0, copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)dstStride);
            copy_ubuf_to_cbuf((__cbuf__ void *)((__cbuf__ fp8_e8m0_t *)pscale + subLoopOffset + vecOffset + i * 32),
                              (__ubuf__ void *)((__ubuf__ fp8_e8m0_t *)pScaleSubLoop0UB + 128), 0,
                              copyCount, (uint16_t)1, (uint16_t)0, (uint16_t)dstStride);
        }
    }

    __aicore__ inline void ub_l1_vec1(__cbuf__ void *p1, uint32_t subLoop, uint32_t createdTaskCount)
    {
        int64_t dstOffset = subBlockIdx * 32 * actMSizeAlign16 + 256 * mBaseSize * subLoop;
        copy_ubuf_to_cbuf((__cbuf__ void *)((__cbuf__ float8_e4m3_t *)p1 + dstOffset),
                          (__ubuf__ void *)((__ubuf__ float8_e4m3_t *)stage1OutQue[createdTaskCount]),
                          0, 8, (uint16_t)actMSizeAlign16,
                          (uint16_t)(vec1Srcstride - actMSizeAlign16), (uint16_t)(128 - actMSizeAlign16));
    }

    // ===================== 行无效处理 =====================
    __aicore__ inline bool CalcBlockNeedRowInvalid(uint32_t gS1Idx, uint32_t actVecMSize, uint32_t vecMbaseIdx,
                                                    int64_t s1FirstValidToken, int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = gS1Idx + vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + actVecMSize - 1;
        int32_t s1StartTdx = vecMStartIdx / gSize;
        int32_t s1EndTdx = vecMEndIdx / gSize;
        return (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
    }

    __aicore__ inline void RowInvalid(__ubuf__ float *vec2ResUb, int64_t mStartVec, uint32_t gS1Idx,
                                      uint32_t actVecMSize, uint32_t vecMbaseIdx, uint32_t mDealSize,
                                      uint32_t mloop, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken = Gemm::Min(Gemm::Max(static_cast<int64_t>(-nextTokensLeftUp), 0L),
                                                   static_cast<int64_t>(actS1Size));
            int64_t s1LastValidToken = Gemm::Min(Gemm::Max(static_cast<int64_t>(preTokens + actS2Size), 0L),
                                                  static_cast<int64_t>(actS1Size));
            s1LastValidToken = Gemm::Max(s1LastValidToken - 1, 0L);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < actS1Size);
            bool batchNeedRowInvalid = isRowInvalidOpen ||
                                       ((sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow);
            if (!batchNeedRowInvalid) {
                return;
            }
            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(gS1Idx, actVecMSize, vecMbaseIdx,
                                                                s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || isRowInvalidOpen;
            if (blockNeedRowInvalid) {
                uint32_t min = 0xFF7FFFFF;
                __ubuf__ float *maxUb = (__ubuf__ float *)softmaxMaxUB[mloop % 3] + mStartVec;
                RowInvalidUpdateVF_MX<float>(vec2ResUb, maxUb, mDealSize, dVBaseSize,
                                             static_cast<uint32_t>(dSizeAligned64), min);
            }
        }
    }

    // ===================== 输出 Cast + CopyOut =====================
    __aicore__ inline void Bmm2DataCopyOutTrans(__ubuf__ OUTPUT_T *attenOutUb, uint32_t vecMIdx,
                                                 uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx,
                                                 uint32_t actVecMSize, uint32_t vecMbaseIdx,
                                                 uint32_t dealRowCount, uint32_t gmDealRowCount)
    {
        uint32_t s1IdxStart = (gS1Idx + vecMbaseIdx) / gSize;
        uint32_t gIdxStart = (gS1Idx + vecMbaseIdx) % gSize;
        uint32_t s1IdxEnd = (gS1Idx + vecMbaseIdx + gmDealRowCount) / gSize;
        uint32_t gIdxEnd = (gS1Idx + vecMbaseIdx + gmDealRowCount) % gSize;
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = gSize - gIdxStart;
        }
        uint64_t tIdx = 0;
        if (bIdx == 0) {
            tIdx = 0;
        } else {
            tIdx = actualSeqLengthsGmQ[bIdx - 1];
        }
        tIdx += (gS1Idx + vecMbaseIdx + vecMIdx) / gSize;
        uint64_t attenOutGmbaseOffset = tIdx * n2Size * gSize * dVBaseSize + n2Idx * gSize * dVBaseSize;
        uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * dVBaseSize;
        uint64_t ubOffset = 0;
        const uint8_t cacheMode = static_cast<uint8_t>(((uint64_t)((__gm__ OUTPUT_T *)attentionOutGm + gmOffset)) >> 60);
        uint32_t blockCount = actS1Size / 2;
        uint32_t blockLen = dVBaseSize * 2 * gSize;
        uint32_t srcStride = 0;
        uint64_t dstStride = n2Size * gSize * dVBaseSize * 2;
        uint32_t srcStride310 = DivCeil(blockLen, 32) * 32;
        copy_ubuf_to_gm_align_v2((__gm__ void *)(attentionOutGm + gmOffset),
                                 (__ubuf__ void *)(attenOutUb + ubOffset), 0, blockCount, blockLen,
                                 cacheMode, dstStride, srcStride310);
    }

    template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
    __simd_vf__ inline void CastIntrinsicsImplVF_MX(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src,
                                                     const uint32_t calCount)
    {
        uint16_t oneRepSize = GetVecLen() / sizeof(SRC_TYPE);
        if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
            oneRepSize = GetVecLen() / sizeof(DST_TYPE);
        }
        uint16_t repeatTime = (calCount + oneRepSize - 1) / oneRepSize;
        uint32_t sreg = static_cast<uint32_t>(calCount);
        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, roundMode};
        Reg::MaskReg preg;
        Reg::RegTensor<SRC_TYPE> srcVreg;
        Reg::RegTensor<DST_TYPE> dstVreg;
        for (uint16_t i = 0; i < repeatTime; ++i) {
            preg = plt_b32(sreg, POST_UPDATE);
            constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
            vlds(srcVreg, src + i * oneRepSize, 0, distValue);
            constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(Reg::RegLayout::ZERO)>();
            constexpr auto modeValue = std::integral_constant<::Mode, static_cast<::Mode>(Reg::MaskMergeMode::ZEROING)>();
            constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
            constexpr auto satModeValue = std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(Reg::SatMode::SAT)>();
            vcvt(dstVreg, srcVreg, preg, roundModeValue, satModeValue, partModeValue, modeValue);
            constexpr auto distValue2 = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<DST_TYPE, Reg::StoreDist::DIST_PACK_B32>())>();
            vsts(dstVreg, dst + i * oneRepSize, 0, distValue2, preg);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(__ubuf__ float *vec2ResUb, uint32_t mStartVec,
                                                  uint32_t bIdx, uint32_t n2Idx, uint32_t gS1Idx,
                                                  uint32_t vecMbaseIdx, uint32_t mDealSize,
                                                  uint32_t gmDealRowCount, uint32_t mloop)
    {
        __ubuf__ OUTPUT_T *attenOut = (__ubuf__ OUTPUT_T *)vec2ResUb;
        int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;
        RowInvalid(vec2ResUb, mStartVec, gS1Idx, actVecMSize, vecMbaseIdx, mDealSize, mloop, dSizeAligned64);
        using DstPrimType = PrimT<OUTPUT_T>;
        using SrcPrimType = PrimT<float>;
        using DST_TYPE = typename CastParam::CastTypeTrait<DstPrimType>::RealType;
        using SRC_TYPE = typename CastParam::CastTypeTrait<SrcPrimType>::RealType;
        __ubuf__ DST_TYPE *dst = reinterpret_cast<__ubuf__ DST_TYPE *>(attenOut);
        __ubuf__ SRC_TYPE *src = reinterpret_cast<__ubuf__ SRC_TYPE *>(vec2ResUb);
        CastIntrinsicsImplVF_MX<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND>(dst, src, mDealSize * dSizeAligned64);

        set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
        Bmm2DataCopyOutTrans(attenOut, mStartVec, bIdx, n2Idx, gS1Idx, actVecMSize, vecMbaseIdx,
                             mDealSize, gmDealRowCount);
    }

    template <typename T1, typename RegType>
    __simd_vf__ inline void DuplicateFromScalarImpl(__ubuf__ T1 *dstLocal, const T1 scalarValue,
                                                     const int32_t calCount)
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
};

} // namespace Block
} // namespace Epilogue
} // namespace Blaze
