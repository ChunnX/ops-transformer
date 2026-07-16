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
 * \file flash_attn_block_vec_mixedquant_gqa.h
 * \brief MixedQuantFlashAttn Vec Block定义
 */

#ifndef FLASH_ATTN_BLOCK_VEC_MIXEDQUANT_GQA_H_
#define FLASH_ATTN_BLOCK_VEC_MIXEDQUANT_GQA_H_

#include "../utils/mixed_quant_flash_attn_common_def.h"
#include "../utils/mixed_quant_flash_attn_utils.h"
#include "../../../common/op_kernel/memcopy/parser.h"
#include "../../../common/op_kernel/arch35/attenmask_gs1.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast.h"
#include "../../../common/op_kernel/arch35/vf/vf_antiquant_w4.h"
#include "../../../common/op_kernel/arch35/vf/vf_antiquant_w8.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/fia_public_define.h"
#include "../../../common/op_kernel/memcopy/copy_ub_to_gm.h"

template <inferFaLayOutTypeEnum LAYOUT>
__aicore__ inline constexpr UbFormat GetOutUbFormatMQ();

using namespace FaVectorApi;

namespace BaseApi {

template <typename QINPUT_T, typename ACC_T, typename OUT_T, typename KV_TYPE, inferFaLayOutTypeEnum inputLayoutType,
           inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
           inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class MixedQuantGqaVecBlockBase {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using KV_T = KV_TYPE;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;

    static constexpr bool HAS_MASK = hasAttenMask;

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);

    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;

    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;

    __aicore__ inline MixedQuantGqaVecBlockBase(ConstInfoX &constInfo) : constInfo(constInfo){};

    const ConstInfoX &constInfo;
};

template <typename QINPUT_T, typename ACC_T, typename OUT_T, typename KV_TYPE, inferFaLayOutTypeEnum inputLayoutType,
           inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
           inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class FAMixedQuantGqaBlockVec
    : public MixedQuantGqaVecBlockBase<QINPUT_T, ACC_T, OUT_T, KV_TYPE, inputLayoutType, outputLayoutType,
                                        s1TemplateType, s2TemplateType, dTemplateType, hasAttenMask, KvLayoutType> {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using KV_T = KV_TYPE;
    using T = ACC_T;
    using OUTPUT_T = OUT_T;

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;
    static constexpr uint32_t dAlign64 = MqfaAlignUp64((uint16_t)dBaseSize);
    static constexpr bool HAS_MASK = hasAttenMask;

    static constexpr bool IS_TND_LAYOUT = (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND);
    static constexpr ActualSeqLensMode Q_MODE = IS_TND_LAYOUT ? ActualSeqLensMode::ACCUM : ActualSeqLensMode::BY_BATCH;
    using QSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                     ActualSeqLensParser<Q_MODE, int32_t>>::type;

    static constexpr uint32_t initOutputEventId = 0U;

    TPipe *tPipe;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<float> softmaxFDSumGm;
    GlobalTensor<float> softmaxFDMaxGm;
    GlobalTensor<int32_t> actualSeqLengthsGmQ;

    QSeqParserType *qActSeqLensParser = nullptr;

<<<<<<< HEAD
=======
    TEventID mte3ToVId[2];
    TEventID vToMte3Id[2];

>>>>>>> 69c719436 (添加C1代码&&修复一些问题)
    bool isFd = false;

    const ConstInfoX &constInfo;
    T negativeFloatScalar;

    using attenMaskGmType = typename std::conditional<hasAttenMask, GlobalTensor<uint8_t>, int8_t>::type;
    attenMaskGmType attenMaskGmInt;
    GlobalTensor<QINPUT_T> sinkGm;

    GlobalTensor<KV_T> rawKeyGm;
    GlobalTensor<KV_T> rawValueGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<QINPUT_T> kDeScaleGm;
    GlobalTensor<QINPUT_T> vDeScaleGm;

    TBuf<TPosition::VECIN> kvInBuf[MQFA_KV_IN_BUF_NUM];
    TBuf<TPosition::VECOUT> kvOutBuf[MQFA_KV_OUT_BUF_NUM];
    TBuf<TPosition::VECIN> scaleInBuf;
    TBuf<> scaleResBuf;

    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t dTemplateAlign64 = ((dBaseSize + 63) >> 6) << 6;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr MaskFormat MASK_LAYOUT = IS_TND_LAYOUT ? MaskFormat::SG : MaskFormat::GS;
    static constexpr uint32_t blockBytes = 32;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;

    TBuf<> commonTBuf;
    TBuf<> vselrIndexesBuf[4];
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TBuf<> stage2OutBuf;
    TBuf<> softmaxMaxBuf[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf[PRELOAD_N + 1];
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECIN, 1> sinkQue;

    __aicore__ inline FAMixedQuantGqaBlockVec(ConstInfoX &constInfo)
        : MixedQuantGqaVecBlockBase<QINPUT_T, ACC_T, OUT_T, KV_TYPE, inputLayoutType, outputLayoutType, s1TemplateType,
                                    s2TemplateType, dTemplateType, hasAttenMask, KvLayoutType>(constInfo),
          constInfo(constInfo){};

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType &qParser)
    {
        this->qActSeqLensParser = &qParser;
    }

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        tPipe = pipe;
        uint32_t tmp1 = MQFA_NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, attenMask, learnableSink, softmaxLse, attentionOut,
                     workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        uint64_t actualLenQSize =
            (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) ? constInfo.cuSeqLensQSize : constInfo.seqUsedQSize;
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, actualLenQSize);

        if constexpr (hasAttenMask) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if (learnableSink != nullptr) {
            sinkGm.SetGlobalBuffer((__gm__ QINPUT_T *)learnableSink);
        }

        if (this->isFd) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (MQFA_CV_RATIO * constInfo.coreNum) - 1) / (MQFA_CV_RATIO * constInfo.coreNum);
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            tSize = qActSeqLensParser->GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize;
        int64_t singleCoreSize =
            (totalOutputSize + (MQFA_CV_RATIO * constInfo.coreNum) - 1) / (MQFA_CV_RATIO * constInfo.coreNum);
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 3e+99);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[], TBuf<> &SharedBuffer3)
    {
        InitAntiquantBuffers(SharedBuffer1, SharedBuffer2, SharedBuffer3);
    }

    __aicore__ inline void InitAntiquantBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[], TBuf<> &SharedBuffer3)
    {
        constexpr uint32_t grpNum = dBaseSize / MQFA_GRP_SIZE;
        // FP4为4bit打包类型，每个字节存储2个FP4元素，因此kvInBufSize按字节计算时需除以2
        constexpr uint32_t kvInBufSize = dAlign64 * s2BaseSize / 2;
        constexpr uint32_t kvOutBufSize = dAlign64 * s2BaseSize * sizeof(QINPUT_T);
        // Scale类型为FP8E8M0，每个元素占1字节
        constexpr uint32_t scaleInBufSize = s2BaseSize * grpNum;
        constexpr uint32_t scaleResBufSize = s2BaseSize * grpNum * sizeof(bfloat16_t);

        for (uint32_t i = 0; i < MQFA_KV_IN_BUF_NUM; i++) {
            tPipe->InitBuffer(kvInBuf[i], kvInBufSize);
        }
        for (uint32_t i = 0; i < MQFA_KV_OUT_BUF_NUM; i++) {
            tPipe->InitBuffer(kvOutBuf[i], kvOutBufSize);
        }
        tPipe->InitBuffer(scaleInBuf, scaleInBufSize);
        tPipe->InitBuffer(scaleResBuf, scaleResBufSize);

        tPipe->InitBuffer(SharedBuffer2[0], 1, 8192);
        tPipe->InitBuffer(SharedBuffer2[1], 1, 8192);
        tPipe->InitBuffer(SharedBuffer3, 32768);
        tPipe->InitBuffer(SharedBuffer1[0], 1, 33024);
        tPipe->InitBuffer(SharedBuffer1[1], 1, 33024);

        tPipe->InitBuffer(softmaxSumBuf[0], 256);
        tPipe->InitBuffer(softmaxSumBuf[1], 256);
        tPipe->InitBuffer(softmaxSumBuf[2], 256);
        tPipe->InitBuffer(softmaxMaxBuf[0], 256);
        tPipe->InitBuffer(softmaxMaxBuf[1], 256);
        tPipe->InitBuffer(softmaxMaxBuf[2], 256);
        tPipe->InitBuffer(softmaxExpBuf[0], 256);
        tPipe->InitBuffer(softmaxExpBuf[1], 256);
        tPipe->InitBuffer(softmaxExpBuf[2], 256);
        tPipe->InitBuffer(maxBrdcst, 1, 2048);
        tPipe->InitBuffer(sumBrdcst, 1, 2048);
        tPipe->InitBuffer(commonTBuf, 512);

        if (constInfo.isSoftmaxLseEnable) {
            tPipe->InitBuffer(softmaxLseQueue, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }

        stage1OutQue[0] = SharedBuffer1[0];
        stage1OutQue[1] = SharedBuffer1[1];
        attenMaskInQue[0] = SharedBuffer2[0];
        attenMaskInQue[1] = SharedBuffer2[1];
        stage2OutBuf = SharedBuffer3;
    }

    __aicore__ inline void InitAntiquantGlobalTensor(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                                      __gm__ uint8_t *blockTable,
                                                      __gm__ uint8_t *kDescale, __gm__ uint8_t *vDescale)
    {
        rawKeyGm.SetGlobalBuffer((__gm__ KV_T *)key);
        rawValueGm.SetGlobalBuffer((__gm__ KV_T *)value);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        kDeScaleGm.SetGlobalBuffer((__gm__ QINPUT_T *)kDescale);
        vDeScaleGm.SetGlobalBuffer((__gm__ QINPUT_T *)vDescale);
    }

    // ===================== Antiquant data-copy helpers =====================

    __aicore__ inline void CopyScalePageAttention(LocalTensor<QINPUT_T> dstLocal,
                                                   GlobalTensor<QINPUT_T>& srcGm,
                                                   RunInfoX &runInfo, uint32_t dealRowCount,
                                                   uint32_t grpNum)
    {
        uint64_t blockTableBaseOffset = runInfo.bIdx * constInfo.maxBlockNumPerBatch;
        uint32_t copyFinishRowCnt = 0;
        uint32_t curSequence = runInfo.s2Idx + constInfo.subBlockIdx *
            MqfaGetRealDealSize(static_cast<uint16_t>(runInfo.actSingleLoopS2Size));
        while (copyFinishRowCnt < dealRowCount) {
            uint64_t blockIdOffset = curSequence / constInfo.blockSize;
            uint64_t remainRowCnt = curSequence % constInfo.blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyRowCnt = constInfo.blockSize - remainRowCnt;
            if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
                copyRowCnt = dealRowCount - copyFinishRowCnt;
            }
            // Scale layout is paged like KV: (Block, BlockSize, N, D/grpNum)
            // Scale类型为FP8E8M0，每个元素占1字节，但GlobalTensor按QINPUT_T声明以匹配DataCopyPad类型要求
            uint64_t curOffset = 0;
            if (constInfo.paLayoutType == static_cast<uint32_t>(MqfaPaLayoutType::BBND)) {
                // BBND: (Block, BlockSize, N, grpNum) — heads interleaved within each block
                curOffset = (idInBlockTable * constInfo.blockSize + remainRowCnt) *
                    constInfo.n2Size * grpNum + (uint64_t)(runInfo.n2Idx * grpNum);
            } else {
                // BNBD: (Block, N, BlockSize, grpNum) — heads separated
                curOffset = idInBlockTable * constInfo.n2Size * constInfo.blockSize * grpNum +
                    (uint64_t)(runInfo.n2Idx * constInfo.blockSize * grpNum) +
                    remainRowCnt * grpNum;
            }
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<QINPUT_T> copyInPadParams{};
            copyInParams.blockCount = copyRowCnt;
            copyInParams.blockLen = grpNum * sizeof(int8_t);
            copyInParams.dstStride = 0;
            copyInParams.srcStride = constInfo.n2Size * grpNum - grpNum;
            if (constInfo.paLayoutType == static_cast<uint32_t>(MqfaPaLayoutType::BNBD)) {
                copyInParams.srcStride = 0;
            }
            copyInPadParams.isPad = false;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = 0;
            copyInPadParams.paddingValue = 0;
            DataCopyPad(dstLocal[(copyFinishRowCnt * grpNum) >> 1], srcGm[curOffset >> 1],
                        copyInParams, copyInPadParams);
            copyFinishRowCnt += copyRowCnt;
            curSequence += copyRowCnt;
        }
    }

    __aicore__ inline void CopyKVPageAttention(LocalTensor<KV_T> dstLocal, GlobalTensor<KV_T>& srcGm,
                                                RunInfoX &runInfo, uint32_t dealRowCount)
    {
        // FP4为4bit打包类型，sizeof(KV_T)=1字节(含2个FP4元素)，typeElementSize为32B块能容纳的FP4元素数
        uint32_t typeElementSize = ONE_BLK_SIZE * 2;
        uint64_t blockTableBaseOffset = runInfo.bIdx * constInfo.maxBlockNumPerBatch;
        uint32_t copyFinishRowCnt = 0;
        uint32_t curSequence = runInfo.s2Idx + constInfo.subBlockIdx *
            MqfaGetRealDealSize(static_cast<uint16_t>(runInfo.actSingleLoopS2Size));
        while (copyFinishRowCnt < dealRowCount) {
            uint64_t blockIdOffset = curSequence / constInfo.blockSize;
            uint64_t remainRowCnt = curSequence % constInfo.blockSize;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyRowCnt = constInfo.blockSize - remainRowCnt;
            if (copyFinishRowCnt + copyRowCnt > dealRowCount) {
                copyRowCnt = dealRowCount - copyFinishRowCnt;
            }
            uint64_t curOffset = 0;
            if (constInfo.paLayoutType == static_cast<uint32_t>(MqfaPaLayoutType::BBND)) {
                // BBND: (Block, BlockSize, N, D) — heads interleaved within each block
                curOffset = (idInBlockTable * constInfo.blockSize + remainRowCnt) *
                    constInfo.n2Size * constInfo.dSize + (uint64_t)(runInfo.n2Idx * constInfo.dSize);
            } else {
                // BNBD: (Block, N, BlockSize, D) — heads separated by block
                curOffset = idInBlockTable * constInfo.n2Size * constInfo.blockSize * constInfo.dSize +
                    (uint64_t)(runInfo.n2Idx * constInfo.blockSize * constInfo.dSize) +
                    remainRowCnt * constInfo.dSize;
            }
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<KV_T> copyInPadParams{};
            copyInParams.blockCount = copyRowCnt;
            // FP4为4bit打包，sizeof(KV_T)=1字节含2个元素，dSize个元素占dSize/2字节
            copyInParams.blockLen = constInfo.dSize * sizeof(KV_T) / 2;
            copyInParams.dstStride = (dBaseSize - constInfo.dSize) / typeElementSize;
            // BBND layout中相邻head在GM中连续排布，需跳过其他head的数据；BNBD layout中每个block只含一个head，srcStride=0
            if (constInfo.paLayoutType == static_cast<uint32_t>(MqfaPaLayoutType::BBND)) {
                copyInParams.srcStride = copyInParams.blockLen * (constInfo.n2Size - 1);
            } else {
                copyInParams.srcStride = 0;
            }
            copyInPadParams.isPad = false;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = 0;
            copyInPadParams.paddingValue = 0;
            DataCopyPad(dstLocal[copyFinishRowCnt * dBaseSize], srcGm[curOffset],
                        copyInParams, copyInPadParams);
            copyFinishRowCnt += copyRowCnt;
            curSequence += copyRowCnt;
        }
    }

    __aicore__ inline void CopyAntiquantResToL1(LocalTensor<QINPUT_T>& antiqResL1,
                                                  LocalTensor<QINPUT_T>& antiqResUb,
                                                  uint32_t dealRowCount, uint32_t s2RealSize,
                                                  uint32_t subBlockIdx)
    {
        uint16_t elementTypeSize = ONE_BLK_SIZE / sizeof(QINPUT_T);
        uint16_t dstStep = MqfaAlignUp16(s2RealSize);
        uint32_t headDimAlignBlock = dAlign64 / elementTypeSize;
        uint64_t outOffset = subBlockIdx * MqfaGetRealDealSize(static_cast<uint16_t>(s2RealSize)) * 16;

        // DataCopy以32B块为粒度搬运，DataCopyParams各字段单位均为32B块：
        // blockCount = D方向块数(dAlign64 / 16)，每块对应16个D元素
        // blockLen = dealRowCount，即当前V核处理的S方向行数，每行16个元素占1个32B块
        // srcStride = 1：VF的StoreAlign(blockStride=dealRowCount+1)在UB中每个D块组之间留了1个32B空隙
        // dstStride = dstStep - dealRowCount：
        //   两个V核(subBlockIdx=0/1)共写同一块L1区域，dstStep=AlignUp16(s2RealSize)为整个S2块在L1中对齐后的步长，
        //   减去当前V核写入的dealRowCount行，剩余空间留给另一个V核的数据及16对齐padding
        // UB和L1中的数据均按NZ格式排列，NZ格式由VF的StoreAlign(DATA_BLOCK_COPY)产生，非硬件强制
        struct DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = headDimAlignBlock;
        dataCopyParams.blockLen = dealRowCount;
        dataCopyParams.srcStride = 1;
        dataCopyParams.dstStride = dstStep - dealRowCount;

        DataCopy(antiqResL1[outOffset], antiqResUb, dataCopyParams);
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_K[0]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_K[1]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_K[0]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_K[1]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_V[0]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_V[1]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_V[0]);
        SetFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_V[1]);
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_K[0]);
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_K[1]);
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_V[0]);
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_V[1]);

<<<<<<< HEAD
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_VEC2[0]);
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_VEC2[1]);
        SetFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[0]);
        SetFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[1]);       
=======
        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
>>>>>>> 69c719436 (添加C1代码&&修复一些问题)
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_K[0]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_K[1]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_K[0]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_K[1]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_V[0]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_SCALE_V[1]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_V[0]);
        WaitFlag<HardEvent::V_MTE2>(MQFA_V_MTE2_KV_V[1]);
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_K[0]);
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_K[1]);
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_V[0]);
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_V[1]);

<<<<<<< HEAD
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_VEC2[0]);
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_VEC2[1]);
        WaitFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[0]);
        WaitFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[1]);  
=======
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
>>>>>>> 69c719436 (添加C1代码&&修复一些问题)
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t bN2Cur)
    {
        uint32_t n2Idx = bN2Cur % constInfo.n2Size;
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        if constexpr (outputLayoutType == inferFaLayOutTypeEnum::LAYOUT_BSND) {
            OffsetCalculator<GmFormat::BSNGD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                  constInfo.dSizeV, *qActSeqLensParser);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outputLayoutType == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
            OffsetCalculator<GmFormat::BNGSD, int32_t> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                  constInfo.dSizeV, *qActSeqLensParser);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if constexpr (outputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            OffsetCalculator<GmFormat::TNGD, int32_t, true> offsetCalculator;
            offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, *qActSeqLensParser);
            DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        }
    }

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>;

    __aicore__ inline void ProcessVec1(MM2_ABUF_T &vec1ResBuf, MM1_DBUF_T &bmm1ResBuf, RunInfoX &runInfo)
    {
#ifdef SKIP_V1
        bmm1ResBuf.WaitCrossCore();
        bmm1ResBuf.SetCrossCore();
        vec1ResBuf.SetCrossCore();
        return;
#endif
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            vec1ResBuf.SetCrossCore();
            return;
        }
        ProcessVec1Nd(vec1ResBuf, bmm1ResBuf, runInfo);
    }

    __aicore__ inline void ProcessVec1Nd(MM2_ABUF_T &vec1ResBuf, MM1_DBUF_T &bmm1ResBuf, RunInfoX runInfo)
    {
        LocalTensor<uint8_t> dropMaskUb;
        LocalTensor<QINPUT_T> nonePseUb;
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (hasAttenMask) {
            attenMaskUb = attenMaskInQue[runInfo.loop & (DB - 1)].template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo);
        }

        LocalTensor<float> sumUb = softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();

        int64_t stage1Offset = runInfo.loop & (DB - 1);
        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;

        LocalTensor<T> mm1Res = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = stage1OutQue[stage1Offset].template AllocTensor<QINPUT_T>();

        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, QINPUT_T, QINPUT_T, false, mBaseSize, s2BaseSize, EQ_128, hasAttenMask,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F,
                    queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, QINPUT_T, QINPUT_T, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAttenMask,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F,
                    queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, QINPUT_T, QINPUT_T, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAttenMask,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F,
                    queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    ProcessVec1Vf<T, QINPUT_T, QINPUT_T, false, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  hasAttenMask, PseTypeEnum::PSE_NONE_TYPE, false>(
                        stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, QINPUT_T, QINPUT_T, true, mBaseSize, s2BaseSize, EQ_128, hasAttenMask,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F,
                    queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, QINPUT_T, QINPUT_T, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAttenMask,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F,
                    queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, QINPUT_T, QINPUT_T, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAttenMask,
                              PseTypeEnum::PSE_NONE_TYPE, false, false, false>(
                    stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                    nonePseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F,
                    queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    ProcessVec1Vf<T, QINPUT_T, QINPUT_T, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256,
                                  hasAttenMask, PseTypeEnum::PSE_NONE_TYPE, false>(
                        stage1CastTensor, vselrIndexesBuf, sumUb, maxUb, mm1Res, expUb, sumUb, maxUb, attenMaskUb,
                        nonePseUb, dropMaskUb, apiTmpBuffer, expUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                        0, 0.0f, 0.0f, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();
        if constexpr (hasAttenMask) {
            attenMaskInQue[runInfo.loop & (DB - 1)].template FreeTensor(attenMaskUb);
        }

        stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        stage1OutQue[stage1Offset].template DeQue<QINPUT_T>();
        LocalTensor<QINPUT_T> vec1L1ResTensor = vec1ResBuf.GetTensor<QINPUT_T>();
        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(vec1L1ResTensor[constInfo.subBlockIdx * (blockBytes / sizeof(QINPUT_T)) *
                                     (runInfo.actMSize - runInfo.actVecMSize)],
                     stage1CastTensor,
                     {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize,
                      (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
        vec1ResBuf.SetCrossCore();

        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2(MM1_DBUF_T &bmm2ResBuf, RunInfoX &runInfo)
    {
#ifdef SKIP_V2
        bmm2ResBuf.WaitCrossCore();
        bmm2ResBuf.SetCrossCore();
        return;
#endif
        bmm2ResBuf.WaitCrossCore();
        ProcessVec2OnUb(bmm2ResBuf, runInfo);
    }

    __aicore__ inline void ProcessVec2OnUb(MM1_DBUF_T &bmm2ResBuf, RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        LocalTensor<T> vec2ResUb = stage2OutBuf.template Get<T>();
        LocalTensor<T> mm2Res = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_VEC2[0]);

        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mm2Res, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;
            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, QINPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mm2Res, vec2ResUb, expUb, pScaleUb,
                    runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, QINPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mm2Res, vec2ResUb, expUb, pScaleUb, sumUb,
                    runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            }
        }
        bmm2ResBuf.SetCrossCore();

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, QINPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(MQFA_MTE3_V_VEC2[0]);
    }

    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb,
                                               uint32_t mStartVec, uint32_t mDealSize)
    {
        if (isFd) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb,
                                                  uint32_t mStartVec, uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        attenOut.SetAddr(vec2ResUb.address_);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[0]);
        WaitFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[0]);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount,
                                      .colCount = (uint32_t)dTemplateAlign64};
        GmCoord gmCoord{.bIdx = info.bIdx, .n2Idx = info.n2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx, .dIdx = 0,
                        .gS1DealSize = dealRowCount, .dDealSize = (uint32_t)constInfo.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if constexpr (outputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV,
                                              *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormatMQ<inputLayoutType>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outputLayoutType == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormatMQ<inputLayoutType>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, *qActSeqLensParser);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormatMQ<inputLayoutType>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb,
                                                uint32_t mStartVec, uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        SetFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[runInfo.loop & (DB - 1)]);
        WaitFlag<HardEvent::V_MTE3>(MQFA_V_MTE3_VEC2[runInfo.loop & (DB - 1)]);
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (blockBytes / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (isFd) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
            if (!runInfo.isS2SplitCore && constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = softmaxLseQueue.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        softmaxLseQueue.template EnQue(lseUb);
        softmaxLseQueue.DeQue<float>();

        if constexpr (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser->GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo.gSize * constInfo.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                           runInfo.actVecMSize, constInfo);
        } else if constexpr (inputLayoutType == inferFaLayOutTypeEnum::LAYOUT_BNSD) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                        runInfo.actVecMSize, constInfo,
                                                        qActSeqLensParser->GetActualSeqLength(runInfo.bIdx));
        } else {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                        runInfo.actVecMSize, constInfo);
        }
        softmaxLseQueue.FreeTensor(lseUb);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * 8;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * 8 + runInfo.vecMbaseIdx * 8;
        LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst.template EnQue(sumOutTensor);
        sumBrdcst.template DeQue<float>();
        DataCopy(softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst.template FreeTensor(sumOutTensor);

        LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeFloatScalar;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL;

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);

        if (IsSkipMask && IsSkipMaskForPre) {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            return;
        }
        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
        }
        if (!IsSkipMaskForPre) {
            LocalTensor<uint8_t> attenMaskUbPre =
                attenMaskInQue[1 - (runInfo.loop & (DB - 1))].template AllocTensor<uint8_t>();
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt, maskInfo, true);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            attenMaskInQue[1 - (runInfo.loop & (DB - 1))].template FreeTensor(attenMaskUbPre);
        }
    }

    // ===================== Antiquant sub-functions =====================

    __aicore__ inline void LoadAndConvertScale(LocalTensor<QINPUT_T>& scaleE8M0Ub,
                                                LocalTensor<QINPUT_T>& scaleBf16Ub,
                                                GlobalTensor<QINPUT_T>& deScaleGm,
                                                RunInfoX &runInfo, uint32_t copyTotalS,
                                                uint32_t grpNum, uint32_t idx,
                                                const uint32_t* mte2VScaleId,
                                                const uint32_t* vMte2ScaleId)
    {
        WaitFlag<HardEvent::V_MTE2>(vMte2ScaleId[idx]);
        CopyScalePageAttention(scaleE8M0Ub, deScaleGm, runInfo, copyTotalS, grpNum);
        SetFlag<HardEvent::MTE2_V>(mte2VScaleId[idx]);

        WaitFlag<HardEvent::MTE2_V>(mte2VScaleId[idx]);
        FaVectorApi::AntiqScaleByVF<QINPUT_T, QINPUT_T>(scaleE8M0Ub, scaleBf16Ub, copyTotalS, grpNum);
        SetFlag<HardEvent::V_MTE2>(vMte2ScaleId[idx]);
    }

    __aicore__ inline void CopyKvToUb(LocalTensor<KV_T>& kvInUb, GlobalTensor<KV_T>& kvGm,
                                       RunInfoX &runInfo, uint32_t copyTotalS, uint32_t idx,
                                       const uint32_t* vMte2KvId,
                                       const uint32_t* mte2VKvId)
    {
        WaitFlag<HardEvent::V_MTE2>(vMte2KvId[idx]);
        CopyKVPageAttention(kvInUb, kvGm, runInfo, copyTotalS);
        SetFlag<HardEvent::MTE2_V>(mte2VKvId[idx]);
    }

    __aicore__ inline void DoAntiquant(LocalTensor<QINPUT_T>& kvOutUb,
                                        LocalTensor<KV_T>& kvInUb,
                                        LocalTensor<QINPUT_T>& scaleBf16Ub,
                                        uint32_t copyTotalS, uint32_t dSize, uint32_t idx,
                                        const uint32_t* mte3VId,
                                        const uint32_t* mte2VKvId,
                                        const uint32_t* vMte2KvId,
                                        const uint32_t* vMte3Id)
    {
        WaitFlag<HardEvent::MTE3_V>(mte3VId[idx]);
        WaitFlag<HardEvent::MTE2_V>(mte2VKvId[idx]);
        LocalTensor<QINPUT_T> dummyOffset;
        // 直接调用vf_antiquant_w4.h中的AntiquantVFImpl，该函数针对fp4x2_e2m1_t有特化重载
        FaVectorApi::AntiquantVFImpl<QINPUT_T, QINPUT_T, dBaseSize, false, false, false>(
            kvInUb, kvOutUb, dummyOffset, scaleBf16Ub, copyTotalS, dSize, copyTotalS);
        SetFlag<HardEvent::V_MTE2>(vMte2KvId[idx]);
        SetFlag<HardEvent::V_MTE3>(vMte3Id[idx]);
    }

    __aicore__ inline void CopyResultToL1(LocalTensor<QINPUT_T>& antiqResL1,
                                           LocalTensor<QINPUT_T>& antiqResUb,
                                           uint32_t copyTotalS, uint32_t s2RealSize,
                                           uint32_t subBlockIdx, uint32_t idx,
                                           const uint32_t* vMte3Id,
                                           const uint32_t* mte3VId)
    {
        CrossCoreWaitFlag<MQFA_SYNC_MODE, PIPE_MTE3>(MQFA_CV_L1_EVENT[idx]);
        WaitFlag<HardEvent::V_MTE3>(vMte3Id[idx]);
        CopyAntiquantResToL1(antiqResL1, antiqResUb, copyTotalS, s2RealSize, subBlockIdx);
        SetFlag<HardEvent::MTE3_V>(mte3VId[idx]);
        CrossCoreSetFlag<MQFA_SYNC_MODE, PIPE_MTE3>(MQFA_VC_L1_EVENT[idx]);
    }

    // ===================== AntiquantKey / AntiquantValue =====================

    __aicore__ inline void AntiquantKey(RunInfoX &runInfo, LocalTensor<QINPUT_T> antiqResL1)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        uint32_t dSize = constInfo.dSize;
        uint32_t grpNum = dSize / MQFA_GRP_SIZE;
        uint32_t subBlockIdx = constInfo.subBlockIdx;
        uint32_t copyTotalS = MqfaGetRealDealSize(static_cast<uint16_t>(s2RealSize));
        if (subBlockIdx == 1) {
            copyTotalS = s2RealSize - copyTotalS;
        }
        uint32_t taskId = runInfo.KvL1loop;
        uint32_t idx = taskId & 3;

        LocalTensor<QINPUT_T> scaleE8M0Ub = scaleInBuf.Get<QINPUT_T>();
        LocalTensor<QINPUT_T> scaleBf16Ub = scaleResBuf.Get<QINPUT_T>();
        LocalTensor<KV_T> kvInUb = kvInBuf[taskId % MQFA_KV_IN_BUF_NUM].Get<KV_T>();
        LocalTensor<QINPUT_T> kvOutUb = kvOutBuf[idx].Get<QINPUT_T>();

        LoadAndConvertScale(scaleE8M0Ub, scaleBf16Ub, kDeScaleGm, runInfo, copyTotalS, grpNum,
                            idx, MQFA_MTE2_V_SCALE_K, MQFA_V_MTE2_SCALE_K);
        CopyKvToUb(kvInUb, rawKeyGm, runInfo, copyTotalS, idx, MQFA_V_MTE2_KV_K, MQFA_MTE2_V_KV_K);
        DoAntiquant(kvOutUb, kvInUb, scaleBf16Ub, copyTotalS, dSize, idx,
                    MQFA_MTE3_V_K, MQFA_MTE2_V_KV_K, MQFA_V_MTE2_KV_K, MQFA_V_MTE3_K);
        CopyResultToL1(antiqResL1, kvOutUb, copyTotalS, s2RealSize, subBlockIdx, idx,
                       MQFA_V_MTE3_K, MQFA_MTE3_V_K);
    }

    __aicore__ inline void AntiquantValue(RunInfoX &runInfo, LocalTensor<QINPUT_T> antiqResL1)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        uint32_t dSize = constInfo.dSize;
        uint32_t grpNum = dSize / MQFA_GRP_SIZE;
        uint32_t subBlockIdx = constInfo.subBlockIdx;
        uint32_t copyTotalS = MqfaGetRealDealSize(static_cast<uint16_t>(s2RealSize));
        if (subBlockIdx == 1) {
            copyTotalS = s2RealSize - copyTotalS;
        }
        uint32_t taskId = runInfo.KvL1loop;
        uint32_t idx = taskId & 3;

        LocalTensor<QINPUT_T> scaleE8M0Ub = scaleInBuf.Get<QINPUT_T>();
        LocalTensor<QINPUT_T> scaleBf16Ub = scaleResBuf.Get<QINPUT_T>();
        LocalTensor<KV_T> kvInUb = kvInBuf[taskId % MQFA_KV_IN_BUF_NUM].Get<KV_T>();
        LocalTensor<QINPUT_T> kvOutUb = kvOutBuf[idx].Get<QINPUT_T>();

        LoadAndConvertScale(scaleE8M0Ub, scaleBf16Ub, vDeScaleGm, runInfo, copyTotalS, grpNum,
                            idx, MQFA_MTE2_V_SCALE_V, MQFA_V_MTE2_SCALE_V);
        CopyKvToUb(kvInUb, rawValueGm, runInfo, copyTotalS, idx, MQFA_V_MTE2_KV_V, MQFA_MTE2_V_KV_V);
        DoAntiquant(kvOutUb, kvInUb, scaleBf16Ub, copyTotalS, dSize, idx,
                    MQFA_MTE3_V_V, MQFA_MTE2_V_KV_V, MQFA_V_MTE2_KV_V, MQFA_V_MTE3_V);
        CopyResultToL1(antiqResL1, kvOutUb, copyTotalS, s2RealSize, subBlockIdx, idx,
                       MQFA_V_MTE3_V, MQFA_MTE3_V_V);
    }
};

} // namespace BaseApi

#endif // FLASH_ATTN_BLOCK_VEC_MIXEDQUANT_GQA_H_
