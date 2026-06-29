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
 * \file flash_attention_mixedquant_block_vec_flashdecode.h
 * \brief MixedQuantFlashAttn FlashDecode Vec Block定义
 */

#ifndef FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_
#define FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_

#include "../utils/mixed_quant_flash_attn_common_def.h"

namespace BaseApi {

template <typename INPUT_T, typename ACC_T, typename OUT_T, inferFaLayOutTypeEnum inputLayoutType,
          inferFaLayOutTypeEnum outputLayoutType, inferS1TemplateType s1TemplateType,
          inferS2TemplateType s2TemplateType, inferDTemplateType dTemplateType, bool hasAttenMask, uint8_t KvLayoutType>
class FAMixedQuantGqaBlockVecFlashDecode {
public:
    using ConstInfoX = MixedQuantConstInfoX;
    using T = ACC_T;
    using OUTPUT_T = OUT_T;
    using SINK_T = INPUT_T;

    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = dBaseSize;

    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

protected:
    GlobalTensor<float> lseSumFdGm;
    GlobalTensor<float> lseMaxFdGm;
    GlobalTensor<float> accumOutGm;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<SINK_T> sinkGm;

    __gm__ uint8_t *keyPtr = nullptr;
    bool learnableSinkFlag = false;
    uint32_t dSizeV_Align;
    uint32_t blockIdx = 0U;

    const ConstInfoX &constInfo;

private:
    LocalTensor<T> fdSumBuf1;
    LocalTensor<T> fdSumBuf2;
    LocalTensor<T> fdMaxBuf1;
    LocalTensor<T> fdMaxBuf2;
    LocalTensor<T> fdLseExpBuf;
    LocalTensor<T> fdMm2ResBuf1;
    LocalTensor<T> fdMm2ResBuf2;
    LocalTensor<T> fdReduceBuf;
    LocalTensor<OUTPUT_T> fdOutputBuf;
    LocalTensor<T> fdSinkCopyInBuf;
    LocalTensor<T> fdSinkValueBuf;
    LocalTensor<T> fdSinkExpBuf;
    LocalTensor<T> fdSinkTmpBuf;

    LocalTensor<T> fdLseMaxUbBuf1;
    LocalTensor<T> fdLseMaxUbBuf2;
    LocalTensor<T> fdLseUbBuf;

public:
    __aicore__ inline FAMixedQuantGqaBlockVecFlashDecode(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitParams()
    {
        this->dSizeV_Align = this->Align(constInfo.dSizeV, FP32_REPEAT_ELEMENT_NUM);
    }

    __aicore__ inline void InitGlobalTensor(GlobalTensor<float> lseMaxFdGm, GlobalTensor<float> lseSumFdGm,
                                            GlobalTensor<float> accumOutGm, GlobalTensor<OUTPUT_T> attentionOutGm,
                                            __gm__ uint8_t *key)
    {
        this->lseMaxFdGm = lseMaxFdGm;
        this->lseSumFdGm = lseSumFdGm;
        this->accumOutGm = accumOutGm;
        this->attentionOutGm = attentionOutGm;
        this->keyPtr = key;
    }

    __aicore__ inline void InitLearnableSinkGm(GlobalTensor<SINK_T> learnableSink)
    {
        learnableSinkFlag = true;
        this->sinkGm = learnableSink;
    }

    __aicore__ inline void InitSoftmaxLseGm(GlobalTensor<float> softmaxLseGm)
    {
        this->softmaxLseGm = softmaxLseGm;
    }

    __aicore__ inline void InitDecodeParams()
    {
        this->blockIdx = GetBlockIdx();
    }

    template <typename U>
    __aicore__ inline U Align(U num, U rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    __aicore__ inline void InitBuffers(TPipe *pipe, TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[], TBuf<> &SharedBuffer3)
    {
        if ASCEND_IS_AIV {
            constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
            constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
            constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
            constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
            fdMm2ResBuf1 = SharedBuffer1[0].template AllocTensor<T>();
            fdMm2ResBuf2 = fdMm2ResBuf1[BUFFER_SIZE_BYTE_16K / sizeof(T)];
            fdReduceBuf = SharedBuffer1[1].template AllocTensor<T>();
            fdOutputBuf = fdReduceBuf[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<OUTPUT_T>();
            fdSumBuf1 = SharedBuffer3.template Get<T>();
            fdSumBuf2 = fdSumBuf1[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdMaxBuf1 = fdSumBuf2[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdMaxBuf2 = fdMaxBuf1[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdLseExpBuf = fdMaxBuf2[(BUFFER_SIZE_BYTE_4K + BUFFER_SIZE_BYTE_2K) / sizeof(T)];
            fdLseMaxUbBuf1 = SharedBuffer2[0].template AllocTensor<T>();
            fdLseMaxUbBuf2 = fdLseMaxUbBuf1[BUFFER_SIZE_BYTE_256B / sizeof(T)];
            fdLseUbBuf = fdLseMaxUbBuf2[BUFFER_SIZE_BYTE_256B / sizeof(T)];
            if (unlikely(learnableSinkFlag)) {
                fdSinkCopyInBuf = fdLseUbBuf[BUFFER_SIZE_BYTE_256B / sizeof(T)];
                fdSinkValueBuf = fdSinkCopyInBuf[BUFFER_SIZE_BYTE_2K / sizeof(T)];
                fdSinkExpBuf = fdSinkValueBuf[BUFFER_SIZE_BYTE_2K / sizeof(T)];
                fdSinkTmpBuf = fdSinkExpBuf[BUFFER_SIZE_BYTE_256B / sizeof(T)];
            }
        }
    }

    __aicore__ inline void FreeBuffers(TQue<QuePosition::VECOUT, 1> SharedBuffer1[],
                                       TQue<QuePosition::VECIN, 1> SharedBuffer2[])
    {
        if ASCEND_IS_AIV {
            SharedBuffer1[0].template FreeTensor(fdMm2ResBuf1);
            SharedBuffer1[1].template FreeTensor(fdReduceBuf);
            SharedBuffer2[0].template FreeTensor(fdLseMaxUbBuf1);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<AscendC::HardEvent::V_MTE2>(0);
        SetFlag<AscendC::HardEvent::V_MTE2>(1);
        SetFlag<AscendC::HardEvent::V_MTE2>(2);
        SetFlag<AscendC::HardEvent::V_MTE2>(3);
        SetFlag<AscendC::HardEvent::MTE3_V>(4);
        SetFlag<AscendC::HardEvent::MTE3_V>(5);
        SetFlag<AscendC::HardEvent::V_MTE2>(6);
        SetFlag<AscendC::HardEvent::V_MTE2>(7);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        WaitFlag<AscendC::HardEvent::V_MTE2>(2);
        WaitFlag<AscendC::HardEvent::V_MTE2>(3);
        WaitFlag<AscendC::HardEvent::MTE3_V>(4);
        WaitFlag<AscendC::HardEvent::MTE3_V>(5);
        WaitFlag<AscendC::HardEvent::V_MTE2>(6);
        WaitFlag<AscendC::HardEvent::V_MTE2>(7);
    }

    __aicore__ inline void FlashDecode(FDparamsX &fdParams)
    {
        // TODO: 待后续填充实现
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_MIXEDQUANT_BLOCK_VEC_FLASHDECODE_H_
