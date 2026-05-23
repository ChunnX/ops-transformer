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
 * \file flash_attn_kernel_base_gqa_template.h
 * \brief Flash Attention GQA kernel entry point
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_
#define FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_

#include "flash_attn_kernel_base_gqa_common.h"
#include "flash_attn_kernel_base_gqa_cube.h"
#include "flash_attn_kernel_base_gqa_vector.h"

namespace BaseApi {

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernel
    : public FlashAttentionCubeProcessor<FlashAttentionVectorProcessor<FlashAttentionNoQuantGqaKernelBase<CubeBlockType, VecFaBlockType, VecFdBlockType>>> {
public:
    using Base = FlashAttentionCubeProcessor<FlashAttentionVectorProcessor<FlashAttentionNoQuantGqaKernelBase<CubeBlockType, VecFaBlockType, VecFdBlockType>>>;
    using Base::Base;

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        this->sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];
        this->isFd = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);

        this->faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                           FA_AIC_CORE_NUM * 16U * this->sectionNum_);
        this->fdMetaDataGm.SetGlobalBuffer(
            (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * this->sectionNum_ * sizeof(uint32_t)),
            FA_AIV_CORE_NUM * 16U * this->sectionNum_);

        this->InitConstInfoCommon();
        this->InitConstInfoCube();
        this->InitConstInfoVector();

        this->keyPtr = key;
        this->valuePtr = value;

        if constexpr (Base::LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            this->cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, this->constInfo.cuSeqLensQSize + 1);
            this->seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, this->constInfo.seqUsedQSize);
        } else {
            this->seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, this->constInfo.seqUsedQSize);
        }
        if constexpr (Base::LAYOUT_KV == LayOutTypeEnum::LAYOUT_TND) {
            this->cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, this->constInfo.cuSeqLensKVSize + 1);
            this->seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, this->constInfo.seqUsedKvSize);
        } else {
            this->seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, this->constInfo.seqUsedKvSize);
        }

        this->InitQCuSeqLensParser();
        this->InitKvCuSeqLensParser();

        this->InitMMResBuf(workspace);

        this->InitVector(cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse,
                         attentionOut, workspace, seqUsedQ, seqUsedKv);
        
        this->InitCube(query, key, value, blockTable);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t sectionIdx = 0; sectionIdx < this->sectionNum_; sectionIdx++) {
            if ASCEND_IS_AIC {
                this->ProcessCube(sectionIdx);
            } else {
                this->ProcessVector(sectionIdx);
            }
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_