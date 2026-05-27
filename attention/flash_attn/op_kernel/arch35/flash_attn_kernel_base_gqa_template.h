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
 * \brief Flash Attention GQA kernel entry point (lightweight dispatch)
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_
#define FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_

#include "flash_attn_kernel_base_gqa_cube.h"
#include "flash_attn_kernel_base_gqa_vector.h"

namespace BaseApi {

template <typename CubeBlockType, typename VecFaBlockType, typename VecFdBlockType>
class FlashAttentionNoQuantGqaKernel {
public:
    using CubeProcessor = FlashAttentionCubeProcessor<CubeBlockType>;
    using VectorProcessor = FlashAttentionVectorProcessor<VecFaBlockType, VecFdBlockType, typename CubeBlockType::Q_T,
                                                          CubeBlockType::LAYOUT, CubeBlockType::PAGE_ATTENTION>;

    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;

    CubeProcessor cubeProc;
    VectorProcessor vectorProc;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        cubeProc.pipe = tPipe;
        cubeProc.tilingData = tiling;
        vectorProc.pipe = tPipe;
        vectorProc.tilingData = tiling;

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ,
                                          tiling->flashAttnBaseParams.cuSeqLensQSize + 1);
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ,
                                        tiling->flashAttnBaseParams.seqUsedQSize);
            cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv,
                                            tiling->flashAttnBaseParams.cuSeqLensKVSize + 1);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv,
                                          tiling->flashAttnBaseParams.seqUsedKvSize);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ,
                                        tiling->flashAttnBaseParams.seqUsedQSize);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv,
                                          tiling->flashAttnBaseParams.seqUsedKvSize);
        }

        cubeProc.Init(query, key, value, blockTable, fiaMetaData, seqUsedQ, seqUsedKv,
                       cuSeqLensGmQ, seqUsedGmQ, cuSeqLensGmKv, seqUsedGmKv, workspace);

        vectorProc.Init(cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse, attentionOut,
                        workspace, fiaMetaData, seqUsedQ, seqUsedKv,
                        cuSeqLensGmQ, seqUsedGmQ, cuSeqLensGmKv, seqUsedGmKv, key);
    }

    __aicore__ inline void Process()
    {
        cubeProc.Process();
        vectorProc.Process();
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_