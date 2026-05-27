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

#include "flash_attn_kernel_base_gqa_common.h"
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

    __aicore__ inline FlashAttentionNoQuantGqaKernel()
    {
    }

    __aicore__ inline void InitConstInfoWithPA()
    {
        if ASCEND_IS_AIC {
            cubeProc.InitConstInfo();
        } else {
            vectorProc.InitConstInfo();
        }
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        if ASCEND_IS_AIC {
            cubeProc.InitMMResBuf(workspace);
        } else {
            vectorProc.InitMMResBuf(workspace);
        }
    }

    __aicore__ inline void InitQCuSeqLensParser()
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            if ASCEND_IS_AIC {
                cubeProc.InitQCuSeqLensParser(cuSeqLensGmQ, seqUsedGmQ);
            }
            if ASCEND_IS_AIV {
                vectorProc.InitQCuSeqLensParser(cuSeqLensGmQ, seqUsedGmQ);
            }
        } else {
            if ASCEND_IS_AIC {
                cubeProc.InitQCuSeqLensParser(cuSeqLensGmQ, seqUsedGmQ);
            }
            if ASCEND_IS_AIV {
                vectorProc.InitQCuSeqLensParser(cuSeqLensGmQ, seqUsedGmQ);
            }
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser()
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            if ASCEND_IS_AIC {
                cubeProc.InitKvCuSeqLensParser(cuSeqLensGmKv, seqUsedGmKv);
            }
            if ASCEND_IS_AIV {
                vectorProc.InitKvCuSeqLensParser(cuSeqLensGmKv, seqUsedGmKv);
            }
        } else {
            if ASCEND_IS_AIC {
                cubeProc.InitKvCuSeqLensParser(cuSeqLensGmKv, seqUsedGmKv);
            }
            if ASCEND_IS_AIV {
                vectorProc.InitKvCuSeqLensParser(cuSeqLensGmKv, seqUsedGmKv);
            }
        }
    }

    __aicore__ inline void InitCube(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                    __gm__ uint8_t *blockTable)
    {
        if ASCEND_IS_AIC {
            cubeProc.InitCubeBlock(query, key, value, blockTable);
        }
    }

    __aicore__ inline void InitVector(__gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                      __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                      __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                      __gm__ uint8_t *workspace, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv)
    {
        if ASCEND_IS_AIV {
            vectorProc.InitVectorBlock(cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink,
                                        softmaxLse, attentionOut, workspace, seqUsedQ, seqUsedKv);
        }
    }

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

        this->InitConstInfoWithPA();

        cubeProc.keyPtr = key;
        vectorProc.keyPtr = key;

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

        this->InitQCuSeqLensParser();
        this->InitKvCuSeqLensParser();

        this->InitMMResBuf(workspace);

        uint32_t sectionNum = ((__gm__ uint32_t *)fiaMetaData)[0];
        cubeProc.sectionNum = sectionNum;
        vectorProc.sectionNum = sectionNum;

        cubeProc.faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                               FA_AIC_CORE_NUM * 16U * sectionNum);
        vectorProc.faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                                 FA_AIC_CORE_NUM * 16U * sectionNum);

        if ASCEND_IS_AIC {
            this->InitCube(query, key, value, blockTable);
        } else {
            vectorProc.isFd = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);
            if (vectorProc.isFd) {
                __gm__ uint32_t *fdMetaDataPtr = (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                                                      FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM *
                                                                          sectionNum * sizeof(uint32_t));
                vectorProc.vecFdBlock.SetFdMetaDataGm(fdMetaDataPtr, sectionNum);
            }
            this->InitVector(cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse, attentionOut, workspace,
                             seqUsedQ, seqUsedKv);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            if (vectorProc.constInfo.aicIdx < vectorProc.tilingData->flashAttnBaseParams.coreNum) {
                vectorProc.CrossCoreBufferInit();
            }
        }

        for (uint32_t sectionIdx = 0; sectionIdx < cubeProc.sectionNum; sectionIdx++) {
            if ASCEND_IS_AIC {
                cubeProc.ProcessCube(sectionIdx);
            } else {
                vectorProc.ProcessVector(sectionIdx);
            }
        }

        if ASCEND_IS_AIC {
            if (cubeProc.constInfo.aicIdx < cubeProc.tilingData->flashAttnBaseParams.coreNum) {
                cubeProc.CrossCoreBufferUnInit();
            }
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_