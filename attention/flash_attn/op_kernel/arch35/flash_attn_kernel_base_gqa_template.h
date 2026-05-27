/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not using this file except in compliance with the License.
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
class FlashAttentionNoQuantGqaKernel : public FlashAttentionGqaContext {
public:
    using CubeData = FlashAttentionCubeData<CubeBlockType>;
    using VectorData = FlashAttentionVectorData<VecFaBlockType, VecFdBlockType, typename CubeBlockType::Q_T,
                                                CubeBlockType::LAYOUT, CubeBlockType::PAGE_ATTENTION>;

    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;

    FlashAttentionCubeProcessor<CubeData> cubeProc;
    FlashAttentionVectorProcessor<VectorData> vectorProc;

    __aicore__ inline FlashAttentionNoQuantGqaKernel() : cubeProc(this), vectorProc(this)
    {
    }

    __aicore__ inline void InitConstInfoWithPA()
    {
        FlashAttentionGqaContext::InitConstInfo();
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        using T = typename CubeBlockType::MM_T;
        using INPUT_T = typename CubeBlockType::Q_T;
        constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
        constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
        constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;
        constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;

        uint32_t mm1OutDtype = sizeof(T);
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        this->l1BufferManager.Init(this->pipe, 524286);
        this->l1PBuffers.Init(this->l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            this->ubBufferManager.Init(this->pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            this->bmm2Buffers.Init(this->ubBufferManager, mm2ResultSize);
        } else {
            this->ubBufferManager.Init(this->pipe, mm1ResultSize * 2);
        }
        this->bmm1Buffers.Init(this->ubBufferManager, mm1ResultSize);

        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSizeCalc = mBaseSize * this->tilingData->flashAttnBaseParams.dSizeV;
            int64_t prevCoretotalOffset = this->constInfo.aicIdx * 3 * mm2ResultSizeCalc;
            workspace = workspace + this->tilingData->flashAttnBaseParams.coreNum * 3 * mm2ResultSizeCalc * sizeof(T);
        }
    }

    __aicore__ inline void InitQCuSeqLensParser()
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            if ASCEND_IS_AIC {
                cubeProc.cubeData.qCuSeqLensParser.Init(this->cuSeqLensGmQ, this->seqUsedGmQ,
                                                        this->tilingData->flashAttnBaseParams.cuSeqLensQSize,
                                                        this->tilingData->flashAttnBaseParams.seqUsedQSize);
            }
            if ASCEND_IS_AIV {
                vectorProc.vectorData.qCuSeqLensParser.Init(this->cuSeqLensGmQ, this->seqUsedGmQ,
                                                            this->tilingData->flashAttnBaseParams.cuSeqLensQSize,
                                                            this->tilingData->flashAttnBaseParams.seqUsedQSize);
            }
        } else {
            if ASCEND_IS_AIC {
                cubeProc.cubeData.qSeqUsedParser.Init(this->seqUsedGmQ,
                                                      this->tilingData->flashAttnBaseParams.seqUsedQSize,
                                                      this->tilingData->flashAttnBaseParams.s1Size);
            }
            if ASCEND_IS_AIV {
                vectorProc.vectorData.qSeqUsedParser.Init(this->seqUsedGmQ,
                                                          this->tilingData->flashAttnBaseParams.seqUsedQSize,
                                                          this->tilingData->flashAttnBaseParams.s1Size);
            }
        }
    }

    __aicore__ inline void InitKvCuSeqLensParser()
    {
        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            if ASCEND_IS_AIC {
                cubeProc.cubeData.kvCuSeqLensParser.Init(this->cuSeqLensGmKv, this->seqUsedGmKv,
                                                         this->tilingData->flashAttnBaseParams.cuSeqLensKVSize,
                                                         this->tilingData->flashAttnBaseParams.seqUsedKvSize);
            }
            if ASCEND_IS_AIV {
                vectorProc.vectorData.kvCuSeqLensParser.Init(this->cuSeqLensGmKv, this->seqUsedGmKv,
                                                             this->tilingData->flashAttnBaseParams.cuSeqLensKVSize,
                                                             this->tilingData->flashAttnBaseParams.seqUsedKvSize);
            }
        } else {
            if ASCEND_IS_AIC {
                cubeProc.cubeData.kvSeqUsedParser.Init(this->seqUsedGmKv,
                                                       this->tilingData->flashAttnBaseParams.seqUsedKvSize,
                                                       this->tilingData->flashAttnBaseParams.s2Size);
            }
            if ASCEND_IS_AIV {
                vectorProc.vectorData.kvSeqUsedParser.Init(this->seqUsedGmKv,
                                                           this->tilingData->flashAttnBaseParams.seqUsedKvSize,
                                                           this->tilingData->flashAttnBaseParams.s2Size);
            }
        }
    }

    __aicore__ inline void InitCube(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                    __gm__ uint8_t *blockTable)
    {
        if ASCEND_IS_AIC {
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                cubeProc.cubeData.cubeBlock.InitCubeBlock(this->pipe, &this->l1BufferManager, query, key, value,
                                                          blockTable, cubeProc.cubeData.qCuSeqLensParser,
                                                          cubeProc.cubeData.kvCuSeqLensParser, this->tilingData);
            } else {
                cubeProc.cubeData.cubeBlock.InitCubeBlock(this->pipe, &this->l1BufferManager, query, key, value,
                                                          blockTable, cubeProc.cubeData.qSeqUsedParser,
                                                          cubeProc.cubeData.kvSeqUsedParser, this->tilingData);
            }
        }
    }

    __aicore__ inline void InitVector(__gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                      __gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                      __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                      __gm__ uint8_t *workspace, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv)
    {
        if ASCEND_IS_AIV {
            vectorProc.vectorData.vecFaBlock.isFd = vectorProc.vectorData.isFd;
            if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                vectorProc.vectorData.vecFaBlock.InitVecBlock(this->pipe, cuSeqLensQ, cuSeqLensKv, attenMask,
                                                              learnableSink, softmaxLse, attentionOut, workspace,
                                                              this->tilingData);
                vectorProc.vectorData.vecFaBlock.SetCuSeqLensParser(vectorProc.vectorData.qCuSeqLensParser);
            } else {
                vectorProc.vectorData.vecFaBlock.InitVecBlock(this->pipe, seqUsedQ, seqUsedKv, attenMask, learnableSink,
                                                              softmaxLse, attentionOut, workspace, this->tilingData);
                vectorProc.vectorData.vecFaBlock.SetCuSeqLensParser(vectorProc.vectorData.qSeqUsedParser);
            }
            vectorProc.vectorData.vecFaBlock.ClearOutput();

            if (vectorProc.vectorData.isFd) {
                vectorProc.vectorData.vecFdBlock.SetTilingData(this->tilingData);
                vectorProc.vectorData.vecFdBlock.InitParams();
                vectorProc.vectorData.vecFdBlock.InitGlobalTensor(
                    vectorProc.vectorData.vecFaBlock.softmaxFDMaxGm, vectorProc.vectorData.vecFaBlock.softmaxFDSumGm,
                    vectorProc.vectorData.vecFaBlock.accumOutGm, vectorProc.vectorData.vecFaBlock.attentionOutGm,
                    this->keyPtr);
                if (learnableSink != nullptr) {
                    GlobalTensor<typename CubeBlockType::Q_T> sinkGm;
                    sinkGm.SetGlobalBuffer((__gm__ typename CubeBlockType::Q_T *)learnableSink);
                    vectorProc.vectorData.vecFaBlock.learnableSinkFlag = true;
                    vectorProc.vectorData.vecFdBlock.InitLearnableSinkGm(sinkGm);
                }
                if (this->tilingData->flashAttnBaseParams.isSoftMaxLseEnable) {
                    GlobalTensor<float> softmaxLseGm;
                    softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
                    vectorProc.vectorData.vecFdBlock.InitSoftmaxLseGm(softmaxLseGm);
                }
                if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
                    vectorProc.vectorData.vecFdBlock.SetCuSeqLensParsers(vectorProc.vectorData.qCuSeqLensParser,
                                                                         vectorProc.vectorData.kvCuSeqLensParser);
                } else {
                    vectorProc.vectorData.vecFdBlock.SetCuSeqLensParsers(vectorProc.vectorData.qSeqUsedParser,
                                                                         vectorProc.vectorData.kvSeqUsedParser);
                }
            }
        }
    }

    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe)
    {
        this->pipe = tPipe;
        this->tilingData = tiling;

        this->InitConstInfoWithPA();

        this->keyPtr = key;

        if constexpr (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND) {
            this->cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ,
                                               this->tilingData->flashAttnBaseParams.cuSeqLensQSize + 1);
            this->seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ,
                                             this->tilingData->flashAttnBaseParams.seqUsedQSize);
            this->cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv,
                                                this->tilingData->flashAttnBaseParams.cuSeqLensKVSize + 1);
            this->seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv,
                                              this->tilingData->flashAttnBaseParams.seqUsedKvSize);
        } else {
            this->seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ,
                                             this->tilingData->flashAttnBaseParams.seqUsedQSize);
            this->seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv,
                                              this->tilingData->flashAttnBaseParams.seqUsedKvSize);
        }

        this->InitQCuSeqLensParser();
        this->InitKvCuSeqLensParser();

        this->InitMMResBuf(workspace);

        this->sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        this->faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                           FA_AIC_CORE_NUM * 16U * this->sectionNum_);

        if ASCEND_IS_AIC {
            this->InitCube(query, key, value, blockTable);
        } else {
            vectorProc.vectorData.isFd = static_cast<bool>(((__gm__ uint32_t *)fiaMetaData)[1]);
            if (vectorProc.vectorData.isFd) {
                __gm__ uint32_t *fdMetaDataPtr = (__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET +
                                                                     FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM *
                                                                         this->sectionNum_ * sizeof(uint32_t));
                vectorProc.vectorData.vecFdBlock.SetFdMetaDataGm(fdMetaDataPtr, this->sectionNum_);
            }
            this->InitVector(cuSeqLensQ, cuSeqLensKv, attenMask, learnableSink, softmaxLse, attentionOut, workspace,
                             seqUsedQ, seqUsedKv);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            if (this->constInfo.aicIdx < this->tilingData->flashAttnBaseParams.coreNum) {
                vectorProc.CrossCoreBufferInit();
            }
        }

        for (uint32_t sectionIdx = 0; sectionIdx < this->sectionNum_; sectionIdx++) {
            if ASCEND_IS_AIC {
                cubeProc.ProcessCube(sectionIdx);
            } else {
                vectorProc.ProcessVector(sectionIdx);
            }
        }

        if ASCEND_IS_AIC {
            if (this->constInfo.aicIdx < this->tilingData->flashAttnBaseParams.coreNum) {
                cubeProc.CrossCoreBufferUnInit();
            }
        }
    }
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_KERNEL_H_