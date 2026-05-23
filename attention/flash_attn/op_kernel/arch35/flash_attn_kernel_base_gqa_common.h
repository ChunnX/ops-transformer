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
 * \file flash_attn_kernel_base_gqa_common.h
 * \brief Flash Attention GQA common definitions and shared methods
 */

#ifndef FLASH_ATTENTION_NOQUANT_GQA_COMMON_H_
#define FLASH_ATTENTION_NOQUANT_GQA_COMMON_H_

#include "../../../common/op_kernel/fia_public_define.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_utils.h"

#include "flash_attn_block_cube_noquant_gqa.h"
#include "flash_attn_block_vec_noquant_gqa.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "flash_attention_noquant_block_vec_flashdecode.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attn_tiling_data.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {

class FlashAttentionGqaContext {
public:
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;
    GlobalTensor<uint32_t> fdMetaDataGm;
    GlobalTensor<float> softmaxLseGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfo_t<FiaKernelType::NO_QUANT> constInfo;

    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData;
    TPipe *pipe;

    uint32_t coreGS1Loops = 0U;
    uint32_t frontGS1Count = 0U;
    uint32_t invalidGS1Size = 0U;
    uint32_t accGS1Loops = 0U;
    uint32_t totalSize = 0U;
    uint32_t createdTaskCount = 0U;
    uint32_t executedTaskCount = 0U;
    uint32_t varlenCalcTimes = 0U;
    bool enableS1OutSplit = false;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    int32_t s1SizeCurrBatch = 0;
    uint32_t curS2Start;
    uint32_t curS2End = 0;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t mloop = 0;
    bool headS2Split = false;
    bool tailS2Split = false;
    bool emptyBatch = false;
    uint32_t dealZerosMSize = 0;

    uint32_t sectionNum_;
    bool isFd;

    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    uint32_t coreFirstTmpOutWsPos_;

    FDparamsX fdParams_;

    TQue<QuePosition::VECOUT, 1> SharedBuffer1[2];
    TQue<QuePosition::VECIN, 1> SharedBuffer2[2];
    TBuf<> SharedBuffer3;

    __aicore__ inline void InitConstInfo()
    {
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aivIdx = GetBlockIdx();
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }
        constInfo.bSize = this->tilingData->flashAttnBaseParams.bSize;
        constInfo.t1Size = this->tilingData->flashAttnBaseParams.t1Size;
        constInfo.t2Size = this->tilingData->flashAttnBaseParams.t2Size;
        constInfo.n2Size = this->tilingData->flashAttnBaseParams.n2Size;
        constInfo.gSize = this->tilingData->flashAttnBaseParams.gSize;
        constInfo.s1Size = this->tilingData->flashAttnBaseParams.s1Size;
        constInfo.s2Size = this->tilingData->flashAttnBaseParams.s2Size;
        constInfo.dSize = this->tilingData->flashAttnBaseParams.dSize;
        constInfo.dSizeV = this->tilingData->flashAttnBaseParams.dSizeV;
        constInfo.cuSeqLensQSize = this->tilingData->flashAttnBaseParams.cuSeqLensQSize;
        constInfo.cuSeqLensKVSize = this->tilingData->flashAttnBaseParams.cuSeqLensKVSize;
        constInfo.seqUsedQSize = this->tilingData->flashAttnBaseParams.seqUsedQSize;
        constInfo.seqUsedKvSize = this->tilingData->flashAttnBaseParams.seqUsedKvSize;
        constInfo.scaleValue = static_cast<float>(this->tilingData->flashAttnBaseParams.scaleValue);
        constInfo.isKvContinuous = this->tilingData->flashAttnBaseParams.isKvContinuous != 0;
        constInfo.coreNum = this->tilingData->flashAttnBaseParams.coreNum;
        constInfo.outputLayout = static_cast<FA_LAYOUT>(this->tilingData->flashAttnBaseParams.outputLayout);

        constInfo.sparseMode = this->tilingData->flashAttnAttenMaskParams.sparseMode;
        constInfo.preTokens = this->tilingData->flashAttnAttenMaskParams.winLefts;
        constInfo.nextTokens = this->tilingData->flashAttnAttenMaskParams.winRights;
        constInfo.attenMaskBatch = this->tilingData->flashAttnAttenMaskParams.attenMaskBatch;
        constInfo.attenMaskS1Size = this->tilingData->flashAttnAttenMaskParams.attenMaskS1Size;
        constInfo.attenMaskS2Size = this->tilingData->flashAttnAttenMaskParams.attenMaskS2Size;
        constInfo.isExistRowInvalid = this->tilingData->flashAttnAttenMaskParams.isExistRowInvalid;

        constInfo.accumOutSize = this->tilingData->flashAttnWorkspaceParams.accumOutSize;
        constInfo.logSumExpSize = this->tilingData->flashAttnWorkspaceParams.logSumExpSize;

        constInfo.isSoftmaxLseEnable = this->tilingData->flashAttnBaseParams.isSoftMaxLseEnable;

        constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FA_FD_METADATA_SIZE * FA_AIV_CORE_NUM * sectionIdx + FA_FD_METADATA_SIZE * coreIdx + metaIdx;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
        coreFirstTmpOutWsPos_ = faMetaDataGm.GetValue(
            GetFAMetaDataIndex(constInfo.aicIdx, FLASH_ATTN_FIRST_FD_DATA_WORKSPACE_IDX_INDEX, sectionIdx));
    }

    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN2Idx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_BN2_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart = fdMetaDataGm.GetValue(GetFDMetaDataIndex(constInfo.aivIdx, FA_FD_M_START_INDEX, sectionIdx));
    }
};

template <typename CubeBlockType>
class FlashAttentionCubeData {
public:
    static constexpr uint32_t mBaseSize = CubeBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = CubeBlockType::s2BaseSize;
    static constexpr uint32_t dBaseSize = CubeBlockType::dBaseSize;
    static constexpr uint32_t dVBaseSize = CubeBlockType::dVBaseSize;

    static constexpr bool USE_DN = CubeBlockType::USE_DN;
    static constexpr bool BMM2_TOUB = CubeBlockType::BMM2_TOUB;
    static constexpr bool HAS_PREFIX = CubeBlockType::HAS_PREFIX;

    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool PAGE_ATTENTION = CubeBlockType::PAGE_ATTENTION;

    static constexpr LayOutTypeEnum LAYOUT_Q = CubeBlockType::LAYOUT;
    static constexpr LayOutTypeEnum LAYOUT_KV = CubeBlockType::LAYOUT;
    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    static constexpr bool IS_TND_LAYOUT = (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND || LAYOUT_Q == LayOutTypeEnum::LAYOUT_NTD);

    using INPUT_T = typename CubeBlockType::Q_T;
    using T = typename CubeBlockType::MM_T;
    using ConstInfoX = typename CubeBlockType::ConstInfoX;

    using QCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                           ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                                           ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t>>::type;
    using KvCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                            ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                            ActualSeqLensParser<KV_MODE, int32_t>>::type;
    using QSeqUsedParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;
    using KvSeqUsedParserType = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;

    QCuSeqLensParserType qCuSeqLensParser;
    KvCuSeqLensParserType kvCuSeqLensParser;
    QSeqUsedParserType qSeqUsedParser;
    KvSeqUsedParserType kvSeqUsedParser;

    CubeBlockType cubeBlock;
    FlashAttentionGqaContext* ctx;

    __aicore__ inline FlashAttentionCubeData(FlashAttentionGqaContext* c) : cubeBlock(c->constInfo), ctx(c){};
};

template <typename VecFaBlockType, typename VecFdBlockType, typename InputT, LayOutTypeEnum LayoutQ = LayOutTypeEnum::None, bool PageAttention = false>
class FlashAttentionVectorData {
public:
    using VecFaBlockTypeAlias = VecFaBlockType;
    static constexpr uint32_t mBaseSize = VecFaBlockType::mBaseSize;
    static constexpr uint32_t s2BaseSize = VecFaBlockType::s2BaseSize;

    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PRELOAD_N + 1;

    static constexpr bool HAS_MASK = VecFaBlockType::HAS_MASK;

    static constexpr LayOutTypeEnum LAYOUT_Q = LayoutQ;
    static constexpr LayOutTypeEnum LAYOUT_KV = LayoutQ;
    static constexpr bool PAGE_ATTENTION = PageAttention;

    static constexpr bool IS_TND_LAYOUT = (LAYOUT_Q == LayOutTypeEnum::LAYOUT_TND || LAYOUT_Q == LayOutTypeEnum::LAYOUT_NTD);

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<LAYOUT_Q>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<LAYOUT_KV, PAGE_ATTENTION>();

    using INPUT_T = InputT;
    using ConstInfoX = typename VecFaBlockType::ConstInfoX;

    using QCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                           ActualSeqLensParser<Q_MODE, int32_t, true>,
                                                           ActualSeqLensParser<Q_MODE, int32_t>>::type;
    using KvCuSeqLensParserType = typename std::conditional<IS_TND_LAYOUT,
                                                            ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                            ActualSeqLensParser<KV_MODE, int32_t>>::type;
    using QSeqUsedParserType = QCuSeqLensParserType;
    using KvSeqUsedParserType = KvCuSeqLensParserType;

    QCuSeqLensParserType qCuSeqLensParser;
    KvCuSeqLensParserType kvCuSeqLensParser;
    QSeqUsedParserType qSeqUsedParser;
    KvSeqUsedParserType kvSeqUsedParser;

    VecFaBlockType vecFaBlock;
    VecFdBlockType vecFdBlock;
    FlashAttentionGqaContext* ctx;

    __aicore__ inline FlashAttentionVectorData(FlashAttentionGqaContext* c) : vecFaBlock(c->constInfo), vecFdBlock(c->constInfo), ctx(c){};
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_COMMON_H_