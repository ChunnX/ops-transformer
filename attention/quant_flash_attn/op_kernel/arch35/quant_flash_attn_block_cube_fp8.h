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
 * \file quant_flash_attn_block_cube_fullquant_fp8.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_FULLQUANT_FP8_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_FULLQUANT_FP8_H_
#if __has_include("../../../common/op_kernel/offset_calculator.h")
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#else
#include "../../common/op_kernel/offset_calculator.h"
#include "../../common/op_kernel/matmul.h"
#include "../../common/op_kernel/FixpipeOut.h"
#include "../../common/op_kernel/arch35/infer_flash_attention_comm_arch35.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/arch35/util_regbase.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "quant_flash_attn_type.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace AttentionCommon;

namespace BaseApi {

template <typename FA_T>
class QuantFlashAttnBlockCubeGqaFp8 {
public:
    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type =
            BuffersPolicyDB<BufferType::L0A, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = std::conditional_t<
            (s2BaseSize == 256 && dBaseSize > 128),
            BuffersPolicySingleBuffer<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK,
                                      IdSource::EXTERNAL>,
            BuffersPolicyDB<BufferType::L0B, SyncType::INNER_CORE_SYNC, SyncMode::LOCK_UNLOCK, IdSource::EXTERNAL>>;
    };

    // fa式核内同步（ADR 0003）：Mutex ID全局单一空间（0-27）须全局互异，AIC侧独立。
    // Q_L1 0-1（驻留双缓冲）、KV_L1 2-5（4buff）、L0A 6/7、L0B 8/9、L0C 10-13（条件4buff）或10/11（DB）。
    // L0A/L0B由matmul内部Wait/Set调用点经policy映射为Lock/Unlock（序列每迭代严格配对，fa同款）。
    static constexpr uint32_t Q_L1_BUFFER_ID0 = 0U;
    static constexpr uint32_t KV_L1_BUFFER_ID0 = 2U;
    static constexpr uint32_t L0A_BUFFER_ID0 = 6U;
    static constexpr uint32_t L0A_BUFFER_ID1 = 7U;
    static constexpr uint32_t L0B_BUFFER_ID0 = 8U;
    static constexpr uint32_t L0B_BUFFER_ID1 = 9U;
    static constexpr uint32_t L0C_BUFFER_ID0 = 10U;

    /* ============FA_T派生的基本块信息（对齐fa）============= */
    using INPUT_T = typename FA_T::inputType;
    using Q_T = typename FA_T::inputType;
    using KV_T = typename FA_T::inputType;
    using MM_T = float;
    using T = float;
    // fp8 原分派为 TND||NTD→cuParser（与 mxfp8 仅 TND 不同），tool 侧 INCLUDE_NTD=true
    using QSeqLensToolType = QFASeqLensTool<FA_T::qLayout, false, true, int32_t, GetQActSeqMode<FA_T::qLayout>()>;
    using KvSeqLensToolType = QFASeqLensTool<FA_T::kvLayout, FA_T::pageAttention, true, int32_t,
                                             GetKvActSeqMode<FA_T::kvLayout, FA_T::pageAttention>()>;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr DTemplateType dTemplateType = FA_T::dBaseSize;
    static constexpr DTemplateType dVTemplateType = FA_T::dVBaseSize;
    static constexpr LayOutTypeEnum LAYOUT = FA_T::qLayout;
    static constexpr LayOutTypeEnum layout = FA_T::qLayout; // 内部小写引用兼容（GetKVGmFormat等）
    static constexpr uint8_t KvLayoutType = FA_T::kvLayoutType;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool USE_DN = FA_T::useDn;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true};
    static constexpr GmFormat Q_FORMAT = GmFormat::NGTD;
    static constexpr GmFormat KV_FORMAT = GmFormat::PA_BnNBsD;

    // fa式L1/L0C布局：裸数组+显式bufId_轮转。L1的P区从0起（保证AIC/AIV跨核同址），
    // Q/KV从P区末尾起算（同vec侧VEC_UB_BASE模式，双侧公式一致）。
    // fp8 无 scale 尾巴：L1_P/L1_Q/L1_KV 均为纯 INPUT_T 尺寸（对照 mxfp8 无 /32 项）。
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_P_REGION_BYTES = L1_P_BUFCNT * L1_P_BUF_BYTES;
    // fa式跨核共享区（cube/vec双侧各自持有同布局裸数组，跨核flag=slotID，AIC侧±16双AIV镜像）：
    // UB: [bmm2Res 1槽][bmm1Res 2槽] 从0起（vec的VEC_UB_BASE紧随其后）；L1: P区3槽从0起（Q/KV紧随其后）
    // fp8 无 s2 拆分（Bmm1 单次覆盖全 s2BaseSize），MM1_RES 公式无 /2（对照现 kernel_fp8 InitMMResBuf）
    static constexpr uint32_t MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * (uint32_t)dVTemplateType * sizeof(T);
    static constexpr uint32_t MM2_RES_BUF_BYTES_TOTAL = MM2_RES_BUF_BYTES + MM1_RES_BUFCNT * MM1_RES_BUF_BYTES;
    // 跨核flag ID（mode 4 独立编号空间；AIC侧每flag须Set/Wait id与id+16两个）
    static constexpr uint32_t CC_MM_BMM1_0 = 0U;
    static constexpr uint32_t CC_MM_BMM1_1 = 1U;
    static constexpr uint32_t CC_MM_BMM2_0 = 2U;
    static constexpr uint32_t CC_L1P_0 = 3U;
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t L1_KV_BUF_BYTES = dBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr bool L0C_IS_SMALL = mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4;
    static constexpr uint32_t L0C_BUFCNT = L0C_IS_SMALL ? 4U : 2U;
    static constexpr uint32_t L0C_BUF_BYTES =
        L0C_IS_SMALL ? (L0C_SIZE / NUM_4) * KB_TO_BYTES : (L0C_SIZE / NUM_2) * KB_TO_BYTES;

    static constexpr bool Q_NEEDS_WZH = true;
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    using ConstInfoX = ConstInfo_t;
    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm_;
    FaGmTensorKV keyGm_;
    FaGmTensorKV valueGm_;
    GlobalTensor<int32_t> blockTableGm_;

    QSeqLensToolType &qSeqLensTool_;
    KvSeqLensToolType &kvSeqLensTool_;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1_;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1_;

    /* =====================LocalBuffer变量（fa式：裸数组+bufId_手工轮转）=====================*/
    BufferManager<BufferType::L0A> l0aBufferManager_;
    BufferManager<BufferType::L0B> l0bBufferManager_;

    LocalTensor<uint8_t> l1PBuffers_;
    LocalTensor<uint8_t> l1QBuffers_;
    LocalTensor<uint8_t> l1KvBuffers_;
    uint32_t qL1BufId_ = 0U;
    uint32_t kvL1BufId_ = 0U;
    uint32_t mm1ResBufId_ = 0U;

    LocalTensor<uint8_t> ubBmm1ResBuffers_;
    LocalTensor<uint8_t> ubBmm2ResBuffers_;

    LocalTensor<uint8_t> l0CBuffers_;
    uint32_t l0cBufId_ = 0U;

    L0AType mmL0ABuffers_;
    L0BType mmL0BBuffers_;

    __gm__ uint8_t *keyPtr_ = nullptr;
    __gm__ uint8_t *valuePtr_ = nullptr;

    const ConstInfoX &constInfo_;

    /* ============================================================================= */
    __aicore__ inline QuantFlashAttnBlockCubeGqaFp8(ConstInfoX &constInfo, QSeqLensToolType &qSeqLensTool,
                                                    KvSeqLensToolType &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        // fa式L1布局：P区(跨核共享,AIV写/AIC读)从0起，Q/KV裸数组从P区末尾手工排布
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, 0U, L1_P_BUFCNT * L1_P_BUF_BYTES);
        l1QBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_P_REGION_BYTES, L1_Q_BUFCNT * L1_Q_BUF_BYTES);
        l1KvBuffers_ = LocalTensor<uint8_t>(TPosition::A1, L1_P_REGION_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES,
                                            L1_KV_BUFCNT * L1_KV_BUF_BYTES);
        // L0A/L0B保留policy（LOCK_UNLOCK经matmul内部Wait/Set调用点映射），policy Init在AllocEventID（fa位置）
        l0aBufferManager_.Init(65536); // 64 * 1024
        l0bBufferManager_.Init(65536); // 64 * 1024
        // L0C裸数组（fa式）
        l0CBuffers_ = LocalTensor<uint8_t>(TPosition::CO1, 0U, L0C_BUFCNT * L0C_BUF_BYTES);
        // 跨核共享UB区（bmm2Res单槽 + bmm1Res双槽，与vec侧同布局）
        ubBmm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, 0U, MM2_RES_BUF_BYTES);
        ubBmm1ResBuffers_ =
            LocalTensor<uint8_t>(TPosition::VECIN, MM2_RES_BUF_BYTES, MM1_RES_BUFCNT * MM1_RES_BUF_BYTES);
    }

    __aicore__ inline void InitCrossCoreSync() {}

    __aicore__ inline void UnInitCrossCoreSync()
    {
        // AIC收尾drain：等两个AIV读完全部mmRes槽（free flag翻转到位）
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM1_1 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm_.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query);

        keyPtr_ = key;
        valuePtr_ = value;
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSize,
                     keyGm_, key, constInfo_.keyStrides.bnStride, constInfo_.keyStrides.n2Stride);
        InitKVBuffer(constInfo_.bSize, constInfo_.s2Size, constInfo_.n2Size, constInfo_.blockSize, constInfo_.dSizeV,
                     valueGm_, value, constInfo_.valueStrides.bnStride, constInfo_.valueStrides.n2Stride);
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, this->qSeqLensTool_.cuSeqLensParser);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim,
                                            this->qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm, uint64_t bnStrides, uint64_t n2Strides)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        } else {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm_,
                                             constInfo_.maxBlockNumPerBatch, bnStrides, n2Strides);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // fa位置：L0A/B policy Init（LOCK_UNLOCK，mutex ID经EXTERNAL注入）
        mmL0ABuffers_.Init(l0aBufferManager_, 32 * 1024, L0A_BUFFER_ID0, L0A_BUFFER_ID1);
        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            mmL0BBuffers_.Init(l0bBufferManager_, 32 * 1024, L0B_BUFFER_ID0);
        } else {
            mmL0BBuffers_.Init(l0bBufferManager_, 32 * 1024, L0B_BUFFER_ID0, L0B_BUFFER_ID1);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        mmL0ABuffers_.Uninit(l0aBufferManager_);
        mmL0BBuffers_.Uninit(l0bBufferManager_);
    }

    // copy query with full s1g
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        uint32_t nopeDealSize = dRealSize;
        uint32_t dstStride = (runInfo.actMSize + 31) >> 5 << 5;
        FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmCoord gmCoord{.bIdx = runInfo.bIdx,
                        .n2Idx = runInfo.realN2Idx,
                        .gS1Idx = runInfo.gS1Idx,
                        .dIdx = dOffset,
                        .gS1DealSize = runInfo.actMSize,
                        .dDealSize = nopeDealSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyQuerySlice(dstTensor, 0, constInfo_.dSize, runInfo);
    }

    // copy key with full s2
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                        uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        uint32_t nopeDealSize = dRealSize;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = nopeDealSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo, uint32_t s2RealSize)
    {
        uint32_t s2Offset = runInfo.s2Idx;
        CopyKeySlice(dstTensor, s2Offset, s2RealSize, 0, constInfo_.dSize, runInfo);
    }

    // copy value with full s2
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t s2Offset, uint32_t s2RealSize,
                                          uint32_t dOffset, uint32_t dRealSize, RunInfoX &runInfo)
    {
        uint32_t dstStride = (s2RealSize + 31) >> 5 << 5;
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = s2Offset,
                          .dIdx = dOffset,
                          .s2DealSize = s2RealSize,
                          .dDealSize = dRealSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, runInfo.s2Idx, runInfo.actSingleLoopS2Size, 0, constInfo_.dSizeV, runInfo);
    }

    __aicore__ inline void IterateBmm1(RunInfoX &runInfo)
    {
        uint32_t mm1ResBufId = mm1ResBufId_;
        mm1ResBufId_ = (mm1ResBufId_ + 1) % MM1_RES_BUFCNT;
        LocalTensor<T> mm1ResUbTensor =
            ubBmm1ResBuffers_[mm1ResBufId * MM1_RES_BUF_BYTES].template ReinterpretCast<T>();
        uint32_t mmSyncIdx = CC_MM_BMM1_0 + mm1ResBufId;
        // 等两个AIV读完该槽（free），fixpipe写入后向两个AIV置ready
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
        IterateBmm1Dn(mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(mmSyncIdx + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo,
                                        uint32_t s2RealSize)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5;
        fixpipeParams.mSize = (s2RealSize + 7) >> 3 << 3;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = 64;
        fixpipeParams.dualDstCtl = 2;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 256, D = 128场景, L1全载/L0全载, K * Q^T, Q矩阵驻留. GS1>80, S=S1*S2 */
    __aicore__ inline void IterateBmm1Dn(LocalTensor<T> &mm1ResUb, RunInfoX &runInfo)
    {
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        LocalTensor<Q_T> qL1Tensor = l1QBuffers_[qL1BufId_ * L1_Q_BUF_BYTES].template ReinterpretCast<Q_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            // Q驻留（fa式锁交接）：MTE2写入后交MTE1持有至本任务末（取代GetPre+裸Set惯用法）
            Mutex::Lock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            CopyQueryTile(qL1Tensor, runInfo);
            Mutex::Unlock<PIPE_MTE2>(Q_L1_BUFFER_ID0 + qL1BufId_);
            Mutex::Lock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
        }

        LocalTensor<KV_T> kL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyKeyTile(kL1Tensor, runInfo, s2CalcSize);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
            Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            MMParam param =
                MakeMMParam((uint32_t)s2CalcSize, (uint32_t)runInfo.actMSize, (uint32_t)constInfo_.dSize, false, true);
            MatmulFull<Q_T, KV_T, T, 256, 128, dBaseSize, ABLayout::MK, ABLayout::KN, L0AType, L0BType>(
                kL1Tensor, qL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param);
            Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
            Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

            FixpipeMm1Dn(mm1ResUb, l0CTensor, runInfo, s2CalcSize);

            Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        if (unlikely(runInfo.isLastS2Loop)) {
            Mutex::Unlock<PIPE_MTE1>(Q_L1_BUFFER_ID0 + qL1BufId_);
            qL1BufId_ = (qL1BufId_ + 1) % L1_Q_BUFCNT;
        }
    }

    __aicore__ inline void IterateBmm2(RunInfoX &runInfo)
    {
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<uint8_t> pL1Buf = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES];
        LocalTensor<T> mm2ResUbTensor = ubBmm2ResBuffers_.template ReinterpretCast<T>();
        // 等两个AIV写完该P槽（ready，MTE3置位）；bmm2Res单槽等free
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(v1c2CrossCoreSyncIdx + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0 + AIV0_AIV1_OFFSET);
        if constexpr (USE_DN) {
            IterateBmm2Dn(mm2ResUbTensor, pL1Buf, runInfo);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_MM_BMM2_0 + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (constInfo_.dSizeV + 7) >> 3 << 3;

        if constexpr (!USE_DN) {
            fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        }

        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2Dn(LocalTensor<T> &mm2ResUbTensor, LocalTensor<uint8_t> &pL1Buf,
                                         RunInfoX &runInfo)
    {
        LocalTensor<INPUT_T> pL1Tensor = pL1Buf.template ReinterpretCast<INPUT_T>();

        LocalTensor<T> l0CTensor = l0CBuffers_[l0cBufId_ * L0C_BUF_BYTES].template ReinterpretCast<T>();
        Mutex::Lock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);

        uint32_t realK = runInfo.actSingleLoopS2Size;
        LocalTensor<KV_T> vL1Tensor = l1KvBuffers_[kvL1BufId_ * L1_KV_BUF_BYTES].template ReinterpretCast<KV_T>();
        Mutex::Lock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        CopyValueSlice(vL1Tensor, runInfo.s2Idx, realK, 0, constInfo_.dSizeV, runInfo);
        Mutex::Unlock<PIPE_MTE2>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        Mutex::Lock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        {
            MMParam param =
                MakeMMParam((uint32_t)mBaseSize, (uint32_t)constInfo_.dSizeV, realK, USE_DN, false, true, true);
            MatmulFull<INPUT_T, KV_T, T, 128, 128, 256, ABLayout::MK, ABLayout::KN, L0AType, L0BType>(
                pL1Tensor, vL1Tensor, mmL0ABuffers_, mmL0BBuffers_, l0CTensor, param);
        }
        Mutex::Unlock<PIPE_MTE1>(KV_L1_BUFFER_ID0 + kvL1BufId_);
        kvL1BufId_ = (kvL1BufId_ + 1) % L1_KV_BUFCNT;

        Mutex::Unlock<PIPE_M>(L0C_BUFFER_ID0 + l0cBufId_);
        Mutex::Lock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);

        FixpipeMm2(mm2ResUbTensor, l0CTensor, runInfo); // 数据搬出

        Mutex::Unlock<PIPE_FIX>(L0C_BUFFER_ID0 + l0cBufId_);
        l0cBufId_ = (l0cBufId_ + 1) % L0C_BUFCNT;
    }
}; // QuantFlashAttnBlockCubeGqaFp8

template <typename FA_T>
class QuantFlashAttnBlockCubeGqaFp8Dummy {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr LayOutTypeEnum LAYOUT = FA_T::qLayout;
    static constexpr LayOutTypeEnum layout = FA_T::qLayout; // 内部小写引用兼容（GetKVGmFormat等）
    static constexpr uint8_t KvLayoutType = FA_T::kvLayoutType;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool USE_DN = FA_T::useDn;

    using Q_T = typename FA_T::inputType;
    using KV_T = typename FA_T::inputType;
    using MM_T = float;

    using ConstInfoX = ConstInfo_t;
    using QSeqLensToolType = QFASeqLensTool<FA_T::qLayout, false, true, int32_t, GetQActSeqMode<FA_T::qLayout>()>;
    using KvSeqLensToolType = QFASeqLensTool<FA_T::kvLayout, FA_T::pageAttention, true, int32_t,
                                             GetKvActSeqMode<FA_T::kvLayout, FA_T::pageAttention>()>;
    __aicore__ inline QuantFlashAttnBlockCubeGqaFp8Dummy(ConstInfoX &constInfo, QSeqLensToolType &,
                                                         KvSeqLensToolType &){};
};
} // namespace BaseApi

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_FULLQUANT_FP8_H_
