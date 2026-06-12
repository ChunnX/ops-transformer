/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file flash_attn_gqa_cube.h
 * \brief Cube-side GQA flash attention: Block layer (FANoQuantGqaCube) + Kernel layer (FlashAttentionNoQuantGqaKernelCube)
 */
#ifndef FLASH_ATTENTION_GQA_CUBE_H_
#define FLASH_ATTENTION_GQA_CUBE_H_
#include "../../../common/op_kernel/fia_public_define.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_cube_def.h"
#include "../utils/flash_attn_utils.h"
#include "../../../common/op_kernel/offset_calculator.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"

#endif
#include "flash_attn_tiling_data.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
using namespace optiling;
// using namespace AttentionCommon;

namespace BaseApi {

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}




template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false, bool hasAttenMask = false>
class FANoQuantGqaCube {
public:
    template <uint32_t dBaseSize>
    struct QL1BuffSel {
        using Type = std::conditional_t<(dBaseSize > 256), BuffersPolicySingleBuffer<BufferType::L1>,
                                        BuffersPolicyDB<BufferType::L1>>;
    };

    /* ============确定Key的L1类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct KVL1BuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>;
    };

    /* ============确定L0A的类型============= */
    struct L0ABuffSel {
        using Type = BuffersPolicyDB<BufferType::L0A>;
    };
    /* ============确定L0B的类型============= */
    template <uint32_t s2BaseSize, uint32_t dBaseSize>
    struct L0BBuffSel {
        using Type = std::conditional_t<(s2BaseSize == 256 && dBaseSize > 128),
                                        BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
    };

    /* ============确定L0C的类型============= */
    template <uint32_t mBaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
    struct L0CBuffSel {
        using Type = std::conditional_t<(mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                         mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
                                        BuffersPolicy4buff<BufferType::L0C>, BuffersPolicyDB<BufferType::L0C>>;
    };

    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_PREFIX = false;
    static constexpr bool SPLITD = splitD;
    static constexpr bool HAS_MASK = hasAttenMask;

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    // static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using MM1_DBUF_T = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using MM2_ABUF_POLICY_T = BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_ABUF_T = Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>;
    using MM2_DBUF_T = mm2ResPos;

    using L1KvType = typename KVL1BuffSel<s2BaseSize, dBaseSize>::Type;
    using L1QType = typename QL1BuffSel<dBaseSize>::Type;
    using L0AType = typename L0ABuffSel::Type;
    using L0BType = typename L0BBuffSel<s2BaseSize, dBaseSize>::Type;
    using L0CType = typename L0CBuffSel<mBaseSize, s2BaseSize, dVBaseSize>::Type;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    using QSeqParserType =
        typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                  ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>>::type;

    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    GlobalTensor<int32_t> blockTableGm;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    /* =====================LocalBuffer变量====================*/
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    L1QType l1QBuffers;
    L1KvType l1KBuffers;
    L1KvType l1VBuffers;

    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;
    L0CType mmL0CBuffers;

    ConstInfoXCube constInfoXCube;

    __aicore__ inline void InitBuffers()
    {
        if constexpr (dBaseSize > 256) {
            /* D大于256的其他dtype场景，Bmm1左矩阵不开DB + 驻留 + 复用 + L1切K
            Bmm2左矩阵3 Buffer循环，Bmm1右矩阵和Bmm2右矩阵在L1上切D轴，并开启DoubleBuffer
            唯一不同的是D=256场景下的D轴只能切分到96，其余都可以切分到128，原因是D=256场景S1Base是128
            */
            constexpr uint32_t mm1LeftSize = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            constexpr uint32_t mm2RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            l1QBuffers.Init(l1BufferManager, mm1LeftSize);
            l1KBuffers.Init(l1BufferManager, mm1RightSize);
            l1VBuffers.Init(l1BufferManager, mm2RightSize);
        } else {
            constexpr uint32_t mm1LeftSize = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(KV_T);
            constexpr uint32_t mm2RightSize = (uint32_t)dVTemplateType * s2BaseSize * sizeof(KV_T);
            l1QBuffers.Init(l1BufferManager, mm1LeftSize);
            l1KBuffers.Init(l1BufferManager, mm1RightSize);
            l1VBuffers.Init(l1BufferManager, mm2RightSize);
        }

        // L0A B C 当前写死，能否通过基础api获取
        l0aBufferManager.Init(pipe, 65536);
        l0bBufferManager.Init(pipe, 65536);
        l0cBufferManager.Init(pipe, 262144);
        // L0A B C当前写死，要改成通过计算获取
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);

        if constexpr (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                      mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
        } else {
            mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfoXCube.bSize, constInfoXCube.n2Size, constInfoXCube.gSize, constInfoXCube.s1Size, constInfoXCube.dSize, queryGm,
                    query);

        if (constInfoXCube.isKvContinuous) {
            InitKVBuffer(constInfoXCube.bSize, constInfoXCube.s2Size, constInfoXCube.n2Size, constInfoXCube.blockSize, constInfoXCube.dSize,
                         keyGm, key);
            InitKVBuffer(constInfoXCube.bSize, constInfoXCube.s2Size, constInfoXCube.n2Size, constInfoXCube.blockSize, constInfoXCube.dSizeV,
                         valueGm, value);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqParser);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfoXCube.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfoXCube.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(kvSeqParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqParser);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // InitBuffers阶段已完成eventId申请和SetFlag，这里为空实现
    }

    __aicore__ inline void FreeEventID()
    {
        l1QBuffers.Uninit(l1BufferManager);
        l1KBuffers.Uninit(l1BufferManager);
        l1VBuffers.Uninit(l1BufferManager);
        mmL0ABuffers.Uninit(l0aBufferManager);
        mmL0BBuffers.Uninit(l0bBufferManager);
        mmL0CBuffers.Uninit(l0cBufferManager);
    }

    // copy query with full s1g and split D
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoXCube &runInfo)
    {
        uint32_t nopeDealSize = dRealSize;

        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        if (nopeDealSize > 0) {
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = dOffset,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = nopeDealSize};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoXCube &runInfo)
    {
        uint32_t dSize = constInfoXCube.dSize;
        CopyQuerySlice(dstTensor, 0, dSize, runInfo);
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfoXCube &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;

        uint32_t nopeDealSize = dRealSize;
        uint32_t ropeDealSize = 0;

        if (nopeDealSize > 0) {
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = constInfoXCube.isKvContinuous ? runInfo.bIdx : 0,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = dOffset,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = nopeDealSize};
            copyKvGmToL1(l1Tensor, keyGm, gmCoord);
        }
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoXCube &runInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = constInfoXCube.isKvContinuous ? runInfo.bIdx : 0,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoXCube &runInfo)
    {
        uint32_t dSize = constInfoXCube.dSize;
        CopyKeySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoXCube &runInfo)
    {
        CopyValueSlice(dstTensor, 0, constInfoXCube.dSizeV, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr, bIdx);
        keyGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm.offsetCalculator.Init(0, constInfoXCube.n2Size, s2Size, constInfoXCube.dSize);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr, bIdx);
        valueGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm.offsetCalculator.Init(0, constInfoXCube.n2Size, s2Size, constInfoXCube.dSize);
    }

    __aicore__ inline void IterateBmm1(MM1_DBUF_T &outputBuf, RunInfoXCube &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateKey(runInfo.bIdx);
            }
        }

#ifdef SKIP_C1
        outputBuf.WaitCrossCore();
        outputBuf.SetCrossCore();
        return;

#endif

        if constexpr (dBaseSize > 256) {
            IterateBmm1NdL1SplitK(outputBuf, runInfo);
            return;
        }

        if constexpr (useDn) {
            if constexpr (dBaseSize > 128) {
                IterateBmm1DnSplitK(outputBuf, runInfo);
            } else {
                IterateBmm1Dn(outputBuf, runInfo);
            }
            return;
        }
        IterateBmm1NdL0Split(outputBuf, runInfo);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoXCube &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1;         // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 128, D > 128场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    __aicore__ inline void IterateBmm1NdL0Split(MM1_DBUF_T &outputBuf, RunInfoXCube &runInfo)
    {
        Buffer<BufferType::L1> mm1A;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A = l1QBuffers.Get();
            mm1A.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
            CopyQueryTile(mm1ATensor, runInfo);
            mm1A.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1A = l1QBuffers.GetPre();
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1B = l1KBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
        CopyKeyTile(mm1BTensor, runInfo);
        mm1B.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)(constInfoXCube.dSize), false, true);
        if constexpr (dBaseSize > 128) {
            if constexpr (s2BaseSize == 256) {
                MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
                    mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
                    param);
            } else {
                MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                    mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
                    param);
            }
        } else {
            MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<Q_T>(), mm1B.GetTensor<KV_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(),
                param);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

        outputBuf.WaitCrossCore();

        FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    /* 针对S1Base=128, S2Base = 128, D > 256场景，L1层面切K，且左矩阵单Buffer+驻留，右矩阵每次重新搬运。*/
    __aicore__ inline void IterateBmm1NdL1SplitK(MM1_DBUF_T &outputBuf, RunInfoXCube &runInfo)
    {
        constexpr uint32_t baseK = l1BaseD;
        uint32_t kLoops = (constInfoXCube.dSize + baseK - 1) / baseK;

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        uint32_t dstNzC0Stride = ((runInfo.actMSize + 15) >> 4 << 4);
        uint64_t l1BaseKOffset = baseK * dstNzC0Stride;

        Buffer<BufferType::L1> mm1A = l1QBuffers.Get();
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1A.Wait<HardEvent::MTE1_MTE2>();
        }

        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            Buffer<BufferType::L1> mm1B; // 左矩阵复用, 但是每次只加载realK列
            if (k == kLoops - 1) {
                realK = constInfoXCube.dSize - k * baseK;
            }
            if (unlikely(runInfo.isFirstS2Loop)) { // sOuter循环第一个基本快：搬运0
                LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();
                CopyQuerySlice(mm1ATensor[k * l1BaseKOffset], k * baseK, realK, runInfo);

                mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
            } else {                              // 非s2的第一次循环直接复用Q
                mm1A = l1QBuffers.GetPre();
                mm1A.Set<HardEvent::MTE2_MTE1>();
            }
            // 加载当前轮的右矩阵到L1
            mm1B = l1KBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<KV_T> mm1BTensor = mm1B.GetTensor<KV_T>();
            CopyKeySlice(mm1BTensor, k * baseK, realK, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();  // 通知
            mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
            mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

            MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size, realK, false,
                                        true, k == 0, k == 0);

            MatmulFull<Q_T, KV_T, T, 128, 128, baseK, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<Q_T>()[k * l1BaseKOffset], mm1BTensor, mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(), param);

            mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            mm1A.Set<HardEvent::MTE1_MTE2>();
        }
        mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
        outputBuf.WaitCrossCore();

        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
        FixpipeMm1(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放
        outputBuf.SetCrossCore();
    }

    /* 针对useDn=true, S1Base=128, S2Base = 128, 128 < D <= 256场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    __aicore__ inline void IterateBmm1DnSplitK(MM1_DBUF_T &outputBuf, RunInfoXCube &runInfo)
    {
        Buffer<BufferType::L1> mm1B;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1B = l1QBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();

        CopyKeyTile(mm1ATensor, runInfo);

        mm1A.Set<HardEvent::MTE2_MTE1>();

        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                    (uint32_t)(constInfoXCube.dSize), false, true);

        MatmulK<KV_T, Q_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<KV_T>(), mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();

        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        FixpipeMm1Dn(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);

        mm1ResL0C.Set<HardEvent::FIX_M>();
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoXCube &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5; // 双目标模式, 按照N方向切分，N必须为32的倍数
        fixpipeParams.mSize = runInfo.actSingleLoopS2Size;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2; // 按照N方向切分
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams); // 将matmul结果从L0C搬运到UB
    }

    __aicore__ inline void IterateBmm1Dn(MM1_DBUF_T &outputBuf, RunInfoXCube &runInfo)
    {
        Buffer<BufferType::L1> mm1B;
        if (unlikely(runInfo.isFirstS2Loop)) {
            mm1B = l1QBuffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<Q_T> mm1BTensor = mm1B.GetTensor<Q_T>();
            CopyQueryTile(mm1BTensor, runInfo);
            mm1B.Set<HardEvent::MTE2_MTE1>();
        } else {
            mm1B = l1QBuffers.GetPre();
            mm1B.Set<HardEvent::MTE2_MTE1>();
        }

        Buffer<BufferType::L1> mm1A = l1KBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<KV_T> mm1ATensor = mm1A.GetTensor<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo);

        mm1A.Set<HardEvent::MTE2_MTE1>();
        mm1A.Wait<HardEvent::MTE2_MTE1>();
        mm1B.Wait<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
        mm1ResL0C.Wait<HardEvent::FIX_M>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                    (uint32_t)(constInfoXCube.dSize), false, true);

        MatmulBase<KV_T, Q_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<KV_T>(), mm1B.GetTensor<Q_T>(), mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), param);

        if (unlikely(runInfo.isLastS2Loop)) {
            mm1B.Set<HardEvent::MTE1_MTE2>();
        }
        mm1A.Set<HardEvent::MTE1_MTE2>();

        mm1ResL0C.Set<HardEvent::M_FIX>();
        mm1ResL0C.Wait<HardEvent::M_FIX>();

        outputBuf.WaitCrossCore();
        FixpipeMm1Dn(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), runInfo);
        mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
        outputBuf.SetCrossCore();
    }

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoXCube &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateValue(runInfo.bIdx);
            }
        }

#ifdef SKIP_C2
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            outputBuf.WaitCrossCore();
        }
        outputBuf.SetCrossCore();
        return;

#endif

        if constexpr ((uint32_t)dVTemplateType > 256 || (uint32_t)dTemplateType > 256) {
            IterateBmm2L1SplitN(outputBuf, inputBuf, runInfo);
        } else {
            IterateBmm2l0Split(outputBuf, inputBuf, runInfo);
        }
    }

    __aicore__ inline void IterateBmm2L1SplitN(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoXCube &runInfo)
    {
        MM2_ABUF_T mm2A = inputBuf.Get();
        mm2A.WaitCrossCore();

        // dVTemplateType > 256, L1切N, 左矩阵不变，右矩阵每次循环搬运S2 * 128
        if constexpr (bmm2Write2Ub) {
            outputBuf.WaitCrossCore();
        }
        constexpr uint32_t baseN = l1BaseD;
        uint32_t nLoops = ((uint32_t)constInfoXCube.dSizeV + baseN - 1) / baseN; // 尾块处理
        uint32_t realN = baseN;
        for (uint32_t n = 0; n < nLoops; ++n) {
            if (n == nLoops - 1) {
                uint32_t tailSize = (uint32_t)constInfoXCube.dSizeV % baseN;
                realN = tailSize ? tailSize : baseN;
            }
            Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
            mm2B.Wait<HardEvent::MTE1_MTE2>();
            LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
            CopyValueSlice(mm2BTensor, n * baseN, realN, runInfo);
            mm2B.Set<HardEvent::MTE2_MTE1>();

            Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
            mm2ResL0C.Wait<HardEvent::FIX_M>();

            MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)realN, (uint32_t)(runInfo.actSingleLoopS2Size),
                                        useDn, false);
            if constexpr (!useDn) {
                param.realM = (uint32_t)runInfo.actMSize;
            }
            mm2B.Wait<HardEvent::MTE2_MTE1>();
            MatmulFull<Q_T, KV_T, T, 128, baseN, 128, ABLayout::MK, ABLayout::KN>(
                mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);

            mm2B.Set<HardEvent::MTE1_MTE2>();
            mm2ResL0C.Set<HardEvent::M_FIX>();
            mm2ResL0C.Wait<HardEvent::M_FIX>();

            FixpipeMm2PartialN(outputBuf.template GetTensor<T>()[n * baseN], mm2ResL0C.GetTensor<T>(), realN, runInfo);
            mm2ResL0C.Set<HardEvent::FIX_M>();
        }
        outputBuf.SetCrossCore();
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              RunInfoXCube &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        if constexpr (bmm2Write2Ub) {
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = realN;
        }

        if constexpr (!useDn) {
            fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        }

        if constexpr (bmm2Write2Ub || splitD) {
            fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfoXCube.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    __aicore__ inline void IterateBmm2l0Split(mm2ResPos &outputBuf, MM2_ABUF_POLICY_T &inputBuf, RunInfoXCube &runInfo)
    {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
        Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
        mm2A.WaitCrossCore();
        mm2B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
        LocalTensor<KV_T> mm2BTensor = mm2B.GetTensor<KV_T>();
        CopyValueTile(mm2BTensor, runInfo);
        mm2B.Set<HardEvent::MTE2_MTE1>(); // 通知

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
        MMParam param = {
            (uint32_t)mBaseSize,                   // singleM 128
            (uint32_t)constInfoXCube.dSizeV,            // singleN 128
            (uint32_t)runInfo.actSingleLoopS2Size, // singleK
            useDn,                                 // isLeftTranspose
            false                                  // isRightTranspose
        };
        if constexpr (!useDn) {
            param.realM = (uint32_t)runInfo.actMSize;
        }
        mm2B.Wait<HardEvent::MTE2_MTE1>(); // 等待

        if constexpr ((uint32_t)dVTemplateType > 128) {
            MatmulN<Q_T, KV_T, T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
        } else {
            if constexpr (s2BaseSize == 128) {
                MatmulFull<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
            } else {
                MatmulBase<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    mm2A.GetTensor<Q_T>(), mm2BTensor, mmL0ABuffers, mmL0BBuffers, mm2ResL0C.GetTensor<T>(), param);
            }
        }

        mm2B.Set<HardEvent::MTE1_MTE2>();   // 释放L1B
        mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
        mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

        if constexpr (bmm2Write2Ub) {
            outputBuf.WaitCrossCore();
        }

        FixpipeMm2PartialN(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), constInfoXCube.dSizeV, runInfo);

        mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
        outputBuf.SetCrossCore();
    }

    static constexpr uint32_t PRELOAD_N = 2;
    // task ring buffer 至少需要 PRELOAD_N+1(=3) 个slot；为了用位掩码(loop & MASK)代替取模(loop % SIZE)，
    // 将容量向上取整到最近的2的幂(=4)。代价仅为多1个RunInfoXCube slot，换取消除内层循环的取模scalar bound。
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1,
                  "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");

    /* =====================CV Buffer变量====================*/
    BufferManager<BufferType::GM> gmBufferManager;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    GlobalTensor<uint32_t> faMetaDataGm;
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    TPipe *pipe;

    /* =====================Loop State变量====================*/
    uint32_t createdTaskCount = 0U;
    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start;
    uint32_t curS2End = 0;
    uint32_t prevBIdx;
    uint32_t prevBN2Idx;
    uint32_t prevGS1Idx;
    uint32_t sectionNum_;
    /* =====================FA metadata变量====================*/
    uint32_t bN2Start_;
    uint32_t bN2End_;
    uint32_t gS1OStart_;
    uint32_t gS1OEnd_;
    uint32_t s2OStart_;
    uint32_t s2OEnd_;
    /* =====================SeqLens Parser变量====================*/
    QSeqParserType qSeqParser;
    KvSeqParserType kvSeqParser;
    /* ==============================fuction=======================================================*/
    __aicore__ inline void InitCube(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                    __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                    __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                    __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling, TPipe *tPipe,
                                    __gm__ uint8_t *blockTable)
    {
        this->pipe = tPipe;

        InitConstInfo(tiling);

        keyPtr = key;
        valuePtr = value;

        GlobalTensor<int32_t> localCuSeqLensGmQ;
        GlobalTensor<int32_t> localSeqUsedGmQ;
        GlobalTensor<int32_t> localCuSeqLensGmKv;
        GlobalTensor<int32_t> localSeqUsedGmKv;

        if constexpr (IS_TND_LAYOUT) {
            localCuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfoXCube.cuSeqLensQSize + 1);
            localSeqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXCube.seqUsedQSize);
            localCuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfoXCube.cuSeqLensKVSize + 1);
            localSeqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXCube.seqUsedKvSize);
        } else {
            localSeqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXCube.seqUsedQSize);
            localSeqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXCube.seqUsedKvSize);
        }

        InitQSeqParser(localCuSeqLensGmQ, localSeqUsedGmQ);
        InitKvSeqParser(localCuSeqLensGmKv, localSeqUsedGmKv);

        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                      FA_AIC_CORE_NUM * 16U * sectionNum_);

        InitMMResBuf(workspace);
        InitCubeInput(query, key, value, blockTable);
    }
    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        uint32_t mm1OutDtype = sizeof(T);
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
        constexpr uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
        constexpr uint32_t mm2LeftSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        l1BufferManager.Init(pipe, 524288); // 512 * 1024
        // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        if constexpr (BMM2_TOUB) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2);
        }
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
        // GM Buffer
        if constexpr (!BMM2_TOUB) {
            int64_t mm2ResultSize =
                mBaseSize * constInfoXCube.dBasicBlock; // 使用Cube计算的总大小，Gm上的数据按照实际的dSize存储
            int64_t prevCoretotalOffset = constInfoXCube.aicIdx * 3 * mm2ResultSize; // 3为preload次数
            // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
            gmBufferManager.Init(workspace + prevCoretotalOffset * sizeof(T));
            bmm2ResGmBuffers.Init(gmBufferManager, mm2ResultSize * sizeof(T));
            workspace = workspace + constInfoXCube.coreNum * 3 * mm2ResultSize * sizeof(T);
        }
    }
    __aicore__ inline void InitQSeqParser(GlobalTensor<int32_t> &cuSeqLensGmQ, GlobalTensor<int32_t> &seqUsedGmQ)
    {
        if constexpr (IS_TND_LAYOUT) {
            qSeqParser.Init(cuSeqLensGmQ, seqUsedGmQ, constInfoXCube.cuSeqLensQSize, constInfoXCube.seqUsedQSize);
        } else {
            qSeqParser.Init(seqUsedGmQ, constInfoXCube.seqUsedQSize, constInfoXCube.s1Size);
        }
    }
    __aicore__ inline void InitKvSeqParser(GlobalTensor<int32_t> &cuSeqLensGmKv, GlobalTensor<int32_t> &seqUsedGmKv)
    {
        if constexpr (IS_TND_LAYOUT) {
            kvSeqParser.Init(cuSeqLensGmKv, seqUsedGmKv, constInfoXCube.cuSeqLensKVSize, constInfoXCube.seqUsedKvSize);
        } else {
            kvSeqParser.Init(seqUsedGmKv, constInfoXCube.seqUsedKvSize, constInfoXCube.s2Size);
        }
    }
    __aicore__ inline void InitConstInfo(const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling)
    {
        constInfoXCube.aicIdx = GetBlockIdx();
        constInfoXCube.bSize = tiling->flashAttnBaseParams.bSize;
        constInfoXCube.n2Size = tiling->flashAttnBaseParams.n2Size;
        constInfoXCube.gSize = tiling->flashAttnBaseParams.gSize;
        constInfoXCube.s1Size = tiling->flashAttnBaseParams.s1Size;
        constInfoXCube.s2Size = tiling->flashAttnBaseParams.s2Size;
        constInfoXCube.dSize = tiling->flashAttnBaseParams.dSize;
        constInfoXCube.dSizeV = tiling->flashAttnBaseParams.dSizeV;
        constInfoXCube.cuSeqLensQSize = tiling->flashAttnBaseParams.cuSeqLensQSize;
        constInfoXCube.cuSeqLensKVSize = tiling->flashAttnBaseParams.cuSeqLensKVSize;
        constInfoXCube.seqUsedQSize = tiling->flashAttnBaseParams.seqUsedQSize;
        constInfoXCube.seqUsedKvSize = tiling->flashAttnBaseParams.seqUsedKvSize;
        constInfoXCube.isKvContinuous = tiling->flashAttnBaseParams.isKvContinuous != 0;
        constInfoXCube.coreNum = tiling->flashAttnBaseParams.coreNum;
        constInfoXCube.sparseMode = tiling->flashAttnAttenMaskParams.sparseMode;
        constInfoXCube.preTokens = tiling->flashAttnAttenMaskParams.winLefts;
        constInfoXCube.nextTokens = tiling->flashAttnAttenMaskParams.winRights;
        if constexpr (PAGE_ATTENTION) {
            constInfoXCube.maxBlockNumPerBatch = tiling->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfoXCube.blockSize = tiling->flashAttnPageAttentionParams.blockSize;
        }
        constInfoXCube.dBasicBlock = Align64Func((uint16_t)constInfoXCube.dSizeV);
    }
    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        // AICPU metadata format: 16 fields per AIC core, 0-indexed (no leading CORE_ENABLE).
        // Kernel field constants ( FLASH_ATTN_BN2_START_INDEX=1, etc.) are 1-based, so subtract 1.
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }
    __aicore__ inline void CrossCoreBufferUnInit()
    {
        bmm1Buffers.Get().WaitCrossCore();
        bmm1Buffers.Get().WaitCrossCore();
        if constexpr (BMM2_TOUB) {
            bmm2Buffers.Get().WaitCrossCore();
            bmm2Buffers.Get().WaitCrossCore();
        }
    }
    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        if (constInfoXCube.aicIdx >= constInfoXCube.coreNum) {
            return;
        }
        GetFASectionInfo(sectionIdx);
        RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE] = {};
        createdTaskCount = 0;
        uint32_t executedTaskCount = 0;
        uint32_t bN2Cur = bN2Start_;
        uint32_t gS1Cur = gS1OStart_;
        uint32_t s2Cur = s2OStart_;
        prevBN2Idx = bN2Cur;
        prevGS1Idx = gS1Cur;
        bool shouldDispatchTask = true;
        uint32_t validTaskCount = 0; // 未执行(有效)的任务数
        while (shouldDispatchTask || validTaskCount) {
            // 分发任务
            shouldDispatchTask = ShouldDispatchTask(bN2Cur, gS1Cur, s2Cur);
            if (shouldDispatchTask) {
                TASK_DEAL_MODE taskDealMode = GetTaskDealMode(bN2Cur, gS1Cur, s2Cur);
                if (taskDealMode == TASK_DEAL_MODE::CREATE_TASK) {
                    // 创建任务
                    CreateTask(createdTaskCount, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
                    createdTaskCount++;
                    validTaskCount++;
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                } else if (taskDealMode == TASK_DEAL_MODE::DEAL_ZERO) {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                } else {
                    UpdateAxisInfo(taskDealMode, bN2Cur, gS1Cur, s2Cur);
                    continue;
                }
            }
            // 执行任务
            if (validTaskCount) {
                ExecuteTask(executedTaskCount, taskRunInfo);
                executedTaskCount++;
                if (executedTaskCount > PRELOAD_N) {
                    validTaskCount--;
                }
            }
        }
    }
    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
    }
    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfoXCube.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfoXCube.cuSeqLensKVSize == 0 && constInfoXCube.seqUsedKvSize == 0 && !constInfoXCube.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorList<LAYOUT>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvSeqParser.GetActualSeqLength(bIdx);
            }
            actSeqLensQ = qSeqParser.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXCube.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        // 计算每一行的起止点，只有当换行时（bN2Cur、gS1Cur更新）才需要重新计算
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }
        if (curS2Start < curS2End && s2Cur >= curS2End) {
            return TASK_DEAL_MODE::S2_END;
        }
        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }
        return TASK_DEAL_MODE::CREATE_TASK;
    }
    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfoXCube.preTokens;
        nextTokenLeftUp = constInfoXCube.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfoXCube.sparseMode);
        if (constInfoXCube.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
        }
        if (constInfoXCube.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        } else if (constInfoXCube.sparseMode == fa_base_vector::BAND) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        }
    }
    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;
        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            curS2Start = s2OStart_;
        }
        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            curS2End = s2OEnd_;
        }
    }
    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        // 1. Calc preTokenLeftUp, nextTokenLeftUp
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);
        // 2. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfoXCube.gSize)) -
                               1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfoXCube.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfoXCube.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                // start and end locate in one G
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                // start and end locate in tow or more G, but working same as crossing one complete block
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }
        // 3. trans index of token to index of block
        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        // no valid token
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        // get valid range
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;
        // 4. Calc curS2Start, curS2End
        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;
        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) { // first line
            curS2Start = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) { // last line
            curS2End = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }
    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXCube &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];                  // 本轮任务
        RunInfoXCube &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK]; // 上PRELOAD_N轮任务
        if (runInfo0.isValid) {
            IterateBmm1(bmm1Buffers.Get(), runInfo0);
        }
        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                if constexpr (BMM2_TOUB) {
                    IterateBmm2(bmm2Buffers.Get(), l1PBuffers, runInfoNegN);
                } else {
                    IterateBmm2(bmm2ResGmBuffers.Get(), l1PBuffers, runInfoNegN);
                }
                runInfoNegN.isValid = false;
            }
        }
    }
    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXCube &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK]; // 本轮任务
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }
    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXCube &info)
    {
        info.bIdx = bN2Cur / constInfoXCube.n2Size;
        info.n2Idx = bN2Cur % constInfoXCube.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.s2Idx = s2Cur * s2BaseSize;
        info.actMSize = mBaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXCube.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size = actSeqLensKv - s2Cur * s2BaseSize;
        }
        if (constInfoXCube.isKvContinuous) {
            info.isChangeBatch = false;
        } else {
            // for tensor-list
            if (loop == 0) {
                info.isChangeBatch = true;
            } else {
                info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
            }
        }
        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
    }
    __aicore__ inline void UpdateAxisInfo(TASK_DEAL_MODE taskDealMode, uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXCube.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        // 当前S2未处理完
        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }
        // 当前BN2未处理完
        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }
        // 当前BN2已处理完
        gS1Cur = 0;
        bN2Cur++;
    }
    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
    }
    __aicore__ inline void Process()
    {
        if (constInfoXCube.aicIdx < constInfoXCube.coreNum) {
            InitBuffers();
            AllocEventID();
        }
        for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
            if (constInfoXCube.aicIdx < constInfoXCube.coreNum) {
                FlashAttention(sectionIdx);
            }
        }
        if (constInfoXCube.aicIdx < constInfoXCube.coreNum) {
            FreeEventID();
            CrossCoreBufferUnInit();
        }
    }

};

} // namespace BaseApi


#endif
