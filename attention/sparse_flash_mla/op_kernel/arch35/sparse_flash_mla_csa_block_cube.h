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
 * \file sparse_flash_mla_csa_block_cube.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H_
#define SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H_
#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"

#if __has_include("../../common/op_kernel/offset_calculator.h")
#include "../../common/op_kernel/offset_calculator.h"
#else
#include "../common/offset_calculator.h"
#endif
#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace SMLAKernel {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

template <SMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

template <SMLA_LAYOUT KV_LAYOUT_T>
__aicore__ inline constexpr GmFormat GetKvGmFormat()
{
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        return GmFormat::TND;
    } else { // BSND
        return GmFormat::BSND;
    }
}

TEMPLATES_DEF
class CSABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline CSABlockCube(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> &l1BufferManager,
                                         __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices,
                                         __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                         __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensQ,
                                         __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                         __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
                                         const ConstInfo &constInfo);

    // SWA/HCA场景, inputRightBuf是INNER_CORE_SYNC类型
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
                                         RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
                                       Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
                                       bool notLastTwoLoop, RunInfo &runInfoNext,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void
    IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
                Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf, const RunInfo &runInfo,
                const ConstInfo &constInfo);

    // CSA场景, inputRightBuf是CROSS_CORE_SYNC_FORWARD类型
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                         Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                         RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
                                       Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                       bool notLastTwoLoop, RunInfo &runInfoNext,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void
    IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
                Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
                const ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer(BufferManager<BufferType::L1> &l1BufferManager);
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                        __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
                                        const ConstInfo &constInfo);
    __aicore__ inline void CalcS2Coord(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    template <typename RightBufType>
    __aicore__ inline void LoadKGmToL1(RightBufType &inputRightBuf, const RunInfo &runInfo,
                                      const ConstInfo &constInfo);

    template <typename RightBufType>
    __aicore__ inline void IterateBmm1Impl(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                           RightBufType &inputRightBuf, bool notLastTwoLoop,
                                           RunInfo &runInfoNext, RunInfo &runInfo, ConstInfo &constInfo);
    template <typename RightBufType>
    __aicore__ inline void
    IterateBmm2Impl(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
                    RightBufType &inputRightBuf, const RunInfo &runInfo, const ConstInfo &constInfo);

    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKvGmFormat<KV_LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == SMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;
    static constexpr bool KV_WITH_ZERO_HEAD = (KV_LAYOUT_T == SMLA_LAYOUT::TND);
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> oriKvGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> cmpKvGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> curKvGm;
    GlobalTensor<int32_t> cuSeqlensQGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t l0ABBufId = 0;
    uint32_t l0CBufId = 0;
    uint32_t l1QBufId = 0;
    uint32_t l1KLoadBufId = 0;
    uint32_t l1KMatmul1BufId = 0;
    uint32_t l1KMatmul2BufId = 0;
    uint32_t l0ABMToMte1FlagId = 3; // {0, 1}, 用于L0AB
    uint32_t l0ABMte1ToMFlagId = 3; // {0, 1}, 用于L0AB
    uint32_t l0CFixToMFlagId = 0; // {0, 1}, 用于L0C
    uint32_t l0CMToFixFlagId = 0; // {0, 1}, 用于L0C
    uint32_t l1QMte1ToMte2FlagId = 0; // {0, 1, 2}, 用于l1Q
    uint32_t l1QMte2ToMte1FlagId = 0; // {0, 1, 2}, 用于l1Q
    uint32_t l1KMte1ToMte2FlagId = 3; // {3, 4, 5}, 用于l1K
    uint32_t l1KMte2ToMte1FlagId = 3; // {3, 4, 5}, 用于l1K
    /* =====================LocalBuffer变量==================== */
    TBuf<TPosition::A1> l1QBuffers;
    TBuf<TPosition::A2> mmL0ABuffers;
    TBuf<TPosition::B2> mmL0BBuffers;
    TBuf<TPosition::CO1> mmL0CBuffers;
    LocalTensor<Q_T> l1QTensor;
    LocalTensor<Q_T> mmL0ATensor;
    LocalTensor<Q_T> mmL0BTensor;
    LocalTensor<T> mmL0CTensor;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitCubeBlock(TPipe *pipe,
                                                                  BufferManager<BufferType::L1> &l1BufferManager,
                                                                  __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer(l1BufferManager);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitCubeInput(
    __gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
    __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
    const ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        this->oriKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)oriKv);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            this->oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
        }
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                      TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
            this->cmpKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)cmpKv);
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                this->cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
            }
        }
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
            this->cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
        }
        InitGmTensor(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitLocalBuffer(BufferManager<BufferType::L1> &l1BufferManager)
{
    tPipe->InitBuffer(l1QBuffers, BUFFER_SIZE_96K);
    l1QTensor = l1QBuffers.Get<Q_T>();

    tPipe->InitBuffer(mmL0ABuffers, BUFFER_SIZE_32K);
    tPipe->InitBuffer(mmL0BBuffers, BUFFER_SIZE_64K);
    tPipe->InitBuffer(mmL0CBuffers, BUFFER_SIZE_256K);
    mmL0ATensor = mmL0ABuffers.Get<Q_T>();
    mmL0BTensor = mmL0BBuffers.Get<Q_T>();
    mmL0CTensor = mmL0CBuffers.Get<T>();
    SetFlag<HardEvent::M_MTE1>(l0ABMToMte1FlagId); // {0, 1}, 用于L0AB
    SetFlag<HardEvent::M_MTE1>(l0ABMToMte1FlagId + 1);
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId); // {0, 1}, 用于L0C
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId); // {0, 1, 2}, 用于l1Q
    SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + 2);
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId); // {3, 4, 5}, 用于l1K
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + 1);
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + 2);
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                                 __gm__ uint8_t *cuSeqlensOriKv,
                                                                 __gm__ uint8_t *cuSeqlensCmpKv,
                                                                 __gm__ uint8_t *seqUsedOriKV,
                                                                 __gm__ uint8_t *seqUsedCmpKV,
                                                                 const ConstInfo &constInfo)
{
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                            constInfo.dSize);
    } else { // SMLA_LAYOUT::TND
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        uint32_t sequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensQ, constInfo.bSize + 1, sequsedQ, sequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, parser);
    }

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.oriBlockSize, constInfo.dSize,
                                            this->oriBlockTableGm, constInfo.oriMaxBlockNumPerBatch);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.cmpBlockSize, constInfo.dSize,
                                                this->cmpBlockTableGm, constInfo.cmpMaxBlockNumPerBatch);
        }
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        uint32_t seqUsedOriKvSize = (seqUsedOriKV == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensOriKv, constInfo.actualSeqLenSize + 1, seqUsedOriKV, seqUsedOriKvSize);
        this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize, parser);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                      TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
            uint32_t seqUseCmpKvSize = (seqUsedCmpKV == nullptr) ? 0 : constInfo.bSize;
            ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
            parser.Init(cuSeqlensCmpKv, constInfo.actualSeqLenSize + 1, seqUsedCmpKV, seqUseCmpKvSize);
            this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize, parser);
        }
    } else {
        // BSND不需要初始化
        this->oriKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.s2Size, constInfo.dSize);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.cmpS2Size,
                                                constInfo.dSize); // 替换constInfo.s2Size / constInfo.cmpRatio
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CalcS2Coord(const RunInfo &runInfo, const ConstInfo &constInfo)
{
    // 计算s2方向偏移
    coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.boIdx;
    if (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) {
        kvCacheBlockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord =
            runInfo.s2StartIdx + (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) * s2BaseSize;
        blockTableGm = cmpBlockTableGm;
        curKvGm = cmpKvGm;
    } else {
        kvCacheBlockSize = constInfo.oriBlockSize;
        maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
        blockTableGm = oriBlockTableGm;
        curKvGm = oriKvGm;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
        runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufId + i) % 3;
        WaitFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + curL1QBufId);
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
            this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize, constInfo.dSize >> 1,
            constInfo.mm1Ka);
        SetFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename RightBufType>
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::LoadKGmToL1(RightBufType &inputRightBuf,
                                                                const RunInfo &runInfo, const ConstInfo &constInfo)
{
    LocalTensor<KV_T> inputRightTensor = inputRightBuf.template GetTensor<KV_T>();
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = coordInfo[runInfo.taskIdMod3].s2Coord;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        shape.pageStride = runInfo.isCmp ? constInfo.cmpKeyStride0 : constInfo.oriKeyStride0;
        GmCopyInToL1PA<KV_T>(inputRightTensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
    } else {
        int64_t keyOffset = this->curKvGm.offsetCalculator.GetOffset(
            coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<KV_T>(inputRightTensor, curKvGm.gmTensor[keyOffset], runInfo.s2RealSize, constInfo.dSize,
                            constInfo.mm1Kb);
    }
}

// SWA/HCA场景: K从GM直接搬运
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
                                           RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    // 加载当前轮的右矩阵到L1
    CalcS2Coord(runInfo, constInfo);
    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KLoadBufId);
    LoadKGmToL1(inputRightBuf, runInfo, constInfo);
    SetFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KLoadBufId);
    l1KLoadBufId = (l1KLoadBufId + 1) % 3;
}

// CSA场景: K来源取决于isSparse
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                           Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    // ORI_SPARSE、ORI_CMP_SPARSE及CSA的cmpKv为v0稀疏搬运，CSA的oriKv为cube搬运
    bool isSparse = false;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        isSparse = true;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        isSparse = runInfo.isCmp ? true : false;
    }

    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KLoadBufId);
    if (!isSparse) {
        // cube直接从kv cache搬运K
        CalcS2Coord(runInfo, constInfo);
        LoadKGmToL1(inputRightBuf, runInfo, constInfo);
    } else {
        LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
        v0ResGm.WaitCrossCore();
        GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
        CopyToL1Nd2Nz<Q_T>(dst, v0ResGmTensor, runInfo.s2RealSize, constInfo.dSize, constInfo.mm1Kb);
    }
    SetFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KLoadBufId);
    l1KLoadBufId = (l1KLoadBufId + 1) % 3;
}

// TODO 两个mm1 mm2可以统一吗
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                         Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
                                         bool notLastTwoLoop, RunInfo &runInfoNext,
                                         RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1Impl(outputBuf, inputRightBuf, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                         Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                         bool notLastTwoLoop, RunInfo &runInfoNext,
                                         RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1Impl(outputBuf, inputRightBuf, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                         BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
                                         Buffer<BufferType::L1, SyncType::INNER_CORE_SYNC> &inputRightBuf,
                                         const RunInfo &runInfo, const ConstInfo &constInfo)
{
    IterateBmm2Impl(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                         BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
                                         Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
                                         const RunInfo &runInfo, const ConstInfo &constInfo)
{
    IterateBmm2Impl(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename RightBufType>
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateBmm1Impl(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                             RightBufType &inputRightBuf, bool notLastTwoLoop,
                                             RunInfo &runInfoNext, RunInfo &runInfo, ConstInfo &constInfo)
{
    WaitFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KMatmul1BufId);
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % 3;
    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);

    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                     static_cast<uint32_t>(constInfo.dSize >> 1),   // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    uint32_t curL1QBufId = l1QBufId;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }

    // m,n不切，k切128，mm1B直接用tensor的数据
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
        inputRightBuf.template GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);

    curL1QBufId = (curL1QBufId + 1) % 3;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;

    // m,n不切，k切128, mm1B直接用tensor的数据
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
        inputRightBuf.template GetTensor<Q_T>()[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + l1QBufId);
        SetFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + curL1QBufId);
        l1QBufId = (l1QBufId + 2) % 3;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    SetFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    WaitFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) //
    // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dstStride = s2BaseSize;
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
template <typename RightBufType>
__aicore__ inline void
CSABlockCube<TEMPLATE_ARGS>::IterateBmm2Impl(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
                                             BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
                                             RightBufType &inputRightBuf, const RunInfo &runInfo,
                                             const ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffer = inputLeftBuffers.Get(); // P直接用无需搬运
    l1PBuffer.WaitCrossCore();

    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN1<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.template GetTensor<Q_T>(),
        inputRightBuf.template GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);

    SetFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    WaitFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KMatmul2BufId);
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % 3;

    outputBuf.WaitCrossCore(); //占用
    // L0C→UB;FixpipeParamsM300:L0C→UB
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.mSize = s1BaseSize;
    // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.srcStride = Align16Func(s1BaseSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], fixpipeParams); // 将matmul结果从L0C搬运到UB
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class CSABlockCubeDummy {
public:
    __aicore__ inline CSABlockCubeDummy(){};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> &l1BufferManager,
                                         __gm__ uint8_t *query)
    {
    }
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices,
                                         __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                         __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensQ,
                                         __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                         __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
                                         const ConstInfo &constInfo)
    {
    }
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS)                                                                     \
    TEMPLATES_DEF_NO_DEFAULT                                                                                           \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> {                                                          \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE)                                                                  \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST)                                                                \
    }

DEFINE_CUBE_BLOCK_TRAITS(CSABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(CSABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS                                                                                                    \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)                                                                       \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace SMLAKernel
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
