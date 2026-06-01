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
 * \file sparse_attn_sharedkv_scfa_block_cube.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H_
#define SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H_
#include "common/offset_calculator.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "sparse_attn_sharedkv_common_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace SASKernel {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

template <SAS_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SAS_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

template <SAS_LAYOUT KV_LAYOUT_T>
__aicore__ inline constexpr GmFormat GetKvGmFormat()
{
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BNBD) {
        return GmFormat::PA_BNBSND;
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        return GmFormat::TND;
    } else { // BSND
        return GmFormat::BSND;
    }
}

TEMPLATES_DEF
class SCFABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline SCFABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, const ConstInfo& constInfo);

    __aicore__ inline void MergeKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos);
    // SWA/CFA场景, inputRightBuf是NO_SYNC类型
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers, 
        Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);
    // SCFA场景, inputRightBuf是CROSS_CORE_SYNC_FORWARD类型
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers, 
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
        __gm__ uint8_t *cuSeqlensCmpKv, const ConstInfo& constInfo);
    __aicore__ inline void CalcS1Coord(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void CalcS2Coord(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<Q_T> inputRightTensor, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1CFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2CFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);

    __aicore__ inline void IterateBmm1SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
        const ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKvGmFormat<KV_LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT> queryGm;
    FaGmTensor<KV_T, KV_FORMAT> oriKvGm;
    FaGmTensor<KV_T, KV_FORMAT> cmpKvGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    FaGmTensor<KV_T, KV_FORMAT> curKvGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    /* =====================L1层 (外部L1K同步flag, 固定编码) ===================== */
    uint32_t l1KMte1ToMte2FlagId = 3;   // {3,4,5} 替代原 mte1ToMte2Id[3]
    uint32_t l1KMte2ToMte1FlagId = 3;   // {3,4,5} 替代原 mte2ToMte1Id[3]

    /* =====================LocalBuffer变量==================== */
    // L1 Q 矩阵 (3路轮转)
    TBuf<TPosition::A1> l1QBuf;
    LocalTensor<Q_T> l1QTensor;
    uint32_t l1QBufId = 0;
    uint32_t l1QMte1ToMte2FlagId = 0;   // {0,1,2}
    uint32_t l1QMte2ToMte1FlagId = 0;   // {0,1,2}

    // L0A / L0B (2路 ping-pong)
    TBuf<TPosition::A2> mmL0ABuf;
    TBuf<TPosition::B2> mmL0BBuf;
    LocalTensor<Q_T> mmL0ATensor;
    LocalTensor<Q_T> mmL0BTensor;

    // L0C (2路 ping-pong)
    TBuf<TPosition::CO1> mmL0CBuf;
    LocalTensor<T> mmL0CTensor;

    uint32_t l0ABBufId = 0;
    uint32_t l0CBufId = 0;
    uint32_t l0ABMToMte1FlagId = 3;     // {3,4}
    uint32_t l0ABMte1ToMFlagId = 3;     // {3,4}
    uint32_t l0CFixToMFlagId = 0;       // {0,1}
    uint32_t l0CMToFixFlagId = 0;       // {0,1}

    // K 三阶段流水
    uint32_t l1KLoadBufId = 0;
    uint32_t l1KMatmul1BufId = 0;
    uint32_t l1KMatmul2BufId = 0;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv,
    __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
    __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        this->oriKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)oriKv);
        if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BNBD) {
            this->oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
        }
        if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE && \
            TEMPLATE_MODE != SASTemplateMode::ORI_SCFA_TEMPLATE_MODE) {
            this->cmpKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)cmpKv);
            if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BNBD) {
                this->cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
            }
        }
        if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
            this->cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
        }
        InitGmTensor(cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitLocalBuffer() {
    // L1 Q: 3路 * 32KB = 96K (每路 dSize>>1=256, mRealSize=64, 64*256*2=32KB per slot)
    tPipe->InitBuffer(l1QBuf, BUFFER_SIZE_96K);
    l1QTensor = l1QBuf.Get<Q_T>();

    // L0A: 2路 * 16K = 32K
    tPipe->InitBuffer(mmL0ABuf, BUFFER_SIZE_32K);
    mmL0ATensor = mmL0ABuf.Get<Q_T>();

    // L0B: 2路 * 32K = 64K
    tPipe->InitBuffer(mmL0BBuf, BUFFER_SIZE_64K);
    mmL0BTensor = mmL0BBuf.Get<Q_T>();

    // L0C: 2路 * 128K = 256K
    tPipe->InitBuffer(mmL0CBuf, BUFFER_SIZE_256K);
    mmL0CTensor = mmL0CBuf.Get<T>();

    // 固定 flag 初始化
    SetFlag<HardEvent::M_MTE1>(3);
    SetFlag<HardEvent::M_MTE1>(4);                     // L0AB 初始可用 (base=3, range {3,4})
    SetFlag<HardEvent::FIX_M>(0);
    SetFlag<HardEvent::FIX_M>(1);                      // L0C 初始可用 (base=0, range {0,1})
    SetFlag<HardEvent::MTE1_MTE2>(0);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    SetFlag<HardEvent::MTE1_MTE2>(2);                  // L1Q 初始可写 (base=0, range {0,1,2})
    SetFlag<HardEvent::MTE1_MTE2>(3);
    SetFlag<HardEvent::MTE1_MTE2>(4);
    SetFlag<HardEvent::MTE1_MTE2>(5);                  // L1K 初始可写 (base=3, range {3,4,5})
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
    __gm__ uint8_t *cuSeqlensCmpKv, const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // SAS_LAYOUT::TND
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            cuSeqlensQGm, constInfo.actualSeqLenSize);
    }
    if constexpr (TEMPLATE_MODE != SASTemplateMode::ORI_SCFA_TEMPLATE_MODE && \
        TEMPLATE_MODE != SASTemplateMode::ORI_CMP_SCFA_TEMPLATE_MODE) {
        if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_BNBD) {
            this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.oriBlockSize, constInfo.dSize,
                this->oriBlockTableGm, constInfo.oriMaxBlockNumPerBatch);
            if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) {
                this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.cmpBlockSize, constInfo.dSize,
                    this->cmpBlockTableGm, constInfo.cmpMaxBlockNumPerBatch);
            }
        } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
            cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize,
                cuSeqlensOriKvGm, constInfo.actualSeqLenSize);
            if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) {
                cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
                this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize,
                    cuSeqlensCmpKvGm, constInfo.actualSeqLenSize);
            }
        } else {
            this->oriKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size,
                constInfo.s2Size, constInfo.dSize);
            if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) {
                this->cmpKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size,
                constInfo.s2Size / constInfo.cmpRatio, constInfo.dSize);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CalcS1Coord(const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    // 计算s1方向偏移
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * runInfo.qSNumInOneBlock;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CalcS2Coord(const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    // 计算s2方向偏移
    coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.boIdx;
    if (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) {
        kvCacheBlockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + \
            (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) * s2BaseSize;
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
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::MergeKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo, const ConstInfo &constInfo,
    int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) { // ori kv阶段无核间操作
            return;
        }
        if (runInfo.s2RealSize > startPos) {
            CalcS2Coord(runInfo, constInfo);
            ProcessSparseKv(outputL1, runInfo, constInfo, startPos);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, const ConstInfo &constInfo, int32_t startPos)
{
    int64_t calNum = (runInfo.s2RealSize - startPos > 16) ? 16 : runInfo.s2RealSize - startPos; // 单次最多处理16行s2
    int64_t s2Loops = CeilDiv(calNum, 2); // 每次循环处理2个topk

    bool meetEnd = false;
    int64_t s2 = startPos;
    int64_t s2End = s2Loops * 2 + startPos;
    int64_t token0Idx, token1Idx; // 拷贝进入的两个token的index
    // 处理一个s2的base块
    int64_t dealRow = startPos;
    while ((s2 < s2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        // 1、copy kv in, gm ->ub
        LocalTensor<Q_T> kvInL1 = outputL1.GetTensor<Q_T>();
        GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
        s2 += 2; // 每次搬运2行
        if (token0Idx== -1 && token1Idx == -1) {
            meetEnd = true;
            break;
        }
        dealRow += CopyInKvSparse(kvInL1, dealRow, token0Idx, token1Idx, runInfo, constInfo);
        if (token1Idx == -1) {
            meetEnd = true;
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, const ConstInfo &constInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.cmpSparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.cmpSparseBlockCount +
            runInfo.s1oIdx * constInfo.cmpSparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx;
    int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= constInfo.cmpSparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = cmpSparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
    topkKIdx += 1;
    if (unlikely(topkKIdx >= constInfo.cmpSparseBlockCount)) {
        token1Idx = -1;
    } else {
        token1Idx = cmpSparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t SCFABlockCube<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<Q_T> inputRightTensor, int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, const ConstInfo &constInfo)
{
    if constexpr (IS_PA) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = token0Idx;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = 1;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        LocalTensor<Q_T> l1Tensor = inputRightTensor[startRow * 16];
        GmCopyInToL1PA<KV_T>(l1Tensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
        if (token1Idx >= 0) {
            startPos.s2Offset = token1Idx;
            l1Tensor = inputRightTensor[(startRow + 1) * 16];
            GmCopyInToL1PA<KV_T>(l1Tensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
        }
    }
    return (token0Idx >= 0) + (token1Idx >= 0);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
        runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufId + i) % 3;
        WaitFlag<HardEvent::MTE1_MTE2>(l1QMte1ToMte2FlagId + curL1QBufId);
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QTensor[curL1QBufId * BUFFER_SIZE_16K], this->queryGm.gmTensor[curGmOffset],
            runInfo.mRealSize, constInfo.dSize >> 1, constInfo.mm1Ka);
        SetFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    CalcS1Coord(runInfo, constInfo);
    CalcS2Coord(runInfo, constInfo);
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KLoadBufId);
    LocalTensor<KV_T> inputRightTensor = inputRightBuf.GetTensor<KV_T>();
    if constexpr (IS_PA) {
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
        GmCopyInToL1PA<KV_T>(inputRightTensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
    } else {
        int64_t keyOffset = this->curKvGm.offsetCalculator.GetOffset(
            coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<KV_T>(inputRightTensor, curKvGm.gmTensor[keyOffset], runInfo.s2RealSize,
                            constInfo.dSize, constInfo.mm1Kb);
    }
    SetFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KLoadBufId);
    l1KLoadBufId = (l1KLoadBufId + 1) % 3;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1CFA(outputBuf, inputRightBuf, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    IterateBmm2CFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1CFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    // BMM1: Q前半×K前半 + Q后半×K后半
    WaitFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KMatmul1BufId);
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % 3;
    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);

    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                        static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                        static_cast<uint32_t>(constInfo.dSize >> 1),   // singleK (first half)
                        0,    // isLeftTranspose
                        1     // isRightTranspose
                    };
    uint32_t curL1QBufId = l1QBufId;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K], inputRightBuf.GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);
    curL1QBufId = (curL1QBufId + 1) % 3;

    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K], inputRightBuf.GetTensor<Q_T>()[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
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
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2CFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::NO_SYNC> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffer = inputLeftBuffers.Get();
    l1PBuffer.WaitCrossCore();

    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN1<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.GetTensor<Q_T>(),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);

    // BMM2完成后释放 K
    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KMatmul2BufId);
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % 3;

    SetFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    WaitFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    fixpipeParams.mSize = s1BaseSize;
    fixpipeParams.srcStride = Align16Func(s1BaseSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    CalcS1Coord(runInfo, constInfo);
    CalcS2Coord(runInfo, constInfo);
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KLoadBufId);
    if constexpr (TEMPLATE_MODE == SASTemplateMode::ORI_SCFA_TEMPLATE_MODE || \
        TEMPLATE_MODE == SASTemplateMode::ORI_CMP_SCFA_TEMPLATE_MODE) {
        LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
        v0ResGm.WaitCrossCore();
        GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
        CopyToL1Nd2Nz<Q_T>(dst, v0ResGmTensor, runInfo.s2RealSize, constInfo.dSize, constInfo.mm1Kb);
    } else {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) {
            LocalTensor<KV_T> inputRightTensor = inputRightBuf.GetTensor<KV_T>();
            if constexpr (IS_PA) {
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
                GmCopyInToL1PA<KV_T>(inputRightTensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
            } else {
                int64_t keyOffset = this->curKvGm.offsetCalculator.GetOffset(
                    coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
                CopyToL1Nd2Nz<KV_T>(inputRightTensor, curKvGm.gmTensor[keyOffset], runInfo.s2RealSize,
                                    constInfo.dSize, constInfo.mm1Kb);
            }
        } else {
            LocalTensor<Q_T> dst = inputRightBuf.GetTensor<Q_T>();
            v0ResGm.WaitCrossCore();
            GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
            CopyToL1Nd2Nz<Q_T>(dst, v0ResGmTensor, runInfo.s2RealSize, constInfo.dSize, constInfo.mm1Kb);
        }
    }
    SetFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KLoadBufId);
    l1KLoadBufId = (l1KLoadBufId + 1) % 3;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1SCFA(outputBuf, inputRightBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    IterateBmm2SCFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    // BMM1: Q前半×K前半 + Q后半×K后半
    WaitFlag<HardEvent::MTE2_MTE1>(l1KMte2ToMte1FlagId + l1KMatmul1BufId);
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % 3;
    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);

    uint32_t curL1QBufId = l1QBufId;
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                        static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                        static_cast<uint32_t>(constInfo.dSize >> 1),   // singleK (first half)
                        0,    // isLeftTranspose
                        1     // isRightTranspose
                    };
    if (unlikely(runInfo.s2LoopCount == 0)) {    
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K], inputRightBuf.GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);

    curL1QBufId = (curL1QBufId + 1) % 3;
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(l1QMte2ToMte1FlagId + curL1QBufId);
    }
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K], inputRightBuf.GetTensor<Q_T>()[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
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
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = s2BaseSize;
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, const RunInfo &runInfo,
    const ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> inputLeftBuf = inputLeftBuffers.Get();
    inputLeftBuf.WaitCrossCore();

    WaitFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN1<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufId],
        param, l0ABMToMte1FlagId, l0ABMte1ToMFlagId, l0ABBufId);

    SetFlag<HardEvent::MTE1_MTE2>(l1KMte1ToMte2FlagId + l1KMatmul2BufId);
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % 3;

    SetFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);
    WaitFlag<HardEvent::M_FIX>(l0CMToFixFlagId + l0CBufId);

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    fixpipeParams.mSize = s1BaseSize;
    fixpipeParams.srcStride = Align16Func(s1BaseSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mmL0CTensor[BUFFER_SIZE_32K * l0CBufId], fixpipeParams);
    SetFlag<HardEvent::FIX_M>(l0CFixToMFlagId + l0CBufId);
    l0CBufId ^= 1;

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class SCFABlockCubeDummy {
public:
    __aicore__ inline SCFABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices,
        __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
        __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, const ConstInfo& constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
