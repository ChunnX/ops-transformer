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
 * \file sparse_flash_mla_csa_block_vector.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
#define SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using AscendC::Reg::StoreDist;

#include "common/flash_decode.h"

#if __has_include("../../common/op_kernel/arch35/vf/vf_flash_decode.h")
#include "../../common/op_kernel/arch35/vf/vf_flash_decode.h"
#else
#include "../common/arch35/vf/vf_flash_decode.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h")
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#else
#include "../../common/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h"
#endif

#if __has_include("../../common/op_kernel/arch35/vf/vf_flashupdate_new.h")
#include "../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#else
#include "../../common/arch35/vf/vf_flashupdate_new.h"
#endif

#if __has_include("../../common/op_kernel/buffers_policy.h")
#include "../../common/op_kernel/buffers_policy.h"
#else
#include "../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/buffer_manager.h")
#include "../../common/op_kernel/buffer_manager.h"
#else
#include "../common/buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/buffer.h")
#include "../../common/op_kernel/buffer.h"
#else
#include "../common/buffer.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;
using AttentionCommon::FdRunInfo;

namespace SMLAKernel {
TEMPLATES_DEF
class CSABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr float R0 = 1.0f;

    // ==================== Functions ======================
    __aicore__ inline CSABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (cuSeqlensOriKv != nullptr) {
                cuSeqlensOriKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensOriKv);
            }
            if (cuSeqlensCmpKv != nullptr) {
                cuSeqlensCmpKvGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensCmpKv);
            }
            if (seqUsedOriKV != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedOriKV);
            }
            if (seqUsedCmpKV != nullptr) {
                actualSeqLengthsCmpKVGm.SetGlobalBuffer((__gm__ int32_t *)seqUsedCmpKV);
            }
            if (cmpResidualKV != nullptr) {
                cmpResidualKVGm.SetGlobalBuffer((__gm__ int32_t *)cmpResidualKV);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void InitS2SplitStaging(__gm__ uint8_t *base)
    {
        s2SplitStagingBase = base;
        stagingOutGm.SetGlobalBuffer((__gm__ float *)base);
    }
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo);
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetKVPhyAddr(
        uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t lastS1EndIdx,
        bool hasActualSeqQlen, bool hasCuSeqlensQ, GlobalTensor<int32_t> actualSeqQlenGm,
        bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv, GlobalTensor<int32_t> actualSeqOriKvlenGm,
        GlobalTensor<int32_t> cuSeqlensOriKvGm, GlobalTensor<int32_t> oriTopkLengthGm,
        bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv, GlobalTensor<int32_t> actualSeqCmpKvlenGm,
        GlobalTensor<int32_t> cuSeqlensCmpKvGm, GlobalTensor<int32_t> cmpTopkLengthGm,
        GlobalTensor<int32_t> cmpResidualKvGm, __gm__ uint8_t *phyAddrWorkspace, ConstInfo &constInfo);

private:
    __aicore__ inline uint32_t GetStagingSlotNum() const
    {
        if constexpr (IS_SPLIT_G) {
            return AttentionCommon::FD_MAX_S2_SPLIT_NUM * (GetBlockNum() >> 1U);
        } else {
            return AttentionCommon::FD_MAX_S2_SPLIT_NUM * GetBlockNum();
        }
    }

    __aicore__ inline uint32_t GetFaStagingWorkspaceIdx(const RunInfo &runInfo) const
    {
        return static_cast<uint32_t>(runInfo.firstFdDataWorkspaceIdx + runInfo.s2SplitIdx);
    }

    __aicore__ inline int64_t GetFaStagingMOffset(const RunInfo &runInfo, const ConstInfo &constInfo) const
    {
        int64_t stagingMOffset = (constInfo.subBlockIdx == 1) ? static_cast<int64_t>(runInfo.firstHalfMRealSize) : 0L;
        if constexpr (IS_SPLIT_G) {
            stagingMOffset += static_cast<int64_t>(runInfo.goIdx);
        }
        return stagingMOffset;
    }

    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                           Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos);
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealS2Addr(int64_t &addr0, int64_t &addr1, int64_t s2IdxInBase,
                                         const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
                                              int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyToOutUb(LocalTensor<Q_T> kvNzUb, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                       ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                          LocalTensor<Q_T> kvNzOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset,
                                          ConstInfo &constInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);
    __aicore__ inline int32_t GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                        GlobalTensor<int32_t> &actualSeqGm, GlobalTensor<int32_t> &cuSeqlensGm,
                                        int64_t defaultSize);
    __aicore__ inline int32_t CalcCurValidS2(
        uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv, bool hasActualSeqKvlen,
        bool hasCuSeqlensKv, GlobalTensor<int32_t> &actualSeqKvlenGm, GlobalTensor<int32_t> &cuSeqlensKvGm,
        GlobalTensor<int32_t> &topkLengthGm, GlobalTensor<int32_t> &cmpResidualKvGm, ConstInfo &constInfo,
        int32_t sparseBlockCount);
    __aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                           int64_t validS2, GlobalTensor<uint32_t> &phyAddrGm,
                                           uint32_t alignedSparseBlockCount);
    __aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                           GlobalTensor<int32_t> &blockTableGm, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                             int64_t validS2, GlobalTensor<int32_t> &sparseIndicesGm,
                                             uint32_t sparseBlockCount);
    __aicore__ inline void GetKVPhyAddrForKvType(
        uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t lastS1EndIdx,
        bool hasActualSeqQlen, bool hasCuSeqlensQ, GlobalTensor<int32_t> actualSeqQlenGm,
        bool hasActualSeqKvlen, bool hasCuSeqlensKv, GlobalTensor<int32_t> actualSeqKvlenGm,
        GlobalTensor<int32_t> cuSeqlensKvGm, GlobalTensor<int32_t> topkLengthGm,
        GlobalTensor<int32_t> cmpResidualKvGm, ConstInfo &constInfo, GlobalTensor<int32_t> &blockTableGm,
        GlobalTensor<int32_t> &sparseIndicesGm, GlobalTensor<uint32_t> &phyAddrGm, uint32_t kvStride,
        uint32_t blockSize, uint32_t maxBlockNumPerBatch, uint32_t sparseBlockCount,
        uint32_t alignedSparseBlockCount, bool isOriKv);

    TPipe *tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cuSeqlensKvGm;
    GlobalTensor<int32_t> oriSparseIndicesGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> cuSeqlensOriKvGm;
    GlobalTensor<int32_t> cuSeqlensCmpKvGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> actualSeqLengthsCmpKVGm;
    GlobalTensor<int32_t> cmpResidualKVGm;
    GlobalTensor<uint32_t> oriKvPhyAddrGm;
    GlobalTensor<uint32_t> cmpKvPhyAddrGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TQue<QuePosition::VECIN, 2> stage0InQue;      // for v0 input, 2份表示可能存在pingpong
    TQue<QuePosition::VECOUT, 1> stage0OutQue;    // for v0 output, 2份表示可能存在pingpong
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2];     // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2];     // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TEventID mte3ToVAttnOutId; // 存放MTE3_V的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID vToMte3AttnOutId; // 存放V_MTE3的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID mte3ToVLseOutId;  // 存放MTE3_V的eventId, 用于V1 LSE拷出阶段的同步
    TEventID vToMte3LseOutId;  // 存放V_MTE3的eventId, 用于V1 LSE拷出阶段的同步
    TEventID mte2ToMte3[2];
    TEventID mte3ToMte2[2];
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> outLseBuf[2];
    TBuf<> vselrIndexesBuf[2];
    AttentionCommon::FdBuffers<TBuf<>> fdBuffers;
    __gm__ uint8_t *s2SplitStagingBase = nullptr;
    GlobalTensor<float> stagingOutGm;
    TEventID mte2ToVV0Id[2];
    TEventID vToMte2V0Id[2];

    T negativeFloatScalar;
    bool isSinks = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;

    TEventID initOutputEventId; // attenOut和lse，刷无效行会用到剩余ub，需要加同步
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
                                                                   int64_t s2IdxInBase, const RunInfo &runInfo,
                                                                   ConstInfo &constInfo)
{
    int64_t sparseBlockCount = 0;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    // CSA、ORI_SPARSE、ORI_CMP_SPARSE均可通过runInfo.isCmp判断
    if (runInfo.isCmp) {
        sparseBlockCount = constInfo.cmpSparseBlockCount;
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
    } else {
        sparseBlockCount = constInfo.oriSparseBlockCount;
    }

    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx +=
            runInfo.boIdx * constInfo.s1Size * sparseBlockCount + runInfo.s1oIdx * sparseBlockCount; // B, S1, N2(1), K
    }

    int64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx + runInfo.s2StartIdx);
    }
    topkKIdx += 1;
    if (unlikely((topkKIdx >= sparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = sparseIndicesGm.GetValue(topkBS1Idx + topkKIdx + runInfo.s2StartIdx);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetRealS2Addr(int64_t &addr0, int64_t &addr1,
                                                                 int64_t s2IdxInBase, const RunInfo &runInfo,
                                                                 ConstInfo &constInfo)
{
    const uint32_t sparseBlockCount =
        runInfo.isCmp ? constInfo.cmpSparseBlockCount : constInfo.oriSparseBlockCount;
    const uint32_t alignedSparseBlockCount =
        runInfo.isCmp ? constInfo.alignedCmpSparseBlockCount : constInfo.alignedOriSparseBlockCount;
    int64_t curS2LoopCnt = runInfo.s2LoopCount;
    GlobalTensor<int64_t> phyAddrGm;
    if (runInfo.isCmp) {
        curS2LoopCnt -= runInfo.oriKvLoopEndIdx;
        phyAddrGm = cmpKvPhyAddrGm.template ReinterpretCast<int64_t>();
    } else {
        phyAddrGm = oriKvPhyAddrGm.template ReinterpretCast<int64_t>();
    }

    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        topkBS1Idx = (cuSeqlensQGm.GetValue(runInfo.boIdx) + runInfo.s1oIdx) * alignedSparseBlockCount;
    } else {
        topkBS1Idx = (runInfo.boIdx * constInfo.s1Size + runInfo.s1oIdx) * alignedSparseBlockCount;
    }
    const uint64_t topkKIdx = s2IdxInBase + curS2LoopCnt * constInfo.s2BaseSize;
    const uint64_t addrIdx = topkBS1Idx + runInfo.s2StartIdx + topkKIdx;
    addr0 = topkKIdx < sparseBlockCount ? phyAddrGm.GetValue(addrIdx) : -1;
    addr1 = (topkKIdx + 1 < sparseBlockCount) && (s2IdxInBase + 1 < sparseS2End) ?
                phyAddrGm.GetValue(addrIdx + 1) :
                -1;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t CSABlockVec<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo,
                                                                   ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKeyStride0 : constInfo.oriKeyStride0;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                        blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        if (runInfo.isCmp) {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.cmpS2Size * constInfo.dSize +
                            runInfo.n2oIdx * constInfo.cmpS2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
        } else {
            realkeyOffset = runInfo.boIdx * constInfo.n2Size * constInfo.s2Size * constInfo.dSize +
                            runInfo.n2oIdx * constInfo.s2Size * constInfo.dSize + s2Idx * constInfo.dSize; // BSN(1)D
        }
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        realkeyOffset = (cuSeqlensKvGm.GetValue(runInfo.boIdx) + s2Idx) * constInfo.n2Size * constInfo.dSize +
                        runInfo.n2oIdx * constInfo.dSize; // TN(1)D
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                  int64_t keyOffset, ConstInfo &constInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = constInfo.dSize * sizeof(KV_T);

    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding =
        (CeilAlign(constInfo.dSize * sizeof(KV_T), BUFFER_SIZE_BYTE_32B) - constInfo.dSize * sizeof(KV_T)) /
        sizeof(KV_T);
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t CSABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                      int64_t token0Idx, int64_t token1Idx,
                                                                      const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0;
    int64_t keyOffset1;
    if constexpr (IS_VEC_S2PHYADDR) {
        keyOffset0 = token0Idx;
        keyOffset1 = token1Idx;
    } else {
        keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
        keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    }
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t combineBytes = constInfo.dSizeVInput * sizeof(KV_T);
    int64_t keySrcStride =
        (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) * sizeof(KV_T) - combineBytes;
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
    } else if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, constInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, constInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = (CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) - combineBytes) / sizeof(KV_T);
        padParams.paddingValue = 0;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * constInfo.dSize], keyGm[keyOffset], intriParams, padParams);
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyToOutUb(LocalTensor<Q_T> kvOutUb, LocalTensor<KV_T> srcTensor,
                                                               int64_t dealRow, ConstInfo &constInfo)
{
    LocalTensor<Q_T> kvNdUb = srcTensor.template ReinterpretCast<Q_T>();
    DataCopy(kvOutUb, kvNdUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                           LocalTensor<Q_T> kvNzOutUb, int64_t dealRow, int64_t s2StartIdx,
                                           const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.dSize / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * 16], kvNzOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx,
                                           const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * constInfo.dSize], kvOutUb, dealRow * constInfo.dSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        int32_t vecCnt = (aicIdx % 2U == 0) ? (GetSubBlockIdx() == 0 ? 0 : 1) : (GetSubBlockIdx() == 0 ? 2 : 3);
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = 0;
            } else {
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        sparseCalSize = GetSubBlockIdx() == 0 ? v0S2SizeFirstCore : runInfo.s2RealSize - v0S2SizeFirstCore;
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : v0S2SizeFirstCore;
        sparseS2End = sparseS2Start + sparseCalSize;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                        const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (runInfo.s2LoopCount < runInfo.oriKvLoopEndIdx) {
            if constexpr (IS_SPLIT_G) {
                CrossCoreSetFlag<0, PIPE_MTE3>(15);
                CrossCoreWaitFlag<0, PIPE_MTE3>(15);
            }
            return;
        }
        keyGm = cmpKVGm;
        cuSeqlensKvGm = cuSeqlensCmpKvGm;
        sparseIndicesGm = cmpSparseIndicesGm;
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            blockTableGm = cmpBlockTableGm;
            blockSize = constInfo.cmpBlockSize;
            maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE3>(15);
            CrossCoreWaitFlag<0, PIPE_MTE3>(15);
        }
        outputL1.SetCrossCore();
        v0ResGm.SetCrossCore();
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                         TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (!runInfo.isCmp) {
            keyGm = oriKVGm;
            cuSeqlensKvGm = cuSeqlensOriKvGm;
            sparseIndicesGm = oriSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = oriBlockTableGm;
                blockSize = constInfo.oriBlockSize;
                maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
            }
        } else {
            keyGm = cmpKVGm;
            cuSeqlensKvGm = cuSeqlensCmpKvGm;
            sparseIndicesGm = cmpSparseIndicesGm;
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                blockTableGm = cmpBlockTableGm;
                blockSize = constInfo.cmpBlockSize;
                maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
            }
        }
        CalSparseCalSize(runInfo, constInfo);
        ProcessSparseKv(outputL1, v0ResGm, runInfo, constInfo, startPos);
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE3>(15);
            CrossCoreWaitFlag<0, PIPE_MTE3>(15);
        }
        outputL1.SetCrossCore();
        v0ResGm.SetCrossCore();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
                                            Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                            const RunInfo &runInfo, ConstInfo &constInfo, int32_t startPos)
{
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (sparseCalSize == 0) {
            return;
        }
        bool meetEnd = false;
        int64_t s2Start = sparseS2Start;
        int64_t s2 = sparseS2Start;
        int64_t token0Idx;
        int64_t token1Idx;
        uint32_t pingPong = 0;
        while ((s2 < sparseS2End) && !meetEnd) {
            int64_t dealRow = 0;
            LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) {
                if constexpr (IS_VEC_S2PHYADDR) {
                    GetRealS2Addr(token0Idx, token1Idx, s2, runInfo, constInfo);
                } else {
                    GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
                }
                s2 += 2;
                if (token0Idx == -1 && token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
                dealRow += CopyInKvSparse(stage0OutUb, dealRow, token0Idx, token1Idx, runInfo, constInfo);
                if (token1Idx == -1) {
                    meetEnd = true;
                    break;
                }
            }
            if (dealRow == 0) {
                SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
                pingPong ^= 1;
                return;
            }
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
            CopyOutKvUb2Gm(v0ResGm, stage0OutUb, dealRow, s2Start, runInfo, constInfo);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            s2Start += dealRow;
            pingPong ^= 1;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                        RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    runInfo.s2RealSizeUpdate = runInfo.s2RealSize;

    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (runInfo.s2LoopCount == 0 && !isSinks) {
        if (likely(runInfo.s2RealSize == 128 &&
                   runInfo.s2RealSizeUpdate == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128 ||
                   runInfo.s2RealSizeUpdate < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (runInfo.s2LoopCount == 0 && isSinks) {
            if (!runInfo.isS2Split || runInfo.isFirstS2SplitCore) {
                // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
                int64_t sinksOffset = 0;
                if constexpr (!IS_SPLIT_G) {
                    sinksOffset = GetBlockIdx() % 2 == 0 ? 0 : runInfo.firstHalfMRealSize;
                } else {
                    sinksOffset = runInfo.goIdx;
                    if (constInfo.subBlockIdx == 1) {
                        sinksOffset += runInfo.firstHalfMRealSize;
                    }
                }
                LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
                InitSoftmaxFromSinks<T>(sumUb, maxUb, sinksUb, sinksOffset, R0, runInfo.halfMRealSize);
            } else {
                // FD非首段split: 初始化maxUb=极小值, sumUb=0（不含sinks），走update模式等价于init
                Duplicate(maxUb, this->negativeFloatScalar, runInfo.halfMRealSize);
                Duplicate(sumUb, static_cast<T>(0), runInfo.halfMRealSize);
            }
        }
        if (likely(runInfo.s2RealSize == 128 &&
                   runInfo.s2RealSizeUpdate == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize < 128 ||
                   runInfo.s2RealSizeUpdate < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, vselrIndexesBuf, runInfo.halfMRealSize,
                runInfo.s2RealSizeUpdate, static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) *
                              (runInfo.mRealSize - runInfo.halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize, (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
                  (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    // ======================================================
    if (runInfo.s2LoopCount != 0 || (runInfo.s2LoopCount == 0 && isSinks)) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
    if (runInfo.isS2Split && runInfo.halfMRealSize > 0 && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(),
                                                                 AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                                 AttentionCommon::FD_REDUCE_CHUNK_ROWS};
        uint32_t workspaceIdx = GetFaStagingWorkspaceIdx(runInfo);
        int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
        LocalTensor<float> tmpUb = this->stage2OutBuf.template Get<float>();
        AttentionCommon::StageVec1Lse(stagingLayout, s2SplitStagingBase, workspaceIdx, stagingMOffset,
                                      static_cast<uint32_t>(runInfo.halfMRealSize), maxUb, sumUb, tmpUb, vToMte3Id[1],
                                      mte3ToVId[1]);
    }
    if (constInfo.returnSoftmaxLse && runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        LocalTensor<float> outLse = this->outLseBuf[runInfo.multiCoreIdxMod2].template Get<float>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(float) * runInfo.halfMRealSize;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
        ComputeLse<float>(outLse, sumUb, maxUb, runInfo.halfMRealSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
        DataCopyPad(this->softmaxLseGm[runInfo.softmaxLseOffset], outLse, dataCopyParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void
CSABlockVec<TEMPLATE_ARGS>::ProcessVec2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                        RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;
    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        if (runInfo.s2RealSizeUpdate > 0) {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
            if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, runInfo.vec2MRealSize, dTemplateAlign64, 1.0,
                    1.0);
            }
        } else {
            if (runInfo.s2LoopCount >= runInfo.s2LoopLimit) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb,
                                                                      runInfo.vec2MRealSize, dTemplateAlign64, 1.0);
            }
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize,
                                                                  dTemplateAlign64, 1.0);
        }

        if (runInfo.isS2Split) {
            AttentionCommon::S2SplitFdStagingLayout stagingLayout = {
                constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(), AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                AttentionCommon::FD_REDUCE_CHUNK_ROWS};
            uint32_t workspaceIdx = GetFaStagingWorkspaceIdx(runInfo);
            int64_t stagingMOffset = GetFaStagingMOffset(runInfo, constInfo);
            AttentionCommon::StageVec2PartialO<T>(
                stagingLayout, stagingOutGm, workspaceIdx, stagingMOffset, static_cast<uint32_t>(runInfo.vec2MRealSize),
                static_cast<uint32_t>(constInfo.dSizeV), vec2ResUb, vToMte3AttnOutId, mte3ToVAttnOutId);
        } else {
            this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitFDBuffers(FdRunInfo &fdRunInfo)
{
    AttentionCommon::InitFDBuffers<T, dTemplateAlign64>(fdRunInfo, this->tPipe, fdBuffers);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo)
{
    InitFDBuffers(fdRunInfo);
    int64_t vec2CalcSize = fdRunInfo.mNum * dTemplateAlign64;
    int64_t seqOffset = 0;
    if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        seqOffset = this->cuSeqlensQGm.GetValue(fdRunInfo.bn2Idx);
    } else {
        seqOffset = fdRunInfo.bn2Idx * constInfo.s1Size;
    }
    int64_t attentionOutOffset =
        seqOffset * constInfo.n2GDv + fdRunInfo.mIdx * constInfo.n2GDv + fdRunInfo.mStartIdx * constInfo.dSizeV;
    LocalTensor<T> accumulatedO = this->fdBuffers.accumOut.template Get<T>();
    LocalTensor<float> lseExpUb = this->fdBuffers.lseExp.template Get<float>();
    LocalTensor<float> blockMaxUb = this->fdBuffers.blockMax.template Get<float>();
    LocalTensor<float> blockSumUb = this->fdBuffers.blockSum.template Get<float>();
    LocalTensor<T> partialOFp32 = this->fdBuffers.partialO.template Get<T>();
    AttentionCommon::S2SplitFdStagingLayout stagingLayout = {constInfo.gSize, dTemplateAlign64, GetStagingSlotNum(),
                                                             AttentionCommon::FD_BROADCAST_ELEMS_PER_ROW,
                                                             AttentionCommon::FD_REDUCE_CHUNK_ROWS};
    AttentionCommon::Reduce<T, dTemplateAlign64>(
        stagingLayout, s2SplitStagingBase, fdRunInfo.workspaceIdx, fdRunInfo.workspaceNum,
        static_cast<uint32_t>(fdRunInfo.mStartIdx), fdRunInfo.mNum, static_cast<uint32_t>(constInfo.dSizeV),
        accumulatedO, lseExpUb, blockMaxUb, blockSumUb, partialOFp32, vToMte2V0Id[0], vToMte2V0Id[1], mte2ToVV0Id[0]);
    RunInfo runInfo;
    runInfo.vec2MRealSize = fdRunInfo.mNum;
    runInfo.attentionOutOffset = attentionOutOffset;
    this->CopyOutAttentionOut(runInfo, constInfo, accumulatedO, 0, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                   LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                   int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                       LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                       int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    if (constInfo.returnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = constInfo.bSize * constInfo.n2Size * constInfo.s1Size * constInfo.gSize;
        } else if constexpr (LAYOUT_T == SMLA_LAYOUT::TND) {
            totalReturnSoftmaxSize = constInfo.n2Size * constInfo.s1Size * constInfo.gSize; // (N2,T1,G)
        }
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            uint64_t singleCoreSoftmaxSize = (totalReturnSoftmaxSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailSoftmaxSize = totalReturnSoftmaxSize - constInfo.aivIdx * singleCoreSoftmaxSize;
            uint64_t singleInitSoftmaxSize =
                tailSoftmaxSize < singleCoreSoftmaxSize ? tailSoftmaxSize : singleCoreSoftmaxSize;
            if (constInfo.aivIdx * singleCoreSoftmaxSize < totalReturnSoftmaxSize && singleInitSoftmaxSize > 0) {
                WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
                matmul::InitOutput<float>(this->softmaxLseGm[constInfo.aivIdx * singleCoreSoftmaxSize],
                                          singleInitSoftmaxSize, 0);
                SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            }
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                               ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxLseGm.SetGlobalBuffer((__gm__ T *)softmaxLse);
        if (constInfo.needInit == 1) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore(constInfo);
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                    __ubuf__ int32_t *blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                    const uint32_t blockSize, const int16_t shiftRightNum,
                                    const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t S2_NUM_PER_LOOP = 128;
    static const uint16_t S2_NUM_PER_REG = 64;
    static const uint16_t OUT_OFFSET_PER_LOOP = 256;
    static const uint16_t OUT_OFFSET_PER_REG = 128;
    static const uint32_t INVALID_VALUE = 0xFFFFFFFFU;
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg addCarryLow1;
    MicroAPI::MaskReg addCarryHigh1;
    MicroAPI::MaskReg addCarryLow2;
    MicroAPI::MaskReg addCarryHigh2;
    MicroAPI::MaskReg pregTailNeg1;
    MicroAPI::MaskReg pregTailNeg2;

    MicroAPI::RegTensor<uint32_t> kvStrideReg;
    MicroAPI::RegTensor<uint32_t> sparseIdx1;
    MicroAPI::RegTensor<uint32_t> sparseIdx2;
    MicroAPI::RegTensor<uint32_t> paBlockIdx1;
    MicroAPI::RegTensor<uint32_t> paBlockIdx2;
    MicroAPI::RegTensor<uint32_t> paBlockBase1;
    MicroAPI::RegTensor<uint32_t> paBlockBase2;
    MicroAPI::RegTensor<uint32_t> paOffset1;
    MicroAPI::RegTensor<uint32_t> paOffset2;
    MicroAPI::RegTensor<uint32_t> phyOffset1;
    MicroAPI::RegTensor<uint32_t> phyOffset2;
    MicroAPI::RegTensor<uint32_t> phyBlockIdx1;
    MicroAPI::RegTensor<uint32_t> phyBlockIdx2;
    MicroAPI::RegTensor<uint32_t> blockOffsetLow1;
    MicroAPI::RegTensor<uint32_t> blockOffsetHigh1;
    MicroAPI::RegTensor<uint32_t> totalOffsetLow1;
    MicroAPI::RegTensor<uint32_t> totalOffsetHigh1;
    MicroAPI::RegTensor<uint32_t> blockOffsetLow2;
    MicroAPI::RegTensor<uint32_t> blockOffsetHigh2;
    MicroAPI::RegTensor<uint32_t> totalOffsetLow2;
    MicroAPI::RegTensor<uint32_t> totalOffsetHigh2;
    MicroAPI::RegTensor<uint32_t> zeroReg;
    MicroAPI::Duplicate(zeroReg, 0U);
    MicroAPI::Duplicate(kvStrideReg, kvStride);

    for (; s2Loop > 1U;) {
        for (uint16_t i = 0; i < s2Loop - 1U; ++i) {
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t> &)sparseIdx1, sparseIdxUb + i * S2_NUM_PER_LOOP);
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t> &)sparseIdx2,
                sparseIdxUb + S2_NUM_PER_REG + i * S2_NUM_PER_LOOP);
            MicroAPI::Muls(sparseIdx1, sparseIdx1, sparseBlockSize, pregAll);
            MicroAPI::Muls(sparseIdx2, sparseIdx2, sparseBlockSize, pregAll);
            MicroAPI::ShiftRights(paBlockIdx1, sparseIdx1, shiftRightNum, pregAll);
            MicroAPI::ShiftRights(paBlockIdx2, sparseIdx2, shiftRightNum, pregAll);
            MicroAPI::Muls(paBlockBase1, paBlockIdx1, blockSize, pregAll);
            MicroAPI::Muls(paBlockBase2, paBlockIdx2, blockSize, pregAll);
            MicroAPI::Sub(paOffset1, sparseIdx1, paBlockBase1, pregAll);
            MicroAPI::Sub(paOffset2, sparseIdx2, paBlockBase2, pregAll);
            MicroAPI::Muls(phyOffset1, paOffset1, kvDim, pregAll);
            MicroAPI::Muls(phyOffset2, paOffset2, kvDim, pregAll);
            DataCopyGather(phyBlockIdx1, blkTableUb, paBlockIdx1, pregAll);
            DataCopyGather(phyBlockIdx2, blkTableUb, paBlockIdx2, pregAll);
            MicroAPI::Mull(blockOffsetLow1, blockOffsetHigh1, phyBlockIdx1, kvStrideReg, pregAll);
            MicroAPI::Mull(blockOffsetLow2, blockOffsetHigh2, phyBlockIdx2, kvStrideReg, pregAll);
            MicroAPI::Add(addCarryLow1, totalOffsetLow1, blockOffsetLow1, phyOffset1, pregAll);
            MicroAPI::Add(addCarryLow2, totalOffsetLow2, blockOffsetLow2, phyOffset2, pregAll);
            MicroAPI::AddC(addCarryHigh1, totalOffsetHigh1, blockOffsetHigh1, zeroReg, addCarryLow1, pregAll);
            MicroAPI::AddC(addCarryHigh2, totalOffsetHigh2, blockOffsetHigh2, zeroReg, addCarryLow2, pregAll);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * OUT_OFFSET_PER_LOOP, totalOffsetLow1, totalOffsetHigh1, pregAll);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + OUT_OFFSET_PER_REG + i * OUT_OFFSET_PER_LOOP, totalOffsetLow2, totalOffsetHigh2,
                pregAll);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1U; i < s2Loop; ++i) {
        MicroAPI::MaskReg pregTail1 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::MaskReg pregTail2 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::Not(pregTailNeg1, pregTail1, pregAll);
        MicroAPI::Not(pregTailNeg2, pregTail2, pregAll);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int32_t> &)sparseIdx1, sparseIdxUb + i * S2_NUM_PER_LOOP);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int32_t> &)sparseIdx2, sparseIdxUb + S2_NUM_PER_REG + i * S2_NUM_PER_LOOP);
        MicroAPI::Muls(sparseIdx1, sparseIdx1, sparseBlockSize, pregTail1);
        MicroAPI::Muls(sparseIdx2, sparseIdx2, sparseBlockSize, pregTail2);
        MicroAPI::ShiftRights(paBlockIdx1, sparseIdx1, shiftRightNum, pregTail1);
        MicroAPI::ShiftRights(paBlockIdx2, sparseIdx2, shiftRightNum, pregTail2);
        MicroAPI::Muls(paBlockBase1, paBlockIdx1, blockSize, pregTail1);
        MicroAPI::Muls(paBlockBase2, paBlockIdx2, blockSize, pregTail2);
        MicroAPI::Sub(paOffset1, sparseIdx1, paBlockBase1, pregTail1);
        MicroAPI::Sub(paOffset2, sparseIdx2, paBlockBase2, pregTail2);
        MicroAPI::Muls(phyOffset1, paOffset1, kvDim, pregTail1);
        MicroAPI::Muls(phyOffset2, paOffset2, kvDim, pregTail2);
        DataCopyGather(phyBlockIdx1, blkTableUb, paBlockIdx1, pregTail1);
        DataCopyGather(phyBlockIdx2, blkTableUb, paBlockIdx2, pregTail2);
        MicroAPI::Mull(blockOffsetLow1, blockOffsetHigh1, phyBlockIdx1, kvStrideReg, pregTail1);
        MicroAPI::Mull(blockOffsetLow2, blockOffsetHigh2, phyBlockIdx2, kvStrideReg, pregTail2);
        MicroAPI::Add(addCarryLow1, totalOffsetLow1, blockOffsetLow1, phyOffset1, pregTail1);
        MicroAPI::Add(addCarryLow2, totalOffsetLow2, blockOffsetLow2, phyOffset2, pregTail2);
        MicroAPI::AddC(addCarryHigh1, totalOffsetHigh1, blockOffsetHigh1, zeroReg, addCarryLow1, pregTail1);
        MicroAPI::AddC(addCarryHigh2, totalOffsetHigh2, blockOffsetHigh2, zeroReg, addCarryLow2, pregTail2);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(totalOffsetLow1, INVALID_VALUE, pregTailNeg1);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(totalOffsetHigh1, INVALID_VALUE, pregTailNeg1);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(totalOffsetLow2, INVALID_VALUE, pregTailNeg2);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(totalOffsetHigh2, INVALID_VALUE, pregTailNeg2);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * OUT_OFFSET_PER_LOOP, totalOffsetLow1, totalOffsetHigh1, pregAll);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + OUT_OFFSET_PER_REG + i * OUT_OFFSET_PER_LOOP, totalOffsetLow2, totalOffsetHigh2, pregAll);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVF(LocalTensor<uint32_t> kvPhyAddrTensor,
                                      LocalTensor<int32_t> sparseIdxTensor, LocalTensor<int32_t> blkTableTensor,
                                      const uint16_t s2Loop, const uint32_t s2Tail, const uint32_t blockSize,
                                      const int16_t shiftRightNum, const uint32_t sparseBlockSize,
                                      const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t *kvPhyAddrUb = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparseIdxUb = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t *blkTableUb = (__ubuf__ int32_t *)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFImpl<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize, shiftRightNum,
                                 sparseBlockSize, kvDim, kvStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::GetSeqLen(int32_t bIdx, bool hasActualSeq, bool hasCuSeqlens,
                                                                GlobalTensor<int32_t> &actualSeqGm,
                                                                GlobalTensor<int32_t> &cuSeqlensGm,
                                                                int64_t defaultSize)
{
    if (hasActualSeq) {
        return actualSeqGm.GetValue(bIdx);
    }
    if (hasCuSeqlens) {
        return cuSeqlensGm.GetValue(bIdx + 1) - cuSeqlensGm.GetValue(bIdx);
    }
    return defaultSize;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t CSABlockVec<TEMPLATE_ARGS>::CalcCurValidS2(
    uint32_t bIdx, int32_t s1Idx, int32_t actualS1Size, bool isOriKv, bool hasActualSeqKvlen,
    bool hasCuSeqlensKv, GlobalTensor<int32_t> &actualSeqKvlenGm, GlobalTensor<int32_t> &cuSeqlensKvGm,
    GlobalTensor<int32_t> &topkLengthGm, GlobalTensor<int32_t> &cmpResidualKvGm, ConstInfo &constInfo,
    int32_t sparseBlockCount)
{
    const uint64_t topkIdx = LAYOUT_T == SMLA_LAYOUT::TND ?
                                 cuSeqlensQGm.GetValue(bIdx) + s1Idx :
                                 static_cast<uint64_t>(bIdx) * constInfo.s1Size + s1Idx;
    if (isOriKv) {
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                      TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            const int32_t topkLen = constInfo.hasOriTopkLength ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
            return Min(topkLen, sparseBlockCount);
        }
        return 0;
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        const int32_t topkLen = constInfo.hasCmpTopkLength ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
        return Min(topkLen, sparseBlockCount);
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        if (constInfo.cmpMaskMode == 0U) {
            const int32_t topkLen = constInfo.hasCmpTopkLength ? topkLengthGm.GetValue(topkIdx) : sparseBlockCount;
            return Min(topkLen, sparseBlockCount);
        }
        const int32_t actualCmpS2Size = GetSeqLen(bIdx, hasActualSeqKvlen, hasCuSeqlensKv, actualSeqKvlenGm,
                                                  cuSeqlensKvGm, constInfo.cmpS2Size);
        const int32_t cmpResidual = cmpResidualKvGm.GetPhyAddr() == nullptr ? 0 : cmpResidualKvGm.GetValue(bIdx);
        const int32_t numerator = actualCmpS2Size * static_cast<int32_t>(constInfo.cmpRatio) + cmpResidual -
                                  actualS1Size + 1 + s1Idx;
        const int32_t maskValidS2 = numerator > 0 ? numerator / static_cast<int32_t>(constInfo.cmpRatio) : 0;
        return Min(maskValidS2, sparseBlockCount);
    }
    return 0;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb,
                                                                   int64_t bS1Idx, int64_t s1Idx, int64_t validS2,
                                                                   GlobalTensor<uint32_t> &phyAddrGm,
                                                                   uint32_t alignedSparseBlockCount)
{
    constexpr int64_t NUM_PER_BLOCK = 32;
    DataCopyParams params;
    params.blockCount = 1U;
    params.blockLen = ((validS2 + 127L) / 128L) * 128L * sizeof(int64_t) / NUM_PER_BLOCK;
    params.srcGap = 0U;
    params.dstGap = 0U;
    DataCopy(phyAddrGm[(bS1Idx + s1Idx) * alignedSparseBlockCount * 2], kvPhyAddrUb, params);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                                                   GlobalTensor<int32_t> &blockTableGm,
                                                                   uint32_t maxBlockNumPerBatch)
{
    DataCopyExtParams params;
    params.blockCount = 1U;
    params.blockLen = maxBlockNumPerBatch * sizeof(int32_t);
    params.srcStride = 0U;
    params.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, blockTableGm[bIdx * maxBlockNumPerBatch], params, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::CopySparseIdxToUb(
    LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx, int64_t validS2,
    GlobalTensor<int32_t> &sparseIndicesGm, uint32_t sparseBlockCount)
{
    DataCopyExtParams params;
    params.blockCount = 1U;
    params.blockLen = static_cast<uint32_t>(validS2 * sizeof(int32_t));
    params.srcStride = 0U;
    params.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(sparseIdxUb, sparseIndicesGm[(bS1Idx + s1Idx) * sparseBlockCount], params, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddrForKvType(
    uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t lastS1EndIdx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, GlobalTensor<int32_t> actualSeqQlenGm, bool hasActualSeqKvlen,
    bool hasCuSeqlensKv, GlobalTensor<int32_t> actualSeqKvlenGm, GlobalTensor<int32_t> cuSeqlensKvGm,
    GlobalTensor<int32_t> topkLengthGm, GlobalTensor<int32_t> cmpResidualKvGm, ConstInfo &constInfo,
    GlobalTensor<int32_t> &blockTableGm, GlobalTensor<int32_t> &sparseIndicesGm,
    GlobalTensor<uint32_t> &phyAddrGm, uint32_t kvStride, uint32_t blockSize, uint32_t maxBlockNumPerBatch,
    uint32_t sparseBlockCount, uint32_t alignedSparseBlockCount, bool isOriKv)
{
    constexpr uint16_t S2_NUM_PER_LOOP = 128;
    constexpr uint32_t VEC_CORE_NUM = IS_SPLIT_G ? 4U : 2U;
    const uint32_t vecCoreIdx = constInfo.aivIdx % VEC_CORE_NUM;

    int16_t shiftRightNum = 0;
    for (uint32_t size = blockSize; size > 1U; size >>= 1U) {
        ++shiftRightNum;
    }

    TBuf<> blkTableBuf;
    TBuf<> sparseIdxBuf;
    TBuf<> kvPhyAddrBuf;
    tPipe->InitBuffer(blkTableBuf, maxBlockNumPerBatch * sizeof(int32_t));
    tPipe->InitBuffer(sparseIdxBuf, alignedSparseBlockCount * sizeof(int32_t));
    tPipe->InitBuffer(kvPhyAddrBuf, alignedSparseBlockCount * sizeof(int64_t));
    LocalTensor<int32_t> blkTableUb = blkTableBuf.template Get<int32_t>();
    LocalTensor<int32_t> sparseIdxUb = sparseIdxBuf.template Get<int32_t>();
    LocalTensor<uint32_t> kvPhyAddrUb = kvPhyAddrBuf.template Get<uint32_t>();

    int64_t totalValidS1 = 0;
    uint32_t currentS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        const bool isLastBatch = bIdx == bN2EndIdx - 1U;
        const int32_t actualS1Size = GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm,
                                               cuSeqlensQGm, constInfo.s1Size);
        const int32_t s1End = isLastBatch && lastS1EndIdx != 0U ? lastS1EndIdx : actualS1Size;
        for (int32_t s1Idx = currentS1Start; s1Idx < s1End; ++s1Idx) {
            const int32_t validS2 = CalcCurValidS2(bIdx, s1Idx, actualS1Size, isOriKv, hasActualSeqKvlen,
                                                   hasCuSeqlensKv, actualSeqKvlenGm, cuSeqlensKvGm, topkLengthGm,
                                                   cmpResidualKvGm, constInfo, sparseBlockCount);
            totalValidS1 += validS2 > 0;
        }
        currentS1Start = 0U;
    }

    const int64_t s1PerVecCore = totalValidS1 / VEC_CORE_NUM;
    const int64_t s1Tail = totalValidS1 % VEC_CORE_NUM;
    const int64_t currentStart = s1PerVecCore * vecCoreIdx + Min(static_cast<int64_t>(vecCoreIdx), s1Tail);
    const int64_t currentCount = s1PerVecCore + (vecCoreIdx < static_cast<uint32_t>(s1Tail) ? 1 : 0);
    if (currentCount == 0) {
        return;
    }

    int64_t validCounter = 0;
    int64_t processedCount = 0;
    currentS1Start = gS1StartIdx;
    bool done = false;
    SetFlag<HardEvent::V_MTE2>(3);
    SetFlag<HardEvent::V_MTE2>(4);
    SetFlag<HardEvent::MTE3_V>(7);
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        const bool isLastBatch = bIdx == bN2EndIdx - 1U;
        const int32_t actualS1Size = GetSeqLen(bIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm,
                                               cuSeqlensQGm, constInfo.s1Size);
        const int32_t s1End = isLastBatch && lastS1EndIdx != 0U ? lastS1EndIdx : actualS1Size;
        const int64_t bS1Idx = LAYOUT_T == SMLA_LAYOUT::TND ?
                                   (hasCuSeqlensQ ? cuSeqlensQGm.GetValue(bIdx) : constInfo.s1Size * bIdx) :
                                   constInfo.s1Size * bIdx;

        WaitFlag<HardEvent::V_MTE2>(3);
        CopyPaTableToUb(blkTableUb, bIdx, blockTableGm, maxBlockNumPerBatch);
        SetFlag<HardEvent::MTE2_V>(8);
        WaitFlag<HardEvent::MTE2_V>(8);
        for (int32_t s1Idx = currentS1Start; s1Idx < s1End; ++s1Idx) {
            const int32_t validS2 = CalcCurValidS2(bIdx, s1Idx, actualS1Size, isOriKv, hasActualSeqKvlen,
                                                   hasCuSeqlensKv, actualSeqKvlenGm, cuSeqlensKvGm, topkLengthGm,
                                                   cmpResidualKvGm, constInfo, sparseBlockCount);
            if (validS2 <= 0) {
                continue;
            }
            if (validCounter < currentStart || validCounter >= currentStart + currentCount) {
                ++validCounter;
                continue;
            }
            ++validCounter;

            const uint16_t s2Loop = (validS2 + S2_NUM_PER_LOOP - 1U) / S2_NUM_PER_LOOP;
            const uint32_t s2Tail = validS2 - (s2Loop - 1U) * S2_NUM_PER_LOOP;
            WaitFlag<HardEvent::V_MTE2>(4);
            CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, validS2, sparseIndicesGm, sparseBlockCount);
            SetFlag<HardEvent::MTE2_V>(6);
            WaitFlag<HardEvent::MTE2_V>(6);
            WaitFlag<HardEvent::MTE3_V>(7);
            GetKVPhyAddrVF<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize, shiftRightNum,
                                     constInfo.sparseBlockSize, constInfo.dSizeVInput, kvStride);
            SetFlag<HardEvent::V_MTE2>(4);
            SetFlag<HardEvent::V_MTE3>(5);
            WaitFlag<HardEvent::V_MTE3>(5);
            CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, validS2, phyAddrGm, alignedSparseBlockCount);
            SetFlag<HardEvent::MTE3_V>(7);
            if (++processedCount >= currentCount) {
                done = true;
                break;
            }
        }
        SetFlag<HardEvent::V_MTE2>(3);
        currentS1Start = 0U;
    }
    WaitFlag<HardEvent::V_MTE2>(3);
    WaitFlag<HardEvent::V_MTE2>(4);
    WaitFlag<HardEvent::MTE3_V>(7);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetKVPhyAddr(
    uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx, uint32_t gS1StartIdx, uint32_t lastS1EndIdx,
    bool hasActualSeqQlen, bool hasCuSeqlensQ, GlobalTensor<int32_t> actualSeqQlenGm,
    bool hasActualSeqOriKvlen, bool hasCuSeqlensOriKv, GlobalTensor<int32_t> actualSeqOriKvlenGm,
    GlobalTensor<int32_t> cuSeqlensOriKvGm, GlobalTensor<int32_t> oriTopkLengthGm,
    bool hasActualSeqCmpKvlen, bool hasCuSeqlensCmpKv, GlobalTensor<int32_t> actualSeqCmpKvlenGm,
    GlobalTensor<int32_t> cuSeqlensCmpKvGm, GlobalTensor<int32_t> cmpTopkLengthGm,
    GlobalTensor<int32_t> cmpResidualKvGm, __gm__ uint8_t *phyAddrWorkspace, ConstInfo &constInfo)
{
    if (hasLoad == 0U) {
        SyncAll();
        tPipe->Reset();
        return;
    }

    const uint64_t totalBS1 = LAYOUT_T == SMLA_LAYOUT::TND ?
                                  static_cast<uint64_t>(constInfo.s1Size) :
                                  static_cast<uint64_t>(constInfo.bSize) * constInfo.s1Size;
    uint64_t oriPhyAddrSize = 0ULL;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize = static_cast<uint64_t>(totalBS1) * constInfo.alignedOriSparseBlockCount * sizeof(int64_t);
        oriKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)phyAddrWorkspace);
        GetKVPhyAddrForKvType(
            bN2StartIdx, bN2EndIdx, gS1StartIdx, lastS1EndIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm,
            hasActualSeqOriKvlen, hasCuSeqlensOriKv, actualSeqOriKvlenGm, cuSeqlensOriKvGm, oriTopkLengthGm,
            cmpResidualKvGm, constInfo, oriBlockTableGm, oriSparseIndicesGm, oriKvPhyAddrGm,
            static_cast<uint32_t>(constInfo.oriKeyStride0), constInfo.oriBlockSize,
            constInfo.oriMaxBlockNumPerBatch, constInfo.oriSparseBlockCount, constInfo.alignedOriSparseBlockCount, true);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpKvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(phyAddrWorkspace + oriPhyAddrSize));
        GetKVPhyAddrForKvType(
            bN2StartIdx, bN2EndIdx, gS1StartIdx, lastS1EndIdx, hasActualSeqQlen, hasCuSeqlensQ, actualSeqQlenGm,
            hasActualSeqCmpKvlen, hasCuSeqlensCmpKv, actualSeqCmpKvlenGm, cuSeqlensCmpKvGm, cmpTopkLengthGm,
            cmpResidualKvGm, constInfo, cmpBlockTableGm, cmpSparseIndicesGm, cmpKvPhyAddrGm,
            static_cast<uint32_t>(constInfo.cmpKeyStride0), constInfo.cmpBlockSize,
            constInfo.cmpMaxBlockNumPerBatch, constInfo.cmpSparseBlockCount, constInfo.alignedCmpSparseBlockCount,
            false);
    }
    SyncAll();
    tPipe->Reset();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV, __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
    __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv, __gm__ uint8_t *cmpResidualKv)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
    }
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
        }
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)oriSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    TEventID mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(mte2ToV);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B
    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }

    if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        tPipe->InitBuffer(stage0OutBuf[0], dVTemplateType * 16 * sizeof(KV_T));
        tPipe->InitBuffer(stage0OutBuf[1], dVTemplateType * 16 * sizeof(KV_T));
    }
    if (constInfo.returnSoftmaxLse) {
        tPipe->InitBuffer(outLseBuf[0], 256);
        tPipe->InitBuffer(outLseBuf[1], 256);
    }

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToVV0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte2ToVV0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    vToMte2V0Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    vToMte2V0Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[0]);
    SetFlag<HardEvent::V_MTE2>(vToMte2V0Id[1]);
    initOutputEventId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class CSABlockVecDummy {
public:
    __aicore__ inline CSABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, ConstInfo &constInfo)
    {
    }
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices,
                                            __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks,
                                            __gm__ uint8_t *sequsedOriKv, __gm__ uint8_t *sequsedCmpKv,
                                            __gm__ uint8_t *cmpResidualKv)
    {
    }
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                        __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                        __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV){};
    __aicore__ inline void InitS2SplitStaging(__gm__ uint8_t *base)
    {
    }
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
    {
    }
    __aicore__ inline void InitFDBuffers(FdRunInfo &fdRunInfo)
    {
    }
    __aicore__ inline void ProcessFlashDecode(FdRunInfo &fdRunInfo, ConstInfo &constInfo)
    {
    }
};
} // namespace SMLAKernel
#endif // SPARSE_FLASH_MLA_CSA_BLOCK_VECTOR_H
