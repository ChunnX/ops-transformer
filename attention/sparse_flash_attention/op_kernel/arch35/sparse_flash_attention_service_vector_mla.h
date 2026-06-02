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
 * \file sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

#include "util_regbase.h"
#include "sparse_flash_attention_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

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
#include "../../common/buffers_policy.h"
#endif
#if __has_include("../../common/op_kernel/buffer_manager.h")
#include "../../common/op_kernel/buffer_manager.h"
#else
#include "../../common/buffer_manager.h"
#endif
#if __has_include("../../common/op_kernel/buffer.h")
#include "../../common/op_kernel/buffer.h"
#else
#include "../../common/buffer.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

TEMPLATES_DEF
class SFAVectorService {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 576;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    // ==================== Functions ======================
    __aicore__ inline SFAVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionTilingDataMla *__restrict tiling,
                                        CVSharedParams &sfaSharedParams, int32_t aicIdx, uint8_t subBlockIdx,
                                        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            tilingData = tiling;
            if (actualSeqLengthsQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ);
            }
            if (actualSeqLengths != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths);
            }
            this->InitCubeVecSharedParams(sfaSharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &sfaConstInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                       __gm__ uint8_t *softmaxSum, ConstInfo &sfaConstInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                            __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax,
                                            __gm__ uint8_t *softmaxSum);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &sfaConstInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo, int32_t startPos);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &sfaRunInfo,
        ConstInfo &sfaConstInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &sfaRunInfo,
        ConstInfo &sfaConstInfo);

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo, int32_t startPos);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
        int64_t s2StartIdx, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo);
    __aicore__ inline void CalSparseCalSize(const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        LocalTensor<Q_T> kvOutUb, int64_t dealRow, int64_t s2StartIdx, const RunInfo &sfaRunInfo,
        ConstInfo &sfaConstInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb,
        int64_t startRow, int64_t keyOffset, ConstInfo &sfaConstInfo);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &sfaRunInfo,
        ConstInfo &sfaConstInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
        int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void CopyFALseToGm(RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo);
    __aicore__ inline void InitCubeVecSharedParams(
        CVSharedParams &sfaSharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &sfaConstInfo);

    TPipe *tPipe;
    const SparseFlashAttentionTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    LocalTensor<float> lseUb;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TBuf<> stage0OutBuf[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVAttnOutId; // 存放MTE3_V的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID vToMte3AttnOutId; // 存放V_MTE3的eventId, 用于V2 attentionOut拷出阶段的同步
    TEventID mte3ToVLseOutId; // 存放MTE3_V的eventId, 用于V1 LSE拷出阶段的同步
    TEventID vToMte3LseOutId; // 存放V_MTE3的eventId, 用于V1 LSE拷出阶段的同步
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> dequantScaleBuff;
    TBuf<> lseBuf;

    TEventID mte2ToV;
    TEventID mte2ToMte3[2];
    TEventID mte3ToMte2[2];

    bool isSinks = false;
    T negativeFloatScalar;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;

    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
SFAVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        uint64_t sfaActualSeqQPrefixSum = sfaRunInfo.sfaBoIdx == 0 ? 0 : cuSeqlensQGm.GetValue(sfaRunInfo.sfaBoIdx - 1);
        topkBS1Idx += (sfaActualSeqQPrefixSum + sfaRunInfo.sfaS1oIdx) * sfaConstInfo.sfaSparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += sfaRunInfo.sfaBoIdx * sfaConstInfo.sfaS1Size * sfaConstInfo.sfaSparseBlockCount +
            sfaRunInfo.sfaS1oIdx * sfaConstInfo.sfaSparseBlockCount; // B, S1, N2(1), K
    }
    int64_t sfaCmpS2LoopCnt = sfaRunInfo.sfaS2LoopCount;
    int64_t sfaTopkKIdx = s2IdxInBase + sfaCmpS2LoopCnt * sfaConstInfo.sfaS2BaseSize;
    if (unlikely(sfaTopkKIdx >= sfaConstInfo.sfaSparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = sparseIndicesGm.GetValue(topkBS1Idx + sfaTopkKIdx) + sfaRunInfo.sfaS2StartIdx;
    }
    sfaTopkKIdx += 1;
    if (unlikely((sfaTopkKIdx >= sfaConstInfo.sfaSparseBlockCount) || (s2IdxInBase + 1 >= sparseS2End))) {
        token1Idx = -1;
    } else {
        token1Idx = sparseIndicesGm.GetValue(topkBS1Idx + sfaTopkKIdx) + sfaRunInfo.sfaS2StartIdx;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
int64_t SFAVectorService<TEMPLATE_ARGS>::GetkeyOffset(
    int64_t s2Idx, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset = blockTableGm.GetValue(sfaRunInfo.sfaBoIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(blockSize) +
            blkTableOffset; // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
            realkeyOffset = (sfaRunInfo.sfaBoIdx * sfaConstInfo.sfaS2Size + s2Idx); // BSN(1)D
        } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            int64_t batchKvStart = (sfaRunInfo.sfaBoIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(sfaRunInfo.sfaBoIdx - 
                1);
            realkeyOffset = (batchKvStart + s2Idx);
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
SFAVectorService<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
    int64_t keyOffset, ConstInfo &sfaConstInfo)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams sfaIntriParams;
    
    sfaIntriParams.blockCount = 1;
    sfaIntriParams.dstStride = 0;
    sfaIntriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> sfaPadParams;
    // 当前仅支持COMBINE模式
    uint32_t combineBytes = 512 * sizeof(KV_T);
    sfaIntriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    sfaPadParams.isPad = true;
    sfaPadParams.leftPadding = 0;
    sfaPadParams.rightPadding = combineDimAlign - combineDim;
    sfaPadParams.paddingValue = 0;
    // 512: Key特征维度; 576: 局部Buffer行跨度
    DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * 512], sfaIntriParams, sfaPadParams);

    sfaIntriParams.blockLen = sfaConstInfo.sfaSparseBlockSize * sfaConstInfo.sfaDSizeRope *sizeof(KV_T);
    sfaIntriParams.dstStride = 512 / BUFFER_SIZE_BYTE_32B;  // 512: 模型特征维度(dSize)
    // 576: 局部Buffer行跨度（Leading Dimension）; 512: Key特征维度（dSize）; 64 :  RoPE索引数据每Token/块的行跨度（Stride）
    DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * 64], sfaIntriParams, sfaPadParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
uint32_t SFAVectorService<TEMPLATE_ARGS>::CopyInKvSparse(
    LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t token0Idx,
    int64_t token1Idx, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, sfaRunInfo, sfaConstInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, sfaRunInfo, sfaConstInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    int64_t blkTableSrcStride =
        ((keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) :
        (keyOffset1 - keyOffset0)) - sfaConstInfo.sfaSparseBlockSize);
    int64_t keySrcStride = blkTableSrcStride * sfaConstInfo.sfaDSizeNope * sizeof(KV_T);
    int64_t keyRopeSrcStride = blkTableSrcStride * sfaConstInfo.sfaDSizeRope * sizeof(KV_T);
    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0, sfaConstInfo);
    } else if (keySrcStride >= INT32_MAX || keySrcStride < 0 || sfaConstInfo.sfaSparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0, sfaConstInfo);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1, sfaConstInfo);
    } else {
        DataCopyExtParams sfaIntriParams;
        sfaIntriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        sfaIntriParams.blockLen = sfaConstInfo.sfaSparseBlockSize * sfaConstInfo.sfaDSizeNope *sizeof(KV_T);
        sfaIntriParams.dstStride = sfaConstInfo.sfaDSizeRope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        sfaIntriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> sfaPadParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * sfaConstInfo.sfaDSizeNope], // 576: 局部Buffer行跨度
                    sfaIntriParams, sfaPadParams); // combineDimAlign

        sfaIntriParams.blockLen = sfaConstInfo.sfaSparseBlockSize * sfaConstInfo.sfaDSizeRope *sizeof(KV_T);
        sfaIntriParams.dstStride = sfaConstInfo.sfaDSizeNope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        sfaIntriParams.srcStride = keyRopeSrcStride;
        // 512: Key特征维度; 576: 局部Buffer行跨度
        DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * sfaConstInfo.sfaDSizeRope],
                    sfaIntriParams, sfaPadParams); // combineDimAlign
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float* ubDstAddr, __ubuf__ int8_t* ubSrcAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<fp8_e8m0_t> vScale1;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res1;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    MicroAPI::RegTensor<float> vScalefp32Res1;
    __ubuf__ int8_t* ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float* ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t>&)vScale0, ubScaleSrcAddrTemp, 640);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vScalefp32Res0, 64, bf16TypeMaskAll);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> kvOutUb,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(v0ResGmTensor[s2StartIdx * 576], kvOutUb, dealRow * 576); // 576: KV Cache 每 Token 的特征宽度
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CalSparseCalSize(
    const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = sfaConstInfo.sfaAivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(sfaRunInfo.sfaS2RealSize, 2);
        uint32_t v0S2SizeSecondCore = sfaRunInfo.sfaS2RealSize - v0S2SizeFirstCore;
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeFirstCore, 2); // 2: Vector split size for first core (first half)
                sparseS2Start = 0;
            } else {
                // 2: Vector split size for first core (second half)
                sparseCalSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                sparseS2Start = CeilDiv(v0S2SizeFirstCore, 2); // 2: Start offset for second half of first core
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                sparseCalSize = CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
                sparseS2Start = v0S2SizeFirstCore;
            } else {
                sparseCalSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
                sparseS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2); // 2: Same as above
            }
        }
        sparseS2End = sparseS2Start + sparseCalSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(sfaRunInfo.sfaS2RealSize, vecNum), s2PerVecLoop);
        sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, sfaRunInfo.sfaS2RealSize);
        sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * 
            s2PerVecLoop, sfaRunInfo.sfaS2RealSize) : sfaRunInfo.sfaS2RealSize;
        sparseCalSize = sparseS2End - sparseS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo, int32_t startPos)
{
    blockSize = sfaConstInfo.sfaOriBlockSize;
    maxBlockNumPerBatch = sfaConstInfo.sfaOriMaxBlockNumPerBatch;

    CalSparseCalSize(sfaRunInfo, sfaConstInfo);
    ProcessSparseKv(outputL1, v0ResGm, sfaRunInfo, sfaConstInfo, startPos);
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE3>(15); // 15: 跨核同步标志位值
        CrossCoreWaitFlag<0, PIPE_MTE3>(15); // 15: 跨核同步标志位值
    }
    outputL1.SetCrossCore();
    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    const RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo, int32_t startPos)
{
    if (sparseCalSize == 0) {
        return;
    }
    bool meetEnd = false;
    int64_t s2Start = sparseS2Start;
    int64_t s2 = sparseS2Start;
    int64_t token0Idx;
    int64_t token1Idx; // 拷贝进入的两个token的index
    // 处理一个s2的base块
    uint32_t pingPong = 0;
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        LocalTensor<Q_T> stage0OutUb = this->stage0OutBuf[pingPong].template Get<Q_T>();
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
        while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) { // 拷贝满16行或者遇到-1
            GetRealCmpS2Idx(token0Idx, token1Idx, s2, sfaRunInfo, sfaConstInfo);
            s2 += 2; // 每次搬运2行
            if (token0Idx == -1 && token1Idx == -1) {
                meetEnd = true;
                break;
            }
            dealRow += CopyInKvSparse(stage0OutUb, dealRow, token0Idx, token1Idx, sfaRunInfo, sfaConstInfo);
            if (token1Idx == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow  == 0) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
            pingPong ^= 1;
            return;
        }
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3[pingPong]);
        // 2、copy kv out, ub -> l1
        CopyOutKvUb2Gm(v0ResGm, stage0OutUb, dealRow, s2Start, sfaRunInfo, sfaConstInfo);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[pingPong]);
        s2Start += dealRow;
        pingPong ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &sfaRunInfo,
    ConstInfo &sfaConstInfo)
{
    bmm1ResBuf.WaitCrossCore();
    LocalTensor<float> sumUb = this->softmaxSumBuf[sfaRunInfo.sfaMultiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[sfaRunInfo.sfaMultiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[sfaRunInfo.sfaTaskIdMod2].template Get<T>();
    int64_t stage1Offset = sfaRunInfo.sfaTaskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (sfaRunInfo.sfaS2LoopCount == 0) { // sink 丢失首token信息，sink会增加首token信息，维度是n1
        if (likely(sfaRunInfo.sfaS2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize
                     sfaRunInfo.sfaS2RealSize,
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize,
                    sfaRunInfo.sfaS2RealSize,
                static_cast<T>(sfaConstInfo.sfaSoftmaxScale), negativeFloatScalar);
        } else if (sfaRunInfo.sfaS2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize,
                sfaRunInfo.sfaS2RealSize, // 实际的计算有效元素，
                static_cast<T>(sfaConstInfo.sfaSoftmaxScale), negativeFloatScalar);
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize
                     sfaRunInfo.sfaS2RealSize,
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize,
                    sfaRunInfo.sfaS2RealSize,
                static_cast<T>(sfaConstInfo.sfaSoftmaxScale), negativeFloatScalar);
        }
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize
                     sfaRunInfo.sfaS2RealSize,
        if (likely(sfaRunInfo.sfaS2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::EQ_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize
                     sfaRunInfo.sfaS2RealSize,
                static_cast<T>(sfaConstInfo.sfaSoftmaxScale), negativeFloatScalar);
        } else if (sfaRunInfo.sfaS2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize
                     sfaRunInfo.sfaS2RealSize,
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize,
                    sfaRunInfo.sfaS2RealSize,
                static_cast<T>(sfaConstInfo.sfaSoftmaxScale), negativeFloatScalar);
        } else if (sfaRunInfo.sfaS2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, FaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize,
                    sfaRunInfo.sfaS2RealSize,
                static_cast<T>(sfaConstInfo.sfaSoftmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>(s2BaseSize * sfaConstInfo.sfaDSizeV);
    if (likely(sfaRunInfo.sfaHalfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[sfaConstInfo.sfaSubBlockIdx * \
            (BLOCK_BYTE / sizeof(Q_T)) * (sfaRunInfo.sfaMRealSize - sfaRunInfo.sfaHalfMRealSize)],
            stage1CastTensor, {s2BaseSize / 16, (uint16_t)sfaRunInfo.sfaHalfMRealSize,
            (uint16_t)(vec1Srcstride - sfaRunInfo.sfaHalfMRealSize),
            (uint16_t)(Align16Func(sfaRunInfo.sfaMRealSize) - sfaRunInfo.sfaHalfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    if (sfaRunInfo.sfaS2LoopCount != 0) {
        SFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, sfaRunInfo.sfaHalfMRealSize);
    }
    if (sfaConstInfo.sfaReturnSoftmaxLse && sfaRunInfo.sfaS2LoopCount == sfaRunInfo.sfaS2LoopLimit) {
        CopyFALseToGm(sfaRunInfo, sfaConstInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &sfaRunInfo,
    ConstInfo &sfaConstInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(sfaRunInfo.sfaVec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }
    
    sfaRunInfo.sfaVec2S1RealSize = sfaRunInfo.sfaVec2S1BaseSize;
    sfaRunInfo.sfaVec2MRealSize = sfaRunInfo.sfaVec2MBaseSize;
    int64_t vec2CalcSize = sfaRunInfo.sfaVec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    if (unlikely(sfaRunInfo.sfaS2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[sfaRunInfo.sfaTaskIdMod2].template Get<T>();
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, sfaRunInfo.sfaVec2MRealSize
                     dTemplateAlign64, 1.0, 1.0);
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sfaRunInfo.sfaVec2MRealSize, dTemplateAlign64, 1.0, 1.0);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[sfaRunInfo.sfaMultiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false, false>(
                vec2ResUb, mmRes, vec2ResUb, expUb, expUb, sumUb, sfaRunInfo.sfaVec2MRealSize,
                    dTemplateAlign64, 1.0, 1.0);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (sfaRunInfo.sfaS2LoopCount == sfaRunInfo.sfaS2LoopLimit) {
        if (unlikely(sfaRunInfo.sfaS2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[sfaRunInfo.sfaMultiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64, false>(
                vec2ResUb, vec2ResUb, sumUb, sfaRunInfo.sfaVec2MRealSize, dTemplateAlign64, 1.0);
        }

        this->CopyOutAttentionOut(sfaRunInfo, sfaConstInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3AttnOutId);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = sfaConstInfo.sfaDSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - sfaConstInfo.sfaDSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = sfaConstInfo.sfaAttentionOutStride;
    dataCopyParams.blockCount = sfaRunInfo.sfaVec2MRealSize;
    RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo, LocalTensor<VEC2_RES_T> &vec2ResUb
         int64_t vec2S1Idx, int64_t vec2CalcSize)
    DataCopyPad(this->attentionOutGm[sfaRunInfo.sfaAttentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
        int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(sfaRunInfo, sfaConstInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}
        totalOutputSize = sfaConstInfo.sfaBSize * sfaConstInfo.sfaGSize * sfaConstInfo.sfaS1Size * 
            sfaConstInfo.sfaDSizeV;
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &sfaConstInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;
    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
        totalOutputSize = sfaConstInfo.sfaBSize * sfaConstInfo.sfaGSize * sfaConstInfo.sfaS1Size * 
            sfaConstInfo.sfaDSizeV;
    } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
        totalOutputSize = sfaConstInfo.sfaS1Size * sfaConstInfo.sfaGSize * sfaConstInfo.sfaDSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - sfaConstInfo.sfaAivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
            totalReturnSoftmaxSize = sfaConstInfo.sfaBSize *  sfaConstInfo.sfaN2Size * sfaConstInfo.sfaS1Size * 
                sfaConstInfo.sfaGSize;
            // (N2,T1,G)
            totalReturnSoftmaxSize = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaS1Size * sfaConstInfo.sfaGSize;
                this->attentionOutGm[sfaConstInfo.sfaAivIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }

    if (sfaConstInfo.sfaReturnSoftmaxLse) {
        uint64_t totalReturnSoftmaxSize = 0;
        if constexpr (LAYOUT_T == SFA_LAYOUT::BSND) {
            totalReturnSoftmaxSize = sfaConstInfo.sfaBSize *  sfaConstInfo.sfaN2Size  *
                sfaConstInfo.sfaS1Size * sfaConstInfo.sfaGSize;
        } else if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            // (N2,T1,G)
            totalReturnSoftmaxSize = sfaConstInfo.sfaN2Size * sfaConstInfo.sfaS1Size * sfaConstInfo.sfaGSize;
        }
        if (coreNum != 0 && totalReturnSoftmaxSize > 0) {
            uint64_t singleCoreSoftmaxSize = (totalReturnSoftmaxSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
            uint64_t tailSoftmaxSize = totalReturnSoftmaxSize - sfaConstInfo.sfaAivIdx * singleCoreSoftmaxSize;
            uint64_t singleInitSoftmaxSize = tailSoftmaxSize < singleCoreSoftmaxSize ?
                                             tailSoftmaxSize : singleCoreSoftmaxSize;
            if (sfaConstInfo.sfaAivIdx * singleCoreSoftmaxSize < totalReturnSoftmaxSize && singleInitSoftmaxSize > 0) {
                matmul::InitOutput<float>(
                    this->softmaxSumGm[sfaConstInfo.sfaAivIdx * singleCoreSoftmaxSize], singleInitSoftmaxSize, 0);
                matmul::InitOutput<float>(
                    this->softmaxMaxGm[sfaConstInfo.sfaAivIdx * singleCoreSoftmaxSize], singleInitSoftmaxSize, 0);
            }
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                                  __gm__ uint8_t *softmaxSum, ConstInfo &sfaConstInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        this->softmaxSumGm.SetGlobalBuffer((__gm__ float *)(softmaxSum));
        this->softmaxMaxGm.SetGlobalBuffer((__gm__ float *)(softmaxMax));
        if (sfaConstInfo.sfaNeedInit == 1) {
            InitOutputSingleCore(sfaConstInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                                       __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                                       __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax,
                                                       __gm__ uint8_t *softmaxSum)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);;
    }
    sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    keyRopeGm.SetGlobalBuffer((__gm__ KV_T *)(keyRope));
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t sfaSoftmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], sfaSoftmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], sfaSoftmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], sfaSoftmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], sfaSoftmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], sfaSoftmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], sfaSoftmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::CopyFALseToGm(RunInfo &sfaRunInfo, ConstInfo &sfaConstInfo)
{
    LocalTensor<float> sumUb = this->softmaxSumBuf[sfaRunInfo.sfaMultiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[sfaRunInfo.sfaMultiCoreIdxMod2].template Get<float>();

    size_t alignedSize = (sizeof(float) * sfaRunInfo.sfaHalfMRealSize + 31) / 32 * 32 / sizeof(float);

    int64_t lseOffset = sfaRunInfo.softmaxLseOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(float) * sfaRunInfo.sfaHalfMRealSize;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    // 拷贝 softmaxMaxUb -> GM
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, maxUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxMaxGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    // 拷贝 softmaxSumUb -> GM
    WaitFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
    DataCopy(lseUb, sumUb, alignedSize);
    SetFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    WaitFlag<HardEvent::V_MTE3>(vToMte3LseOutId);
    DataCopyPad(this->softmaxSumGm[lseOffset], lseUb, dataCopyParams);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitSinksBuffer(
    ConstInfo &sfaConstInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = sfaConstInfo.sfaGSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> sfaPadParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, sfaPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline
void SFAVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &sfaConstInfo)
{
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512); // sinksBuf内存申请512B
    tPipe->InitBuffer(lseBuf, 512); // lseBuf内存申请512B
    lseUb = this->lseBuf.template Get<float>();

    tPipe->InitBuffer(stage0OutBuf[0], 576 * 16 * sizeof(KV_T)); // 576: 模型特征维度(dSize); 16 : 注意力头数 (num_heads)
    tPipe->InitBuffer(stage0OutBuf[1], 576 * 16 * sizeof(KV_T)); // 576: 模型特征维度(dSize); 16 : 注意力头数 (num_heads)

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVAttnOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVLseOutId = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVAttnOutId);
    SetFlag<HardEvent::MTE3_V>(mte3ToVLseOutId);

    vToMte3AttnOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3LseOutId = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();

    mte2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte3ToMte2[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    mte3ToMte2[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2[1]);
    mte2ToMte3[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
    mte2ToMte3[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sfaSharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sfaSharedParams.sfaBSize = sparseAttnSharedkvBaseParams.batchSize;
    sfaSharedParams.sfaN2Size = 1;
    sfaSharedParams.sfaGSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup;
    sfaSharedParams.sfaS1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sfaSharedParams.sfaS2Size = sparseAttnSharedkvBaseParams.seqSize;
    sfaSharedParams.sfaSparseBlockCount = sparseAttnSharedkvBaseParams.sfaSparseBlockCount;
    sfaSharedParams.sfaCmpRatio = 1; // 走sparse， 但不压缩
    sfaSharedParams.sfaOriMaskMode = sparseAttnSharedkvBaseParams.sfaSparseMode;
    sfaSharedParams.sfaOriWinLeft = -1;
    sfaSharedParams.sfaOriWinRight = 0;
    sfaSharedParams.sfaLayoutType = sparseAttnSharedkvBaseParams.outputLayout;
    sfaSharedParams.sfaDSizeRope = 64; // 64: 编码维度
    sfaSharedParams.sfaSoftmaxScale = sparseAttnSharedkvBaseParams.scaleValue;
    sfaSharedParams.sfaDSize = 512; // 512: 模型特征维度(dSize)
    sfaSharedParams.sfaDSizeVInput = 512; // 512: 模型特征维度(dSize)
    sfaSharedParams.sfaUsedCoreNum = this->tilingData->singleCoreParams.sfaUsedCoreNum;

    // pageAttention, rope在C侧搬运时使用
    if constexpr (isPa) {
        sfaSharedParams.sfaOriBlockSize = sparseAttnSharedkvBaseParams.blockSize;
        sfaSharedParams.sfaOriMaxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
    }
    
    // actQ->TND, actKV pa场景任意layout均有
    sfaSharedParams.sfaIsActualSeqLengthsNull = sparseAttnSharedkvBaseParams.sfaIsActualLenDimsNull;
    sfaSharedParams.sfaIsActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams.sfaIsActualLenDimsKVNull;
    sfaSharedParams.sfaReturnSoftmaxLse = sparseAttnSharedkvBaseParams.sfaReturnSoftmaxLse;
    sfaSharedParams.sfaNeedInit = 0;
    for (uint32_t bIdx = 0; bIdx < sfaSharedParams.sfaBSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == SFA_LAYOUT::TND) {
            s2Size = bIdx == 0 ? actualSeqLengthsKVGm.GetValue(bIdx) : \
            actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else {
            if (sfaSharedParams.sfaIsActualSeqLengthsKVNull) {
                s2Size = sfaSharedParams.sfaS2Size;
            } else {
                s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            }
        }
        int64_t s1Size;
        if constexpr (LAYOUT_T == SFA_LAYOUT::TND) {
            s1Size = bIdx == 0 ? cuSeqlensQGm.GetValue(bIdx) : \
            cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1);
        } else {
            if (sfaSharedParams.sfaIsActualSeqLengthsNull) {
                s1Size = sfaSharedParams.sfaS1Size;
            } else {
                s1Size = cuSeqlensQGm.GetValue(bIdx);
            }
        }
        if (s1Size > s2Size || (LAYOUT_T == SFA_LAYOUT::BSND && s1Size < sfaSharedParams.sfaS1Size)) {
            sfaSharedParams.sfaNeedInit = 1;
            break;
        }
    }

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sfaSharedParams);
#pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAVectorService<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}


TEMPLATES_DEF class SFAVectorServiceDummy {
public:
    __aicore__ inline SFAVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxMax,
                                       __gm__ uint8_t *softmaxSum, ConstInfo &sfaConstInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                            __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *blockTable, __gm__ uint8_t *softmaxMax,
                                            __gm__ uint8_t *softmaxSum) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionTilingDataMla *__restrict tiling,
        CVSharedParams &sfaSharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ,
        __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &sfaConstInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &sfaRunInfo,
        ConstInfo &sfaConstInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &sfaRunInfo,
        ConstInfo &sfaConstInfo) {}
};
}
#endif // SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
