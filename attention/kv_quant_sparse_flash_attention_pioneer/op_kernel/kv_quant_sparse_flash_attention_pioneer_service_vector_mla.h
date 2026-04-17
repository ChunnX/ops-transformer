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
 * \file kv_quant_sparse_flash_attention_pioneer_service_vector_mla.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_PIONEER_SERVICE_VECTOR_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_PIONEER_SERVICE_VECTOR_MLA_H

#include "util_regbase.h"
#include "kv_quant_sparse_flash_attention_pioneer_common.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_qsfa.h"
#include "vf/vf_flashupdate_new_qsfa.h"
#include "vf/vf_flash_decode.h"

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
using namespace QSFAPVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

TEMPLATES_DEF
class QSFAVectorService {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = 512;
    static constexpr uint32_t dTemplateAlign64 = Align64Func(dVTemplateType);
    static constexpr uint32_t dVTemplateTypeInput = 672;
    static constexpr float R0 = 1.0f;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;

    // ==================== Functions ======================
    __aicore__ inline QSFAVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseFlashAttentionPioneerTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths)
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
            this->InitCubeVecSharedParams(sharedParams, aicIdx, subBlockIdx);
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *workspace);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

    // [FD] Phase 2 方法
    __aicore__ inline void InitFDBuffers(ConstInfo &constInfo);
    __aicore__ inline void FlashDecodeCompute(ConstInfo &constInfo, __gm__ int32_t *actualSeqKvlenAddr);

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx, int64_t s2IdxInBase,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t v0Loop, int64_t dealRow,
        int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb , int64_t startRow, int64_t token0Idx,
        int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
        int64_t s2ProcessBaseSize, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        LocalTensor<Q_T> antiKvTensorAsB16, int64_t v0Loop, int64_t dealRow, int64_t s2StartIdx,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyOutMrgeResult(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        int64_t mte2Size, int64_t mte3Size, int64_t s2keyOffset, int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(
        RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void InitCubeVecSharedParams(CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx);
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    // [FD] Phase 2 私有方法
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2FDOut(RunInfo &runInfo, ConstInfo &constInfo,
        LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(ConstInfo &constInfo);
    __aicore__ inline void CombineSplitKVRes(ConstInfo &constInfo, int64_t attenOutOffset,
        uint32_t bIdx, uint32_t n2Idx, uint32_t actualCombineLoopSize);

    TPipe *tPipe;
    const KvQuantSparseFlashAttentionPioneerTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> SparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

    // [FD] GM 变量
    GlobalTensor<T> accumOutGm;           // accumOut 存储 T 类型（与 vec2ResUb 一致）
    GlobalTensor<float> softmaxFDMaxGm;   // softmax max 存储 float（与 softmaxMaxBuf 一致）
    GlobalTensor<float> softmaxFDSumGm;   // softmax sum 存储 float（与 softmaxSumBuf 一致）

    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2]; // 2份表示可能存在pingpong
    TQue<QuePosition::VECIN, 2> stage0InQue; // for v0 input, 2份表示可能存在pingpong
    TQue<QuePosition::VECOUT, 2> stage0OutQue; // for v0 output, 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];

    T negativeFloatScalar;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;

    // [FD] Phase 2 UB buffers（与 Phase 1 时分复用，InitFDBuffers 中分配）
    TBuf<> fdLseTmpBuf;                          // 归约临时空间 + scale 结果
    TQue<QuePosition::VECIN, 1> fdMaxInputQue;   // 加载 per-split softmaxMax
    TQue<QuePosition::VECIN, 1> fdSumInputQue;   // 加载 per-split softmaxSum
    TQue<QuePosition::VECIN, 1> fdAccumInQue;    // 加载 per-split accumOut
    TQue<QuePosition::VECOUT, 1> fdResOutQue;    // 最终归约结果输出
};


TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
    int64_t s2IdxInBase, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = runInfo.boIdx == 0 ? 0 : cuSeqlensQGm.GetValue(runInfo.boIdx - 1);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * constInfo.sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx += runInfo.boIdx * constInfo.s1Size * constInfo.sparseBlockCount +
            runInfo.s1oIdx * constInfo.sparseBlockCount; // B, S1, N2(1), K
    }
    int64_t cmpS2LoopCnt = runInfo.s2LoopCount;
    if constexpr (hasSink) {
        cmpS2LoopCnt -= 1;
    }
    int64_t topkKIdx = s2IdxInBase + cmpS2LoopCnt * constInfo.s2BaseSize;
    if (unlikely(topkKIdx >= constInfo.sparseBlockCount)) {
        token0Idx = -1;
    } else {
        token0Idx = SparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
    topkKIdx += 1;
    if (unlikely(topkKIdx >= constInfo.sparseBlockCount)) {
        token1Idx = -1;
    } else {
        token1Idx = SparseIndicesGm.GetValue(topkBS1Idx + topkKIdx) + runInfo.s2StartIdx;
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t QSFAVectorService<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(constInfo.keyBlockStride) +
            blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
    } else {
        if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
            realkeyOffset = (runInfo.boIdx * constInfo.s2Size + s2Idx) * constInfo.dSizeVInput; // BSN(1)D
        } else if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            int64_t batchKvStart = (runInfo.boIdx == 0) ? 0 : actualSeqLengthsKVGm.GetValue(runInfo.boIdx - 1);
            realkeyOffset = (batchKvStart + s2Idx) * constInfo.dSizeVInput;
        }
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void
QSFAVectorService<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;
    
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    uint32_t combineBytes = 672;
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t QSFAVectorService<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb , int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo);
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }
    uint32_t combineBytes = constInfo.dSizeVInput;
    int64_t keySrcStride = (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) :
        (keyOffset1 - keyOffset0)) - combineBytes;
    if (keySrcStride >= INT32_MAX || keySrcStride < 0 || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = combineBytes;
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }

        // 当前仅支持COMBINE模式
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(kvInUb[startRow *  combineDimAlign], keyGm[keyOffset], intriParams, padParams);
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
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ int8_t* ubSrcAddr, __ubuf__ Q_T* ubDstAddr, // output first
    __ubuf__ float* ubScaleSrcAddr, uint32_t dealRowCount)
{
    uint32_t combineDim = 672; // 128对齐 640->672
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<half> vKvDataHalf0;
    MicroAPI::RegTensor<half> vKvDataHalf1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<float> vMulRes0;
    MicroAPI::RegTensor<float> vMulRes1;
    MicroAPI::RegTensor<float> vScale0;
    MicroAPI::RegTensor<float> vScale1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t blockStride = 17; // +1 to solve bank confict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 512;  // 448->512
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 1;
    const uint32_t tileSize = 128;
    
    // tilesize is 128, deal 128 b8 kv, deal 1 fp32 scale
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t* ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ float* ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T* ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData0, ubSrcTemp, tileSize / 2);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t>&)vKvData1, ubSrcTemp, combineDim - tileSize / 2);

            MicroAPI::LoadAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<float>&)vScale0, ubScaleSrcAddrTemp, combineDim / 4);

            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastFp32Res0, vScale0, fp32MaskAll);
            MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastFp32Res1, vScale0, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vMulRes0, fp32MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vMulRes1, fp32MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vCastRes0, vCastRes1);
            
            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T>& outputUb,  LocalTensor<KV_T>& inputUb, uint32_t dealRowCount)
{
    __ubuf__ int8_t* ubSrcAddr = (__ubuf__ int8_t*)(inputUb.GetPhyAddr()); // nope改成在左，所以起始位置是0
    __ubuf__ Q_T* ubDstAddr = (__ubuf__ Q_T*)(outputUb.GetPhyAddr());
    __ubuf__ float* ubScaleAddr = (__ubuf__ float*)(inputUb[512 + 64 * 2].GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubSrcAddr, ubDstAddr, ubScaleAddr, dealRowCount);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, int64_t s2ProcessBaseSize, ConstInfo &constInfo)
{
    // srcTensor是nope(512) + nope(64) + scale + pad, dstTensor是nope(512) + rope(64)
    AntiquantVFFp8D448<Q_T, KV_T>(antiKvTensorAsB16, srcTensor, dealRow);

    LocalTensor<Q_T> kRopeUb = srcTensor[constInfo.dSizeNope].template ReinterpretCast<Q_T>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)]; // V0单次处理16行数据
    Copy(kRopeUbNz, kRopeUb,
        constInfo.dSizeRope, // mask 处理多少列数据
        static_cast<uint8_t>(dealRow), // repeatTime, 每次处理多少个block
        {
            17, // dst stride
            1, // src stride
            1, // dst repeat stride
            21 // src repeat stride, 640 / 32   // 640 -> 672 : 20 -> 21
        });
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    LocalTensor<Q_T> antiKvTensorAsB16, int64_t v0Loop, int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo, ConstInfo &constInfo)
{
    outputL1.WaitCrossCore(); // 核间同步
    
    if (hasSink && runInfo.s2LoopCount == 0) {
        outputL1.SetCrossCore();
        return;
    }

    blockSize = constInfo.blockSize;
    maxBlockNumPerBatch = constInfo.maxBlockNumPerBatch;

    ProcessSparseKv(outputL1, runInfo, constInfo);

    outputL1.SetCrossCore(); // 核间同步
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t s2ProcessBaseSize = 32;
    int64_t s2PerVecLoop = 2LL;
    int64_t vecNum = 2LL;
    int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);

    // Left-closed, right-open interval
    // 4x = 2x + 2x
    // 4x + 1 = (2x + 2) + (2x - 1)
    // 4x + 2 = (2x + 2) + (2x)
    // 4x + 3 = (2x + 2) + (2x + 1)
    int64_t s2Start = GetSubBlockIdx() == 0 ? 0 : s2Loops * s2PerVecLoop;
    int64_t s2End = GetSubBlockIdx() == 0 ? s2Loops * s2PerVecLoop: runInfo.s2RealSize;
    bool meetEnd = false;
    int64_t s2 = s2Start;
    int64_t token0Idx, token1Idx; // 拷贝进入的两个token的index
    // 处理一个s2的base块
    while ((s2 < s2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        LocalTensor<KV_T> kvInUb = stage0InQue.AllocTensor<KV_T>();
        while (dealRow < 16) { // 拷贝满16行或者遇到-1
            GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
            s2 += 2; // 每次搬运2行
            if (token0Idx== -1 && token1Idx == -1) {
                meetEnd = true;
                break;
            }
            dealRow += CopyInKvSparse(kvInUb, dealRow, token0Idx, token1Idx, runInfo, constInfo);
            if (token1Idx == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow  == 0) {
            stage0InQue.FreeTensor(kvInUb);
            return;
        }
        stage0InQue.EnQue(kvInUb);
        kvInUb = stage0InQue.DeQue<KV_T>();

        // 2、dequant by vf
        LocalTensor<Q_T> kvDequantOutUb = stage0OutQue.AllocTensor<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, s2ProcessBaseSize, constInfo);
        stage0InQue.FreeTensor(kvInUb);
        stage0OutQue.EnQue(kvDequantOutUb);
        kvDequantOutUb = stage0OutQue.DeQue<Q_T>();

        // 3、copy kv out, ub -> l1
        CopyOutKvUb2L1(outputL1, kvDequantOutUb, 0, dealRow, s2Start, runInfo, constInfo);
        s2Start += dealRow;
        stage0OutQue.FreeTensor(kvDequantOutUb);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.multiCoreIdxMod2].template Get<float>();
    LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
    int64_t stage1Offset = runInfo.taskIdMod2;
    auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    if (runInfo.s2LoopCount == 0) {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, QSFAPVectorApi::OriginNRange::EQ_128_QSFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, QSFAPVectorApi::OriginNRange::GT_0_AND_LTE_64_QSFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, QSFAPVectorApi::OriginNRange::GT_64_AND_LTE_128_QSFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, QSFAPVectorApi::OriginNRange::EQ_128_QSFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, QSFAPVectorApi::OriginNRange::GT_0_AND_LTE_64_QSFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, QSFAPVectorApi::OriginNRange::GT_64_AND_LTE_128_QSFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
    this->stage1OutQue[stage1Offset].template DeQue<Q_T>();

    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>(s2BaseSize * constInfo.dSizeV);
    if (likely(runInfo.halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - runInfo.halfMRealSize)],
            stage1CastTensor, {s2BaseSize / 16, (uint16_t)runInfo.halfMRealSize,
            (uint16_t)(vec1Srcstride - runInfo.halfMRealSize),
            (uint16_t)(Align16Func(runInfo.mRealSize) - runInfo.halfMRealSize)});
    }

    this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

    outputBuf.SetCrossCore();
    if (runInfo.s2LoopCount != 0) {
        QSFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = runInfo.vec2MBaseSize;
    int64_t vec2CalcSize = runInfo.vec2MRealSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();

    WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    if (unlikely(runInfo.s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        LocalTensor<T> expUb = softmaxExpBuf[runInfo.taskIdMod2].template Get<T>();
        if (runInfo.s2LoopCount < runInfo.s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2MRealSize);
        } else {
            // [FD] FD 模式不做最终归一化，保留 unnormalized accumOut
            if constexpr (isFd) {
                FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, runInfo.vec2MRealSize);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2MRealSize);
            }
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        // [FD] FD 模式跳过 LastDivNew，保留 unnormalized 结果
        if constexpr (!isFd) {
            if (unlikely(runInfo.s2LoopCount == 0)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
                LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize);
            }
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);

        // [FD] Phase 1 结束后写入 softmax statistics 到 workspace
        if constexpr (isFd) {
            this->ComputeLogSumExpAndCopyToGm(constInfo);
        }
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &runInfo, ConstInfo &constInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
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
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    if constexpr (isFd) {
        this->Bmm2FDOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
    } else {
        this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if constexpr (LAYOUT_T == QSFA_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *key,
__gm__ uint8_t *value, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable, __gm__ uint8_t *workspace)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    SparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    if constexpr (isFd) {
        // FD workspace 布局: [accumOut(T) | softmaxFDMax(float) | softmaxFDSum(float)]
        // accumOutSize 单位: T 元素数; logSumExpSize 单位: float 元素数
        uint32_t accumOutSize = this->tilingData->splitKVParams.accumOutSize;
        uint32_t logSumExpSize = this->tilingData->splitKVParams.logSumExpSize;
        __gm__ uint8_t *wsBytes = (__gm__ uint8_t *)workspace;
        accumOutGm.SetGlobalBuffer((__gm__ T *)wsBytes);
        uint64_t accumOutBytes = static_cast<uint64_t>(accumOutSize) * sizeof(T);
        __gm__ float *lseBase = (__gm__ float *)(wsBytes + accumOutBytes);
        softmaxFDMaxGm.SetGlobalBuffer(lseBase);
        softmaxFDSumGm.SetGlobalBuffer(lseBase + logSumExpSize / 2);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(stage0InQue, 2, dVTemplateTypeInput * 16 * sizeof(KV_T)); // V0阶段每次处理16个seq, 开2 buffer
    tPipe->InitBuffer(stage0OutQue, 2, 576 * (16 + 1) * sizeof(Q_T)); // kv输入D轴640, V0阶段每次处理16个seq, 开2 buffer

    tPipe->InitBuffer(stage1OutQue[0], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutQue[1], 1, vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, (s1BaseSize / CV_RATIO) * dTemplateAlign64 * sizeof(T));

    mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();

    vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
    CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx)
{
    auto &sparseAttnSharedkvBaseParams = this->tilingData->baseParams;
    sharedParams.bSize = sparseAttnSharedkvBaseParams.batchSize;
    sharedParams.n2Size = 1;
    sharedParams.gSize = sparseAttnSharedkvBaseParams.nNumOfQInOneGroup; 
    sharedParams.s1Size = sparseAttnSharedkvBaseParams.qSeqSize;
    sharedParams.s2Size = sparseAttnSharedkvBaseParams.seqSize;
    sharedParams.sparseBlockCount = sparseAttnSharedkvBaseParams.sparseBlockCount;
    sharedParams.maskMode = sparseAttnSharedkvBaseParams.sparseMode;
    sharedParams.layoutType = sparseAttnSharedkvBaseParams.outputLayout; 
    sharedParams.dSizeRope = 64;
    sharedParams.softmaxScale = SOFTMAX_SCALE;
    sharedParams.dSize = 576;
    sharedParams.dSizeVInput = sparseAttnSharedkvBaseParams.dSizeVInput;
    sharedParams.keyBlockStride = sparseAttnSharedkvBaseParams.keyBlockStride;
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;
    sharedParams.splitKVNum = this->tilingData->splitKVParams.s2;
    if constexpr (isPa) {
        sharedParams.blockSize = sparseAttnSharedkvBaseParams.blockSize;
        sharedParams.maxBlockNumPerBatch = sparseAttnSharedkvBaseParams.maxBlockNumPerBatch;
    }
    
    // actQ->TND, actKV pa场景任意layout均有
    sharedParams.isActualSeqLengthsNull = sparseAttnSharedkvBaseParams.isActualLenDimsNull;
    sharedParams.isActualSeqLengthsKVNull = sparseAttnSharedkvBaseParams.isActualLenDimsKVNull;

    sharedParams.needInit = 0;
    for (uint32_t bIdx = 0; bIdx < sharedParams.bSize; bIdx++) {
        int64_t s2Size;
        if constexpr (KV_LAYOUT_T == QSFA_LAYOUT::TND) {
            s2Size = bIdx == 0 ? actualSeqLengthsKVGm.GetValue(bIdx) : actualSeqLengthsKVGm.GetValue(bIdx) - actualSeqLengthsKVGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsKVNull) {
                s2Size = sharedParams.s2Size;
            } else {
                s2Size = actualSeqLengthsKVGm.GetValue(bIdx);
            }
        }
        
        int64_t s1Size;
        if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            s1Size = bIdx == 0 ? cuSeqlensQGm.GetValue(bIdx) : cuSeqlensQGm.GetValue(bIdx) - cuSeqlensQGm.GetValue(bIdx - 1);
        } else {
            if (sharedParams.isActualSeqLengthsNull) {
                s1Size = sharedParams.s1Size;
            } else {
                s1Size = cuSeqlensQGm.GetValue(bIdx);
            }
        }
        if (s1Size > s2Size) {
            sharedParams.needInit = 1;
            break;
        }
    }

    if ASCEND_IS_AIV {
        if (subBlockIdx == 0) {
            auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
            auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
            #pragma unroll
            for (int i = 0; i < sizeof(CVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                *tempTilingSSbuf = *tempTiling;
            }
            CrossCoreSetFlag<SYNC_MODE, PIPE_S>(15);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}

// ==================== [FD] Phase 1: Bmm2FDOut ====================
TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::Bmm2FDOut(RunInfo &runInfo, ConstInfo &constInfo,
    LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    // FD 模式：将 FP32 中间结果直接写入 workspace accumOutGm，不做 Cast
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;
    // accumOut 布局: [fdCoreIdx][gSize * dSizeV]，subBlock 1 偏移 firstHalfMRealSize * dSizeV
    // attentionOutOffset = boIdx * gSize * dSizeV + subBlockOffset (BSND, s1=1, n2=1)
    // fdOffset = fdCoreIdx * gSize * dSizeV + subBlockOffset
    //          = attentionOutOffset + (fdCoreIdx - boIdx) * gSize * dSizeV
    int64_t fdExtraOffset = (static_cast<int64_t>(constInfo.fdCoreIdx) - runInfo.boIdx) *
        constInfo.gSize * constInfo.dSizeV;
    int64_t fdOffset = runInfo.attentionOutOffset + fdExtraOffset;

    SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) * sizeof(T) / BUFFER_SIZE_BYTE_32B;
    dataCopyParams.dstStride = 0;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->accumOutGm[fdOffset], vec2ResUb, dataCopyParams);
}

// ==================== [FD] Phase 1: ComputeLogSumExpAndCopyToGm ====================
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::ComputeLogSumExpAndCopyToGm(
    ConstInfo &constInfo)
{
    // [FD Sync] 等待 Bmm2FDOut 的 DataCopyPad 完成，再发起新的 MTE3 搬运
    SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
    WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

    // softmaxFDMax/Sum GM 布局: [fdCoreIdx][gSize][8]，8 = 32B / sizeof(float)
    // 每个 G head 占 32B（8 个 float），其中只有第一个是有效值（BroadcastMaxSum 展开）
    constexpr uint32_t LSE_BLOCK_ELEMS = 8; // 32B / sizeof(float)

    // subBlock 偏移: subBlock0 写前半 G heads, subBlock1 写后半
    uint32_t halfM = (constInfo.gSize + 1) >> 1; // firstHalfMRealSize (for s1=1)
    uint32_t halfMRealSize = (constInfo.subBlockIdx == 0) ? halfM : (constInfo.gSize - halfM);
    uint32_t mStart = constInfo.subBlockIdx * halfM;
    int64_t lseOffset = static_cast<int64_t>(constInfo.fdCoreIdx) * constInfo.gSize * LSE_BLOCK_ELEMS
                      + mStart * LSE_BLOCK_ELEMS;

    // softmaxMaxBuf/SumBuf: 连续布局, 1 float/row, halfMRealSize 行 = 256 bytes
    // multiCoreInnerIdx 从 1 开始 (FD s1=1), mod2 = 1
    LocalTensor<float> maxUb = this->softmaxMaxBuf[1].template Get<float>();
    LocalTensor<float> sumUb = this->softmaxSumBuf[1].template Get<float>();

    // BroadcastMaxSum: 连续 → 32B 对齐 (1 float/row → 8 floats/row)
    // 使用 stage2OutBuf 作为临时 broadcast 缓冲区 (64KB, 足够 halfMRealSize * 8 * 4B)
    LocalTensor<float> broadcastBuf = this->stage2OutBuf.template Get<float>();
    BroadcastMaxSum(broadcastBuf, maxUb, halfMRealSize);
    uint32_t copySize = halfMRealSize * LSE_BLOCK_ELEMS;
    DataCopy(this->softmaxFDMaxGm[lseOffset], broadcastBuf, copySize);

    BroadcastMaxSum(broadcastBuf, sumUb, halfMRealSize);
    DataCopy(this->softmaxFDSumGm[lseOffset], broadcastBuf, copySize);
}

// ==================== [FD] Phase 2: InitFDBuffers ====================
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::InitFDBuffers(
    ConstInfo &constInfo)
{
    // Phase 2: Reset UB，重新分配 FD 归约所需的缓冲区
    // Phase 1 和 Phase 2 的 UB 时分复用，Reset 释放 Phase 1 的所有 buffer
    tPipe->Reset();

    // 分配 5×32KB UB 缓冲区用于 FD 归约
    constexpr uint32_t FD_BUF_SIZE = 32 * 1024; // 32KB per buffer
    tPipe->InitBuffer(fdLseTmpBuf, FD_BUF_SIZE);
    tPipe->InitBuffer(fdMaxInputQue, 1, FD_BUF_SIZE);
    tPipe->InitBuffer(fdSumInputQue, 1, FD_BUF_SIZE);
    tPipe->InitBuffer(fdAccumInQue, 1, FD_BUF_SIZE);
    tPipe->InitBuffer(fdResOutQue, 1, FD_BUF_SIZE);
}

// ==================== [FD] Phase 2: FlashDecodeCompute ====================
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::FlashDecodeCompute(
    ConstInfo &constInfo, __gm__ int32_t *actualSeqKvlenAddr)
{
    // 仅前 B × N2 个 AIV 核执行归约
    uint32_t aivIdx = constInfo.aivIdx;
    uint32_t bIdx = aivIdx / constInfo.n2Size;
    uint32_t n2Idx = aivIdx % constInfo.n2Size;
    if (aivIdx >= constInfo.bSize * constInfo.n2Size) {
        return;
    }

    // 获取当前 batch 的实际 KV 长度，计算有效分片数
    int64_t actualSeqLen = constInfo.s2Size;
    if (actualSeqKvlenAddr != nullptr) {
        if constexpr (isPa) {
            actualSeqLen = actualSeqKvlenAddr[bIdx];
        } else if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
            actualSeqLen = (bIdx == 0) ? actualSeqKvlenAddr[0] :
                actualSeqKvlenAddr[bIdx] - actualSeqKvlenAddr[bIdx - 1];
        } else {
            actualSeqLen = actualSeqKvlenAddr[bIdx];
        }
    }
    uint32_t actualCombineLoopSize = (actualSeqLen + constInfo.sInnerLoopSize - 1) / constInfo.sInnerLoopSize;
    if (actualCombineLoopSize > constInfo.splitKVNum) {
        actualCombineLoopSize = constInfo.splitKVNum;
    }

    // 计算 attentionOut 偏移（用于最终 CopyOut）
    int64_t attenOutOffset = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        int64_t seqOffset = (bIdx == 0) ? 0 : cuSeqlensQGm.GetValue(bIdx - 1);
        attenOutOffset = seqOffset * constInfo.n2GDv + n2Idx * constInfo.gDv;
    } else {
        attenOutOffset = bIdx * constInfo.s1Size * constInfo.n2GDv + n2Idx * constInfo.gDv;
    }

    CombineSplitKVRes(constInfo, attenOutOffset, bIdx, n2Idx, actualCombineLoopSize);
}

// ==================== [FD] Phase 2: CombineSplitKVRes ====================
TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void QSFAVectorService<TEMPLATE_ARGS>::CombineSplitKVRes(
    ConstInfo &constInfo, int64_t attenOutOffset, uint32_t bIdx, uint32_t n2Idx,
    uint32_t actualCombineLoopSize)
{
    constexpr uint32_t LSE_BLOCK_ELEMS = 8; // 32B / sizeof(float)
    constexpr uint32_t FD_BUF_SIZE = 32 * 1024;
    uint32_t gSize = constInfo.gSize;
    uint32_t dSizeV = constInfo.dSizeV;
    uint32_t dSizeAligned64 = Align64Func(dSizeV);

    // s1BlockIdx: 对于 decode (s1=1)，linearS1Idx = bIdx
    uint32_t s1BlockIdx = bIdx;
    // LSE GM 基址: [s1BlockIdx * splitKVNum * gSize * LSE_BLOCK_ELEMS]
    int64_t lseBaseOffset = static_cast<int64_t>(s1BlockIdx) * constInfo.splitKVNum * gSize * LSE_BLOCK_ELEMS;
    // accumOut GM 基址: [s1BlockIdx * splitKVNum * gSize * dSizeV]
    int64_t accumBaseOffset = static_cast<int64_t>(s1BlockIdx) * constInfo.splitKVNum * gSize * dSizeV;

    // G 轴分片: gSplitSize 受 LSE(float)、accumOut 加载(T)、float cast/result 三个约束
    // LSE: splitKVNum * gSplit * 8 * sizeof(float) <= 32KB
    // accumOut 加载: gSplit * dSizeV * sizeof(T) <= 32KB
    // float cast/result: gSplit * dSizeAligned64 * sizeof(float) <= 32KB
    uint32_t gSplitByLse = FD_BUF_SIZE / (constInfo.splitKVNum * LSE_BLOCK_ELEMS * sizeof(float));
    uint32_t gSplitByAccum = FD_BUF_SIZE / (dSizeV * sizeof(T));
    uint32_t gSplitByFloat = FD_BUF_SIZE / (dSizeAligned64 * sizeof(float));
    uint32_t gSplitSize = gSplitByLse < gSplitByAccum ? gSplitByLse : gSplitByAccum;
    gSplitSize = gSplitSize < gSplitByFloat ? gSplitSize : gSplitByFloat;
    if (gSplitSize > gSize) {
        gSplitSize = gSize;
    }
    if (gSplitSize == 0) {
        gSplitSize = 1;
    }

    for (uint32_t gOffset = 0; gOffset < gSize; gOffset += gSplitSize) {
        uint32_t dealGSize = (gOffset + gSplitSize > gSize) ? (gSize - gOffset) : gSplitSize;

        // Step 1: 加载所有分片的 max/sum (float 类型) 到 UB
        LocalTensor<float> lseMaxUb = fdMaxInputQue.AllocTensor<float>();
        LocalTensor<float> lseSumUb = fdSumInputQue.AllocTensor<float>();
        for (uint32_t k = 0; k < actualCombineLoopSize; k++) {
            int64_t lseKOffset = lseBaseOffset + k * gSize * LSE_BLOCK_ELEMS + gOffset * LSE_BLOCK_ELEMS;
            uint32_t lseUbOffset = k * dealGSize * LSE_BLOCK_ELEMS;
            DataCopy(lseMaxUb[lseUbOffset], this->softmaxFDMaxGm[lseKOffset],
                dealGSize * LSE_BLOCK_ELEMS);
            DataCopy(lseSumUb[lseUbOffset], this->softmaxFDSumGm[lseKOffset],
                dealGSize * LSE_BLOCK_ELEMS);
        }
        fdMaxInputQue.EnQue(lseMaxUb);
        fdSumInputQue.EnQue(lseSumUb);
        lseMaxUb = fdMaxInputQue.DeQue<float>();
        lseSumUb = fdSumInputQue.DeQue<float>();

        // Step 2: 计算归约权重 scale[k] (全 float 运算)
        // globalMax = max(max_0, ..., max_{splitKVNum-1})
        // scale[k] = exp(max_k - globalMax) × sum_k / totalSum
        LocalTensor<float> lseTmpUb = fdLseTmpBuf.template Get<float>();
        LocalTensor<OUTPUT_T> dummySinkUb; // hasSink=false 时不使用
        dummySinkUb.SetAddr(lseTmpUb.address_);
        FaVectorApi::ComputeScaleValue_VF<float, OUTPUT_T>(
            dummySinkUb, lseMaxUb, lseSumUb, lseTmpUb,
            dealGSize, actualCombineLoopSize, false, false);

        // Step 3: 逐分片加权求和 (float 精度)
        LocalTensor<float> dstUb = fdResOutQue.AllocTensor<float>();
        for (uint32_t k = 0; k < actualCombineLoopSize; k++) {
            // 加载 split k 的 accumOut (T 类型, 如 bfloat16_t)
            LocalTensor<T> accumUbT = fdAccumInQue.AllocTensor<T>();
            int64_t accumKOffset = accumBaseOffset + k * gSize * dSizeV + gOffset * dSizeV;
            DataCopyExtParams accumCopyParams;
            accumCopyParams.blockCount = dealGSize;
            accumCopyParams.blockLen = dSizeV * sizeof(T);
            accumCopyParams.srcStride = 0;
            accumCopyParams.dstStride = 0;
            DataCopyPadExtParams<T> accumPadParams;
            accumPadParams.isPad = false;
            accumPadParams.leftPadding = 0;
            accumPadParams.rightPadding = 0;
            DataCopyPad(accumUbT, this->accumOutGm[accumKOffset], accumCopyParams, accumPadParams);
            fdAccumInQue.EnQue(accumUbT);
            accumUbT = fdAccumInQue.DeQue<T>();

            // Cast T → float (复用 lseTmpBuf, Step 2 已完成不再使用)
            LocalTensor<float> accumUbFloat = fdLseTmpBuf.template Get<float>();
            int64_t castLen = static_cast<int64_t>(dealGSize) * dSizeAligned64;
            Cast(accumUbFloat, accumUbT, RoundMode::CAST_NONE, castLen);

            // dst = sum(scale[k] * accumOut[k])
            FaVectorApi::ReduceFinalRes_VF<float>(dstUb, lseSumUb, accumUbFloat,
                dealGSize, static_cast<uint64_t>(dSizeAligned64), k);

            fdAccumInQue.FreeTensor(accumUbT);
        }

        fdMaxInputQue.FreeTensor(lseMaxUb);
        fdSumInputQue.FreeTensor(lseSumUb);

        // Step 4: Cast float → OUTPUT_T + CopyOut 到 attentionOutGm
        fdResOutQue.EnQue(dstUb);
        dstUb = fdResOutQue.DeQue<float>();

        LocalTensor<OUTPUT_T> castOut;
        castOut.SetAddr(dstUb.address_);
        int64_t castSize = static_cast<int64_t>(dealGSize) * dSizeAligned64;
        Cast(castOut, dstUb, RoundMode::CAST_ROUND, castSize);

        int64_t outOffset = attenOutOffset + gOffset * dSizeV;
        DataCopyExtParams outCopyParams;
        outCopyParams.blockCount = dealGSize;
        outCopyParams.blockLen = dSizeV * sizeof(OUTPUT_T);
        outCopyParams.srcStride = (dSizeAligned64 - dSizeV) * sizeof(OUTPUT_T) / BUFFER_SIZE_BYTE_32B;
        outCopyParams.dstStride = 0;
        DataCopyPad(this->attentionOutGm[outOffset], castOut, outCopyParams);

        fdResOutQue.FreeTensor(dstUb);
    }
}


TEMPLATES_DEF class QSFAVectorServiceDummy {
public:
    __aicore__ inline QSFAVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *workspace) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const KvQuantSparseFlashAttentionPioneerTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
    __aicore__ inline void InitFDBuffers(ConstInfo &constInfo) {}
    __aicore__ inline void FlashDecodeCompute(ConstInfo &constInfo, __gm__ int32_t *actualSeqKvlenAddr) {}
};
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_PIONEER_SERVICE_VECTOR_MLA_H
