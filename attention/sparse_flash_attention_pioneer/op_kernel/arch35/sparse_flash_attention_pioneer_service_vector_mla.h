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
#ifndef SPARSE_ATTN_SHAREDKV_QSFA_BLOCK_VECTOR_H
#define SPARSE_ATTN_SHAREDKV_QSFA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "sparse_flash_attention_pioneer_common_arch35.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_sfap.h"
#include "vf/vf_flashupdate_new_sfap.h"

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
using namespace QSFaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {

TEMPLATES_DEF
class SFAPVectorService {
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
    __aicore__ inline SFAPVectorService() {};
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionPioneerTilingDataMla *__restrict tiling,
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
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

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
    __aicore__ inline void CopyKvNdToNz(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
        int64_t s2ProcessBaseSize, ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
        LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
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
    __aicore__ inline void CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo);

    TPipe *tPipe;
    const SparseFlashAttentionPioneerTilingDataMla *__restrict tilingData;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<int32_t> SparseIndicesGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;

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

    int64_t sparseCalSize;
    int64_t sparseS2Start;
    int64_t sparseS2End;
};


TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t &token0Idx, int64_t &token1Idx,
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

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline int64_t SFAPVectorService<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        // realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
        //     static_cast<int64_t>(blockSize) * constInfo.dSizeVInput +
        //     blkTableOffset * constInfo.dSizeVInput; // BlockNum, BlockSize, N(1), D
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) *
            static_cast<int64_t>(blockSize) + blkTableOffset; // BlockNum, BlockSize, N(1), D
    } else {
        //todo
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
SFAPVectorService<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset)
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
    uint32_t combineBytes = 512 * sizeof(KV_T); //512*2
    intriParams.blockLen = combineBytes;
    uint32_t combineDim = combineBytes / sizeof(KV_T);
    uint32_t combineDimAlign = CeilAlign(combineBytes, BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * 512], intriParams, padParams);
    intriParams.blockLen = 64 * sizeof(KV_T);
    intriParams.dstStride = 512 / BUFFER_SIZE_BYTE_32B;  // 因为NopeRope目的地拼在一起，
    //intriParams.srcStride = 0;
    DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * /*constInfo.dSizeRope*/ 64], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline uint32_t SFAPVectorService<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb , int64_t startRow,
    int64_t token0Idx, int64_t token1Idx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t keyOffset0 = GetkeyOffset(token0Idx, runInfo, constInfo); // offset计算里的D被去掉
    int64_t keyOffset1 = GetkeyOffset(token1Idx, runInfo, constInfo);
    if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
        return 0;
    }

    int64_t blkTableSrcStride = ((keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - constInfo.sparseBlockSize);
    int64_t keySrcStride = blkTableSrcStride * constInfo.dSizeNope * sizeof(KV_T);
    int64_t keyRopeSrcStride = blkTableSrcStride * constInfo.dSizeRope * sizeof(KV_T);

    if (unlikely(keyOffset1 < 0)) {
        CopyInSingleKv(kvInUb, startRow, keyOffset0);
    } else if (keySrcStride >= INT32_MAX || keySrcStride < 0 || constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(kvInUb, startRow, keyOffset0);
        CopyInSingleKv(kvInUb, startRow + 1, keyOffset1);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeNope * sizeof(KV_T);
        intriParams.dstStride = constInfo.dSizeRope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
        if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
            keyOffset = keyOffset1;
        }
        DataCopyPad(kvInUb[startRow * 576], keyGm[keyOffset * constInfo.dSizeNope], intriParams, padParams);

        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.dSizeRope * sizeof(KV_T);
        intriParams.dstStride = constInfo.dSizeNope * sizeof(KV_T) / BUFFER_SIZE_BYTE_32B;
        intriParams.srcStride = keyRopeSrcStride;
        DataCopyPad(kvInUb[startRow * 576 + 512], keyRopeGm[keyOffset * constInfo.dSizeRope], intriParams, padParams);
    }
    return (keyOffset0 > -1) + (keyOffset1 > -1);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::CopyKvNdToNz(LocalTensor<Q_T> antiKvTensorAsB16,
    LocalTensor<KV_T> srcTensor, int64_t dealRow, int64_t s2ProcessBaseSize, ConstInfo &constInfo)
{
    // 非量化模式: 输入已是 fp16/bf16, 只需 ND→NZ 格式转换
    LocalTensor<Q_T> kUb = srcTensor; // 输入 ND 格式
    LocalTensor<Q_T> kUbNz = antiKvTensorAsB16; // 输出 NZ 格式

    constexpr uint16_t srcRepeatStride = 576 / 16; // = 36

    // Nope 部分: D=512, 每次处理128列, 分4段
    constexpr uint32_t tileSize = 128;
    constexpr uint32_t nopeDim = 512;
    for (uint32_t j = 0; j < nopeDim / tileSize; j++) {
        Copy(kUbNz[j * tileSize * 17], kUb[j * tileSize],
            tileSize,                        // mask: 128 元素
            static_cast<uint8_t>(dealRow),   // repeatTime
            {
                17,                           // dst stride: NZ fractal 间隔
                1,                            // src stride: 列间紧邻
                1,                            // dst repeat stride: 行间连续
                srcRepeatStride               // src repeat stride: 36
            });
    }

    // Rope 部分: D=64, 1 段
    Copy(kUbNz[nopeDim * 17], kUb[nopeDim],
        constInfo.dSizeRope,                 // mask: 64 元素
        static_cast<uint8_t>(dealRow),       // repeatTime
        {
            17,                               // dst stride
            1,                                // src stride
            1,                                // dst repeat stride
            srcRepeatStride                   // src repeat stride: 36
        });
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::CopyOutKvUb2L1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, LocalTensor<Q_T> kvOutUb,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t blockElementNum = 16;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(runInfo.s2RealSize) - dealRow;

    LocalTensor<Q_T> dst = outputL1.GetTensor<Q_T>();
    DataCopy(dst[s2StartIdx * blockElementNum], kvOutUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::CalSparseCalSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    int64_t s2PerVecLoop = 2LL;
    int64_t vecNum = 2LL;
    int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
    sparseS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
    sparseS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
    sparseCalSize = sparseS2End - sparseS2Start;
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1, const RunInfo &runInfo, ConstInfo &constInfo)
{
    outputL1.WaitCrossCore(); // 核间同步

    blockSize = constInfo.blockSize;
    maxBlockNumPerBatch = constInfo.maxBlockNumPerBatch;

    CalSparseCalSize(runInfo, constInfo);
    ProcessSparseKv(outputL1, runInfo, constInfo);

    outputL1.SetCrossCore(); // 核间同步
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputL1,
    const RunInfo &runInfo, ConstInfo &constInfo)
{
    if (sparseCalSize == 0) {
        return;
    }

    bool meetEnd = false;
    int64_t s2Start = sparseS2Start;
    int64_t s2 = sparseS2Start;
    int64_t token0Idx, token1Idx; // 拷贝进入的两个token的index
    // 处理一个s2的base块
    uint32_t pingPong = 0;
    while ((s2 < sparseS2End) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        // 1、copy kv in, gm ->ub
        LocalTensor<Q_T> kvInUb = stage0InQue.AllocTensor<KV_T>();
        while (dealRow < Min(16, sparseCalSize) && s2 < sparseS2End) { // 拷贝满16行或者遇到-1
            GetRealCmpS2Idx(token0Idx, token1Idx, s2, runInfo, constInfo);
            s2 += 2; // 每次搬运2行
            if (token0Idx== -1 && token1Idx == -1) {
                meetEnd = true;
                break;
            }
            PRINTF("dealRow:   %d\n", dealRow);
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

        // 2、nd to nz
        LocalTensor<Q_T> kvDequantOutUb = stage0OutQue.AllocTensor<Q_T>();
        CopyKvNdToNz(kvDequantOutUb, kvInUb, dealRow, 32, constInfo); // TODO 换接口名
        stage0InQue.FreeTensor(kvInUb);
        stage0OutQue.EnQue(kvDequantOutUb);
        kvDequantOutUb = stage0OutQue.DeQue<Q_T>();

        // 2、copy kv out, ub -> l1
        // DumpTensor(kvDequantOutUb, 777, 576*17);
        CopyOutKvUb2L1(outputL1, kvDequantOutUb, dealRow, s2Start, runInfo, constInfo);
        s2Start += dealRow;
        stage0OutQue.FreeTensor(kvDequantOutUb);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::ProcessVec1(
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
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, QSFaVectorApi::OriginNRange::EQ_128_SFAP>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, QSFaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFAP>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, QSFaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFAP>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        }
    } else {
        if (likely(runInfo.s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, QSFaVectorApi::OriginNRange::EQ_128_SFAP>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if (runInfo.s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, QSFaVectorApi::OriginNRange::GT_0_AND_LTE_64_SFAP>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, runInfo.halfMRealSize, runInfo.s2RealSize,
                static_cast<T>(constInfo.softmaxScale), negativeFloatScalar);
        } else if(runInfo.s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, QSFaVectorApi::OriginNRange::GT_64_AND_LTE_128_SFAP>(
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

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::ProcessVec2(
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
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(
                vec2ResUb, mmRes, vec2ResUb, expUb, sumUb, runInfo.vec2MRealSize);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (runInfo.s2LoopCount == runInfo.s2LoopLimit) {
        if (unlikely(runInfo.s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, runInfo.vec2MRealSize);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::Bmm2DataCopyOut (RunInfo &runInfo, ConstInfo &constInfo,
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
__aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::CopyOutAttentionOut(
    RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
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

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) 
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *key,
__gm__ uint8_t *value, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *blockTable)
{
    keyGm.SetGlobalBuffer((__gm__ KV_T *)(key));
    keyRopeGm.SetGlobalBuffer((__gm__ KV_T *)(keyRope));
    SparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
    if constexpr (isPa) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    // ub buffer
    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(stage0InQue, 2, 576 * 16 * sizeof(KV_T)); // V0阶段每次处理16个seq, 开2 buffer
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

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::InitCubeVecSharedParams(
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
    sharedParams.dSize = 512;  //675->512
    sharedParams.dSizeVInput = 512;
    sharedParams.usedCoreNum = this->tilingData->singleCoreParams.usedCoreNum;
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

TEMPLATES_DEF_NO_DEFAULT __aicore__ inline void SFAPVectorService<TEMPLATE_ARGS>::GetExtremeValue(
    T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
    negativeScalar = *((float *)&tmp1);
}


TEMPLATES_DEF class SFAPVectorServiceDummy {
public:
    __aicore__ inline SFAPVectorServiceDummy() {};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *keyRope, __gm__ uint8_t *sparseIndices,
        __gm__ uint8_t *blockTable) {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, const SparseFlashAttentionPioneerTilingDataMla *__restrict tiling,
        CVSharedParams &sharedParams, int32_t aicIdx, uint8_t subBlockIdx, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
};
}
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
