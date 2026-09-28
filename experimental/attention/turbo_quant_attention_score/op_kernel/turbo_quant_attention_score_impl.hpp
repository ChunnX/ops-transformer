/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TURBO_QUANT_ATTENTION_SCORE_IMPL_HPP_
#define TURBO_QUANT_ATTENTION_SCORE_IMPL_HPP_

#include "kernel_operator.h"
#include "turbo_quant_attention_score_tiling_def.h"

namespace NsTurboQuantAttentionScore {
using namespace AscendC;

// True IEEE -inf so masked scores match golden (-INFINITY) / Softmax masks.
constexpr float NEG_INF = -__builtin_huge_valf();
constexpr uint32_t MAX_CENTROIDS = 16;
constexpr uint32_t SCALAR_SLOT = 8;
constexpr uint32_t NORM_SLOT = 0 * SCALAR_SLOT;
constexpr uint32_t GAMMA_SLOT = 1 * SCALAR_SLOT;
constexpr uint32_t DOT_SLOT = 2 * SCALAR_SLOT;

__aicore__ inline uint32_t AlignUpBytes(uint32_t bytes, uint32_t ubBlockSize)
{
    // ubBlockSize comes from Ops::Base::GetUbBlockSize on host; do not hardcode 32.
    if (ubBlockSize <= 1U) {
        return bytes;
    }
    return (bytes + ubBlockSize - 1U) / ubBlockSize * ubBlockSize;
}

// NormT = bfloat16_t (BF16 binary) or half (FP16 binary); query stays BF16.
template <typename NormT>
class TurboQuantAttentionScoreImpl {
public:
    __aicore__ inline TurboQuantAttentionScoreImpl() = default;

    __aicore__ inline void Init(GM_ADDR queryRotated, GM_ADDR queryQjl, GM_ADDR keyCacheIdx, GM_ADDR keyCacheQjl,
                                GM_ADDR keyCacheNorm, GM_ADDR keyCacheGamma, GM_ADDR blockTable, GM_ADDR seqLens,
                                GM_ADDR attnScores, const optiling::TurboQuantAttentionScoreTilingData *tiling,
                                uint32_t mseBits)
    {
        numTokens_ = tiling->numTokens;
        numQHeads_ = tiling->numQHeads;
        numKvHeads_ = tiling->numKvHeads;
        headDim_ = tiling->headDim;
        blockSize_ = tiling->blockSize;
        numBlocks_ = tiling->numBlocks;
        idxPackedBytes_ = tiling->idxPackedBytes;
        qjlPackedBytes_ = tiling->qjlPackedBytes;
        maxBatch_ = tiling->maxBatch;
        maxBlocksPerSeq_ = tiling->maxBlocksPerSeq;
        maxKvLen_ = tiling->maxKvLen;
        kvTile_ = tiling->kvTile;
        usedCoreNum_ = tiling->usedCoreNum;
        groups_ = tiling->groups;
        ubBlockSize_ = tiling->ubBlockSize;
        qjlCoeff_ = tiling->qjlCoeff;
        mseBits_ = mseBits;
        totalRows_ = static_cast<uint64_t>(numTokens_) * numQHeads_;
        kShare_ = (groups_ >= 4U) ? 4U : 2U;

        const uint64_t queryElems = totalRows_ * headDim_;
        const uint64_t idxElems = static_cast<uint64_t>(numBlocks_) * blockSize_ * numKvHeads_ * idxPackedBytes_;
        const uint64_t qjlElems = static_cast<uint64_t>(numBlocks_) * blockSize_ * numKvHeads_ * qjlPackedBytes_;
        const uint64_t scalarElems = static_cast<uint64_t>(numBlocks_) * blockSize_ * numKvHeads_;
        const uint64_t tableElems = static_cast<uint64_t>(maxBatch_) * maxBlocksPerSeq_;
        const uint64_t outElems = totalRows_ * maxKvLen_;

        queryRotatedGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(queryRotated), queryElems);
        queryQjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(queryQjl), queryElems);
        keyIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(keyCacheIdx), idxElems);
        keyQjlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(keyCacheQjl), qjlElems);
        keyNormGm_.SetGlobalBuffer(reinterpret_cast<__gm__ NormT *>(keyCacheNorm), scalarElems);
        keyGammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ NormT *>(keyCacheGamma), scalarElems);
        blockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable), tableElems);
        seqLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqLens), maxBatch_);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attnScores), outElems);

        pipe_.InitBuffer(queryInQue_, 1, AlignUpBytes(headDim_ * sizeof(bfloat16_t), ubBlockSize_));
        pipe_.InitBuffer(qjlInQue_, 1, AlignUpBytes(headDim_ * sizeof(bfloat16_t), ubBlockSize_));
        // Contiguous [run, KvH, packed] tile so one GM burst covers a block run.
        pipe_.InitBuffer(idxInQue_, 1, AlignUpBytes(kvTile_ * numKvHeads_ * idxPackedBytes_, ubBlockSize_));
        pipe_.InitBuffer(qjlPackedInQue_, 1, AlignUpBytes(kvTile_ * numKvHeads_ * qjlPackedBytes_, ubBlockSize_));
        pipe_.InitBuffer(normInQue_, 1, AlignUpBytes(kvTile_ * numKvHeads_ * sizeof(NormT), ubBlockSize_));
        pipe_.InitBuffer(gammaInQue_, 1, AlignUpBytes(kvTile_ * numKvHeads_ * sizeof(NormT), ubBlockSize_));
        pipe_.InitBuffer(outputQue_, 1, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));

        pipe_.InitBuffer(queryFp32Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(qjlFp32Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(queryG1Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(qjlG1Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(queryG2Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(qjlG2Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(queryG3Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(qjlG3Buf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(dequantBuf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(signsBuf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(tmpBuf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(reduceWorkBuf_, AlignUpBytes(headDim_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(centroidBuf_, AlignUpBytes(MAX_CENTROIDS * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(scalarBuf_, AlignUpBytes(3U * SCALAR_SLOT * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(scoreTileBuf_, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(scoreTile1Buf_, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(scoreTile2Buf_, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(scoreTile3Buf_, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(normFpBuf_, AlignUpBytes(kvTile_ * numKvHeads_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(gammaFpBuf_, AlignUpBytes(kvTile_ * numKvHeads_ * sizeof(float), ubBlockSize_));
        pipe_.InitBuffer(gatherOffBuf_, AlignUpBytes(headDim_ * sizeof(uint32_t), ubBlockSize_));
        pipe_.InitBuffer(signsLutBuf_, AlignUpBytes(256U * 8U * sizeof(float), ubBlockSize_));

        // Prefill: cache dequant/signs for one (kvHead, tile) across all tokens.
        isPrefill_ = (numTokens_ > 1U && maxBatch_ == 1U);
        if (isPrefill_) {
            pipe_.InitBuffer(dequantCacheBuf_, AlignUpBytes(kvTile_ * headDim_ * sizeof(float), ubBlockSize_));
            pipe_.InitBuffer(signsCacheBuf_, AlignUpBytes(kvTile_ * headDim_ * sizeof(float), ubBlockSize_));
            pipe_.InitBuffer(normRowBuf_, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));
            pipe_.InitBuffer(gammaRowBuf_, AlignUpBytes(kvTile_ * sizeof(float), ubBlockSize_));
            // Scratch for batched Mul across kvTile rows (fewer PIPE_V barriers).
            pipe_.InitBuffer(dotTmpBuf_, AlignUpBytes(kvTile_ * headDim_ * sizeof(float), ubBlockSize_));
            // Scratch for BlockReduceSum: 8 floats per row.
            pipe_.InitBuffer(reduceTvBuf_, AlignUpBytes(kvTile_ * 8U * sizeof(float), ubBlockSize_));
        }

        InitCentroids();
        InitSignsLut();
    }

    __aicore__ inline void Process()
    {
        const uint64_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum_) {
            return;
        }
        if (isPrefill_) {
            ProcessPrefill(blockIdx);
            return;
        }
        const uint32_t numKvTiles = (maxKvLen_ + kvTile_ - 1U) / kvTile_;
        const uint64_t totalUnits = totalRows_ * numKvTiles;
        for (uint64_t unit = blockIdx; unit < totalUnits; unit += usedCoreNum_) {
            const uint64_t row = unit / numKvTiles;
            const uint32_t tileIdx = static_cast<uint32_t>(unit % numKvTiles);
            const uint32_t kvStart = tileIdx * kvTile_;
            const uint32_t token = static_cast<uint32_t>(row / numQHeads_);
            const uint32_t qHead = static_cast<uint32_t>(row % numQHeads_);
            // Same-core GQA share: only the group-base qHead unpacks once per group;
            // other units for sibling heads are no-ops (their rows are written by the leader).
            if (groups_ >= 2U) {
                if ((qHead % groups_) != 0U) {
                    continue;
                }
                const uint32_t kvHead = qHead / groups_;
                const uint32_t qHeadBase = kvHead * groups_;
                if (groups_ >= 4U) {
                    ProcessKvShared4(token, kvHead, kvStart, qHeadBase);
                } else {
                    for (uint32_t g = 0; g < groups_; g += 2U) {
                        ProcessKvShared2(token, kvHead, kvStart, qHeadBase + g);
                    }
                }
            } else {
                ProcessQueryRow(row, kvStart);
            }
        }
    }

private:
    __aicore__ inline void InitCentroids()
    {
        LocalTensor<float> centroids = centroidBuf_.Get<float>();
        if (mseBits_ == 2U) {
            centroids.SetValue(0, -0.1335033178F);
            centroids.SetValue(1, -0.04002048075F);
            centroids.SetValue(2, 0.04002048075F);
            centroids.SetValue(3, 0.1335033178F);
        } else if (mseBits_ == 4U) {
            centroids.SetValue(0, -0.2414890379F);
            centroids.SetValue(1, -0.1828317791F);
            centroids.SetValue(2, -0.1429702938F);
            centroids.SetValue(3, -0.1109927073F);
            centroids.SetValue(4, -0.08325428516F);
            centroids.SetValue(5, -0.05802082643F);
            centroids.SetValue(6, -0.03428063914F);
            centroids.SetValue(7, -0.01134236995F);
            centroids.SetValue(8, 0.01134236995F);
            centroids.SetValue(9, 0.03428063914F);
            centroids.SetValue(10, 0.05802082643F);
            centroids.SetValue(11, 0.08325428516F);
            centroids.SetValue(12, 0.1109927073F);
            centroids.SetValue(13, 0.1429702938F);
            centroids.SetValue(14, 0.1828317791F);
            centroids.SetValue(15, 0.2414890379F);
        } else {
            centroids.SetValue(0, -0.19020693F);
            centroids.SetValue(1, -0.1187859178F);
            centroids.SetValue(2, -0.06682205945F);
            centroids.SetValue(3, -0.02166347019F);
            centroids.SetValue(4, 0.02166347019F);
            centroids.SetValue(5, 0.06682205945F);
            centroids.SetValue(6, 0.1187859178F);
            centroids.SetValue(7, 0.19020693F);
        }
    }

    __aicore__ inline void InitSignsLut()
    {
        LocalTensor<float> lut = signsLutBuf_.Get<float>();
        for (uint32_t byte = 0; byte < 256U; ++byte) {
            for (uint32_t s = 0; s < 8U; ++s) {
                const float sign = ((byte >> s) & 1U) == 0U ? -1.0F : 1.0F;
                lut.SetValue(byte * 8U + s, sign);
            }
        }
    }

    __aicore__ inline void CopyBf16Row(const LocalTensor<bfloat16_t> &dst, const GlobalTensor<bfloat16_t> &src)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(headDim_ * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, copy, pad);
    }

    __aicore__ inline void CopyBytes(const LocalTensor<uint8_t> &dst, const GlobalTensor<uint8_t> &src, uint32_t bytes)
    {
        DataCopyExtParams copy{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, copy, pad);
    }

    __aicore__ inline void CopyScores(uint64_t offset, const LocalTensor<float> &src, uint32_t count)
    {
        DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(outputGm_[offset], src, copy);
    }

    __aicore__ inline void LoadQueryPair(uint32_t token, uint32_t qHead, const LocalTensor<float> &qDst,
                                         const LocalTensor<float> &sDst)
    {
        const uint64_t queryOffset = (static_cast<uint64_t>(token) * numQHeads_ + qHead) * headDim_;
        LocalTensor<bfloat16_t> queryLocal = queryInQue_.AllocTensor<bfloat16_t>();
        CopyBf16Row(queryLocal, queryRotatedGm_[queryOffset]);
        queryInQue_.EnQue(queryLocal);
        LocalTensor<bfloat16_t> qjlLocal = qjlInQue_.AllocTensor<bfloat16_t>();
        CopyBf16Row(qjlLocal, queryQjlGm_[queryOffset]);
        qjlInQue_.EnQue(qjlLocal);
        queryLocal = queryInQue_.DeQue<bfloat16_t>();
        qjlLocal = qjlInQue_.DeQue<bfloat16_t>();
        Cast(qDst, queryLocal, RoundMode::CAST_NONE, headDim_);
        Cast(sDst, qjlLocal, RoundMode::CAST_NONE, headDim_);
        PipeBarrier<PIPE_V>();
        queryInQue_.FreeTensor(queryLocal);
        qjlInQue_.FreeTensor(qjlLocal);
    }

    __aicore__ inline void WriteScoreRow(uint32_t token, uint32_t qHead, uint32_t kvStart, uint32_t rows,
                                         const LocalTensor<float> &scores)
    {
        // Direct UB->GM; caller resolves S/V→MTE3 and may batch several writes before MTE3_S.
        const uint64_t outOff = (static_cast<uint64_t>(token) * numQHeads_ + qHead) * maxKvLen_ + kvStart;
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
        DataCopyPad(outputGm_[outOff], scores, copy);
    }

    __aicore__ inline uint64_t CacheScalarOffset(uint32_t physBlock, uint32_t offsetInBlock, uint32_t kvHead) const
    {
        return (static_cast<uint64_t>(physBlock) * blockSize_ + offsetInBlock) * numKvHeads_ + kvHead;
    }

    __aicore__ inline void UnpackPrimaryAt(const LocalTensor<uint8_t> &packed, uint32_t base,
                                           const LocalTensor<float> &dst, const LocalTensor<float> &centroids)
    {
        LocalTensor<uint32_t> gatherOff = gatherOffBuf_.Get<uint32_t>();
        uint32_t out = 0;
        if (mseBits_ == 3U) {
            for (uint32_t t = 0; t < 16U; ++t) {
                const uint32_t b0 = packed.GetValue(base + t * 3U);
                const uint32_t b1 = packed.GetValue(base + t * 3U + 1U);
                const uint32_t b2 = packed.GetValue(base + t * 3U + 2U);
                const uint32_t word = b0 | (b1 << 8) | (b2 << 16);
                for (uint32_t s = 0; s < 24U; s += 3U) {
                    const uint32_t index = (word >> s) & 7U;
                    gatherOff.SetValue(out++, index * static_cast<uint32_t>(sizeof(float)));
                }
            }
        } else {
            const uint32_t mask = (1U << mseBits_) - 1U;
            for (uint32_t b = 0; b < idxPackedBytes_; ++b) {
                const uint32_t value = packed.GetValue(base + b);
                for (uint32_t s = 0; s < 8U; s += mseBits_) {
                    const uint32_t index = (value >> s) & mask;
                    gatherOff.SetValue(out++, index * static_cast<uint32_t>(sizeof(float)));
                }
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Gather(dst, centroids, gatherOff, 0, headDim_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UnpackSignsAt(const LocalTensor<uint8_t> &packed, uint32_t base,
                                         const LocalTensor<float> &dst)
    {
        LocalTensor<float> lut = signsLutBuf_.Get<float>();
        LocalTensor<uint32_t> gatherOff = gatherOffBuf_.Get<uint32_t>();
        uint32_t out = 0;
        for (uint32_t b = 0; b < qjlPackedBytes_; ++b) {
            const uint32_t value = packed.GetValue(base + b);
            const uint32_t lutBase = value * 8U;
            for (uint32_t s = 0; s < 8U; ++s) {
                gatherOff.SetValue(out++, (lutBase + s) * static_cast<uint32_t>(sizeof(float)));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Gather(dst, lut, gatherOff, 0, headDim_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float DotProduct(const LocalTensor<float> &a, const LocalTensor<float> &b,
                                       const LocalTensor<float> &tmp, const LocalTensor<float> &work,
                                       const LocalTensor<float> &scalar)
    {
        Mul(tmp, a, b, headDim_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(scalar[DOT_SLOT], tmp, work, headDim_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return scalar.GetValue(DOT_SLOT);
    }

    // Prefill: ReduceSum writes the scalar directly into dstSlot (e.g. scores[i]) — no GetValue.
    __aicore__ inline void DotToSlot(const LocalTensor<float> &a, const LocalTensor<float> &b,
                                     const LocalTensor<float> &tmp, const LocalTensor<float> &work,
                                     const LocalTensor<float> &dstSlot)
    {
        Mul(tmp, a, b, headDim_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(dstSlot, tmp, work, headDim_);
        PipeBarrier<PIPE_V>();
    }

    // Batched GEMV: broadcast-q Mul, fold 128→64, BlockReduceSum rows→scalars.
    __aicore__ inline void DotBatchRows(const LocalTensor<float> &q, const LocalTensor<float> &mat,
                                        const LocalTensor<float> &tmpMat, const LocalTensor<float> &work,
                                        const LocalTensor<float> &dstScores, uint32_t rows)
    {
        (void)work;
        if (rows == 0U) {
            return;
        }
        constexpr uint8_t kRowBlocks = 16; // headDim=128 floats / 8
        constexpr uint64_t kVecLen = 64;
        const BinaryRepeatParams mulBrp(1, 1, 1, kRowBlocks, 0, kRowBlocks);
        const uint8_t rep = static_cast<uint8_t>(rows);
        Mul(tmpMat, q, mat, kVecLen, rep, mulBrp);
        Mul(tmpMat[kVecLen], q[kVecLen], mat[kVecLen], kVecLen, rep, mulBrp);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams addBrp(1, 1, 1, kRowBlocks, kRowBlocks, kRowBlocks);
        Add(tmpMat, tmpMat, tmpMat[kVecLen], kVecLen, rep, addBrp);
        PipeBarrier<PIPE_V>();
        // 64→8 per row (src stride = 128 floats = 16 blocks), then 8→1.
        LocalTensor<float> tv = reduceTvBuf_.Get<float>();
        BlockReduceSum<float, false>(tv, tmpMat, static_cast<int32_t>(rows), 0, 1, 1, static_cast<int32_t>(kRowBlocks));
        PipeBarrier<PIPE_V>();
        const int32_t stage2Rep = static_cast<int32_t>((static_cast<uint32_t>(rows) * 8U + 63U) / 64U);
        BlockReduceSum<float, false>(dstScores, tv, stage2Rep, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    // score = norm * main + qjlCoeff * gamma * corr  (vector, over n KV positions)
    __aicore__ inline void FuseNormGammaCorr(const LocalTensor<float> &scores, const LocalTensor<float> &corr,
                                             const LocalTensor<float> &normRow, const LocalTensor<float> &gammaRow,
                                             uint32_t n)
    {
        Mul(scores, scores, normRow, n);
        PipeBarrier<PIPE_V>();
        Mul(corr, corr, gammaRow, n);
        PipeBarrier<PIPE_V>();
        Muls(corr, corr, qjlCoeff_, n);
        PipeBarrier<PIPE_V>();
        Add(scores, scores, corr, n);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MaskTailNegInf(const LocalTensor<float> &scores, uint32_t validRows, uint32_t rows)
    {
        for (uint32_t i = validRows; i < rows; ++i) {
            scores.SetValue(i, NEG_INF);
        }
    }

    // Unpack once; dual query dots; MTE3-sync between the two score GM writes.
    __aicore__ inline void ProcessKvShared2(uint32_t token, uint32_t kvHead, uint32_t kvStart, uint32_t qHeadBase)
    {
        const uint32_t rows = (kvStart + kvTile_ <= maxKvLen_) ? kvTile_ : (maxKvLen_ - kvStart);
        const uint32_t batchIdx = (maxBatch_ == 1U) ? 0U : token;

        LocalTensor<float> q0 = queryFp32Buf_.Get<float>();
        LocalTensor<float> s0 = qjlFp32Buf_.Get<float>();
        LocalTensor<float> q1 = queryG1Buf_.Get<float>();
        LocalTensor<float> s1 = qjlG1Buf_.Get<float>();
        LocalTensor<float> scores0 = scoreTileBuf_.Get<float>();
        LocalTensor<float> scores1 = scoreTile1Buf_.Get<float>();
        LocalTensor<float> dequant = dequantBuf_.Get<float>();
        LocalTensor<float> signs = signsBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> centroids = centroidBuf_.Get<float>();
        LocalTensor<float> normFp = normFpBuf_.Get<float>();
        LocalTensor<float> gammaFp = gammaFpBuf_.Get<float>();

        LoadQueryPair(token, qHeadBase, q0, s0);
        LoadQueryPair(token, qHeadBase + 1U, q1, s1);

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        const int32_t seqLen = seqLensGm_.GetValue(batchIdx);

        uint32_t i = 0;
        while (i < rows) {
            const uint32_t kvPos = kvStart + i;
            if (static_cast<int32_t>(kvPos) >= seqLen) {
                for (uint32_t j = i; j < rows; ++j) {
                    scores0.SetValue(j, NEG_INF);
                    scores1.SetValue(j, NEG_INF);
                }
                break;
            }

            const uint32_t logical0 = kvPos / blockSize_;
            const uint32_t offset0 = kvPos % blockSize_;
            const int32_t phys = blockTableGm_.GetValue(static_cast<uint64_t>(batchIdx) * maxBlocksPerSeq_ + logical0);

            uint32_t run = 1;
            while (i + run < rows) {
                const uint32_t p = kvStart + i + run;
                if (static_cast<int32_t>(p) >= seqLen) {
                    break;
                }
                if ((p / blockSize_) != logical0 || (p % blockSize_) != (offset0 + run)) {
                    break;
                }
                ++run;
            }

            const uint64_t baseScalar = (static_cast<uint64_t>(phys) * blockSize_ + offset0) * numKvHeads_;
            const uint32_t span = run * numKvHeads_;

            LocalTensor<uint8_t> idxLocal = idxInQue_.AllocTensor<uint8_t>();
            CopyBytes(idxLocal, keyIdxGm_[baseScalar * idxPackedBytes_], span * idxPackedBytes_);
            idxInQue_.EnQue(idxLocal);
            LocalTensor<uint8_t> qjlPackedLocal = qjlPackedInQue_.AllocTensor<uint8_t>();
            CopyBytes(qjlPackedLocal, keyQjlGm_[baseScalar * qjlPackedBytes_], span * qjlPackedBytes_);
            qjlPackedInQue_.EnQue(qjlPackedLocal);
            LocalTensor<NormT> normLocal = normInQue_.AllocTensor<NormT>();
            {
                DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                DataCopyPad(normLocal, keyNormGm_[baseScalar], c, p);
            }
            normInQue_.EnQue(normLocal);
            LocalTensor<NormT> gammaLocal = gammaInQue_.AllocTensor<NormT>();
            {
                DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                DataCopyPad(gammaLocal, keyGammaGm_[baseScalar], c, p);
            }
            gammaInQue_.EnQue(gammaLocal);

            idxLocal = idxInQue_.DeQue<uint8_t>();
            qjlPackedLocal = qjlPackedInQue_.DeQue<uint8_t>();
            normLocal = normInQue_.DeQue<NormT>();
            gammaLocal = gammaInQue_.DeQue<NormT>();

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(normFp, normLocal, RoundMode::CAST_NONE, span);
            Cast(gammaFp, gammaLocal, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);

            for (uint32_t r = 0; r < run; ++r) {
                const uint32_t slot = r * numKvHeads_ + kvHead;
                UnpackPrimaryAt(idxLocal, slot * idxPackedBytes_, dequant, centroids);
                UnpackSignsAt(qjlPackedLocal, slot * qjlPackedBytes_, signs);
                const float normVal = normFp.GetValue(slot);
                const float gammaVal = gammaFp.GetValue(slot);

                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                const float main0 = DotProduct(q0, dequant, tmp, work, scalar);
                const float corr0 = DotProduct(s0, signs, tmp, work, scalar);
                const float main1 = DotProduct(q1, dequant, tmp, work, scalar);
                const float corr1 = DotProduct(s1, signs, tmp, work, scalar);
                scores0.SetValue(i + r, normVal * main0 + qjlCoeff_ * gammaVal * corr0);
                scores1.SetValue(i + r, normVal * main1 + qjlCoeff_ * gammaVal * corr1);
            }

            idxInQue_.FreeTensor(idxLocal);
            qjlPackedInQue_.FreeTensor(qjlPackedLocal);
            normInQue_.FreeTensor(normLocal);
            gammaInQue_.FreeTensor(gammaLocal);
            i += run;
        }

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WriteScoreRow(token, qHeadBase, kvStart, rows, scores0);
        WriteScoreRow(token, qHeadBase + 1U, kvStart, rows, scores1);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // One KV walk for 4 GQA queries (groups=4); MTE3-sync after each score write.
    __aicore__ inline void ProcessKvShared4(uint32_t token, uint32_t kvHead, uint32_t kvStart, uint32_t qHeadBase)
    {
        const uint32_t rows = (kvStart + kvTile_ <= maxKvLen_) ? kvTile_ : (maxKvLen_ - kvStart);
        const uint32_t batchIdx = (maxBatch_ == 1U) ? 0U : token;

        LocalTensor<float> q0 = queryFp32Buf_.Get<float>();
        LocalTensor<float> s0 = qjlFp32Buf_.Get<float>();
        LocalTensor<float> q1 = queryG1Buf_.Get<float>();
        LocalTensor<float> s1 = qjlG1Buf_.Get<float>();
        LocalTensor<float> q2 = queryG2Buf_.Get<float>();
        LocalTensor<float> s2 = qjlG2Buf_.Get<float>();
        LocalTensor<float> q3 = queryG3Buf_.Get<float>();
        LocalTensor<float> s3 = qjlG3Buf_.Get<float>();
        LocalTensor<float> scores0 = scoreTileBuf_.Get<float>();
        LocalTensor<float> scores1 = scoreTile1Buf_.Get<float>();
        LocalTensor<float> scores2 = scoreTile2Buf_.Get<float>();
        LocalTensor<float> scores3 = scoreTile3Buf_.Get<float>();
        LocalTensor<float> dequant = dequantBuf_.Get<float>();
        LocalTensor<float> signs = signsBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> centroids = centroidBuf_.Get<float>();
        LocalTensor<float> normFp = normFpBuf_.Get<float>();
        LocalTensor<float> gammaFp = gammaFpBuf_.Get<float>();

        LoadQueryPair(token, qHeadBase, q0, s0);
        LoadQueryPair(token, qHeadBase + 1U, q1, s1);
        LoadQueryPair(token, qHeadBase + 2U, q2, s2);
        LoadQueryPair(token, qHeadBase + 3U, q3, s3);

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        const int32_t seqLen = seqLensGm_.GetValue(batchIdx);

        uint32_t i = 0;
        while (i < rows) {
            const uint32_t kvPos = kvStart + i;
            if (static_cast<int32_t>(kvPos) >= seqLen) {
                for (uint32_t j = i; j < rows; ++j) {
                    scores0.SetValue(j, NEG_INF);
                    scores1.SetValue(j, NEG_INF);
                    scores2.SetValue(j, NEG_INF);
                    scores3.SetValue(j, NEG_INF);
                }
                break;
            }

            const uint32_t logical0 = kvPos / blockSize_;
            const uint32_t offset0 = kvPos % blockSize_;
            const int32_t phys = blockTableGm_.GetValue(static_cast<uint64_t>(batchIdx) * maxBlocksPerSeq_ + logical0);

            uint32_t run = 1;
            while (i + run < rows) {
                const uint32_t p = kvStart + i + run;
                if (static_cast<int32_t>(p) >= seqLen) {
                    break;
                }
                if ((p / blockSize_) != logical0 || (p % blockSize_) != (offset0 + run)) {
                    break;
                }
                ++run;
            }

            const uint64_t baseScalar = (static_cast<uint64_t>(phys) * blockSize_ + offset0) * numKvHeads_;
            const uint32_t span = run * numKvHeads_;

            LocalTensor<uint8_t> idxLocal = idxInQue_.AllocTensor<uint8_t>();
            CopyBytes(idxLocal, keyIdxGm_[baseScalar * idxPackedBytes_], span * idxPackedBytes_);
            idxInQue_.EnQue(idxLocal);
            LocalTensor<uint8_t> qjlPackedLocal = qjlPackedInQue_.AllocTensor<uint8_t>();
            CopyBytes(qjlPackedLocal, keyQjlGm_[baseScalar * qjlPackedBytes_], span * qjlPackedBytes_);
            qjlPackedInQue_.EnQue(qjlPackedLocal);
            LocalTensor<NormT> normLocal = normInQue_.AllocTensor<NormT>();
            {
                DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                DataCopyPad(normLocal, keyNormGm_[baseScalar], c, p);
            }
            normInQue_.EnQue(normLocal);
            LocalTensor<NormT> gammaLocal = gammaInQue_.AllocTensor<NormT>();
            {
                DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                DataCopyPad(gammaLocal, keyGammaGm_[baseScalar], c, p);
            }
            gammaInQue_.EnQue(gammaLocal);

            idxLocal = idxInQue_.DeQue<uint8_t>();
            qjlPackedLocal = qjlPackedInQue_.DeQue<uint8_t>();
            normLocal = normInQue_.DeQue<NormT>();
            gammaLocal = gammaInQue_.DeQue<NormT>();

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(normFp, normLocal, RoundMode::CAST_NONE, span);
            Cast(gammaFp, gammaLocal, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);

            for (uint32_t r = 0; r < run; ++r) {
                const uint32_t slot = r * numKvHeads_ + kvHead;
                UnpackPrimaryAt(idxLocal, slot * idxPackedBytes_, dequant, centroids);
                UnpackSignsAt(qjlPackedLocal, slot * qjlPackedBytes_, signs);
                const float normVal = normFp.GetValue(slot);
                const float gammaVal = gammaFp.GetValue(slot);

                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                const float main0 = DotProduct(q0, dequant, tmp, work, scalar);
                const float corr0 = DotProduct(s0, signs, tmp, work, scalar);
                const float main1 = DotProduct(q1, dequant, tmp, work, scalar);
                const float corr1 = DotProduct(s1, signs, tmp, work, scalar);
                const float main2 = DotProduct(q2, dequant, tmp, work, scalar);
                const float corr2 = DotProduct(s2, signs, tmp, work, scalar);
                const float main3 = DotProduct(q3, dequant, tmp, work, scalar);
                const float corr3 = DotProduct(s3, signs, tmp, work, scalar);
                scores0.SetValue(i + r, normVal * main0 + qjlCoeff_ * gammaVal * corr0);
                scores1.SetValue(i + r, normVal * main1 + qjlCoeff_ * gammaVal * corr1);
                scores2.SetValue(i + r, normVal * main2 + qjlCoeff_ * gammaVal * corr2);
                scores3.SetValue(i + r, normVal * main3 + qjlCoeff_ * gammaVal * corr3);
            }

            idxInQue_.FreeTensor(idxLocal);
            qjlPackedInQue_.FreeTensor(qjlPackedLocal);
            normInQue_.FreeTensor(normLocal);
            gammaInQue_.FreeTensor(gammaLocal);
            i += run;
        }

        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WriteScoreRow(token, qHeadBase, kvStart, rows, scores0);
        WriteScoreRow(token, qHeadBase + 1U, kvStart, rows, scores1);
        WriteScoreRow(token, qHeadBase + 2U, kvStart, rows, scores2);
        WriteScoreRow(token, qHeadBase + 3U, kvStart, rows, scores3);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    // Prefill: schedule by (kvHead, tile); unpack once then fan out over tokens.
    __aicore__ inline void ProcessPrefill(uint64_t blockIdx)
    {
        const uint32_t numKvTiles = (maxKvLen_ + kvTile_ - 1U) / kvTile_;
        const uint64_t totalUnits = static_cast<uint64_t>(numKvHeads_) * numKvTiles;
        for (uint64_t unit = blockIdx; unit < totalUnits; unit += usedCoreNum_) {
            const uint32_t kvHead = static_cast<uint32_t>(unit / numKvTiles);
            const uint32_t tileIdx = static_cast<uint32_t>(unit % numKvTiles);
            const uint32_t kvStart = tileIdx * kvTile_;
            const uint32_t tokenStart = 0U;
            const uint32_t tokenEnd = numTokens_;
            if (groups_ >= 4U) {
                ProcessPrefillKvShared4(kvHead, kvStart, tokenStart, tokenEnd);
            } else if (groups_ >= 2U) {
                const uint32_t qHeadBase = kvHead * groups_;
                for (uint32_t g = 0; g < groups_; g += 2U) {
                    ProcessPrefillKvShared2(kvHead, kvStart, qHeadBase + g, tokenStart, tokenEnd);
                }
            } else {
                ProcessPrefillKvSingle(kvHead, kvStart, tokenStart, tokenEnd);
            }
        }
    }

    // Unpack one (kvHead, tile) into UB caches; returns valid row count and fills mask via NEG_INF later.
    __aicore__ inline void UnpackTileToCache(uint32_t kvHead, uint32_t kvStart, uint32_t rows,
                                             const LocalTensor<float> &dequantCache,
                                             const LocalTensor<float> &signsCache, const LocalTensor<float> &normRow,
                                             const LocalTensor<float> &gammaRow, int32_t seqLen)
    {
        LocalTensor<float> centroids = centroidBuf_.Get<float>();
        LocalTensor<float> normFp = normFpBuf_.Get<float>();
        LocalTensor<float> gammaFp = gammaFpBuf_.Get<float>();

        uint32_t i = 0;
        while (i < rows) {
            const uint32_t kvPos = kvStart + i;
            if (static_cast<int32_t>(kvPos) >= seqLen) {
                for (uint32_t j = i; j < rows; ++j) {
                    normRow.SetValue(j, 0.0F);
                    gammaRow.SetValue(j, 0.0F);
                }
                break;
            }

            const uint32_t logical0 = kvPos / blockSize_;
            const uint32_t offset0 = kvPos % blockSize_;
            const int32_t phys = blockTableGm_.GetValue(static_cast<uint64_t>(logical0));

            uint32_t run = 1;
            while (i + run < rows) {
                const uint32_t p = kvStart + i + run;
                if (static_cast<int32_t>(p) >= seqLen) {
                    break;
                }
                if ((p / blockSize_) != logical0 || (p % blockSize_) != (offset0 + run)) {
                    break;
                }
                ++run;
            }

            const uint64_t baseScalar = (static_cast<uint64_t>(phys) * blockSize_ + offset0) * numKvHeads_;
            const uint32_t span = run * numKvHeads_;

            LocalTensor<uint8_t> idxLocal = idxInQue_.AllocTensor<uint8_t>();
            CopyBytes(idxLocal, keyIdxGm_[baseScalar * idxPackedBytes_], span * idxPackedBytes_);
            idxInQue_.EnQue(idxLocal);
            LocalTensor<uint8_t> qjlPackedLocal = qjlPackedInQue_.AllocTensor<uint8_t>();
            CopyBytes(qjlPackedLocal, keyQjlGm_[baseScalar * qjlPackedBytes_], span * qjlPackedBytes_);
            qjlPackedInQue_.EnQue(qjlPackedLocal);
            LocalTensor<NormT> normLocal = normInQue_.AllocTensor<NormT>();
            {
                DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                DataCopyPad(normLocal, keyNormGm_[baseScalar], c, p);
            }
            normInQue_.EnQue(normLocal);
            LocalTensor<NormT> gammaLocal = gammaInQue_.AllocTensor<NormT>();
            {
                DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                DataCopyPad(gammaLocal, keyGammaGm_[baseScalar], c, p);
            }
            gammaInQue_.EnQue(gammaLocal);

            idxLocal = idxInQue_.DeQue<uint8_t>();
            qjlPackedLocal = qjlPackedInQue_.DeQue<uint8_t>();
            normLocal = normInQue_.DeQue<NormT>();
            gammaLocal = gammaInQue_.DeQue<NormT>();

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(normFp, normLocal, RoundMode::CAST_NONE, span);
            Cast(gammaFp, gammaLocal, RoundMode::CAST_NONE, span);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            SetFlag<HardEvent::V_S>(EVENT_ID1);
            WaitFlag<HardEvent::V_S>(EVENT_ID1);

            for (uint32_t r = 0; r < run; ++r) {
                const uint32_t slot = r * numKvHeads_ + kvHead;
                const uint32_t rowOff = (i + r) * headDim_;
                UnpackPrimaryAt(idxLocal, slot * idxPackedBytes_, dequantCache[rowOff], centroids);
                UnpackSignsAt(qjlPackedLocal, slot * qjlPackedBytes_, signsCache[rowOff]);
                normRow.SetValue(i + r, normFp.GetValue(slot));
                gammaRow.SetValue(i + r, gammaFp.GetValue(slot));
            }

            idxInQue_.FreeTensor(idxLocal);
            qjlPackedInQue_.FreeTensor(qjlPackedLocal);
            normInQue_.FreeTensor(normLocal);
            gammaInQue_.FreeTensor(gammaLocal);
            i += run;
        }
    }

    __aicore__ inline void ProcessPrefillKvShared4(uint32_t kvHead, uint32_t kvStart, uint32_t tokenStart,
                                                   uint32_t tokenEnd)
    {
        const uint32_t rows = (kvStart + kvTile_ <= maxKvLen_) ? kvTile_ : (maxKvLen_ - kvStart);
        const uint32_t qHeadBase = kvHead * groups_;

        LocalTensor<float> dequantCache = dequantCacheBuf_.Get<float>();
        LocalTensor<float> signsCache = signsCacheBuf_.Get<float>();
        LocalTensor<float> normRow = normRowBuf_.Get<float>();
        LocalTensor<float> gammaRow = gammaRowBuf_.Get<float>();

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        const int32_t seqLen = seqLensGm_.GetValue(0);

        UnpackTileToCache(kvHead, kvStart, rows, dequantCache, signsCache, normRow, gammaRow, seqLen);

        const uint32_t validRows =
            (static_cast<int32_t>(kvStart) >= seqLen) ?
                0U :
                ((kvStart + rows <= static_cast<uint32_t>(seqLen)) ? rows : (static_cast<uint32_t>(seqLen) - kvStart));

        LocalTensor<float> q0 = queryFp32Buf_.Get<float>();
        LocalTensor<float> s0 = qjlFp32Buf_.Get<float>();
        LocalTensor<float> q1 = queryG1Buf_.Get<float>();
        LocalTensor<float> s1 = qjlG1Buf_.Get<float>();
        LocalTensor<float> q2 = queryG2Buf_.Get<float>();
        LocalTensor<float> s2 = qjlG2Buf_.Get<float>();
        LocalTensor<float> q3 = queryG3Buf_.Get<float>();
        LocalTensor<float> s3 = qjlG3Buf_.Get<float>();
        LocalTensor<float> scores0 = scoreTileBuf_.Get<float>();
        LocalTensor<float> scores1 = scoreTile1Buf_.Get<float>();
        LocalTensor<float> scores2 = scoreTile2Buf_.Get<float>();
        LocalTensor<float> scores3 = scoreTile3Buf_.Get<float>();
        // Reuse decode-only / unpack scratch as corr tiles (headDim / KvH*tile >= kvTile).
        LocalTensor<float> corr0 = dequantBuf_.Get<float>();
        LocalTensor<float> corr1 = signsBuf_.Get<float>();
        LocalTensor<float> corr2 = normFpBuf_.Get<float>();
        LocalTensor<float> corr3 = gammaFpBuf_.Get<float>();
        LocalTensor<float> tmpMat = dotTmpBuf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();

        for (uint32_t token = tokenStart; token < tokenEnd; ++token) {
            LoadQueryPair(token, qHeadBase, q0, s0);
            LoadQueryPair(token, qHeadBase + 1U, q1, s1);
            LoadQueryPair(token, qHeadBase + 2U, q2, s2);
            LoadQueryPair(token, qHeadBase + 3U, q3, s3);

            if (validRows > 0U) {
                DotBatchRows(q0, dequantCache, tmpMat, work, scores0, validRows);
                DotBatchRows(s0, signsCache, tmpMat, work, corr0, validRows);
                DotBatchRows(q1, dequantCache, tmpMat, work, scores1, validRows);
                DotBatchRows(s1, signsCache, tmpMat, work, corr1, validRows);
                DotBatchRows(q2, dequantCache, tmpMat, work, scores2, validRows);
                DotBatchRows(s2, signsCache, tmpMat, work, corr2, validRows);
                DotBatchRows(q3, dequantCache, tmpMat, work, scores3, validRows);
                DotBatchRows(s3, signsCache, tmpMat, work, corr3, validRows);
                FuseNormGammaCorr(scores0, corr0, normRow, gammaRow, validRows);
                FuseNormGammaCorr(scores1, corr1, normRow, gammaRow, validRows);
                FuseNormGammaCorr(scores2, corr2, normRow, gammaRow, validRows);
                FuseNormGammaCorr(scores3, corr3, normRow, gammaRow, validRows);
            }

            if (validRows < rows) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                MaskTailNegInf(scores0, validRows, rows);
                MaskTailNegInf(scores1, validRows, rows);
                MaskTailNegInf(scores2, validRows, rows);
                MaskTailNegInf(scores3, validRows, rows);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            WriteScoreRow(token, qHeadBase, kvStart, rows, scores0);
            WriteScoreRow(token, qHeadBase + 1U, kvStart, rows, scores1);
            WriteScoreRow(token, qHeadBase + 2U, kvStart, rows, scores2);
            WriteScoreRow(token, qHeadBase + 3U, kvStart, rows, scores3);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessPrefillKvShared2(uint32_t kvHead, uint32_t kvStart, uint32_t qHeadBase,
                                                   uint32_t tokenStart, uint32_t tokenEnd)
    {
        const uint32_t rows = (kvStart + kvTile_ <= maxKvLen_) ? kvTile_ : (maxKvLen_ - kvStart);

        LocalTensor<float> dequantCache = dequantCacheBuf_.Get<float>();
        LocalTensor<float> signsCache = signsCacheBuf_.Get<float>();
        LocalTensor<float> normRow = normRowBuf_.Get<float>();
        LocalTensor<float> gammaRow = gammaRowBuf_.Get<float>();

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        const int32_t seqLen = seqLensGm_.GetValue(0);

        UnpackTileToCache(kvHead, kvStart, rows, dequantCache, signsCache, normRow, gammaRow, seqLen);

        const uint32_t validRows =
            (static_cast<int32_t>(kvStart) >= seqLen) ?
                0U :
                ((kvStart + rows <= static_cast<uint32_t>(seqLen)) ? rows : (static_cast<uint32_t>(seqLen) - kvStart));

        LocalTensor<float> q0 = queryFp32Buf_.Get<float>();
        LocalTensor<float> s0 = qjlFp32Buf_.Get<float>();
        LocalTensor<float> q1 = queryG1Buf_.Get<float>();
        LocalTensor<float> s1 = qjlG1Buf_.Get<float>();
        LocalTensor<float> scores0 = scoreTileBuf_.Get<float>();
        LocalTensor<float> scores1 = scoreTile1Buf_.Get<float>();
        LocalTensor<float> corr0 = dequantBuf_.Get<float>();
        LocalTensor<float> corr1 = signsBuf_.Get<float>();
        LocalTensor<float> tmpMat = dotTmpBuf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();

        for (uint32_t token = tokenStart; token < tokenEnd; ++token) {
            LoadQueryPair(token, qHeadBase, q0, s0);
            LoadQueryPair(token, qHeadBase + 1U, q1, s1);

            if (validRows > 0U) {
                DotBatchRows(q0, dequantCache, tmpMat, work, scores0, validRows);
                DotBatchRows(s0, signsCache, tmpMat, work, corr0, validRows);
                DotBatchRows(q1, dequantCache, tmpMat, work, scores1, validRows);
                DotBatchRows(s1, signsCache, tmpMat, work, corr1, validRows);
                FuseNormGammaCorr(scores0, corr0, normRow, gammaRow, validRows);
                FuseNormGammaCorr(scores1, corr1, normRow, gammaRow, validRows);
            }

            if (validRows < rows) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                MaskTailNegInf(scores0, validRows, rows);
                MaskTailNegInf(scores1, validRows, rows);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            WriteScoreRow(token, qHeadBase, kvStart, rows, scores0);
            WriteScoreRow(token, qHeadBase + 1U, kvStart, rows, scores1);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessPrefillKvSingle(uint32_t kvHead, uint32_t kvStart, uint32_t tokenStart,
                                                  uint32_t tokenEnd)
    {
        const uint32_t rows = (kvStart + kvTile_ <= maxKvLen_) ? kvTile_ : (maxKvLen_ - kvStart);

        LocalTensor<float> dequantCache = dequantCacheBuf_.Get<float>();
        LocalTensor<float> signsCache = signsCacheBuf_.Get<float>();
        LocalTensor<float> normRow = normRowBuf_.Get<float>();
        LocalTensor<float> gammaRow = gammaRowBuf_.Get<float>();

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        const int32_t seqLen = seqLensGm_.GetValue(0);

        UnpackTileToCache(kvHead, kvStart, rows, dequantCache, signsCache, normRow, gammaRow, seqLen);

        const uint32_t validRows =
            (static_cast<int32_t>(kvStart) >= seqLen) ?
                0U :
                ((kvStart + rows <= static_cast<uint32_t>(seqLen)) ? rows : (static_cast<uint32_t>(seqLen) - kvStart));

        LocalTensor<float> queryFp32 = queryFp32Buf_.Get<float>();
        LocalTensor<float> qjlFp32 = qjlFp32Buf_.Get<float>();
        LocalTensor<float> scoreTile = scoreTileBuf_.Get<float>();
        LocalTensor<float> corr = dequantBuf_.Get<float>();
        LocalTensor<float> tmpMat = dotTmpBuf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();

        for (uint32_t token = tokenStart; token < tokenEnd; ++token) {
            LoadQueryPair(token, kvHead, queryFp32, qjlFp32);

            if (validRows > 0U) {
                DotBatchRows(queryFp32, dequantCache, tmpMat, work, scoreTile, validRows);
                DotBatchRows(qjlFp32, signsCache, tmpMat, work, corr, validRows);
                FuseNormGammaCorr(scoreTile, corr, normRow, gammaRow, validRows);
            }

            if (validRows < rows) {
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                MaskTailNegInf(scoreTile, validRows, rows);
                SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            }
            WriteScoreRow(token, kvHead, kvStart, rows, scoreTile);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessQueryRow(uint64_t row, uint32_t kvStart)
    {
        const uint32_t token = static_cast<uint32_t>(row / numQHeads_);
        const uint32_t qHead = static_cast<uint32_t>(row % numQHeads_);
        const uint32_t kvHead = qHead / groups_;
        const uint32_t batchIdx = (maxBatch_ == 1U) ? 0U : token;
        const uint64_t outRowOffset = row * maxKvLen_;

        LocalTensor<float> queryFp32 = queryFp32Buf_.Get<float>();
        LocalTensor<float> qjlFp32 = qjlFp32Buf_.Get<float>();
        LoadQueryPair(token, qHead, queryFp32, qjlFp32);

        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        const int32_t seqLen = seqLensGm_.GetValue(batchIdx);

        LocalTensor<float> dequant = dequantBuf_.Get<float>();
        LocalTensor<float> signs = signsBuf_.Get<float>();
        LocalTensor<float> tmp = tmpBuf_.Get<float>();
        LocalTensor<float> work = reduceWorkBuf_.Get<float>();
        LocalTensor<float> scalar = scalarBuf_.Get<float>();
        LocalTensor<float> centroids = centroidBuf_.Get<float>();
        LocalTensor<float> scoreTile = scoreTileBuf_.Get<float>();
        LocalTensor<float> normFp = normFpBuf_.Get<float>();
        LocalTensor<float> gammaFp = gammaFpBuf_.Get<float>();

        {
            const uint32_t rows = kvStart + kvTile_ <= maxKvLen_ ? kvTile_ : maxKvLen_ - kvStart;
            uint32_t i = 0;
            while (i < rows) {
                const uint32_t kvPos = kvStart + i;
                if (static_cast<int32_t>(kvPos) >= seqLen) {
                    for (uint32_t j = i; j < rows; ++j) {
                        scoreTile.SetValue(j, NEG_INF);
                    }
                    break;
                }

                const uint32_t logical0 = kvPos / blockSize_;
                const uint32_t offset0 = kvPos % blockSize_;
                const int32_t phys =
                    blockTableGm_.GetValue(static_cast<uint64_t>(batchIdx) * maxBlocksPerSeq_ + logical0);

                uint32_t run = 1;
                while (i + run < rows) {
                    const uint32_t p = kvStart + i + run;
                    if (static_cast<int32_t>(p) >= seqLen) {
                        break;
                    }
                    if ((p / blockSize_) != logical0 || (p % blockSize_) != (offset0 + run)) {
                        break;
                    }
                    ++run;
                }

                const uint64_t baseScalar = (static_cast<uint64_t>(phys) * blockSize_ + offset0) * numKvHeads_;
                const uint32_t span = run * numKvHeads_;

                LocalTensor<uint8_t> idxLocal = idxInQue_.AllocTensor<uint8_t>();
                CopyBytes(idxLocal, keyIdxGm_[baseScalar * idxPackedBytes_], span * idxPackedBytes_);
                idxInQue_.EnQue(idxLocal);
                LocalTensor<uint8_t> qjlPackedLocal = qjlPackedInQue_.AllocTensor<uint8_t>();
                CopyBytes(qjlPackedLocal, keyQjlGm_[baseScalar * qjlPackedBytes_], span * qjlPackedBytes_);
                qjlPackedInQue_.EnQue(qjlPackedLocal);
                LocalTensor<NormT> normLocal = normInQue_.AllocTensor<NormT>();
                {
                    DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                    DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                    DataCopyPad(normLocal, keyNormGm_[baseScalar], c, p);
                }
                normInQue_.EnQue(normLocal);
                LocalTensor<NormT> gammaLocal = gammaInQue_.AllocTensor<NormT>();
                {
                    DataCopyExtParams c{1, static_cast<uint32_t>(span * sizeof(NormT)), 0, 0, 0};
                    DataCopyPadExtParams<NormT> p{false, 0, 0, 0};
                    DataCopyPad(gammaLocal, keyGammaGm_[baseScalar], c, p);
                }
                gammaInQue_.EnQue(gammaLocal);

                idxLocal = idxInQue_.DeQue<uint8_t>();
                qjlPackedLocal = qjlPackedInQue_.DeQue<uint8_t>();
                normLocal = normInQue_.DeQue<NormT>();
                gammaLocal = gammaInQue_.DeQue<NormT>();

                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Cast(normFp, normLocal, RoundMode::CAST_NONE, span);
                Cast(gammaFp, gammaLocal, RoundMode::CAST_NONE, span);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                SetFlag<HardEvent::V_S>(EVENT_ID1);
                WaitFlag<HardEvent::V_S>(EVENT_ID1);

                for (uint32_t r = 0; r < run; ++r) {
                    const uint32_t slot = r * numKvHeads_ + kvHead;
                    UnpackPrimaryAt(idxLocal, slot * idxPackedBytes_, dequant, centroids);
                    UnpackSignsAt(qjlPackedLocal, slot * qjlPackedBytes_, signs);
                    const float normVal = normFp.GetValue(slot);
                    const float gammaVal = gammaFp.GetValue(slot);

                    SetFlag<HardEvent::S_V>(EVENT_ID0);
                    WaitFlag<HardEvent::S_V>(EVENT_ID0);
                    const float mainDot = DotProduct(queryFp32, dequant, tmp, work, scalar);
                    const float corrDot = DotProduct(qjlFp32, signs, tmp, work, scalar);
                    scoreTile.SetValue(i + r, normVal * mainDot + qjlCoeff_ * gammaVal * corrDot);
                }

                idxInQue_.FreeTensor(idxLocal);
                qjlPackedInQue_.FreeTensor(qjlPackedLocal);
                normInQue_.FreeTensor(normLocal);
                gammaInQue_.FreeTensor(gammaLocal);
                i += run;
            }

            SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
            LocalTensor<float> outLocal = outputQue_.AllocTensor<float>();
            for (uint32_t i = 0; i < rows; ++i) {
                outLocal.SetValue(i, scoreTile.GetValue(i));
            }
            outputQue_.EnQue(outLocal);
            outLocal = outputQue_.DeQue<float>();
            CopyScores(outRowOffset + kvStart, outLocal, rows);
            outputQue_.FreeTensor(outLocal);
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> queryInQue_;
    TQue<TPosition::VECIN, 1> qjlInQue_;
    TQue<TPosition::VECIN, 1> idxInQue_;
    TQue<TPosition::VECIN, 1> qjlPackedInQue_;
    TQue<TPosition::VECIN, 1> normInQue_;
    TQue<TPosition::VECIN, 1> gammaInQue_;
    TQue<TPosition::VECOUT, 1> outputQue_;
    TBuf<TPosition::VECCALC> queryFp32Buf_;
    TBuf<TPosition::VECCALC> qjlFp32Buf_;
    TBuf<TPosition::VECCALC> queryG1Buf_;
    TBuf<TPosition::VECCALC> qjlG1Buf_;
    TBuf<TPosition::VECCALC> queryG2Buf_;
    TBuf<TPosition::VECCALC> qjlG2Buf_;
    TBuf<TPosition::VECCALC> queryG3Buf_;
    TBuf<TPosition::VECCALC> qjlG3Buf_;
    TBuf<TPosition::VECCALC> dequantBuf_;
    TBuf<TPosition::VECCALC> signsBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> reduceWorkBuf_;
    TBuf<TPosition::VECCALC> centroidBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> scoreTileBuf_;
    TBuf<TPosition::VECCALC> scoreTile1Buf_;
    TBuf<TPosition::VECCALC> scoreTile2Buf_;
    TBuf<TPosition::VECCALC> scoreTile3Buf_;
    TBuf<TPosition::VECCALC> normFpBuf_;
    TBuf<TPosition::VECCALC> gammaFpBuf_;
    TBuf<TPosition::VECCALC> gatherOffBuf_;
    TBuf<TPosition::VECCALC> signsLutBuf_;
    TBuf<TPosition::VECCALC> dequantCacheBuf_;
    TBuf<TPosition::VECCALC> signsCacheBuf_;
    TBuf<TPosition::VECCALC> normRowBuf_;
    TBuf<TPosition::VECCALC> gammaRowBuf_;
    TBuf<TPosition::VECCALC> dotTmpBuf_;
    TBuf<TPosition::VECCALC> reduceTvBuf_;

    GlobalTensor<bfloat16_t> queryRotatedGm_;
    GlobalTensor<bfloat16_t> queryQjlGm_;
    GlobalTensor<uint8_t> keyIdxGm_;
    GlobalTensor<uint8_t> keyQjlGm_;
    GlobalTensor<NormT> keyNormGm_;
    GlobalTensor<NormT> keyGammaGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<float> outputGm_;

    uint32_t numTokens_ = 0;
    uint32_t numQHeads_ = 0;
    uint32_t numKvHeads_ = 0;
    uint32_t headDim_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t idxPackedBytes_ = 0;
    uint32_t qjlPackedBytes_ = 0;
    uint32_t maxBatch_ = 0;
    uint32_t maxBlocksPerSeq_ = 0;
    uint32_t maxKvLen_ = 0;
    uint32_t kvTile_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t groups_ = 0;
    uint32_t ubBlockSize_ = 0;
    uint32_t kShare_ = 2;
    uint32_t mseBits_ = 3;
    uint64_t totalRows_ = 0;
    float qjlCoeff_ = 0.0F;
    bool isPrefill_ = false;
};
} // namespace NsTurboQuantAttentionScore
#endif // TURBO_QUANT_ATTENTION_SCORE_IMPL_HPP_
