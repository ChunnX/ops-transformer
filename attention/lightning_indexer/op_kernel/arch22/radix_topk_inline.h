#ifndef LIGHTNING_INDEXER_RADIX_TOPK_INLINE_H
#define LIGHTNING_INDEXER_RADIX_TOPK_INLINE_H

#include "kernel_operator.h"

namespace LIKernel {
using namespace AscendC;

static constexpr uint32_t RDX_BITS = 2;
static constexpr uint32_t RDX_BINS = 4;
static constexpr uint32_t RDX_ROUNDS = 8;
static constexpr int16_t RDX_XOR = 0x8000;

#define RDX_V_S_SYNC() do { event_t _e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S)); \
    SetFlag<HardEvent::V_S>(_e); WaitFlag<HardEvent::V_S>(_e); } while(0)
#define RDX_S_V_SYNC() do { event_t _e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V)); \
    SetFlag<HardEvent::S_V>(_e); WaitFlag<HardEvent::S_V>(_e); } while(0)

template <typename T = void>
__aicore__ inline void InlineRadixTopK(
    TPipe *pipe,
    __gm__ uint8_t *scoreGmAddr,
    __gm__ uint8_t *indicesGmAddr,
    __gm__ uint8_t *wsGmAddr,
    uint32_t sortLen,
    uint32_t sortLenAlign,
    uint32_t kValue,
    uint32_t tileLen,
    uint32_t coreId,
    uint32_t coreNum,
    uint32_t debugBatchId = 0xFFFFFFFF)  // debug: which batch to trace
{
    const uint32_t UB_BLOCK_SIZE = 32;
    const uint32_t REPEAT_SIZE = 256;

    uint32_t tileLen_ = tileLen;
    uint32_t tileLenAlign_ = (tileLen + 7) / 8 * 8;
    uint32_t totalTiles = (sortLen + tileLen - 1) / tileLen;
    if (totalTiles == 0) { totalTiles = 1; }
    uint32_t tileNum_ = totalTiles;
    uint32_t tailTileLen_ = sortLen - (tileNum_ - 1) * tileLen;
    if (tailTileLen_ == 0) { tailTileLen_ = tileLen_; }
    uint32_t tileNumAlign_ = (tileNum_ + 7) / 8 * 8;

    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> outValueBuf;
    TBuf<TPosition::VECCALC> outIndexBuf;
    TBuf<TPosition::VECCALC> globalHistBuf;
    TBuf<TPosition::VECCALC> tileTopKBuf;
    TBuf<TPosition::VECCALC> tempBuf;

    pipe->Reset();
    pipe->InitBuffer(xBuf, tileLenAlign_ * sizeof(float) * 2 + 64);
    pipe->InitBuffer(outValueBuf, tileLenAlign_ * sizeof(float) + 64);
    pipe->InitBuffer(outIndexBuf, tileLenAlign_ * sizeof(int32_t) + 64);
    pipe->InitBuffer(globalHistBuf, RDX_BINS * sizeof(int32_t) + 64);
    pipe->InitBuffer(tileTopKBuf, tileNum_ * sizeof(int32_t) + 64);
    uint32_t cmpMaskSize = (tileLenAlign_ + 7) / 8 * 8;
    uint32_t tileHistSize = tileNumAlign_ * RDX_BINS * sizeof(int32_t);
    pipe->InitBuffer(tempBuf, tileHistSize + cmpMaskSize + 64);

    GlobalTensor<bfloat16_t> xGm;
    xGm.SetGlobalBuffer((__gm__ bfloat16_t *)scoreGmAddr);
    GlobalTensor<int32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t *)indicesGmAddr);
    GlobalTensor<int32_t> dbgGm;
    dbgGm.SetGlobalBuffer((__gm__ int32_t *)wsGmAddr + coreNum * RDX_BINS + coreNum);  // after coreTopK

    bool dbg = false;
    uint32_t dbgPos = 0;

    LocalTensor<bfloat16_t> xLocalB16 = xBuf.Get<bfloat16_t>();
    LocalTensor<int16_t> xLocalInt16 = xLocalB16[tileLen_].template ReinterpretCast<int16_t>();
    LocalTensor<int32_t> xLocalInt32 = xLocalInt16.template ReinterpretCast<int32_t>();
    LocalTensor<float> xLocalFp32 = xLocalInt16.template ReinterpretCast<float>();

    LocalTensor<int16_t> signMaskTensor = outValueBuf.Get<int16_t>();
    LocalTensor<int32_t> maskTensor = outValueBuf.Get<int32_t>();
    LocalTensor<half> tempHalf = outValueBuf.Get<half>();
    LocalTensor<float> outValueLocalFp32 = outValueBuf.Get<float>();

    LocalTensor<uint8_t> cmpMaskTensor = outIndexBuf.Get<uint8_t>();
    LocalTensor<int16_t> tempTensor = outIndexBuf.Get<int16_t>();
    LocalTensor<int32_t> outIndexLocal = outIndexBuf.Get<int32_t>();

    // Debug: skip RadixTopK for debug batch, just output score values
    if (dbg) {
        for (uint32_t i = 5; i < kValue; i++) { indexGm.SetValue(i, -1); }
        return;
    }

    // cmpMaskTopK reuses outValueBuf during TileTopK phase (signMaskTensor not needed then)
    LocalTensor<uint8_t> cmpMaskTopK = outValueBuf.Get<uint8_t>();
    LocalTensor<int32_t> tileHist = tempBuf.Get<int32_t>();

    LocalTensor<int32_t> globalHist = globalHistBuf.Get<int32_t>();
    LocalTensor<int32_t> tileTopK = tileTopKBuf.Get<int32_t>();

    int16_t involvedMask16_ = 0;
    int16_t andMask16_ = static_cast<int16_t>(0xC000);
    int32_t totalDefinitelyInTopK_ = 0;
    int32_t boundaryBin = 0;
    int32_t boundaryBinPrev = 1;
    int32_t globalHistBoundaryNum_ = static_cast<int32_t>((tileNum_ - 1) * tileLen_ + tailTileLen_);
    int32_t remainK_ = static_cast<int32_t>(kValue);

    Duplicate<int32_t>(tileTopK, 0, tileNum_);
    PipeBarrier<PIPE_V>();

    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

    for (int32_t roundId = RDX_ROUNDS - 1; roundId >= 0; roundId--) {
        // ClearHist
        if (roundId == static_cast<int32_t>(RDX_ROUNDS) - 1) {
            Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), tileNum_);
            RDX_V_S_SYNC();
            if (tailTileLen_ != tileLen_) {
                tileHist.SetValue(tileNum_ - 1, static_cast<int32_t>(tailTileLen_));
            }
            globalHistBoundaryNum_ = static_cast<int32_t>((tileNum_ - 1) * tileLen_ + tailTileLen_);
            RDX_S_V_SYNC();
        } else {
            RDX_V_S_SYNC();
            if (boundaryBin < static_cast<int32_t>(RDX_BINS) - 1) {
                for (uint32_t i = 0; i < tileNum_; i++) {
                    tileHist.SetValue(i, tileHist.GetValue(boundaryBin * tileNumAlign_ + i) -
                                         tileHist.GetValue(boundaryBinPrev * tileNumAlign_ + i));
                }
            } else {
                for (uint32_t i = 0; i < tileNum_; i++) {
                    tileHist.SetValue(i, tileHist.GetValue(boundaryBin * tileNumAlign_ + i));
                }
            }
            RDX_S_V_SYNC();
        }

        RDX_S_V_SYNC();
        globalHist.SetValue(0, globalHistBoundaryNum_);
        for (uint32_t i = 1; i < RDX_BINS; i++) {
            globalHist.SetValue(i, 0);
        }
        for (uint32_t i = tileNumAlign_; i < tileNumAlign_ * RDX_BINS; i++) {
            tileHist.SetValue(i, 0);
        }
        RDX_S_V_SYNC();

        for (uint32_t tileId = 0; tileId < tileNum_; tileId++) {
            uint32_t curTileLen = (tileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
            if (curTileLen == 0) { continue; }

            DataCopyPad(xLocalB16[tileLen_], xGm[tileId * tileLen_], DataCopyExtParams{
                static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(bfloat16_t) * curTileLen), 0, 0, 0},
                DataCopyPadExtParams<bfloat16_t>{true, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);

            uint64_t calcLen = curTileLen;
            uint64_t curTileLenAlign = (curTileLen + UB_BLOCK_SIZE / sizeof(int16_t) - 1) /
                (UB_BLOCK_SIZE / sizeof(int16_t)) * (UB_BLOCK_SIZE / sizeof(int16_t));

            // TwiddleInB16
            LocalTensor<int16_t> temp1 = tempTensor[curTileLenAlign];
            ShiftRight(tempTensor, xLocalInt16, (int16_t)15, curTileLen);
            Duplicate<int16_t>(signMaskTensor, RDX_XOR, curTileLen);
            PipeBarrier<PIPE_V>();
            Or(tempTensor, tempTensor, signMaskTensor, calcLen);
            PipeBarrier<PIPE_V>();
            And(temp1, xLocalInt16, tempTensor, calcLen);
            Or(xLocalInt16, xLocalInt16, tempTensor, calcLen);
            PipeBarrier<PIPE_V>();
            Not(temp1, temp1, calcLen);
            PipeBarrier<PIPE_V>();
            And(xLocalInt16, xLocalInt16, temp1, calcLen);

            // DoAndMask
            Duplicate<int16_t>(maskTensor.template ReinterpretCast<int16_t>(), andMask16_, curTileLen);
            PipeBarrier<PIPE_V>();
            And(xLocalInt16, xLocalInt16, maskTensor.template ReinterpretCast<int16_t>(), calcLen);

            // CalcCumsumHistogram
            Cast(xLocalFp32, xLocalInt16, RoundMode::CAST_NONE, curTileLen);
            PipeBarrier<PIPE_V>();
            Cast(xLocalInt32, xLocalFp32, RoundMode::CAST_RINT, curTileLen);
            PipeBarrier<PIPE_V>();

            int32_t cumSumOne = 0;
            uint64_t curTileLenAlign32 = (curTileLen + REPEAT_SIZE / sizeof(int32_t) - 1) /
                (REPEAT_SIZE / sizeof(int32_t)) * (REPEAT_SIZE / sizeof(int32_t));
            for (int16_t binMask = RDX_BINS - 1; binMask > 0; binMask--) {
                int32_t involvedMaskTemp = static_cast<int32_t>(
                    involvedMask16_ | static_cast<int16_t>(binMask << (roundId * RDX_BITS)));
                Compares<int32_t, uint8_t>(cmpMaskTensor, xLocalInt32, involvedMaskTemp,
                    AscendC::CMPMODE::EQ, curTileLenAlign32);
                PipeBarrier<PIPE_V>();
                uint64_t rsvdCnt = 0;
                GatherMask(tempHalf, tempHalf, cmpMaskTensor.template ReinterpretCast<uint16_t>(),
                           true, curTileLen, {1, 1, 0, 0}, rsvdCnt);
                RDX_V_S_SYNC();
                cumSumOne += static_cast<int32_t>(rsvdCnt);
                tileHist[binMask * tileNumAlign_].SetValue(tileId, cumSumOne);
                globalHist.SetValue(binMask, globalHist.GetValue(binMask) + cumSumOne);
                RDX_S_V_SYNC();
            }
        }

        // FindBoundaryBin
        RDX_V_S_SYNC();
        boundaryBin = -1;
        boundaryBinPrev = -1;
        for (int32_t binMask = RDX_BINS - 1; binMask >= 0; binMask--) {
            if (globalHist.GetValue(binMask) >= remainK_) {
                boundaryBin = binMask;
                boundaryBinPrev = binMask + 1;
                break;
            }
        }
        involvedMask16_ = involvedMask16_ |
            (static_cast<int16_t>(boundaryBin) << (roundId * RDX_BITS));
        RDX_S_V_SYNC();

        // Update globalHistBoundaryNum_ for next round's ClearHist
        RDX_V_S_SYNC();
        globalHistBoundaryNum_ = globalHist.GetValue(boundaryBin);
        if (boundaryBin < static_cast<int32_t>(RDX_BINS) - 1) {
            globalHistBoundaryNum_ -= globalHist.GetValue(boundaryBinPrev);
        }
        RDX_S_V_SYNC();

        // AddTileHistToTileTopK
        RDX_V_S_SYNC();
        if (globalHist.GetValue(boundaryBin) == remainK_) {
            for (uint32_t i = 0; i < tileNum_; i++) {
                tileTopK.SetValue(i, tileTopK.GetValue(i) +
                    tileHist.GetValue(boundaryBin * tileNumAlign_ + i));
            }
            totalDefinitelyInTopK_ += globalHist.GetValue(boundaryBin);
        } else if (boundaryBinPrev < static_cast<int32_t>(RDX_BINS)) {
            for (uint32_t i = 0; i < tileNum_; i++) {
                tileTopK.SetValue(i, tileTopK.GetValue(i) +
                    tileHist.GetValue(boundaryBinPrev * tileNumAlign_ + i));
            }
            totalDefinitelyInTopK_ += globalHist.GetValue(boundaryBinPrev);
        }
        RDX_S_V_SYNC();

        remainK_ = static_cast<int32_t>(kValue) - totalDefinitelyInTopK_;

        // HandleLastRoundBoundary
        if (roundId == 0 && remainK_ > 0) {
            RDX_V_S_SYNC();
            int32_t remainToDistribute = remainK_;
            for (uint32_t tileId = 0; tileId < tileNum_ && remainToDistribute > 0; tileId++) {
                int32_t binCount;
                if (boundaryBin < static_cast<int32_t>(RDX_BINS) - 1) {
                    binCount = tileHist.GetValue(boundaryBin * tileNumAlign_ + tileId) -
                               tileHist.GetValue(boundaryBinPrev * tileNumAlign_ + tileId);
                } else {
                    binCount = tileHist.GetValue(boundaryBin * tileNumAlign_ + tileId);
                }
                int32_t add = (remainToDistribute < binCount) ? remainToDistribute : binCount;
                tileTopK.SetValue(tileId, tileTopK.GetValue(tileId) + add);
                remainToDistribute -= add;
            }
            totalDefinitelyInTopK_ = static_cast<int32_t>(kValue);
            remainK_ = 0;
            RDX_S_V_SYNC();
        }

        andMask16_ >>= RDX_BITS;
        if (remainK_ == 0 || roundId == 0) break;
    }

    // Compute coreTopKOffset
    RDX_V_S_SYNC();
    int32_t coreTopKOffset = 0;
    for (uint32_t i = 0; i < tileNum_; i++) {
        coreTopKOffset += tileTopK.GetValue(i);
    }
    RDX_S_V_SYNC();

    int16_t boundaryNumber16 = involvedMask16_ ^ ((~(involvedMask16_ >> 15)) | 0x8000);
    bfloat16_t boundaryBf16 = *reinterpret_cast<bfloat16_t *>(&boundaryNumber16);
    float boundaryFloat = AscendC::Cast(boundaryBf16);

    uint32_t writeOffset = 0;
    for (uint32_t tileId = 0; tileId < tileNum_; tileId++) {
        RDX_V_S_SYNC();
        int32_t curTileK = tileTopK.GetValue(tileId);
        RDX_S_V_SYNC();
        if (curTileK <= 0) { continue; }

        uint32_t curTileLen = (tileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;

        DataCopyPad(xLocalB16[tileLen_], xGm[tileId * tileLen_], DataCopyExtParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(bfloat16_t) * curTileLen), 0, 0, 0},
            DataCopyPadExtParams<bfloat16_t>{true, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        int32_t tileOffset = static_cast<int32_t>(tileId * tileLen_);

        // Scalar GT + EQ: read each score, compare with boundary, write index if match
        RDX_V_S_SYNC();
        for (uint32_t i = 0; i < curTileLen && curTileK > 0; i++) {
            bfloat16_t score = xLocalB16[tileLen_].GetValue(i);
            float scoreFloat = AscendC::Cast(score);
            if (scoreFloat > boundaryFloat) {
                indexGm.SetValue(writeOffset, static_cast<int32_t>(i) + tileOffset);
                writeOffset++;
                curTileK--;
            }
        }
        for (uint32_t i = 0; i < curTileLen && curTileK > 0; i++) {
            bfloat16_t score = xLocalB16[tileLen_].GetValue(i);
            float scoreFloat = AscendC::Cast(score);
            if (scoreFloat == boundaryFloat) {
                indexGm.SetValue(writeOffset, static_cast<int32_t>(i) + tileOffset);
                writeOffset++;
                curTileK--;
            }
        }
        RDX_S_V_SYNC();
    }

    for (uint32_t i = writeOffset; i < kValue; i++) {
        indexGm.SetValue(i, -1);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV);
}

}

#endif
