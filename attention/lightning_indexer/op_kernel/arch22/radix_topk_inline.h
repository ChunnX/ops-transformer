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

#define RDX_UB2UB(dst, src, count) do { \
    uint32_t _c = (count); \
    DataCopy(dst, src, _c > 2048 ? 2048 : _c); \
    if (_c > 2048) { DataCopy(dst[2048], src[2048], _c - 2048); } \
} while(0)

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
    uint32_t debugBatchId = 0xFFFFFFFF)
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

    TBuf<TPosition::VECCALC> xBufPing;
    TBuf<TPosition::VECCALC> xBufPong;
    TBuf<TPosition::VECCALC> outValueBuf;
    TBuf<TPosition::VECCALC> outIndexBuf;
    TBuf<TPosition::VECCALC> globalHistBuf;
    TBuf<TPosition::VECCALC> tileTopKBuf;
    TBuf<TPosition::VECCALC> tempBuf;
    TBuf<TPosition::VECCALC> scoreCacheBuf;

    pipe->Reset();
    uint32_t xBufSize = tileLenAlign_ * sizeof(float) + tileLenAlign_ * sizeof(bfloat16_t) + 64;
    pipe->InitBuffer(xBufPing, xBufSize);
    pipe->InitBuffer(xBufPong, xBufSize);
    pipe->InitBuffer(outValueBuf, tileLenAlign_ * sizeof(float) + 64);
    pipe->InitBuffer(outIndexBuf, tileLenAlign_ * sizeof(int32_t) + 64);
    pipe->InitBuffer(globalHistBuf, RDX_BINS * sizeof(int32_t) + 64);
    pipe->InitBuffer(tileTopKBuf, tileNum_ * sizeof(int32_t) + 64);
    uint32_t cmpMaskSize = (tileLenAlign_ + 7) / 8 * 8;
    uint32_t tileHistSize = tileNumAlign_ * RDX_BINS * sizeof(int32_t);
    pipe->InitBuffer(tempBuf, tileHistSize + cmpMaskSize + 64);
    // Cache all score data in UB to avoid repeated GM reads (8 rounds + 1 extraction = 9 reads)
    uint32_t scoreCacheSize = sortLenAlign * sizeof(float);
    pipe->InitBuffer(scoreCacheBuf, scoreCacheSize + 64);

    GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer((__gm__ float *)scoreGmAddr);
    GlobalTensor<int32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t *)indicesGmAddr);

    // outValueBuf: signMask, mask, tempHalf
    LocalTensor<int16_t> signMaskTensor = outValueBuf.Get<int16_t>();
    LocalTensor<int32_t> maskTensor = outValueBuf.Get<int32_t>();
    LocalTensor<half> tempHalf = outValueBuf.Get<half>();

    LocalTensor<uint8_t> cmpMaskTensor = outIndexBuf.Get<uint8_t>();
    LocalTensor<int16_t> tempTensor = outIndexBuf.Get<int16_t>();
    LocalTensor<int32_t> outIndexLocal = outIndexBuf.Get<int32_t>();

    LocalTensor<uint8_t> cmpMaskTopK = outValueBuf.Get<uint8_t>();
    LocalTensor<int32_t> tileHist = tempBuf.Get<int32_t>();

    LocalTensor<int32_t> globalHist = globalHistBuf.Get<int32_t>();
    LocalTensor<int32_t> tileTopK = tileTopKBuf.Get<int32_t>();
    LocalTensor<float> scoreCache = scoreCacheBuf.Get<float>();

    int16_t involvedMask16_ = 0;
    int16_t andMask16_ = static_cast<int16_t>(0xC000);  // signed: arithmetic shift accumulates
    int32_t totalDefinitelyInTopK_ = 0;
    int32_t boundaryBin = 0;
    int32_t boundaryBinPrev = 1;
    int32_t globalHistBoundaryNum_ = static_cast<int32_t>((tileNum_ - 1) * tileLen_ + tailTileLen_);
    int32_t remainK_ = static_cast<int32_t>(kValue);

    Duplicate<int32_t>(tileTopK, 0, tileNum_);
    PipeBarrier<PIPE_V>();
    RDX_V_S_SYNC();

    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

    // One-time load all score data from GM to UB cache
    DataCopyPad(scoreCache, xGm, DataCopyExtParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * sortLen), 0, 0, 0},
        DataCopyPadExtParams<float>{true, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    WaitFlag<HardEvent::MTE2_V>(mte2ToV);

    for (int32_t roundId = RDX_ROUNDS - 1; roundId >= 0; roundId--) {
        // ClearHist
        if (roundId == static_cast<int32_t>(RDX_ROUNDS) - 1) {
            Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), tileNum_);
            PipeBarrier<PIPE_V>();
            RDX_V_S_SYNC();
            if (tailTileLen_ != tileLen_) {
                tileHist.SetValue(tileNum_ - 1, static_cast<int32_t>(tailTileLen_));
            }
            globalHistBoundaryNum_ = static_cast<int32_t>((tileNum_ - 1) * tileLen_ + tailTileLen_);
            RDX_S_V_SYNC();
        } else {
            // Vectorized ClearHist: tileHist[0..tileNum_) = tileHist[boundaryBin] - tileHist[boundaryBinPrev]
            if (boundaryBin < static_cast<int32_t>(RDX_BINS) - 1) {
                Sub<int32_t>(tileHist, tileHist[boundaryBin * tileNumAlign_],
                             tileHist[boundaryBinPrev * tileNumAlign_], tileNumAlign_);
            } else {
                DataCopy(tileHist, tileHist[boundaryBin * tileNumAlign_], tileNumAlign_);
            }
            PipeBarrier<PIPE_V>();
        }

        // Vectorized clear: globalHist and tileHist[1..BINS-1]
        Duplicate<int32_t>(globalHist, 0, RDX_BINS);
        Duplicate<int32_t>(tileHist[tileNumAlign_], 0, tileNumAlign_ * (RDX_BINS - 1));
        PipeBarrier<PIPE_V>();
        RDX_V_S_SYNC();
        if (tailTileLen_ != tileLen_) {
            tileHist.SetValue(tileNum_ - 1, static_cast<int32_t>(tailTileLen_));
        }
        globalHistBoundaryNum_ = static_cast<int32_t>((tileNum_ - 1) * tileLen_ + tailTileLen_);
        // Initialize scalar globalHist accumulators (avoid per-tile GetValue)
        int32_t gh[4] = {globalHistBoundaryNum_, 0, 0, 0};
        RDX_S_V_SYNC();

        // Ping-Pong tile loop: overlap UB→UB DataCopy with V computation
        // Prefetch first tile to xBufPing
        RDX_UB2UB(xBufPing.Get<float>(), scoreCache[0], tileLenAlign_);
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        bool pingPongFlag = true;

        for (uint32_t tileId = 0; tileId < tileNum_; tileId++) {
            uint32_t curTileLen = (tileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
            if (curTileLen == 0) { continue; }

            // Select current and next buffer
            TBuf<TPosition::VECCALC>& curBuf = pingPongFlag ? xBufPing : xBufPong;
            TBuf<TPosition::VECCALC>& nextBuf = pingPongFlag ? xBufPong : xBufPing;

            // Prefetch next tile (overlap with current tile's V computation)
            if (tileId < tileNum_ - 1) {
                RDX_UB2UB(nextBuf.Get<float>(), scoreCache[(tileId + 1) * tileLen_], tileLenAlign_);
            }

            // Wait for current tile data ready
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);

            // Get LocalTensor from current buffer
            LocalTensor<float> xLocalFp32 = curBuf.Get<float>();
            LocalTensor<int32_t> xLocalInt32 = xLocalFp32.template ReinterpretCast<int32_t>();
            LocalTensor<bfloat16_t> xLocalB16Base = curBuf.Get<bfloat16_t>();
            LocalTensor<bfloat16_t> xLocalB16 = xLocalB16Base[tileLenAlign_ * 2];
            LocalTensor<int16_t> xLocalInt16 = xLocalB16.template ReinterpretCast<int16_t>();

            // Cast float -> bf16 into xLocalB16 (xBuf back, no overlap with xLocalFp32 in front)
            Cast(xLocalB16, xLocalFp32, RoundMode::CAST_ROUND, curTileLen);
            PipeBarrier<PIPE_V>();

            // TwiddleInB16 operates on xLocalInt16 (xBuf back)
            LocalTensor<int16_t> xLocalInt16Cast = xLocalInt16;

            uint64_t calcLen = curTileLen;
            uint64_t curTileLenAlign = (curTileLen + UB_BLOCK_SIZE / sizeof(int16_t) - 1) /
                (UB_BLOCK_SIZE / sizeof(int16_t)) * (UB_BLOCK_SIZE / sizeof(int16_t));

            // TwiddleInB16
            LocalTensor<int16_t> temp1 = tempTensor[curTileLenAlign];
            ShiftRight(tempTensor, xLocalInt16Cast, (int16_t)15, curTileLen);
            Duplicate<int16_t>(signMaskTensor, RDX_XOR, curTileLen);
            PipeBarrier<PIPE_V>();
            Or(tempTensor, tempTensor, signMaskTensor, calcLen);
            PipeBarrier<PIPE_V>();
            And(temp1, xLocalInt16Cast, tempTensor, calcLen);
            Or(xLocalInt16Cast, xLocalInt16Cast, tempTensor, calcLen);
            PipeBarrier<PIPE_V>();
            Not(temp1, temp1, calcLen);
            PipeBarrier<PIPE_V>();
            And(xLocalInt16Cast, xLocalInt16Cast, temp1, calcLen);

            // DoAndMask: andMask16_ accumulates via arithmetic shift (int16_t)
            Duplicate<int16_t>(maskTensor.template ReinterpretCast<int16_t>(), andMask16_, curTileLen);
            PipeBarrier<PIPE_V>();
            And(xLocalInt16Cast, xLocalInt16Cast, maskTensor.template ReinterpretCast<int16_t>(), calcLen);
            PipeBarrier<PIPE_V>();

            // CalcCumsumHistogram: Cast int16 -> float -> int32, compare int32 (matching math repo)
            Cast(xLocalFp32, xLocalInt16Cast, RoundMode::CAST_NONE, curTileLen);
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
                gh[binMask] += cumSumOne;
            }

            // Signal next tile data ready, flip ping-pong flag
            if (tileId < tileNum_ - 1) {
                SetFlag<HardEvent::MTE2_V>(mte2ToV);
            }
            pingPongFlag = !pingPongFlag;
        }

        // Write globalHist from scalar accumulators (already in S after tile loop)
        globalHist.SetValue(0, gh[0]);
        globalHist.SetValue(1, gh[1]);
        globalHist.SetValue(2, gh[2]);
        globalHist.SetValue(3, gh[3]);

        // FindBoundaryBin (all S ops, no sync needed between S operations)
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

        globalHistBoundaryNum_ = globalHist.GetValue(boundaryBin);
        if (boundaryBin < static_cast<int32_t>(RDX_BINS) - 1) {
            globalHistBoundaryNum_ -= globalHist.GetValue(boundaryBinPrev);
        }

        int32_t globalHistBoundary = globalHist.GetValue(boundaryBin);
        int32_t globalHistBoundaryPrev = (boundaryBinPrev < static_cast<int32_t>(RDX_BINS))
                                         ? globalHist.GetValue(boundaryBinPrev) : 0;
        RDX_S_V_SYNC();

        // AddTileHistToTileTopK (vectorized: tileTopK += tileHist[bin])
        if (globalHistBoundary == remainK_) {
            Add<int32_t>(tileTopK, tileTopK, tileHist[boundaryBin * tileNumAlign_], tileNumAlign_);
            PipeBarrier<PIPE_V>();
            totalDefinitelyInTopK_ += globalHistBoundary;
        } else if (boundaryBinPrev < static_cast<int32_t>(RDX_BINS)) {
            Add<int32_t>(tileTopK, tileTopK, tileHist[boundaryBinPrev * tileNumAlign_], tileNumAlign_);
            PipeBarrier<PIPE_V>();
            totalDefinitelyInTopK_ += globalHistBoundaryPrev;
        }

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

    // Compute coreTopKOffset (vectorized ReduceSum)
    RDX_V_S_SYNC();
    int32_t coreTopKOffset = 0;
    // tileNum_ is small (<=16 for 8192/512), scalar sum is fast enough
    for (uint32_t i = 0; i < tileNum_; i++) {
        coreTopKOffset += tileTopK.GetValue(i);
    }
    RDX_S_V_SYNC();

    // ExtractBoundaryNumber (inverse twiddle for int16)
    int16_t boundaryNumber16 = involvedMask16_ ^ ((~(involvedMask16_ >> 15)) | 0x8000);
    bfloat16_t boundaryBf16 = *reinterpret_cast<bfloat16_t *>(&boundaryNumber16);
    float boundaryFloat = AscendC::Cast(boundaryBf16);

    // Reuse outValueBuf for vecIndex during extraction
    LocalTensor<int32_t> vecIndex = outValueBuf.Get<int32_t>();

    // Generate sequential index [0, 1, 2, ..., tileLen_-1] once (reused for all tiles)
    RDX_S_V_SYNC();
    for (uint32_t i = 0; i < tileLen_; i++) {
        vecIndex.SetValue(i, static_cast<int32_t>(i));
    }
    RDX_S_V_SYNC();

    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    event_t mte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    bool mte3Pending = false;

    uint32_t writeOffset = 0;
    RDX_UB2UB(xBufPing.Get<float>(), scoreCache[0], tileLenAlign_);
    SetFlag<HardEvent::MTE2_V>(mte2ToV);
    bool ppFlag = true;

    for (uint32_t tileId = 0; tileId < tileNum_; tileId++) {
        RDX_V_S_SYNC();
        int32_t curTileK = tileTopK.GetValue(tileId);
        RDX_S_V_SYNC();
        if (curTileK <= 0) {
            if (tileId < tileNum_ - 1) {
                TBuf<TPosition::VECCALC>& nextBuf = ppFlag ? xBufPong : xBufPing;
                RDX_UB2UB(nextBuf.Get<float>(), scoreCache[(tileId + 1) * tileLen_], tileLenAlign_);
            }
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);
            if (tileId < tileNum_ - 1) {
                SetFlag<HardEvent::MTE2_V>(mte2ToV);
            }
            ppFlag = !ppFlag;
            continue;
        }

        if (mte3Pending) {
            WaitFlag<HardEvent::MTE3_V>(mte3ToV);
            mte3Pending = false;
        }

        uint32_t curTileLen = (tileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
        TBuf<TPosition::VECCALC>& curBuf = ppFlag ? xBufPing : xBufPong;
        TBuf<TPosition::VECCALC>& nextBuf = ppFlag ? xBufPong : xBufPing;
        if (tileId < tileNum_ - 1) {
            RDX_UB2UB(nextBuf.Get<float>(), scoreCache[(tileId + 1) * tileLen_], tileLenAlign_);
        }
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        LocalTensor<float> xLocalFp32 = curBuf.Get<float>();
        LocalTensor<bfloat16_t> xLocalB16Base = curBuf.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> xLocalB16 = xLocalB16Base[tileLenAlign_ * 2];

        Cast(xLocalB16, xLocalFp32, RoundMode::CAST_ROUND, curTileLen);
        PipeBarrier<PIPE_V>();
        Cast(xLocalFp32, xLocalB16, RoundMode::CAST_NONE, curTileLen);
        PipeBarrier<PIPE_V>();

        int32_t tileOffset = static_cast<int32_t>(tileId * tileLen_);
        uint64_t curTileLenAlign32 = (curTileLen + REPEAT_SIZE / sizeof(int32_t) - 1) /
            (REPEAT_SIZE / sizeof(int32_t)) * (REPEAT_SIZE / sizeof(int32_t));

        // GT pass
        Compares<float, uint8_t>(cmpMaskTensor, xLocalFp32, boundaryFloat,
            AscendC::CMPMODE::GT, curTileLenAlign32);
        PipeBarrier<PIPE_V>();

        uint64_t gtCnt = 0;
        GatherMask(outIndexLocal, vecIndex, cmpMaskTensor.template ReinterpretCast<uint32_t>(),
                   true, curTileLen, {1, 1, 0, 0}, gtCnt);
        PipeBarrier<PIPE_V>();
        RDX_V_S_SYNC();
        uint32_t gtCount = static_cast<uint32_t>(gtCnt);
        if (gtCount > static_cast<uint32_t>(curTileK)) { gtCount = static_cast<uint32_t>(curTileK); }

        if (gtCount > 0) {
            RDX_S_V_SYNC();
            Adds(outIndexLocal, outIndexLocal, tileOffset, gtCount);
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopyPad(indexGm[writeOffset], outIndexLocal, DataCopyExtParams{
                static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * gtCount), 0, 0, 0});
            SetFlag<HardEvent::MTE3_V>(mte3ToV);
            mte3Pending = true;
            writeOffset += gtCount;
            curTileK -= static_cast<int32_t>(gtCount);
        }

        if (curTileK > 0) {
            if (mte3Pending) {
                WaitFlag<HardEvent::MTE3_V>(mte3ToV);
                mte3Pending = false;
            }
            Compares<float, uint8_t>(cmpMaskTensor, xLocalFp32, boundaryFloat,
                AscendC::CMPMODE::EQ, curTileLenAlign32);
            PipeBarrier<PIPE_V>();

            uint64_t eqCnt = 0;
            GatherMask(outIndexLocal, vecIndex, cmpMaskTensor.template ReinterpretCast<uint32_t>(),
                       true, curTileLen, {1, 1, 0, 0}, eqCnt);
            PipeBarrier<PIPE_V>();
            RDX_V_S_SYNC();
            uint32_t eqCount = static_cast<uint32_t>(eqCnt);
            if (eqCount > static_cast<uint32_t>(curTileK)) { eqCount = static_cast<uint32_t>(curTileK); }

            if (eqCount > 0) {
                RDX_S_V_SYNC();
                Adds(outIndexLocal, outIndexLocal, tileOffset, eqCount);
                SetFlag<HardEvent::V_MTE3>(vToMte3);
                WaitFlag<HardEvent::V_MTE3>(vToMte3);
                DataCopyPad(indexGm[writeOffset], outIndexLocal, DataCopyExtParams{
                    static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * eqCount), 0, 0, 0});
                SetFlag<HardEvent::MTE3_V>(mte3ToV);
                mte3Pending = true;
                writeOffset += eqCount;
                curTileK -= static_cast<int32_t>(eqCount);
            }
        }

        if (tileId < tileNum_ - 1) {
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
        }
        ppFlag = !ppFlag;
    }

    // Fill remaining with -1
    if (mte3Pending) {
        WaitFlag<HardEvent::MTE3_V>(mte3ToV);
        mte3Pending = false;
    }
    uint32_t remainCount = kValue - writeOffset;
    if (remainCount > 0) {
        Duplicate<int32_t>(outIndexLocal, static_cast<int32_t>(-1), remainCount);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vToMte3);
        WaitFlag<HardEvent::V_MTE3>(vToMte3);
        DataCopyPad(indexGm[writeOffset], outIndexLocal, DataCopyExtParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * remainCount), 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(mte3ToV);
        mte3Pending = true;
    }

    if (mte3Pending) {
        WaitFlag<HardEvent::MTE3_V>(mte3ToV);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV);
}

}

#endif
