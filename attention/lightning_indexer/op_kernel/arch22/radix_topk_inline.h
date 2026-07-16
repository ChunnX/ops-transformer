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

// Multi-core divide-and-conquer RadixTopK
// All AIV cores process the same S1 token simultaneously, splitting sortLen across cores.
// Uses workspace GM for global histogram reduction via SyncAll.
// No early exit — always runs 8 rounds so AIC knows exact SyncAll count.
//
// SyncAll count per batch: 8 (rounds) + 1 (coreTopK) + 1 (end) = 10
// AIC must call SyncAll the same number of times.
template <typename T = void>
__aicore__ inline void InlineRadixTopK(
    TPipe *pipe,
    __gm__ uint8_t *scoreGmAddr,
    __gm__ uint8_t *indicesGmAddr,
    __gm__ uint8_t *wsGmAddr,         // workspace: [coreNum*4] globalHist + [coreNum] coreTopK
    uint32_t sortLen,
    uint32_t sortLenAlign,
    uint32_t kValue,
    uint32_t tileLen,
    uint32_t coreId,                  // this AIV's core ID
    uint32_t coreNum)                 // total AIV cores
{
    uint32_t totalTiles = (sortLen + tileLen - 1) / tileLen;
    if (totalTiles == 0) { totalTiles = 1; }
    uint32_t tileLenAlign = (tileLen + 7) / 8 * 8;
    uint32_t cmpMaskLen = tileLenAlign / 8;
    uint32_t alignLen = (tileLenAlign + 63) / 64 * 64;

    // Tile distribution across cores (same as original RadixTopK)
    uint32_t formerTileNum = (totalTiles + coreNum - 1) / coreNum;
    uint32_t tailTileNum = totalTiles / coreNum;
    uint32_t formerCoreNum = totalTiles % coreNum;
    uint32_t tailCoreNum = tailTileNum ? (coreNum - formerCoreNum) : 0;
    if (formerCoreNum == 0) { formerCoreNum = coreNum; formerTileNum = tailTileNum; }

    uint32_t myTileNum = (coreId < formerCoreNum) ? formerTileNum : tailTileNum;
    uint32_t myTileStart;
    if (coreId < formerCoreNum) {
        myTileStart = coreId * formerTileNum;
    } else {
        myTileStart = formerCoreNum * formerTileNum + (coreId - formerCoreNum) * tailTileNum;
    }

    TBuf<TPosition::VECCALC> scoreBuf;
    TBuf<TPosition::VECCALC> twiddleBuf;
    TBuf<TPosition::VECCALC> histBuf;
    TBuf<TPosition::VECCALC> cmpMaskBuf;
    TBuf<TPosition::VECCALC> indexBuf;
    TBuf<TPosition::VECCALC> outIdxBuf;
    TBuf<TPosition::VECCALC> tempBuf;

    // Note: pipe->Reset() is needed to free ProcessVec buffers and allocate RadixTopK buffers
    pipe->Reset();
    pipe->InitBuffer(scoreBuf, tileLenAlign * sizeof(float) + 64);
    pipe->InitBuffer(twiddleBuf, tileLenAlign * sizeof(int16_t) * 2 + 64);
    pipe->InitBuffer(histBuf, RDX_BINS * sizeof(int32_t) * 2 + 64);
    pipe->InitBuffer(cmpMaskBuf, cmpMaskLen + 64);
    pipe->InitBuffer(indexBuf, tileLenAlign * sizeof(int32_t) + 64);
    pipe->InitBuffer(outIdxBuf, tileLenAlign * sizeof(int32_t) + 64);
    pipe->InitBuffer(tempBuf, tileLenAlign * sizeof(uint32_t) + tileLenAlign * sizeof(int32_t) + 256);

    GlobalTensor<bfloat16_t> scoreGm;
    scoreGm.SetGlobalBuffer((__gm__ bfloat16_t *)scoreGmAddr);
    GlobalTensor<int32_t> indicesGm;
    indicesGm.SetGlobalBuffer((__gm__ int32_t *)indicesGmAddr);

    // Workspace layout: [coreNum, 4] globalHist | [coreNum] coreTopK
    GlobalTensor<int32_t> globalHistGm;
    globalHistGm.SetGlobalBuffer((__gm__ int32_t *)wsGmAddr);
    GlobalTensor<int32_t> coreTopKGm;
    coreTopKGm.SetGlobalBuffer((__gm__ int32_t *)wsGmAddr + coreNum * RDX_BINS);

    LocalTensor<bfloat16_t> scoreLocal = scoreBuf.Get<bfloat16_t>();
    LocalTensor<int16_t> scoreI16Raw = scoreBuf.Get<int16_t>();
    LocalTensor<int16_t> twiddleLocal = twiddleBuf.Get<int16_t>();
    LocalTensor<int32_t> localHist = histBuf.Get<int32_t>();
    LocalTensor<int32_t> tileHist = localHist[RDX_BINS];
    LocalTensor<uint8_t> cmpMask = cmpMaskBuf.Get<uint8_t>();
    LocalTensor<int32_t> indexLocal = indexBuf.Get<int32_t>();
    LocalTensor<int32_t> outIdxLocal = outIdxBuf.Get<int32_t>();

    LocalTensor<int16_t> tempI16 = tempBuf.Get<int16_t>();
    LocalTensor<int32_t> tempI32 = tempBuf.Get<int32_t>();
    LocalTensor<float> tempF32 = tempI32.ReinterpretCast<float>();

    int16_t involvedMask = 0;
    int32_t remainK = static_cast<int32_t>(kValue);

    event_t mte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t vToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    // === 8 rounds of 2-bit radix (no early exit) ===
    for (int32_t roundId = RDX_ROUNDS - 1; roundId >= 0; roundId--) {
        Duplicate<int32_t>(localHist, 0, RDX_BINS);
        PipeBarrier<PIPE_V>();
        RDX_V_S_SYNC();  // Ensure localHist is zeroed before S pipe reads

        // Each core processes its assigned tiles
        for (uint32_t t = 0; t < myTileNum; t++) {
            uint32_t tileId = myTileStart + t;
            uint32_t curTileLen = (tileId == totalTiles - 1) ?
                (sortLen - tileId * tileLen) : tileLen;
            if (curTileLen == 0) { continue; }

            DataCopyPad(scoreLocal[tileLen], scoreGm[tileId * tileLen], DataCopyExtParams{
                1, static_cast<uint32_t>(sizeof(bfloat16_t) * curTileLen), 0, 0, 0},
                DataCopyPadExtParams<bfloat16_t>{true, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(mte2ToV);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV);

            // TwiddleInB16 — match math repo: operate on [tileLen] offset, not in-place
            LocalTensor<int16_t> scoreI16 = scoreI16Raw[tileLen];
            LocalTensor<int16_t> signMaskTensor = tempI16;
            LocalTensor<int16_t> tempTensor = twiddleLocal;
            uint64_t curTileLenAlign = (curTileLen + 15) / 16 * 16;  // 32-byte align for int16
            LocalTensor<int16_t> temp1 = tempTensor[curTileLenAlign];

            ShiftRight(tempTensor, scoreI16, (int16_t)15, curTileLenAlign);
            Duplicate<int16_t>(signMaskTensor, RDX_XOR, curTileLenAlign);
            Or(tempTensor, tempTensor, signMaskTensor, curTileLenAlign);

            And(temp1, scoreI16, tempTensor, curTileLenAlign);
            Or(scoreI16, scoreI16, tempTensor, curTileLenAlign);
            Not(temp1, temp1, curTileLenAlign);
            And(scoreI16, scoreI16, temp1, curTileLenAlign);

            // AndMask not needed - use full twiddled value for histogram
            // involvedTemp includes accumulated boundary bits from previous rounds
            // so we need the full value, not just current 2 bits

            // Cast int16 -> float -> int32 (use twiddleBuf for intermediate)
            LocalTensor<float> castF32 = twiddleLocal.template ReinterpretCast<float>();
            Cast(castF32, scoreI16, RoundMode::CAST_NONE, curTileLenAlign);
            PipeBarrier<PIPE_V>();
            Cast(tempI32, castF32, RoundMode::CAST_RINT, curTileLenAlign);
            PipeBarrier<PIPE_V>();

            // Histogram for this tile
            int32_t cumSum = 0;
            LocalTensor<float> gatherDst = tempF32[tileLenAlign];  // use second half to avoid overwriting tempI32
            for (int32_t bin = RDX_BINS - 1; bin > 0; bin--) {
                int32_t involvedTemp = static_cast<int32_t>(
                    involvedMask | static_cast<int16_t>(bin << (roundId * RDX_BITS)));
                Compares<int32_t, uint8_t>(cmpMask, tempI32, involvedTemp, CMPMODE::EQ, alignLen);
                PipeBarrier<PIPE_V>();
                uint64_t rsvdCnt = 0;
                GatherMask(gatherDst, gatherDst, cmpMask.template ReinterpretCast<uint32_t>(),
                           true, tileLenAlign, {1, 1, 0, 0}, rsvdCnt);
                RDX_V_S_SYNC();
                cumSum += static_cast<int32_t>(rsvdCnt);
                tileHist.SetValue(bin, cumSum);
                RDX_S_V_SYNC();
            }
            tileHist.SetValue(0, static_cast<int32_t>(curTileLen));

            // Accumulate to localHist
            RDX_V_S_SYNC();
            for (uint32_t b = 0; b < RDX_BINS; b++) {
                localHist.SetValue(b, localHist.GetValue(b) + tileHist.GetValue(b));
            }
            RDX_S_V_SYNC();
        }

        // Write localHist to workspace GM
        RDX_V_S_SYNC();
        for (uint32_t b = 0; b < RDX_BINS; b++) {
            globalHistGm.SetValue(coreId * RDX_BINS + b, localHist.GetValue(b));
        }
        RDX_S_V_SYNC();

        // Single-core mode: skip SyncAll and globalHist reduction, use localHist directly
        if (coreNum > 1) {
            SyncAll();

            // Reduce globalHist: all cores read all cores' histograms
            Duplicate<int32_t>(localHist, 0, RDX_BINS);
            PipeBarrier<PIPE_V>();
            RDX_V_S_SYNC();
            for (uint32_t c = 0; c < coreNum; c++) {
                for (uint32_t b = 0; b < RDX_BINS; b++) {
                    localHist.SetValue(b, localHist.GetValue(b) + globalHistGm.GetValue(c * RDX_BINS + b));
                }
            }
            RDX_S_V_SYNC();
        }

        // FindBoundaryBin (all cores compute the same result)
        RDX_V_S_SYNC();
        int32_t boundaryBin = -1;
        int32_t boundaryBinPrev = -1;
        for (int32_t bin = RDX_BINS - 1; bin >= 0; bin--) {
            if (localHist.GetValue(bin) >= remainK) {
                boundaryBin = bin;
                boundaryBinPrev = bin + 1;
                break;
            }
        }
        // When boundaryBin == -1, involvedMask |= (int16_t)(-1) << shift sets all bits
        // This makes boundary = max value, so all elements are in TopK
        RDX_S_V_SYNC();

        involvedMask |= static_cast<int16_t>(static_cast<uint16_t>(boundaryBin) << (roundId * RDX_BITS));

        int32_t definitelyIn = 0;
        if (boundaryBinPrev < static_cast<int32_t>(RDX_BINS) && boundaryBinPrev >= 0) {
            RDX_V_S_SYNC();
            definitelyIn = localHist.GetValue(boundaryBinPrev);
            RDX_S_V_SYNC();
        }
        remainK = static_cast<int32_t>(kValue) - definitelyIn;

        // andMask no longer needed - using full twiddled value
    }

    // Extract boundary number
    int16_t boundaryNumber16 = involvedMask ^ ((~(involvedMask >> 15)) | 0x8000);
    uint32_t boundaryBits = static_cast<uint32_t>(static_cast<uint16_t>(boundaryNumber16)) << 16;
    float boundaryFloat = *reinterpret_cast<float *>(&boundaryBits);

    // === TileTopK extraction ===
    ArithProgression<int32_t>(indexLocal, 0, 1, tileLenAlign);
    PipeBarrier<PIPE_V>();

    uint32_t coreTopKOffset = 0;
    // Phase 1: count GT + EQ per tile
    for (uint32_t t = 0; t < myTileNum; t++) {
        uint32_t tileId = myTileStart + t;
        uint32_t curTileLen = (tileId == totalTiles - 1) ?
            (sortLen - tileId * tileLen) : tileLen;
        if (curTileLen == 0) { continue; }

        DataCopyPad(scoreLocal[tileLen], scoreGm[tileId * tileLen], DataCopyExtParams{
            1, static_cast<uint32_t>(sizeof(bfloat16_t) * curTileLen), 0, 0, 0},
            DataCopyPadExtParams<bfloat16_t>{true, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        LocalTensor<bfloat16_t> scoreB16 = scoreI16Raw[tileLen].template ReinterpretCast<bfloat16_t>();
        LocalTensor<float> scoreFp32 = tempF32;
        Cast(scoreFp32, scoreB16, RoundMode::CAST_NONE, alignLen);
        PipeBarrier<PIPE_V>();

        // GT count
        Compares<float, uint8_t>(cmpMask, scoreFp32, boundaryFloat, CMPMODE::GT, alignLen);
        uint64_t rsvdCntI = 0;
        GatherMask(outIdxLocal, indexLocal, cmpMask.template ReinterpretCast<uint32_t>(),
                   true, tileLenAlign, {1, 1, 0, 0}, rsvdCntI);
        RDX_V_S_SYNC();
        int32_t gtCount = static_cast<int32_t>(rsvdCntI);
        RDX_S_V_SYNC();

        // EQ count
        int32_t eqCount = 0;
        int32_t remaining = static_cast<int32_t>(kValue) - static_cast<int32_t>(coreTopKOffset) - gtCount;
        if (remaining > 0) {
            Compares<float, uint8_t>(cmpMask, scoreFp32, boundaryFloat, CMPMODE::EQ, alignLen);
            LocalTensor<int32_t> eqIdx = tempI32;
            uint64_t rsvdCntEq = 0;
            GatherMask(eqIdx, indexLocal, cmpMask.template ReinterpretCast<uint32_t>(),
                       true, tileLenAlign, {1, 1, 0, 0}, rsvdCntEq);
            RDX_V_S_SYNC();
            eqCount = static_cast<int32_t>(rsvdCntEq);
            int32_t eqCopied = (remaining < eqCount) ? remaining : eqCount;
            RDX_S_V_SYNC();
            coreTopKOffset += gtCount + eqCopied;
        } else {
            coreTopKOffset += gtCount;
        }
    }

    // Write this core's total TopK count to workspace
    RDX_V_S_SYNC();
    coreTopKGm.SetValue(coreId, static_cast<int32_t>(coreTopKOffset));
    RDX_S_V_SYNC();

    // SyncAll for multi-core mode
    uint32_t outOffset = 0;
    if (coreNum > 1) {
        SyncAll();
        RDX_V_S_SYNC();
        for (uint32_t c = 0; c < coreId; c++) {
            outOffset += static_cast<uint32_t>(coreTopKGm.GetValue(c));
        }
        RDX_S_V_SYNC();
    }

    // Phase 2: write indices to GM
    uint32_t writeOffset = outOffset;
    for (uint32_t t = 0; t < myTileNum; t++) {
        uint32_t tileId = myTileStart + t;
        uint32_t curTileLen = (tileId == totalTiles - 1) ?
            (sortLen - tileId * tileLen) : tileLen;
        if (curTileLen == 0) { continue; }

        DataCopyPad(scoreLocal[tileLen], scoreGm[tileId * tileLen], DataCopyExtParams{
            1, static_cast<uint32_t>(sizeof(bfloat16_t) * curTileLen), 0, 0, 0},
            DataCopyPadExtParams<bfloat16_t>{true, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(mte2ToV);
        WaitFlag<HardEvent::MTE2_V>(mte2ToV);

        ArithProgression<int32_t>(indexLocal, 0, 1, tileLenAlign);
        PipeBarrier<PIPE_V>();

        LocalTensor<bfloat16_t> scoreB16 = scoreI16Raw[tileLen].template ReinterpretCast<bfloat16_t>();
        LocalTensor<float> scoreFp32 = tempF32;
        Cast(scoreFp32, scoreB16, RoundMode::CAST_NONE, alignLen);
        PipeBarrier<PIPE_V>();

        int32_t tileOffset = static_cast<int32_t>(tileId * tileLen);

        // GT extraction
        Compares<float, uint8_t>(cmpMask, scoreFp32, boundaryFloat, CMPMODE::GT, alignLen);
        uint64_t rsvdCntI = 0;
        GatherMask(outIdxLocal, indexLocal, cmpMask.template ReinterpretCast<uint32_t>(),
                   true, tileLenAlign, {1, 1, 0, 0}, rsvdCntI);
        RDX_V_S_SYNC();
        int32_t gtCount = static_cast<int32_t>(rsvdCntI);
        RDX_S_V_SYNC();

        if (gtCount > 0) {
            int32_t gtAlign = (gtCount + 7) / 8 * 8;
            if (gtAlign > static_cast<int32_t>(tileLenAlign)) { gtAlign = tileLenAlign; }
            Adds<int32_t>(outIdxLocal, outIdxLocal, tileOffset, gtAlign);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3);
            WaitFlag<HardEvent::V_MTE3>(vToMte3);
            DataCopyPad(indicesGm[writeOffset], outIdxLocal, DataCopyExtParams{
                1, static_cast<uint32_t>(sizeof(int32_t) * gtCount), 0, 0, 0});
            writeOffset += gtCount;
        }

        // EQ extraction
        int32_t remaining = static_cast<int32_t>(kValue) - static_cast<int32_t>(writeOffset);
        if (remaining > 0) {
            Compares<float, uint8_t>(cmpMask, scoreFp32, boundaryFloat, CMPMODE::EQ, alignLen);
            LocalTensor<int32_t> eqIdx = tempI32;
            uint64_t rsvdCntEq = 0;
            GatherMask(eqIdx, indexLocal, cmpMask.template ReinterpretCast<uint32_t>(),
                       true, tileLenAlign, {1, 1, 0, 0}, rsvdCntEq);
            RDX_V_S_SYNC();
            int32_t eqCount = static_cast<int32_t>(rsvdCntEq);
            int32_t eqCopied = (remaining < eqCount) ? remaining : eqCount;
            RDX_S_V_SYNC();
            if (eqCopied > 0) {
                int32_t eqAlign = (eqCopied + 7) / 8 * 8;
                if (eqAlign > static_cast<int32_t>(tileLenAlign)) { eqAlign = tileLenAlign; }
                Adds<int32_t>(eqIdx, eqIdx, tileOffset, eqAlign);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(vToMte3);
                WaitFlag<HardEvent::V_MTE3>(vToMte3);
                DataCopyPad(indicesGm[writeOffset], eqIdx, DataCopyExtParams{
                    1, static_cast<uint32_t>(sizeof(int32_t) * eqCopied), 0, 0, 0});
                writeOffset += eqCopied;
            }
        }
    }

    // Fill remaining with -1
    if (coreId == 0) {
        for (uint32_t i = writeOffset; i < kValue; i++) {
            indicesGm.SetValue(i, -1);
        }
    }

    // SyncAll for multi-core
    if (coreNum > 1) { SyncAll(); }

    RDX_S_V_SYNC();

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3);
}

}

#endif
