#ifndef BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H
#define BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"

#ifndef BSA_TOPK_DEBUG
#define BSA_TOPK_DEBUG 0
#endif

#ifndef BSA_TOPK_SYNC_PROBE
#define BSA_TOPK_SYNC_PROBE 0
#endif

#ifndef BSA_TOPK_ENABLE_MTE3_TO_S_SYNC
#define BSA_TOPK_ENABLE_MTE3_TO_S_SYNC 1
#endif

#ifndef BSA_TOPK_INIT_ATTNSCORE_PROBE
#define BSA_TOPK_INIT_ATTNSCORE_PROBE 1
#endif

#ifndef BSA_TOPK_SMALL_SORT_DEBUG
#define BSA_TOPK_SMALL_SORT_DEBUG 0
#endif

#define BSA_TOPK_PRINTF(...) do {} while (0)

#define BSA_TOPK_SYNC_PRINTF(...) do {} while (0)

#define BSA_TOPK_DUMP_TENSOR(...) do {} while (0)

#define BSA_TOPK_MTE3_TO_S_SYNC() do { \
    if (BSA_TOPK_ENABLE_MTE3_TO_S_SYNC) { \
        MTE3ToSSync(); \
    } \
} while (0)

template <typename BSAT>
class BSARadixTopKService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;

    __aicore__ inline BSARadixTopKService() {};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGM(GlobalTensor<IN_T> &attnScoreGm,
                                   GlobalTensor<int32_t> &topkWorkspaceGm,
                                   GlobalTensor<uint8_t> &maskTmpGm,
                                   GlobalTensor<uint8_t> &maskOutGmU8);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx);
private:
    __aicore__ inline void InitAttnScoreGmForRobustness(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void NegateDataForLargest(TBuf<TPosition::VECIN>& xBuf, uint64_t curTileLen);
    __aicore__ inline void TwiddleInB16(TBuf<TPosition::VECIN>& xBuf, uint64_t curTileLen,
                                        int32_t roundId, int32_t tileId);
    __aicore__ inline void DoAndMask(TBuf<TPosition::VECIN>& xBuf, uint64_t curTileLen);
    __aicore__ inline void CalcCumsumHistogram(TBuf<TPosition::VECIN>& xBuf, int32_t roundId, 
                                                int32_t tileId, uint64_t curTileLen);
    __aicore__ inline void CopyIn(TBuf<TPosition::VECIN>& xBuf, uint64_t dataNum, uint64_t xOffset);
    __aicore__ inline void ClearTileTopKInWs();
    __aicore__ inline void ClearTileHistInWs(uint32_t dstOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileHistWs2Ub(LocalTensor<int32_t>& dstTensor,
                                              uint32_t srcOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileTopKWs2Ub(LocalTensor<int32_t>& dstTensor,
                                              uint32_t srcOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileTopKUb2Ws(LocalTensor<int32_t>& srcTensor,
                                              uint32_t dstOffset, uint32_t dataNum);
    __aicore__ inline void CopyTileHistInWs(uint32_t dstOffset, uint32_t srcOffset);
    __aicore__ inline void SubTileHistWs2Ub(LocalTensor<int32_t>& dstTensor,
                                             LocalTensor<int32_t>& srcTensor,
                                             uint32_t srcOffset0, uint32_t srcOffset1,
                                             uint32_t dataNum);
    __aicore__ inline void SubTileHistInWs(uint32_t dstOffset, uint32_t srcOffset0,
                                            uint32_t srcOffset1, uint32_t dataNum);
    __aicore__ inline void SubTileHistInWsAll(uint32_t dstOffset, uint32_t srcOffset0,
                                               uint32_t srcOffset1);
    __aicore__ inline void AddTileHist2TileTopKInWs(uint64_t tileHistOffset);
    __aicore__ inline void ClearHistInWs(int32_t roundId);
    template<bool isInit>
    __aicore__ inline void CopyOutHistToWs(uint64_t maskLen, uint64_t gmOffset);
    __aicore__ inline bool Update(int32_t roundId);
    __aicore__ inline void CopyInGlobalHist(LocalTensor<int32_t>& globalHist);
    __aicore__ inline void ReduceGlobalHist(LocalTensor<int32_t>& globalHist);
    __aicore__ inline void FindBoundaryBin(const LocalTensor<int32_t>& globalHist, int32_t roundId);
    __aicore__ inline void AddTileHistToTileTopK(LocalTensor<int32_t>& globalHist);
    __aicore__ inline void HandleLastRoundBoundary(LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);
    __aicore__ inline void WriteCoreTopKFromWs(LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32);
    __aicore__ inline void CopyOutBoundaryBinCumSum(LocalTensor<int32_t>& resTensor);
    __aicore__ inline void ComputeCumSumPrev(LocalTensor<int32_t>& resTensor, int32_t& cumSumValuePrev);
    __aicore__ inline void CopyOutCoreTopK(LocalTensor<int32_t>& resTensor, int32_t totalTileTopKInCore);
    __aicore__ inline void SubTopKAndWriteMaskGT(LocalTensor<half>& xLocal, half boundaryHalf,
                                                  uint64_t curTileLen, uint64_t outputGmOffset,
                                                  int32_t& curTileK);
    __aicore__ inline void SubTopKAndWriteMaskEQ(LocalTensor<half>& xLocal, half boundaryHalf,
                                                  uint64_t curTileLen, uint64_t outputGmOffset,
                                                  int32_t& curTileK);
    __aicore__ inline void TileTopK(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void VToSSync();
    __aicore__ inline void SToVSync();
    __aicore__ inline void MTE2ToMTE3Sync();
    __aicore__ inline void MTE3ToMTE2Sync();
    __aicore__ inline void MTE3ToVSync();
    __aicore__ inline void VToMTE3Sync();
    __aicore__ inline void MTE2ToVSync();
    __aicore__ inline void MTE2ToSSync();
    __aicore__ inline void MTE3ToSSync();
    __aicore__ inline void SToMTE3Sync();

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> attnScoreGmLocal;
    GlobalTensor<int32_t> topkWorkspaceGmLocal;
    GlobalTensor<uint8_t> maskTmpGmLocal;
    GlobalTensor<uint8_t> maskOutGmU8Local;

    GlobalTensor<int32_t> globalHistGm_;
    GlobalTensor<int32_t> boundaryBinCumSumGm_;
    GlobalTensor<int32_t> coreTopKGm_;
    GlobalTensor<int32_t> tileTopKGm_;
    GlobalTensor<int32_t> tileHistGm_;

    TBuf<TPosition::VECIN> xBufPing_;
    TBuf<TPosition::VECIN> xBufPong_;
    TBuf<TPosition::VECOUT> outValueBuf_;
    TBuf<TPosition::VECOUT> outIndexBuf_;
    TBuf<TPosition::VECOUT> maskLocalBuf_;
    TBuf<TPosition::VECOUT> globalHistBuf_;
    TBuf<TPosition::VECCALC> tileHistAndTopKBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;

    int16_t involvedMask16_;
    int16_t andMask16_;
    int32_t boundaryBin;
    int32_t boundaryBinPrev;
    uint64_t globalHistBoundaryNum_;
    uint64_t totalDefinitelyInTopK_;
    uint64_t remainK_;
    uint64_t blockOffset_;

    uint64_t tileLen_;
    uint64_t tileStartId_;
    uint64_t tileNum_;
    uint64_t tileNumAlign_;
    uint64_t sortLen_;
    uint64_t kValue_;
    uint32_t round_;
    uint32_t numValue_;
    uint32_t bitsPerRound_;

    uint64_t totalTileNum_;
    uint64_t formerCoreNum_;
    uint64_t tailCoreNum_;
    uint64_t formerTileNum_;
    uint64_t tailTileNum_;
    uint64_t formerTileLen_;
    uint64_t tailTileLen_;
    uint32_t tileNumRepeatTimes_;
    uint32_t tileNumRemain_;
    uint32_t tileNumBy2RepeatTimes_;
    uint32_t tileNumBy2Remain_;
    uint32_t tileNumBy3RepeatTimes_;
    uint32_t tileNumBy3Remain_;

    event_t eventIDMTE2ToVForX_;
    bool pingPongFlag;
};

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitParams(const BSAConstInfo &constInfo,
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;

    this->sortLen_ = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks;
    this->kValue_ = constInfo.topKValue;
    this->bitsPerRound_ = RADIX_BITS_PER_ROUND;
    this->numValue_ = RADIX_NUM_BINS;
    this->round_ = RADIX_ROUNDS_FP16;

    this->tileLen_ = BSAMin(static_cast<uint64_t>(7168), BSAMax(static_cast<uint64_t>(128), sortLen_));
    this->totalTileNum_ = (sortLen_ + tileLen_ - 1) / tileLen_;
    this->tileNum_ = totalTileNum_;
    this->tileNumAlign_ = BSACeilDiv(tileNum_, static_cast<uint64_t>(8)) * 8;

    uint32_t coreNum = constInfo.aivNum;
    uint64_t baseTilesPerCore = totalTileNum_ / coreNum;
    uint64_t extraCores = totalTileNum_ % coreNum;
    this->formerCoreNum_ = extraCores;
    this->tailCoreNum_ = coreNum - extraCores;
    this->formerTileNum_ = baseTilesPerCore + (extraCores > 0 ? 1 : 0);
    this->tailTileNum_ = baseTilesPerCore;

    uint64_t myTileNum = (GetBlockIdx() < formerCoreNum_) ? formerTileNum_ : tailTileNum_;
    this->tileStartId_ = (GetBlockIdx() < formerCoreNum_)
        ? GetBlockIdx() * formerTileNum_
        : formerCoreNum_ * formerTileNum_ + (GetBlockIdx() - formerCoreNum_) * tailTileNum_;
    this->tileNum_ = myTileNum;
    this->tileNumAlign_ = BSACeilDiv(myTileNum, static_cast<uint64_t>(8)) * 8;
    this->formerTileLen_ = tileLen_;
    uint64_t lastTileLen = sortLen_ - (totalTileNum_ - 1) * tileLen_;
    if (lastTileLen == 0) {
        lastTileLen = tileLen_;
    }
    this->tailTileLen_ = (myTileNum > 0 && tileStartId_ + myTileNum == totalTileNum_) ? lastTileLen : tileLen_;

    this->tileNumRepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB);
    this->tileNumRemain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB);
    this->tileNumBy2RepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB_BY2);
    this->tileNumBy2Remain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB_BY2);
    this->tileNumBy3RepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB_BY3);
    this->tileNumBy3Remain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB_BY3);

    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF(
            "[SDBG100] sortLen=%lu k=%lu tileLen=%lu totalTile=%lu tileStart=%lu tileNum=%lu "
            "tailTileLen=%lu repeat=%u remain=%u aiv=%u\n",
            sortLen_, kValue_, tileLen_, totalTileNum_, tileStartId_, tileNum_,
            tailTileLen_, tileNumRepeatTimes_, tileNumRemain_, constInfo.aivNum);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(xBufPing_, tileLen_ * sizeof(float));
    pipe->InitBuffer(xBufPong_, tileLen_ * sizeof(float));
    pipe->InitBuffer(outValueBuf_, tileLen_ * sizeof(float));
    pipe->InitBuffer(outIndexBuf_, tileLen_ * sizeof(float));
    pipe->InitBuffer(tileHistAndTopKBuf_, MAX_TILE_NUM_IN_UB * sizeof(int32_t));

    uint32_t cmpMaskSize = tileLen_ * sizeof(uint8_t) / BYTE_SIZE;
    pipe->InitBuffer(tempBuf_, cmpMaskSize);

    pipe->InitBuffer(globalHistBuf_, numValue_ * sizeof(int32_t));
    pipe->InitBuffer(maskLocalBuf_, tileLen_ * sizeof(int32_t) + tileLen_ * sizeof(int16_t) + tileLen_ * sizeof(uint8_t));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitGM(
    GlobalTensor<IN_T> &attnScoreGm,
    GlobalTensor<int32_t> &topkWorkspaceGm,
    GlobalTensor<uint8_t> &maskTmpGm,
    GlobalTensor<uint8_t> &maskOutGmU8)
{
    this->attnScoreGmLocal = attnScoreGm;
    this->topkWorkspaceGmLocal = topkWorkspaceGm;
    this->maskTmpGmLocal = maskTmpGm;
    this->maskOutGmU8Local = maskOutGmU8;

    uint32_t coreNum = constInfo.aivNum;
    uint32_t headerSize = (numValue_ * coreNum + coreNum + coreNum);
    this->globalHistGm_.SetGlobalBuffer((__gm__ int32_t *)topkWorkspaceGm.GetPhyAddr());
    this->boundaryBinCumSumGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + numValue_ * coreNum * sizeof(int32_t)));
    this->coreTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + (numValue_ * coreNum + coreNum) * sizeof(int32_t)));

    uint64_t tileTopKOffset = tileStartId_;
    uint64_t tileHistOffset = numValue_ * tileTopKOffset;

    this->tileTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + headerSize * sizeof(int32_t) + tileTopKOffset * sizeof(int32_t)));
    this->tileHistGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + headerSize * sizeof(int32_t) + totalTileNum_ * sizeof(int32_t) + tileHistOffset * sizeof(int32_t)));

}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AllocEventID()
{
    eventIDMTE2ToVForX_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::FreeEventID()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitAttnScoreGmForRobustness(uint32_t batchIdx, uint32_t headIdx)
{
    if (!BSA_TOPK_INIT_ATTNSCORE_PROBE || tileNum_ == 0) {
        return;
    }

    (void)batchIdx;
    (void)headIdx;
    // attnScoreGmLocal is a per batch/head scratch buffer reused by the outer loop.
    uint64_t baseOffset = 0;
    for (uint64_t tileId = 0; tileId < tileNum_; ++tileId) {
        uint64_t globalTileId = tileStartId_ + tileId;
        uint64_t curTileLen = (tileId + 1 == tileNum_) ? tailTileLen_ : tileLen_;
        uint64_t offset = baseOffset + globalTileId * tileLen_;
        AscendC::InitOutput(attnScoreGmLocal[offset], curTileLen, static_cast<IN_T>(0.0f));
        PipeBarrier<PIPE_ALL>();

        uint64_t oneLen = curTileLen >= 64 ? 8 : (curTileLen >= 16 ? 4 : 1);
        uint64_t oneStart = curTileLen / 2;
        if (oneStart + oneLen > curTileLen) {
            oneStart = curTileLen > oneLen ? curTileLen - oneLen : 0;
        }
        AscendC::InitOutput(attnScoreGmLocal[offset + oneStart], oneLen, static_cast<IN_T>(1.0f));

        uint64_t gap = curTileLen >= 64 ? 16 : 4;
        uint64_t setStart = oneStart + oneLen + gap;
        if (setStart < curTileLen) {
            uint64_t stride = curTileLen >= 64 ? 7 : 3;
            uint64_t jitter = globalTileId % stride;
            for (uint64_t i = 0; i < 4; ++i) {
                uint64_t pos = setStart + jitter + i * stride;
                if (pos < curTileLen) {
                    IN_T targetValue = (i % 2 == 0) ? static_cast<IN_T>(2.0f) : static_cast<IN_T>(-2.0f);
                    attnScoreGmLocal.SetValue(offset + pos, targetValue);
                }
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx)
{
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[TopK] v6.8: main flow assembled\n");
    }
    // 1. Calculate blockOffset
    this->blockOffset_ = tileStartId_ * tileLen_;
    if (BSA_TOPK_INIT_ATTNSCORE_PROBE) {
        InitAttnScoreGmForRobustness(batchIdx, headIdx);
        SyncAll();
    }

    // 3. ClearTileTopKInWs() — batch zero tileTopK in WS
    ClearTileTopKInWs();

    // 4. Reset state variables
    involvedMask16_ = 0;
    andMask16_ = static_cast<int16_t>(((1 << bitsPerRound_) - 1) << (16 - bitsPerRound_));
    totalDefinitelyInTopK_ = 0;
    boundaryBin = 0;
    boundaryBinPrev = 1;
    globalHistBoundaryNum_ = (tileNum_ > 0) ? (tileNum_ - 1) * tileLen_ + tailTileLen_ : 0;
    remainK_ = kValue_;

    // 4.5. Clear globalHist for all AIVs (including those with tileNum_=0)
    // This ensures ReduceGlobalHist doesn't sum garbage data from idle AIVs
    {
        LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();
        SToVSync();
        Duplicate<int32_t>(globalHist, 0, numValue_);
        VToMTE3Sync();
        DataCopyExtParams params{1, static_cast<uint32_t>(numValue_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(globalHistGm_[GetBlockIdx() * numValue_], globalHist, params);
    }

    // 5. RADIX LOOP
    bool earlyExit = false;
    for (int32_t roundId = round_ - 1; roundId >= 0; roundId--) {
        // a. Prefetch first tile (double buffering)
        if (tileNum_ > 0) {
            uint64_t firstTileLen = tileNum_ > 1 ? tileLen_ : tailTileLen_;
            CopyIn(xBufPing_, firstTileLen, blockOffset_);
        }
        pingPongFlag = true;

        // b. CopyOutHistToWs<true>
        CopyOutHistToWs<true>(numValue_, GetBlockIdx() * numValue_);

        // c. ClearHistInWs — batch init/reset tileHist in WS
        ClearHistInWs(roundId);

        // d. Signal first tile ready
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

        // e. Batched tile loop (each batch MAX_TILE_NUM_IN_UB_BY3 = 2048)
        int32_t repeatTimes = static_cast<int32_t>(tileNumBy3RepeatTimes_ + (tileNumBy3Remain_ > 0 ? 1 : 0));
        for (int32_t repeatId = 0; repeatId < repeatTimes; repeatId++) {
            int32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB_BY3;
            int32_t endTileId = BSAMin(startTileId + MAX_TILE_NUM_IN_UB_BY3,
                                       static_cast<int32_t>(tileNum_));

            for (int32_t tileId = startTileId; tileId < endTileId; tileId++) {
                uint64_t curTileLen = (tileId == static_cast<int32_t>(tileNum_) - 1)
                                    ? tailTileLen_ : tileLen_;
                TBuf<TPosition::VECIN>& curBuf = pingPongFlag ? xBufPing_ : xBufPong_;
                TBuf<TPosition::VECIN>& nextBuf = pingPongFlag ? xBufPong_ : xBufPing_;

                // Prefetch next tile (double buffering)
                if (tileId < static_cast<int32_t>(tileNum_) - 1) {
                    uint64_t nextOffset = blockOffset_ + (tileId + 1) * tileLen_;
                    uint64_t nextTileLen = (tileId == static_cast<int32_t>(tileNum_) - 2)
                                         ? tailTileLen_ : tileLen_;
                    CopyIn(nextBuf, nextTileLen, nextOffset);
                }

                // Wait current tile ready
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

                // Core processing
                NegateDataForLargest(curBuf, curTileLen);
                if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0) {
                    LocalTensor<int16_t> rawInt16 = curBuf.Get<int16_t>();
                    BSA_TOPK_PRINTF("[DUMP440] after CopyIn/Negate, xBuf int16[tileLen_] first 64:\n");
                    BSA_TOPK_DUMP_TENSOR(rawInt16[tileLen_], 440, 64);
                }
                TwiddleInB16(curBuf, curTileLen, roundId, tileId);
                PipeBarrier<PIPE_V>();
                DoAndMask(curBuf, curTileLen);
                PipeBarrier<PIPE_V>();
                VToSSync();
                if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0) {
                    LocalTensor<int16_t> maskedInt16 = curBuf.Get<int16_t>();
                    BSA_TOPK_PRINTF("[DUMP442] after DoAndMask, xBuf int16[tileLen_] first 64:\n");
                    BSA_TOPK_DUMP_TENSOR(maskedInt16[tileLen_], 442, 64);
                    if (curTileLen > 1023) {
                        BSA_TOPK_PRINTF("[DUMP443] maskedInt16 key points: idx0=%d idx31=%d idx32=%d idx639=%d idx640=%d idx895=%d idx896=%d idx1023=%d\n",
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(0)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(31)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(32)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(639)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(640)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(895)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(896)),
                            static_cast<int32_t>(maskedInt16[tileLen_].GetValue(1023)));
                    }
                }
                CalcCumsumHistogram(curBuf, roundId, tileId - startTileId, curTileLen);

                // === DUMP 400: tileHist after CalcCumsumHistogram (first round, first tile) ===
                if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0) {
                    LocalTensor<int32_t> tileHistDump = tileHistAndTopKBuf_.Get<int32_t>();
                    // bin 1, 2, 3 each have 1 entry for tile 0
                    // layout: [bin1_tile0, bin1_tile1, ..., bin2_tile0, ..., bin3_tile0, ...]
                    // For 1 tile: bin1 at offset 0, bin2 at offset 2048, bin3 at offset 4096
                    BSA_TOPK_PRINTF("[DUMP400] tileHist bin3=%d bin2=%d bin1=%d\n",
                        tileHistDump.GetValue(2 * MAX_TILE_NUM_IN_UB_BY3),
                        tileHistDump.GetValue(1 * MAX_TILE_NUM_IN_UB_BY3),
                        tileHistDump.GetValue(0));
                }

                // Signal next tile ready
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

                pingPongFlag = !pingPongFlag;  // Switch double buffering
            }

            // Write this batch's tileHist to WS
            SToMTE3Sync();
            LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
            if (endTileId > startTileId) {
                for (int32_t binMask = numValue_ - 1; binMask > 0; binMask--) {
                    uint32_t tileHistOffset = binMask * tileNum_ + repeatId * MAX_TILE_NUM_IN_UB_BY3;
                    DataCopyPad(tileHistGm_[tileHistOffset],
                               tileHist[(binMask - 1) * MAX_TILE_NUM_IN_UB_BY3],
                               DataCopyParams(1, sizeof(int32_t) * (endTileId - startTileId), 0, 0));
                }
                if (GetBlockIdx() == 0) {
                    BSA_TOPK_SYNC_PRINTF("[SYNC_PROBE] after tileHist ws write round=%d repeat=%d\n",
                        roundId, repeatId);
                }
            }
        }

        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

        // f. CopyOutHistToWs<false>
        CopyOutHistToWs<false>(numValue_, GetBlockIdx() * numValue_);

        // g. SyncAll
        SyncAll();

        // h. Update
        earlyExit = Update(roundId);
        if (GetBlockIdx() == 0) {
            BSA_TOPK_SYNC_PRINTF("[SYNC_PROBE] after Update round=%d remainK=%lu total=%lu\n",
                roundId, remainK_, totalDefinitelyInTopK_);
        }

        // === DUMP 500: After Update - boundaryBin, involvedMask16_, remainK_ ===
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
            BSA_TOPK_PRINTF("[DUMP500] round=%d boundaryBin=%d boundaryBinPrev=%d involvedMask16_=%d remainK_=%lu totalDefinitelyInTopK_=%lu\n",
                roundId, boundaryBin, boundaryBinPrev, static_cast<int32_t>(involvedMask16_), remainK_, totalDefinitelyInTopK_);
        }

        // i. Shift mask
        andMask16_ >>= RADIX_BITS_PER_ROUND;

        // j. Early exit
        if (earlyExit) break;
    }

    // 6. SyncAll before final selection
    SyncAll();

    // 7. TileTopK (batched processing)
    TileTopK(batchIdx, headIdx);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::NegateDataForLargest(
    TBuf<TPosition::VECIN>& xBuf, uint64_t curTileLen)
{
    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        LocalTensor<IN_T> xLocal = xBuf.Get<IN_T>();
        int32_t ubOffset = (sizeof(IN_T) == 2) ? tileLen_ : 0;
        uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));
        if constexpr (IsSameType<IN_T, bfloat16_t>::value) {
            Muls(xLocal[ubOffset].template ReinterpretCast<half>(),
                 xLocal[ubOffset].template ReinterpretCast<half>(), (half)-1, alignLen);
        } else {
            Muls(xLocal[ubOffset], xLocal[ubOffset], (IN_T)-1, alignLen);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TwiddleInB16(
    TBuf<TPosition::VECIN>& xBuf, uint64_t curTileLen, int32_t roundId, int32_t tileId)
{
    (void)roundId;
    (void)tileId;
    LocalTensor<IN_T> xLocalB16 = xBuf.Get<IN_T>();
    LocalTensor<int16_t> xLocalInt16 = xLocalB16[tileLen_].template ReinterpretCast<int16_t>();

    LocalTensor<int16_t> signMaskTensor = outValueBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensor = outIndexBuf_.Get<int16_t>();

    uint64_t calcLen = curTileLen * (sizeof(IN_T) / sizeof(int16_t));
    uint64_t curTileLenAlign = BSAAlignTo(curTileLen, static_cast<uint64_t>(UB_BLOCK_SIZE / sizeof(int16_t)));
    LocalTensor<int16_t> temp1 = tempTensor[curTileLenAlign];

    ShiftRight(tempTensor, xLocalInt16, (int16_t)15, curTileLen);
    Duplicate<int16_t>(signMaskTensor, XOR_OP_VALUE_16, curTileLen);
    PipeBarrier<PIPE_V>();
    Or(tempTensor, tempTensor, signMaskTensor, calcLen);
    PipeBarrier<PIPE_V>();

    And(temp1, xLocalInt16, tempTensor, calcLen);
    Or(xLocalInt16, xLocalInt16, tempTensor, calcLen);
    PipeBarrier<PIPE_V>();
    Not(temp1, temp1, calcLen);
    PipeBarrier<PIPE_V>();
    And(xLocalInt16, xLocalInt16, temp1, calcLen);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::DoAndMask(TBuf<TPosition::VECIN>& xBuf, uint64_t curTileLen)
{
    LocalTensor<int16_t> xLocalInt16 = xBuf.Get<int16_t>();
    uint64_t calcLen = curTileLen * (sizeof(IN_T) / sizeof(int16_t));
    LocalTensor<int32_t> maskTensor = outValueBuf_.Get<int32_t>();

    Duplicate<int16_t>(maskTensor.ReinterpretCast<int16_t>(), andMask16_, curTileLen);
    PipeBarrier<PIPE_V>();
    And(xLocalInt16[tileLen_], xLocalInt16[tileLen_],
        maskTensor.ReinterpretCast<int16_t>(), calcLen);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CalcCumsumHistogram(
    TBuf<TPosition::VECIN>& xBuf, int32_t roundId, int32_t tileId, uint64_t curTileLen)
{
    LocalTensor<int16_t> xLocalInt16 = xBuf.Get<int16_t>();
    LocalTensor<int32_t> xLocalInt32 = xLocalInt16.ReinterpretCast<int32_t>();
    LocalTensor<float> xLocalFp32 = xLocalInt16.ReinterpretCast<float>();

    Cast(xLocalFp32, xLocalInt16[tileLen_], RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
    Cast(xLocalInt32, xLocalFp32, RoundMode::CAST_RINT, curTileLen);
    PipeBarrier<PIPE_V>();

    // === DUMP 451: Cast chain results (first round, first tile) ===
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0) {
        BSA_TOPK_PRINTF("[DUMP451] After Cast chain (xLocalInt32):\n");
        BSA_TOPK_DUMP_TENSOR(xLocalInt32, 451, 32);
    }

    LocalTensor<half> tempHalf = outValueBuf_.Get<half>();
    LocalTensor<uint8_t> cmpMaskTensor = outIndexBuf_.Get<uint8_t>();
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();

    const int32_t tileHistBinStride = MAX_TILE_NUM_IN_UB_BY3;
    const int32_t tileHistBinOffset = MAX_TILE_NUM_IN_UB_BY3;

    int32_t cumSumOne = 0;
    for (int16_t binMask = numValue_ - 1; binMask > 0; binMask--) {
        int32_t involvedMaskTemp = static_cast<int32_t>(
            involvedMask16_ | static_cast<int16_t>(binMask << (roundId * bitsPerRound_)));

        // === DUMP 460: involvedMaskTemp (first round, first tile, first 3 bins) ===
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0 && binMask >= 1) {
            BSA_TOPK_PRINTF("[DUMP460] binMask=%d involvedMaskTemp=%d (0x%x)\n",
                binMask, involvedMaskTemp, involvedMaskTemp);
        }

        // === DUMP 465: xLocalInt32 before Compares (first round, first tile, bin 2 only) ===
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0 && binMask == 2) {
            BSA_TOPK_PRINTF("[DUMP465] xLocalInt32 before Compares (first 32):\n");
            BSA_TOPK_DUMP_TENSOR(xLocalInt32, 465, 32);
        }

        uint64_t curTileLenAlign = BSAAlignTo(curTileLen, static_cast<uint64_t>(REPEAT_SIZE / sizeof(int32_t)));
        Compares<int32_t, uint8_t>(cmpMaskTensor, xLocalInt32, involvedMaskTemp,
                                   AscendC::CMPMODE::EQ, curTileLenAlign);
        PipeBarrier<PIPE_V>();

        // === DUMP 466: cmpMaskTensor after Compares for bin 2 ===
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0 && binMask == 2) {
            BSA_TOPK_PRINTF("[DUMP466] cmpMaskTensor after bin2 Compares first 64:\n");
            BSA_TOPK_DUMP_TENSOR(cmpMaskTensor, 466, 64);
        }

        // === DUMP 470: cmpMaskTensor after Compares (first round, first tile, bin 1 only) ===
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0 && binMask == 1) {
            BSA_TOPK_DUMP_TENSOR(cmpMaskTensor, 470, 32);
        }

        uint64_t rsvdCnt = 0;
        GatherMask(tempHalf, tempHalf, cmpMaskTensor.ReinterpretCast<uint16_t>(),
                   true, curTileLen, {1, 1, 0, 0}, rsvdCnt);
        VToSSync();

        // === DUMP 480: rsvdCnt after GatherMask (first round, first tile) ===
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && roundId == static_cast<int32_t>(round_) - 1 && tileId == 0) {
            BSA_TOPK_PRINTF("[DUMP480] binMask=%d rsvdCnt=%lu cumSumOne=%d\n",
                binMask, rsvdCnt, cumSumOne + static_cast<int32_t>(rsvdCnt));
        }

        cumSumOne += rsvdCnt;

        tileHist[binMask * tileHistBinStride - tileHistBinOffset]
            .SetValue(tileId, cumSumOne);

        globalHist.SetValue(binMask, globalHist.GetValue(binMask) + cumSumOne);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::VToSSync()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SToVSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE2ToMTE3Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventID);
    WaitFlag<HardEvent::MTE2_MTE3>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE3ToMTE2Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE3ToVSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventID);
    WaitFlag<HardEvent::MTE3_V>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::VToMTE3Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE2ToVSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventID);
    WaitFlag<HardEvent::MTE2_V>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE2ToSSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventID);
    WaitFlag<HardEvent::MTE2_S>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::MTE3ToSSync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventID);
    WaitFlag<HardEvent::MTE3_S>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SToMTE3Sync()
{
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventID);
    WaitFlag<HardEvent::S_MTE3>(eventID);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyIn(
    TBuf<TPosition::VECIN>& xBuf, uint64_t dataNum, uint64_t xOffset)
{
    LocalTensor<IN_T> xLocal = xBuf.Get<IN_T>();
    int32_t ubOffset = 0;
    if constexpr (sizeof(IN_T) == 2) {
        ubOffset = tileLen_;
    }
    uint64_t xAlign = UB_BLOCK_SIZE / sizeof(IN_T);
    uint8_t xPadNum = static_cast<uint8_t>((xAlign - dataNum % xAlign) % xAlign);
    DataCopyExtParams xCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(IN_T) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<IN_T> xPadParams{true, 0, xPadNum, 0};
    DataCopyPad(xLocal[ubOffset], attnScoreGmLocal[xOffset], xCopyParams, xPadParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileHistWs2Ub(
    LocalTensor<int32_t>& dstTensor, uint32_t srcOffset, uint32_t dataNum)
{
    DataCopyExtParams params{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, tileHistGm_[srcOffset], params, padParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileTopKWs2Ub(
    LocalTensor<int32_t>& dstTensor, uint32_t srcOffset, uint32_t dataNum)
{
    DataCopyExtParams params{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(dstTensor, tileTopKGm_[srcOffset], params, padParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileTopKUb2Ws(
    LocalTensor<int32_t>& srcTensor, uint32_t dstOffset, uint32_t dataNum)
{
    DataCopyExtParams params{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPad(tileTopKGm_[dstOffset], srcTensor, params);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearTileTopKInWs()
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    for (int i = 0; i < static_cast<int>(tileNumRepeatTimes_); i++) {
        Duplicate<int32_t>(tileTopK, 0, MAX_TILE_NUM_IN_UB);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, i * MAX_TILE_NUM_IN_UB, MAX_TILE_NUM_IN_UB);
    }
    if (tileNumRemain_ > 0) {
        Duplicate<int32_t>(tileTopK, 0, tileNumRemain_);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, tileNumRepeatTimes_ * MAX_TILE_NUM_IN_UB, tileNumRemain_);
    }
    MTE3ToVSync();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearTileHistInWs(
    uint32_t dstOffset, uint32_t dataNum)
{
    if (dataNum == 0) return;
    uint32_t repeatTimes = dataNum / MAX_TILE_NUM_IN_UB;
    uint32_t remain = dataNum % MAX_TILE_NUM_IN_UB;
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    for (uint32_t i = 0; i < repeatTimes; i++) {
        Duplicate<int32_t>(tileHist, 0, MAX_TILE_NUM_IN_UB);
        VToMTE3Sync();
        DataCopyExtParams params{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(sizeof(int32_t) * MAX_TILE_NUM_IN_UB), 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + i * MAX_TILE_NUM_IN_UB], tileHist, params);
    }
    if (remain > 0) {
        Duplicate<int32_t>(tileHist, 0, remain);
        VToMTE3Sync();
        DataCopyExtParams params{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(sizeof(int32_t) * remain), 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + repeatTimes * MAX_TILE_NUM_IN_UB], tileHist, params);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyTileHistInWs(
    uint32_t dstOffset, uint32_t srcOffset)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    for (uint32_t i = 0; i <= tileNumRepeatTimes_; i++) {
        uint32_t dataNum = (i == tileNumRepeatTimes_) ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0) continue;
        CopyTileHistWs2Ub(tileHist, srcOffset + i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToMTE3Sync();
        DataCopyExtParams params{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
        DataCopyPad(tileHistGm_[dstOffset + i * MAX_TILE_NUM_IN_UB], tileHist, params);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTileHistWs2Ub(
    LocalTensor<int32_t>& dstTensor, LocalTensor<int32_t>& srcTensor,
    uint32_t srcOffset0, uint32_t srcOffset1, uint32_t dataNum)
{
    DataCopyExtParams params{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(srcTensor, tileHistGm_[srcOffset0], params, padParams);
    DataCopyPad(srcTensor[MAX_TILE_NUM_IN_UB_BY2], tileHistGm_[srcOffset1], params, padParams);
    MTE2ToVSync();
    Sub(dstTensor, srcTensor, srcTensor[MAX_TILE_NUM_IN_UB_BY2], dataNum);
    PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTileHistInWs(
    uint32_t dstOffset, uint32_t srcOffset0, uint32_t srcOffset1, uint32_t dataNum)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    SubTileHistWs2Ub(tileHist, tileHist, srcOffset0, srcOffset1, dataNum);
    VToMTE3Sync();
    DataCopyExtParams params{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * dataNum), 0, 0, 0};
    DataCopyPad(tileHistGm_[dstOffset], tileHist, params);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTileHistInWsAll(
    uint32_t dstOffset, uint32_t srcOffset0, uint32_t srcOffset1)
{
    for (uint32_t i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = (i == tileNumBy2RepeatTimes_) ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0) continue;
        SubTileHistInWs(dstOffset + i * MAX_TILE_NUM_IN_UB_BY2,
            srcOffset0 + i * MAX_TILE_NUM_IN_UB_BY2,
            srcOffset1 + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE3ToMTE2Sync();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AddTileHist2TileTopKInWs(uint64_t tileHistOffset)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> tileTopK = tileHist[MAX_TILE_NUM_IN_UB_BY2];
    for (uint32_t i = 0; i <= tileNumBy2RepeatTimes_; i++) {
        uint32_t dataNum = (i == tileNumBy2RepeatTimes_) ? tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0) continue;
        CopyTileHistWs2Ub(tileHist, tileHistOffset + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE2ToVSync();
        Add(tileTopK, tileTopK, tileHist, dataNum);
        VToMTE3Sync();
        CopyTileTopKUb2Ws(tileTopK, i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
        MTE3ToMTE2Sync();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearHistInWs(int32_t roundId)
{
    LocalTensor<int32_t> tileHist = tileHistAndTopKBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();

    if (roundId == static_cast<int32_t>(round_) - 1) {
        for (int i = 0; i < static_cast<int>(tileNumRepeatTimes_); i++) {
            Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), MAX_TILE_NUM_IN_UB);
            if (tileNumRemain_ == 0 && tailTileLen_ != tileLen_) {
                VToSSync();
                tileHist.SetValue(MAX_TILE_NUM_IN_UB - 1, static_cast<int32_t>(tailTileLen_));
                SToMTE3Sync();
            } else {
                VToMTE3Sync();
            }
            DataCopyExtParams params{
                static_cast<uint16_t>(1),
                static_cast<uint32_t>(sizeof(int32_t) * MAX_TILE_NUM_IN_UB), 0, 0, 0};
            DataCopyPad(tileHistGm_[i * MAX_TILE_NUM_IN_UB], tileHist, params);
        }
        if (tileNumRemain_ > 0) {
            Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), tileNumRemain_);
            if (tailTileLen_ != tileLen_) {
                VToSSync();
                tileHist.SetValue(tileNumRemain_ - 1, static_cast<int32_t>(tailTileLen_));
                SToMTE3Sync();
            } else {
                VToMTE3Sync();
            }
            DataCopyExtParams params{
                static_cast<uint16_t>(1),
                static_cast<uint32_t>(sizeof(int32_t) * tileNumRemain_), 0, 0, 0};
            DataCopyPad(tileHistGm_[tileNumRepeatTimes_ * MAX_TILE_NUM_IN_UB], tileHist, params);
        }
        globalHistBoundaryNum_ = (tileNum_ > 0) ? (tileNum_ - 1) * tileLen_ + tailTileLen_ : 0;
    } else {
        if (boundaryBin < static_cast<int32_t>(numValue_) - 1) {
            SubTileHistInWsAll(0, boundaryBin * static_cast<uint32_t>(tileNum_),
                boundaryBinPrev * static_cast<uint32_t>(tileNum_));
        } else {
            CopyTileHistInWs(0, boundaryBin * static_cast<uint32_t>(tileNum_));
        }
    }

    SToVSync();
    globalHist.SetValue(0, static_cast<int32_t>(globalHistBoundaryNum_));
    MTE3ToVSync();
    ClearTileHistInWs(static_cast<uint32_t>(tileNum_),
                      (numValue_ - 1) * static_cast<uint32_t>(tileNum_));
    MTE3ToMTE2Sync();
}

template <typename BSAT>
template<bool isInit>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutHistToWs(uint64_t maskLen, uint64_t gmOffset)
{
    LocalTensor<int32_t> globalHist = globalHistBuf_.Get<int32_t>();
    if constexpr (isInit) {
        globalHistBoundaryNum_ = globalHist.GetValue(boundaryBin);
        if (boundaryBin < static_cast<int32_t>(numValue_) - 1) {
            globalHistBoundaryNum_ -= globalHist.GetValue(boundaryBinPrev);
        }
        SToVSync();
        Duplicate<int32_t>(globalHist, 0, maskLen);
        VToMTE3Sync();
    } else {
        SToMTE3Sync();
    }
    DataCopyExtParams globalHistParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * maskLen), 0, 0, 0};
    DataCopyPad(globalHistGm_[gmOffset], globalHist, globalHistParams);
    BSA_TOPK_MTE3_TO_S_SYNC();
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyInGlobalHist(LocalTensor<int32_t>& globalHist)
{
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(sizeof(int32_t) * numValue_ * constInfo.aivNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(globalHist, globalHistGm_, copyParams, padParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ReduceGlobalHist(LocalTensor<int32_t>& globalHist)
{
    int32_t coresInBlock = UB_BLOCK_SIZE / sizeof(int32_t) / numValue_;
    int32_t splitPoint = 32;
    while (splitPoint >= static_cast<int32_t>(constInfo.aivNum) && splitPoint > coresInBlock) {
        splitPoint /= 2;
    }
    int32_t remain = static_cast<int32_t>(constInfo.aivNum);
    while (remain > coresInBlock) {
        Add(globalHist, globalHist, globalHist[splitPoint * numValue_], (remain - splitPoint) * numValue_);
        PipeBarrier<PIPE_V>();
        remain = splitPoint;
        splitPoint /= 2;
    }
    VToSSync();
    for (int32_t binMask = 0; binMask < static_cast<int32_t>(numValue_); binMask++) {
        int32_t sum = 0;
        for (int32_t i = 0; i < BSAMin(remain, coresInBlock); i++) {
            sum += globalHist.GetValue(binMask + i * numValue_);
        }
        globalHist.SetValue(binMask, sum);
    }

    // === DUMP 520: Reduced globalHist (core 0 only) ===
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[DUMP520] ReduceGlobalHist: coresInBlock=%d remain=%d aivNum=%u\n",
            coresInBlock, remain, constInfo.aivNum);
        BSA_TOPK_PRINTF("[DUMP520] globalHist[0]=%d [1]=%d [2]=%d [3]=%d sum=%d\n",
            globalHist.GetValue(0), globalHist.GetValue(1),
            globalHist.GetValue(2), globalHist.GetValue(3),
            globalHist.GetValue(0) + globalHist.GetValue(1) +
            globalHist.GetValue(2) + globalHist.GetValue(3));
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::FindBoundaryBin(
    const LocalTensor<int32_t>& globalHist, int32_t roundId)
{
    boundaryBin = -1;
    boundaryBinPrev = -1;
    for (int32_t binMask = static_cast<int32_t>(numValue_) - 1; binMask >= 0; binMask--) {
        if (globalHist.GetValue(binMask) >= static_cast<int32_t>(remainK_)) {
            boundaryBin = binMask;
            boundaryBinPrev = binMask + 1;
            break;
        }
    }
    involvedMask16_ = involvedMask16_ |
        (static_cast<int16_t>(boundaryBin) << (roundId * bitsPerRound_));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AddTileHistToTileTopK(LocalTensor<int32_t>& globalHist)
{
    // === DUMP 510: AddTileHistToTileTopK decision ===
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[DUMP510] AddTileHistToTileTopK: boundaryBin=%d boundaryBinPrev=%d globalHist[boundaryBin]=%d globalHist[boundaryBinPrev]=%d remainK_=%lu\n",
            boundaryBin, boundaryBinPrev,
            globalHist.GetValue(boundaryBin),
            (boundaryBinPrev < static_cast<int32_t>(numValue_)) ? globalHist.GetValue(boundaryBinPrev) : -1,
            remainK_);
    }

    if (globalHist.GetValue(boundaryBin) == static_cast<int32_t>(remainK_)) {
        AddTileHist2TileTopKInWs(boundaryBin * tileNum_);
        totalDefinitelyInTopK_ += globalHist.GetValue(boundaryBin);
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
            BSA_TOPK_PRINTF("[DUMP511] Branch 1: added boundaryBin, totalDefinitelyInTopK_=%lu\n", totalDefinitelyInTopK_);
        }
    } else if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
        AddTileHist2TileTopKInWs(boundaryBinPrev * tileNum_);
        totalDefinitelyInTopK_ += globalHist.GetValue(boundaryBinPrev);
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
            BSA_TOPK_PRINTF("[DUMP511] Branch 2: added boundaryBinPrev, totalDefinitelyInTopK_=%lu\n", totalDefinitelyInTopK_);
        }
    } else {
        if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
            BSA_TOPK_PRINTF("[DUMP511] Branch 3: nothing added\n");
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutBoundaryBinCumSum(LocalTensor<int32_t>& resTensor)
{
    DataCopyExtParams boundaryBinSumCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0, 0, 0};
    DataCopyPad(boundaryBinCumSumGm_[GetBlockIdx()], resTensor, boundaryBinSumCopyParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ComputeCumSumPrev(
    LocalTensor<int32_t>& resTensor, int32_t& cumSumValuePrev)
{
    DataCopyExtParams copyInParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(sizeof(int32_t) * constInfo.aivNum), 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(resTensor, boundaryBinCumSumGm_, copyInParams, padParams);
    MTE2ToSSync();
    cumSumValuePrev = 0;
    for (int32_t coreId = 0; coreId < static_cast<int32_t>(GetBlockIdx()); coreId++) {
        cumSumValuePrev += resTensor.GetValue(coreId);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutCoreTopK(
    LocalTensor<int32_t>& resTensor, int32_t totalTileTopKInCore)
{
    resTensor.SetValue(0, totalTileTopKInCore);
    SToMTE3Sync();
    DataCopyExtParams coreTopKParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * 1), 0, 0, 0};
    DataCopyPad(coreTopKGm_[GetBlockIdx()], resTensor, coreTopKParams);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::HandleLastRoundBoundary(
    LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32)
{
    uint32_t reduceSumValue = 0;
    for (int32_t i = 0; i <= static_cast<int32_t>(tileNumBy2RepeatTimes_); i++) {
        uint32_t dataNum = (i == static_cast<int32_t>(tileNumBy2RepeatTimes_)) ?
            tileNumBy2Remain_ : MAX_TILE_NUM_IN_UB_BY2;
        if (dataNum == 0) continue;

        if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
            SubTileHistWs2Ub(resTensor, resTensor,
                boundaryBin * static_cast<uint32_t>(tileNum_) + i * MAX_TILE_NUM_IN_UB_BY2,
                boundaryBinPrev * static_cast<uint32_t>(tileNum_) + i * MAX_TILE_NUM_IN_UB_BY2,
                dataNum);
            MTE2ToVSync();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
            PipeBarrier<PIPE_V>();
        } else {
            CopyTileHistWs2Ub(resTensor,
                boundaryBin * static_cast<uint32_t>(tileNum_) + i * MAX_TILE_NUM_IN_UB_BY2, dataNum);
            MTE2ToVSync();
            Cast(tileHistFp32, resTensor, RoundMode::CAST_NONE, dataNum);
            PipeBarrier<PIPE_V>();
        }

        int32_t int32Align = UB_BLOCK_SIZE / sizeof(int32_t);
        int32_t safeSumTileNum = FLOAT32_SAFE_INT / static_cast<int32_t>(tileLen_);
        safeSumTileNum = (safeSumTileNum / int32Align) * int32Align;
        // Ensure safeSumTileNum is at least int32Align to avoid segSize = 0
        if (safeSumTileNum == 0) {
            safeSumTileNum = int32Align;
        }
        int32_t segSize = static_cast<int32_t>(dataNum) < safeSumTileNum ?
            static_cast<int32_t>(dataNum) : safeSumTileNum;
        for (int32_t segOff = 0; segOff < static_cast<int32_t>(dataNum); segOff += segSize) {
            int32_t curSegLen = BSAMin(segSize, static_cast<int32_t>(dataNum) - segOff);
            ReduceSum(tileHistFp32, tileHistFp32[segOff], tileHistFp32[segOff], curSegLen);
            VToSSync();
            reduceSumValue += static_cast<uint32_t>(static_cast<int32_t>(tileHistFp32.GetValue(0)));
        }
    }

    resTensor.SetValue(0, static_cast<int32_t>(reduceSumValue));
    SToMTE3Sync();
    CopyOutBoundaryBinCumSum(resTensor);
    SyncAll();

    int32_t cumSumValuePrev = 0;
    ComputeCumSumPrev(resTensor, cumSumValuePrev);
    int32_t remainCoreBoundaryNum = static_cast<int32_t>(remainK_) - cumSumValuePrev;

    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF(
            "[SDBG550] reduceBoundary=%u cumPrev=%d remainCore=%d remainK=%lu boundary=%d prev=%d\n",
            reduceSumValue, cumSumValuePrev, remainCoreBoundaryNum, remainK_,
            boundaryBin, boundaryBinPrev);
    }

    // === DUMP 550: HandleLastRoundBoundary key values ===
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[DUMP550] HandleLastRoundBoundary: reduceSumValue=%u cumSumValuePrev=%d remainCoreBoundaryNum=%d remainK_=%lu\n",
            reduceSumValue, cumSumValuePrev, remainCoreBoundaryNum, remainK_);
    }

    if (remainCoreBoundaryNum > 0) {
        LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
        LocalTensor<int32_t> tileHistBoundary = tileTopK[MAX_TILE_NUM_IN_UB_BY3];
        LocalTensor<int32_t> tileHistBoundaryPrev = tileTopK[2 * MAX_TILE_NUM_IN_UB_BY3];
        for (int32_t i = 0; i <= static_cast<int32_t>(tileNumBy3RepeatTimes_); i++) {
            uint32_t dataNum = (i == static_cast<int32_t>(tileNumBy3RepeatTimes_)) ?
                tileNumBy3Remain_ : MAX_TILE_NUM_IN_UB_BY3;
            if (dataNum == 0) continue;
            uint32_t startTileId = i * MAX_TILE_NUM_IN_UB_BY3;
            uint32_t endTileId = startTileId + dataNum;

            CopyTileHistWs2Ub(tileHistBoundary,
                boundaryBin * static_cast<uint32_t>(tileNum_) + startTileId, dataNum);
            CopyTileHistWs2Ub(tileHistBoundaryPrev,
                boundaryBinPrev * static_cast<uint32_t>(tileNum_) + startTileId, dataNum);
            CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
            MTE2ToSSync();

            for (int32_t tileId = static_cast<int32_t>(startTileId);
                 tileId < static_cast<int32_t>(endTileId); tileId++) {
                int32_t curTileBoundaryNum =
                    tileHistBoundary.GetValue(tileId - static_cast<int32_t>(startTileId));
                if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
                    curTileBoundaryNum -=
                        tileHistBoundaryPrev.GetValue(tileId - static_cast<int32_t>(startTileId));
                }

                // === DUMP 560: Per-tile boundary allocation (core 0, first tile only) ===
                if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && tileId == 0) {
                    BSA_TOPK_PRINTF("[DUMP560] tileId=%d curTileBoundaryNum=%d remainCoreBoundaryNum=%d tileTopK_before=%d\n",
                        tileId, curTileBoundaryNum, remainCoreBoundaryNum,
                        tileTopK.GetValue(tileId - static_cast<int32_t>(startTileId)));
                }

                if (curTileBoundaryNum < remainCoreBoundaryNum) {
                    remainCoreBoundaryNum -= curTileBoundaryNum;
                    tileTopK.SetValue(tileId - static_cast<int32_t>(startTileId),
                        tileTopK.GetValue(tileId - static_cast<int32_t>(startTileId)) + curTileBoundaryNum);
                } else {
                    tileTopK.SetValue(tileId - static_cast<int32_t>(startTileId),
                        tileTopK.GetValue(tileId - static_cast<int32_t>(startTileId)) + remainCoreBoundaryNum);
                    remainCoreBoundaryNum = 0;
                    break;
                }
            }
            SToMTE3Sync();
            CopyTileTopKUb2Ws(tileTopK, startTileId, dataNum);
            MTE3ToMTE2Sync();
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::WriteCoreTopKFromWs(
    LocalTensor<int32_t>& resTensor, LocalTensor<float>& tileHistFp32)
{
    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    int32_t totalTileTopKInCore = 0;
    int32_t int32Align = UB_BLOCK_SIZE / sizeof(int32_t);
    int32_t safeSumTileNum = FLOAT32_SAFE_INT / static_cast<int32_t>(tileLen_);
    safeSumTileNum = (safeSumTileNum / int32Align) * int32Align;
    if (safeSumTileNum == 0) {
        safeSumTileNum = int32Align;
    }
    for (int32_t i = 0; i <= static_cast<int32_t>(tileNumRepeatTimes_); i++) {
        uint32_t dataNum = (i == static_cast<int32_t>(tileNumRepeatTimes_)) ?
            tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0) continue;
        CopyTileTopKWs2Ub(tileTopK, i * MAX_TILE_NUM_IN_UB, dataNum);
        MTE2ToVSync();
        int32_t segSize = static_cast<int32_t>(dataNum) < safeSumTileNum ?
            static_cast<int32_t>(dataNum) : safeSumTileNum;
        for (int32_t segOff = 0; segOff < static_cast<int32_t>(dataNum); segOff += segSize) {
            int32_t curSegLen = BSAMin(segSize, static_cast<int32_t>(dataNum) - segOff);
            SToVSync();
            Cast(tileHistFp32, tileTopK[segOff], RoundMode::CAST_NONE, curSegLen);
            PipeBarrier<PIPE_V>();
            ReduceSum(tileHistFp32, tileHistFp32, tileHistFp32, curSegLen);
            VToSSync();
            totalTileTopKInCore += static_cast<int32_t>(tileHistFp32.GetValue(0));
        }
    }

    // === DUMP 570: WriteCoreTopKFromWs totalTileTopKInCore ===
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[DUMP570] WriteCoreTopKFromWs: totalTileTopKInCore=%d\n", totalTileTopKInCore);
    }
    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[SDBG570] totalTileTopKInCore=%d\n", totalTileTopKInCore);
    }

    CopyOutCoreTopK(resTensor, totalTileTopKInCore);
}

template <typename BSAT>
__aicore__ inline bool BSARadixTopKService<BSAT>::Update(int32_t roundId)
{
    LocalTensor<int32_t> resTensor = outIndexBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;

    CopyInGlobalHist(globalHist);
    MTE2ToVSync();
    ReduceGlobalHist(globalHist);
    FindBoundaryBin(globalHist, roundId);

    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF(
            "[SDBG500] round=%d remainBefore=%lu hist=[%d,%d,%d,%d] boundary=%d prev=%d involved=0x%04x\n",
            roundId, remainK_, globalHist.GetValue(0), globalHist.GetValue(1),
            globalHist.GetValue(2), globalHist.GetValue(3), boundaryBin, boundaryBinPrev,
            static_cast<uint16_t>(involvedMask16_));
    }

    AddTileHistToTileTopK(globalHist);

    remainK_ = kValue_ - totalDefinitelyInTopK_;
    LocalTensor<float> tileHistFp32 = resTensor.ReinterpretCast<float>();

    if (roundId == 0 && remainK_ > 0) {
        HandleLastRoundBoundary(resTensor, tileHistFp32);
    }

    bool isLastRound = (remainK_ == 0 || roundId == 0);
    if (isLastRound) {
        WriteCoreTopKFromWs(resTensor, tileHistFp32);
    }

    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF(
            "[SDBG590] round=%d remainAfter=%lu totalDefinitely=%lu isLast=%d\n",
            roundId, remainK_, totalDefinitelyInTopK_, static_cast<int32_t>(isLastRound));
    }

    return isLastRound;
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMaskGT(
    LocalTensor<half>& xLocal, half boundaryHalf,
    uint64_t curTileLen, uint64_t outputGmOffset, int32_t& curTileK)
{
    uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

    if (sortLen_ <= 64) {
        LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
        uint8_t defaultValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 0 : 1;
        uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
        uint32_t writeCnt = 0;
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(curTileLen); ++idx) {
            maskLocal.SetValue(idx, defaultValue);
        }
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(curTileLen) && writeCnt < static_cast<uint32_t>(curTileK);
             ++idx) {
            float value = static_cast<float>(xLocal.GetValue(idx));
            float boundary = static_cast<float>(boundaryHalf);
            bool selected = (constInfo.sparsityMode == BSASparseMode::BottomK) ? (value < boundary) : (value > boundary);
            if (selected) {
                maskLocal.SetValue(idx, writeValue);
                ++writeCnt;
            }
        }

        if (BSA_TOPK_SMALL_SORT_DEBUG && GetBlockIdx() == 0 && outputGmOffset == 0) {
            BSA_TOPK_PRINTF("[SDBG705] scalar GT writeCnt=%u curTileKBefore=%d boundary=%f\n",
                writeCnt, curTileK, static_cast<float>(boundaryHalf));
        }

        SToMTE3Sync();
        DataCopyPad(maskOutGmU8Local[outputGmOffset], maskLocal,
                    DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
        MTE3ToMTE2Sync();

        curTileK -= static_cast<int32_t>(writeCnt);
        return;
    }

    LocalTensor<uint8_t> cmpMask = tempBuf_.Get<uint8_t>();
    AscendC::CMPMODE cmpMode = (constInfo.sparsityMode == BSASparseMode::BottomK) ?
        AscendC::CMPMODE::LT : AscendC::CMPMODE::GT;
    Compares<half, uint8_t>(cmpMask, xLocal, boundaryHalf, cmpMode, alignLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> tempHalf = outIndexBuf_.Get<half>();
    uint64_t rsvdCnt = 0;
    GatherMask(tempHalf, tempHalf, cmpMask.template ReinterpretCast<uint16_t>(),
               true, curTileLen, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0 && outputGmOffset == 0) {
        BSA_TOPK_PRINTF("[SDBG700] GT boundary=%f rsvdCnt=%lu curTileKBefore=%d\n",
            static_cast<float>(boundaryHalf), rsvdCnt, curTileK);
    }

    LocalTensor<half> maskHalf = outValueBuf_.Get<half>();
    LocalTensor<half> onesHalf = outIndexBuf_.Get<half>();

    if (constInfo.sparsityMode == BSASparseMode::TopK) {
        Duplicate<half>(maskHalf, (half)0, alignLen);
        Duplicate<half>(onesHalf, (half)1, alignLen);
        PipeBarrier<PIPE_V>();
        Select<half>(maskHalf, cmpMask, onesHalf, (half)0,
                    SELMODE::VSEL_TENSOR_SCALAR_MODE, alignLen);
    } else {
        Duplicate<half>(maskHalf, (half)1, alignLen);
        Duplicate<half>(onesHalf, (half)0, alignLen);
        PipeBarrier<PIPE_V>();
        Select<half>(maskHalf, cmpMask, onesHalf, (half)1,
                    SELMODE::VSEL_TENSOR_SCALAR_MODE, alignLen);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> maskU8 = outIndexBuf_.Get<uint8_t>();
    Cast<uint8_t, half>(maskU8, maskHalf, RoundMode::CAST_NONE, alignLen);
    PipeBarrier<PIPE_V>();

    VToMTE3Sync();
    DataCopyPad(maskOutGmU8Local[outputGmOffset], maskU8,
                DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
    MTE3ToSSync();
    MTE3ToMTE2Sync();

    VToSSync();

    // === DUMP 730: rsvdCnt after GatherMask ===
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && outputGmOffset == 0) {
        BSA_TOPK_PRINTF("[DUMP730] GT rsvdCnt=%lu curTileK=%d->%d\n",
            rsvdCnt, curTileK, curTileK - static_cast<int32_t>(rsvdCnt));
    }

    curTileK -= static_cast<int32_t>(rsvdCnt);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMaskEQ(
    LocalTensor<half>& xLocal, half boundaryHalf,
    uint64_t curTileLen, uint64_t outputGmOffset, int32_t& curTileK)
{
    uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

    if (sortLen_ <= 64) {
        LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
        DataCopyPad(maskLocal, maskOutGmU8Local[outputGmOffset],
                    DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(uint8_t)), 0, 0, 0),
                    DataCopyPadExtParams<uint8_t>(false, 0,
                        static_cast<uint8_t>(alignLen - curTileLen), 0));
        MTE2ToSSync();

        uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
        uint32_t writeCnt = 0;
        for (uint32_t idx = 0; idx < static_cast<uint32_t>(curTileLen) && writeCnt < static_cast<uint32_t>(curTileK);
             ++idx) {
            if (static_cast<float>(xLocal.GetValue(idx)) == static_cast<float>(boundaryHalf)) {
                maskLocal.SetValue(idx, writeValue);
                ++writeCnt;
            }
        }

        if (BSA_TOPK_SMALL_SORT_DEBUG && GetBlockIdx() == 0 && outputGmOffset == 0) {
            BSA_TOPK_PRINTF("[SDBG805] scalar EQ writeCnt=%u curTileKBefore=%d\n", writeCnt, curTileK);
        }

        SToMTE3Sync();
        DataCopyPad(maskOutGmU8Local[outputGmOffset], maskLocal,
                    DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
        MTE3ToMTE2Sync();

        curTileK -= static_cast<int32_t>(writeCnt);
        return;
    }

    LocalTensor<uint8_t> cmpMask = tempBuf_.Get<uint8_t>();
    Compares<half, uint8_t>(cmpMask, xLocal, boundaryHalf, AscendC::CMPMODE::EQ, alignLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> vecIndex = outIndexBuf_.Get<int32_t>();
    for (uint64_t i = 0; i < curTileLen; i++) {
        vecIndex.SetValue(i, static_cast<int32_t>(i));
    }
    SToVSync();

    uint64_t eqRsvdCnt = 0;
    GatherMask(vecIndex, vecIndex, cmpMask.template ReinterpretCast<uint32_t>(),
               true, curTileLen, {1, 1, 0, 0}, eqRsvdCnt);
    VToSSync();

    uint32_t writeCnt = BSAMin(static_cast<uint32_t>(curTileK), static_cast<uint32_t>(eqRsvdCnt));

    if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0 && outputGmOffset == 0) {
        BSA_TOPK_PRINTF("[SDBG800] EQ boundary=%f eqRsvdCnt=%lu writeCnt=%u curTileKBefore=%d\n",
            static_cast<float>(boundaryHalf), eqRsvdCnt, writeCnt, curTileK);
        uint32_t printCnt = BSAMin(writeCnt, static_cast<uint32_t>(12));
        for (uint32_t i = 0; i < printCnt; i++) {
            BSA_TOPK_PRINTF("[SDBG801] eqIdx[%u]=%d\n", i, vecIndex.GetValue(i));
        }
    }

    LocalTensor<uint8_t> maskLocal = maskLocalBuf_.Get<uint8_t>();
    DataCopyPad(maskLocal, maskOutGmU8Local[outputGmOffset],
                DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(uint8_t)), 0, 0, 0),
                DataCopyPadExtParams<uint8_t>(false, 0,
                    static_cast<uint8_t>(alignLen - curTileLen), 0));
    MTE2ToSSync();

    uint8_t writeValue = (constInfo.sparsityMode == BSASparseMode::TopK) ? 1 : 0;
    for (uint32_t i = 0; i < writeCnt; i++) {
        int32_t idx = vecIndex.GetValue(i);
        maskLocal.SetValue(idx, writeValue);
    }

    SToMTE3Sync();
    DataCopyPad(maskOutGmU8Local[outputGmOffset], maskLocal,
                DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));
    MTE3ToMTE2Sync();

    curTileK -= static_cast<int32_t>(writeCnt);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TileTopK(uint32_t batchIdx, uint32_t headIdx)
{
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[TopK] v6.6b: TileTopK implemented\n");
    }

    LocalTensor<int32_t> tileTopK = tileHistAndTopKBuf_.Get<int32_t>();
    uint64_t outputOffset = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) * sortLen_;

    // Restore boundary value (twiddle inverse transform)
    int16_t boundaryInt16 = involvedMask16_ ^ ((~(involvedMask16_ >> 15)) | 0x8000);
    half boundaryHalf = *reinterpret_cast<half*>(&boundaryInt16);
    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        boundaryHalf = static_cast<half>(-static_cast<float>(boundaryHalf));
    }

    // === DUMP 600: TileTopK entry - boundaryHalf ===
    if (GetBlockIdx() == 0) {
        BSA_TOPK_SYNC_PRINTF("[SYNC_PROBE] tileTopK boundary involved=0x%04x int16=0x%04x half=%f\n",
            involvedMask16_, boundaryInt16, static_cast<float>(boundaryHalf));
    }
    if (BSA_TOPK_DEBUG && GetBlockIdx() == 0) {
        BSA_TOPK_PRINTF("[DUMP600] TileTopK: involvedMask16_=0x%04x boundaryInt16=0x%04x boundaryHalf=%f\n",
            involvedMask16_, boundaryInt16, static_cast<float>(boundaryHalf));
    }

    // WS version: batch read tileTopK, use ping-pong double buffering
    for (uint32_t repeatId = 0; repeatId <= tileNumRepeatTimes_; repeatId++) {
        uint32_t startTileId = repeatId * MAX_TILE_NUM_IN_UB;
        uint32_t dataNum = (repeatId == tileNumRepeatTimes_) ? tileNumRemain_ : MAX_TILE_NUM_IN_UB;
        if (dataNum == 0) continue;

        CopyTileTopKWs2Ub(tileTopK, startTileId, dataNum);
        MTE2ToSSync();

        if (BSA_TOPK_SMALL_SORT_DEBUG && sortLen_ <= 64 && GetBlockIdx() == 0) {
            BSA_TOPK_PRINTF("[SDBG610] repeat=%u dataNum=%u tileTopK0=%d\n",
                repeatId, dataNum, tileTopK.GetValue(0));
        }

        // Prefetch first tile
        uint32_t firstTileId = startTileId;
        uint64_t firstTileLen = (firstTileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
        uint64_t firstOffset = blockOffset_ + firstTileId * tileLen_;
        CopyIn(xBufPing_, firstTileLen, firstOffset);
        pingPongFlag = true;
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

        for (uint32_t i = 0; i < dataNum; i++) {
            uint64_t tileId = startTileId + i;
            int32_t curTileK = tileTopK.GetValue(i);
            // v2.3: no longer skip curTileK==0, GT phase naturally handles all tiles

            uint64_t curTileLen = (tileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
            uint64_t tileGmOffset = blockOffset_ + tileId * tileLen_;
            uint64_t outputGmOffset = outputOffset + tileGmOffset;

            TBuf<TPosition::VECIN>& curBuf = pingPongFlag ? xBufPing_ : xBufPong_;
            TBuf<TPosition::VECIN>& nextBuf = pingPongFlag ? xBufPong_ : xBufPing_;

            // Prefetch next tile
            if (i < dataNum - 1) {
                uint64_t nextTileId = tileId + 1;
                uint64_t nextTileLen = (nextTileId == tileNum_ - 1) ? tailTileLen_ : tileLen_;
                uint64_t nextOffset = blockOffset_ + nextTileId * tileLen_;
                CopyIn(nextBuf, nextTileLen, nextOffset);
            }

            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);

            LocalTensor<half> xLocal = curBuf.template Get<half>()[tileLen_].template ReinterpretCast<half>();

            // === DUMP 700: Before GT phase ===
            int32_t curTileKBeforeGT = curTileK;
            if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && tileId == 0) {
                BSA_TOPK_PRINTF("[DUMP700] Before GT: tileId=%lu curTileK=%d curTileLen=%lu tileGmOffset=%lu\n",
                    tileId, curTileK, curTileLen, tileGmOffset);
            }

            SubTopKAndWriteMaskGT(xLocal, boundaryHalf, curTileLen, outputGmOffset, curTileK);

            // === DUMP 701: After GT phase ===
            if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && tileId == 0) {
                BSA_TOPK_PRINTF("[DUMP701] After GT: curTileK=%d (was %d, selected %d)\n",
                    curTileK, curTileKBeforeGT, curTileKBeforeGT - curTileK);
            }

            if (curTileK > 0) {
                // === DUMP 800: Before EQ phase ===
                int32_t curTileKBeforeEQ = curTileK;
                if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && tileId == 0) {
                    BSA_TOPK_PRINTF("[DUMP800] Before EQ: curTileK=%d\n", curTileK);
                }

                SubTopKAndWriteMaskEQ(xLocal, boundaryHalf, curTileLen, outputGmOffset, curTileK);

                // === DUMP 801: After EQ phase ===
                if (BSA_TOPK_DEBUG && GetBlockIdx() == 0 && tileId == 0) {
                    BSA_TOPK_PRINTF("[DUMP801] After EQ: curTileK=%d (was %d, selected %d)\n",
                        curTileK, curTileKBeforeEQ, curTileKBeforeEQ - curTileK);
                }
            }

            if (i < dataNum - 1) {
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToVForX_);
            }
            pingPongFlag = !pingPongFlag;
        }
    }
}

#undef BSA_TOPK_PRINTF
#undef BSA_TOPK_SYNC_PRINTF
#undef BSA_TOPK_DUMP_TENSOR
#undef BSA_TOPK_MTE3_TO_S_SYNC

#endif // BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H
