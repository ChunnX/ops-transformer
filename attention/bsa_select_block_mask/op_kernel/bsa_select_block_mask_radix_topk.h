#ifndef BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H
#define BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"

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
                                   GlobalTensor<uint8_t> &maskTmpGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void CopyMaskToOutput(uint32_t batchIdx, uint32_t headIdx,
                                             GlobalTensor<OUT_T> &maskOutGm);

private:
    __aicore__ inline void TwiddleInB16(LocalTensor<int16_t> &xLocal, uint64_t dataLen);
    __aicore__ inline void DoAndMask(LocalTensor<int16_t> &xLocal, uint64_t dataLen);
    __aicore__ inline void CalcCumsumHistogram(LocalTensor<int16_t> &xLocal, int32_t roundId,
                                                int32_t tileId, uint64_t curTileLen);
    __aicore__ inline void CopyOutHistToWs();
    __aicore__ inline bool Update(int32_t roundId);
    __aicore__ inline void ClearHist(int32_t roundId);
    __aicore__ inline void TopKPerTile(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void SubTopKAndWriteMask(LocalTensor<float> &xLocalFp32, float boundaryNumber,
                                                uint64_t curTileLen, uint64_t tileGmOffset,
                                                int32_t &curTileK);
    __aicore__ inline void NegateAttnScore();

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> attnScoreGmLocal;
    GlobalTensor<int32_t> topkWorkspaceGmLocal;
    GlobalTensor<uint8_t> maskTmpGmLocal;

    GlobalTensor<int32_t> globalHistGm_;
    GlobalTensor<int32_t> boundaryBinCumSumGm_;
    GlobalTensor<int32_t> coreTopKGm_;

    TBuf<TPosition::VECIN> xBuf0_;
    TBuf<TPosition::VECIN> xBuf1_;
    TBuf<TPosition::VECOUT> maskLocalBuf_;
    TBuf<TPosition::VECOUT> globalHistBuf_;
    TBuf<TPosition::VECCALC> tileTopKBuf_;
    TBuf<TPosition::VECCALC> tempBuf_;

    int16_t involvedMask16_;
    int16_t andMask16_;
    int32_t boundaryBin;
    int32_t boundaryBinPrev;
    int32_t globalHistBoundaryNum_;
    uint64_t totalDefinitelyInTopK_;
    uint64_t remainK_;

    uint64_t tileLen_;
    uint64_t tileNum_;
    uint64_t tileNumAlign_;
    uint64_t sortLen_;
    uint64_t kValue_;
    uint32_t round_;
    uint32_t numValue_;
    uint32_t bitsPerRound_;

    event_t eventBuf0_;
    bool xBufUse0_;
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

    this->tileLen_ = BSAMin(static_cast<uint64_t>(7168), sortLen_);
    this->tileNum_ = (sortLen_ + tileLen_ - 1) / tileLen_;
    this->tileNumAlign_ = BSACeilDiv(tileNum_, static_cast<uint64_t>(8)) * 8;
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(xBuf0_, tileLen_ * sizeof(float));
    pipe->InitBuffer(xBuf1_, tileLen_ * sizeof(float));
    pipe->InitBuffer(tileTopKBuf_, tileNumAlign_ * sizeof(int32_t));

    uint32_t tileHistSize = numValue_ * tileNumAlign_ * sizeof(int32_t);
    uint32_t cmpMaskSize = tileLen_ * sizeof(uint8_t) / BYTE_SIZE;
    uint32_t tempBufSize = BSAMax(tileHistSize, cmpMaskSize);
    pipe->InitBuffer(tempBuf_, tempBufSize);

    pipe->InitBuffer(globalHistBuf_, numValue_ * sizeof(int32_t));
    pipe->InitBuffer(maskLocalBuf_, tileLen_ * sizeof(int32_t) + tileLen_ * sizeof(int16_t) + tileLen_ * sizeof(uint8_t));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitGM(
    GlobalTensor<IN_T> &attnScoreGm,
    GlobalTensor<int32_t> &topkWorkspaceGm,
    GlobalTensor<uint8_t> &maskTmpGm)
{
    this->attnScoreGmLocal = attnScoreGm;
    this->topkWorkspaceGmLocal = topkWorkspaceGm;
    this->maskTmpGmLocal = maskTmpGm;

    uint32_t coreNum = constInfo.coreNum;
    this->globalHistGm_.SetGlobalBuffer((__gm__ int32_t *)topkWorkspaceGm.GetPhyAddr());
    this->boundaryBinCumSumGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + numValue_ * coreNum * sizeof(int32_t)));
    this->coreTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + (numValue_ * coreNum + coreNum) * sizeof(int32_t)));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AllocEventID()
{
    eventBuf0_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::FreeEventID()
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx)
{
    involvedMask16_ = 0;
    andMask16_ = static_cast<int16_t>(0xC000);
    totalDefinitelyInTopK_ = 0;
    boundaryBin = 0;
    boundaryBinPrev = 1;
    remainK_ = kValue_;

    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        NegateAttnScore();
    }

    bool earlyExit = false;
    for (int32_t roundId = round_ - 1; roundId >= 0; roundId--) {
        CopyOutHistToWs();
        ClearHist(roundId);

        for (uint64_t tileIdx = 0; tileIdx < tileNum_; tileIdx++) {
            uint64_t curTileLen = BSAMin(tileLen_, sortLen_ - tileIdx * tileLen_);
            uint64_t tileGmOffset = tileIdx * tileLen_;

            LocalTensor<IN_T> xLocalIN = xBuf0_.Get<IN_T>();
            uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

            DataCopyPad(xLocalIN, attnScoreGmLocal[tileGmOffset],
                        DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(IN_T)), 0, 0, 0),
                        DataCopyPadExtParams<IN_T>(false, 0,
                            static_cast<uint8_t>(alignLen - curTileLen), 0));
            PipeBarrier<PIPE_V>();

            LocalTensor<int16_t> xLocalInt16 = xLocalIN.template ReinterpretCast<int16_t>();
            TwiddleInB16(xLocalInt16, curTileLen);
            DoAndMask(xLocalInt16, curTileLen);
            CalcCumsumHistogram(xLocalInt16, roundId, tileIdx, curTileLen);
        }

        CopyOutHistToWs();
        SyncAll();
        earlyExit = Update(roundId);
        andMask16_ >>= bitsPerRound_;

        if (earlyExit) {
            break;
        }
    }

    SyncAll();
    TopKPerTile(batchIdx, headIdx);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::NegateAttnScore()
{
    for (uint64_t tileIdx = 0; tileIdx < tileNum_; tileIdx++) {
        uint64_t curTileLen = BSAMin(tileLen_, sortLen_ - tileIdx * tileLen_);
        uint64_t tileGmOffset = tileIdx * tileLen_;

        LocalTensor<IN_T> xLocal = xBuf0_.Get<IN_T>();
        uint64_t alignLen = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

        DataCopyPad(xLocal, attnScoreGmLocal[tileGmOffset],
                    DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(IN_T)), 0, 0, 0),
                    DataCopyPadExtParams<IN_T>(false, 0,
                        static_cast<uint8_t>(alignLen - curTileLen), 0));
        PipeBarrier<PIPE_V>();

        if constexpr (IsSameType<IN_T, bfloat16_t>::value) {
            Muls(xLocal.template ReinterpretCast<half>(),
                 xLocal.template ReinterpretCast<half>(), (half)-1, alignLen);
        } else {
            Muls(xLocal, xLocal, (IN_T)-1, alignLen);
        }
        PipeBarrier<PIPE_V>();

        DataCopyPad(attnScoreGmLocal[tileGmOffset], xLocal,
                    DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(IN_T)), 0, 0));
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TwiddleInB16(LocalTensor<int16_t> &xLocal, uint64_t dataLen)
{
    LocalTensor<int16_t> signMaskTensor = maskLocalBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensor = tempBuf_.Get<int16_t>();

    uint64_t curTileLenAlign = BSAAlignTo(dataLen, static_cast<uint64_t>(VEC_ALIGN_SIZE / sizeof(int16_t)));
    LocalTensor<int16_t> temp1 = tempBuf_.GetWithOffset<int16_t>(
        curTileLenAlign, curTileLenAlign * sizeof(int16_t));

    ShiftRight(tempTensor, xLocal, static_cast<int16_t>(15), dataLen);
    Duplicate<int16_t>(signMaskTensor, XOR_OP_VALUE_16, dataLen);
    Or(tempTensor, tempTensor, signMaskTensor, dataLen);

    And(temp1, xLocal, tempTensor, dataLen);
    Or(xLocal, xLocal, tempTensor, dataLen);
    Not(temp1, temp1, dataLen);
    And(xLocal, xLocal, temp1, dataLen);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::DoAndMask(LocalTensor<int16_t> &xLocal, uint64_t dataLen)
{
    LocalTensor<int16_t> maskTensor = maskLocalBuf_.Get<int16_t>();

    Duplicate<int16_t>(maskTensor, andMask16_, dataLen);
    And(xLocal, xLocal, maskTensor, dataLen);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CalcCumsumHistogram(
    LocalTensor<int16_t> &xLocal, int32_t roundId, int32_t tileId, uint64_t curTileLen)
{
    LocalTensor<float> xLocalFp32 = xBuf1_.Get<float>();
    LocalTensor<int32_t> xLocalInt32 = maskLocalBuf_.Get<int32_t>();

    Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, curTileLen);
    PipeBarrier<PIPE_V>();
    Cast(xLocalInt32, xLocalFp32, RoundMode::CAST_RINT, curTileLen);
    PipeBarrier<PIPE_V>();

    LocalTensor<half> tempHalf = globalHistBuf_.Get<half>();
    LocalTensor<uint8_t> cmpMaskTensor = tempBuf_.Get<uint8_t>();
    LocalTensor<int32_t> tileHist = tempBuf_.GetWithOffset<int32_t>(
        curTileLen / BYTE_SIZE, curTileLen / BYTE_SIZE);
    LocalTensor<int32_t> globalHist = maskLocalBuf_.GetWithOffset<int32_t>(
        numValue_, curTileLen * sizeof(int32_t));

    uint32_t curTileLenAlign = BSAAlignTo(curTileLen, static_cast<uint64_t>(REPEAT_SIZE / sizeof(int32_t)));

    int32_t cumSumOne = 0;
    for (int32_t binMask = numValue_ - 1; binMask > 0; binMask--) {
        int32_t involvedMaskTemp = static_cast<int32_t>(involvedMask16_) |
            static_cast<int32_t>(binMask << (roundId * bitsPerRound_));

        Compares<int32_t, uint8_t>(
            cmpMaskTensor,
            xLocalInt32,
            involvedMaskTemp,
            AscendC::CMPMODE::EQ,
            curTileLenAlign);

        uint64_t rsvdCnt = 0;
        GatherMask(tempHalf, tempHalf, cmpMaskTensor.template ReinterpretCast<uint16_t>(),
                   true, curTileLen, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        cumSumOne += static_cast<int32_t>(rsvdCnt);
        tileHist[binMask * tileNumAlign_].SetValue(tileId, cumSumOne);
        globalHist.SetValue(binMask, globalHist.GetValue(binMask) + cumSumOne);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutHistToWs()
{
    LocalTensor<int32_t> globalHist = maskLocalBuf_.Get<int32_t>();

    DataCopyExtParams globalHistParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * numValue_), 0, 0, 0};
    DataCopyPad(globalHistGm_[GetBlockIdx() * numValue_], globalHist, globalHistParams);
}

template <typename BSAT>
__aicore__ inline bool BSARadixTopKService<BSAT>::Update(int32_t roundId)
{
    LocalTensor<int32_t> tileHist = tempBuf_.GetWithOffset<int32_t>(
        tileLen_ / BYTE_SIZE, tileLen_ / BYTE_SIZE);
    LocalTensor<int32_t> resTensor = maskLocalBuf_.Get<int32_t>();
    LocalTensor<int32_t> globalHist = resTensor;

    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int32_t) * numValue_ * constInfo.coreNum),
        0, 0, 0};
    DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
    DataCopyPad(globalHist, globalHistGm_, copyParams, padParams);
    PipeBarrier<PIPE_V>();

    for (int32_t binMask = 0; binMask < static_cast<int32_t>(numValue_); binMask++) {
        for (int32_t coreId = 1; coreId < static_cast<int32_t>(constInfo.coreNum); coreId++) {
            globalHist.SetValue(binMask, globalHist(binMask) + globalHist(coreId * numValue_ + binMask));
        }
    }

    boundaryBin = -1;
    boundaryBinPrev = -1;
    for (int32_t binMask = numValue_ - 1; binMask >= 0; binMask--) {
        if (globalHist(binMask) >= static_cast<int32_t>(remainK_)) {
            boundaryBin = binMask;
            boundaryBinPrev = binMask + 1;
            break;
        }
    }

    involvedMask16_ = involvedMask16_ | static_cast<int16_t>(boundaryBin << (roundId * bitsPerRound_));

    LocalTensor<int32_t> tileTopK = tileTopKBuf_.Get<int32_t>();
    if (globalHist(boundaryBin) == static_cast<int32_t>(remainK_)) {
        Add(tileTopK, tileTopK, tileHist[boundaryBin * tileNumAlign_], tileNum_);
        totalDefinitelyInTopK_ += globalHist(boundaryBin);
    } else if (boundaryBinPrev < static_cast<int32_t>(numValue_)) {
        Add(tileTopK, tileTopK, tileHist[boundaryBinPrev * tileNumAlign_], tileNum_);
        totalDefinitelyInTopK_ += globalHist(boundaryBinPrev);
    }

    remainK_ = kValue_ - totalDefinitelyInTopK_;

    return (remainK_ == 0 || roundId == 0);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearHist(int32_t roundId)
{
    LocalTensor<int32_t> tileHist = tempBuf_.GetWithOffset<int32_t>(
        tileLen_ / BYTE_SIZE, tileLen_ / BYTE_SIZE);
    LocalTensor<int32_t> globalHist = maskLocalBuf_.Get<int32_t>();

    if (roundId == static_cast<int32_t>(round_) - 1) {
        Duplicate<int32_t>(tileHist, static_cast<int32_t>(tileLen_), tileNum_);
        if (sortLen_ % tileLen_ != 0) {
            tileHist.SetValue(tileNum_ - 1, static_cast<int32_t>(sortLen_ % tileLen_));
        }
        globalHistBoundaryNum_ = static_cast<int32_t>(sortLen_);
    } else {
        if (boundaryBin < static_cast<int32_t>(numValue_) - 1) {
            Sub(tileHist, tileHist[boundaryBin * tileNumAlign_],
                tileHist[boundaryBinPrev * tileNumAlign_], tileNum_);
        } else {
            Copy(tileHist, tileHist[boundaryBin * tileNumAlign_], tileNum_);
        }
    }

    globalHist.SetValue(0, globalHistBoundaryNum_);
    Duplicate<int32_t>(tileHist[tileNumAlign_], 0, (numValue_ - 1) * tileNumAlign_);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TopKPerTile(uint32_t batchIdx, uint32_t headIdx)
{
    uint64_t aivIdx = GetBlockIdx();
    uint64_t aivNum = constInfo.aivNum;
    uint64_t tilesPerCore = tileNum_ / aivNum;
    uint64_t extraTiles = tileNum_ % aivNum;

    uint64_t coreTileStart, coreTileEnd;
    if (aivIdx < extraTiles) {
        coreTileStart = aivIdx * (tilesPerCore + 1);
        coreTileEnd = coreTileStart + tilesPerCore + 1;
    } else {
        coreTileStart = extraTiles * (tilesPerCore + 1) + (aivIdx - extraTiles) * tilesPerCore;
        coreTileEnd = coreTileStart + tilesPerCore;
    }

    int16_t boundaryNumber16 = involvedMask16_ ^ ((~(involvedMask16_ >> 15)) | static_cast<int16_t>(0x8000));
    IN_T boundaryIN = *reinterpret_cast<IN_T *>(&boundaryNumber16);
    float boundaryNumber = static_cast<float>(boundaryIN);

    LocalTensor<int32_t> tileTopK = tileTopKBuf_.Get<int32_t>();

    for (uint64_t tileIdx = coreTileStart; tileIdx < coreTileEnd; tileIdx++) {
        uint64_t curTileLen = BSAMin(tileLen_, sortLen_ - tileIdx * tileLen_);
        uint64_t tileGmOffset = tileIdx * tileLen_;
        int32_t curTileK = tileTopK.GetValue(tileIdx);

        LocalTensor<IN_T> xLocalIN = xBuf0_.Get<IN_T>();
        uint64_t topkTileAlign = BSAAlignTo(curTileLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));
        DataCopyPad(xLocalIN, attnScoreGmLocal[tileGmOffset],
                    DataCopyExtParams(1, static_cast<uint32_t>(curTileLen * sizeof(IN_T)), 0, 0, 0),
                    DataCopyPadExtParams<IN_T>(false, 0,
                        static_cast<uint8_t>(topkTileAlign - curTileLen), 0));
        PipeBarrier<PIPE_V>();

        LocalTensor<float> xLocalFp32 = xBuf1_.Get<float>();
        Cast<float, IN_T>(xLocalFp32, xLocalIN, RoundMode::CAST_NONE, topkTileAlign);
        PipeBarrier<PIPE_V>();

        SubTopKAndWriteMask(xLocalFp32, boundaryNumber, curTileLen, tileGmOffset, curTileK);
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMask(
    LocalTensor<float> &xLocalFp32, float boundaryNumber,
    uint64_t curTileLen, uint64_t tileGmOffset, int32_t &curTileK)
{
    if (curTileK <= 0) {
        return;
    }

    uint64_t curTileLenAlign = BSAAlignTo(curTileLen, static_cast<uint64_t>(REPEAT_SIZE / sizeof(float)));
    uint32_t cmpMaskBytes = static_cast<uint32_t>(curTileLenAlign / BYTE_SIZE);
    uint32_t cmpMaskBytesAlign = BSAAlignTo(cmpMaskBytes, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    LocalTensor<uint8_t> cmpMaskTensor = tempBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskLocal8 = tempBuf_.GetWithOffset<uint8_t>(
        cmpMaskBytesAlign, cmpMaskBytesAlign);

    Compares<float, uint8_t>(cmpMaskTensor, xLocalFp32, boundaryNumber,
                              CMPMODE::GT, curTileLenAlign);

    LocalTensor<int32_t> maskLocal32 = maskLocalBuf_.Get<int32_t>();

    if (constInfo.sparsityMode == BSASparseMode::TopK) {
        Duplicate<int32_t>(maskLocal32, static_cast<int32_t>(1), curTileLenAlign);
    } else {
        Duplicate<int32_t>(maskLocal32, static_cast<int32_t>(0), curTileLenAlign);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<float> maskFloat = maskLocal32.template ReinterpretCast<float>();
    if (constInfo.sparsityMode == BSASparseMode::TopK) {
        Select<float>(maskFloat, cmpMaskTensor, maskFloat, 0.0f,
                      SELMODE::VSEL_TENSOR_SCALAR_MODE, curTileLenAlign);
    } else {
        Select<float>(maskFloat, cmpMaskTensor, maskFloat, 1.0f,
                      SELMODE::VSEL_TENSOR_SCALAR_MODE, curTileLenAlign);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<int16_t> maskLocal16 = maskLocal32.template ReinterpretCast<int16_t>();
    Cast<int16_t, int32_t>(maskLocal16, maskLocal32, RoundMode::CAST_NONE, curTileLenAlign);
    PipeBarrier<PIPE_V>();

    Cast<uint8_t, int16_t>(maskLocal8, maskLocal16, RoundMode::CAST_NONE, curTileLenAlign);
    PipeBarrier<PIPE_V>();

    DataCopyPad(maskTmpGmLocal[tileGmOffset], maskLocal8,
                DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(uint8_t)), 0, 0));

    LocalTensor<half> tempHalf = globalHistBuf_.Get<half>();
    uint64_t rsvdCnt = 0;
    GatherMask(tempHalf, tempHalf, cmpMaskTensor.ReinterpretCast<uint16_t>(),
               true, curTileLen, {1, 1, 0, 0}, rsvdCnt);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    curTileK -= static_cast<int32_t>(rsvdCnt);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyMaskToOutput(uint32_t batchIdx, uint32_t headIdx,
                                                                     GlobalTensor<OUT_T> &maskOutGm)
{
    uint64_t maskSizePerHead = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks;
    uint64_t outputOffset = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) * maskSizePerHead;

    uint64_t copyTileLen = BSAMin(tileLen_, maskSizePerHead);

    for (uint64_t offset = 0; offset < maskSizePerHead; offset += copyTileLen) {
        uint64_t curLen = BSAMin(copyTileLen, maskSizePerHead - offset);
        uint64_t alignLen = BSAAlignTo(curLen, static_cast<uint64_t>(VEC_ALIGN_SIZE));

        LocalTensor<uint8_t> maskLocal8 = maskLocalBuf_.Get<uint8_t>();

        DataCopyPad(maskLocal8, maskTmpGmLocal[offset],
                    DataCopyExtParams(1, static_cast<uint32_t>(curLen * sizeof(uint8_t)), 0, 0, 0),
                    DataCopyPadExtParams<uint8_t>(false, 0,
                        static_cast<uint8_t>(alignLen - curLen), 0));
        PipeBarrier<PIPE_V>();

        LocalTensor<OUT_T> maskLocalOut = maskLocal8.template ReinterpretCast<OUT_T>();
        DataCopyPad(maskOutGm[outputOffset + offset], maskLocalOut,
                    DataCopyParams(1, static_cast<uint16_t>(curLen * sizeof(OUT_T)), 0, 0));
    }
}

#endif // BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H
