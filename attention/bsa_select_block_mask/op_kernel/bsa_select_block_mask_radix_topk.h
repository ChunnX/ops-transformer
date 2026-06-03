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
    __aicore__ inline void NegateAttnScore();

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> attnScoreGmLocal;
    GlobalTensor<int32_t> topkWorkspaceGmLocal;
    GlobalTensor<uint8_t> maskTmpGmLocal;

    GlobalTensor<int32_t> globalHistGm_;
    GlobalTensor<int32_t> boundaryBinCumSumGm_;
    GlobalTensor<int32_t> coreTopKGm_;
    GlobalTensor<int32_t> tileTopKGm_;
    GlobalTensor<int32_t> tileHistGm_;

    TBuf<TPosition::VECIN> xBufPing_;
    TBuf<TPosition::VECIN> xBufPong_;
    TBuf<TPosition::VECOUT> maskLocalBuf_;
    TBuf<TPosition::VECOUT> globalHistBuf_;
    TBuf<TPosition::VECCALC> tileHistAndTopKBuf_;
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

    this->tileLen_ = BSAMin(static_cast<uint64_t>(7168), sortLen_);
    this->totalTileNum_ = (sortLen_ + tileLen_ - 1) / tileLen_;
    this->tileNum_ = totalTileNum_;
    this->tileNumAlign_ = BSACeilDiv(tileNum_, static_cast<uint64_t>(8)) * 8;

    this->formerTileLen_ = tileLen_;
    this->tailTileLen_ = sortLen_ - (totalTileNum_ - 1) * tileLen_;
    if (tailTileLen_ == 0) {
        tailTileLen_ = tileLen_;
    }

    uint32_t coreNum = constInfo.coreNum;
    uint64_t baseTilesPerCore = totalTileNum_ / coreNum;
    uint64_t extraCores = totalTileNum_ % coreNum;
    this->formerCoreNum_ = extraCores;
    this->tailCoreNum_ = coreNum - extraCores;
    this->formerTileNum_ = baseTilesPerCore + (extraCores > 0 ? 1 : 0);
    this->tailTileNum_ = baseTilesPerCore;

    uint64_t myTileNum = (GetBlockIdx() < formerCoreNum_) ? formerTileNum_ : tailTileNum_;
    this->tileNum_ = myTileNum;
    this->tileNumAlign_ = BSACeilDiv(myTileNum, static_cast<uint64_t>(8)) * 8;

    this->tileNumRepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB);
    this->tileNumRemain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB);
    this->tileNumBy2RepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB_BY2);
    this->tileNumBy2Remain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB_BY2);
    this->tileNumBy3RepeatTimes_ = static_cast<uint32_t>(myTileNum / MAX_TILE_NUM_IN_UB_BY3);
    this->tileNumBy3Remain_ = static_cast<uint32_t>(myTileNum % MAX_TILE_NUM_IN_UB_BY3);
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(xBufPing_, tileLen_ * sizeof(float));
    pipe->InitBuffer(xBufPong_, tileLen_ * sizeof(float));
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
    GlobalTensor<uint8_t> &maskTmpGm)
{
    this->attnScoreGmLocal = attnScoreGm;
    this->topkWorkspaceGmLocal = topkWorkspaceGm;
    this->maskTmpGmLocal = maskTmpGm;

    uint32_t coreNum = constInfo.coreNum;
    uint32_t headerSize = (numValue_ * coreNum + coreNum + coreNum);
    this->globalHistGm_.SetGlobalBuffer((__gm__ int32_t *)topkWorkspaceGm.GetPhyAddr());
    this->boundaryBinCumSumGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + numValue_ * coreNum * sizeof(int32_t)));
    this->coreTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + (numValue_ * coreNum + coreNum) * sizeof(int32_t)));

    uint64_t tileTopKOffset = 0;
    if (GetBlockIdx() < formerCoreNum_) {
        tileTopKOffset = GetBlockIdx() * formerTileNum_;
    } else {
        tileTopKOffset = formerCoreNum_ * formerTileNum_ + (GetBlockIdx() - formerCoreNum_) * tailTileNum_;
    }
    uint64_t tileHistOffset = numValue_ * tileTopKOffset;

    this->tileTopKGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + headerSize * sizeof(int32_t) + tileTopKOffset * sizeof(int32_t)));
    this->tileHistGm_.SetGlobalBuffer(
        (__gm__ int32_t *)(topkWorkspaceGm.GetPhyAddr() + headerSize * sizeof(int32_t) + totalTileNum_ * sizeof(int32_t) + tileHistOffset * sizeof(int32_t)));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::AllocEventID()
{
    eventIDMTE2ToVForX_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::FreeEventID()
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ProcessRadixTopKAndWriteMask(uint32_t batchIdx, uint32_t headIdx)
{
    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        NegateAttnScore();
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::NegateAttnScore()
{
    for (uint64_t tileIdx = 0; tileIdx < tileNum_; tileIdx++) {
        uint64_t curTileLen = BSAMin(tileLen_, sortLen_ - tileIdx * tileLen_);
        uint64_t tileGmOffset = tileIdx * tileLen_;

        LocalTensor<IN_T> xLocal = xBufPing_.Get<IN_T>();
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
__aicore__ inline void BSARadixTopKService<BSAT>::CopyMaskToOutput(uint32_t batchIdx, uint32_t headIdx,
                                                                     GlobalTensor<OUT_T> &maskOutGm)
{
}

#endif // BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H