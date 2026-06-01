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
    __aicore__ inline void TwiddleInB16(LocalTensor<int16_t> &xLocal, uint64_t dataLen);
    __aicore__ inline void DoAndMask(LocalTensor<int16_t> &xLocal, uint64_t dataLen);
    __aicore__ inline void CalcCumsumHistogram(LocalTensor<int16_t> &xLocal, int32_t roundId,
                                                int32_t tileId, uint64_t curTileLen);
    __aicore__ inline void CopyOutHistToWs();
    __aicore__ inline bool Update(int32_t roundId);
    __aicore__ inline void ClearHist(int32_t roundId);
    __aicore__ inline void TopKPerTile(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void SubTopKAndWriteMask(LocalTensor<half> &xLocalHalf, half boundaryHalf,
                                                uint64_t curTileLen, uint64_t tileGmOffset,
                                                int32_t &curTileK);

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
    if (GetBlockIdx() == 0) {
        AscendC::printf("STEP6.2: ProcessRadixTopKAndWriteMask start, sparsityMode=%d\n",
                        static_cast<int32_t>(constInfo.sparsityMode));
    }

    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        NegateAttnScore();
    }

    if (GetBlockIdx() == 0) {
        AscendC::printf("STEP6.2: NegateAttnScore done (or skipped for TopK)\n");
    }
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

        if (GetBlockIdx() == 0 && tileIdx == 0) {
            AscendC::printf("STEP6.2: NegateAttnScore before Muls, tileLen=%lu\n", curTileLen);
        }

        if constexpr (IsSameType<IN_T, bfloat16_t>::value) {
            Muls(xLocal.template ReinterpretCast<half>(),
                 xLocal.template ReinterpretCast<half>(), (half)-1, alignLen);
        } else {
            Muls(xLocal, xLocal, (IN_T)-1, alignLen);
        }
        PipeBarrier<PIPE_V>();

        DataCopyPad(attnScoreGmLocal[tileGmOffset], xLocal,
                    DataCopyParams(1, static_cast<uint16_t>(curTileLen * sizeof(IN_T)), 0, 0));

        if (GetBlockIdx() == 0 && tileIdx == 0) {
            AscendC::printf("STEP6.2: NegateAttnScore after writeback\n");
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TwiddleInB16(LocalTensor<int16_t> &xLocal, uint64_t dataLen)
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::DoAndMask(LocalTensor<int16_t> &xLocal, uint64_t dataLen)
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CalcCumsumHistogram(
    LocalTensor<int16_t> &xLocal, int32_t roundId, int32_t tileId, uint64_t curTileLen)
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutHistToWs()
{
}

template <typename BSAT>
__aicore__ inline bool BSARadixTopKService<BSAT>::Update(int32_t roundId)
{
    return true;
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearHist(int32_t roundId)
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TopKPerTile(uint32_t batchIdx, uint32_t headIdx)
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::SubTopKAndWriteMask(
    LocalTensor<half> &xLocalHalf, half boundaryHalf,
    uint64_t curTileLen, uint64_t tileGmOffset, int32_t &curTileK)
{
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyMaskToOutput(uint32_t batchIdx, uint32_t headIdx,
                                                                     GlobalTensor<OUT_T> &maskOutGm)
{
    AscendC::printf("STEP6.1: CopyMaskToOutput baseline (empty)\n");
}

#endif // BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H