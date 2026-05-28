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
    __aicore__ inline void InitGM(GlobalTensor<half> &attnScoreGm,
                                   GlobalTensor<int32_t> &topkWorkspaceGm,
                                   GlobalTensor<uint8_t> &maskTmpGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void ProcessRadixTopK(uint32_t batchIdx, uint32_t headIdx);

private:
    __aicore__ inline void TwiddleInB16(LocalTensor<int16_t> &xLocal, uint64_t dataLen);
    __aicore__ inline void DoAndMask(LocalTensor<int16_t> &xLocal, uint64_t dataLen);
    __aicore__ inline void CalcCumsumHistogram(LocalTensor<int16_t> &xLocal, int32_t roundId,
                                                int32_t tileId, uint64_t curTileLen);
    __aicore__ inline void CopyOutHistToWs();
    __aicore__ inline bool Update(int32_t roundId);
    __aicore__ inline void ClearHist(int32_t roundId);
    __aicore__ inline void TopKPerTile(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void NegateAttnScore();

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<half> attnScoreGmLocal;
    GlobalTensor<int32_t> topkWorkspaceGmLocal;
    GlobalTensor<uint8_t> maskTmpGmLocal;

    GlobalTensor<int32_t> globalHistGm_;
    GlobalTensor<int32_t> boundaryBinCumSumGm_;
    GlobalTensor<int32_t> coreTopKGm_;

    TBuf<TPosition::VECIN> xBuf0_;
    TBuf<TPosition::VECIN> xBuf1_;
    TBuf<TPosition::VECOUT> outValueBuf_;
    TBuf<TPosition::VECOUT> outIndexBuf_;
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

    // TODO: 根据 UB 容量计算 tileLen 和 tileNum
    this->tileLen_ = 7168;
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
    pipe->InitBuffer(outValueBuf_, tileLen_ * sizeof(float));
    pipe->InitBuffer(outIndexBuf_, tileLen_ * sizeof(int32_t));
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::InitGM(
    GlobalTensor<half> &attnScoreGm,
    GlobalTensor<int32_t> &topkWorkspaceGm,
    GlobalTensor<uint8_t> &maskTmpGm)
{
    this->attnScoreGmLocal = attnScoreGm;
    this->topkWorkspaceGmLocal = topkWorkspaceGm;
    this->maskTmpGmLocal = maskTmpGm;

    // workspace 复用 topkWorkspaceGm 空间
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
    // eventBuf0_ 是 FetchEventID，不需要 Release
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ProcessRadixTopK(uint32_t batchIdx, uint32_t headIdx)
{
    // TODO: 完整 Radix TopK 流程
    // 1. 如果 BottomK 模式: NegateAttnScore()
    // 2. 初始化状态: involvedMask16_=0, andMask16_=0xC000, remainK_=kValue_
    // 3. Phase 1: Radix 筛选 (for roundId = round_-1 → 0)
    //    a. CopyOutHistToWs<true> (清零+保存上轮信息)
    //    b. ClearHist(roundId)
    //    c. for each tile: CopyIn → TwiddleInB16 → DoAndMask → CalcCumsumHistogram
    //    d. CopyOutHistToWs<false>
    //    e. SyncAll()
    //    f. Update(roundId) → 如果 earlyExit 则 break
    //    g. andMask16_ >>= 2
    // 4. SyncAll()
    // 5. Phase 2: TopKPerTile → 收集结果写入mask

    involvedMask16_ = 0;
    andMask16_ = static_cast<int16_t>(0xC000);
    totalDefinitelyInTopK_ = 0;
    boundaryBin = 0;
    boundaryBinPrev = 1;
    remainK_ = kValue_;

    if (constInfo.sparsityMode == BSASparseMode::BottomK) {
        NegateAttnScore();
        SyncAll();
    }

    // TODO: Phase 1 + Phase 2 完整实现
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::NegateAttnScore()
{
    // TODO: 对 attn_score 取反 (FP16 翻转符号位)
    // 逐tile: CopyIn → Muls(-1) → CopyOut
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TwiddleInB16(LocalTensor<int16_t> &xLocal,
                                                                  uint64_t dataLen)
{
    // TODO: FP16 Twiddle 变换
    // 1. ShiftRight 提取符号位
    // 2. Or 构造 flipMask
    // 3. AND/OR/NOT 组合实现 XOR
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::DoAndMask(LocalTensor<int16_t> &xLocal,
                                                               uint64_t dataLen)
{
    // TODO: 用 andMask16_ 做位与提取当前 2bit
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CalcCumsumHistogram(LocalTensor<int16_t> &xLocal,
    int32_t roundId, int32_t tileId, uint64_t curTileLen)
{
    // TODO: 累积直方图统计
    // for binMask = 3 → 1:
    //   involvedMaskTemp = involvedMask16_ | (binMask << (roundId * 2))
    //   Compares + GatherMask 统计匹配数
    //   cumSumOne += rsvdCnt
    //   tileHist[binMask][tileId] = cumSumOne
    //   globalHist[binMask] += cumSumOne
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::CopyOutHistToWs()
{
    // TODO: 将 globalHist 写入 GM workspace
}

template <typename BSAT>
__aicore__ inline bool BSARadixTopKService<BSAT>::Update(int32_t roundId)
{
    // TODO: 跨核聚合直方图 + 更新边界
    // 1. 从 GM 读取所有核的 globalHist
    // 2. 向量树形归约
    // 3. 找 boundaryBin
    // 4. 更新 involvedMask16_
    // 5. 更新 tileTopK
    // 6. 最后一轮精确分配
    // return (remainK_ == 0 || roundId == 0)
    return false;
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::ClearHist(int32_t roundId)
{
    // TODO: 初始化/重置直方图
    // 第一轮: tileHist[bin0] = tileLen, globalHist[0] = 总元素数
    // 后续轮: tileHist[bin0] = 上轮boundaryBin内元素数
}

template <typename BSAT>
__aicore__ inline void BSARadixTopKService<BSAT>::TopKPerTile(uint32_t batchIdx, uint32_t headIdx)
{
    // TODO: 按边界值收集 TopK 结果并写入 mask
    // for each tile:
    //   Compares(GT) + GatherMask → 收集 > 边界值的位置 → mask写1
    //   Compares(EQ) + GatherMask → 收集 = 边界值的位置(按配额) → mask写1
}

#endif // BSA_SELECT_BLOCK_MASK_RADIX_TOPK_H
