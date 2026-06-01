#ifndef BSA_SELECT_BLOCK_MASK_VECTOR_H
#define BSA_SELECT_BLOCK_MASK_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"

template <typename BSAT>
class BSAVectorService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;

    __aicore__ inline BSAVectorService() {};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGM(GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
                                   GlobalTensor<IN_T> &attnScoreGm, GlobalTensor<T> &softmaxTmpGm,
                                   GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
                                   GlobalTensor<int64_t> &actualBlockLenQGm,
                                   GlobalTensor<int64_t> &actualBlockLenKVGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void PoolingSingleQBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t qBlockIdx,
                                                GlobalTensor<IN_T> &queryGm,
                                                GlobalTensor<int64_t> &actualBlockLenQGm,
                                                GlobalTensor<IN_T> &qCmpGm);
    __aicore__ inline void PoolingSingleKBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t kBlockIdx,
                                                GlobalTensor<IN_T> &keyGm,
                                                GlobalTensor<int64_t> &actualBlockLenKVGm,
                                                GlobalTensor<IN_T> &kCmpGm);
    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                        uint32_t kChunkStart, uint32_t kChunkSize,
                                                        T &mFinal, T &lFinal);
    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qBlockIdx, T mFinal, T lFinal);

private:
    __aicore__ inline void PoolingSingleBlockImpl(GlobalTensor<IN_T> &srcGm, uint64_t srcOffset,
                                                   uint32_t actualLen, uint32_t blockSize,
                                                   GlobalTensor<IN_T> &dstGm, uint64_t dstOffset);

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> qCmpGmLocal, kCmpGmLocal;
    GlobalTensor<IN_T> attnScoreGmLocal;
    GlobalTensor<T> softmaxTmpGmLocal;
    GlobalTensor<IN_T> queryGmLocal, keyGmLocal;
    GlobalTensor<int64_t> actualBlockLenQGmLocal, actualBlockLenKVGmLocal;

    TBuf<> uBuf_;

    LocalTensor<IN_T> poolInputUb_;
    LocalTensor<T> poolAccumUb_;
    LocalTensor<IN_T> poolOutputUb_;
    LocalTensor<T> scoreUb_;
    LocalTensor<T> softmaxTmpUb_;
    LocalTensor<T> castOutputUb_;
    LocalTensor<T> mScalarUb_;
    LocalTensor<T> lScalarUb_;

    event_t eventIdMte2ToV;
    event_t eventIdVToMte3;
    event_t eventIdMte3ToV;
};

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitParams(const BSAConstInfo &constInfo,
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(uBuf_, BSAConstInfo::BUFFER_SIZE_BYTE_192K);

    uint32_t ubOffset = 0;
    uint32_t poolInputSize = constInfo.blockShapeX * constInfo.dSize;
    poolInputUb_ = uBuf_.GetWithOffset<IN_T>(poolInputSize, ubOffset);
    ubOffset += poolInputSize * sizeof(IN_T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    poolAccumUb_ = uBuf_.GetWithOffset<T>(constInfo.dSize, ubOffset);
    ubOffset += constInfo.dSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    poolOutputUb_ = uBuf_.GetWithOffset<IN_T>(constInfo.dSize, ubOffset);
    ubOffset += constInfo.dSize * sizeof(IN_T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t scoreSize = constInfo.yBlocks;
    scoreUb_ = uBuf_.GetWithOffset<T>(scoreSize, ubOffset);
    ubOffset += scoreSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    softmaxTmpUb_ = uBuf_.GetWithOffset<T>(scoreSize, ubOffset);
    ubOffset += scoreSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    castOutputUb_ = uBuf_.GetWithOffset<T>(scoreSize, ubOffset);
    ubOffset += scoreSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    mScalarUb_ = uBuf_.GetWithOffset<T>(1, ubOffset);
    ubOffset += sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    lScalarUb_ = uBuf_.GetWithOffset<T>(1, ubOffset);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitGM(
    GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
    GlobalTensor<IN_T> &attnScoreGm, GlobalTensor<T> &softmaxTmpGm,
    GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenQGm, GlobalTensor<int64_t> &actualBlockLenKVGm)
{
    this->qCmpGmLocal = qCmpGm;
    this->kCmpGmLocal = kCmpGm;
    this->attnScoreGmLocal = attnScoreGm;
    this->softmaxTmpGmLocal = softmaxTmpGm;
    this->queryGmLocal = queryGm;
    this->keyGmLocal = keyGm;
    this->actualBlockLenQGmLocal = actualBlockLenQGm;
    this->actualBlockLenKVGmLocal = actualBlockLenKVGm;
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::AllocEventID()
{
    eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
    eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleQBlock(
    uint32_t batchIdx, uint32_t headIdx, uint32_t qBlockIdx,
    GlobalTensor<IN_T> &queryGm,
    GlobalTensor<int64_t> &actualBlockLenQGm,
    GlobalTensor<IN_T> &qCmpGm)
{
    // TODO: 单个query block的均值池化
    // 1. 计算 srcOffset: 根据 layout(BNSD/TND) + batchIdx + headIdx + qBlockIdx + blockShapeX
    // 2. 获取 actualLen:
    //    useActualBlockLenQ ? actualBlockLenQGm[batchIdx*xBlocks+qBlockIdx] : blockShapeX
    //    末尾不完整块: min(actualLen, maxQSeqlen - qBlockIdx*blockShapeX)
    // 3. 计算 dstOffset: qBlockIdx * dSize
    // 4. PoolingSingleBlockImpl(queryGm, srcOffset, actualLen, blockShapeX, qCmpGm, dstOffset)
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleKBlock(
    uint32_t batchIdx, uint32_t headIdx, uint32_t kBlockIdx,
    GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<IN_T> &kCmpGm)
{
    // TODO: 单个key block的均值池化（同PoolingSingleQBlock逻辑）
    // 1. 计算 srcOffset: 根据 layout + batchIdx + headIdx + kBlockIdx + blockShapeY
    // 2. 获取 actualLen:
    //    useActualBlockLenK ? actualBlockLenKVGm[batchIdx*yBlocks+kBlockIdx] : blockShapeY
    // 3. 计算 dstOffset: kBlockIdx * dSize
    // 4. PoolingSingleBlockImpl(keyGm, srcOffset, actualLen, blockShapeY, kCmpGm, dstOffset)
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleBlockImpl(
    GlobalTensor<IN_T> &srcGm, uint64_t srcOffset,
    uint32_t actualLen, uint32_t blockSize,
    GlobalTensor<IN_T> &dstGm, uint64_t dstOffset)
{
    // TODO: 单个block的均值池化实现
    // if actualLen == 0:
    //   Duplicate(poolOutputUb_, 0, dSize) → DataCopyPad写出0向量
    // else:
    //   1. DataCopyPad: GM→UB 搬入 actualLen×D 数据到 poolInputUb_
    //   2. Cast: poolInputUb_(IN_T) → poolAccumUb_(FP32)
    //   3. ReduceSum: 沿token维度(actualLen)求和 → poolAccumUb_
    //   4. Divs: poolAccumUb_ 除以 actualLen
    //   5. Cast: poolAccumUb_(FP32) → poolOutputUb_(IN_T)
    //   6. DataCopyPad: UB→GM 写出 D 个元素到 dstGm[dstOffset]
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::OnlineSoftmaxFirstPassChunk(
    uint32_t qChunkStart, uint32_t qChunkSize,
    uint32_t kChunkStart, uint32_t kChunkSize,
    T &mFinal, T &lFinal)
{
    // TODO: Online Softmax 第一遍（对一个kChunk的score块）
    // 输入: score[qChunkStart:qChunkStart+qChunkSize, kChunkStart:kChunkStart+kChunkSize]
    //       存储在 softmaxTmpGm (FP32)
    // 对每个 qBlock 行维护 m(标量) 和 l(标量):
    //   搬入 score_partial (qChunkSize × kChunkSize 个 FP32)
    //   m_new = max(m, max(score_partial))
    //   l_new = l * exp(m - m_new) + Σexp(score_partial - m_new)
    //   将原始score_partial写回 softmaxTmpGm
    //   m = m_new, l = l_new
    // 输出: mFinal, lFinal 通过引用返回（跨kChunk累积）
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::SoftmaxSecondPassAndCast(
    uint32_t qBlockIdx, T mFinal, T lFinal)
{
    // TODO: SimpleSoftmax 第二遍归一化 + Cast FP32→FP16
    // 对 qBlockIdx 行的所有 Yblocks 个 score:
    //   逐chunk读 softmaxTmpGm 中的 FP32 score
    //   attn_score = exp(score - mFinal) / lFinal
    //   Cast FP32 → FP16
    //   写入 attnScoreGm[qBlockIdx * yBlocks :] (FP16)
}

#endif // BSA_SELECT_BLOCK_MASK_VECTOR_H
