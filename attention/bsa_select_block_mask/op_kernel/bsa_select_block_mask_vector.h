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
                                   GlobalTensor<half> &attnScoreGm, GlobalTensor<T> &softmaxTmpGm,
                                   GlobalTensor<uint8_t> &maskTmpGm,
                                   GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
                                   GlobalTensor<int64_t> &actualBlockLenQGm,
                                   GlobalTensor<int64_t> &actualBlockLenKVGm,
                                   GlobalTensor<OUT_T> &maskOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void ProcessPooling(uint32_t batchIdx, uint32_t headIdx,
                                          GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
                                          GlobalTensor<int64_t> &actualBlockLenQGm,
                                          GlobalTensor<int64_t> &actualBlockLenKVGm,
                                          GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm);
    __aicore__ inline void ProcessSoftmaxAndCast(uint32_t qBlockIdx);
    __aicore__ inline void ProcessMaskGen(uint32_t batchIdx, uint32_t headIdx);

private:
    __aicore__ inline void PoolingQ(uint32_t batchIdx, uint32_t headIdx,
                                     GlobalTensor<IN_T> &queryGm,
                                     GlobalTensor<int64_t> &actualBlockLenQGm,
                                     GlobalTensor<IN_T> &qCmpGm);
    __aicore__ inline void PoolingK(uint32_t batchIdx, uint32_t headIdx,
                                     GlobalTensor<IN_T> &keyGm,
                                     GlobalTensor<int64_t> &actualBlockLenKVGm,
                                     GlobalTensor<IN_T> &kCmpGm);
    __aicore__ inline void PoolingSingleBlock(GlobalTensor<IN_T> &srcGm, uint64_t srcOffset,
                                               uint32_t actualLen, uint32_t blockSize,
                                               GlobalTensor<IN_T> &dstGm, uint64_t dstOffset);
    __aicore__ inline void OnlineSoftmaxFirstPass(uint32_t qBlockIdx, T &mFinal, T &lFinal);
    __aicore__ inline void SimpleSoftmaxSecondPass(uint32_t qBlockIdx, T mFinal, T lFinal);
    __aicore__ inline void CastAttnScoreToFp16(uint32_t qBlockIdx);

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    GlobalTensor<IN_T> qCmpGmLocal, kCmpGmLocal;
    GlobalTensor<half> attnScoreGmLocal;
    GlobalTensor<T> softmaxTmpGmLocal;
    GlobalTensor<uint8_t> maskTmpGmLocal;
    GlobalTensor<IN_T> queryGmLocal, keyGmLocal;
    GlobalTensor<int64_t> actualBlockLenQGmLocal, actualBlockLenKVGmLocal;
    GlobalTensor<OUT_T> maskOutGmLocal;

    TBuf<> uBuf_;

    LocalTensor<IN_T> poolInputUb_;
    LocalTensor<T> poolAccumUb_;
    LocalTensor<IN_T> poolOutputUb_;
    LocalTensor<T> scoreUb_;
    LocalTensor<T> softmaxTmpUb_;
    LocalTensor<half> castOutputUb_;
    LocalTensor<uint8_t> maskUb_;
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

    poolAccumUb_ = uBuf_.GetWithOffset<T>(constInfo.dSize, ubOffset);
    ubOffset += constInfo.dSize * sizeof(T);

    poolOutputUb_ = uBuf_.GetWithOffset<IN_T>(constInfo.dSize, ubOffset);
    ubOffset += constInfo.dSize * sizeof(IN_T);

    uint32_t scoreSize = constInfo.yBlocks;
    scoreUb_ = uBuf_.GetWithOffset<T>(scoreSize, ubOffset);
    ubOffset += scoreSize * sizeof(T);

    softmaxTmpUb_ = uBuf_.GetWithOffset<T>(scoreSize, ubOffset);
    ubOffset += scoreSize * sizeof(T);

    castOutputUb_ = uBuf_.GetWithOffset<half>(scoreSize, ubOffset);
    ubOffset += scoreSize * sizeof(half);

    maskUb_ = uBuf_.GetWithOffset<uint8_t>(constInfo.yBlocks, ubOffset);
    ubOffset += constInfo.yBlocks;

    mScalarUb_ = uBuf_.GetWithOffset<T>(1, ubOffset);
    ubOffset += sizeof(T);

    lScalarUb_ = uBuf_.GetWithOffset<T>(1, ubOffset);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::InitGM(
    GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
    GlobalTensor<half> &attnScoreGm, GlobalTensor<T> &softmaxTmpGm,
    GlobalTensor<uint8_t> &maskTmpGm,
    GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenQGm, GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<OUT_T> &maskOutGm)
{
    this->qCmpGmLocal = qCmpGm;
    this->kCmpGmLocal = kCmpGm;
    this->attnScoreGmLocal = attnScoreGm;
    this->softmaxTmpGmLocal = softmaxTmpGm;
    this->maskTmpGmLocal = maskTmpGm;
    this->queryGmLocal = queryGm;
    this->keyGmLocal = keyGm;
    this->actualBlockLenQGmLocal = actualBlockLenQGm;
    this->actualBlockLenKVGmLocal = actualBlockLenKVGm;
    this->maskOutGmLocal = maskOutGm;
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
__aicore__ inline void BSAVectorService<BSAT>::ProcessPooling(
    uint32_t batchIdx, uint32_t headIdx,
    GlobalTensor<IN_T> &queryGm, GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenQGm, GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm)
{
    // TODO: 实现均值池化压缩
    // 1. PoolingQ: 对当前(batch,head)的query做分核池化 → qCmpGm
    // 2. PoolingK: 对当前(batch,head)的key做分核池化 → kCmpGm
    // 每个block: 搬入blockShape个token → Cast FP32 → ReduceSum → Divs → Cast回原类型 → 写出
    // actualBlockLen=0时填0向量
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingQ(
    uint32_t batchIdx, uint32_t headIdx,
    GlobalTensor<IN_T> &queryGm,
    GlobalTensor<int64_t> &actualBlockLenQGm,
    GlobalTensor<IN_T> &qCmpGm)
{
    // TODO: 对query做均值池化
    // for qBlock in 本核负责的qBlocks:
    //   actualLen = useActualBlockLenQ ? actualBlockLenQGm[batchIdx*xBlocks+qBlock] : blockShapeX
    //   PoolingSingleBlock(queryGm, srcOffset, actualLen, blockShapeX, qCmpGm, dstOffset)
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingK(
    uint32_t batchIdx, uint32_t headIdx,
    GlobalTensor<IN_T> &keyGm,
    GlobalTensor<int64_t> &actualBlockLenKVGm,
    GlobalTensor<IN_T> &kCmpGm)
{
    // TODO: 对key做均值池化（同PoolingQ逻辑）
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::PoolingSingleBlock(
    GlobalTensor<IN_T> &srcGm, uint64_t srcOffset,
    uint32_t actualLen, uint32_t blockSize,
    GlobalTensor<IN_T> &dstGm, uint64_t dstOffset)
{
    // TODO: 单个block的均值池化
    // if actualLen == 0: Duplicate 0向量 → 写出
    // else:
    //   1. DataCopyPad: GM→UB 搬入 actualLen×D 数据
    //   2. Cast: IN_T → FP32
    //   3. ReduceSum: 沿token维度求和
    //   4. Divs: 除以 actualLen
    //   5. Cast: FP32 → IN_T
    //   6. DataCopyPad: UB→GM 写出 D 个元素
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::ProcessSoftmaxAndCast(uint32_t qBlockIdx)
{
    // TODO: Online Softmax 两遍循环 + Cast FP32→FP16
    // 第一遍: OnlineSoftmaxFirstPass → 得到 mFinal, lFinal
    // 第二遍: SimpleSoftmaxSecondPass → 归一化
    // Cast: CastAttnScoreToFp16 → 写入workspace
    T mFinal = 0.0f;
    T lFinal = 0.0f;
    OnlineSoftmaxFirstPass(qBlockIdx, mFinal, lFinal);
    SimpleSoftmaxSecondPass(qBlockIdx, mFinal, lFinal);
    CastAttnScoreToFp16(qBlockIdx);
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::OnlineSoftmaxFirstPass(uint32_t qBlockIdx,
                                                                       T &mFinal, T &lFinal)
{
    // TODO: Online Softmax 第一遍
    // m = -inf, l = 0
    // for kIter in kBlock迭代:
    //   搬入 score_partial (从workspace)
    //   m_new = max(m, max(score_partial))
    //   l_new = l * exp(m - m_new) + Σexp(score_partial - m_new)
    //   将原始score_partial写回workspace
    //   m = m_new, l = l_new
    // mFinal = m, lFinal = l
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::SimpleSoftmaxSecondPass(uint32_t qBlockIdx,
                                                                        T mFinal, T lFinal)
{
    // TODO: SimpleSoftmax 第二遍归一化
    // for each stored score chunk:
    //   attn_score = exp(score - m_final) / l_final
    //   写回workspace (FP32)
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::CastAttnScoreToFp16(uint32_t qBlockIdx)
{
    // TODO: Cast attn_score FP32 → FP16 写入workspace
    // 逐chunk读FP32 score → Cast → 写FP16
}

template <typename BSAT>
__aicore__ inline void BSAVectorService<BSAT>::ProcessMaskGen(uint32_t batchIdx, uint32_t headIdx)
{
    // TODO: 生成 BlockSparseMask
    // if sparsityMode == TopK:
    //   初始化mask全0 → topK位置写1
    // else (BottomK):
    //   初始化mask全1 → bottomK位置写0
    // DataCopyPad: mask workspace → 输出GM
}

#endif // BSA_SELECT_BLOCK_MASK_VECTOR_H
