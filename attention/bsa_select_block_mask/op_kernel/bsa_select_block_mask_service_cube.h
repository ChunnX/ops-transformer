#ifndef BSA_SELECT_BLOCK_MASK_SERVICE_CUBE_H
#define BSA_SELECT_BLOCK_MASK_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "bsa_select_block_mask_common.h"

template <typename BSAT> class BSAMatmulService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using MM_OUT_T = T;

    __aicore__ inline BSAMatmulService() {};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitGM(GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
                                   GlobalTensor<MM_OUT_T> &attnScoreGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void ComputeMatmulChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                               uint32_t kChunkStart, uint32_t kChunkSize);
    __aicore__ inline void EmitFixpipeProbe(uint32_t kChunkFlagIdx);
    __aicore__ inline void FreeEventID();

private:
    __aicore__ inline void CopyInQToL1(uint32_t qBlockIdx);
    __aicore__ inline void CopyInKToL1(uint32_t kBlockStart, uint32_t kBlockLen, uint8_t pingpongFlag);
    __aicore__ inline void MmadInner(uint32_t qBlockIdx, uint32_t kBlockStart, uint32_t kBlockLen,
                                      uint8_t l0abPingPongFlag, uint8_t l0cPingPongFlag);
    __aicore__ inline void FixCopyOut(uint32_t qBlockIdx, uint32_t kBlockStart, uint32_t kBlockLen,
                                       uint8_t l0cPingPongFlag);

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    BSAConstInfo constInfo{};

    GlobalTensor<IN_T> qCmpGmLocal;
    GlobalTensor<IN_T> kCmpGmLocal;
    GlobalTensor<MM_OUT_T> scoreGmLocal;

    TBuf<TPosition::A1> qBuf;
    TBuf<TPosition::B1> kBuf[2];
    TBuf<TPosition::A2> tmpBufL0A[2];
    TBuf<TPosition::B2> tmpBufL0B[2];
    TBuf<TPosition::CO1> tmpBufL0C[2];

    LocalTensor<IN_T> l1QTensor;
    LocalTensor<IN_T> l1KTensor[2];
    LocalTensor<IN_T> aL0TensorPingPong[2];
    LocalTensor<IN_T> bL0TensorPingPong[2];
    LocalTensor<MM_OUT_T> cL0TensorPingPong[2];
};

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::InitParams(const BSAConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(qBuf, CUBE_BASE_BLOCK * constInfo.dSize * sizeof(IN_T));
    pipe->InitBuffer(kBuf[0], CUBE_BASE_BLOCK * constInfo.dSize * sizeof(IN_T));
    pipe->InitBuffer(kBuf[1], CUBE_BASE_BLOCK * constInfo.dSize * sizeof(IN_T));

    l1QTensor = qBuf.Get<IN_T>();
    l1KTensor[0] = kBuf[0].Get<IN_T>();
    l1KTensor[1] = kBuf[1].Get<IN_T>();

    pipe->InitBuffer(tmpBufL0A[0], L0A_PP_SIZE);
    pipe->InitBuffer(tmpBufL0A[1], L0A_PP_SIZE);
    pipe->InitBuffer(tmpBufL0B[0], L0B_PP_SIZE);
    pipe->InitBuffer(tmpBufL0B[1], L0B_PP_SIZE);
    pipe->InitBuffer(tmpBufL0C[0], L0C_PP_SIZE);
    pipe->InitBuffer(tmpBufL0C[1], L0C_PP_SIZE);

    aL0TensorPingPong[0] = tmpBufL0A[0].Get<IN_T>();
    aL0TensorPingPong[1] = tmpBufL0A[1].Get<IN_T>();
    bL0TensorPingPong[0] = tmpBufL0B[0].Get<IN_T>();
    bL0TensorPingPong[1] = tmpBufL0B[1].Get<IN_T>();
    cL0TensorPingPong[0] = tmpBufL0C[0].Get<MM_OUT_T>();
    cL0TensorPingPong[1] = tmpBufL0C[1].Get<MM_OUT_T>();
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::InitGM(GlobalTensor<IN_T> &qCmpGm,
                                                        GlobalTensor<IN_T> &kCmpGm,
                                                        GlobalTensor<MM_OUT_T> &attnScoreGm)
{
    this->qCmpGmLocal = qCmpGm;
    this->kCmpGmLocal = kCmpGm;
    this->scoreGmLocal = attnScoreGm;
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::AllocEventID()
{
    // TODO: 分配 MTE1<->MTE2, M<->MTE1, FIX<->M 事件
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::FreeEventID()
{
    // TODO: 等待并释放所有事件
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::ComputeMatmulChunk(
    uint32_t qChunkStart, uint32_t qChunkSize,
    uint32_t kChunkStart, uint32_t kChunkSize)
{
    // TODO: 对 qChunkStart 开始的 qChunkSize 行，与 kChunkStart 开始的 kChunkSize 列做 Matmul
    // 1. for qBlockIdx in [qChunkStart, qChunkStart+qChunkSize):
    //      CopyInQToL1(qBlockIdx)
    // 2. CopyInKToL1(kChunkStart, kChunkSize, pingpongFlag)
    // 3. MmadInner + FixCopyOut
    // score[qChunkStart:qChunkStart+qChunkSize, kChunkStart:kChunkStart+kChunkSize]
    //   = q_cmp[qChunkStart:] × k_cmp[kChunkStart:].T
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::EmitFixpipeProbe(uint32_t kChunkFlagIdx)
{
    uint64_t gmOffset = (static_cast<uint64_t>(constInfo.aicIdx) * 2 + (kChunkFlagIdx & 1)) *
                        BSA_FIXPIPE_PROBE_ELEMS;

    FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = BSA_FIXPIPE_PROBE_N;
    fixpipeParams.mSize = BSA_FIXPIPE_PROBE_M;
    fixpipeParams.srcStride = BSA_FIXPIPE_PROBE_M;
    fixpipeParams.dstStride = BSA_FIXPIPE_PROBE_M;
    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 0;
    fixpipeParams.dstNdStride = 0;
    fixpipeParams.reluEn = false;

    Fixpipe<MM_OUT_T, MM_OUT_T>(scoreGmLocal[gmOffset],
                                cL0TensorPingPong[kChunkFlagIdx & 1],
                                fixpipeParams);
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::CopyInQToL1(uint32_t qBlockIdx)
{
    // TODO: GM -> L1 ND2NZ 搬运 q_cmp[qBlockIdx, :]
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::CopyInKToL1(uint32_t kBlockStart, uint32_t kBlockLen,
                                                             uint8_t pingpongFlag)
{
    // TODO: GM -> L1 ND2NZ 搬运 k_cmp[kBlockStart:kBlockStart+kBlockLen, :]
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::MmadInner(uint32_t qBlockIdx, uint32_t kBlockStart,
                                                           uint32_t kBlockLen,
                                                           uint8_t l0abPingPongFlag, uint8_t l0cPingPongFlag)
{
    // TODO: L1->L0A/L0B LoadData + Mmad
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::FixCopyOut(uint32_t qBlockIdx, uint32_t kBlockStart,
                                                            uint32_t kBlockLen, uint8_t l0cPingPongFlag)
{
    // TODO: L0C -> GM Fixpipe 输出 score 到 workspace
}

#endif // BSA_SELECT_BLOCK_MASK_SERVICE_CUBE_H
