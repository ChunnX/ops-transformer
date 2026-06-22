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
    using T = float;                        ///< 中间计算类型 (float)，统一 FP32
    using IN_T = typename BSAT::inputT;       ///< Cube A/B 输入类型
    using MM_OUT_T = T;                     ///< Matmul 输出类型 = float

    __aicore__ inline BSAMatmulService() {};

    __aicore__ inline void InitParams(const BSAConstInfo &constInfo);

    __aicore__ inline void InitBuffers(TPipe *pipe);

    __aicore__ inline void InitGM(GlobalTensor<IN_T> &qCmpGm, GlobalTensor<IN_T> &kCmpGm,
                                   GlobalTensor<MM_OUT_T> &attnScoreGm);

    __aicore__ inline void AllocEventID();

    __aicore__ inline void ComputeMatmulChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                               uint32_t kChunkStart, uint32_t kChunkSize);

    __aicore__ inline void FreeEventID();

private:
    __aicore__ inline void CopyGmToL1(LocalTensor<IN_T> &l1Tensor, GlobalTensor<IN_T> &gmSrcTensor,
                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride);

    __aicore__ inline void CopyInQToL1(uint32_t qChunkStart, uint32_t qChunkSize);

    __aicore__ inline void CopyInKToL1(uint32_t kBlockStart, uint32_t kCubeBlockLen, uint8_t pingpongFlag);

    __aicore__ inline void LoadDataMmA(LocalTensor<IN_T> &aL0Tensor, LocalTensor<IN_T> &aL1Tensor,
                                        uint32_t mSize, uint32_t kSize);

    __aicore__ inline void LoadDataMmB(LocalTensor<IN_T> &bL0Tensor, LocalTensor<IN_T> &bL1Tensor,
                                        uint32_t kSize, uint32_t nSize);

    __aicore__ inline void MmadInner(uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                      bool isL0CInit, uint8_t l0PingPongFlag);

    __aicore__ inline void FixCopyOut(GlobalTensor<MM_OUT_T> &resGm, uint32_t mSize, uint32_t nSize,
                                       uint32_t dstStride, uint8_t l0PingPongFlag);

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    static constexpr uint32_t SYNC_MTE21_FLAG_Q = EVENT_ID0;
    static constexpr uint32_t SYNC_MTE21_FLAG_K[2] = {EVENT_ID1, EVENT_ID2};
    static constexpr uint32_t SYNC_MTE1MM_FLAG[2] = {EVENT_ID3, EVENT_ID4};
    static constexpr uint32_t SYNC_MMFIX_FLAG[2] = {EVENT_ID5, EVENT_ID6};

    BSAConstInfo constInfo{};  ///< 运行时常量缓存 (dSize, yBlocks 等)

    GlobalTensor<IN_T> qCmpGmLocal;        ///< Query 压缩矩阵 [xBlocks, dSize] (float)
    GlobalTensor<IN_T> kCmpGmLocal;        ///< Key 压缩矩阵 [yBlocks, dSize] (float)
    GlobalTensor<MM_OUT_T> scoreGmLocal;///< 输出注意力分数 [xBlocks, yBlocks] (float)
    GlobalTensor<T> scoreFp32OutGm;

    // ---- L1 缓冲区 ----
    TBuf<TPosition::A1> qBuf;           ///< Q L1 单缓冲 [CUBE_BASE_BLOCK * dSize]
    TBuf<TPosition::B1> kBuf[2];        ///< K L1 双缓冲 [2][CUBE_BASE_BLOCK * dSize]

    // ---- L0 缓冲区 ----
    TBuf<TPosition::A2> tmpBufL0A[2];   ///< L0A 双缓冲 (Ping-Pong)
    TBuf<TPosition::B2> tmpBufL0B[2];   ///< L0B 双缓冲 (Ping-Pong)
    TBuf<TPosition::CO1> tmpBufL0C[2];  ///< L0C 双缓冲 (Ping-Pong)

    LocalTensor<IN_T> l1QTensor;               ///< Q L1 LocalTensor (单缓冲)
    LocalTensor<IN_T> l1KTensor[2];            ///< K L1 LocalTensor [2] (双缓冲)
    LocalTensor<IN_T> aL0TensorPingPong[2];    ///< L0A LocalTensor [2] (双缓冲)
    LocalTensor<IN_T> bL0TensorPingPong[2];    ///< L0B LocalTensor [2] (双缓冲)
    LocalTensor<MM_OUT_T> cL0TensorPingPong[2];///< L0C LocalTensor [2] (双缓冲)
};


// ==================== BSAMatmulService 成员函数实现 ====================

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::InitParams(const BSAConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::InitBuffers(TPipe *pipe)
{
    // ---- L1 缓冲区: Q 单缓冲 + K 双缓冲 ----
    // Q 单缓冲: 尺寸 = CUBE_BASE_BLOCK(128) × dSize(128) × sizeof(float)= 64KB
    pipe->InitBuffer(qBuf, CUBE_BASE_BLOCK * constInfo.dSize * sizeof(IN_T));
    // K 双缓冲: 尺寸同上 ×2，Ping-Pong 以隐藏 K L1 搬运延迟
    pipe->InitBuffer(kBuf[0], CUBE_BASE_BLOCK * constInfo.dSize * sizeof(IN_T));
    pipe->InitBuffer(kBuf[1], CUBE_BASE_BLOCK * constInfo.dSize * sizeof(IN_T));

    // 获取 LocalTensor 句柄
    l1QTensor = qBuf.Get<IN_T>();
    l1KTensor[0] = kBuf[0].Get<IN_T>();
    l1KTensor[1] = kBuf[1].Get<IN_T>();

    // ---- L0 缓冲区: L0A/L0B/L0C 三组双缓冲 ----
    pipe->InitBuffer(tmpBufL0A[0], L0A_PP_SIZE);   // 32KB
    pipe->InitBuffer(tmpBufL0A[1], L0A_PP_SIZE);   // 32KB
    pipe->InitBuffer(tmpBufL0B[0], L0B_PP_SIZE);   // 32KB
    pipe->InitBuffer(tmpBufL0B[1], L0B_PP_SIZE);   // 32KB
    pipe->InitBuffer(tmpBufL0C[0], L0C_PP_SIZE);   // 32KB
    pipe->InitBuffer(tmpBufL0C[1], L0C_PP_SIZE);   // 32KB

    // 获取 LocalTensor 句柄
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
    // MTE1↔MTE2: Q 数据搬运同步
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);

    // MTE1↔MTE2: K 数据搬运同步 (Ping-Pong)
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_K[0]);
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_K[1]);

    // M↔MTE1: L0A/L0B 加载完成同步 (Ping-Pong)
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[0]);
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[1]);

    // FIX↔M: L0C 释放同步 (Ping-Pong)
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[0]);
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[1]);
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_K[0]);
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_K[1]);

    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[0]);
    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[1]);

    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[0]);
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[1]);
}


template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::CopyGmToL1(LocalTensor<IN_T> &l1Tensor,
                                                           GlobalTensor<IN_T> &gmSrcTensor,
                                                           uint32_t srcN,
                                                           uint32_t srcD,
                                                           uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;                                // ND 矩阵个数
    nd2nzPara.nValue = srcN;                            // 1个nd矩阵行数
    nd2nzPara.dValue = srcD;                            // 1个nd矩阵列数 (=dSize=128)
    nd2nzPara.srcDValue = srcDstride;                   // GM 中行 stride (=xBlocks*dSize 或 =yBlocks*dSize)
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16;   // NZ 格式 C0 stride，对齐到16， 即=原nd矩阵 行数
    nd2nzPara.dstNzNStride = 1;                         // N 方向 stride
    nd2nzPara.srcNdMatrixStride = 0; // 源操作数相邻ND矩阵起始地址间的偏移， 目前只搬1个矩阵设为0不影响
    nd2nzPara.dstNzMatrixStride = 0; // 相邻NZ矩阵起始地址间的偏移， 目前只搬1个矩阵设为0不影响
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}



template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::CopyInQToL1(uint32_t qChunkStart, uint32_t qChunkSize)
{
    // qCmp 完整 shape: [xBlocks, dSize]
    // 搬运 qCmp[qChunkStart : qChunkStart+qChunkSize, :]
    // GM 行 stride = xBlocks * dSize（跨 batch/head 后的全量行宽，此处 xBlocks 为全 Q 维度）
    uint32_t qGmOffset = qChunkStart * constInfo.dSize;
    auto qSrcGm = qCmpGmLocal[qGmOffset];
    // GM stride: qCmp 是 workspace 内连续存放的 [xBlocks, dSize]，stride = dSize * xBlocks
    // 但实际上 qCmp 是 [xBlocks, dSize]，调用方传入的 qCmpGm 是全量；stride = xBlocks * dSize
    // 注意：此处的 stride 应为 qCmp 在 GM 中的行 stride
    // 由于 qCmp 按 xBlocks*dSize 连续存放，行 stride = dSize
    // 但跨 batch/head 时需要更大的 stride。当前设计：单 batch/head 内 qCmp 行 stride = dSize
    CopyGmToL1(l1QTensor, qSrcGm, qChunkSize, constInfo.dSize, constInfo.dSize);
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::CopyInKToL1(uint32_t kBlockStart, uint32_t kCubeBlockLen,
                                                             uint8_t pingpongFlag)
{
    // kCmp 完整 shape: [yBlocks, dSize]
    // 搬运 kCmp[kBlockStart : kBlockStart+kCubeBlockLen, :]
    uint32_t kGmOffset = kBlockStart * constInfo.dSize;
    auto kSrcGm = kCmpGmLocal[kGmOffset];
    CopyGmToL1(l1KTensor[pingpongFlag & 1], kSrcGm, kCubeBlockLen, constInfo.dSize, constInfo.dSize);
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::LoadDataMmA(LocalTensor<IN_T> &aL0Tensor,
                                                            LocalTensor<IN_T> &aL1Tensor,
                                                            uint32_t mSize, uint32_t kSize)
{
    // M, K 对齐到 C0_SIZE=16
    uint32_t alignM = BSAAlignTo(mSize, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignK = BSAAlignTo(kSize, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = alignK / C0_SIZE;   // 每行分为 K/16 段
    loadData2DParams.srcStride = alignM / C0_SIZE;      // L1 中行 stride（以 C0 为单位）
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;                // 非转置搬运

    // 逐行加载：L1 的每一行 → L0A 的对应位置
    for (int32_t i = 0; i < static_cast<int32_t>(alignM / C0_SIZE); i++) {
        LoadData(aL0Tensor[i * alignK * C0_SIZE],
                 aL1Tensor[i * CUBE_MATRIX_SIZE],
                 loadData2DParams);
    }
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::LoadDataMmB(LocalTensor<IN_T> &bL0Tensor,
                                                            LocalTensor<IN_T> &bL1Tensor,
                                                            uint32_t kSize, uint32_t nSize)
{
    uint32_t alignK = BSAAlignTo(kSize, static_cast<uint32_t>(C0_SIZE));
    uint32_t alignN = BSAAlignTo(nSize, static_cast<uint32_t>(C0_SIZE));

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = alignK * alignN / CUBE_MATRIX_SIZE;  // 整体搬完
    loadData2DParams.srcStride = 1;                                      // L1 连续搬
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;                                // Cube 硬件 NZ 格式自动处理 K^T

    LoadData(bL0Tensor, bL1Tensor, loadData2DParams);
}


template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::MmadInner(uint32_t mSize, uint32_t nSize, uint32_t kSize,
                                                          bool isL0CInit, uint8_t l0PingPongFlag)
{
    uint8_t flagVal = l0PingPongFlag & 1;

    // 获取当前乒乓槽位的 L0 tensors
    LocalTensor<IN_T> l0aTensor = aL0TensorPingPong[flagVal];
    LocalTensor<IN_T> l0bTensor = bL0TensorPingPong[flagVal];
    LocalTensor<MM_OUT_T> l0cTensor = cL0TensorPingPong[flagVal];

    // Step 1: 等待 Cube 释放 L0A/L0B → 可以覆盖加载
    WaitFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[flagVal]);

    // Step 2: L1→L0A/L0B 数据加载
    //   Q: L1→L0A, shape [M, K]
    //   K: L1→L0B, shape [N, K] (Cube 硬件 NZ 格式处理 K^T)
    LoadDataMmA(l0aTensor, l1QTensor, mSize, kSize);
    LoadDataMmB(l0bTensor, l1KTensor[flagVal], kSize, nSize);

    // Step 3: 通知 Cube L0A/L0B 已就绪
    SetFlag<AscendC::HardEvent::MTE1_M>(SYNC_MTE1MM_FLAG[flagVal]);

    // Step 4: 等待 MTE1 完成 + Fixpipe 释放 L0C
    WaitFlag<AscendC::HardEvent::MTE1_M>(SYNC_MTE1MM_FLAG[flagVal]);
    WaitFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[flagVal]);

    // Step 5: Mmad 矩阵乘加
    MmadParams mmadParams;
    mmadParams.m = (mSize == 1) ? C0_SIZE : mSize;   // M ≥ 16（Cube 最低要求）
    mmadParams.n = nSize;                              // N 维度
    mmadParams.k = kSize;                              // K 维度
    mmadParams.cmatrixInitVal = isL0CInit;             // 是否清零 L0C
    Mmad(l0cTensor, l0aTensor, l0bTensor, mmadParams);
    // Step 6: 通知 MTE1 L0A/L0B 已释放（下一轮可覆盖）
    SetFlag<AscendC::HardEvent::M_MTE1>(SYNC_MTE1MM_FLAG[flagVal]);
}

template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::FixCopyOut(GlobalTensor<MM_OUT_T> &resGm,
                                                            uint32_t mSize, uint32_t nSize,
                                                            uint32_t dstStride, uint8_t l0PingPongFlag)
{
    uint8_t flagVal = l0PingPongFlag & 1;

    // Step 1: 通知 Fixpipe 可以开始
    SetFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[flagVal]);
    // Step 2: 等待 Cube Mmad 完成
    WaitFlag<AscendC::HardEvent::M_FIX>(SYNC_MMFIX_FLAG[flagVal]);

    // Step 3: Fixpipe — L0C→GM 搬移（无 ReLU，纯数据搬移）
    FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = nSize;
    fixpipeParams.mSize = mSize;
    fixpipeParams.srcStride = ((mSize + C0_SIZE - 1) / C0_SIZE) * C0_SIZE;   // L0C 行 stride（对齐到 16）
    fixpipeParams.dstStride = dstStride;                                      // GM 行 stride (= yBlocks)
    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 0;
    fixpipeParams.dstNdStride = 0;
    fixpipeParams.reluEn = false;    // 不做 ReLU

    LocalTensor<MM_OUT_T> l0cTensor = cL0TensorPingPong[flagVal];
    Fixpipe<MM_OUT_T, MM_OUT_T>(resGm, l0cTensor, fixpipeParams);
    // Step 4: 通知 Cube L0C 已释放
    SetFlag<AscendC::HardEvent::FIX_M>(SYNC_MMFIX_FLAG[flagVal]);
}


template <typename BSAT>
__aicore__ inline void BSAMatmulService<BSAT>::ComputeMatmulChunk(
    uint32_t qChunkStart, uint32_t qChunkSize,
    uint32_t kChunkStart, uint32_t kChunkSize)
{
    uint32_t dSize = constInfo.dSize;
    uint32_t yBlocks = constInfo.yBlocks;
    uint32_t xBlocks = constInfo.xBlocks;
    uint32_t numHeads = constInfo.numHeads;

    uint64_t perHead = static_cast<uint64_t>(xBlocks) * static_cast<uint64_t>(yBlocks);
    uint64_t perBatch = perHead * numHeads;

    uint32_t singleM = qChunkSize;                          // M: Q 行数 (≤128)
    uint32_t singleK = constInfo.dSize;                     // K: Head 维度 (128)
    uint32_t dstStride = constInfo.yBlocks;                 // GM 行 stride（attnScore 全量 K 维度）
    uint32_t nLoopTimes = (kChunkSize + CUBE_BASE_BLOCK - 1) / CUBE_BASE_BLOCK;  // N 方向分块数

    // ---- Step 1: Q 一次性加载到 L1 ----
    WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
    CopyInQToL1(qChunkStart, qChunkSize);
    SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_Q);
    WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_Q);

    // ---- Step 2: K 分块 Ping-Pong 循环 ----
    uint8_t l0PingPongFlag = 0;

    for (uint32_t nBlockIdx = 0; nBlockIdx < nLoopTimes; nBlockIdx++) {
        // 当前 K Block 的 N 大小（尾块可能 < 128）
        uint32_t kBlockStart = kChunkStart + nBlockIdx * CUBE_BASE_BLOCK;
        uint32_t curN = (nBlockIdx == nLoopTimes - 1)
                            ? (kChunkSize - nBlockIdx * CUBE_BASE_BLOCK)
                            : CUBE_BASE_BLOCK;

        // ---- 2a: 等待上一轮 K L1 释放，搬运当前 K Block ----
        WaitFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_K[l0PingPongFlag & 1]);
        CopyInKToL1(kBlockStart, curN, l0PingPongFlag);
        SetFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_K[l0PingPongFlag & 1]);
        WaitFlag<AscendC::HardEvent::MTE2_MTE1>(SYNC_MTE21_FLAG_K[l0PingPongFlag & 1]);

        // ---- 2b: Mmad (含 L1→L0 加载) ----
        MmadInner(singleM, curN, singleK, true /*isL0CInit*/, l0PingPongFlag);

        // ---- 2c: 释放 K L1，通知 MTE2 可搬运下一轮 ----
        SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_K[l0PingPongFlag & 1]);

        // ---- 2d: Fixpipe 写回 attnScoreGm ----
        // GM 偏移: attnScore[qChunkStart, kBlockStart]
        uint32_t scoreGmOffset = qChunkStart * dstStride + kBlockStart;
        auto dstScoreGm = scoreGmLocal[scoreGmOffset];

        FixCopyOut(dstScoreGm, singleM, curN, dstStride, l0PingPongFlag);

        // ---- 2e: 翻转乒乓旗标 ----
        l0PingPongFlag = (l0PingPongFlag + 1) & 1;
    }

    // ---- 收尾: 释放 Q L1 ----
    SetFlag<AscendC::HardEvent::MTE1_MTE2>(SYNC_MTE21_FLAG_Q);
}

#endif // BSA_SELECT_BLOCK_MASK_SERVICE_CUBE_H
