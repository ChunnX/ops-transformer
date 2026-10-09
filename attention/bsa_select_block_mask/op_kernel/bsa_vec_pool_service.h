/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file bsa_vec_pool_service.h
 * \brief
 */
#ifndef BSA_VEC_POOL_SERVICE_H
#define BSA_VEC_POOL_SERVICE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"

template <typename BSAT>
class BSAVecPoolService {
public:
    using T = float;                    ///< 中间计算类型 (float, FP32)
    using IN_T = typename BSAT::inputT; ///< 输入数据类型 (FP16/BF16)
    using OUT_T = half;                 ///< 输出数据类型 (float16)

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;   // BNSD or TND
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV; // BNSD or TND

    __aicore__ inline BSAVecPoolService(){};
    __aicore__ inline void InitParams(const BSAConstInfo& constInfo,
                                      const optiling::BSASelectBlockMaskTilingData* __restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<>* uBuf_);

    __aicore__ inline void InitGM(GlobalTensor<OUT_T>& qCmpGm, GlobalTensor<OUT_T>& kCmpGm, GlobalTensor<IN_T>& queryGm,
                                  GlobalTensor<IN_T>& keyGm, GlobalTensor<int64_t>& actualBlockLenQGm,
                                  GlobalTensor<int64_t>& actualBlockLenKVGm, GlobalTensor<int64_t>& actualSeqLensQGm,
                                  GlobalTensor<int64_t>& actualSeqLensKVGm);

    __aicore__ inline void AllocEventID();

    __aicore__ inline void FreeEventID();

    __aicore__ inline void PoolingSingleQBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t qBlockIdx,
                                               GlobalTensor<IN_T>& queryGm, GlobalTensor<int64_t>& actualBlockLenQGm,
                                               GlobalTensor<OUT_T>& qCmpGm, uint64_t seqPrefixSumQ,
                                               uint64_t blockPrefixSumQ);

    __aicore__ inline void PoolingSingleKBlock(uint32_t batchIdx, uint32_t headIdx, uint32_t kBlockIdx,
                                               GlobalTensor<IN_T>& keyGm, GlobalTensor<int64_t>& actualBlockLenKVGm,
                                               GlobalTensor<OUT_T>& kCmpGm, uint64_t seqPrefixSumKV,
                                               uint64_t blockPrefixSumKV);

    __aicore__ inline bool FastPoolingQRange(uint32_t batchIdx, uint32_t headIdx, uint32_t startQBlockIdx,
                                             uint32_t endQBlockIdx, uint64_t seqPrefixSumQ);

    __aicore__ inline bool FastPoolingKRange(uint32_t batchIdx, uint32_t headIdx, uint32_t startKBlock,
                                             uint32_t endKBlock, uint64_t seqPrefixSumKV);

private:
    __aicore__ inline void PoolingSingleBlockImpl(GlobalTensor<IN_T>& srcGm, uint64_t srcOffset, uint32_t actualLen,
                                                  uint32_t blockSize, GlobalTensor<OUT_T>& dstGm, uint64_t dstOffset,
                                                  bool isTnd, uint32_t timeStrideElems);

    __aicore__ inline void BatchPoolingRange(GlobalTensor<IN_T>& srcGm, uint64_t srcFirstElem, uint32_t rowPitchElems,
                                             uint32_t blockSize, uint32_t nBlocks, GlobalTensor<OUT_T>& dstGm,
                                             uint64_t dstFirstElem);

    __aicore__ inline void PoolingLen0Impl(GlobalTensor<IN_T>& srcGm, GlobalTensor<OUT_T>& dstGm, uint64_t srcOffset,
                                           uint64_t dstOffset);

    __aicore__ inline void PoolingLen256Impl(GlobalTensor<IN_T>& srcGm, GlobalTensor<OUT_T>& dstGm, uint64_t srcOffset,
                                             uint64_t dstOffset, uint32_t actualLen, bool isTnd,
                                             uint32_t timeStrideElems);

    __aicore__ inline void PoolingLenAllImpl(GlobalTensor<IN_T>& srcGm, GlobalTensor<OUT_T>& dstGm, uint64_t srcOffset,
                                             uint64_t dstOffset, uint32_t actualLen, bool isTnd,
                                             uint32_t timeStrideElems);

    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData* __restrict tilingData;

    GlobalTensor<OUT_T> qCmpWrkTensor; ///< Query 压缩矩阵 [xBlocks, dSize]
    GlobalTensor<OUT_T> kCmpWrkTensor; ///< Key 压缩矩阵 [yBlocks, dSize]

    GlobalTensor<IN_T> queryGmTensor;               ///< 原始 Query [B,N,S,D] or [T,N,D]
    GlobalTensor<IN_T> keyGmTensor;                 ///< 原始 Key [B,N,S,D] or [T,N,D]
    GlobalTensor<int64_t> actualBlockLenQGmTensor;  ///< Q 实际 Block 长度 (可为 NULL)
    GlobalTensor<int64_t> actualBlockLenKVGmTensor; ///< KV 实际 Block 长度 (可为 NULL)
    GlobalTensor<int64_t> actualSeqLensQGmTensor;   ///< Q 各 batch 实际序列长度 (TND 必选)
    GlobalTensor<int64_t> actualSeqLensKVGmTensor;  ///< KV 各 batch 实际序列长度 (TND 必选)

    constexpr static int32_t BUFFER = 2;   // 开启double buffer
    LocalTensor<IN_T> poolInputUb[BUFFER]; ///< 池化输入双缓冲 [2][blockShapeX * dSize] (FP16/BF16)
    LocalTensor<T> poolInFp32Ub[BUFFER];   ///< Cast后的FP32数据 [blockShapeX * dSize] (FP32)

    LocalTensor<T> poolOut;
    LocalTensor<OUT_T> poolOutCast;
    LocalTensor<T> poolOutTemp; // 临时存储out
    LocalTensor<uint8_t> reduceSharedTemp[BUFFER];

    event_t eventPing;
    event_t eventPong;

    // —— 批量池化快速路径（poolQ/poolK）：转置式跨步读 + 单次大宽 ReduceMean ——
    // 逐块路径每 2KB block 携带 6 对 Set/Wait + 4 次 barrier（0.84µs/块，纯同步开销），
    // 批量路径按 batchBlk 块一轮：8 次跨步 GM 读转置成 [bs][W] + 1 Cast + 1 ReduceMean{bs,W}
    // + 1 Cast + 1 连续写回，每轮仅 2 对 Set/Wait。缓冲与逐块缓冲同相位覆盖复用（≤180KB）。
    uint32_t batchBlk = 0;           // 每轮 block 数（0=批量不可用，回退逐块路径）
    uint32_t batchWMax = 0;          // 每轮列数 = batchBlk * dSize
    LocalTensor<IN_T> batchFp16;     // 转置布局 [blockShapeX][W]（overlay poolInputUb 区，cast 后失效）
    LocalTensor<T> batchFp32;        // [blockShapeX][W]（overlay poolInFp32Ub 区）
    LocalTensor<T> batchOutFp32;     // ReduceMean 输出 [W]
    LocalTensor<OUT_T> batchOutFp16; // 输出 fp16 [W]
};

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::InitParams(
    const BSAConstInfo& constInfo, const optiling::BSASelectBlockMaskTilingData* __restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::InitBuffers(TBuf<>* uBuf_)
{
    // 初始化统一 UB 缓冲区（192KB）
    uint32_t ubOffset = 0;

    // 池化步骤内存分配
    /*
    FP16 :30KB
    ||{fp16[0] | fp16[1] }||{ fp32[0] | fp32[1]}|| outTemp 9 * 128 * 4 || fina out 128 * 4||
    || reduceSum ||

    sum: 30 * 2 + 60 * 2 + 4.5 + 0.5 = 185k
    */

    // input fp16 ub
    for (int32_t i = 0; i < BUFFER; i++) {
        uint32_t poolInputElements = BSAConstInfo::BUFFER_SIZE_BYTE_60K / BUFFER / sizeof(IN_T);
        poolInputUb[i] = uBuf_->GetWithOffset<IN_T>(poolInputElements, ubOffset);

        // reduceMean temp ub 复用fp16 ub
        uint32_t sharedTensorMaxEles = BSAConstInfo::BUFFER_SIZE_BYTE_60K / BUFFER / sizeof(uint8_t);
        reduceSharedTemp[i] = uBuf_->GetWithOffset<uint8_t>(sharedTensorMaxEles, 0);

        ubOffset += poolInputElements * sizeof(IN_T);
        ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));
    }

    // cast input fp32 ub
    uint32_t poolFp32Elements = BSAConstInfo::BUFFER_SIZE_BYTE_120K / BUFFER / sizeof(T);
    for (int32_t i = 0; i < BUFFER; i++) {
        poolInFp32Ub[i] = uBuf_->GetWithOffset<T>(poolFp32Elements, ubOffset);
        ubOffset += poolFp32Elements * sizeof(T);
        ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));
    }

    uint32_t poolOutTempMaxEles = 9 * 128; // (8 + 1) * 128
    poolOutTemp = uBuf_->GetWithOffset<T>(poolOutTempMaxEles, ubOffset);
    ubOffset += poolOutTempMaxEles * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t poolOutFinaEles = 1 * 128;
    poolOut = uBuf_->GetWithOffset<T>(poolOutFinaEles, ubOffset);
    ubOffset += poolOutFinaEles * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t poolOutCastEles = 1 * 128;
    poolOutCast = uBuf_->GetWithOffset<OUT_T>(poolOutCastEles, ubOffset);

    // 批量快速路径覆盖视图：fp16 域 [0,60KB)（30720 elems）、fp32 域 [60KB,180KB)（30720 elems）
    // 内重划——与逐块缓冲同相位串行复用（poolQ/poolK 相位互斥，入口 PIPE_ALL barrier 切换）。
    // fp32 域须容纳 bs*W(输入) + W(输出)，fp16 域须容纳 bs*W，由此定每轮 block 上限。
    {
        const uint32_t bs = constInfo.blockShapeX;
        const uint32_t d = constInfo.dSize;
        uint32_t blkCap = (bs > 0 && d > 0) ? (30720 / (bs + 1) / d) : 0;
        batchBlk = BSAMin(blkCap, static_cast<uint32_t>(24));
        if (batchBlk >= 4) {
            batchWMax = batchBlk * d;
            batchFp16 = uBuf_->GetWithOffset<IN_T>(bs * batchWMax, 0);
            batchFp32 = uBuf_->GetWithOffset<T>(bs * batchWMax, 60 * 1024);
            uint32_t batchOff = 60 * 1024 + bs * batchWMax * sizeof(T);
            batchOutFp32 = uBuf_->GetWithOffset<T>(batchWMax, batchOff);
            batchOff += batchWMax * sizeof(T);
            batchOutFp16 = uBuf_->GetWithOffset<OUT_T>(batchWMax, batchOff);
        } else {
            batchBlk = 0;
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::InitGM(GlobalTensor<OUT_T>& qCmpGm, GlobalTensor<OUT_T>& kCmpGm,
                                                       GlobalTensor<IN_T>& queryGm, GlobalTensor<IN_T>& keyGm,
                                                       GlobalTensor<int64_t>& actualBlockLenQGm,
                                                       GlobalTensor<int64_t>& actualBlockLenKVGm,
                                                       GlobalTensor<int64_t>& actualSeqLensQGm,
                                                       GlobalTensor<int64_t>& actualSeqLensKVGm)
{
    this->qCmpWrkTensor = qCmpGm;
    this->kCmpWrkTensor = kCmpGm;
    this->queryGmTensor = queryGm;
    this->keyGmTensor = keyGm;
    this->actualBlockLenQGmTensor = actualBlockLenQGm;
    this->actualBlockLenKVGmTensor = actualBlockLenKVGm;
    this->actualSeqLensQGmTensor = actualSeqLensQGm;
    this->actualSeqLensKVGmTensor = actualSeqLensKVGm;
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::AllocEventID()
{
    eventPing = (event_t)0;
    eventPong = (event_t)1;
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::FreeEventID()
{
    return;
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::PoolingSingleQBlock(uint32_t batchIdx, uint32_t headIdx,
                                                                    uint32_t qBlockIdx, GlobalTensor<IN_T>& queryGm,
                                                                    GlobalTensor<int64_t>& actualBlockLenQGm,
                                                                    GlobalTensor<OUT_T>& qCmpGm, uint64_t seqPrefixSumQ,
                                                                    uint64_t blockPrefixSumQ)
{
    uint32_t dSize = constInfo.dSize;
    uint64_t blockShapeX = constInfo.blockShapeX;
    uint32_t xBlocks = constInfo.xBlocks;
    uint32_t numHeads = constInfo.numHeads;
    uint32_t maxQSeqlen = constInfo.maxQSeqlen;

    // 1. 计算 srcOffset（元素偏移）
    uint64_t srcOffset;
    if constexpr (BSAT::layoutQ == BSALayout::BNSD) {
        uint64_t perHeadBytes = static_cast<uint64_t>(maxQSeqlen) * dSize * sizeof(IN_T);
        uint64_t perBatchBytes = perHeadBytes * numHeads;
        srcOffset = batchIdx * perBatchBytes + headIdx * perHeadBytes + qBlockIdx * blockShapeX * dSize * sizeof(IN_T);
        srcOffset = srcOffset / sizeof(IN_T);
    } else {
        // TND: 用 seqPrefixSumQ（前缀和）替代 batchIdx * maxQSeqlen，支持变长拼接
        uint64_t perSeqHeadBytes = static_cast<uint64_t>(numHeads) * dSize * sizeof(IN_T);
        srcOffset = seqPrefixSumQ * perSeqHeadBytes + (qBlockIdx * blockShapeX) * perSeqHeadBytes +
                    headIdx * dSize * sizeof(IN_T);
        srcOffset = srcOffset / sizeof(IN_T);
    }

    // 2. 获取 actualLen
    uint32_t actualLen;
    if (tilingData->baseParams.useActualBlockLenQ) {
        uint64_t blkIdx = (BSAT::layoutQ == BSALayout::TND) ? (blockPrefixSumQ + qBlockIdx) :
                                                              (static_cast<uint64_t>(batchIdx) * xBlocks + qBlockIdx);
        actualLen = static_cast<uint32_t>(actualBlockLenQGm.GetValue(blkIdx));

        // 防止 actualLen > blockShapeX情形
        actualLen = BSAMin(actualLen, blockShapeX);
    } else {
        actualLen = static_cast<uint32_t>(blockShapeX);
    }

    // TND 下各 batch 序列长度不同，需按当前 batch 的实际 seqlen 计算剩余有效长度，
    // 避免短序列 batch 读取到下一个 batch 的数据
    uint32_t tokenStart = qBlockIdx * static_cast<uint32_t>(blockShapeX);
    uint32_t curBatchSeqlen;
    if (tilingData->baseParams.useActualSeqLenQ) {
        curBatchSeqlen = static_cast<uint32_t>(actualSeqLensQGmTensor.GetValue(batchIdx));
    } else {
        curBatchSeqlen = maxQSeqlen;
    }
    uint32_t remaining = (curBatchSeqlen > tokenStart) ? (curBatchSeqlen - tokenStart) : 0;
    actualLen = BSAMin(actualLen, remaining);

    // 3. 计算 dstOffset（元素索引）
    uint64_t wrkDstOffset = qBlockIdx * dSize;

    // 4. 调用底层池化实现
    // TND [T,N,D] 中 head 的数据逐 timestep 交错，相邻 timestep 间隔 numHeads*dSize 个元素
    constexpr bool isTndQ = (BSAT::layoutQ == BSALayout::TND);
    uint32_t timeStrideElems = isTndQ ? (numHeads * dSize) : 0;
    PoolingSingleBlockImpl(queryGm, srcOffset, actualLen, static_cast<uint32_t>(blockShapeX), qCmpGm, wrkDstOffset,
                           isTndQ, timeStrideElems);
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::PoolingSingleKBlock(uint32_t batchIdx, uint32_t headIdx,
                                                                    uint32_t kBlockIdx, GlobalTensor<IN_T>& keyGm,
                                                                    GlobalTensor<int64_t>& actualBlockLenKVGm,
                                                                    GlobalTensor<OUT_T>& kCmpGm,
                                                                    uint64_t seqPrefixSumKV, uint64_t blockPrefixSumKV)
{
    uint32_t dSize = constInfo.dSize;
    uint64_t blockShapeY = constInfo.blockShapeY;
    uint32_t yBlocks = constInfo.yBlocks;
    uint32_t numHeads = constInfo.numHeads;
    uint32_t maxKvSeqlen = constInfo.maxKvSeqlen;

    // 1. 计算 srcOffset（字节偏移）
    uint64_t srcOffset = 0;
    if constexpr (BSAT::layoutKV == BSALayout::BNSD) {
        uint64_t perHeadBytes = static_cast<uint64_t>(maxKvSeqlen) * dSize * sizeof(IN_T);
        uint64_t perBatchBytes = perHeadBytes * numHeads;
        srcOffset = batchIdx * perBatchBytes + headIdx * perHeadBytes + kBlockIdx * blockShapeY * dSize * sizeof(IN_T);
        srcOffset = srcOffset / sizeof(IN_T);
    } else {
        // TND: 用 seqPrefixSumKV（前缀和）替代 batchIdx * maxKvSeqlen，支持变长拼接
        uint64_t perSeqHeadBytes = static_cast<uint64_t>(numHeads) * dSize * sizeof(IN_T);
        srcOffset = seqPrefixSumKV * perSeqHeadBytes + (kBlockIdx * blockShapeY) * perSeqHeadBytes +
                    headIdx * dSize * sizeof(IN_T);
        srcOffset = srcOffset / sizeof(IN_T);
    }

    // 2. 获取 actualLen
    uint32_t actualLen;
    if (tilingData->baseParams.useActualBlockLenK) {
        uint64_t blkIdx = (BSAT::layoutKV == BSALayout::TND) ?
                              (blockPrefixSumKV + kBlockIdx) :
                              (static_cast<uint64_t>(batchIdx) * constInfo.yBlocks + kBlockIdx);
        actualLen = static_cast<uint32_t>(actualBlockLenKVGm.GetValue(blkIdx));
        // 防止 actualLen > blockShapeX情形
        actualLen = BSAMin(actualLen, blockShapeY);
    } else {
        actualLen = static_cast<uint32_t>(blockShapeY);
    }

    // 考虑末端实际不足blockShapeY场景，即seqlen无法整除blocksize场景
    // TND 下各 batch 序列长度不同，需按当前 batch 的实际 seqlen 计算剩余有效长度
    uint32_t tokenStart = kBlockIdx * static_cast<uint32_t>(blockShapeY);
    uint32_t curBatchSeqlen;
    if (tilingData->baseParams.useActualSeqLenK) {
        curBatchSeqlen = static_cast<uint32_t>(actualSeqLensKVGmTensor.GetValue(batchIdx));
    } else {
        curBatchSeqlen = maxKvSeqlen;
    }
    uint32_t remaining = (curBatchSeqlen > tokenStart) ? (curBatchSeqlen - tokenStart) : 0;
    actualLen = BSAMin(actualLen, remaining);

    // 3. 计算 dstOffset（元素索引）
    uint64_t wrkDstOffset = kBlockIdx * dSize;

    // 4. 调用底层池化实现
    // TND [T,N,D] 中 head 的数据逐 timestep 交错，相邻 timestep 间隔 numHeads*dSize 个元素
    constexpr bool isTndKV = (BSAT::layoutKV == BSALayout::TND);
    uint32_t timeStrideElems = isTndKV ? (numHeads * dSize) : 0;
    PoolingSingleBlockImpl(keyGm, srcOffset, actualLen, static_cast<uint32_t>(blockShapeY), kCmpGm, wrkDstOffset,
                           isTndKV, timeStrideElems);
}

// —— 批量池化核心：nBlocks 个满块（actualLen == blockSize）一轮批处理 ——
// src 布局：block g（0 起）行 r 首元素 = srcFirstElem + g*rowPitchElems + r*dSize（BNSD/TND 统一），
// 转置式跨步读固定行 r 收集 nb 个 block → fp16 [blockSize][nb*d]；ReduceMean 沿 blockSize 行归约。
template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::BatchPoolingRange(GlobalTensor<IN_T>& srcGm, uint64_t srcFirstElem,
                                                                  uint32_t rowPitchElems, uint32_t blockSize,
                                                                  uint32_t nBlocks, GlobalTensor<OUT_T>& dstGm,
                                                                  uint64_t dstFirstElem)
{
    const uint32_t d = constInfo.dSize;

    AscendC::PipeBarrier<PIPE_ALL>();
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<IN_T> padInParams = {false, 0, 0, 0};
    copyInParams.blockLen = d * sizeof(IN_T);
    copyInParams.blockCount = 0; // 每轮填
    copyInParams.srcStride = (rowPitchElems - d) * sizeof(IN_T);
    copyInParams.dstStride = 0;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    for (uint32_t b0 = 0; b0 < nBlocks; b0 += batchBlk) {
        const uint32_t nb = BSAMin(batchBlk, nBlocks - b0);
        const uint32_t w = nb * d;
        // 跨轮排空：上轮 MTE3（GM 写读 batchOutFp16）必须先于本轮 V（Cast 覆写 batchOutFp16）——
        // MTE3_MTE2 Set/Wait 排空上轮 MTE3，经 MTE2_V 传递保证本轮 V 在其后（与逐块路径同协议）
        if (b0 > 0) {
            SetFlag<HardEvent::MTE3_MTE2>(eventPing);
            WaitFlag<HardEvent::MTE3_MTE2>(eventPing);
        }
        // 8 次跨步 GM 读：行 r 的 nb 个 block（各 d 列）聚拢到 batchFp16[r*w]
        copyInParams.blockCount = nb;
        for (uint32_t r = 0; r < blockSize; r++) {
            DataCopyPad(batchFp16[r * w], srcGm[srcFirstElem + b0 * rowPitchElems + r * d], copyInParams, padInParams);
        }
        SetFlag<HardEvent::MTE2_V>(eventPing);
        WaitFlag<HardEvent::MTE2_V>(eventPing);

        Cast(batchFp32, batchFp16, RoundMode::CAST_NONE, blockSize * w);
        AscendC::PipeBarrier<PIPE_V>();

        // 手工复刻 RA ReduceMean{blockSize, w} 的内部序列（BinaryReduceByFirstAxis 连续半区树
        // + Muls(1/first)，逐位等价）：大 W 下 RA 接口的 counter-mode 单 repeat 大计数异常，
        // 此处用高层 Add/Muls（内部按 64 elem/repeat 循环 + 全 mask）规避该上限。
        // 前置条件：blockSize 为 2 的幂（树结构要求，快速路径入口已校验）。
        for (uint32_t half = blockSize / 2; half >= 1; half >>= 1) {
            AscendC::Add(batchFp32, batchFp32, batchFp32[half * w], half * w);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // 与 RA ReduceMean 内部一致：1.0f / int32(blockSize)（aicore 禁 float↔unsigned 显式转换）
        const float invBlockSize = 1.0f / static_cast<int32_t>(blockSize);
        AscendC::Muls(batchOutFp32, batchFp32, invBlockSize, w);
        AscendC::PipeBarrier<PIPE_V>();

        Cast(batchOutFp16, batchOutFp32, RoundMode::CAST_NONE, w);
        SetFlag<HardEvent::V_MTE3>(eventPing);
        WaitFlag<HardEvent::V_MTE3>(eventPing);

        copyOutParams.blockLen = w * sizeof(OUT_T);
        DataCopyPad(dstGm[dstFirstElem + b0 * d], batchOutFp16, copyOutParams);
    }
}

template <typename BSAT>
__aicore__ inline bool BSAVecPoolService<BSAT>::FastPoolingQRange(uint32_t batchIdx, uint32_t headIdx,
                                                                  uint32_t startQBlockIdx, uint32_t endQBlockIdx,
                                                                  uint64_t seqPrefixSumQ)
{
    if (batchBlk == 0 || startQBlockIdx >= endQBlockIdx) {
        return false;
    }
    // 快速路径条件：无变长 block/序列（actualLen 恒 = blockShapeX），且范围末端满块不出界
    if (tilingData->baseParams.useActualBlockLenQ || tilingData->baseParams.useActualSeqLenQ) {
        return false;
    }
    if ((constInfo.blockShapeX & (constInfo.blockShapeX - 1)) != 0) { // 手工树要求 2 的幂
        return false;
    }
    if (static_cast<uint64_t>(endQBlockIdx) * constInfo.blockShapeX > constInfo.maxQSeqlen) {
        return false;
    }
    const uint32_t d = constInfo.dSize;
    const uint64_t bs = constInfo.blockShapeX;
    uint64_t srcFirst;
    uint32_t rowPitch;
    if constexpr (BSAT::layoutQ == BSALayout::BNSD) {
        const uint64_t perHeadElems = static_cast<uint64_t>(constInfo.maxQSeqlen) * d;
        srcFirst = static_cast<uint64_t>(batchIdx) * perHeadElems * constInfo.numHeads +
                   static_cast<uint64_t>(headIdx) * perHeadElems + startQBlockIdx * bs * d;
        rowPitch = static_cast<uint32_t>(bs * d);
    } else {
        const uint64_t perSeqHeadElems = static_cast<uint64_t>(constInfo.numHeads) * d;
        srcFirst = seqPrefixSumQ * perSeqHeadElems + startQBlockIdx * bs * perSeqHeadElems + headIdx * d;
        rowPitch = static_cast<uint32_t>(bs * perSeqHeadElems);
    }
    BatchPoolingRange(queryGmTensor, srcFirst, rowPitch, static_cast<uint32_t>(bs), endQBlockIdx - startQBlockIdx,
                      qCmpWrkTensor, static_cast<uint64_t>(startQBlockIdx) * d);
    return true;
}

template <typename BSAT>
__aicore__ inline bool BSAVecPoolService<BSAT>::FastPoolingKRange(uint32_t batchIdx, uint32_t headIdx,
                                                                  uint32_t startKBlock, uint32_t endKBlock,
                                                                  uint64_t seqPrefixSumKV)
{
    if (batchBlk == 0 || startKBlock >= endKBlock) {
        return false;
    }
    if (tilingData->baseParams.useActualBlockLenK || tilingData->baseParams.useActualSeqLenK) {
        return false;
    }
    if ((constInfo.blockShapeY & (constInfo.blockShapeY - 1)) != 0) { // 手工树要求 2 的幂
        return false;
    }
    if (static_cast<uint64_t>(endKBlock) * constInfo.blockShapeY > constInfo.maxKvSeqlen) {
        return false;
    }
    const uint32_t d = constInfo.dSize;
    const uint64_t bs = constInfo.blockShapeY;
    uint64_t srcFirst;
    uint32_t rowPitch;
    if constexpr (BSAT::layoutKV == BSALayout::BNSD) {
        const uint64_t perHeadElems = static_cast<uint64_t>(constInfo.maxKvSeqlen) * d;
        srcFirst = static_cast<uint64_t>(batchIdx) * perHeadElems * constInfo.numHeads +
                   static_cast<uint64_t>(headIdx) * perHeadElems + startKBlock * bs * d;
        rowPitch = static_cast<uint32_t>(bs * d);
    } else {
        const uint64_t perSeqHeadElems = static_cast<uint64_t>(constInfo.numHeads) * d;
        srcFirst = seqPrefixSumKV * perSeqHeadElems + startKBlock * bs * perSeqHeadElems + headIdx * d;
        rowPitch = static_cast<uint32_t>(bs * perSeqHeadElems);
    }
    BatchPoolingRange(keyGmTensor, srcFirst, rowPitch, static_cast<uint32_t>(bs), endKBlock - startKBlock,
                      kCmpWrkTensor, static_cast<uint64_t>(startKBlock) * d);
    return true;
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::PoolingSingleBlockImpl(GlobalTensor<IN_T>& srcGm, uint64_t srcOffset,
                                                                       uint32_t actualLen, uint32_t blockSize,
                                                                       GlobalTensor<OUT_T>& dstGm, uint64_t dstOffset,
                                                                       bool isTnd, uint32_t timeStrideElems)
{
    uint32_t dSize = constInfo.dSize;
    uint32_t maxLen = BSAConstInfo::BUFFER_SIZE_BYTE_60K / BUFFER / sizeof(IN_T) / dSize;
    // 等上个步骤结束，防止内存踩踏
    AscendC::PipeBarrier<PIPE_ALL>();
    if (actualLen == 0) {
        PoolingLen0Impl(srcGm, dstGm, srcOffset, dstOffset);
        return;
    } else if (actualLen <= maxLen) {
        PoolingLen256Impl(srcGm, dstGm, srcOffset, dstOffset, actualLen, isTnd, timeStrideElems);
        return;
    } else {
        PoolingLenAllImpl(srcGm, dstGm, srcOffset, dstOffset, actualLen, isTnd, timeStrideElems);
        return;
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::PoolingLen0Impl(GlobalTensor<IN_T>& srcGm, GlobalTensor<OUT_T>& dstGm,
                                                                uint64_t srcOffset, uint64_t dstOffset)
{
    uint32_t dSize = constInfo.dSize;

    event_t eventId = eventPing;
    SetFlag<HardEvent::MTE3_V>(eventPing);
    WaitFlag<HardEvent::MTE3_V>(eventPing);
    SetFlag<HardEvent::MTE3_V>(eventPong);
    WaitFlag<HardEvent::MTE3_V>(eventPong);

    Duplicate(poolOutCast, (OUT_T)0.0f, dSize);
    AscendC::PipeBarrier<PIPE_V>();
    Duplicate(poolOut, 0.0f, dSize);

    SetFlag<HardEvent::V_MTE3>(eventPing);
    WaitFlag<HardEvent::V_MTE3>(eventPing);

    DataCopyExtParams copyOutParams;
    copyOutParams.blockLen = dSize * sizeof(float);
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    DataCopyExtParams copyOutCastParams;
    copyOutCastParams.blockLen = dSize * sizeof(OUT_T);
    copyOutCastParams.blockCount = 1;
    copyOutCastParams.srcStride = 0;
    copyOutCastParams.dstStride = 0;
    DataCopyPad(dstGm[dstOffset], poolOutCast, copyOutCastParams);
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::PoolingLen256Impl(GlobalTensor<IN_T>& srcGm, GlobalTensor<OUT_T>& dstGm,
                                                                  uint64_t srcOffset, uint64_t dstOffset,
                                                                  uint32_t actualLen, bool isTnd,
                                                                  uint32_t timeStrideElems)
{
    uint32_t dSize = constInfo.dSize;

    uint32_t bufferId = 0;

    SetFlag<HardEvent::MTE3_MTE2>(eventPing);
    WaitFlag<HardEvent::MTE3_MTE2>(eventPing);
    SetFlag<HardEvent::MTE3_MTE2>(eventPong);
    WaitFlag<HardEvent::MTE3_MTE2>(eventPong);

    // Step 1: DataCopyPad GM→UB，搬入 actualLen × D 个 IN_T 元素到 poolInputUb[bufferId]
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<IN_T> padInParams = {false, 0, 0, 0};
    if (isTnd) {
        // TND [T,N,D]: head 的数据逐 timestep 交错，一次读 actualLen*D 会读到其他 head 的数据。
        // 用 blockCount=actualLen、srcStride=(numHeads-1)*D 单次批量搬运：每 block 读 D 个元素，
        // 相邻 block 间跳过 (numHeads-1)*D 个元素（总步长 numHeads*D，减去已读的 D）。
        copyInParams.blockLen = dSize * sizeof(IN_T);
        copyInParams.blockCount = actualLen;
        copyInParams.srcStride = (constInfo.numHeads - 1) * dSize * sizeof(IN_T);
        copyInParams.dstStride = 0;
        DataCopyPad(poolInputUb[bufferId], srcGm[srcOffset], copyInParams, padInParams);
    } else {
        // BNSD: head 数据连续，一次读 actualLen*D 个元素
        copyInParams.blockLen = actualLen * dSize * sizeof(IN_T);
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(poolInputUb[bufferId], srcGm[srcOffset], copyInParams, padInParams);
    }

    SetFlag<HardEvent::MTE2_V>(eventPing);
    WaitFlag<HardEvent::MTE2_V>(eventPing);

    // Step 2: Cast 全部 [actualLen, D] IN_T → FP32 到 poolInFp32Ub
    Cast(poolInFp32Ub[bufferId], poolInputUb[bufferId], RoundMode::CAST_NONE, actualLen * dSize);
    AscendC::PipeBarrier<PIPE_V>();

    // Step 3: 使用 ReduceSum 沿 seqlen 维度求和
    uint32_t shape[] = {actualLen, dSize};
    bool isSrcInnerPad = true;
    if (actualLen % FLOAT_DATA_BLOCK_NUM != 0) {
        isSrcInnerPad = false;
    }
    constexpr bool isReuseSource = true;
    AscendC::ReduceMean<T, AscendC::Pattern::Reduce::RA, true>(poolOut, poolInFp32Ub[bufferId],
                                                               reduceSharedTemp[bufferId], shape, isSrcInnerPad);
    AscendC::PipeBarrier<PIPE_V>();

    // Step4 : cast fp32 to INT
    Cast(poolOutCast, poolOut, RoundMode::CAST_NONE, dSize);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventPing);
    WaitFlag<HardEvent::V_MTE3>(eventPing);

    // step5 : copy to gm
    DataCopyExtParams copyOutCastParams;
    copyOutCastParams.blockLen = dSize * sizeof(OUT_T);
    copyOutCastParams.blockCount = 1;
    copyOutCastParams.srcStride = 0;
    copyOutCastParams.dstStride = 0;
    DataCopyPad(dstGm[dstOffset], poolOutCast, copyOutCastParams);
}

template <typename BSAT>
__aicore__ inline void BSAVecPoolService<BSAT>::PoolingLenAllImpl(GlobalTensor<IN_T>& srcGm, GlobalTensor<OUT_T>& dstGm,
                                                                  uint64_t srcOffset, uint64_t dstOffset,
                                                                  uint32_t actualLen, bool isTnd,
                                                                  uint32_t timeStrideElems)
{
    uint32_t dSize = constInfo.dSize;
    uint32_t numHeads = constInfo.numHeads;
    uint32_t maxLen = BSAConstInfo::BUFFER_SIZE_BYTE_60K / BUFFER / sizeof(IN_T) / dSize;
    uint32_t loops = (actualLen + maxLen - 1) / maxLen;
    uint32_t tailLen = actualLen % maxLen == 0 ? maxLen : actualLen % maxLen;

    DataCopyExtParams copyInParams;

    uint32_t loopNum = 0;
    uint32_t finalOutLocation = 0;
    bool isFirstMerge = true;
    uint32_t bufferId = 0;
    event_t eventId;

    SetFlag<HardEvent::V_MTE2>(eventPing);
    SetFlag<HardEvent::V_MTE2>(eventPong);

    SetFlag<HardEvent::MTE3_MTE2>(eventPing);
    WaitFlag<HardEvent::MTE3_MTE2>(eventPing);
    SetFlag<HardEvent::MTE3_MTE2>(eventPong);
    WaitFlag<HardEvent::MTE3_MTE2>(eventPong);

    for (uint32_t loopIdx = 0; loopIdx < loops; loopIdx++) {
        loopNum++;

        bufferId = loopIdx % BUFFER;
        eventId = bufferId == 0 ? eventPing : eventPong;

        // TND 下相邻 timestep 在 GM 中间隔 numHeads*dSize 个元素，单次 loop 推进 maxLen 个 timestep
        // 对应 GM 偏移 maxLen*numHeads*dSize；BNSD 下 head 数据连续，推进 maxLen*dSize。
        uint64_t inputGmOffset;
        if (isTnd) {
            inputGmOffset = static_cast<uint64_t>(loopIdx) * maxLen * numHeads * dSize + srcOffset;
        } else {
            inputGmOffset = loopIdx * maxLen * dSize + srcOffset;
        }
        uint32_t excuteLen = (loopIdx == (loops - 1) ? tailLen : maxLen);

        WaitFlag<HardEvent::V_MTE2>(eventId);

        // Step 1: DataCopyPad GM->UB
        DataCopyPadExtParams<IN_T> padInParams = {false, 0, 0, 0};
        if (isTnd) {
            // TND: 用 blockCount=excuteLen、srcStride=(numHeads-1)*D 单次批量搬运
            copyInParams.blockLen = dSize * sizeof(IN_T);
            copyInParams.blockCount = excuteLen;
            copyInParams.srcStride = (constInfo.numHeads - 1) * dSize * sizeof(IN_T);
            copyInParams.dstStride = 0;
            DataCopyPad(poolInputUb[bufferId], srcGm[inputGmOffset], copyInParams, padInParams);
        } else {
            // BNSD: 一次读 excuteLen*D 个连续元素
            copyInParams.blockLen = excuteLen * dSize * sizeof(IN_T);
            copyInParams.blockCount = 1;
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            DataCopyPad(poolInputUb[bufferId], srcGm[inputGmOffset], copyInParams, padInParams);
        }

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        // Step 2: Cast IN_T -> FP32
        Cast(poolInFp32Ub[bufferId], poolInputUb[bufferId], RoundMode::CAST_NONE, excuteLen * dSize);
        AscendC::PipeBarrier<PIPE_V>();

        // Step 3: ReduceSum per chunk (raw sum, NOT mean)
        uint32_t shape[] = {excuteLen, dSize};
        bool isSrcInnerPad = true;

        // Determine output row offset in poolOutTemp (same logic as original)
        uint32_t outTempOffset = ((loopNum - 1) % FLOAT_DATA_BLOCK_NUM + 1) * dSize;
        if (finalOutLocation == 0 && isFirstMerge) {
            outTempOffset = ((loopNum - 1) % FLOAT_DATA_BLOCK_NUM) * dSize;
        }

        // ReduceSum (NOT ReduceMean) - accumulate raw sum for this chunk
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(poolOutTemp[outTempOffset], poolInFp32Ub[bufferId],
                                                                  reduceSharedTemp[bufferId], shape, isSrcInnerPad);
        AscendC::PipeBarrier<PIPE_V>();

        // Intermediate merge: ReduceSum over stored rows when poolOutTemp is full
        if ((isFirstMerge && loopNum == FLOAT_DATA_BLOCK_NUM) ||
            (!isFirstMerge && (loopNum % (FLOAT_DATA_BLOCK_NUM - 1) == 0))) {
            if (isFirstMerge) {
                isFirstMerge = false;
            }
            uint32_t mergeOutOffset = finalOutLocation == 0 ? FLOAT_DATA_BLOCK_NUM * dSize : 0;
            uint32_t mergeInOffset = finalOutLocation == 0 ? 0 : 1 * dSize;

            uint32_t mergeShape[] = {FLOAT_DATA_BLOCK_NUM, dSize};
            bool mergeIsSrcInnerPad = true;

            // ReduceSum for merge (accumulate raw sums, NOT mean-of-means)
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(
                poolOutTemp[mergeOutOffset], poolOutTemp[mergeInOffset], reduceSharedTemp[bufferId], mergeShape,
                mergeIsSrcInnerPad);
            AscendC::PipeBarrier<PIPE_V>();

            finalOutLocation = 1 - finalOutLocation;
            loopNum = 0;
        }

        SetFlag<HardEvent::V_MTE2>(eventId);
    }

    WaitFlag<HardEvent::V_MTE2>(eventPing);
    WaitFlag<HardEvent::V_MTE2>(eventPong);

    // Final merge: ReduceSum over remaining rows in poolOutTemp
    if (loopNum != 0) {
        if (!isFirstMerge && finalOutLocation == 1) {
            // Step 1: ReduceSum new chunks into row 0
            uint32_t chunkShape[] = {loopNum, dSize};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(poolOutTemp[0], poolOutTemp[dSize],
                                                                      reduceSharedTemp[bufferId], chunkShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            // Step 2: Add accumulated sum (row 8) to chunk sum (row 0)
            Add(poolOutTemp[0], poolOutTemp[0], poolOutTemp[FLOAT_DATA_BLOCK_NUM * dSize], static_cast<int32_t>(dSize));
            AscendC::PipeBarrier<PIPE_V>();
            finalOutLocation = 0; // result in row 0
        } else {
            // - isFirstMerge: chunks in rows 0..loopNum-1, rDim=loopNum
            uint32_t mergeOutOffset = FLOAT_DATA_BLOCK_NUM * dSize; // write to row 8
            uint32_t mergeInOffset = 0;
            uint32_t rDim = isFirstMerge ? loopNum : (loopNum + 1);
            uint32_t finalShape[] = {rDim, dSize};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(
                poolOutTemp[mergeOutOffset], poolOutTemp[mergeInOffset], reduceSharedTemp[bufferId], finalShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            finalOutLocation = 1; // result in row 8
        }
    }

    // Get the final accumulated sum from poolOutTemp
    // row 0 if finalOutLocation==0, row 8 if finalOutLocation==1
    uint32_t finalOffset = finalOutLocation == 0 ? 0 : FLOAT_DATA_BLOCK_NUM * dSize;
    auto sumTensor = poolOutTemp[finalOffset];

    // Step 4: Divide accumulated sum by actualLen (hardware Cast avoids prohibited int->float C++ cast)
    LocalTensor<int32_t> lenInt32Ub = reduceSharedTemp[0].ReinterpretCast<int32_t>();
    Duplicate(lenInt32Ub, static_cast<int32_t>(actualLen), static_cast<int32_t>(dSize));
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> lenFloatUb = reduceSharedTemp[0].ReinterpretCast<T>()[dSize];
    Cast(lenFloatUb, lenInt32Ub, RoundMode::CAST_NONE, dSize);
    AscendC::PipeBarrier<PIPE_V>();
    Div(poolOut, sumTensor, lenFloatUb, static_cast<int32_t>(dSize));
    AscendC::PipeBarrier<PIPE_V>();

    // Step 5: Cast FP32 -> OUT_T
    Cast(poolOutCast, poolOut, RoundMode::CAST_NONE, dSize);
    AscendC::PipeBarrier<PIPE_V>();

    eventId = bufferId == 0 ? eventPing : eventPong;
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);

    // Step 6: Write to GM
    DataCopyExtParams copyOutCastParams;
    copyOutCastParams.blockLen = dSize * sizeof(OUT_T);
    copyOutCastParams.blockCount = 1;
    copyOutCastParams.srcStride = 0;
    copyOutCastParams.dstStride = 0;
    DataCopyPad(dstGm[dstOffset], poolOutCast, copyOutCastParams);
}
#endif // BSA_VEC_POOL_SERVICE_H
