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
 * \file bsa_vec_sm_service_arch35.h
 * \brief Online Softmax 服务（融合 post pool）
 *        aiv0/aiv1 行切分由 BSAVecPostPoolService::GetChunkRowSlices 统一给出：
 *        非 post 模式为经典对半切分；post 模式按 postXBlock 组对齐（跨 chunk 片元归 aiv0），
 *        两趟 softmax 与 post pool 折叠共用同一行切分，满足 online softmax 一致性要求。
 *        二趟在归一化 FP32 tile 上直接调 post pool 折叠（post 模式不再写 attnScore GM）。
 */
#ifndef BSA_VEC_SM_SERVICE_ARCH35_H
#define BSA_VEC_SM_SERVICE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common_arch35.h"
#include "bsa_select_block_mask_tiling_data_arch35.h"
#include "bsa_vec_post_pool_service_arch35.h"
#include "bsa_vec_sm_vf.h"

// 行切片数组容量：单元数上限 + 超大单元拆分余量
static constexpr uint32_t BSA_SM_SLICES_MAX_ARCH35 = BSA_POST_POOL_UNITS_MAX + 2;

template <typename BSAT>
class BSAVecSmService {
public:
    using T = float;                    //< 中间计算类型 (float, FP32)
    using IN_T = typename BSAT::inputT; //< 输入数据类型 (FP16/BF16)
    using OUT_T = half;                 //< softmax 输出数据类型 (FP16)

    using PostRowSlice = typename BSAVecPostPoolService<BSAT>::PostRowSlice;

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;   // BNSD or TND
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV; // BNSD or TND

    __aicore__ inline BSAVecSmService(){};
    __aicore__ inline void InitParams(const BSAConstInfo &constInfo,
                                      const optiling::BSASelectBlockMaskTilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<> *uBuf_);

    __aicore__ inline void InitGM(GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<OUT_T> &attnScoreFp16Gm,
                                  GlobalTensor<OUT_T> &pooledScoreGm, GlobalTensor<T> &postStagingGm);

    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                       uint32_t kChunkSize, uint32_t validXBlocks,
                                                       uint32_t validYBlocks);

    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                    uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx,
                                                    uint32_t validXBlocks, uint32_t validYBlocks);

private:
    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData;

    static constexpr int32_t BUFFER = 1;
    uint32_t vecSubBlockIdx = 0; // vector核在aic的序号： 0 or 1

    // ---- post pool 融合服务（post 模式下二趟直接折叠）----
    BSAVecPostPoolService<BSAT> postPoolOP;

    // ---- GM 全局张量引用 ----
    GlobalTensor<OUT_T> attnScoreFp16GmTensor; //< 注意力分数 [xBlocks * yBlocks] (FP32→FP16)，非 post 模式写出
    GlobalTensor<T> scoreFp32GmTensor;         //< Softmax 临时缓冲 (FP32)
    TBuf<> *uBufPtr = nullptr;                 //< 统一 UB 缓冲句柄（post 服务在其后 carve）

    LocalTensor<uint8_t> reduceSharedTemp; // reduce相关 高阶接口的临时存储空间，暂定24k
    LocalTensor<T> tmpMax;                 //< Softmax Max (FP32) * 8 (即FLOAT_DATA_BLOCK_NUM)
    LocalTensor<T> tmpSum;                 //< Softmax Sum (FP32) * 8 (即FLOAT_DATA_BLOCK_NUM)
    LocalTensor<T> globalMax;              //< Softmax old Max (FP32) 在一轮k里会驻守 <= 128
    LocalTensor<T> globalSum;              //< Softmax old Sum (FP32) 在一轮k里会驻守 <= 128
    LocalTensor<T> globalMaxBroc;          //< Softmax Max (FP32) 在一轮k里会驻守 <= 128
    LocalTensor<T> globalSumBroc;          //< Softmax Sum (FP32) 在一轮k里会驻守 <= 128
    LocalTensor<T> diffMax;

    LocalTensor<T> scoreUb;         // out shape [x, y] < [64, 128 * 5]
    LocalTensor<OUT_T> scoreFp16Ub; // 上一轮矩阵乘结果 input [x, y] < [64, 128 * 5]

    // ---- 硬件同步事件 ----
    event_t eventSft = (event_t)3;
};

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitParams(
    const BSAConstInfo &constInfo, const optiling::BSASelectBlockMaskTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    vecSubBlockIdx = constInfo.subBlockIdx; // 0 or 1
    postPoolOP.InitParams(constInfo, tilingData);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitBuffers(TBuf<> *uBuf_)
{
    // ub 划分（softmax 相位约 162.5KB，post 相位约 11KB，合计 < 192KB）
    uint32_t ubOffset = 0;
    uint32_t chunkSize = Q_CHUNK_SIZE;

    globalMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalMaxBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSum = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSumBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    diffMax = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpSum = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t scoreElements = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO;
    scoreUb = uBuf_->GetWithOffset<T>(scoreElements, ubOffset);
    scoreFp16Ub = uBuf_->GetWithOffset<OUT_T>(scoreElements, ubOffset); // 复用fp32空间

    ubOffset += scoreElements * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t sharedReduceEles = BSAConstInfo::BUFFER_SIZE_BYTE_24K / sizeof(uint8_t);
    reduceSharedTemp = uBuf_->GetWithOffset<uint8_t>(sharedReduceEles, ubOffset);
    ubOffset += sharedReduceEles * sizeof(uint8_t);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uBufPtr = uBuf_;
    if (constInfo.usePostBlockShape) {
        // post 相位缓冲紧随 softmax 之后 carve，reduce 临时空间两相位复用
        postPoolOP.InitBuffers(uBuf_, ubOffset, reduceSharedTemp);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitGM(GlobalTensor<T> &ScoreFp32Gm, GlobalTensor<OUT_T> &attnScoreFp16Gm,
                                                     GlobalTensor<OUT_T> &pooledScoreGm, GlobalTensor<T> &postStagingGm)
{
    this->attnScoreFp16GmTensor = attnScoreFp16Gm;
    this->scoreFp32GmTensor = ScoreFp32Gm;
    if (constInfo.usePostBlockShape) {
        postPoolOP.InitGM(pooledScoreGm, postStagingGm);
    }
}

/**
 * @brief Online Softmax 第一遍——对一个 K-Chunk 的 Score 块完成局部 softmax 的 m/l 累积
 * @details 按 GetChunkRowSlices 给出的行切片逐片处理（切分与二趟/post pool 完全一致）：
 *          - 从 softmaxTmpGm（FP32）逐切片读入 score_partial 子块
 *          - m_new = max(m, row_max(score_partial))；l_new = l × exp(m - m_new) + row_sum(exp(·- m_new))
 *          - m = m_new, l = l_new 跨所有 K-Chunk 累积，供二趟归一化
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                          uint32_t kChunkStart, uint32_t kChunkSize,
                                                                          uint32_t validXBlocks, uint32_t validYBlocks)
{
    PostRowSlice slices[BSA_SM_SLICES_MAX_ARCH35];
    uint32_t sliceNum = postPoolOP.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, slices);
    if (sliceNum == 0) {
        return;
    }

    // 32字节对齐
    uint32_t kChunkSizeAlign = (kChunkSize + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM * FLOAT_DATA_BLOCK_NUM;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    // 分块连续布局：当前 kTile 基址（元素）
    uint64_t tileBase = BSAScratchTileOffset(constInfo.aicIdx, kChunkStart, constInfo.yBlocks);

    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice &sl = slices[s];

        SetFlag<HardEvent::V_MTE2>(eventSft);
        WaitFlag<HardEvent::V_MTE2>(eventSft);

        DataCopyPad(scoreUb, scoreFp32GmTensor[tileBase + sl.chunkRowStart * K_TILE_COLS],
                    {static_cast<uint16_t>(sl.rowCount), static_cast<uint32_t>(kChunkSize * sizeof(float)),
                     static_cast<uint32_t>((K_TILE_COLS - kChunkSize) * sizeof(float)), 0, 0},
                    {true, 0, static_cast<uint8_t>(kChunkSizeAlign - kChunkSize), SOFTMAX_NEG_INF});

        SetFlag<HardEvent::MTE2_V>(eventSft);
        WaitFlag<HardEvent::MTE2_V>(eventSft);

        Muls(scoreUb, scoreUb, (float)(constInfo.scaleValue), sl.rowCount * kChunkSizeAlign);
        AscendC::PipeBarrier<PIPE_V>();

        // 更new max：M = ReduceMax(score_partial)，m_new = max(old_m, M)
        uint32_t maxShape[] = {sl.rowCount, kChunkSizeAlign};
        bool isSrcInnerPad = true;

        AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(tmpMax[sl.localRowStart], scoreUb, reduceSharedTemp,
                                                                   maxShape, isSrcInnerPad);
        AscendC::PipeBarrier<PIPE_V>();

        if (kChunkStart != 0) {
            AscendC::Max(tmpMax[sl.localRowStart], tmpMax[sl.localRowStart], globalMax[sl.localRowStart], sl.rowCount);
        } else {
            // 第一次k循环初始化
            AscendC::Duplicate(globalMax[sl.localRowStart], float(SOFTMAX_NEG_INF), sl.rowCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(globalSum[sl.localRowStart], float(0.0f), sl.rowCount);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // step 1: t3 = global_sum * exp(global_max - newMax)
        AscendC::Sub(diffMax[sl.localRowStart], globalMax[sl.localRowStart], tmpMax[sl.localRowStart], sl.rowCount);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Exp(diffMax[sl.localRowStart], diffMax[sl.localRowStart], sl.rowCount);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(globalSum[sl.localRowStart], globalSum[sl.localRowStart], diffMax[sl.localRowStart], sl.rowCount);
        AscendC::PipeBarrier<PIPE_V>();

        // step 2: global_sum += exp(score - newMax).sum(dim=-1)
        uint32_t brocRepeat = BSACeilDiv(sl.rowCount, static_cast<uint32_t>(BROC_BASE_NUM));
        Brcb(globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], tmpMax[sl.localRowStart], brocRepeat, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();

        BSASubRowBroadcast(scoreUb, scoreUb, globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                           kChunkSizeAlign);

        AscendC::Exp(scoreUb, scoreUb, sl.rowCount * kChunkSizeAlign);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t sumShape[] = {sl.rowCount, kChunkSizeAlign};
        if (kChunkSizeAlign <= FLOAT_REPEAT_NUM) {
            // Set the mask in the instruction instead of relying on the high-level reuse implementation's mask state.
            AscendC::WholeReduceSum<T, true>(tmpSum[sl.localRowStart], scoreUb, static_cast<int32_t>(kChunkSizeAlign),
                                             static_cast<int32_t>(sl.rowCount), 1, 1,
                                             static_cast<int32_t>(kChunkSizeAlign / FLOAT_DATA_BLOCK_NUM));
        } else {
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(tmpSum[sl.localRowStart], scoreUb,
                                                                      reduceSharedTemp, sumShape, isSrcInnerPad);
        }
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Add(globalSum[sl.localRowStart], tmpSum[sl.localRowStart], globalSum[sl.localRowStart], sl.rowCount);
        AscendC::PipeBarrier<PIPE_V>();

        // 更新globalMax
        AscendC::Max(globalMax[sl.localRowStart], tmpMax[sl.localRowStart], globalMax[sl.localRowStart], sl.rowCount);
    }

    // 最后一次k循环存储sumbroc（供二趟归一化），覆盖本 AIV 全部行
    if (kChunkStart + kChunkSize >= validYBlocks) {
        uint32_t rowsTotal = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
        uint32_t brocRepeat = BSACeilDiv(rowsTotal, static_cast<uint32_t>(BROC_BASE_NUM));
        Brcb(globalSumBroc, globalSum, brocRepeat, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
}

/**
 * @brief Softmax 第二遍：逐切片读入 score、归一化；post 模式直接在 UB tile 上做 post pool 折叠，
 *        非 post 模式 Cast FP16 后写出 attnScore GM
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                       uint32_t kChunkStart, uint32_t kChunkSize,
                                                                       uint32_t batchIdx, uint32_t headIdx,
                                                                       uint32_t validXBlocks, uint32_t validYBlocks)
{
    PostRowSlice slices[BSA_SM_SLICES_MAX_ARCH35];
    uint32_t sliceNum = postPoolOP.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, slices);
    if (sliceNum == 0) {
        return;
    }

    bool isLastKTile = (kChunkStart + kChunkSize >= validYBlocks);
    if (constInfo.usePostBlockShape && kChunkStart == 0) {
        postPoolOP.BeginChunk(batchIdx, headIdx, qChunkStart, qChunkSize, validXBlocks, validYBlocks);
    }

    // 32字节对齐 (FP32)
    uint32_t kChunkSizeAlign32 = (kChunkSize + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM * FLOAT_DATA_BLOCK_NUM;
    // 32字节对齐 (FP16)
    uint32_t kChunkSizeAlign16 = (kChunkSize + FP16_DATA_BLOCK_NUM - 1) / FP16_DATA_BLOCK_NUM * FP16_DATA_BLOCK_NUM;
    uint32_t kChunkSizeAlign = kChunkSizeAlign16;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    // 分块连续布局：当前 kTile 基址（元素）
    uint64_t tileBase = BSAScratchTileOffset(constInfo.aicIdx, kChunkStart, constInfo.yBlocks);

    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice &sl = slices[s];

        SetFlag<HardEvent::V_MTE2>(eventSft);
        WaitFlag<HardEvent::V_MTE2>(eventSft);

        if (kChunkSizeAlign32 * sizeof(IN_T) % VEC_ALIGN_SIZE == 0) {
            // CAST TO FP16时候保持32字节对齐
            DataCopyPad(scoreUb, scoreFp32GmTensor[tileBase + sl.chunkRowStart * K_TILE_COLS],
                        {static_cast<uint16_t>(sl.rowCount), static_cast<uint32_t>(kChunkSize * sizeof(float)),
                         static_cast<uint32_t>((K_TILE_COLS - kChunkSize) * sizeof(float)), 0, 0},
                        {true, 0, static_cast<uint8_t>(kChunkSizeAlign32 - kChunkSize), SOFTMAX_NEG_INF});
        } else {
            // CAST TO FP16时候未保持32字节对齐，需要加一个1个fp32的block
            DataCopyPad(scoreUb, scoreFp32GmTensor[tileBase + sl.chunkRowStart * K_TILE_COLS],
                        {static_cast<uint16_t>(sl.rowCount), static_cast<uint32_t>(kChunkSize * sizeof(float)),
                         static_cast<uint32_t>((K_TILE_COLS - kChunkSize) * sizeof(float)), 1, 0},
                        {true, 0, static_cast<uint8_t>(kChunkSizeAlign32 - kChunkSize), SOFTMAX_NEG_INF});
        }

        SetFlag<HardEvent::MTE2_V>(eventSft);
        WaitFlag<HardEvent::MTE2_V>(eventSft);

        Muls(scoreUb, scoreUb, (float)(constInfo.scaleValue), sl.rowCount * kChunkSizeAlign);
        AscendC::PipeBarrier<PIPE_V>();

        BSASubRowBroadcast(scoreUb, scoreUb, globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                           kChunkSizeAlign);

        AscendC::Exp(scoreUb, scoreUb, sl.rowCount * kChunkSizeAlign);
        AscendC::PipeBarrier<PIPE_V>();

        BSADivRowBroadcast(scoreUb, scoreUb, globalSumBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                           kChunkSizeAlign);

        if (constInfo.usePostBlockShape) {
            // 融合路径：归一化 FP32 tile 直接折叠为 pooledScore
            postPoolOP.FoldNormalizedSlice(scoreUb, sl, kChunkStart, kChunkSize, kChunkSizeAlign, isLastKTile);
        } else {
            AscendC::Cast(scoreFp16Ub, scoreUb, AscendC::RoundMode::CAST_ROUND, sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(eventSft);
            WaitFlag<HardEvent::V_MTE3>(eventSft);

            uint32_t sftOutQRow = qChunkStart + sl.chunkRowStart;
            uint64_t attnScoreOffset = static_cast<uint64_t>(sftOutQRow) * validYBlocks + kChunkStart;
            GlobalTensor<OUT_T> attnScoreChunkGm;
            attnScoreChunkGm.SetGlobalBuffer((__gm__ OUT_T *)(attnScoreFp16GmTensor.GetPhyAddr() + attnScoreOffset));
            DataCopyPad(attnScoreChunkGm[0], scoreFp16Ub,
                        {static_cast<uint16_t>(sl.rowCount), static_cast<uint32_t>(kChunkSize * sizeof(OUT_T)), 0,
                         static_cast<uint32_t>((validYBlocks - kChunkSize) * sizeof(OUT_T)), 0});
        }
    }

    if (constInfo.usePostBlockShape) {
        postPoolOP.FlushTile();
    }
}

#endif // BSA_VEC_SM_SERVICE_ARCH35_H
