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
 * \file bsa_vec_sm_service.h
 * \brief Online Softmax 服务（融合 post pool）
 *        aiv0/aiv1 行切分由 BSAVecPostPoolService::GetChunkRowSlices 统一给出：
 *        非 post 模式为经典对半切分；post 模式按 postXBlock 组对齐（跨 chunk 片元归 aiv0），
 *        两趟 softmax 与 post pool 折叠共用同一行切分，满足 online softmax 一致性要求。
 *        二趟在归一化 FP32 tile 上直接调 post pool 折叠（post 模式不再写 attnScore GM）。
 */
#ifndef BSA_VEC_SM_SERVICE_H
#define BSA_VEC_SM_SERVICE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"
#include "bsa_vec_post_pool_service.h"

// 行切片数组容量：单元数上限 + 超大单元拆分余量
static constexpr uint32_t BSA_SM_SLICES_MAX = BSA_POST_POOL_UNITS_MAX + 2;

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
                                  GlobalTensor<OUT_T> &pooledScoreGm, GlobalTensor<T> &postStagingGm,
                                  GlobalTensor<T> &partSumGm, GlobalTensor<T> &partMaxGm);
    // softmax 单趟化收尾：per-tile 部分和 × exp(m_i−m_c) / l_c → 行折叠 → pooledScore（替代 pass2）
    __aicore__ inline void SoftmaxFinalize(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t batchIdx,
                                           uint32_t headIdx, uint32_t validXBlocks, uint32_t validYBlocks);

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

    /**
     * @brief NZ fractal 布局 → ND 行主序 UB 的重排读取
     * @details AIC 侧 Fixpipe CFG_NZ 直拷后，scratch tile 为 NZ 布局：
     *          线性地址 = 列块b(0..31)×2048 + 行r×16 + 块内列c(0..15)，对应逻辑列 b*16+c。
     *          本函数按列块搬运：每个列块一次 DataCopyPad（blockCount=行数、blockLen=16列=64B），
     *          源行距(2048-16)*4B、目的行距(ubRowWidth-16)*4B，把 fractal 列段拼回行主序。
     *          尾列块（kChunkSize%16≠0）同样整读 16 列，padding 列由调用方掩 NEG_INF。
     * @param[in/out] dstUb   目的 UB（行主序，行宽 ubRowWidth，≥16 对齐）
     * @param[in] tileBase    本 kTile 在 scratch 的基址（元素）
     * @param[in] rowStart    本切片起始行（tile 内行号）
     * @param[in] rowCount    本切片行数
     * @param[in] validCols   有效列数（kChunkSize）
     * @param[in] ubRowWidth  UB 行宽（对齐后，≥validCols，16 的倍数）
     */
    __aicore__ inline void NzToNdCopyIn(LocalTensor<T> &dstUb, uint64_t tileBase, uint32_t rowStart, uint32_t rowCount,
                                        uint32_t validCols, uint32_t ubRowWidth);

    // 行归约三级级联（BlockReduce 8→1：512→64→8→1），对照 PFA Rowsum<float,SPEC_TILE_512> 参数模式
    __aicore__ inline void RowMaxCascade(LocalTensor<T> &dstRow, LocalTensor<T> &srcTile, uint32_t rows,
                                         uint32_t rowStride, LocalTensor<uint8_t> &tmp);
    __aicore__ inline void RowSumCascade(LocalTensor<T> &dstRow, LocalTensor<T> &srcTile, uint32_t rows,
                                         uint32_t rowStride, LocalTensor<uint8_t> &tmp);

    // ---- post pool 融合服务（post 模式下二趟直接折叠）----
    BSAVecPostPoolService<BSAT> postPoolOP;

    // ---- GM 全局张量引用 ----
    GlobalTensor<OUT_T> attnScoreFp16GmTensor; //< 注意力分数 [xBlocks * yBlocks] (FP32→FP16)，非 post 模式写出
    GlobalTensor<T> scoreFp32GmTensor;         //< Softmax 临时缓冲 (FP32)
    GlobalTensor<T> partSumGmTensor;           //< softmax 单趟化：per-tile 列组部分和
    GlobalTensor<T> partMaxGmTensor;           //< softmax 单趟化：per-tile 局部行 max
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
                                                     GlobalTensor<OUT_T> &pooledScoreGm, GlobalTensor<T> &postStagingGm,
                                                     GlobalTensor<T> &partSumGm, GlobalTensor<T> &partMaxGm)
{
    this->attnScoreFp16GmTensor = attnScoreFp16Gm;
    this->scoreFp32GmTensor = ScoreFp32Gm;
    this->partSumGmTensor = partSumGm;
    this->partMaxGmTensor = partMaxGm;
    if (constInfo.usePostBlockShape) {
        postPoolOP.InitGM(pooledScoreGm, postStagingGm);
    }
}

/**
 * @brief NZ fractal 布局 → ND 行主序 UB 的重排读取（列块外层版）
 * @details AIC 侧 Fixpipe CFG_NZ 直拷后，scratch tile 为 NZ 布局：
 *          逻辑(行r, 列j) 位于 tileBase + (j/16)*2048 + r*16 + (j%16)
 *          （每个 fractal 列块 = 128 行 × 16 列行主序连续，列块间跳 2048 元素）。
 *          按列块搬运：每个列块一次 DataCopyPad，源为"切片行带 × 16 列"的连续 run
 *          （blockCount=行数、blockLen=16列=64B、srcStride=0），目的按行主序拼回
 *          （dstStride=(行宽-16)*4B/32B，DataCopyExtParams 的 dstStride 为 32B 块单位）。
 *          尾列块（validCols%16≠0）同样整读 16 列，padding 列由调用方掩 NEG_INF。
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::NzToNdCopyIn(LocalTensor<T> &dstUb, uint64_t tileBase, uint32_t rowStart,
                                                           uint32_t rowCount, uint32_t validCols, uint32_t ubRowWidth)
{
    constexpr uint32_t NZ_FRAC_COLS = 16;                            // fractal 列宽
    constexpr uint32_t NZ_BLOCK_ELEMS = Q_CHUNK_SIZE * NZ_FRAC_COLS; // 列块元素数 = 128*16 = 2048
    uint32_t blockNum = (validCols + NZ_FRAC_COLS - 1) / NZ_FRAC_COLS;
    for (uint32_t b = 0; b < blockNum; b++) {
        DataCopyPad(dstUb[b * NZ_FRAC_COLS],
                    scoreFp32GmTensor[tileBase + static_cast<uint64_t>(b) * NZ_BLOCK_ELEMS +
                                      static_cast<uint64_t>(rowStart) * NZ_FRAC_COLS],
                    {static_cast<uint16_t>(rowCount), static_cast<uint32_t>(NZ_FRAC_COLS * sizeof(float)), 0,
                     static_cast<uint32_t>((ubRowWidth - NZ_FRAC_COLS) * sizeof(float) / 32), 0},
                    {false, 0, 0, 0});
    }
}


/**
 * @brief 行方向归约三级级联（BlockReduce 8→1）：[rows × 512] → [rows × 1]
 * @details 对照 prompt_flash_attention Rowsum<float,SPEC_TILE_512>：
 *          L1: [rows×512] 组内 8→1 → [rows×64]；L2: → [rows×8]；L3: → [rows×1]
 *          tmp 布局（float 元素）：[0, rows*64) L1 结果，[rows*64, rows*72) L2 结果（≤24KB reduceSharedTemp）
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::RowMaxCascade(LocalTensor<T> &dstRow, LocalTensor<T> &srcTile,
                                                            uint32_t rows, uint32_t rowStride,
                                                            LocalTensor<uint8_t> &tmpBytes)
{
    // BlockReduce(vcgmax) 硬件语义：每 repeat 64 元素 → 8 个组内(8:1)结果
    //（对照 PFA FLOAT_VECTOR_SIZE=64 的 Rowsum 级联系数），三级 512→64→8→1
    constexpr uint32_t VEC = 64;                                        // PFA FLOAT_VECTOR_SIZE
    constexpr uint32_t REDUCE_WIDTH = K_TILE_COLS;                      // 512
    LocalTensor<T> tmp = tmpBytes.ReinterpretCast<T>();
    const uint32_t l1Elems = rows * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM); // rows×64（L1 输出元素数）
    AscendC::BlockReduceMax<float, false>(tmp, srcTile, rows * rowStride / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(tmp[l1Elems], tmp, l1Elems / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(dstRow, tmp[l1Elems], rows * FLOAT_DATA_BLOCK_NUM / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::RowSumCascade(LocalTensor<T> &dstRow, LocalTensor<T> &srcTile,
                                                            uint32_t rows, uint32_t rowStride,
                                                            LocalTensor<uint8_t> &tmpBytes)
{
    constexpr uint32_t VEC = 64;
    constexpr uint32_t REDUCE_WIDTH = K_TILE_COLS; // 512
    LocalTensor<T> tmp = tmpBytes.ReinterpretCast<T>();
    const uint32_t l1Elems = rows * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM); // rows×64
    AscendC::BlockReduceSum<float, false>(tmp, srcTile, rows * rowStride / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum<float, false>(tmp[l1Elems], tmp, l1Elems / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum<float, false>(dstRow, tmp[l1Elems], rows * FLOAT_DATA_BLOCK_NUM / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
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
    PostRowSlice slices[BSA_SM_SLICES_MAX];
    uint32_t sliceNum = postPoolOP.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, slices);
    if (sliceNum == 0) {
        return;
    }

    // NZ 读法行宽：对齐到 fractal 列宽 16（≥16），尾块 padding 段统一掩 SOFTMAX_NEG_INF（数学无害）
    uint32_t kChunkSizeAlign = (kChunkSize + 15) / 16 * 16;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    // 分块连续布局：当前 kTile 基址（元素）
    uint64_t tileBase = BSAScratchTileOffset(constInfo.aicIdx, kChunkStart, constInfo.yBlocks);

    // ---- band 整读 + 逐 slice 计算 ----
    // 连续 slice 合并为 band 一次读入（32 列块 × blockCount=bandRows 连续 run，burst 大），
    // band 行数受 scoreUb 行容量约束（对齐场景整子核 64 行 = 1 band；不对齐片元场景自动多 band）
    uint32_t bandCap = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO / kChunkSizeAlign;
    uint32_t s = 0;
    while (s < sliceNum) {
        uint32_t e = s + 1;
        while (e < sliceNum) {
            bool contiguous = (slices[e].chunkRowStart == slices[e - 1].chunkRowStart + slices[e - 1].rowCount) &&
                              (slices[e].localRowStart == slices[e - 1].localRowStart + slices[e - 1].rowCount);
            uint32_t mergedRows =
                slices[e - 1].localRowStart + slices[e - 1].rowCount - slices[s].localRowStart + slices[e].rowCount;
            if (!contiguous || mergedRows > bandCap) {
                break;
            }
            e++;
        }
        const uint32_t bandLocalStart = slices[s].localRowStart;
        const uint32_t bandChunkStart = slices[s].chunkRowStart;
        const uint32_t bandRows = slices[e - 1].localRowStart + slices[e - 1].rowCount - bandLocalStart;

        SetFlag<HardEvent::V_MTE2>(eventSft);
        WaitFlag<HardEvent::V_MTE2>(eventSft);

        // NZ 重排读：fractal 列块布局 → 行主序 UB（band 整读，列块外层）
        NzToNdCopyIn(scoreUb, tileBase, bandChunkStart, bandRows, kChunkSize, kChunkSizeAlign);

        SetFlag<HardEvent::MTE2_V>(eventSft);
        WaitFlag<HardEvent::MTE2_V>(eventSft);

        // 尾块掩码：AIC 侧 nSize 对齐 16 写出的 padding 列是 L0C 残留，必须掩成 NEG_INF
        if (kChunkSize < kChunkSizeAlign) {
            for (uint32_t r = 0; r < bandRows; r++) {
                AscendC::Duplicate(scoreUb[static_cast<uint32_t>(r) * kChunkSizeAlign + kChunkSize],
                                   float(SOFTMAX_NEG_INF), kChunkSizeAlign - kChunkSize);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t i = s; i < e; i++) {
            const PostRowSlice &sl = slices[i];
            LocalTensor<T> tile = scoreUb[(sl.localRowStart - bandLocalStart) * kChunkSizeAlign];

            Muls(tile, tile, (float)(constInfo.scaleValue), sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();

            // 行归约：全 tile(512) 走 BlockReduce 三级级联（快），尾 tile 走 AR 原路径
            uint32_t maxShape[] = {sl.rowCount, kChunkSizeAlign};
            bool isSrcInnerPad = true;

            if (kChunkSizeAlign == K_TILE_COLS) {
                LocalTensor<T> dstMax = tmpMax[sl.localRowStart];
                RowMaxCascade(dstMax, tile, sl.rowCount, kChunkSizeAlign, reduceSharedTemp);
            } else {
                AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(tmpMax[sl.localRowStart], tile,
                                                                           reduceSharedTemp, maxShape, isSrcInnerPad);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if (kChunkStart != 0) {
                AscendC::Max(tmpMax[sl.localRowStart], tmpMax[sl.localRowStart], globalMax[sl.localRowStart],
                             sl.rowCount);
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

            AscendC::Mul(globalSum[sl.localRowStart], globalSum[sl.localRowStart], diffMax[sl.localRowStart],
                         sl.rowCount);
            AscendC::PipeBarrier<PIPE_V>();

            // step 2: global_sum += exp(score - newMax).sum(dim=-1)
            uint32_t brocRepeat = BSACeilDiv(sl.rowCount, static_cast<uint32_t>(BROC_BASE_NUM));
            Brcb(globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], tmpMax[sl.localRowStart], brocRepeat, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();

            BSASubRowBroadcast(tile, tile, globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                               kChunkSizeAlign);

            AscendC::Exp(tile, tile, sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();

            if (kChunkSizeAlign == K_TILE_COLS) {
                LocalTensor<T> dstSum = tmpSum[sl.localRowStart];
                RowSumCascade(dstSum, tile, sl.rowCount, kChunkSizeAlign, reduceSharedTemp);
            } else {
                uint32_t sumShape[] = {sl.rowCount, kChunkSizeAlign};
                if (kChunkSizeAlign <= FLOAT_REPEAT_NUM) {
                    AscendC::WholeReduceSum<T, true>(tmpSum[sl.localRowStart], tile,
                                                     static_cast<int32_t>(kChunkSizeAlign),
                                                     static_cast<int32_t>(sl.rowCount), 1, 1,
                                                     static_cast<int32_t>(kChunkSizeAlign / FLOAT_DATA_BLOCK_NUM));
                } else {
                    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, true>(tmpSum[sl.localRowStart], tile,
                                                                              reduceSharedTemp, sumShape,
                                                                              isSrcInnerPad);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::Add(globalSum[sl.localRowStart], tmpSum[sl.localRowStart], globalSum[sl.localRowStart],
                         sl.rowCount);
            AscendC::PipeBarrier<PIPE_V>();

            // 更新globalMax
            AscendC::Max(globalMax[sl.localRowStart], tmpMax[sl.localRowStart], globalMax[sl.localRowStart],
                         sl.rowCount);

            // ---- softmax 单趟化：本 tile 的列组部分和 reduce_sum1 + 局部 max 写出到 GM 槽 ----
            // partSum 槽布局：[aivIdx*KTILES + tileIdx][128 行][32 列组]；partMax 槽：[..][128]
            {
                const uint32_t KTILES = (constInfo.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
                const uint32_t tileIdx = kChunkStart / K_TILE_COLS;
                // partSum 槽：[槽][128 行][32 组]（行主序，每行 32 组 fp32 连续）；partMax 槽：[槽][行]
                const uint64_t psSlotBase =
                    (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + tileIdx) * Q_CHUNK_SIZE * 32;
                const uint64_t pmSlotBase =
                    (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + tileIdx) * Q_CHUNK_SIZE;
                // GM 槽布局（行主序）[槽][128 行][32 组]。两步折叠（原语语义全透明，每 slice 共 2 次调用/16 行）：
                //   Step A 块配对 Add：foldMid[g][j] = tile[16g+j] + tile[16g+8+j]（16 列组 = 2 相邻块）
                //   Step B BlockReduceSum 8:1：foldOut[r][g] = Σ_j foldMid[r][g][j]（稠密 dst）
                LocalTensor<T> groupTmp = reduceSharedTemp.ReinterpretCast<T>();
                LocalTensor<T> foldOut = groupTmp[4096]; // 行主序 [rows][32]，≤ 64*32 = 2048 float
                // 无效组清零：GM 槽行宽恒 32，收尾仅消费有效组列
                AscendC::Duplicate(foldOut, float(0.0f), 32 * sl.rowCount);
                AscendC::PipeBarrier<PIPE_V>();
                {
                    // 尾 tile（行宽 < 128）需先整形成 64 列虚宽行布局（块粒度跨距 DataCopy），
                    // 否则 128 列虚宽假设跨行错位；整形区 32-63 列为残留，只流入无效组（无消费方）
                    LocalTensor<T> foldSrc = tile;
                    uint32_t srcStrideCols = kChunkSizeAlign;
                    LocalTensor<T> foldMid = groupTmp[0]; // 全 tile：[16 行][32 组][8 lane] = 4096 float
                    if (kChunkSizeAlign < 128) {
                        LocalTensor<T> reshapeDst = groupTmp[0];     // [rows][64]
                        foldMid = groupTmp[1024];                    // [16 行][4 组][8 lane] = 512 float
                        AscendC::DataCopy(reshapeDst, tile,
                                          {static_cast<uint16_t>(sl.rowCount), 4, 0, 4});
                        AscendC::PipeBarrier<PIPE_ALL>();
                        foldSrc = reshapeDst;
                        srcStrideCols = 64;
                    }
                    const uint32_t kColsVirtual = srcStrideCols;
                    const uint32_t groupsVirtual = kColsVirtual / 16;
                    AscendC::BinaryRepeatParams bp;
                    bp.dstBlkStride = 1;
                    bp.src0BlkStride = 2;
                    bp.src1BlkStride = 2;
                    bp.dstRepStride = 8;
                    bp.src0RepStride = 16;
                    bp.src1RepStride = 16;
                    for (uint32_t rb = 0; rb < sl.rowCount; rb += 16) {
                        const uint32_t rc = BSAMin(uint32_t(16), sl.rowCount - rb);
                        const uint32_t repA = rc * groupsVirtual / 8; // 尾 tile 需 rc 为偶数（本 case 恒 16）
                        AscendC::Add<T, false>(foldMid, foldSrc[rb * kColsVirtual],
                                               foldSrc[rb * kColsVirtual + 8], static_cast<uint64_t>(0),
                                               static_cast<uint8_t>(repA), bp);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::BlockReduceSum<float, false>(foldOut[rb * 32], foldMid, groupsVirtual * rc / 8, 0,
                                                               1, 1, 8);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                // V→MTE3 同步必须在写出【之前】（规范顺序，对照旧 pass2）：
                // MTE3 读 foldOut/tmpMax 前须等本 slice 全部 V 计算落定，否则竞态读到前缀+零值
                SetFlag<HardEvent::V_MTE3>(eventSft);
                WaitFlag<HardEvent::V_MTE3>(eventSft);
                // 整 slice 一次写出（GM 侧 chunk 行连续；blockLen 字节单位）
                DataCopyPad(partSumGmTensor[psSlotBase + sl.chunkRowStart * 32], foldOut,
                            {1, static_cast<uint16_t>(32 * sl.rowCount * sizeof(float)), 0, 0});
                // 写出 partMax：rows 个 fp32 连续（blockLen 单位为字节）
                DataCopyPad(partMaxGmTensor[pmSlotBase + sl.chunkRowStart], tmpMax[sl.localRowStart],
                            {1, static_cast<uint16_t>(sl.rowCount * sizeof(float)), 0, 0});
                // WAR 保护：MTE3 写出读 foldOut/tmpMax 期间，下一 slice 的 Duplicate/Add 不得先行覆写
                SetFlag<HardEvent::MTE3_V>(eventSft);
                WaitFlag<HardEvent::MTE3_V>(eventSft);
            }
        }
        s = e;
    }

    // 最后一次k循环存储sumbroc（供收尾归一化），覆盖本 AIV 全部行
    if (kChunkStart + kChunkSize >= validYBlocks) {
        uint32_t rowsTotal = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
        uint32_t brocRepeat = BSACeilDiv(rowsTotal, static_cast<uint32_t>(BROC_BASE_NUM));
        Brcb(globalSumBroc, globalSum, brocRepeat, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
}


/**
 * @brief softmax 单趟化收尾（替代 SoftmaxSecondPassAndCast 的 post 模式主路径）
 * @details 数学：output[px,py] = Σ_{r∈px} p_i[r,g]·exp(m_i[r]−m_c[r]) / l_c[r]
 *          其中 p_i = per-tile 列组部分和（pass1 已写出 GM），i 由 py 决定；
 *          乘法指数律严格等价于 Σ exp(x−m_c)/l_c 的 post 折叠（见推导）。
 *          非 post 模式仍走 SoftmaxSecondPassAndCast 原路径。
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxFinalize(uint32_t qChunkStart, uint32_t qChunkSize,
                                                              uint32_t batchIdx, uint32_t headIdx,
                                                              uint32_t validXBlocks, uint32_t validYBlocks)
{
    if (!constInfo.usePostBlockShape) {
        return; // 非 post 模式由调用方走原 pass2
    }
    GlobalTensor<OUT_T> pooledScoreGmTensor = postPoolOP.GetPooledScoreGm();
    const uint32_t validPostYCur = (validYBlocks + constInfo.postBlockShapeY - 1) / constInfo.postBlockShapeY;
    PostRowSlice slices[BSA_SM_SLICES_MAX];
    uint32_t sliceNum = postPoolOP.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, slices);
    if (sliceNum == 0) {
        return;
    }
    const uint32_t KTILES = (constInfo.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    const uint32_t unitRows = constInfo.postBlockShapeX; // 16
    // pooled 头基址（[B, N, postX, postY] 展开）
    const uint64_t pooledHeadOffset = (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) *
                                      constInfo.postXBlocks * constInfo.postYBlocks;

    // 共享临时区规划（fp32 元素，GetReduceTemp = reduceSharedTemp 24KB = 6144 float）：
    //   [0,32) fold acc；[64,192) fp16Row[8 unit][32 half]（Cast 目标槽）；[192,224) invCols[32]
    //   [1024,2048) RA 归约 tmp；[4096,4224) corrAll[128]；[4224,4352) diffSlice[128]
    //   [5120,6144) corrBroc[128*8]（Brcb 行广播布局）
    LocalTensor<T> rs = postPoolOP.GetReduceTemp();
    LocalTensor<T> acc = rs[0];
    LocalTensor<OUT_T> fp16Row = rs[64].ReinterpretCast<OUT_T>();
    LocalTensor<uint8_t> raTmp = rs[1024].ReinterpretCast<uint8_t>();
    LocalTensor<T> corrAll = rs[4096];
    LocalTensor<T> diffSlice = rs[4224];
    LocalTensor<T> psUb = scoreUb;
    LocalTensor<T> pmUb = tmpMax; // [128] 复用（chunk 行序的 m_t）

    for (uint32_t t = 0; t < KTILES; t++) {
        const uint32_t colLo = t * K_TILE_COLS;
        const uint32_t groupsValid = BSAMin(K_TILE_COLS, constInfo.yBlocks - colLo) / 16;
        if (groupsValid == 0) {
            continue;
        }
        const uint64_t slotBase = (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + t) * Q_CHUNK_SIZE * 32;
        const uint64_t pmSlotBase = (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + t) * Q_CHUNK_SIZE;

        // —— 读 partSum 槽（16KB 行主序 [128 行][32 组]，4x4KB）+ partMax 槽（512B）——
        // 注意：DataCopyPad 的 blockLen 单位为【字节】（32B 块语义仅适用于非 pad 版 DataCopy）
        SetFlag<HardEvent::MTE3_MTE2>(eventSft);
        WaitFlag<HardEvent::MTE3_MTE2>(eventSft);
        SetFlag<HardEvent::V_MTE2>(eventSft);
        WaitFlag<HardEvent::V_MTE2>(eventSft);
        DataCopyPad(psUb, partSumGmTensor[slotBase], {1, 4096, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(psUb[1024], partSumGmTensor[slotBase + 1024], {1, 4096, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(psUb[2048], partSumGmTensor[slotBase + 2048], {1, 4096, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(psUb[3072], partSumGmTensor[slotBase + 3072], {1, 4096, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(pmUb, partMaxGmTensor[pmSlotBase], {1, static_cast<uint16_t>(Q_CHUNK_SIZE * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(eventSft);
        WaitFlag<HardEvent::MTE2_V>(eventSft);

        // —— 修正系数：corr[chunkRow] = exp(m_t[chunkRow] - m_c[localRow]) / l_c[localRow] ——
        // pass1 语义：partSum 槽 = Σ exp(x - m_t)（m_t 为截至本 tile 的 running max）、partMax 槽 = m_t、
        // globalMax/globalSum = 最终 m_c / l_c；乘 corr 后 = Σ exp(x - m_c) / l_c（指数律严格等价）。
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice &sl = slices[s];
            AscendC::Sub(diffSlice, pmUb[sl.chunkRowStart], globalMax[sl.localRowStart], sl.rowCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(diffSlice, diffSlice, sl.rowCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(corrAll[sl.chunkRowStart], diffSlice, globalSum[sl.localRowStart], sl.rowCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
        // corr 行广播乘入：ps[r][0..32) *= corr[r]（Brcb 行广播 + BSAvRowBroadcastMul 全槽一次；
        // 非本 AIV 的行 corrAll 为陈旧值，但其 psUb 行同样不会被折叠，结果无消费方，安全）
        LocalTensor<T> corrBroc = rs[5120];
        Brcb(corrBroc, corrAll, Q_CHUNK_SIZE / FLOAT_DATA_BLOCK_NUM, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        BSAvRowBroadcastMul(psUb, corrBroc, Q_CHUNK_SIZE, 32);
        AscendC::PipeBarrier<PIPE_V>();

        // —— tile 级除数向量：invCols[g] = 1/组内列数 ——
        // golden 语义 pooled = ΣΣP / (rowsInPx × colsInG)（对照旧 pass2 SegmentMulsCastOut invCount）
        LocalTensor<T> invCols = rs[192];
        const uint32_t py0 = t * (K_TILE_COLS / 16);
        if ((py0 + groupsValid) * 16 <= validYBlocks) {
            AscendC::Duplicate(invCols, float(1.0f / 16.0f), 32); // 全 full 组，向量化快路径
        } else {
            for (uint32_t g = 0; g < 32; g++) {
                uint32_t colsG = BSAMin(uint32_t(16), validYBlocks - (py0 + g) * 16);
                invCols.SetValue(g, 1.0f / static_cast<float>(static_cast<int32_t>(colsG == 0 ? 16 : colsG)));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        // —— 逐 unit：RA 单调用行折叠（旧 pass2 同款原语）→ 行/列均值 → Cast 各 unit 槽 ——
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice &sl = slices[s];
            for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                const uint32_t cnt = BSAMin(unitRows, sl.rowCount - uo);
                const uint32_t chunkRow = sl.chunkRowStart + uo;
                const uint32_t px = (qChunkStart + chunkRow) / unitRows;
                const uint32_t rowsInPx = BSAMin(unitRows, validXBlocks - px * unitRows);
                uint32_t uShape[] = {cnt, 32};
                AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(acc, psUb[chunkRow * 32], raTmp, uShape,
                                                                           true);
                AscendC::PipeBarrier<PIPE_V>();
                int32_t rowsInPxI = static_cast<int32_t>(rowsInPx);
                float invRows = 1.0f / static_cast<float>(rowsInPxI);
                AscendC::Muls(acc, acc, invRows, 32); // 行均值
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(acc, acc, invCols, 32); // 组内列均值
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(fp16Row[(chunkRow / unitRows) * 32], acc, AscendC::RoundMode::CAST_ROUND, 32);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        // 批量写出：V→MTE3 一次同步 + 逐 unit 写（尾 tile 仅写有效组防越界）+ MTE3→V WAR 收尾
        SetFlag<HardEvent::V_MTE3>(eventSft);
        WaitFlag<HardEvent::V_MTE3>(eventSft);
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice &sl = slices[s];
            for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                const uint32_t chunkRow = sl.chunkRowStart + uo;
                const uint32_t px = (qChunkStart + chunkRow) / unitRows;
                const uint64_t gmCol =
                    pooledHeadOffset + static_cast<uint64_t>(px) * validPostYCur + t * (K_TILE_COLS / 16);
                DataCopyPad(pooledScoreGmTensor[gmCol], fp16Row[(chunkRow / unitRows) * 32],
                            {1, static_cast<uint16_t>(groupsValid * sizeof(OUT_T)), 0, 0});
            }
        }
        SetFlag<HardEvent::MTE3_V>(eventSft);
        WaitFlag<HardEvent::MTE3_V>(eventSft);
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
    PostRowSlice slices[BSA_SM_SLICES_MAX];
    uint32_t sliceNum = postPoolOP.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, slices);
    if (sliceNum == 0) {
        return;
    }

    bool isLastKTile = (kChunkStart + kChunkSize >= validYBlocks);
    if (constInfo.usePostBlockShape && kChunkStart == 0) {
        postPoolOP.BeginChunk(batchIdx, headIdx, qChunkStart, qChunkSize, validXBlocks, validYBlocks);
    }

    // NZ 读法行宽：对齐到 fractal 列宽 16（与 FP16_DATA_BLOCK_NUM=16 对齐值一致），
    // 尾块 padding 段统一掩 SOFTMAX_NEG_INF（数学无害：exp→0 对折叠无贡献）
    uint32_t kChunkSizeAlign = (kChunkSize + 15) / 16 * 16;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    // 分块连续布局：当前 kTile 基址（元素）
    uint64_t tileBase = BSAScratchTileOffset(constInfo.aicIdx, kChunkStart, constInfo.yBlocks);

    // ---- band 整读 + 逐 slice 计算（与 pass1 同结构）----
    uint32_t bandCap = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO / kChunkSizeAlign;
    uint32_t s = 0;
    while (s < sliceNum) {
        uint32_t e = s + 1;
        while (e < sliceNum) {
            bool contiguous = (slices[e].chunkRowStart == slices[e - 1].chunkRowStart + slices[e - 1].rowCount) &&
                              (slices[e].localRowStart == slices[e - 1].localRowStart + slices[e - 1].rowCount);
            uint32_t mergedRows =
                slices[e - 1].localRowStart + slices[e - 1].rowCount - slices[s].localRowStart + slices[e].rowCount;
            if (!contiguous || mergedRows > bandCap) {
                break;
            }
            e++;
        }
        const uint32_t bandLocalStart = slices[s].localRowStart;
        const uint32_t bandChunkStart = slices[s].chunkRowStart;
        const uint32_t bandRows = slices[e - 1].localRowStart + slices[e - 1].rowCount - bandLocalStart;

        SetFlag<HardEvent::V_MTE2>(eventSft);
        WaitFlag<HardEvent::V_MTE2>(eventSft);

        // NZ 重排读：fractal 列块布局 → 行主序 UB（band 整读，列块外层）
        NzToNdCopyIn(scoreUb, tileBase, bandChunkStart, bandRows, kChunkSize, kChunkSizeAlign);

        SetFlag<HardEvent::MTE2_V>(eventSft);
        WaitFlag<HardEvent::MTE2_V>(eventSft);

        // 尾块掩码：AIC 侧 nSize 对齐 16 写出的 padding 列是 L0C 残留，必须掩成 NEG_INF
        if (kChunkSize < kChunkSizeAlign) {
            for (uint32_t r = 0; r < bandRows; r++) {
                AscendC::Duplicate(scoreUb[static_cast<uint32_t>(r) * kChunkSizeAlign + kChunkSize],
                                   float(SOFTMAX_NEG_INF), kChunkSizeAlign - kChunkSize);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t i = s; i < e; i++) {
            const PostRowSlice &sl = slices[i];
            LocalTensor<T> tile = scoreUb[(sl.localRowStart - bandLocalStart) * kChunkSizeAlign];

            Muls(tile, tile, (float)(constInfo.scaleValue), sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();

            BSASubRowBroadcast(tile, tile, globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                               kChunkSizeAlign);

            AscendC::Exp(tile, tile, sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();

            BSADivRowBroadcast(tile, tile, globalSumBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                               kChunkSizeAlign);

            if (constInfo.usePostBlockShape) {
                // 融合路径：归一化 FP32 tile 直接折叠为 pooledScore
                postPoolOP.FoldNormalizedSlice(tile, sl, kChunkStart, kChunkSize, kChunkSizeAlign, isLastKTile);
            } else {
                AscendC::Cast(scoreFp16Ub, tile, AscendC::RoundMode::CAST_ROUND, sl.rowCount * kChunkSizeAlign);
                AscendC::PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_MTE3>(eventSft);
                WaitFlag<HardEvent::V_MTE3>(eventSft);

                uint32_t sftOutQRow = qChunkStart + sl.chunkRowStart;
                uint64_t attnScoreOffset = static_cast<uint64_t>(sftOutQRow) * validYBlocks + kChunkStart;
                GlobalTensor<OUT_T> attnScoreChunkGm;
                attnScoreChunkGm.SetGlobalBuffer(
                    (__gm__ OUT_T *)(attnScoreFp16GmTensor.GetPhyAddr() + attnScoreOffset));
                DataCopyPad(attnScoreChunkGm[0], scoreFp16Ub,
                            {static_cast<uint16_t>(sl.rowCount), static_cast<uint32_t>(kChunkSize * sizeof(OUT_T)), 0,
                             static_cast<uint32_t>((validYBlocks - kChunkSize) * sizeof(OUT_T)), 0});
            }
        }
        s = e;
    }

    if (constInfo.usePostBlockShape) {
        postPoolOP.FlushTile();
    }
}

#endif // BSA_VEC_SM_SERVICE_H
