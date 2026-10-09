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
    __aicore__ inline void InitParams(const BSAConstInfo& constInfo,
                                      const optiling::BSASelectBlockMaskTilingData* __restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<>* uBuf_);

    __aicore__ inline void InitGM(GlobalTensor<T>& ScoreFp32Gm, GlobalTensor<OUT_T>& attnScoreFp16Gm,
                                  GlobalTensor<OUT_T>& attnScoreGm, GlobalTensor<T>& postStagingGm,
                                  GlobalTensor<T>& partSumGm, GlobalTensor<T>& partMaxGm);
    // softmax 单趟化收尾：per-tile 部分和 × exp(m_i−m_c) / l_c → 行折叠 → attnScore 槽（替代 pass2）
    __aicore__ inline void SoftmaxFinalize(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t batchIdx,
                                           uint32_t headIdx, uint32_t validXBlocks, uint32_t validYBlocks);
    // 收尾批处理体（BATCH 个 tile 一次同步轮）：corr 修正 + RA 行折叠 + 批量写出
    __aicore__ inline void SoftmaxFinalizeBatch(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                                uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                                uint32_t validYBlocks, uint32_t t0, uint32_t k, uint32_t groupsValid,
                                                bool boundary);
    // 通用收尾（postBlockShapeY != 16 慢路径）：逐 tile corr 修正 + RA 行折叠 + 跨 tile 列组装
    // （列宽=实际 postBlockShapeY，组宽 foldW，支持列跨 tile 与尾列实际宽度）
    __aicore__ inline void GeneralFinalize(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                           uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                           uint32_t validYBlocks);
    // 子带 NZ 读（软件流水用）：槽空闲门(V_MTE2) + NZ 读 + 数据就绪门(MTE2_V)
    __aicore__ inline void NzReadSubBand(uint64_t tileBase, uint32_t subChunkStart, uint32_t subRows,
                                         uint32_t kChunkSize, uint32_t kColsAlign, uint32_t slot, event_t ev);

    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                       uint32_t kChunkSize, uint32_t validXBlocks,
                                                       uint32_t validYBlocks);

    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                    uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx,
                                                    uint32_t validXBlocks, uint32_t validYBlocks);

private:
    BSAConstInfo constInfo;
    const optiling::BSASelectBlockMaskTilingData* __restrict tilingData;

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
    __aicore__ inline void NzToNdCopyIn(LocalTensor<T>& dstUb, uint64_t tileBase, uint32_t rowStart, uint32_t rowCount,
                                        uint32_t validCols, uint32_t ubRowWidth);

    // 行归约三级级联（BlockReduce 8→1：512→64→8→1），对照 PFA Rowsum<float,SPEC_TILE_512> 参数模式
    __aicore__ inline void RowMaxCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile, uint32_t rows,
                                         uint32_t rowStride, LocalTensor<uint8_t>& tmp);
    __aicore__ inline void RowSumCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile, uint32_t rows,
                                         uint32_t rowStride, LocalTensor<uint8_t>& tmp);

    // ---- post pool 融合服务（post 模式下二趟直接折叠）----
    BSAVecPostPoolService<BSAT> postPoolOP;

    // ---- GM 全局张量引用 ----
    GlobalTensor<OUT_T> attnScoreFp16GmTensor; //< 注意力分数 [xBlocks * yBlocks] (FP32→FP16)，非 post 模式写出
    GlobalTensor<T> scoreFp32GmTensor;         //< Softmax 临时缓冲 (FP32)
    GlobalTensor<T> partSumGmTensor;           //< softmax 单趟化：per-tile 列组部分和
    GlobalTensor<T> partMaxGmTensor;           //< softmax 单趟化：per-tile 局部行 max
    TBuf<>* uBufPtr = nullptr;                 //< 统一 UB 缓冲句柄（post 服务在其后 carve）

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

public:
    // ---- L2'' 跨 tile 预读 + receipt 前移 ----
    // evCross(6)：band 末 partSum 写出的 MTE3→V WAR 信号。原 band 末 Set+Wait（标量阻塞）
    // 改为只 Set，Wait 挪到下 tile 入口 M2 发射之后——M2(t+1) 与 M3(t) 写出重叠。
    static constexpr event_t evCross = (event_t)6;
    // finalize 专用事件（协议重构试验：与 pass1 的 eventSft(3) 隔离，消除复用竞态窗口）
    static constexpr event_t eventFin = (event_t)7;
    bool nextHeadPrefetched = false; // 下 tile 首子带已在 slot0（入口跳过首读）
    bool preWaitedNextReady = false; // 已提前 Wait 掉下 tile ready（base 循环跳过，Set/Wait 1:1）
    bool receiptSentByPass1 = false; // 本 tile receipt 已在 band 末发出（base 跳过）

public:
    // ---- pass1 chunk 上下文：每 chunk 首 tile 填充一次，其余 194 tile 直接消费 ----
    static constexpr uint32_t SUB_ROWS = 32;
    struct Pass1Band {
        uint32_t sliceS, sliceE; // band 覆盖的 slice 区间 [s, e)
        uint32_t localStart;     // band 首行局部行号
        uint32_t chunkStart;     // band 首 chunk 行号
        uint32_t rows;           // band 总行数
        uint32_t nSub;           // 子带数（32 行切分对齐校验后）
    };
    PostRowSlice cachedSlices[BSA_SM_SLICES_MAX];
    uint32_t cachedSliceNum = 0;
    uint32_t cachedChunkKey = 0xFFFFFFFF;     // 无效哨兵（合法 qChunkStart 从 0 起）
    uint32_t cachedValidXBlocks = 0xFFFFFFFF; // 缓存键第二维：批间 validXBlocks 不同（TND/seqlen
                                              // 变长）时 qChunkStart 相同也必须重填，否则复用上一
                                              // 批的 slice 几何（尾 unit 行数差 → 尾行丢失）
    Pass1Band ctxBands[BSA_SM_SLICES_MAX];
    uint32_t ctxBandNum = 0;
    uint64_t ctxChunkGlobalTile = 0; // (qChunkStart/Q_CHUNK_SIZE)*KTILES（chunk 不变量）
    uint64_t ctxTileElems = 0;       // Q_CHUNK_SIZE*K_TILE_COLS（kernel 不变量）
    uint64_t ctxAicRingBase = 0;     // aicIdx*环槽数*tileElems（kernel 不变量）
    uint64_t ctxAivSlotStride = 0;   // aivIdx*KTILES（kernel 不变量）

    __aicore__ inline void FillPass1Ctx(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t validXBlocks);
    __aicore__ inline uint32_t MergeBands(uint32_t kChunkSizeAlign, Pass1Band* out);

private:
};

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitParams(
    const BSAConstInfo& constInfo, const optiling::BSASelectBlockMaskTilingData* __restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    vecSubBlockIdx = constInfo.subBlockIdx; // 0 or 1
    postPoolOP.InitParams(constInfo, tilingData);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitBuffers(TBuf<>* uBuf_)
{
    // ub 划分（softmax 相位约 161.5KB，post 相位约 11KB，合计 < 192KB）
    uint32_t ubOffset = 0;
    uint32_t chunkSize = Q_CHUNK_SIZE;

    // globalMax/globalSum 常驻 pool 服务 carve（[0,185.25KB)）之上的尾区：poolQ/poolK 相位
    // 会覆写该区间，而 chunk 边界重排（poolQ(c) 先于 finalize(c-1)）后这两个跨相位累积
    // 状态必须存活到 finalize。每 chunk 由 pass1 首 tile（kChunkStart==0 分支）重新初始化。
    constexpr uint32_t GLOBAL_STATE_UB_OFFSET = 186 * 1024;
    globalMax = uBuf_->GetWithOffset<T>(chunkSize, GLOBAL_STATE_UB_OFFSET);
    globalSum = uBuf_->GetWithOffset<T>(chunkSize, GLOBAL_STATE_UB_OFFSET + chunkSize * sizeof(T));

    globalMaxBroc = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
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
__aicore__ inline void BSAVecSmService<BSAT>::InitGM(GlobalTensor<T>& ScoreFp32Gm, GlobalTensor<OUT_T>& attnScoreFp16Gm,
                                                     GlobalTensor<OUT_T>& attnScoreGm, GlobalTensor<T>& postStagingGm,
                                                     GlobalTensor<T>& partSumGm, GlobalTensor<T>& partMaxGm)
{
    this->attnScoreFp16GmTensor = attnScoreFp16Gm;
    this->scoreFp32GmTensor = ScoreFp32Gm;
    this->partSumGmTensor = partSumGm;
    this->partMaxGmTensor = partMaxGm;
    if (constInfo.usePostBlockShape) {
        postPoolOP.InitGM(attnScoreGm, postStagingGm);
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
__aicore__ inline void BSAVecSmService<BSAT>::NzToNdCopyIn(LocalTensor<T>& dstUb, uint64_t tileBase, uint32_t rowStart,
                                                           uint32_t rowCount, uint32_t validCols, uint32_t ubRowWidth)
{
    constexpr uint32_t NZ_FRAC_COLS = 16;                            // fractal 列宽
    constexpr uint32_t NZ_BLOCK_ELEMS = Q_CHUNK_SIZE * NZ_FRAC_COLS; // 列块元素数 = 128*16 = 2048
    uint32_t blockNum = (validCols + NZ_FRAC_COLS - 1) / NZ_FRAC_COLS;
    for (uint32_t b = 0; b < blockNum; b++) {
        // WORKAROUND-MTE2: DataCopyPad(64B blockLen) 的 c24-31 段间歇性不落地 UB（残留前次
        // 使用者值；DataCopy 同症状、调用越频繁越严重 → MTE2 通道层段丢失）。双读规避：
        // 同源连读两次（第二次覆盖第一次），漏搬率平方化（0.3%→~1e-5）。
        // 仅尾块非 16 列对齐（validCols%16≠0，实证漏搬形态：kCmp=31/30）时双读；
        // 对齐场景（kChunkSize=512/32/16，未观测到漏搬）保持单读，避免读带宽翻倍
        // （客户场景 794880_bx8_post16 实测双读 +24% 回归）
        // FIX-PAD32: 尾 fractal 读宽 8 列对齐（32B）+ pad 补齐到 16 列 NEG_INF——
        // blockLen 须 32B 对齐（12B 实测丢段破坏整队列）；pad 列由 MTE2 硬件填 NEG_INF，
        // 迟到写覆盖亦同值。残留窗口从 13 列缩至 ≤7 列（Exp置0 兜底）
        const uint32_t fracValid = BSAMin(NZ_FRAC_COLS, validCols - b * NZ_FRAC_COLS);
        const uint32_t fracRead = (fracValid + 7u) / 8u * 8u; // 读列数（8 倍数，32B 对齐）
        const bool fracPad = (fracRead < NZ_FRAC_COLS);
        // FIX-ROWREAD: 恒整行 16 列读（blockLen=64B，与 90d949921 原始实现对齐）。FIX-PAD32
        // 的缩列读（blockLen=32B）在 fracRead=8 时 srcStride 语义无法表达 NZ 行距（实测
        // srcStride=1/配 dstStride 修正均大幅劣化 256→1920+），而其想解决的两大问题已另有
        // 根治：①迟到写覆盖 NEG_INF → d3b757d81 V管Mul掩码（Exp 后掩，迟到写免疫）；
        // ②非 32B blockLen 丢段 → 恒 64B 对齐天然免疫。读入的 GM padding 列由尾块掩码
        // （kChunkSize..align 段 NEG_INF）覆盖，数学无害（756 dump 实证掩码生效）
        (void)fracValid;
        (void)fracRead;
        (void)fracPad;
        const uint32_t copyReps = (validCols % NZ_FRAC_COLS != 0) ? 2u : 1u;
        for (uint32_t rep = 0; rep < copyReps; rep++) {
            DataCopyPad(dstUb[b * NZ_FRAC_COLS],
                        scoreFp32GmTensor[tileBase + static_cast<uint64_t>(b) * NZ_BLOCK_ELEMS +
                                          static_cast<uint64_t>(rowStart) * NZ_FRAC_COLS],
                        {static_cast<uint16_t>(rowCount), static_cast<uint32_t>(NZ_FRAC_COLS * sizeof(float)), 0,
                         static_cast<uint32_t>((ubRowWidth - NZ_FRAC_COLS) * sizeof(float) / 32), 0},
                        {false, 0, 0, 0});
        }
    }
}

/**
 * @brief 行方向归约三级级联（BlockReduce 8→1）：[rows × 512] → [rows × 1]
 * @details 对照 prompt_flash_attention Rowsum<float,SPEC_TILE_512>：
 *          L1: [rows×512] 组内 8→1 → [rows×64]；L2: → [rows×8]；L3: → [rows×1]
 *          tmp 布局（float 元素）：[0, rows*64) L1 结果，[rows*64, rows*72) L2 结果（≤24KB reduceSharedTemp）
 */
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::RowMaxCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile,
                                                            uint32_t rows, uint32_t rowStride,
                                                            LocalTensor<uint8_t>& tmpBytes)
{
    // BlockReduce(vcgmax) 硬件语义：每 repeat 64 元素 → 8 个组内(8:1)结果
    // （对照 PFA FLOAT_VECTOR_SIZE=64 的 Rowsum 级联系数），三级 512→64→8→1
    // L1 分块：向量指令 repeat 域 8bit(≤255)，rows>16 时 rows*rowStride/64 > 255 溢出——
    // 按 16 行/段（repeat=128）分块执行，段间 src/dst 区域互斥无依赖、无需 barrier
    constexpr uint32_t VEC = 64;                   // PFA FLOAT_VECTOR_SIZE
    constexpr uint32_t REDUCE_WIDTH = K_TILE_COLS; // 512
    constexpr uint32_t L1_CHUNK_ROWS = 16;         // 16*512/64 = 128 ≤ 255
    LocalTensor<T> tmp = tmpBytes.ReinterpretCast<T>();
    const uint32_t l1Elems = rows * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM); // rows×64（L1 输出元素数）
    for (uint32_t r = 0; r < rows; r += L1_CHUNK_ROWS) {
        const uint32_t cnt = BSAMin(L1_CHUNK_ROWS, rows - r);
        AscendC::BlockReduceMax<float, false>(tmp[r * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM)], srcTile[r * rowStride],
                                              cnt * rowStride / VEC, 0, 1, 1, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(tmp[l1Elems], tmp, l1Elems / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(dstRow, tmp[l1Elems], rows * FLOAT_DATA_BLOCK_NUM / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::RowSumCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile,
                                                            uint32_t rows, uint32_t rowStride,
                                                            LocalTensor<uint8_t>& tmpBytes)
{
    constexpr uint32_t VEC = 64;
    constexpr uint32_t REDUCE_WIDTH = K_TILE_COLS; // 512
    constexpr uint32_t L1_CHUNK_ROWS = 16;         // repeat 域 8bit 限制，同 RowMaxCascade
    LocalTensor<T> tmp = tmpBytes.ReinterpretCast<T>();
    const uint32_t l1Elems = rows * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM); // rows×64
    for (uint32_t r = 0; r < rows; r += L1_CHUNK_ROWS) {
        const uint32_t cnt = BSAMin(L1_CHUNK_ROWS, rows - r);
        AscendC::BlockReduceSum<float, false>(tmp[r * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM)], srcTile[r * rowStride],
                                              cnt * rowStride / VEC, 0, 1, 1, 8);
    }
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
// ---- pass1 chunk 上下文：band 合并预演（原 while(s<sliceNum) 合并循环 + 子带对齐校验的纯函数化）----
template <typename BSAT>
__aicore__ inline uint32_t BSAVecSmService<BSAT>::MergeBands(uint32_t kChunkSizeAlign, Pass1Band* out)
{
    const uint32_t bandCap = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO / kChunkSizeAlign;
    uint32_t num = 0;
    uint32_t s = 0;
    while (s < cachedSliceNum) {
        uint32_t e = s + 1;
        while (e < cachedSliceNum) {
            bool contiguous =
                (cachedSlices[e].chunkRowStart == cachedSlices[e - 1].chunkRowStart + cachedSlices[e - 1].rowCount) &&
                (cachedSlices[e].localRowStart == cachedSlices[e - 1].localRowStart + cachedSlices[e - 1].rowCount);
            uint32_t mergedRows = cachedSlices[e - 1].localRowStart + cachedSlices[e - 1].rowCount -
                                  cachedSlices[s].localRowStart + cachedSlices[e].rowCount;
            if (!contiguous || mergedRows > bandCap) {
                break;
            }
            e++;
        }
        const uint32_t bandLocalStart = cachedSlices[s].localRowStart;
        const uint32_t bandChunkStart = cachedSlices[s].chunkRowStart;
        const uint32_t bandRows = cachedSlices[e - 1].localRowStart + cachedSlices[e - 1].rowCount - bandLocalStart;
        uint32_t nSub = (bandRows + SUB_ROWS - 1) / SUB_ROWS;
        if (nSub > 1) {
            // 32 行切点必须与 slice 边界重合；片元不对齐时整带作单子带（串行等价原路径）
            for (uint32_t b = SUB_ROWS; b < bandRows; b += SUB_ROWS) {
                bool hit = false;
                for (uint32_t i2 = s; i2 < e; i2++) {
                    if (cachedSlices[i2].localRowStart + cachedSlices[i2].rowCount - bandLocalStart == b) {
                        hit = true;
                        break;
                    }
                }
                if (!hit) {
                    nSub = 1;
                    break;
                }
            }
        }
        out[num].sliceS = s;
        out[num].sliceE = e;
        out[num].localStart = bandLocalStart;
        out[num].chunkStart = bandChunkStart;
        out[num].rows = bandRows;
        out[num].nSub = nSub;
        num++;
        s = e;
    }
    return num;
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FillPass1Ctx(uint32_t qChunkStart, uint32_t qChunkSize,
                                                           uint32_t validXBlocks)
{
    PostRowSlice raw[BSA_SM_SLICES_MAX];
    uint32_t rawNum = postPoolOP.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, raw);
    // （slice 合并）相邻连续 slice 合并到 ≤MERGED_SLICE_ROWS 行/条：V 链条数减半（每条加宽），
    // barrier/发射的固定开销减半（元素总量不变、逐行归约语义不变）。finalize 的 unit 循环
    // （uo += unitRows）天然兼容加宽 slice。片元不连续时保持原样。
    constexpr uint32_t MERGED_SLICE_ROWS = 32;
    cachedSliceNum = 0;
    for (uint32_t i = 0; i < rawNum;) {
        uint32_t rows = raw[i].rowCount;
        uint32_t j = i + 1;
        while (j < rawNum && rows < MERGED_SLICE_ROWS) {
            const PostRowSlice& a = raw[j - 1];
            const PostRowSlice& b = raw[j];
            if (b.chunkRowStart != a.chunkRowStart + a.rowCount || b.localRowStart != a.localRowStart + a.rowCount ||
                rows + b.rowCount > MERGED_SLICE_ROWS) {
                break;
            }
            rows += b.rowCount;
            j++;
        }
        cachedSlices[cachedSliceNum] = {raw[i].chunkRowStart, raw[i].localRowStart, rows};
        cachedSliceNum++;
        i = j;
    }
    cachedChunkKey = qChunkStart;
    cachedValidXBlocks = validXBlocks;
    const uint32_t ktiles = (constInfo.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    ctxChunkGlobalTile = (static_cast<uint64_t>(qChunkStart) / Q_CHUNK_SIZE) * ktiles;
    ctxTileElems = static_cast<uint64_t>(Q_CHUNK_SIZE) * K_TILE_COLS;
    ctxAicRingBase = static_cast<uint64_t>(constInfo.aicIdx) * BSA_SCRATCH_RING_TILES * ctxTileElems;
    ctxAivSlotStride = static_cast<uint64_t>(constInfo.aivIdx) * ktiles;
    ctxBandNum = MergeBands(K_TILE_COLS, ctxBands);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::NzReadSubBand(uint64_t tileBase, uint32_t subChunkStart, uint32_t subRows,
                                                            uint32_t kChunkSize, uint32_t kColsAlign, uint32_t slot,
                                                            event_t ev)
{
    // 槽空闲门：V 队列排空（该槽上轮计算完成）后 MTE2 才可覆写（Set+Wait 配对，无残留态）
    SetFlag<HardEvent::V_MTE2>(ev);
    WaitFlag<HardEvent::V_MTE2>(ev);
    // NZ 重排读：fractal 列块 → 行主序 UB（本子带行 × 全列宽，落入乒乓槽 slot）
    LocalTensor<T> slotUb = scoreUb[slot * SUB_ROWS * kColsAlign];
    NzToNdCopyIn(slotUb, tileBase, subChunkStart, subRows, kChunkSize, kColsAlign);
    // 数据就绪门：MTE2 队列排空（读完成）后置位，供 V 侧 Wait 消费
    SetFlag<HardEvent::MTE2_V>(ev);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                          uint32_t kChunkStart, uint32_t kChunkSize,
                                                                          uint32_t validXBlocks, uint32_t validYBlocks)
{
    // chunk 上下文：首 tile 填充（slices 解析 + band 预演 + 寻址基址），其余 194 tile 直接消费
    if (cachedChunkKey != qChunkStart || cachedValidXBlocks != validXBlocks) {
        FillPass1Ctx(qChunkStart, qChunkSize, validXBlocks);
    }
    const PostRowSlice* slices = cachedSlices;
    const uint32_t sliceNum = cachedSliceNum;
    if (sliceNum == 0) {
        return;
    }

    // NZ 读法行宽：对齐到 fractal 列宽 16（≥16），尾块 padding 段统一掩 SOFTMAX_NEG_INF（数学无害）
    const bool fullTile = (kChunkSize == K_TILE_COLS);
    const uint32_t kChunkSizeAlign = fullTile ? K_TILE_COLS : (kChunkSize + 15) / 16 * 16;
    // 折叠组宽由实际 postBlockShapeY 决定：%16==0 走 16 组宽（客户场景快路径），否则 8 组宽
    // （BLOCK_SHAPE_ALIGN=8 保证 postBlockShapeY 为 8 的倍数；与 tiling partSumSize 同公式）
    const uint32_t foldW = (constInfo.usePostBlockShape && constInfo.postBlockShapeY % 16 != 0) ? 8u : 16u;
    const uint32_t foldG = K_TILE_COLS / foldW;
    // 折叠列宽取对齐倍数（BRS 以连续 64 float 步进消费，行距非倍数会跨行错位；pad 列由尾块
    // 掩码覆 NEG_INF，exp 后为 0，对 max/sum/折叠均无贡献）：w8 取 64 倍数（8 列组不跨行）；
    // w16 取 128 倍数（16 列组数 g16 为 8 倍数 → 配对 Add/ GM 写步进均为整数块，免掩码）；
    // kcs≥385 时 w16 升格 512=K_TILE_COLS 自动走级联快路径
    const uint32_t foldAlign = (foldW == 8) ? 64u : 128u;
    const uint32_t foldCols = (kChunkSize + foldAlign - 1) / foldAlign * foldAlign;

    // 入口 MTE3_MTE2 排空：band 末已 Wait MTE3_V（M3 排空才返回），此处排空对恢复以保护
    // band 末 M3 写出与本 tile M2 读的时序（bisect 定论：延迟消费架构 50% 挂死不可用）
    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);

    // 当前 kTile 基址：ctx 缓存的三层不变量 + tileIdx 偏移（等价原 BSAScratchTileOffset）
    const uint32_t tileIdx = kChunkStart / K_TILE_COLS;
    const uint64_t tileBase = ctxAicRingBase + ((ctxChunkGlobalTile + tileIdx) % BSA_SCRATCH_RING_TILES) * ctxTileElems;
    const uint64_t psSlotBase = (ctxAivSlotStride + tileIdx) * Q_CHUNK_SIZE * foldG;
    const uint64_t pmSlotBase = (ctxAivSlotStride + tileIdx) * Q_CHUNK_SIZE;
    LocalTensor<T> foldOut = reduceSharedTemp.ReinterpretCast<T>()[4096]; // [本 aiv ≤64 行][32 组] band 累计
    // foldW==8 时折叠输出 [≤32 行][64 组]（与 foldOut 同一 rs 尾区，不同相位互斥），按子带暂存+写出
    LocalTensor<T> foldOut8 = reduceSharedTemp.ReinterpretCast<T>()[4096];

    // band 布局：full tile 用 ctx 预演结果（194/195 命中）；尾 tile bandCap 随列宽变化，就地重演
    const Pass1Band* bands;
    Pass1Band tailBands[BSA_SM_SLICES_MAX];
    uint32_t bandNum;
    if (fullTile) {
        bands = ctxBands;
        bandNum = ctxBandNum;
    } else {
        bandNum = MergeBands(foldCols, tailBands);
        bands = tailBands;
    }
    // —— 子带流水化：scoreUb 按 32 行切 2 乒乓槽（容量不变），MTE2 读子带 j+2 与 V 算子带 j 重叠 ——
    // 槽事件 4/5：V_MTE2=槽空闲（V 队列排空置位）、MTE2_V=数据就绪（MTE2 排空置位）；
    // Set/Wait 严格配对无残留态，与 eventSft(3)/eventPost(2)/pool(0/1) 不冲突
    const event_t evSlot[2] = {(event_t)4, (event_t)5};
    for (uint32_t bi = 0; bi < bandNum; bi++) {
        const Pass1Band& bd = bands[bi];

        // 预读子带 0/1（MTE2 队列串行；子带 1 的读与子带 0 的 V 计算重叠）
        // （L2''）若上一 tile band 末已预读本 band 首子带进 slot0，跳过首读直接消费
        if (!nextHeadPrefetched) {
            NzReadSubBand(tileBase, bd.chunkStart, BSAMin(SUB_ROWS, bd.rows), kChunkSize, foldCols, 0, evSlot[0]);
        }
        nextHeadPrefetched = false;
        // M3(上 tile) 排空后本 tile V 链方可覆写 foldOut/tmpMax——band 末立即 Wait 已保证，
        // 此处无需额外门（mte3vPending 延迟消费逻辑已随 bisect 定论移除：50% 挂死、收益归零）
        if (bd.nSub > 1) {
            NzReadSubBand(tileBase, bd.chunkStart + SUB_ROWS, BSAMin(SUB_ROWS, bd.rows - SUB_ROWS), kChunkSize,
                          foldCols, 1, evSlot[1]);
        }
        uint32_t sliceIdx = bd.sliceS;
        for (uint32_t j = 0; j < bd.nSub; j++) {
            const uint32_t subLocalStart = bd.localStart + j * SUB_ROWS;
            const uint32_t subChunkStart = bd.chunkStart + j * SUB_ROWS;
            const uint32_t subRows = BSAMin(SUB_ROWS, bd.rows - j * SUB_ROWS);
            const uint32_t slot = j & 1;
            LocalTensor<T> subUb = scoreUb[slot * SUB_ROWS * foldCols];
            WaitFlag<HardEvent::MTE2_V>(evSlot[slot]); // V 门：本子带数据就绪
            // DIAG-755: GM scratch 尾fractal c464-496 (对齐32B: 464%8=0 → 1856B ✓)
            // 尾块掩码：AIC 侧 nSize 对齐 16 写出的 padding 列是 L0C 残留，必须掩成 NEG_INF
            if (kChunkSize < foldCols) {
                // VEC 32B 对齐：kChunkSize%8≠0 时 Duplicate 目标未对齐（kCols=31→+124B 实锤
                // VEC 队列死亡→下游所有 MTE3_V/V_MTE3 等待永挂）——未对齐尾段(≤7列)标量写，
                // 对齐段一次性 Duplicate
                const uint32_t padAlign = (kChunkSize + 7u) / 8u * 8u;
                for (uint32_t r = 0; r < subRows; r++) {
                    for (uint32_t c = kChunkSize; c < padAlign; c++) {
                        subUb.SetValue(static_cast<uint32_t>(r) * foldCols + c, float(SOFTMAX_NEG_INF));
                    }
                }
                if (padAlign < foldCols) {
                    for (uint32_t r = 0; r < subRows; r++) {
                        AscendC::Duplicate(subUb[static_cast<uint32_t>(r) * foldCols + padAlign],
                                           float(SOFTMAX_NEG_INF), foldCols - padAlign);
                    }
                }
                // FIX-SCALAR-V: 尾段掩码含 SetValue 标量写（kCols%8≠0 时），标量管→V 管无
                // 隐式可见性保证——PIPE_V 只同步 V 队列，标量写须 PIPE_ALL 才对后续 V 读
                // （Muls/ReduceMax）可见。缺此屏障时 L0C 残留列间歇参与 ReduceMax → 行 max
                // 偏大 → 整行 attnScore 缩小（basevar B1 vy=30 偶数行 -0.43% 实锤）
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            uint32_t i = sliceIdx;
            while (i < bd.sliceE) {
                const PostRowSlice& sl = slices[i];
                if (sl.localRowStart >= subLocalStart + subRows) {
                    break; // 本子带的 slice 已处理完
                }
                LocalTensor<T> tile = subUb[(sl.localRowStart - subLocalStart) * foldCols];

                Muls(tile, tile, (float)(constInfo.scaleValue), sl.rowCount * foldCols);
                AscendC::PipeBarrier<PIPE_V>();

                // 行归约：全 tile(512) 走 BlockReduce 三级级联（快），尾 tile 走 AR 原路径
                uint32_t maxShape[] = {sl.rowCount, foldCols};
                bool isSrcInnerPad = true;

                if (foldCols == K_TILE_COLS) {
                    LocalTensor<T> dstMax = tmpMax[sl.localRowStart];
                    RowMaxCascade(dstMax, tile, sl.rowCount, foldCols, reduceSharedTemp);
                } else {
                    AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(
                        tmpMax[sl.localRowStart], tile, reduceSharedTemp, maxShape, isSrcInnerPad);
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
                AscendC::Sub(diffMax[sl.localRowStart], globalMax[sl.localRowStart], tmpMax[sl.localRowStart],
                             sl.rowCount);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Exp(diffMax[sl.localRowStart], diffMax[sl.localRowStart], sl.rowCount);
                AscendC::PipeBarrier<PIPE_V>();

                AscendC::Mul(globalSum[sl.localRowStart], globalSum[sl.localRowStart], diffMax[sl.localRowStart],
                             sl.rowCount);
                AscendC::PipeBarrier<PIPE_V>();

                // step 2: global_sum += exp(score - newMax).sum(dim=-1)
                uint32_t brocRepeat = BSACeilDiv(sl.rowCount, static_cast<uint32_t>(BROC_BASE_NUM));
                Brcb(globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], tmpMax[sl.localRowStart], brocRepeat,
                     {1, 8});
                AscendC::PipeBarrier<PIPE_V>();

                BSASubRowBroadcast(tile, tile, globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                                   foldCols);

                AscendC::Exp(tile, tile, sl.rowCount * foldCols);
                AscendC::PipeBarrier<PIPE_V>();
                // 仅 post 启用。FIX-VMASK: 残留列清 0 改 V 管 Mul 掩码——SetValue（标量管）
                // 与 MTE2 排序弱，被迟到写覆盖回 scratch 残留分数（近均匀下≈max→exp≈1，
                // g05 行级 12x 实锤）；V 管操作在 MTE2_V flag 后发射，晚于一切 MTE2 写。
                // 掩码：最后一个混合 8 列块 [maskBase, maskBase+8) 前 rem 列保留其余清 0，
                // 后续整块由 Duplicate NEG_INF（已有）→ exp=0
                if (kChunkSize < foldCols && constInfo.usePostBlockShape) {
                    // FIX-VMASK2: 掩码段扩展到整个 padding 尾巴 [kCmp 下取整 8 对齐, foldCols)。
                    // 旧版只掩 rem≠0 的混合块——kCmp%8==0（如 g01 尾 tile 424）完全跳过，唯一
                    // 保护 Duplicate NEG_INF 会被整行读（FIX-ROWREAD）第二遍迟到写用 GM raw
                    // padding(0) 覆盖 → exp(0−max)=0.055 污染折叠和（outside boundary 实锤）。
                    // Mul 掩码在 Exp 后 V 管执行，晚于一切 MTE2 写，迟到免疫
                    const uint32_t rem = kChunkSize % 8u;       // 混合块内有效列数
                    const uint32_t maskBase = kChunkSize - rem; // 8 列对齐起点（32B）
                    if (maskBase < foldCols) {
                        // 混合块用 Mul 掩码（借 diffMax[0]，旧版验证安全区）；后续整 padding 块
                        // 用 V 管 Duplicate 0 直清（无需第二掩码区——mskZero=diffMax[16] 越界
                        // 破坏活数据致 py2 分数丢，g15 稳定 116 错实锤）。两者均在 Exp 后 V 管
                        // 执行，晚于一切 MTE2 迟到写，免疫
                        LocalTensor<T> mskMix = diffMax[0];
                        for (uint32_t c = 0; c < 8u; c++) {
                            mskMix.SetValue(c, c < rem ? 1.0f : 0.0f);
                        }
                        AscendC::PipeBarrier<PIPE_ALL>(); // 标量写掩码 → V 可见
                        for (uint32_t r = 0; r < sl.rowCount; r++) {
                            AscendC::Mul(tile[static_cast<uint32_t>(r) * foldCols + maskBase],
                                         tile[static_cast<uint32_t>(r) * foldCols + maskBase], mskMix, 8);
                        }
                        for (uint32_t off = maskBase + 8u; off < foldCols; off += 8u) {
                            for (uint32_t r = 0; r < sl.rowCount; r++) {
                                AscendC::Duplicate(tile[static_cast<uint32_t>(r) * foldCols + off], 0.0f, 8);
                            }
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                }
                // DEBUG-757: Exp 后尾组段（c190-207，尾块 padding 应≈0）；desc=qChunkStart
                if (kChunkSize < foldCols && constInfo.usePostBlockShape) {
                    SetFlag<HardEvent::V_MTE3>(eventSft);
                    WaitFlag<HardEvent::V_MTE3>(eventSft);
                }

                if (foldCols == K_TILE_COLS) {
                    LocalTensor<T> dstSum = tmpSum[sl.localRowStart];
                    RowSumCascade(dstSum, tile, sl.rowCount, foldCols, reduceSharedTemp);
                } else {
                    uint32_t sumShape[] = {sl.rowCount, foldCols};
                    if (foldCols <= FLOAT_REPEAT_NUM) {
                        AscendC::WholeReduceSum<T, true>(tmpSum[sl.localRowStart], tile, static_cast<int32_t>(foldCols),
                                                         static_cast<int32_t>(sl.rowCount), 1, 1,
                                                         static_cast<int32_t>(foldCols / FLOAT_DATA_BLOCK_NUM));
                    } else {
                        // isReuseSource 必须为 false：true 时 AR 实现以 src（本 tile）作 staging 缓冲，
                        // 大行宽(>64)路径 ReduceSumInLargeLast 原地 halving 归约会覆写 tile——
                        // 随后 w8 折叠 BRS / w16 尾 tile 折叠 DataCopy 读到混叠数据
                        // （blk8 partSum 蝴蝶混叠值错根因，p16var/tndtail/bfvar 同源）
                        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, false>(
                            tmpSum[sl.localRowStart], tile, reduceSharedTemp, sumShape, isSrcInnerPad);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }

                AscendC::Add(globalSum[sl.localRowStart], tmpSum[sl.localRowStart], globalSum[sl.localRowStart],
                             sl.rowCount);
                AscendC::PipeBarrier<PIPE_V>();

                // 更新globalMax
                AscendC::Max(globalMax[sl.localRowStart], tmpMax[sl.localRowStart], globalMax[sl.localRowStart],
                             sl.rowCount);

                // ---- softmax 单趟化：per-slice 折叠累计到 foldOut（band 末统一写出）----
                // partSum 槽布局（行主序）[aivIdx*KTILES + tileIdx][128 行][foldG 组]；partMax 槽：[槽][行]
                // 仅 post 模式折叠（非 post 无消费方；且单行 slice 走 AR 尾路径 repA=rc*4/8=0 会触发
                // repeat=0 的 VEC 异常——base0 S=256 单行 chunk 实锤）
                if (foldW == 16 && constInfo.usePostBlockShape) {
                    LocalTensor<T> groupTmp = reduceSharedTemp.ReinterpretCast<T>();
                    // 无效组清零（本 slice 行区）：GM 槽行宽恒 32，收尾仅消费有效组列
                    AscendC::Duplicate(foldOut[sl.localRowStart * 32], float(0.0f), 32 * sl.rowCount);
                    AscendC::PipeBarrier<PIPE_V>();
                    if (foldCols == K_TILE_COLS) {
                        // 全 tile：复用 RowSumCascade 的 L1 输出（[rows][64] 8 列组和，本 slice 归约完成后
                        // 仍驻留 reduceSharedTemp[0..rows*64)）——PairReduceSum 相邻对相加得 [rows][32]
                        // 16 列组部分和（替代全矩阵块配对 Add + BlockReduceSum 两步，元素量 4096 vs 98304）。
                        // 参数（实测标定）：repeat 数 = rows，每 repeat 消费 64 元素产 32 输出；
                        // dstRepStride 单位 = repeat 输出宽（32 float），=1 即逐行连续；srcRepStride=8 块
                        // DIAG-764: L1 级联输出 ([行][64组] 前1024)
                        AscendC::ReducePairElem<AscendC::ReduceType::SUM, float, float, true>(
                            foldOut[sl.localRowStart * 32], groupTmp[0], static_cast<int32_t>(sl.rowCount * 64),
                            static_cast<int32_t>(sl.rowCount), 1, 1, 8);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        // 尾 tile（AR 行归约路径无 L1 输出）：tile 的 8-float 块直接配对 Add（相邻块
                        // 相加 = 16 列组部分和）→ BRS 块和成标量紧排 [rc][g16] → 即时 GM 写
                        // （槽行宽 32 组，dstStride 补齐；w8 路径同款模式）。
                        // 旧实现（DataCopy 32 列整形 + Add/BRS）输出 [2r+g] 紧凑节距与消费端
                        // [r*32+g] 错位，仅 row0 落对、其余行 partSum 槽为陈旧内容（p16var 根因）。
                        // foldCols=128 倍数 → g16 为 8 倍数：Add 全 8 块免掩码、GM dstStride 整数块。
                        const uint32_t g16 = foldCols / 16; // 16 列组数（8 的倍数）
                        const uint32_t quadCalls = g16 / 8;
                        LocalTensor<T> chunks = groupTmp[0]; // [rc][F/2] ≤ 16×192 = 3072 float
                        LocalTensor<T> compact = reduceSharedTemp.ReinterpretCast<T>()[4096]; // [rc][g16] ≤ 512
                        AscendC::BinaryRepeatParams bpPair;
                        bpPair.dstBlkStride = 1;  // 组内 8 float 连续
                        bpPair.src0BlkStride = 2; // 相邻 8-float 块间隔 16 float（16 列组）
                        bpPair.src1BlkStride = 2;
                        bpPair.dstRepStride = foldCols / 16; // chunks 行距 F/2 float
                        bpPair.src0RepStride = foldCols / 8; // tile 行距
                        bpPair.src1RepStride = foldCols / 8;
                        for (uint32_t rb = 0; rb < sl.rowCount; rb += 16) {
                            const uint32_t rc = BSAMin(uint32_t(16), sl.rowCount - rb);
                            for (uint32_t qb = 0; qb < quadCalls; qb++) {
                                // dst[r*F/2 + qb*64 + j*8..+8] = tile[r*F + 128qb + 16j..+8] + tile[...+8..+16]
                                AscendC::Add<T, false>(chunks[qb * 64], tile[rb * foldCols + qb * 128],
                                                       tile[rb * foldCols + qb * 128 + 8], static_cast<uint64_t>(0),
                                                       static_cast<uint8_t>(rc), bpPair);
                                AscendC::PipeBarrier<PIPE_V>();
                            }
                            // BRS：每 repeat 消费 8 个相邻 8-float 块（一行的 8 个 16 列组）→ 8 标量
                            AscendC::BlockReduceSum<float, false>(compact[0], chunks[0], rc * g16 / 8, 0, 1, 1, 8);
                            AscendC::PipeBarrier<PIPE_V>();
                            SetFlag<HardEvent::V_MTE3>(eventSft);
                            WaitFlag<HardEvent::V_MTE3>(eventSft);
                            DataCopyPad(partSumGmTensor[psSlotBase + (sl.chunkRowStart + rb) * 32], compact[0],
                                        {static_cast<uint16_t>(rc), static_cast<uint32_t>(g16 * sizeof(float)), 0,
                                         static_cast<uint32_t>((32 - g16) * sizeof(float)), 0});
                            SetFlag<HardEvent::MTE3_V>(eventSft);
                            WaitFlag<HardEvent::MTE3_V>(eventSft);
                        }
                    }
                } else {
                    // foldW==8 通用路径（postBlockShapeY%16!=0）：BRS 一级级联产紧排 [rows][foldCols/8]
                    // 8 列组（foldCols 为 64 倍数保证组不跨行；padding 列 exp 后为 0 无贡献；组数按
                    // validYBlocks 收敛消费）。按 ≤32 行块暂存并即时跨步写出（GM 槽行宽恒 64 组，
                    // dstStride=(64-组数)×4B，组数为 8 倍数故 32B 对齐）
                    const uint32_t grpW = foldCols / 8;
                    for (uint32_t r0 = 0; r0 < sl.rowCount; r0 += SUB_ROWS) {
                        const uint32_t rc = BSAMin(SUB_ROWS, sl.rowCount - r0);
                        for (uint32_t r = 0; r < rc; r += 16) { // repeat 域 8bit：16 行×512/64=128 ≤ 255
                            const uint32_t cnt = BSAMin(uint32_t(16), rc - r);
                            AscendC::BlockReduceSum<float, false>(foldOut8[r * grpW], tile[(r0 + r) * foldCols],
                                                                  cnt * foldCols / 64, 0, 1, 1, 8);
                            AscendC::PipeBarrier<PIPE_V>();
                        }
                        SetFlag<HardEvent::V_MTE3>(eventSft);
                        WaitFlag<HardEvent::V_MTE3>(eventSft);
                        DataCopyPad(partSumGmTensor[psSlotBase + (sl.chunkRowStart + r0) * 64], foldOut8[0],
                                    {static_cast<uint16_t>(rc), static_cast<uint32_t>(grpW * sizeof(float)), 0,
                                     static_cast<uint32_t>((64 - grpW) * sizeof(float)), 0});
                        SetFlag<HardEvent::MTE3_V>(eventSft);
                        WaitFlag<HardEvent::MTE3_V>(eventSft);
                    }
                }
                i++;
            }
            sliceIdx = i;
            // 预读子带 j+2（同槽 j&1；NzReadSubBand 内部 V_MTE2 门保证子带 j 计算排空后才覆写）
            if (j + 2 < bd.nSub) {
                NzReadSubBand(tileBase, subChunkStart + 2 * SUB_ROWS, BSAMin(SUB_ROWS, bd.rows - (j + 2) * SUB_ROWS),
                              kChunkSize, foldCols, slot, evSlot[slot]);
            }
            // （M3 拆半实测 -0.65ms：band 中 V_MTE3 Set+Wait 把 scalar 锁到 slot0 排空才发射
            // slot1 的 V 链，V 管道多 ~0.2µs 气泡/tile > 藏掉的 M3 传输 ~0.1µs，净损失，不做）
        }
        // band 末批量写出：V→MTE3 一次同步 + partSum（4KB 粒度）+ partMax 各一段
        // （同步对数从每 slice 2 对降为每 band 2 对；foldOut 容量恰 [64 行][32 组]）
        SetFlag<HardEvent::V_MTE3>(eventSft);
        WaitFlag<HardEvent::V_MTE3>(eventSft);
        // （receipt 前移）tile t 的 scratch 读取早已全部完成（V_MTE3 对刚返回），此刻发 receipt：
        // MTE2 队列为空 → 立即置位，AIC 零延迟。若留在 base 循环发，预读 M2 排在 MTE2 队列里
        // 会推迟 receipt 置位 ~一个搬运时长，N=2 环下直接节流 AIC。仅末 band 发。
        if (bd.sliceE == sliceNum && bi == bandNum - 1) {
            // 仅单趟化路径（SoftmaxFinalize，pass2 不再读 scratch）才可前移 receipt；
            // 非 post / debugOldPass2 的老 pass2 要重读本 tile，receipt 必须延迟到 pass2 读完
            // （否则 AIC 下一 (b,h) 的 fixpipe 在 pass2 读槽期间覆写 → 读到旧/半旧数据）
            if (constInfo.usePostBlockShape && !constInfo.debugOldPass2) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V2_TO_C2_FLAG[tileIdx % SYNC_C1_TO_V1_FLAG_NUMS]);
                receiptSentByPass1 = true;
            }
        }
        // （L2''）band 末 V 已排空——此处等 ready(t+1) 不阻塞任何 V 发射；预读下 tile 首子带
        // 进 slot0（V(t:0) 早已消费完），M2(t+1:0) 与 M3(t) 写出/返回/旗标/下 tile 入口全程重叠。
        // 仅限末 band、双子带对齐、下 tile 存在且全宽、非 chunk 末（下 tile 属下 chunk，
        // 中隔 finalize 用 scoreUb）。
        if (bi == bandNum - 1 && bd.nSub == 2 && kChunkStart + kChunkSize < validYBlocks &&
            kChunkStart + 2 * K_TILE_COLS <= validYBlocks) {
            AscendC::CrossCoreWaitFlag(SYNC_C1_TO_V1_FLAG[(tileIdx + 1) % SYNC_C1_TO_V1_FLAG_NUMS]);
            preWaitedNextReady = true;
            const uint64_t nextTileBase =
                ctxAicRingBase + ((ctxChunkGlobalTile + tileIdx + 1) % BSA_SCRATCH_RING_TILES) * ctxTileElems;
            NzReadSubBand(nextTileBase, bd.chunkStart, BSAMin(SUB_ROWS, bd.rows), K_TILE_COLS, K_TILE_COLS, 0,
                          evSlot[0]);
            nextHeadPrefetched = true;
        }
        {
            if (foldW == 16 && foldCols == K_TILE_COLS) {
                // DIAG-763: foldOut band末写出前 ([行][32组] 前512值=16行)
                // band 末批量写出 partSum（仅全宽 tile；尾 tile 已在 slice 内按即时 GM 写出——
                // 尾路径不写 foldOut，此处若照搬会把 foldOut 陈旧内容灌进槽）
                const uint32_t bandElems = bd.rows * 32;
                for (uint32_t off = 0; off < bandElems; off += 1024) {
                    const uint32_t n = BSAMin(uint32_t(1024), bandElems - off);
                    DataCopyPad(partSumGmTensor[psSlotBase + bd.chunkStart * 32 + off],
                                foldOut[bd.localStart * 32 + off], {1, static_cast<uint16_t>(n * sizeof(float)), 0, 0});
                }
            }
            DataCopyPad(partMaxGmTensor[pmSlotBase + bd.chunkStart], tmpMax[bd.localStart],
                        {1, static_cast<uint16_t>(bd.rows * sizeof(float)), 0, 0});
        }
        // MTE3→V WAR：Set 后立即 Wait（bisect 定论：延迟消费虽有小收益但 50% 挂死，M3 段
        // 仅 0.5µs/tile 不值；立即 Wait 下 M3(t) 排空后才返回——foldOut 保护完备、协议简洁）
        SetFlag<HardEvent::MTE3_V>(evCross);
        WaitFlag<HardEvent::MTE3_V>(evCross);
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
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxFinalizeBatch(const PostRowSlice* slices, uint32_t sliceNum,
                                                                   uint32_t qChunkStart, uint32_t batchIdx,
                                                                   uint32_t headIdx, uint32_t validXBlocks,
                                                                   uint32_t validYBlocks, uint32_t t0, uint32_t k,
                                                                   uint32_t groupsValid, bool boundary)
{
    GlobalTensor<OUT_T> attnScoreGmTensor = postPoolOP.GetAttnScoreGm();
    const uint32_t validPostYCur = (validYBlocks + constInfo.postBlockShapeY - 1) / constInfo.postBlockShapeY;
    const uint32_t KTILES = (constInfo.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    const uint32_t unitRows = constInfo.postBlockShapeX;
    // attnScore 单槽复用：逐 (b,h) 覆写偏移 0
    const uint64_t pooledHeadOffset = 0;
    // rs 布局（fp32 元素）：[0,1024) RA tmp；[1024,1280) fp16Row[unit][tile][32 half]；[1280,1312) invCols
    //   [1312,1824) unitRes[unit][tile][32]；[1824,1952) corrAll；[1952,2080) diffSlice
    //   [2080,2592) pmT[4][128]；[2592,3616) corrBroc
    // rs 布局（fp32 元素，BATCH=8）：[0,1024) RA tmp；[1024,1536) fp16Row[unit][tile][32 half]
    //   [1536,1568) invCols；[1568,2592) unitRes[unit][tile][32]
    //   [2592,2720) corrAll；[2720,2848) diffSlice
    //   [2848,3872) pmT[8][128]；[3872,4384) corrBroc
    //   （pass2 与 pass1 foldOut[4096..6144) 不同相位无冲突；总 4384 < 6144）
    LocalTensor<T> rs = postPoolOP.GetReduceTemp();
    LocalTensor<uint8_t> raTmp = rs[0].ReinterpretCast<uint8_t>();
    LocalTensor<OUT_T> fp16Row = rs[1024].ReinterpretCast<OUT_T>();
    LocalTensor<T> invCols = rs[1536];
    LocalTensor<T> unitRes = rs[1568];
    LocalTensor<T> corrAll = rs[2592];
    LocalTensor<T> diffSlice = rs[2720];
    LocalTensor<T> pmT = rs[2848];
    LocalTensor<T> corrBroc = rs[3872];
    LocalTensor<T> psUb = scoreUb; // 批槽 [批内 tile][2048 float]（归属行段，rel 行索引）

    const uint32_t ownRows = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
    // FIX-CHUNK-DOMAIN(B''): mine 行集空洞判定与 chunk 跨度——原版读回/标量 corr 在无空洞
    // 场景零改动(泛化/客户零扰动), 仅空洞场景补洞读+逐行 corr(d_s1long1 三层根因)
    const uint32_t spanRows =
        slices[sliceNum - 1].chunkRowStart + slices[sliceNum - 1].rowCount - slices[0].chunkRowStart;
    const bool hasHole = (spanRows != ownRows);
    const uint32_t chunkBase = slices[0].chunkRowStart;
    const uint32_t nUnits = (ownRows + unitRows - 1) / unitRows;
    const bool invRowsUniform = (validXBlocks % unitRows == 0);

    // —— 批量读：k 个 partSum 槽归属行段（4KB 粒度）+ k 个 partMax → pmT 批槽 ——
    SetFlag<HardEvent::MTE3_MTE2>(eventSft);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft);
    SetFlag<HardEvent::V_MTE2>(eventSft);
    WaitFlag<HardEvent::V_MTE2>(eventSft);
    for (uint32_t b = 0; b < k; b++) {
        const uint64_t slotB = (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + t0 + b) * Q_CHUNK_SIZE * 32;
        const uint64_t pmSlotB = (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + t0 + b) * Q_CHUNK_SIZE;
        for (uint32_t off = 0; off < ownRows * 32; off += 1024) {
            const uint32_t n = BSAMin(uint32_t(1024), ownRows * 32 - off);
            DataCopyPad(psUb[b * 2048 + off], partSumGmTensor[slotB + chunkBase * 32 + off],
                        {1, static_cast<uint16_t>(n * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        }
        DataCopyPad(pmT[b * Q_CHUNK_SIZE + chunkBase], partMaxGmTensor[pmSlotB + chunkBase],
                    {1, static_cast<uint16_t>(ownRows * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        // FIX-CHUNK-DOMAIN(B''): 空洞场景补洞——原读回按 ownRows(local 跨度)只覆盖
        // [chunkBase, +ownRows), 空洞后的 mine slice(整段在 ownRows 之外)逐段补读
        if (hasHole) {
            for (uint32_t si = 0; si < sliceNum; si++) {
                const PostRowSlice& slH = slices[si];
                if (slH.chunkRowStart >= chunkBase + ownRows) {
                    DataCopyPad(psUb[b * 2048 + slH.chunkRowStart * 32],
                                partSumGmTensor[slotB + chunkBase * 32 + slH.chunkRowStart * 32],
                                {1, static_cast<uint16_t>(slH.rowCount * 32 * sizeof(float)), 0, 0}, {false, 0, 0, 0});
                    DataCopyPad(pmT[b * Q_CHUNK_SIZE + chunkBase + slH.chunkRowStart],
                                partMaxGmTensor[pmSlotB + chunkBase + slH.chunkRowStart],
                                {1, static_cast<uint16_t>(slH.rowCount * sizeof(float)), 0, 0}, {false, 0, 0, 0});
                }
            }
        }
    }
    SetFlag<HardEvent::MTE2_V>(eventSft);
    WaitFlag<HardEvent::MTE2_V>(eventSft);
    // DIAG-761: Batch psUb 读回后 (b=0, [16行][32组]), desc=(b,h)

    // —— 尾 tile 重算 invCols[g]=1/组内列数（全 full tile 用循环外默认 1/16）——
    if (boundary) {
        const uint32_t py0 = t0 * (K_TILE_COLS / 16);
        for (uint32_t g = 0; g < 32; g++) {
            uint32_t colsG = BSAMin(uint32_t(16), validYBlocks - (py0 + g) * 16);
            invCols.SetValue(g, 1.0f / static_cast<float>(static_cast<int32_t>(colsG == 0 ? 16 : colsG)));
        }
        // FIX-SCALAR-V: SetValue 标量写 → V 读(Mul invCols) 须 PIPE_ALL（PIPE_V 不保标量落地，
        // 尾 py 列读到默认 1/16 → c30 恰好 0.5x——bx8_s3904 dump 实锤，与 GeneralFinalize
        // 列组装同款 bug 的 Batch 路径翻版）
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // —— 逐 tile：corr（归属行整段 3 次调用）+ 行广播乘 + nUnits 个 RA 行折叠 ——
    // pass1 语义：partSum 槽 = Σ exp(x - m_t)、partMax 槽 = m_t、globalMax/globalSum = 最终 m_c/l_c；
    // 乘 corr = exp(m_t − m_c)/l_c 后 = Σ exp(x - m_c)/l_c（指数律严格等价）。
    for (uint32_t b = 0; b < k; b++) {
        if (hasHole) {
            // 空洞: 域转换副本(读回段后写, corr 段消费) + 逐行 m_c/l_c + spanRows 维度
            LocalTensor<T> gmC = rs[0];
            LocalTensor<T> gsC = rs[128];
            for (uint32_t si = 0; si < sliceNum; si++) {
                const PostRowSlice& slD = slices[si];
                AscendC::Adds(gmC[slD.chunkRowStart], globalMax[slD.localRowStart], 0.0f, slD.rowCount);
                AscendC::Adds(gsC[slD.chunkRowStart], globalSum[slD.localRowStart], 0.0f, slD.rowCount);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(diffSlice, pmT[b * Q_CHUNK_SIZE + chunkBase], gmC[chunkBase], spanRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(diffSlice, diffSlice, spanRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(corrAll[chunkBase], diffSlice, gsC[chunkBase], spanRows);
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<T> corrBrocH = rs[256]; // 空洞需 spanRows×8≤1024, 原 512 区越界
            Brcb(corrBrocH, corrAll[chunkBase], (spanRows + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            BSAvRowBroadcastMul(psUb[b * 2048], corrBrocH, spanRows, 32);
        } else {
            AscendC::Sub(diffSlice, pmT[b * Q_CHUNK_SIZE + chunkBase], globalMax[0], ownRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(diffSlice, diffSlice, ownRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(corrAll[chunkBase], diffSlice, globalSum[0], ownRows);
            AscendC::PipeBarrier<PIPE_V>();
            Brcb(corrBroc, corrAll[chunkBase], (ownRows + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            BSAvRowBroadcastMul(psUb[b * 2048], corrBroc, ownRows, 32);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice& sl = slices[s];
            for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                const uint32_t cnt = BSAMin(unitRows, sl.rowCount - uo);
                const uint32_t rel = sl.chunkRowStart + uo - chunkBase;
                const uint32_t u = (sl.localRowStart + uo) / unitRows;
                uint32_t uShape[] = {cnt, 32};
                AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(
                    unitRes[(u * k + b) * 32], psUb[b * 2048 + rel * 32], raTmp, uShape, true);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    // —— 整批 scale（行均值）+ invCols 列广播 + Cast（unitRes 布局 [unit][tile][32]）——
    if (invRowsUniform) {
        AscendC::Muls(unitRes, unitRes, float(1.0f / 16.0f), k * nUnits * 32);
    } else {
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice& sl = slices[s];
            for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                const uint32_t chunkRow = sl.chunkRowStart + uo;
                const uint32_t px = (qChunkStart + chunkRow) / unitRows;
                const uint32_t rowsInPx = BSAMin(unitRows, validXBlocks - px * unitRows);
                const uint32_t u = (sl.localRowStart + uo) / unitRows;
                for (uint32_t b = 0; b < k; b++) {
                    int32_t rowsInPxI = static_cast<int32_t>(rowsInPx);
                    AscendC::Muls(unitRes[(u * k + b) * 32], unitRes[(u * k + b) * 32],
                                  1.0f / static_cast<float>(rowsInPxI), 32);
                }
            }
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    {
        // invCols 列广播：unitRes[u][b][c] *= invCols[c]（src1RepStride=0 复用）
        AscendC::BinaryRepeatParams bpCols;
        bpCols.dstBlkStride = 1;
        bpCols.src0BlkStride = 1;
        bpCols.src1BlkStride = 1;
        bpCols.dstRepStride = 4;
        bpCols.src0RepStride = 4;
        bpCols.src1RepStride = 0;
        AscendC::Mul<T, false>(unitRes, unitRes, invCols, static_cast<uint64_t>(32), static_cast<uint8_t>(k * nUnits),
                               bpCols);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(fp16Row, unitRes, AscendC::RoundMode::CAST_ROUND, k * nUnits * 32);
    AscendC::PipeBarrier<PIPE_V>();

    // —— 批量写出：V→MTE3 一次同步 + 每 unit 一段 k×groupsValid 组连续（列跨 tile 连续）——
    SetFlag<HardEvent::V_MTE3>(eventSft);
    WaitFlag<HardEvent::V_MTE3>(eventSft);
    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice& sl = slices[s];
        for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
            const uint32_t chunkRow = sl.chunkRowStart + uo;
            const uint32_t px = (qChunkStart + chunkRow) / unitRows;
            const uint32_t u = (sl.localRowStart + uo) / unitRows;
            const uint64_t gmCol =
                pooledHeadOffset + static_cast<uint32_t>(px) * validPostYCur + t0 * (K_TILE_COLS / 16);
            // FIX-OUT-ALIGN: 写出宽 groupsValid×k×2B 非 32B 对齐时 MTE3 写侧间歇丢尾段
            // （与 radix 读入丢段同款硬件行为，FIX-RDX-ALIGN 修读侧后写侧独立存在）。
            // 双写规避：丢段率平方化；对齐块单写（大块翻倍实测 +1.4ms）
            const uint32_t outBytes = groupsValid * k * sizeof(OUT_T);
            const uint32_t writeReps = (outBytes < 32 || outBytes % 32 != 0) ? 2u : 1u;
            for (uint32_t wr = 0; wr < writeReps; wr++) {
                DataCopyPad(attnScoreGmTensor[gmCol], fp16Row[u * k * 32],
                            {1, static_cast<uint16_t>(groupsValid * k * sizeof(OUT_T)), 0, 0});
            }
        }
    }
    SetFlag<HardEvent::MTE3_V>(eventSft);
    WaitFlag<HardEvent::MTE3_V>(eventSft);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxFinalize(uint32_t qChunkStart, uint32_t qChunkSize,
                                                              uint32_t batchIdx, uint32_t headIdx,
                                                              uint32_t validXBlocks, uint32_t validYBlocks)
{
    if (!constInfo.usePostBlockShape) {
        return; // 非 post 模式由调用方走原 pass2
    }
    GlobalTensor<OUT_T> attnScoreGmTensor = postPoolOP.GetAttnScoreGm();
    const uint32_t validPostYCur = (validYBlocks + constInfo.postBlockShapeY - 1) / constInfo.postBlockShapeY;
    // chunk 上下文：与 pass1 共用（finalize(c) 在重排后先于 pass1(c) 执行，此处多为 miss 填充一次）
    if (cachedChunkKey != qChunkStart || cachedValidXBlocks != validXBlocks) {
        FillPass1Ctx(qChunkStart, qChunkSize, validXBlocks);
    }
    const PostRowSlice* slices = cachedSlices;
    const uint32_t sliceNum = cachedSliceNum;
    if (sliceNum == 0) {
        return;
    }
    const uint32_t unitRows = constInfo.postBlockShapeX; // 16
    // attnScore 单槽复用：逐 (b,h) 覆写偏移 0
    const uint64_t pooledHeadOffset = 0;

    // invCols 全 full tile 默认值（循环外一次构建；仅尾 tile 重算）
    LocalTensor<T> rs0 = postPoolOP.GetReduceTemp();
    LocalTensor<T> invCols = rs0[1536]; // 与 SoftmaxFinalizeBatch 的 rs 布局同步（BATCH=8 布局）
    AscendC::Duplicate(invCols, float(1.0f / 16.0f), 32);
    AscendC::PipeBarrier<PIPE_V>();

    // full/尾 tile 数按【本批 validYBlocks】取（原 constInfo.yBlocks 为总量/最大批值，多批
    // 变长（TND/seqlen）下批间有效列数不同——按总量算写宽度会跨 head 蔽写/漏写）
    const uint32_t lastFull = validYBlocks / K_TILE_COLS; // 批内 full tile 数
    const uint32_t ktilesBatch = (validYBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    // postBlockShapeY != 16 走通用收尾（列宽=实际参数，支持跨 tile 列与尾列；慢路径）
    if (constInfo.usePostBlockShape && constInfo.postBlockShapeY != 16) {
        GeneralFinalize(slices, sliceNum, qChunkStart, batchIdx, headIdx, validXBlocks, validYBlocks);
        return;
    }
    constexpr uint32_t BATCH = 8; // 每 batch 处理的 tile 数
    // full tile 按 BATCH=8 批处理
    for (uint32_t t0 = 0; t0 < lastFull; t0 += BATCH) {
        const uint32_t k = BSAMin(BATCH, lastFull - t0);
        SoftmaxFinalizeBatch(slices, sliceNum, qChunkStart, batchIdx, headIdx, validXBlocks, validYBlocks, t0, k,
                             K_TILE_COLS / 16, false);
    }
    // 尾 tile（批内 validYBlocks % K_TILE_COLS ≠ 0 时）单批收尾；写宽度 = 余数列组的上取整
    if (lastFull < ktilesBatch) {
        const uint32_t groupsValid = (validYBlocks - lastFull * K_TILE_COLS + 15) / 16;
        SoftmaxFinalizeBatch(slices, sliceNum, qChunkStart, batchIdx, headIdx, validXBlocks, validYBlocks, lastFull, 1,
                             groupsValid, true);
    }
}

// —— 通用收尾（postBlockShapeY != 16 慢路径）——
// 列 c 覆盖 yBlocks [c*pbY, (c+1)*pbY)，组 g 覆盖 [g*foldW, (g+1)*foldW)；foldW | pbY（tiling 保证）
// → 列 c = 全局组 [c*P, (c+1)*P)（P = pbY/foldW），全局组号跨 k-tile 连续，天然支持列跨 tile；
// 列均值按实际宽度（尾列 min((c+1)*pbY, validYBlocks) - c*pbY）。
// UB（scoreUb 相位独占，32768 fp32）：psUb[ownRows][G] | pm/corr/corrBroc | unitGrp[G]
//   | acc[nUnits][groupsWin]（列对齐分窗）| outRow/outF16（窗内列）。
template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralFinalize(const PostRowSlice* slices, uint32_t sliceNum,
                                                              uint32_t qChunkStart, uint32_t batchIdx, uint32_t headIdx,
                                                              uint32_t validXBlocks, uint32_t validYBlocks)
{
    GlobalTensor<OUT_T> attnScoreGmTensor = postPoolOP.GetAttnScoreGm();
    const uint32_t pbY = constInfo.postBlockShapeY;
    const uint32_t foldW = (pbY % 16 == 0) ? 16u : 8u; // 仅 post 模式可达（分派已 gate usePostBlockShape）
    const uint32_t G = K_TILE_COLS / foldW;            // 每 k-tile 组数（32/64）
    const uint32_t P = pbY / foldW;                    // 每列组数
    const uint32_t unitRows = constInfo.postBlockShapeX;
    const uint32_t validPostYCur = (validYBlocks + pbY - 1) / pbY;
    const uint32_t globalGroups = (validYBlocks + foldW - 1) / foldW;
    const uint32_t KTILES = (constInfo.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    // attnScore 单槽复用：逐 (b,h) 覆写偏移 0
    const uint64_t pooledHeadOffset = 0;

    const uint32_t ownRows = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
    // FIX-CHUNK-DOMAIN(B''): 同 Batch
    const uint32_t spanRows =
        slices[sliceNum - 1].chunkRowStart + slices[sliceNum - 1].rowCount - slices[0].chunkRowStart;
    const bool hasHole = (spanRows != ownRows);
    const uint32_t chunkBase = slices[0].chunkRowStart;
    const uint32_t nUnits = (ownRows + unitRows - 1) / unitRows; // unitRows≥8（BLOCK_SHAPE_ALIGN）→ ≤16

    LocalTensor<T> rsG = postPoolOP.GetReduceTemp();
    LocalTensor<uint8_t> raTmp0 = rsG[0].ReinterpretCast<uint8_t>();
    LocalTensor<T> ub = scoreUb;
    LocalTensor<T> psUb = ub[0];        // [ownRows ≤ 128][G ≤ 64] = 8192
    LocalTensor<T> pmRow = ub[8192];    // [128]
    LocalTensor<T> corrRow = ub[8320];  // [128]
    LocalTensor<T> corrBroc = ub[8448]; // [128][8] = 1024
    LocalTensor<T> unitGrp = ub[9472];  // [G ≤ 64]
    // VEC 32B 对齐约束：窗口起点须满足 c0*P%8==0（stepCols=8/gcd(P,8)），acc 行距 8 倍数，
    // outF16 基址 8 元素对齐（否则 aivector UB 未对齐异常挂死，plog 实锤）
    const uint32_t accBase = 9600;
    const uint32_t accCap = 32768 - accBase - 768;
    uint32_t gcdP8 = P;
    uint32_t gcdR = 8;
    while (gcdR != 0) { // gcd(P, 8)
        const uint32_t t = gcdP8 % gcdR;
        gcdP8 = gcdR;
        gcdR = t;
    }
    const uint32_t stepCols = 8 / gcdP8;
    // 窗口列数恒为 stepCols 倍数（≥stepCols）：保证窗口起点 c0*P%8==0 → Add 源 unitGrp[lg0]
    // 与 acc 偏移均 32B 对齐（小 validPostY 也须如此，c1 由 validPostYCur 收敛）
    const uint32_t colsCap = accCap / nUnits / P;
    // 上限 256：固定尾区布局（outRow [31744,32000) fp32 + outF16 [32256..) fp16 2KB 间距
    // + colScale [32512,32768)——Cast 相邻/重叠布局实测 VEC 越界（n1 实锤），固定大间距稳定）
    const uint32_t colsPerWin = BSAMax(stepCols, (BSAMin(colsCap, uint32_t(256)) / stepCols) * stepCols);
    const uint32_t groupsWinA = (colsPerWin * P + 7) / 8 * 8; // acc 行距（8 倍数）
    LocalTensor<T> acc = ub[accBase];                         // [nUnits][groupsWinA]
    LocalTensor<T> outRow = ub[31744];                        // [≤256] fp32 尾区（固定偏移）
    LocalTensor<T> colScale = ub[32512]; // [≤256] fp32 列缩放（固定尾区，窗口开头标量构建）
    LocalTensor<OUT_T> outF16 = ub[32256].ReinterpretCast<OUT_T>(); // [512B] fp16 尾区（固定偏移）

    for (uint32_t c0 = 0; c0 < validPostYCur; c0 += colsPerWin) {
        const uint32_t c1 = BSAMin(c0 + colsPerWin, validPostYCur);
        const uint32_t gLo = c0 * P;
        const uint32_t gHi = BSAMin(c1 * P, globalGroups);
        const uint32_t winGroups = gHi - gLo;

        // —— 窗口复位 + 排空（上一窗写出 M3 与 V 均已由窗尾 MTE3_V 收敛，此处再成对保险）——
        SetFlag<HardEvent::MTE3_MTE2>(eventFin);
        WaitFlag<HardEvent::MTE3_MTE2>(eventFin);
        SetFlag<HardEvent::V_MTE2>(eventFin);
        WaitFlag<HardEvent::V_MTE2>(eventFin);
        AscendC::Duplicate(acc, float(0.0f), nUnits * groupsWinA);
        AscendC::PipeBarrier<PIPE_V>();

        // —— 逐 k-tile：读 partSum/partMax 槽 → corr 修正 → RA 行折叠 → 累加到全局组窗 ——
        for (uint32_t t = 0; t < KTILES; t++) {
            const uint32_t tG = t * G;
            if (tG >= gHi || tG + G <= gLo) {
                continue; // 本 tile 组区间与窗口无交
            }
            const uint64_t slotB = (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + t) * Q_CHUNK_SIZE * G;
            const uint64_t pmSlotB = (static_cast<uint64_t>(constInfo.aivIdx) * KTILES + t) * Q_CHUNK_SIZE;
            SetFlag<HardEvent::V_MTE2>(eventFin);
            WaitFlag<HardEvent::V_MTE2>(eventFin);
            for (uint32_t off = 0; off < ownRows * G; off += 1024) {
                const uint32_t n = BSAMin(uint32_t(1024), ownRows * G - off);
                DataCopyPad(psUb[off], partSumGmTensor[slotB + chunkBase * G + off],
                            {1, static_cast<uint16_t>(n * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            }
            DataCopyPad(pmRow[0], partMaxGmTensor[pmSlotB + chunkBase],
                        {1, static_cast<uint16_t>(ownRows * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            // FIX-CHUNK-DOMAIN(B''): 补洞读
            if (hasHole) {
                for (uint32_t si = 0; si < sliceNum; si++) {
                    const PostRowSlice& slH = slices[si];
                    if (slH.chunkRowStart >= chunkBase + ownRows) {
                        DataCopyPad(
                            psUb[slH.chunkRowStart * G], partSumGmTensor[slotB + chunkBase * G + slH.chunkRowStart * G],
                            {1, static_cast<uint16_t>(slH.rowCount * G * sizeof(float)), 0, 0}, {false, 0, 0, 0});
                        DataCopyPad(pmRow[slH.chunkRowStart], partMaxGmTensor[pmSlotB + chunkBase + slH.chunkRowStart],
                                    {1, static_cast<uint16_t>(((slH.rowCount + 7u) / 8u * 8u) * sizeof(float)), 0, 0},
                                    {false, 0, 0, 0});
                    }
                }
            }
            SetFlag<HardEvent::MTE2_V>(eventFin);
            WaitFlag<HardEvent::MTE2_V>(eventFin);

            // corr = exp(m_t − m_c) / l_c（逐行）
            if (hasHole) {
                LocalTensor<T> gmC = ub[32000];
                LocalTensor<T> gsC = ub[32064];
                for (uint32_t si = 0; si < sliceNum; si++) {
                    const PostRowSlice& slD = slices[si];
                    AscendC::Adds(gmC[slD.chunkRowStart], globalMax[slD.localRowStart], 0.0f, slD.rowCount);
                    AscendC::Adds(gsC[slD.chunkRowStart], globalSum[slD.localRowStart], 0.0f, slD.rowCount);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(corrRow, pmRow, gmC[chunkBase], spanRows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(corrRow, corrRow, spanRows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(corrRow, corrRow, gsC[chunkBase], spanRows);
                AscendC::PipeBarrier<PIPE_V>();
                Brcb(corrBroc, corrRow, (spanRows + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                BSAvRowBroadcastMul(psUb, corrBroc, spanRows, G);
            } else {
                AscendC::Sub(corrRow, pmRow, globalMax[0], ownRows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Exp(corrRow, corrRow, ownRows);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(corrRow, corrRow, globalSum[0], ownRows);
                AscendC::PipeBarrier<PIPE_V>();
                Brcb(corrBroc, corrRow, (ownRows + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM, {1, 8});
                AscendC::PipeBarrier<PIPE_V>();
                BSAvRowBroadcastMul(psUb, corrBroc, ownRows, G);
            }
            AscendC::PipeBarrier<PIPE_V>();

            // 逐 unit 段 RA 行折叠 → 本 tile 与窗口交叠组段累加进 acc
            const uint32_t lg0 = (gLo > tG) ? (gLo - tG) : 0; // 交叠起始本地组
            const uint32_t lg1 = BSAMin(G, gHi - tG);         // 交叠结束本地组
            for (uint32_t s = 0; s < sliceNum; s++) {
                const PostRowSlice& sl = slices[s];
                for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                    const uint32_t cnt = BSAMin(unitRows, sl.rowCount - uo);
                    const uint32_t rel = sl.chunkRowStart + uo - chunkBase;
                    const uint32_t u = (sl.localRowStart + uo) / unitRows;
                    uint32_t uShape[] = {cnt, G};
                    AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(unitGrp, psUb[rel * G], raTmp0, uShape,
                                                                              true);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(acc[u * groupsWinA + (tG + lg0 - gLo)], acc[u * groupsWinA + (tG + lg0 - gLo)],
                                 unitGrp[lg0], lg1 - lg0);
                    AscendC::PipeBarrier<PIPE_V>();
                }
            }
        }

        // 行均值：acc=Σ_rows 部分和，需除以 px 实际行数（对齐 SoftmaxFinalizeBatch 的 1/unitRows 语义）。
        // 遗漏导致折叠分暂存 = 行和而非行均值（postBlockShapeX=8 时恰 8×，dump 实证）；
        // top-k 尺度不变故 mask/boundary 未暴露，仅 pooled_score 输出张量受影响。
        // 置于标量读 acc 的 PIPE_ALL 屏障之前，避免 V 写后标量读（2201 挂死模式）。
        if (validXBlocks % unitRows == 0) {
            AscendC::Muls(acc, acc, 1.0f / static_cast<float>(static_cast<int32_t>(unitRows)), nUnits * groupsWinA);
        } else {
            for (uint32_t s = 0; s < sliceNum; s++) {
                const PostRowSlice& sl = slices[s];
                for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                    const uint32_t chunkRow = sl.chunkRowStart + uo;
                    const uint32_t px = (qChunkStart + chunkRow) / unitRows;
                    const uint32_t rowsInPx = BSAMin(unitRows, validXBlocks - px * unitRows);
                    const uint32_t u = (sl.localRowStart + uo) / unitRows;
                    AscendC::Muls(acc[u * groupsWinA], acc[u * groupsWinA],
                                  1.0f / static_cast<float>(static_cast<int32_t>(rowsInPx)), winGroups);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        // —— 列组装（列均值按实际宽度）+ Cast + 写出 ——
        AscendC::PipeBarrier<PIPE_ALL>(); // 标量读 acc 前排空 V
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice& sl = slices[s];
            for (uint32_t uo = 0; uo < sl.rowCount; uo += unitRows) {
                const uint32_t chunkRow = sl.chunkRowStart + uo;
                const uint32_t px = (qChunkStart + chunkRow) / unitRows;
                const uint32_t u = (sl.localRowStart + uo) / unitRows;
                const LocalTensor<T> accRow = acc[u * groupsWinA];
                if (P == 1 && gHi == gLo + (c1 - c0)) {
                    // 列=组：预构建每列缩放（尾列按实际宽度）——标量写在任何 V op 之前
                    // （invCols 同模式），单次向量 Mul。原「Muls 后 GetValue/SetValue 修尾列」
                    // 在 2201 实测挂死（V 写后标量读触发 VEC 异常，tail 实锤）
                    for (uint32_t c = c0; c < c1; c++) {
                        const uint32_t width = BSAMin((c + 1) * pbY, validYBlocks) - c * pbY;
                        colScale.SetValue(c - c0, 1.0f / static_cast<float>(static_cast<int32_t>(width)));
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(outRow, accRow, colScale, winGroups);
                    AscendC::PipeBarrier<PIPE_V>();
                } else {
                    for (uint32_t c = c0; c < c1; c++) {
                        const uint32_t g0 = (c - c0) * P;
                        const uint32_t gN = BSAMin(P, gHi - c * P);
                        float sum = 0.0f;
                        for (uint32_t pj = 0; pj < gN; pj++) {
                            sum += accRow.GetValue(g0 + pj);
                        }
                        const uint32_t width = BSAMin((c + 1) * pbY, validYBlocks) - c * pbY;
                        outRow.SetValue(c - c0, sum / static_cast<float>(static_cast<int32_t>(width)));
                    }
                }
                // FIX-SCALAR-V: 标量 SetValue → V 读(Cast) 须 PIPE_ALL（PIPE_V 只同步 V 队列，
                // 标量写未落地时 Cast 读到 UB 残留——g05 尾列 c3=-3.77 垃圾实锤，间歇边沿性）
                AscendC::PipeBarrier<PIPE_ALL>(); // S 写 → V 可见
                AscendC::Cast(outF16, outRow, AscendC::RoundMode::CAST_ROUND, BSAMax(uint32_t(16), c1 - c0));
                AscendC::PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(eventFin);
                WaitFlag<HardEvent::V_MTE3>(eventFin);
                // WORKAROUND-MTE3: <32B 非对齐小块写出间歇丢列（m01 实锤）——小块双写
                // 丢块率平方化；≥32B 对齐块单写（大块翻倍实测 +1.4ms）
                const uint32_t outBytes = (c1 - c0) * sizeof(OUT_T);
                const uint32_t writeReps = (outBytes % 32 != 0) ? 2u : 1u; // FIX-WIN32(重放)
                for (uint32_t wr = 0; wr < writeReps; wr++) {
                    DataCopyPad(attnScoreGmTensor[pooledHeadOffset + static_cast<uint64_t>(px) * validPostYCur + c0],
                                outF16, {1, static_cast<uint16_t>((c1 - c0) * sizeof(OUT_T)), 0, 0});
                }
                SetFlag<HardEvent::MTE3_V>(eventFin);
                WaitFlag<HardEvent::MTE3_V>(eventFin);
            }
        }
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
    uint64_t tileBase = BSAScratchTileOffset(constInfo.aicIdx, qChunkStart, kChunkStart, constInfo.yBlocks);

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
        // DEBUG-UBR: pass2 读后 fp32 tile（mask 前），desc 编码 (b,h)；测完恢复

        // 尾块掩码：AIC 侧 nSize 对齐 16 写出的 padding 列是 L0C 残留，必须掩成 NEG_INF
        if (kChunkSize < kChunkSizeAlign) {
            // 同 pass1：kCols%8≠0 时 Duplicate 未对齐——尾段标量 + 对齐段 Duplicate
            const uint32_t padAlign2 = (kChunkSize + 7u) / 8u * 8u;
            for (uint32_t r = 0; r < bandRows; r++) {
                for (uint32_t c = kChunkSize; c < padAlign2; c++) {
                    scoreUb.SetValue(static_cast<uint32_t>(r) * kChunkSizeAlign + c, float(SOFTMAX_NEG_INF));
                }
            }
            if (padAlign2 < kChunkSizeAlign) {
                for (uint32_t r = 0; r < bandRows; r++) {
                    AscendC::Duplicate(scoreUb[static_cast<uint32_t>(r) * kChunkSizeAlign + padAlign2],
                                       float(SOFTMAX_NEG_INF), kChunkSizeAlign - padAlign2);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t i = s; i < e; i++) {
            const PostRowSlice& sl = slices[i];
            LocalTensor<T> tile = scoreUb[(sl.localRowStart - bandLocalStart) * kChunkSizeAlign];

            Muls(tile, tile, (float)(constInfo.scaleValue), sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();

            BSASubRowBroadcast(tile, tile, globalMaxBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                               kChunkSizeAlign);

            AscendC::Exp(tile, tile, sl.rowCount * kChunkSizeAlign);
            AscendC::PipeBarrier<PIPE_V>();
            // FIX-P2VMASK: pass2 重读的尾块掩码（读入后 NEG_INF）可被双读第二遍迟到写用 GM
            // raw padding(0) 覆盖（exp 前落地→exp(0−max) 污染尾列，basevar 62-146 波动 FAIL /
            // g07 464→28 波动实锤）。Exp 后兜底（值幂等方案，零借用零竞态）：
            // ① [kChunkSize, padAlign2) rem≤7 列标量 SetValue 0——迟到写覆盖也是 0，值幂等；
            // ② [padAlign2, kChunkSizeAlign) 8 对齐段 V 管 Duplicate 0——晚于一切 MTE2 写。
            // 尾列输出 0 与 golden exp(−inf)/sum=0 一致；有效列不受影响（迟到写携带 GM 正确
            // 分数，exp 前覆盖语义等价）
            if (kChunkSize < kChunkSizeAlign) {
                const uint32_t padAlign2 = (kChunkSize + 7u) / 8u * 8u;
                for (uint32_t r = 0; r < sl.rowCount; r++) {
                    for (uint32_t c = kChunkSize; c < padAlign2; c++) {
                        tile.SetValue(static_cast<uint32_t>(r) * kChunkSizeAlign + c, 0.0f);
                    }
                }
                if (padAlign2 < kChunkSizeAlign) {
                    for (uint32_t r = 0; r < sl.rowCount; r++) {
                        AscendC::Duplicate(tile[static_cast<uint32_t>(r) * kChunkSizeAlign + padAlign2], 0.0f,
                                           kChunkSizeAlign - padAlign2);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
            }

            BSADivRowBroadcast(tile, tile, globalSumBroc[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                               kChunkSizeAlign);

            if (constInfo.usePostBlockShape) {
                // 融合路径：归一化 FP32 tile 直接折叠为 attnScore 槽折叠分
                postPoolOP.FoldNormalizedSlice(tile, sl, kChunkStart, kChunkSize, kChunkSizeAlign, isLastKTile);
            } else {
                AscendC::Cast(scoreFp16Ub, tile, AscendC::RoundMode::CAST_ROUND, sl.rowCount * kChunkSizeAlign);
                AscendC::PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_MTE3>(eventSft);
                WaitFlag<HardEvent::V_MTE3>(eventSft);

                uint32_t sftOutQRow = qChunkStart + sl.chunkRowStart;
                uint64_t attnScoreOffset = static_cast<uint64_t>(sftOutQRow) * validYBlocks + kChunkStart;
                GlobalTensor<OUT_T> attnScoreChunkGm;
                attnScoreChunkGm.SetGlobalBuffer((__gm__ OUT_T*)(attnScoreFp16GmTensor.GetPhyAddr() + attnScoreOffset));
                // FIX-P2OUT32(重放): blockLen=kChunkSize×2B 非 32B 对齐(变长 kC=24→48B)双写
                for (uint32_t pwr = 0; pwr < 2; pwr++) {
                    DataCopyPad(attnScoreChunkGm[0], scoreFp16Ub,
                                {static_cast<uint16_t>(sl.rowCount), static_cast<uint32_t>(kChunkSize * sizeof(OUT_T)),
                                 0, static_cast<uint32_t>((validYBlocks - kChunkSize) * sizeof(OUT_T)), 0});
                }
                // MTE3 写出完成等待：SetFlag/WaitFlag 确保本核 MTE3 队列排空（GM 写落地）
                SetFlag<HardEvent::MTE3_V>(eventSft);
                WaitFlag<HardEvent::MTE3_V>(eventSft);
            }
        }
        s = e;
    }

    if (constInfo.usePostBlockShape) {
        postPoolOP.FlushTile();
    }
}

#endif // BSA_VEC_SM_SERVICE_H
