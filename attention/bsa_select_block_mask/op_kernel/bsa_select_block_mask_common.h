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
 * \file bsa_select_block_mask_common.h
 * \brief
 */
#ifndef BSA_SELECT_BLOCK_MASK_COMMON_H
#define BSA_SELECT_BLOCK_MASK_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

constexpr uint8_t SYNC_V1_TO_C1_FLAG[2] = {0, 1};
constexpr uint8_t SYNC_C1_TO_V1_FLAG[9] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
constexpr uint32_t SYNC_C1_TO_V1_FLAG_NUMS = 9;
// NZ 快写背压回执环（AIV pass1 消费完 → 放行 AIC 复用同槽）：
// AIC 发第 i 个 C1_TO_V1 前须先收到第 i-9 个回执，领先幅度钉死 < 9 个 kChunk（一个复用周期）
constexpr uint8_t SYNC_V2_TO_C2_FLAG[9] = {11, 12, 13, 14, 15, 16, 17, 18, 19};
constexpr uint32_t CV_EXEC_RATIO = 4;
constexpr uint8_t SYNC_MODE = 2;
constexpr uint32_t BSA_FIXPIPE_SYNC_ANCHOR_M = 16;
constexpr uint32_t BSA_FIXPIPE_SYNC_ANCHOR_N = 16;
constexpr uint32_t BSA_FIXPIPE_SYNC_ANCHOR_ELEMS = BSA_FIXPIPE_SYNC_ANCHOR_M * BSA_FIXPIPE_SYNC_ANCHOR_N;

static constexpr uint32_t N_WORKSPACE_SIZE = 1024;
static constexpr uint32_t S1_BASE_STEP = 128;
static constexpr uint32_t S2_BASE_STEP = 1024;
static constexpr uint32_t CUBE_BASE_BLOCK = 128;
static constexpr uint32_t C0_SIZE = 16;
static constexpr uint32_t VEC_ALIGN_SIZE = 32;
static constexpr uint32_t CUBE_MATRIX_SIZE = 256;
static constexpr uint32_t AIC_AIV_RATIO = 2;
static constexpr uint32_t FLOAT_DATA_BLOCK_NUM = 8;
static constexpr uint32_t FP16_DATA_BLOCK_NUM = 16;
static constexpr uint32_t FLOAT_REPEAT_NUM = 64;
static constexpr uint32_t D_SIZE = 128;
static constexpr uint32_t Q_CHUNK_SIZE = 128;
static constexpr uint32_t K_CHUNK_SIZE = 128;
// score scratch 分块连续布局：每 kTile（K_TILE_COLS 列）在 bank 内连续存放，
// fixpipe 写/AIV 读均为大块连续传输（消除按行跨 yBlocks 的碎片访问）
static constexpr uint32_t K_TILE_COLS = K_CHUNK_SIZE * CV_EXEC_RATIO; // 512
static constexpr uint32_t REPEAT_SIZE = 256;
static constexpr uint32_t BROC_BASE_NUM = 8;
static constexpr uint32_t BYTE_SIZE = 8;
static constexpr uint16_t XOR_OP_VALUE_16 = 0x8000;
static constexpr uint32_t HALF_SAFE_MAX = 2048;
static constexpr uint32_t RADIX_BITS_PER_ROUND = 2;
static constexpr uint32_t RADIX_NUM_BINS = 4;
static constexpr uint32_t RADIX_ROUNDS_FP16 = 8;
static constexpr float SOFTMAX_NEG_INF = -1e38f;
static constexpr uint32_t MAX_TILE_NUM_IN_UB = 6144;
static constexpr uint32_t MAX_TILE_NUM_IN_UB_BY2 = 3072;
static constexpr uint32_t MAX_TILE_NUM_IN_UB_BY3 = 2048;
static constexpr uint32_t FLOAT32_SAFE_INT = 16777216;
static constexpr uint32_t BLOCK_SIZE = 196608;
static constexpr uint32_t UB_BLOCK_SIZE = 32;

enum class BSALayout {
    TND = 0,
    BNSD = 1
};

enum class BSASparseMode {
    TopK = 0,
    BottomK = 1
};

template <typename InputT, typename OutputT, BSALayout LayoutQ = BSALayout::BNSD, BSALayout LayoutKV = BSALayout::BNSD,
          typename... Args>
struct BSAType {
    using inputT = InputT;
    using outputT = OutputT;
    static constexpr BSALayout layoutQ = LayoutQ;
    static constexpr BSALayout layoutKV = LayoutKV;
};

struct BSAConstInfo {
    static constexpr uint32_t BUFFER_SIZE_BYTE_64 = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_128 = 128;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256 = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512 = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_60K = 60 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64K = 64 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_120K = 120 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_192K = 192 * 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_24K = 24 * 1024;

    uint32_t aicIdx;
    uint32_t aivIdx;
    uint32_t subBlockIdx;
    uint32_t aivNum;
    uint32_t aicNum;
    uint32_t activeYVecCoreNum;

    uint32_t batchSize;
    uint32_t numHeads;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t dSize;
    uint64_t blockShapeX;
    uint64_t blockShapeY;
    uint32_t xBlocks;
    uint32_t yBlocks;
    float scaleValue;
    float sparsity;
    uint64_t topKValue;
    BSASparseMode sparsityMode;

    uint32_t coreNum;
    uint32_t activeCoreNum;
    uint32_t rowsPerCore;
    uint32_t extraCores;
    uint32_t totalRows;
    uint32_t yBlocksPerCore;
    uint32_t extraYCores;
    uint32_t qChunkSize;
    uint32_t kChunkSize;

    uint64_t qCmpOffset;
    uint64_t kCmpOffset;
    uint64_t attnScoreOffset;
    uint64_t softmaxTmpOffset;
    uint64_t topkWorkspaceOffset;
    uint64_t pooledScoreOffset;
    uint64_t postStagingOffset;

    uint8_t usePostBlockShape;
    uint16_t postBlockShapeX;
    uint16_t postBlockShapeY;
    uint32_t postXBlocks;
    uint32_t postYBlocks;
};

// score scratch 分块连续布局偏移（元素单位）：
//   layout = [aicIdx][kTileIdx][chunkRow][K_TILE_COLS]
//   每 AIC 区大小 = Q_CHUNK_SIZE * yBlocks；每 tile 大小 = Q_CHUNK_SIZE * K_TILE_COLS
__aicore__ inline uint64_t BSAScratchTileOffset(uint32_t aicIdx, uint32_t kChunkStart, uint32_t yBlocks)
{
    uint64_t coreBase = static_cast<uint64_t>(aicIdx) * Q_CHUNK_SIZE * yBlocks;
    uint64_t tileIdx = kChunkStart / K_TILE_COLS;
    return coreBase + tileIdx * Q_CHUNK_SIZE * K_TILE_COLS;
}

struct BSARunInfo {
    uint32_t taskId;
    uint32_t batchIdx;
    uint32_t headIdx;
    uint32_t qBlockStart;
    uint32_t qBlockEnd;
    uint32_t curQBlocks;
    bool isValid;
};

template <typename T>
__aicore__ inline T BSACeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 BSAMax(T1 a, T2 b)
{
    return (a < b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 BSAMin(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T>
__aicore__ inline T BSAAlignTo(T n, T alignSize)
{
    if (alignSize == 0) {
        return 0;
    }
    return (n + alignSize - 1) & (~(alignSize - 1));
}

// 广播二元操作：ubIn1 为 Brcb 8 倍展开的逐行向量，跨 [row, col] 矩阵按行广播。
// 供 softmax 二趟（max/sum 归一化）与 post pool（逐行除 count）共用。
__aicore__ inline void BSASubRowBroadcast(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                          LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
{
    // 默认 col 按照32字节对齐
    // 分核逻辑的关系，矩阵shape不会超过[128, 128 * 5], 若启动俩个vector，shape不会超过[64, 128 * 5]
    // 所以，一次sub， 最大可计算[255, 64], 迭代次数为行数, 一次迭代计算元素大小最多为64
    // 所以，对列按照64切分，循环sub计算
    uint32_t countEachRepeat = 256 / sizeof(float);                   // 每次迭代计算的元素 64
    uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat; // 上取整
    uint32_t remain = col % countEachRepeat;
    uint64_t mask = countEachRepeat; // 参与计算的元素个数
    uint8_t repeatTimes = row;
    AscendC::BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src0RepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src1RepStride = 1;

    for (uint32_t i = 0; i < colLoop; i++) {
        if (i == colLoop - 1 && remain != 0) {
            mask = remain;
        }
        AscendC::Sub(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// 广播除法：ubIn1 为 Brcb 8 倍展开的逐行向量，跨 [row, col] 矩阵按行广播。
__aicore__ inline void BSADivRowBroadcast(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                          LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
{
    uint32_t countEachRepeat = 256 / sizeof(float);                   // 每次迭代计算的元素 64
    uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat; // 上取整
    uint32_t remain = col % countEachRepeat;
    uint64_t mask = countEachRepeat; // 参与计算的元素个数
    uint8_t repeatTimes = row;
    AscendC::BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src0RepStride = col / FLOAT_DATA_BLOCK_NUM;
    repeatParams.src1RepStride = 1;

    for (uint32_t i = 0; i < colLoop; i++) {
        if (i == colLoop - 1 && remain != 0) {
            mask = remain;
        }
        AscendC::Div(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

#endif // BSA_SELECT_BLOCK_MASK_COMMON_H
