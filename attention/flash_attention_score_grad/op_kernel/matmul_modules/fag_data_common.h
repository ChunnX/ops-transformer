/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fag_data_common.h
 * \brief
 */

#ifndef FAG_DATA_COMMON_KERNEL_H_
#define FAG_DATA_COMMON_KERNEL_H_

namespace AscendC {

constexpr uint32_t NUM_10 = 10;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_448 = 448;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_512 = 512;

// mm-dyv/qk
constexpr uint8_t MM_DYV_INDEX = 0;
constexpr uint8_t MM_QK_INDEX = 1;
// mm-dq/k/v
constexpr uint8_t MM_DQ_INDEX = 2;
constexpr uint8_t MM_DK0_INDEX = 3;
constexpr uint8_t MM_DV_INDEX = 4;
constexpr uint8_t MM_DK1_INDEX = 5;

constexpr uint32_t TOTAL_L1_BUF_NUMS = 4; // 写死4块

/**
 * Template 1.2
 */
constexpr uint8_t TOTAL_MATMUL_NUMS = 5;
constexpr uint8_t T12_MM_DYV = 0;
constexpr uint8_t T12_MM_QK = 1;
constexpr uint8_t T12_MM_DQ = 2;
constexpr uint8_t T12_MM_DK = 3;
constexpr uint8_t T12_MM_DV = 4;

constexpr uint8_t T12_TOTAL_L1_BUF_NUMS = 7;
constexpr uint8_t T12_DY_PING = 0;
constexpr uint8_t T12_Q_PING = 1;
constexpr uint8_t T12_K_PING = 2;
constexpr uint8_t T12_DY_PONG = 3;
constexpr uint8_t T12_Q_PONG = 4;
constexpr uint8_t T12_K_PONG = 5;
constexpr uint8_t T12_TEMP = 6;

// matmul与对应pingpong buffer的关系表
constexpr int32_t MM_BUFFER_TABLE_A[TOTAL_MATMUL_NUMS][2] = {{T12_DY_PING, T12_DY_PONG},
                                                             {T12_Q_PING, T12_Q_PONG},
                                                             {T12_TEMP, T12_TEMP},
                                                             {T12_TEMP, T12_TEMP},
                                                             {T12_TEMP, T12_TEMP}};

constexpr int32_t MM_BUFFER_TABLE_B[TOTAL_MATMUL_NUMS][2] = {{T12_TEMP, T12_TEMP},
                                                             {T12_K_PING, T12_K_PONG},
                                                             {T12_K_PING, T12_K_PONG},
                                                             {T12_Q_PING, T12_Q_PONG},
                                                             {T12_DY_PING, T12_DY_PONG}};
/**
 * Template SAB
 */
constexpr uint32_t TSAB_MM345_SINGLE_A_SIZE = NUM_512 * NUM_512; // mm345的A矩阵single块的大小
constexpr uint32_t TSAB_MM345_BASE_A_SIZE = NUM_128 * NUM_128;   // mm345的A矩阵base块的大小
constexpr uint32_t TSAB_MM345_BASE_A_NUM =
    TSAB_MM345_SINGLE_A_SIZE / TSAB_MM345_BASE_A_SIZE; // mm345可以容纳的base块数量
// queue index
constexpr uint32_t BUF_0 = 0;
constexpr uint32_t BUF_1 = 1;
constexpr uint32_t BUF_2 = 2;

// 8字节以下不会影响通信开销，且需要4字节对齐
struct FagFlagData {
    uint8_t mmIdx;
    uint8_t pingPongIdx;
    uint8_t reserve[2];
};

// 定义L1 buffer全局管理结构体
struct GlobalTscmArray {
    __aicore__ inline GlobalTscmArray(){};
    TQue<TPosition::A1, 1> localQue;
    TBuffAddr srcAddr;
    int32_t cacheSize{0}; // 表示已缓存base块的个数
    int32_t bufferSize{0};
};

__BLOCK_LOCAL__ __inline__ GlobalTscmArray *tscmGlobal;

template <typename T, const STemplateType S1TEMPLATETYPE = STemplateType::NotAligned,
          const STemplateType S2TEMPLATETYPE = STemplateType::NotAligned,
          const DTemplateType DTEMPLATETYPE = DTemplateType::NotAligned>
__aicore__ inline void
InitTscmBuffer(TPipe *tPipe, GlobalTscmArray *tscmGlobal,
               const FlashAttentionScoreGradTilingDataS1s2Bn2gs1s2SameAb *__restrict ordTilingData)
{
    if (g_coreType == AIC) {
        // Tiling中需要保证分块不能超过L1 buffer的总大小
        if constexpr (S1TEMPLATETYPE == STemplateType::Aligned512 && S2TEMPLATETYPE == STemplateType::Aligned512) {
            if constexpr (DTEMPLATETYPE == DTemplateType::Aligned128) {
                tPipe->InitBuffer(tscmGlobal[BUF_0].localQue, 1, NUM_512 * NUM_128 * sizeof(T));
                tPipe->InitBuffer(tscmGlobal[BUF_1].localQue, 1, NUM_512 * NUM_128 * sizeof(T));
                tPipe->InitBuffer(tscmGlobal[BUF_2].localQue, 1, NUM_512 * NUM_256 * sizeof(T));
            } else if constexpr (DTEMPLATETYPE == DTemplateType::Aligned64) {
                /**
                 * Limit: d<=128
                 */
                int64_t dAlign = (ordTilingData->s1s2BNGS1S2BaseParams.d + 16 - 1) & ~(16 - 1);
                int64_t buf0Size = NUM_512 * (NUM_512 - dAlign);
                int64_t buf1Size = NUM_512 * dAlign;
                tPipe->InitBuffer(tscmGlobal[BUF_0].localQue, 1, buf0Size * sizeof(T));
                tPipe->InitBuffer(tscmGlobal[BUF_1].localQue, 1, buf1Size * sizeof(T));
                tscmGlobal[BUF_0].bufferSize = buf0Size / TSAB_MM345_BASE_A_SIZE;
            } else {
                tPipe->InitBuffer(tscmGlobal[BUF_0].localQue, 1, NUM_512 * NUM_256 * sizeof(T));
                tPipe->InitBuffer(tscmGlobal[BUF_1].localQue, 1, NUM_512 * NUM_256 * sizeof(T));
            }
        }
    }
}

template <typename T>
__aicore__ inline void InitSabTSCMBuffer(TPipe *pipeBase, GlobalTscmArray *tscmGlobal,
                                         const FlashAttentionScoreGradTilingDataS1s2Bn2 *__restrict ordTilingData)
{
    if (g_coreType == AIC) {
        int64_t d = ordTilingData->opInfo.D;
        int64_t s1CvInner = ordTilingData->splitCoreParams.singleM;
        int64_t s2CvInner = ordTilingData->splitCoreParams.singleN;
        int64_t dAlign = (d + 15) / 16 * 16;

        // 自主申请的L1Buffer总和不能超512*1024
        pipeBase->InitBuffer(tscmGlobal[T12_DY_PING].localQue, 1, s1CvInner * dAlign * sizeof(T)); // dy ping
        pipeBase->InitBuffer(tscmGlobal[T12_Q_PING].localQue, 1, s1CvInner * dAlign * sizeof(T));  // q ping
        pipeBase->InitBuffer(tscmGlobal[T12_K_PING].localQue, 1, s2CvInner * dAlign * sizeof(T));  // k ping
        pipeBase->InitBuffer(tscmGlobal[T12_DY_PONG].localQue, 1, s1CvInner * dAlign * sizeof(T)); // dy pong
        pipeBase->InitBuffer(tscmGlobal[T12_Q_PONG].localQue, 1, s1CvInner * dAlign * sizeof(T));  // q pong
        pipeBase->InitBuffer(tscmGlobal[T12_K_PONG].localQue, 1, s2CvInner * dAlign * sizeof(T));  // k pong
        pipeBase->InitBuffer(tscmGlobal[T12_TEMP].localQue, 1, NUM_128 * NUM_128 * sizeof(T)); // v, pScm, dsScm ping
    }
}

namespace Impl {
namespace Detail {

template <typename IMPL, class INPUT_TYPE, const auto &MM_CFG> class FagContext {
    using SrcT = typename INPUT_TYPE::TRANS_T;

public:
    __aicore__ inline FagContext()
    {
    }
    __aicore__ inline void Init()
    {
    }
    int32_t mmIdx{0};

    int32_t tscmIndexA = BUF_2;
    LocalTensor<SrcT> CurrentTensorA;
    int32_t baseBlockSizeA{0};
    bool needAllocA{true};
    bool needFreeA{false};
    bool isFirstBaseBlockA{true};
    bool isLastBaseBlockA{false};
    int32_t maxCacheSizePartA{0};

    int32_t tscmIndexB = BUF_2;
    LocalTensor<SrcT> CurrentTensorB;
    int32_t baseBlockSizeB{0};
    bool needAllocB{true};
    bool needFreeB{false};
    bool isFirstBaseBlockB{true};
    bool isLastBaseBlockB{false};
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // FAG_DATA_COMMON_KERNEL_H_
