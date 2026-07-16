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
 * \file quant_lightning_indexer_v2_common.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_COMMON_H
#define QUANT_LIGHTNING_INDEXER_V2_COMMON_H
using namespace AscendC;

// ===================== MXFP8 类型定义 =====================
using FP8E4M3 = fp8_e4m3fn_t;
using FP8E8M0 = fp8_e8m0_t;

constexpr uint32_t MXFP8_BLOCK_SIZE = 32;  // 每32个D维度元素一个scale
constexpr uint32_t FP8_TWO = 2;            // 2个fp8_e8m0打包成1个bf16

// ===================== AIC:AIV核数比值宏定义 =====================
// CUBE与VEC核数比值（AIC:AIV Ratio）
#define CVRATIO 1

namespace QLIV2Common {

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename QK_T, typename SCORE_T, typename OUT_T,
    const bool PAGE_ATTENTION = false,
          LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BBND,
          typename W_T = float, typename SCALE_T = half, typename... Args>
struct QLIV2Type {
    static_assert(
        (std::is_same_v<QK_T, float> &&
         (std::is_same_v<SCORE_T, uint32_t> || std::is_same_v<SCORE_T, uint16_t>)) ||
        (std::is_same_v<QK_T, bfloat16_t> &&
         std::is_same_v<SCORE_T, uint16_t>),
        "Invalid combination of QK_T and SCORE_T"
    );
    using queryType = Q_T;
    using keyType = K_T;
    using queryKeyType = QK_T;
    using scoreType = SCORE_T;
    using outputType = OUT_T;
    using weightType = W_T;
    using scaleType = SCALE_T;

    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
    static constexpr bool isMxFp8 = std::is_same_v<Q_T, FP8E4M3> && std::is_same_v<SCALE_T, FP8E8M0>;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;
    uint32_t s2Start;
    uint32_t s2LoopEnd;
    uint32_t validS2Len;
    uint32_t qScaleLoop;
    uint32_t kScaleLoop;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    uint32_t curCuSeqlensQ;
    uint32_t curCuSeqlensK;
    uint32_t curSequsedQ;
    uint32_t curSequsedK;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorQScaleOffset;   // MXFP8: qScale在Cube核使用的偏移
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;
    uint64_t valueOutOffset;
    uint64_t outputIdxCoreOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
    bool isNeedLD = false;
    bool isOutputIdxOffsetValid = false;
    bool needTndPadding = false;
    uint32_t saveWorkSpaceIdx = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t QLIV2_SYNC_MODE4 = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;
    static constexpr uint32_t CROSS_VC_EVENT = 0;
    static constexpr uint32_t CROSS_CV_EVENT = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;
    static constexpr uint16_t NEG_INF_BFLOAT = 0xFF80;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;              // topK选取大小
    uint64_t kSeqSize = 0ULL;          // kv最大S长度
    uint64_t qSeqSize = 1ULL;          // q最大S长度
    uint32_t kCacheBlockSize = 0;      // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0;  // PA场景的最大单batch block number
    LI_LAYOUT outputLayout;            // 输出的格式
    bool attenMaskFlag = false;
    uint32_t cmpRatio = 1;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    int32_t maxSeqlenQ = -1;
    
    uint32_t actualLenQDims = 0U;  // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;   // KV 的actualSeqLength 的维度
    uint32_t cmpResiduaKLenDims = 0U;   // cmpResidualK的维度
    bool isAccumSeqS1 = false;     // 是否累加模式
    bool isAccumSeqS2 = false;     // 是否累加模式
    bool isLDOpen = false;
    bool returnValue = false;
};

struct LdSplitCoreInfo {
    bool isLdCoreEnable = false;     // 当前核是否参与规约任务
    uint32_t saveWorkSpaceIdx = 0U;  // 存放LD参数的地址
    uint32_t bn2Idx = 0U;            // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U;      // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U;      // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0U;  // 最终输出索引搬出Topk的初始偏移地址
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;  // S2的起始位置
    uint32_t s2End = 0U;    // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;  // 当前核是否需要进行Decode归约任务
    bool isCoreEnable = false;
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
}  // namespace QLIV2Common

// ============================================================================
// bank冲突优化 — 新芯片 bank 布局
// ============================================================================
// 新芯片 Vector UB = 384KB, bank 布局:
//   8 bank groups × 4 banks/group × 32B/bank
//   每 1KB 地址范围内:
//     bank0 = [0,   255B], bank1 = [256, 511B]
//     bank2 = [512, 767B], bank3 = [768,1023B]
//   每个 bank 含 2 个 sub-bank × 16B (950 为 4×8B)
//
// 冲突判定: 两地址 A,B 同 bank ⟺
//   (A/256)%4 == (B/256)%4  且  (A/1024)%8 == (B/1024)%8
//   └── bank index (1KB内4选1) ──┘  └── group index (8KB内8选1) ──┘
//
// 解冲突手段:
//   ① 错开 nK+256/512/768B → 落同 group 不同 bank
//   ② 错开 nK+1KB → 落不同 group
//   ③ 交换读写方向 (读/写不冲突)
// ============================================================================

#define UB_BLOCK              32   // 32B (不变)
#define UB_BANK_GROUPS        8    // 8 个 bank group (不变)
#define UB_BANKS              4    // 每 group 4 个 bank (原 2 → 4)
#define UB_BANK_DEPTH         384  // 384KB / 1024B = 384 行 (原 512)

// UB_BANK_STRIDE=256B: 新芯片语义为"同 group 内跨 bank" (原 950 为"跨 group 同 bank")
//   ping-pong 偏移 256B 在新芯片上天然跨 bank → ping/pong 无冲突 (利好, 无需改动)
#define UB_BANK_STRIDE        (UB_BANK_GROUPS * UB_BLOCK)               // 256B (值不变, 语义变更)

// UB_BANK_GROUP_STRIDE: 一个 bank group 宽度 = 4×256B = 1024B (原 32B)
#define UB_BANK_GROUP_STRIDE  (UB_BANKS * UB_BANK_STRIDE)               // 1024B

// ⚠️ UB_BANK_DEPTH_STRIDE 保持 512B 不变!
//   原因: vector1.h 内层循环 "qk + 128*i" 的 128 是硬编码 S2Base(float元素数),
//         锁死 rowStride = 128 float = 512B。若改为 1024B, g维读取错位 → 算子崩。
//   注意: 此处 512B = UB_BANKS_OLD(2) × UB_BANK_GROUPS(8) × UB_BLOCK(32),
//         而非新芯片的 UB_BANKS(4) × ... = 1024B。
#define UB_BANK_DEPTH_STRIDE  (2 * UB_BANK_GROUPS * UB_BLOCK)           // 512B (锁死, 勿改!)

// ============================================================================
// bank 偏移宏 — 用于将同时访问的 buffer 散到不同 bank/group
// ============================================================================
// QK 4路并发 (MulWeightAndReduceSum2_MxFP8):
//   A = qk0 + 128*i               (s1行A, g=i)
//   C = qk0 + qkStride + 128*i    (s1行B, g=i)  → qkStride 附加 +256B → bank+1
//   B = qk0 + qkVLStride + 128*i  (配对路,  g=i) → qkVLStride 附加 +512B → bank+2 (TODO, 见下)
//   D = qk0 + qkStride + qkVLStride + 128*i     → bank+3 (自动)
//
// 跨 buffer:
//   weight/weightTemp 与 QK 并发读 → base 错 bank
//   out 写与 QK 读不并发 (循环后), 但仍错开 group 防边界冲突
// ============================================================================

// qkStride 附加偏移: s1行B 落 bank+1 (解 A/C 同 bank 冲突)
#define QLIV2_BANK_OFF_QK_S1     256    // 256B → bank+1

// qkVLStride 附加偏移: 配对路落 bank+2 (解 A/B 同 bank 冲突)
// ⚠️ TODO: 需确认 cube Fixpipe 写配对路(pong块)的路径后才能启用。
//    当前不启用 (vector端不加, cube端不加), 配对路冲突保持, A/B 仍同 bank。
//    启用方法: vector 端 qkVLstride += (QLIV2_BANK_OFF_QK_VL/sizeof(QK_T)),
//              cube 端 Fixpipe 写配对路时同步偏移。
#define QLIV2_BANK_OFF_QK_VL     512    // 512B → bank+2

// weight base 偏移 → bank3 (与 QK bank0/1 错开)
#define QLIV2_BANK_OFF_WEIGHT    768    // 768B → bank3

// weightTemp base 偏移 → 跨 group1 (与 QK group0 彻底分开)
#define QLIV2_BANK_OFF_WTEMP     1024   // 1KB → group1

// out base 偏移 → 跨 group1 (与 QK group0 彻底分开)
#define QLIV2_BANK_OFF_OUT       1024   // 1KB → group1

#endif  // QUANT_LIGHTNING_INDEXER_V2_COMMON_H