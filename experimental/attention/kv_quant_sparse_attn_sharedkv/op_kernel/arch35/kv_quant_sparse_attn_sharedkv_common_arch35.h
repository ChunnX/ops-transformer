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
 * \file kv_quant_sparse_attn_sharedkv_common_arch35.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_AHSREDKV_COMMON_ARCH35_H
#define KV_QUANT_SPARSE_ATTN_AHSREDKV_COMMON_ARCH35_H
#include <type_traits>
#include "kernel_tiling/kernel_tiling.h"
#include "../kv_quant_sparse_attn_sharedkv_common.h"

constexpr uint64_t BLOCK_BYTE = 32;
constexpr uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;

constexpr uint32_t BUFFER_SIZE_8K = 8192; // 8192表示8 * 1024

constexpr uint32_t BUFFER_SIZE_16K = 16384; // 16384表示16 * 1024
constexpr uint32_t BUFFER_SIZE_32K = 32768; // 32768表示32 * 1024
constexpr uint32_t BUFFER_SIZE_64K = 65536; // 65536表示64 * 1024
constexpr uint32_t BUFFER_SIZE_96K = 98304; // 98304表示96 * 1024
constexpr uint32_t BUFFER_SIZE_256K = 262144; // 262144表示256 * 1024

constexpr uint32_t CV_RATIO = 2;
constexpr uint64_t SYNC_MODE = 4;

// ===== C 侧 buffer 元素个数 (element count, 用于 tensor 偏移及地址递增) =====
constexpr uint32_t L1Q_ELEM_PER_BUF       = 16384;     // Q_T 元素, 每个 L1Q buffer
constexpr uint32_t L1_RIGHT_ELEM_PER_BLOCK = 65536;     // Q_T 元素, 每个 L1Right 块
constexpr uint32_t L0A_ELEM_PER_BUF       = 8192;       // Q_T 元素, 每个 L0A buffer
constexpr uint32_t L0B_ELEM_PER_BUF       = 16384;      // Q_T 元素, 每个 L0B buffer
constexpr uint32_t L0C_ELEM_PER_BUF       = 32768;      // T   元素, 每个 L0C buffer

// ===== V 侧主流程核内 MutexID (范围 0~7, 上限 27) =====
#define INNERCORE_STAGE0_IN(s)       (s)           // 0, 1: UB stage0In, MTE2->V
#define INNERCORE_STAGE0_OUT(s)      (2 + (s))     // 2, 3: UB stage0Out, V->MTE3
#define INNERCORE_STAGE1(s)          (4 + (s))     // 4, 5: UB stage1Out, V->MTE3
#define INNERCORE_STAGE2             (6)           // 6:   UB stage2Out, V->MTE3

// TODO sinks init阶段执行 可复用id 与向量化一起分配
#define INNERCORE_SINKS_SYNC         (7)           // 7:   UB sinksBuf, MTE2->V
// ===== GetKVPhyAddr 核内 MutexID (独立阶段, ID 可与主流程复用) =====
#define INNERCORE_PHYADDR_BLKTABLE   (0)
#define INNERCORE_PHYADDR_SPARSEIDX  (1)
#define INNERCORE_PHYADDR_KVADDR     (2)

// ===== C 侧核内 MutexID (范围 0~12, 上限 27) =====
#define INNERCORE_L1Q(s)             (s)           // 0, 1, 2: L1 Q, MTE2->MTE1
#define INNERCORE_L1KV(s)            (3 + (s))     // 3~8:     L1 KV shared, MTE2->MTE1
#define INNERCORE_L0AB(s)            (9 + (s))     // 9, 10:   L0A/L0B shared, MTE1->M
#define INNERCORE_L0C(s)             (11 + (s))    // 11, 12:  L0C, M->FIX

// ===== 跨核 flag ID (正向反向共用, +16 dual-dest sub-block 1, 上限 15) =====
#define CROSSCORE_BMM1(s)            (6 + s)           // 0, 1: bmm1 DB
#define CROSSCORE_BMM1_D(s)          (6 + 16 + (s))    // 16, 17: bmm1 dual-dest
#define CROSSCORE_BMM2               (8)           // 2:    bmm2 Single
#define CROSSCORE_BMM2_D             (24)          // 18:   bmm2 dual-dest
#define CROSSCORE_L1P(s)            (9 + s)           // 0, 1: L1 P buffer forward sync (AIV→AIC)
#define CROSSCORE_L1P_D(s)            (9 + 16 + s)

template <typename ElemT>
struct StaticBuffer {
    LocalTensor<ElemT> tensor;
    uint32_t idx;
};

template <typename ElemT>
struct RingBuffer {
    StaticBuffer<ElemT> *bufs;
    uint32_t             bufNum;
    uint32_t             curId;

    __aicore__ inline RingBuffer() : bufs(nullptr), bufNum(0), curId(0) {}
    __aicore__ inline RingBuffer(StaticBuffer<ElemT> *b, uint32_t n)
        : bufs(b), bufNum(n), curId(n - 1) {}

    __aicore__ inline StaticBuffer<ElemT> &GetNext()
    {
        curId = (curId + 1) % bufNum;
        return bufs[curId];
    }
};

constexpr uint32_t s1BaseSize = 64;
constexpr uint32_t s2BaseSize = 128;
constexpr uint32_t bSize = 1;
constexpr uint32_t n2Size = 1;
constexpr uint32_t gSize = 128;
constexpr uint32_t s1Size = 4096;
constexpr uint32_t s2Size = 8192;
constexpr uint32_t dSize = 512;
constexpr uint32_t dSizeV = 512;
constexpr uint32_t dSizeVInput = 640;
constexpr uint32_t dSizeRope = 64;
constexpr uint32_t dSizeNope = dSize - dSizeRope;
constexpr int32_t oriWinLeft = 127;
constexpr int32_t oriWinRight = 0;
constexpr uint32_t cmpRatio = 4;
constexpr uint32_t sparseBlockCount = 1024;
constexpr uint32_t sparseBlockSize = 1;
constexpr bool isActualLenDimsKVNull = false;
constexpr uint32_t actualSeqLenSize = bSize + 1;
constexpr uint32_t actualSeqLenKVSize = bSize;
constexpr uint32_t oriBlockSize = 128;
constexpr uint32_t cmpBlockSize = 128;
constexpr uint32_t oriMaxBlockNumPerBatch = 64;
constexpr uint32_t cmpMaxBlockNumPerBatch = 64;
constexpr uint32_t oriKvStride = oriBlockSize * dSizeVInput;
constexpr uint32_t cmpKvStride = cmpBlockSize * dSizeVInput;
constexpr float softmaxScale = 0.04419417;

namespace BaseApi {
__aicore__ constexpr uint64_t Align2Func(uint64_t data) {
    return (data + 1UL) >> 1UL << 1UL; // 向上2对齐, +1移位2
}

__aicore__ constexpr uint64_t Align8Func(uint64_t data) {
    return (data + 7UL) >> 3UL << 3UL; // 向上8对齐, +7移位3
}

__aicore__ constexpr uint64_t Align16Func(uint64_t data) {
    return (data + 15UL) >> 4UL << 4UL; // 向上16对齐, +15移位4
}

__aicore__ constexpr uint64_t Align64Func(uint64_t data) {
    return (data + 63UL) >> 6UL << 6UL; // 向上64对齐, +63移位6
}
}

#define TEMPLATE_INTF \
    template <typename Q_T, typename KV_T, typename T, typename OUTPUT_T, bool isFd, bool isPa, SAS_LAYOUT LAYOUT_T, \
    SAS_LAYOUT KV_LAYOUT_T, SASTemplateMode TEMPLATE_MODE, bool IS_SPLIT_G, bool IS_VEC_S2PHYADDR>

#define TEMPLATE_INTF_ARGS \
    Q_T, KV_T, T, OUTPUT_T, isFd, isPa, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE, IS_SPLIT_G, IS_VEC_S2PHYADDR

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X) \
    X(Q_T) \
    X(KV_T) \
    X(T) \
    X(OUTPUT_T) \

#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(isFd, bool, false) \
    X(isPa, bool, true) \
    X(LAYOUT_T, SAS_LAYOUT, SAS_LAYOUT::BSND) \
    X(KV_LAYOUT_T, SAS_LAYOUT, SAS_LAYOUT::PA_ND) \
    X(TEMPLATE_MODE, SASTemplateMode, SASTemplateMode::SCFA_TEMPLATE_MODE) \
    X(IS_SPLIT_G, bool, false) \
    X(IS_VEC_S2PHYADDR, bool, false)


/* 1. 生成带默认值的模版Template */
#define GEN_TYPE_PARAM(name) typename name,
#define GEN_CONST_PARAM(name, type, default_val) type name = default_val,

#define TEMPLATES_DEF \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>

/* 2. 生成不带带默认值的模版Template */
#define GEN_TEMPLATE_TYPE_NODEF(name) typename name,
#define GEN_TEMPLATE_CONST_NODEF(name, type, default_val) type name,
#define TEMPLATES_DEF_NO_DEFAULT \
template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TEMPLATE_TYPE_NODEF) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TEMPLATE_CONST_NODEF) bool end>

/* 3. 生成有默认值的Args */
#define GEN_ARG_NAME(name, ...) name,
#define TEMPLATE_ARGS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARG_NAME) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARG_NAME) end

#endif
