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
 * \file flash_attn_gqa_cube.h
 * \brief FlashAttn arch35 AIC-side kernel (non-quant GQA)
 */

#ifndef FLASH_ATTN_GQA_CUBE_H_
#define FLASH_ATTN_GQA_CUBE_H_

#include "../../../common/op_kernel/fia_public_define.h"
#include "../../../common/op_kernel/offset_calculator.h"
#include "fa_static_buffer.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../utils/flash_attn_common_def.h"
#include "../utils/flash_attn_cube_def.h"
#include "../utils/flash_attn_utils.h"

#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "flash_attn_tiling_data.h"
#include "flash_attn_task_loop_hooks.h"

using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;

namespace BaseApi {
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

/* ============确定L0A的类型（仅供matmul使用）============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};
/* ============确定L0B的类型（仅供matmul使用）============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> ||
                                        (s2BaseSize == 256 && dBaseSize > 128 &&
                                         !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                           std::is_same_v<INPUT_T, fp8_e5m2_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                                    BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128,
          uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false,
          bool HasMask = false>
class FlashAttentionNoQuantGqaKernel {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t l1BaseD = 128;
    static constexpr LayOutTypeEnum LAYOUT = layout;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);
    static constexpr bool BMM2_TOUB = bmm2Write2Ub;
    static constexpr bool USE_DN = useDn;
    static constexpr bool SPLITD = splitD;
    static constexpr bool HAS_PREFIX = false;
    static constexpr bool HAS_MASK = HasMask;

    static constexpr uint32_t PRELOAD_N = 2;
    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = 4;
    static constexpr uint32_t PRELOAD_TASK_CACHE_MASK = PRELOAD_TASK_CACHE_SIZE - 1;
    static_assert(PRELOAD_TASK_CACHE_SIZE >= PRELOAD_N + 1,
                  "PRELOAD_TASK_CACHE_SIZE must be at least PRELOAD_N + 1");
    static_assert((PRELOAD_TASK_CACHE_SIZE & PRELOAD_TASK_CACHE_MASK) == 0,
                  "PRELOAD_TASK_CACHE_SIZE must be a power of two for bitmask indexing");

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr ActualSeqLensMode KV_MODE = GetKvActSeqMode<layout, PAGE_ATTENTION>();

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using L0AType = typename L0ABuffSel<INPUT_T>::Type;
    using L0BType = typename L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;

    using ConstInfoX = ConstInfoXCube;
    using RunInfoType = RunInfoXCube;

    struct TaskLoopHooks {
        static constexpr bool HAS_DEAL_ZERO = false;

        static __aicore__ inline void InitLoopState(FlashAttentionNoQuantGqaKernel &) {}

        static __aicore__ inline void ResetLoopState(FlashAttentionNoQuantGqaKernel &) {}

        static __aicore__ inline void GetSectionInfo(FlashAttentionNoQuantGqaKernel &self, uint32_t sectionIdx)
        {
            self.GetFASectionInfo(sectionIdx);
        }

        static __aicore__ inline void CreateTask(FlashAttentionNoQuantGqaKernel &self,
                                                  RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE],
                                                  uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
        {
            self.CreateTask(loop, bN2Cur, gS1Cur, s2Cur, taskRunInfo);
        }

        static __aicore__ inline void ExecuteTask(FlashAttentionNoQuantGqaKernel &self,
                                                   RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE],
                                                   uint64_t loop)
        {
            self.ExecuteTask(loop, taskRunInfo);
        }

        static __aicore__ inline void UpdateAxisInfo(FlashAttentionNoQuantGqaKernel &self, TASK_DEAL_MODE,
                                                      uint32_t &bN2Cur, uint32_t &gS1Cur, uint32_t &s2Cur)
        {
            self.UpdateAxisInfo(bN2Cur, gS1Cur, s2Cur);
        }

        static __aicore__ inline void DealZeroActSeqLen(FlashAttentionNoQuantGqaKernel &, uint32_t) {}
    };

    // ====== L1/L0C 缓冲区布局常量 ======
    static constexpr bool L1Q_IS_SINGLE = (dBaseSize > 256);
    static constexpr bool L1KV_IS_SINGLE = (s2BaseSize == 256 && dBaseSize > 128);
    static constexpr bool L0C_IS_4BUFF = (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                          mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4);

    // ====== 核间同步常量 ======
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t AIV0_AIV1_OFFSET = 16;

    static constexpr bool IS_TND_LAYOUT =
        (LAYOUT == LayOutTypeEnum::LAYOUT_TND || LAYOUT == LayOutTypeEnum::LAYOUT_NTD);
    static constexpr bool Q_NEEDS_WZH = (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND);
    static constexpr bool KV_NEEDS_WZH = (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND);

    // ====== 静态事件ID常量（编译期固定，绝不调用AllocEventID动态分配）======
    // 使用 TEventID（与 Buffer 成员字段类型一致），static_cast 保证跨类型安全
    // L1同步事件 (c2p: MTE1→MTE2)；由AllocEventID()中显式SetFlag<MTE1_MTE2>预置
    static constexpr TEventID FA_L1_Q_C2P_0 = static_cast<TEventID>(0);  // l1Q ping/single
    static constexpr TEventID FA_L1_Q_C2P_1 = static_cast<TEventID>(1);  // l1Q pong
    static constexpr TEventID FA_L1_K_C2P_0 = static_cast<TEventID>(2);  // l1K ping/single
    static constexpr TEventID FA_L1_K_C2P_1 = static_cast<TEventID>(3);  // l1K pong
    static constexpr TEventID FA_L1_V_C2P_0 = static_cast<TEventID>(4);  // l1V ping/single
    static constexpr TEventID FA_L1_V_C2P_1 = static_cast<TEventID>(5);  // l1V pong
    // L0A同步事件 (c2p: M→MTE1)；L0A始终为DB；由AllocEventID()中显式SetFlag<M_MTE1>预置
    static constexpr TEventID FA_L0A_C2P_0  = static_cast<TEventID>(0);  // L0A ping
    static constexpr TEventID FA_L0A_C2P_1  = static_cast<TEventID>(1);  // L0A pong
    // L0B同步事件 (c2p: M→MTE1)；与L0A同类型但内存独立；由AllocEventID()中预置
    static constexpr TEventID FA_L0B_C2P_0  = static_cast<TEventID>(2);  // L0B ping/single
    static constexpr TEventID FA_L0B_C2P_1  = static_cast<TEventID>(3);  // L0B pong
    // L0C同步事件 (c2p: FIX→M)；事件空间与L0A/L0B独立；由AllocEventID()中预置
    static constexpr TEventID FA_L0C_C2P_0  = static_cast<TEventID>(0);  // L0C slot0
    static constexpr TEventID FA_L0C_C2P_1  = static_cast<TEventID>(1);  // L0C slot1
    static constexpr TEventID FA_L0C_C2P_2  = static_cast<TEventID>(2);  // L0C slot2
    static constexpr TEventID FA_L0C_C2P_3  = static_cast<TEventID>(3);  // L0C slot3

    // ====== L1 内存布局常量（全部 static constexpr，运行时不再访问 BufferManager）======
    // L1 从 offset=0 起，布局为：
    //   [0 .. 3*mm2LeftSize) : l1PBuffers（3buff，由 InitMMResBuf 中 l1PBuffers.Init 分配）
    //   [FA_L1_CUBE_BASE ...) : cube block 的 l1QBuffers / l1KBuffers / l1VBuffers
    // mm2LeftSize = mBaseSize * s2BaseSize * sizeof(Q_T)，三个 slot 合计 3 倍
    static constexpr uint32_t FA_L1_CUBE_BASE = 3U * mBaseSize * s2BaseSize * sizeof(Q_T);

    using QSeqParserType =
        typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true>,
                                  ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>>::type;

    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType qSeqLensParser;
    KvSeqParserType kvSeqLensParser;

    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    /* =====================LocalBuffer变量====================*/
    // L1 裸 LocalTensor 数组 + 旋转状态（替代 BuffersPolicy）
    LocalTensor<uint8_t> l1QSlot[2];   // [0]=ping/single, [1]=pong
    LocalTensor<uint8_t> l1KSlot[2];
    LocalTensor<uint8_t> l1VSlot[2];
    uint32_t l1QFlag = 0;
    uint32_t l1KFlag = 0;
    uint32_t l1VFlag = 0;

    // L0C 裸 LocalTensor 数组 + 旋转状态
    LocalTensor<uint8_t> l0CSlot[4];   // max 4 slots (4buff or DB)
    uint32_t l0CHead = 0;              // for 4buff: 0,1,2,3,0,...
    uint32_t l0CFlag = 0;              // for DB: XOR toggle

    // L0A/L0B 保留给matmul API
    L0AType mmL0ABuffers;
    L0BType mmL0BBuffers;

    // ====== L1 slot 旋转辅助方法 ======
    __aicore__ inline uint32_t L1QGet() {
        if constexpr (L1Q_IS_SINGLE) { return 0; }
        uint32_t idx = l1QFlag ^ 1; l1QFlag ^= 1; return idx;
    }
    __aicore__ inline uint32_t L1QGetPre() {
        if constexpr (L1Q_IS_SINGLE) { return 0; }
        return l1QFlag;
    }
    __aicore__ inline uint32_t L1KGet() {
        if constexpr (L1KV_IS_SINGLE) { return 0; }
        uint32_t idx = l1KFlag ^ 1; l1KFlag ^= 1; return idx;
    }
    __aicore__ inline uint32_t L1VGet() {
        if constexpr (L1KV_IS_SINGLE) { return 0; }
        uint32_t idx = l1VFlag ^ 1; l1VFlag ^= 1; return idx;
    }
    __aicore__ inline uint32_t L0CGet() {
        if constexpr (L0C_IS_4BUFF) {
            uint32_t idx = l0CHead % 4;
            l0CHead++;
            return idx;
        } else {
            uint32_t idx = l0CFlag ^ 1; l0CFlag ^= 1; return idx;
        }
    }
    // 事件ID查询
    __aicore__ inline TEventID L1QEvt(uint32_t idx) { return (idx == 0) ? FA_L1_Q_C2P_0 : FA_L1_Q_C2P_1; }
    __aicore__ inline TEventID L1KEvt(uint32_t idx) { return (idx == 0) ? FA_L1_K_C2P_0 : FA_L1_K_C2P_1; }
    __aicore__ inline TEventID L1VEvt(uint32_t idx) { return (idx == 0) ? FA_L1_V_C2P_0 : FA_L1_V_C2P_1; }
    __aicore__ inline TEventID L0CEvt(uint32_t idx) { return static_cast<TEventID>(idx); }

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;

    ConstInfoXCube constInfoXCube;

    __aicore__ inline void InitBuffers()
    {
        if constexpr (dBaseSize > 256) {
            // L1Q = Single (dBaseSize>256), L1KV: Single if s2==256&&d>128, else DB
            constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            constexpr uint32_t l1QOff       = FA_L1_CUBE_BASE;
            constexpr uint32_t l1KOff       = l1QOff + mm1LeftSize;

            l1QSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1QOff, mm1LeftSize);

            if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
                // Single + Single + Single
                constexpr uint32_t l1VOff = l1KOff + mm1RightSize;
                l1KSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1KOff, mm1RightSize);
                l1VSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1VOff, mm1RightSize);
            } else {
                // Single + DB + DB
                constexpr uint32_t l1VOff = l1KOff + 2U * mm1RightSize;
                l1KSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1KOff, mm1RightSize);
                l1KSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1KOff + mm1RightSize, mm1RightSize);
                l1VSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1VOff, mm1RightSize);
                l1VSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1VOff + mm1RightSize, mm1RightSize);
            }
        } else {
            // L1Q = DB (dBaseSize<=256), L1KV: Single if s2==256&&d>128, else DB
            constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(KV_T);
            constexpr uint32_t mm2RightSize = (uint32_t)dVTemplateType * s2BaseSize * sizeof(KV_T);
            constexpr uint32_t l1QOff       = FA_L1_CUBE_BASE;
            constexpr uint32_t l1KOff       = l1QOff + 2U * mm1LeftSize;  // DB: ping+pong

            l1QSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1QOff, mm1LeftSize);
            l1QSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1QOff + mm1LeftSize, mm1LeftSize);

            if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
                // DB + Single + Single
                constexpr uint32_t l1VOff = l1KOff + mm1RightSize;
                l1KSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1KOff, mm1RightSize);
                l1VSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1VOff, mm2RightSize);
            } else {
                // DB + DB + DB
                constexpr uint32_t l1VOff = l1KOff + 2U * mm1RightSize;
                l1KSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1KOff, mm1RightSize);
                l1KSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1KOff + mm1RightSize, mm1RightSize);
                l1VSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1VOff, mm2RightSize);
                l1VSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1VOff + mm2RightSize, mm2RightSize);
            }
        }

        // L0A：始终为 DB，ping=offset 0，pong=offset 32KB
        constexpr uint32_t FA_L0_BUF_SIZE = 32U * 1024U;
        mmL0ABuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, FA_L0_BUF_SIZE,
                                      FA_L0A_C2P_0, FA_L0A_C2P_0,
                                      FA_L0A_C2P_1, FA_L0A_C2P_1);

        // L0B：Single if s2==256&&d>128, else DB
        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            mmL0BBuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, FA_L0B_C2P_0, FA_L0B_C2P_0);
        } else {
            mmL0BBuffers.InitStaticDirect(0U, FA_L0_BUF_SIZE, FA_L0_BUF_SIZE,
                                          FA_L0B_C2P_0, FA_L0B_C2P_0,
                                          FA_L0B_C2P_1, FA_L0B_C2P_1);
        }

        // L0C：4buff (各64KB) 或 DB (各128KB) —— 裸 LocalTensor 构造
        // 注意 L0C 的 TPosition 标准是 CO1（对应 common/op_kernel/buffer.h::BufferInfo<L0C>::Position），
        // 不是 C1 — 之前用 C1 会让 LocalTensor 指向错误的物理区域，导致 fixpipe 读到的 L0C 数据有缺失，
        // 表现为部分 m-block(尤其是 m=0 和 S2-split 的尾 m)精度异常。
        if constexpr (L0C_IS_4BUFF) {
            constexpr uint32_t l0cSlotSize = (uint32_t)(L0C_SIZE / NUM_4) * (uint32_t)KB_TO_BYTES;
            l0CSlot[0] = LocalTensor<uint8_t>(TPosition::CO1, 0U, l0cSlotSize);
            l0CSlot[1] = LocalTensor<uint8_t>(TPosition::CO1, l0cSlotSize, l0cSlotSize);
            l0CSlot[2] = LocalTensor<uint8_t>(TPosition::CO1, 2U * l0cSlotSize, l0cSlotSize);
            l0CSlot[3] = LocalTensor<uint8_t>(TPosition::CO1, 3U * l0cSlotSize, l0cSlotSize);
        } else {
            constexpr uint32_t l0cSlotSize = (uint32_t)(L0C_SIZE / NUM_2) * (uint32_t)KB_TO_BYTES;
            l0CSlot[0] = LocalTensor<uint8_t>(TPosition::CO1, 0U, l0cSlotSize);
            l0CSlot[1] = LocalTensor<uint8_t>(TPosition::CO1, l0cSlotSize, l0cSlotSize);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfoXCube.bSize, constInfoXCube.n2Size, constInfoXCube.gSize, constInfoXCube.s1Size, constInfoXCube.dSize, queryGm,
                    query);

        keyPtr = key;
        valuePtr = value;
        if (constInfoXCube.isKvContinuous) {
            InitKVBuffer(constInfoXCube.bSize, constInfoXCube.s2Size, constInfoXCube.n2Size, constInfoXCube.blockSize, constInfoXCube.dSize,
                         keyGm, key);
            InitKVBuffer(constInfoXCube.bSize, constInfoXCube.s2Size, constInfoXCube.n2Size, constInfoXCube.blockSize, constInfoXCube.dSizeV,
                         valueGm, value);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensParser);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensParser);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfoXCube.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfoXCube.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(kvSeqLensParser);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, kvSeqLensParser);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        // 与 VEC 侧完全对称：AllocEventID 统一完成所有流水预置 SetFlag（无动态 AllocEventID 调用）
        // L1 c2p 预置：MTE1 告知 MTE2 当前 slot 空闲可以覆写
        SetFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_0);
        if constexpr (!L1Q_IS_SINGLE) {
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_1);
        }
        SetFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_0);
        if constexpr (!L1KV_IS_SINGLE) {
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_1);
        }
        SetFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_0);
        if constexpr (!L1KV_IS_SINGLE) {
            SetFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_1);
        }
        // L0A c2p 预置：CUBE 告知 MTE1 当前 slot 空闲可重新加载（L0A 始终为 DB）
        SetFlag<HardEvent::M_MTE1>(FA_L0A_C2P_0);
        SetFlag<HardEvent::M_MTE1>(FA_L0A_C2P_1);
        // L0B c2p 预置：Single if s2==256&&d>128, else DB
        if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
            SetFlag<HardEvent::M_MTE1>(FA_L0B_C2P_0);
        } else {
            SetFlag<HardEvent::M_MTE1>(FA_L0B_C2P_0);
            SetFlag<HardEvent::M_MTE1>(FA_L0B_C2P_1);
        }
        // L0C c2p 预置：FixPipe 告知 CUBE 当前 slot 空闲可写入（4buff 或 DB）
        if constexpr (L0C_IS_4BUFF) {
            SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_0);
            SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_1);
            SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_2);
            SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_3);
        } else {
            SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_0);
            SetFlag<HardEvent::FIX_M>(FA_L0C_C2P_1);
        }
    }

    __aicore__ inline void FreeEventID()
    {
        // 排干所有 c2p 流水：WaitFlag 等待最后一次 Set 完成
        // L1: MTE1_MTE2 事件
        if constexpr (L1Q_IS_SINGLE) {
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_0);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_0);
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_Q_C2P_1);
        }
        if constexpr (L1KV_IS_SINGLE) {
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_0);
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_0);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_0);
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_K_C2P_1);
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_0);
            WaitFlag<HardEvent::MTE1_MTE2>(FA_L1_V_C2P_1);
        }
        // L0A/L0B: 保留 Buffer 接口
        mmL0ABuffers.UninitStaticDirect();
        mmL0BBuffers.UninitStaticDirect();
        // L0C: FIX_M 事件
        if constexpr (L0C_IS_4BUFF) {
            WaitFlag<HardEvent::FIX_M>(FA_L0C_C2P_0);
            WaitFlag<HardEvent::FIX_M>(FA_L0C_C2P_1);
            WaitFlag<HardEvent::FIX_M>(FA_L0C_C2P_2);
            WaitFlag<HardEvent::FIX_M>(FA_L0C_C2P_3);
        } else {
            WaitFlag<HardEvent::FIX_M>(FA_L0C_C2P_0);
            WaitFlag<HardEvent::FIX_M>(FA_L0C_C2P_1);
        }
    }

    // copy query with full s1g and split D
    __aicore__ inline void CopyQuerySlice(const LocalTensor<Q_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoXCube &runInfo)
    {
        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        if (dRealSize > 0) {
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = dOffset,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = dRealSize};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoXCube &runInfo)
    {
        uint32_t dSize = constInfoXCube.dSize;
        CopyQuerySlice(dstTensor, 0, dSize, runInfo);
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfoXCube &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;

        if (dRealSize > 0) {
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = constInfoXCube.isKvContinuous ? runInfo.bIdx : 0,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = dOffset,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = dRealSize};
            copyKvGmToL1(l1Tensor, keyGm, gmCoord);
        }
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoXCube &runInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = constInfoXCube.isKvContinuous ? runInfo.bIdx : 0,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoXCube &runInfo)
    {
        uint32_t dSize = constInfoXCube.dSize;
        CopyKeySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoXCube &runInfo)
    {
        CopyValueSlice(dstTensor, 0, constInfoXCube.dSizeV, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr, bIdx);
        keyGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm.offsetCalculator.Init(0, constInfoXCube.n2Size, s2Size, constInfoXCube.dSize);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr, bIdx);
        valueGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm.offsetCalculator.Init(0, constInfoXCube.n2Size, s2Size, constInfoXCube.dSize);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoXCube &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateKey(runInfo.bIdx);
            }
        }

#ifdef SKIP_C1
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
        return;
#endif

        if constexpr (dBaseSize > 256) {
            IterateBmm1NdL1SplitK(outputTensor, ccId0, ccId1, runInfo);
            return;
        }

        if constexpr (useDn) {
            if constexpr (dBaseSize > 128) {
                IterateBmm1DnSplitK(outputTensor, ccId0, ccId1, runInfo);
            } else {
                IterateBmm1Dn(outputTensor, ccId0, ccId1, runInfo);
            }
            return;
        }
        IterateBmm1NdL0Split(outputTensor, ccId0, ccId1, runInfo);
    }

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoXCube &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小, 使能NZ2ND, nSize*sizeof(T) 必须是32B的倍数
        fixpipeParams.nSize = (runInfo.actSingleLoopS2Size + 7) >> 3 << 3;
        // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
        // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T)
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element
        fixpipeParams.dualDstCtl = 1;         // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;

        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams);
    }

    /* 针对S1Base=128, S2Base = 128, D > 128场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1NdL0Split(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoXCube &runInfo)
    {
        uint32_t qIdx;
        if (unlikely(runInfo.isFirstS2Loop)) {
            qIdx = L1QGet();
            WaitFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
            CopyQueryTile(l1QSlot[qIdx].template ReinterpretCast<Q_T>(), runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        } else {
            qIdx = L1QGetPre();
            SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        }

        uint32_t kIdx = L1KGet();
        WaitFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));
        LocalTensor<KV_T> mm1BTensor = l1KSlot[kIdx].template ReinterpretCast<KV_T>();
        CopyKeyTile(mm1BTensor, runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));
        WaitFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        WaitFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));

        uint32_t cIdx = L0CGet();
        WaitFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        LocalTensor<T> l0CTensor = l0CSlot[cIdx].template ReinterpretCast<T>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size,
                                    (uint32_t)(constInfoXCube.dSize), false, true);
        LocalTensor<Q_T> mm1ATensor = l1QSlot[qIdx].template ReinterpretCast<Q_T>();
        if constexpr (dBaseSize > 128) {
            if constexpr (s2BaseSize == 256) {
                MatmulN<Q_T, KV_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
                    mm1ATensor, mm1BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);
            } else {
                MatmulK<Q_T, KV_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                    mm1ATensor, mm1BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);
            }
        } else {
            MatmulBase<Q_T, KV_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
                mm1ATensor, mm1BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));   // 释放L1B
        SetFlag<HardEvent::M_FIX>(L0CEvt(cIdx));       // 通知
        WaitFlag<HardEvent::M_FIX>(L0CEvt(cIdx));      // 等待L0C

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);

        FixpipeMm1(outputTensor, l0CTensor, runInfo);

        SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    /* 针对S1Base=128, S2Base = 128, D > 256场景，L1层面切K，且左矩阵单Buffer+驻留，右矩阵每次重新搬运。*/
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1NdL1SplitK(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoXCube &runInfo)
    {
        constexpr uint32_t baseK = l1BaseD;
        uint32_t kLoops = (constInfoXCube.dSize + baseK - 1) / baseK;

        uint32_t cIdx = L0CGet();
        WaitFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        LocalTensor<T> l0CTensor = l0CSlot[cIdx].template ReinterpretCast<T>();

        uint32_t dstNzC0Stride = ((runInfo.actMSize + 15) >> 4 << 4);
        uint64_t l1BaseKOffset = baseK * dstNzC0Stride;

        uint32_t qIdx = L1QGet();
        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
        }

        uint32_t realK = baseK;
        for (uint32_t k = 0; k < kLoops; k++) {
            if (k == kLoops - 1) {
                realK = constInfoXCube.dSize - k * baseK;
            }
            if (unlikely(runInfo.isFirstS2Loop)) { // sOuter循环第一个基本快：搬运0
                LocalTensor<Q_T> mm1ATensor = l1QSlot[qIdx].template ReinterpretCast<Q_T>();
                CopyQuerySlice(mm1ATensor[k * l1BaseKOffset], k * baseK, realK, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx)); // 通知
            } else {                              // 非s2的第一次循环直接复用Q
                qIdx = L1QGetPre();
                SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
            }
            // 加载当前轮的右矩阵到L1
            uint32_t kIdx = L1KGet();
            WaitFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));
            LocalTensor<KV_T> mm1BTensor = l1KSlot[kIdx].template ReinterpretCast<KV_T>();
            CopyKeySlice(mm1BTensor, k * baseK, realK, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));  // 通知
            WaitFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx)); // 等待L1A
            WaitFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx)); // 等待L1B

            MMParam param = MakeMMParam((uint32_t)runInfo.actMSize, (uint32_t)runInfo.actSingleLoopS2Size, realK, false,
                                        true, k == 0, k == 0);

            LocalTensor<Q_T> mm1AForMatmul = l1QSlot[qIdx].template ReinterpretCast<Q_T>();
            MatmulFull<Q_T, KV_T, T, 128, 128, baseK, ABLayout::MK, ABLayout::KN>(
                mm1AForMatmul[k * l1BaseKOffset], mm1BTensor, mmL0ABuffers, mmL0BBuffers,
                l0CTensor, param);

            SetFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx)); // 释放L1B
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
        }
        SetFlag<HardEvent::M_FIX>(L0CEvt(cIdx)); // 通知
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);

        WaitFlag<HardEvent::M_FIX>(L0CEvt(cIdx)); // 等待L0C
        FixpipeMm1(outputTensor, l0CTensor, runInfo);

        SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx)); // 释放
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    /* 针对useDn=true, S1Base=128, S2Base = 128, 128 < D <= 256场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1DnSplitK(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoXCube &runInfo)
    {
        uint32_t qIdx;
        if (unlikely(runInfo.isFirstS2Loop)) {
            qIdx = L1QGet();
            WaitFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
            CopyQueryTile(l1QSlot[qIdx].template ReinterpretCast<Q_T>(), runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        } else {
            qIdx = L1QGetPre();
            SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        }

        uint32_t kIdx = L1KGet();
        WaitFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));
        LocalTensor<KV_T> mm1ATensor = l1KSlot[kIdx].template ReinterpretCast<KV_T>();

        CopyKeyTile(mm1ATensor, runInfo);

        SetFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));

        WaitFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));
        WaitFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));

        uint32_t cIdx = L0CGet();
        WaitFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        LocalTensor<T> l0CTensor = l0CSlot[cIdx].template ReinterpretCast<T>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                    (uint32_t)(constInfoXCube.dSize), false, true);

        MatmulK<KV_T, Q_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
            mm1ATensor, l1QSlot[qIdx].template ReinterpretCast<Q_T>(), mmL0ABuffers, mmL0BBuffers, l0CTensor, param);

        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));

        SetFlag<HardEvent::M_FIX>(L0CEvt(cIdx));
        WaitFlag<HardEvent::M_FIX>(L0CEvt(cIdx));

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);
        FixpipeMm1Dn(outputTensor, l0CTensor, runInfo);

        SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoXCube &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (runInfo.actMSize + 31) >> 5 << 5; // 双目标模式, 按照N方向切分，N必须为32的倍数
        fixpipeParams.mSize = runInfo.actSingleLoopS2Size;
        fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        fixpipeParams.dstStride = fixpipeParams.nSize / 2;
        fixpipeParams.dualDstCtl = 2; // 按照N方向切分
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(dstTensor, l0C, fixpipeParams); // 将matmul结果从L0C搬运到UB
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1Dn(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoXCube &runInfo)
    {
        uint32_t qIdx;
        if (unlikely(runInfo.isFirstS2Loop)) {
            qIdx = L1QGet();
            WaitFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
            CopyQueryTile(l1QSlot[qIdx].template ReinterpretCast<Q_T>(), runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        } else {
            qIdx = L1QGetPre();
            SetFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));
        }

        uint32_t kIdx = L1KGet();
        WaitFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));
        LocalTensor<KV_T> mm1ATensor = l1KSlot[kIdx].template ReinterpretCast<KV_T>();
        CopyKeyTile(mm1ATensor, runInfo);

        SetFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));
        WaitFlag<HardEvent::MTE2_MTE1>(L1KEvt(kIdx));
        WaitFlag<HardEvent::MTE2_MTE1>(L1QEvt(qIdx));

        uint32_t cIdx = L0CGet();
        WaitFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        LocalTensor<T> l0CTensor = l0CSlot[cIdx].template ReinterpretCast<T>();

        MMParam param = MakeMMParam((uint32_t)runInfo.actSingleLoopS2Size, (uint32_t)runInfo.actMSize,
                                    (uint32_t)(constInfoXCube.dSize), false, true);

        MatmulBase<KV_T, Q_T, T, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
            mm1ATensor, l1QSlot[qIdx].template ReinterpretCast<Q_T>(), mmL0ABuffers, mmL0BBuffers, l0CTensor, param);

        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::MTE1_MTE2>(L1QEvt(qIdx));
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1KEvt(kIdx));

        SetFlag<HardEvent::M_FIX>(L0CEvt(cIdx));
        WaitFlag<HardEvent::M_FIX>(L0CEvt(cIdx));

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId1 + AIV0_AIV1_OFFSET);
        FixpipeMm1Dn(outputTensor, l0CTensor, runInfo);
        SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx)); // 释放L0C
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(ccId0 + AIV0_AIV1_OFFSET);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                        LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoXCube &runInfo)
    {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            if (runInfo.isChangeBatch) {
                UpdateValue(runInfo.bIdx);
            }
        }

#ifdef SKIP_C2
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);
        if constexpr (bmm2Write2Ub) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
        return;
#endif

        if constexpr ((uint32_t)dVTemplateType > 256 || (uint32_t)dTemplateType > 256) {
            IterateBmm2L1SplitN(outputTensor, outCcId0, outCcId1, l1PTensor, l1PCcId, runInfo);
        } else {
            IterateBmm2l0Split(outputTensor, outCcId0, outCcId1, l1PTensor, l1PCcId, runInfo);
        }
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2L1SplitN(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                                LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoXCube &runInfo)
    {
        // L1 P 由两个 AIV 子核分别写入上下两半（行 0-63 / 64-127），
        // AIC 必须同时等待 AIV0(l1PCcId) 与 AIV1(l1PCcId+OFFSET) 两个 flag。
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);

        // dVTemplateType > 256, L1切N, 左矩阵不变，右矩阵每次循环搬运S2 * 128
        if constexpr (bmm2Write2Ub) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }
        constexpr uint32_t baseN = l1BaseD;
        uint32_t nLoops = ((uint32_t)constInfoXCube.dSizeV + baseN - 1) / baseN; // 尾块处理
        uint32_t realN = baseN;
        for (uint32_t n = 0; n < nLoops; ++n) {
            if (n == nLoops - 1) {
                uint32_t tailSize = (uint32_t)constInfoXCube.dSizeV % baseN;
                realN = tailSize ? tailSize : baseN;
            }
            uint32_t vIdx = L1VGet();
            WaitFlag<HardEvent::MTE1_MTE2>(L1VEvt(vIdx));
            LocalTensor<KV_T> mm2BTensor = l1VSlot[vIdx].template ReinterpretCast<KV_T>();
            CopyValueSlice(mm2BTensor, n * baseN, realN, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(L1VEvt(vIdx));

            uint32_t cIdx = L0CGet();
            WaitFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
            LocalTensor<T> l0CTensor = l0CSlot[cIdx].template ReinterpretCast<T>();

            MMParam param = MakeMMParam((uint32_t)mBaseSize, (uint32_t)realN, (uint32_t)(runInfo.actSingleLoopS2Size),
                                        useDn, false);
            if constexpr (!useDn) {
                param.realM = (uint32_t)runInfo.actMSize;
            }
            WaitFlag<HardEvent::MTE2_MTE1>(L1VEvt(vIdx));
            MatmulFull<Q_T, KV_T, T, 128, baseN, 128, ABLayout::MK, ABLayout::KN>(
                l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);

            SetFlag<HardEvent::MTE1_MTE2>(L1VEvt(vIdx));
            SetFlag<HardEvent::M_FIX>(L0CEvt(cIdx));
            WaitFlag<HardEvent::M_FIX>(L0CEvt(cIdx));

            FixpipeMm2PartialN(outputTensor[n * baseN], l0CTensor, realN, runInfo);
            SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx));
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
    }

    template <typename DST_TENSOR_T>
    __aicore__ inline void FixpipeMm2PartialN(const DST_TENSOR_T &dstTensor, const LocalTensor<T> &l0C, uint32_t realN,
                                              RunInfoXCube &runInfo)
    {
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        if constexpr (bmm2Write2Ub) {
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
        } else {
            fixpipeParams.nSize = realN;
        }

        if constexpr (!useDn) {
            fixpipeParams.mSize = (runInfo.actMSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16;
        } else {
            fixpipeParams.mSize = mBaseSize;
            fixpipeParams.srcStride = ((mBaseSize + 15) / 16) * 16;
        }

        if constexpr (bmm2Write2Ub || splitD) {
            fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfoXCube.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2l0Split(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                               LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoXCube &runInfo)
    {
        // L1 P 由两个 AIV 子核分别写入上下两半（行 0-63 / 64-127），
        // AIC 必须同时等待 AIV0(l1PCcId) 与 AIV1(l1PCcId+OFFSET) 两个 flag，
        // 否则会在 AIV1 写完前读取 L1 下半部分，导致行 64-127 读到脏数据。
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(l1PCcId + AIV0_AIV1_OFFSET);
        uint32_t vIdx = L1VGet();
        WaitFlag<HardEvent::MTE1_MTE2>(L1VEvt(vIdx)); // 占用L1B
        LocalTensor<KV_T> mm2BTensor = l1VSlot[vIdx].template ReinterpretCast<KV_T>();
        CopyValueTile(mm2BTensor, runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(L1VEvt(vIdx)); // 通知

        uint32_t cIdx = L0CGet();
        WaitFlag<HardEvent::FIX_M>(L0CEvt(cIdx)); // 占用
        LocalTensor<T> l0CTensor = l0CSlot[cIdx].template ReinterpretCast<T>();
        MMParam param = {
            (uint32_t)mBaseSize,                   // singleM 128
            (uint32_t)constInfoXCube.dSizeV,            // singleN 128
            (uint32_t)runInfo.actSingleLoopS2Size, // singleK
            useDn,                                 // isLeftTranspose
            false                                  // isRightTranspose
        };
        if constexpr (!useDn) {
            param.realM = (uint32_t)runInfo.actMSize;
        }
        WaitFlag<HardEvent::MTE2_MTE1>(L1VEvt(vIdx)); // 等待

        if constexpr ((uint32_t)dVTemplateType > 128) {
            MatmulN<Q_T, KV_T, T, mBaseSize, 128, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);
        } else {
            if constexpr (s2BaseSize == 128) {
                MatmulFull<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);
            } else {
                MatmulBase<Q_T, KV_T, T, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
                    l1PTensor, mm2BTensor, mmL0ABuffers, mmL0BBuffers, l0CTensor, param);
            }
        }

        SetFlag<HardEvent::MTE1_MTE2>(L1VEvt(vIdx));   // 释放L1B
        SetFlag<HardEvent::M_FIX>(L0CEvt(cIdx));       // 通知
        WaitFlag<HardEvent::M_FIX>(L0CEvt(cIdx));      // 等待

        if constexpr (bmm2Write2Ub) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId1 + AIV0_AIV1_OFFSET);
        }

        FixpipeMm2PartialN(outputTensor, l0CTensor, constInfoXCube.dSizeV, runInfo);

        SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx)); // 释放
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
    }

    // ===== 核间共享 buffer slots =====
    LocalTensor<uint8_t> bmm1UbSlot[2];
    LocalTensor<uint8_t> bmm2UbSlot[2];
    LocalTensor<uint8_t> l1PSlot[3];
    GlobalTensor<MM_T>   bmm2GmSlot[3];

    // ===== 静态 cross-core 事件 ID =====
    static constexpr uint32_t CC_BMM1_0 = 0U;
    static constexpr uint32_t CC_BMM1_1 = 1U;
    static constexpr uint32_t CC_BMM2_0 = 2U;
    static constexpr uint32_t CC_BMM2_1 = 3U;
    static constexpr uint32_t CC_BMM2_2 = 4U;
    static constexpr uint32_t CC_L1P_0  = 5U;
    static constexpr uint32_t CC_L1P_1  = 8U;
    static constexpr uint32_t CC_L1P_2  = 9U;

    GlobalTensor<int32_t> cuSeqLensGmQ;
    GlobalTensor<int32_t> cuSeqLensGmKv;
    GlobalTensor<int32_t> seqUsedGmQ;
    GlobalTensor<int32_t> seqUsedGmKv;
    GlobalTensor<uint32_t> faMetaDataGm;

    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData;

    uint32_t createdTaskCount = 0U;

    uint64_t actSeqLensKv = 0;
    uint64_t actSeqLensQ = 0;
    uint32_t curS2Start = 0;
    uint32_t curS2End = 0;
    uint32_t prevBIdx = 0;
    uint32_t prevBN2Idx = 0;
    uint32_t prevGS1Idx = 0;

    uint32_t sectionNum_ = 0;
    uint32_t bN2Start_ = 0;
    uint32_t bN2End_ = 0;
    uint32_t gS1OStart_ = 0;
    uint32_t gS1OEnd_ = 0;
    uint32_t s2OStart_ = 0;
    uint32_t s2OEnd_ = 0;



    __aicore__ inline FlashAttentionNoQuantGqaKernel() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *fiaMetaData,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv,
                                const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tiling)
    {
        this->tilingData = tiling;

        InitConstInfo();

        keyPtr = key;
        valuePtr = value;

        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            cuSeqLensGmQ.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensQ, constInfoXCube.cuSeqLensQSize + 1);
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXCube.seqUsedQSize);
            cuSeqLensGmKv.SetGlobalBuffer((__gm__ int32_t *)cuSeqLensKv, constInfoXCube.cuSeqLensKVSize + 1);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXCube.seqUsedKvSize);
        } else {
            seqUsedGmQ.SetGlobalBuffer((__gm__ int32_t *)seqUsedQ, constInfoXCube.seqUsedQSize);
            seqUsedGmKv.SetGlobalBuffer((__gm__ int32_t *)seqUsedKv, constInfoXCube.seqUsedKvSize);
        }
        sectionNum_ = ((__gm__ uint32_t *)fiaMetaData)[0];

        faMetaDataGm.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + FA_METADATA_HEADER_OFFSET),
                                     FA_AIC_CORE_NUM * 16U * sectionNum_);

        InitQSeqLensParser();
        InitKvSeqLensParser();

        InitMMResBuf(workspace);
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitMMResBuf(__gm__ uint8_t *&workspace)
    {
        constexpr uint32_t mm1SlotSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        constexpr uint32_t mm2SlotSize = BMM2_TOUB ?
            (mBaseSize / CV_RATIO * dVBaseSize * sizeof(T)) : 0U;
        constexpr uint32_t l1PSlotSize = mBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t bmm1UbBase  = 2U * mm2SlotSize;

        if constexpr (BMM2_TOUB) {
            bmm2UbSlot[0] = LocalTensor<uint8_t>(TPosition::VECIN, 0U,          mm2SlotSize);
            bmm2UbSlot[1] = LocalTensor<uint8_t>(TPosition::VECIN, mm2SlotSize, mm2SlotSize);
        }
        bmm1UbSlot[0] = LocalTensor<uint8_t>(TPosition::VECIN, bmm1UbBase,               mm1SlotSize);
        bmm1UbSlot[1] = LocalTensor<uint8_t>(TPosition::VECIN, bmm1UbBase + mm1SlotSize, mm1SlotSize);

        l1PSlot[0] = LocalTensor<uint8_t>(TPosition::A1, 0U,               l1PSlotSize);
        l1PSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1PSlotSize,      l1PSlotSize);
        l1PSlot[2] = LocalTensor<uint8_t>(TPosition::A1, 2U * l1PSlotSize, l1PSlotSize);

        if constexpr (!BMM2_TOUB) {
            int64_t mm2GmSlotElems = mBaseSize * constInfoXCube.dBasicBlock;
            int64_t prevCoreOffset = constInfoXCube.aicIdx * 3 * mm2GmSlotElems;
            bmm2GmSlot[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset);
            bmm2GmSlot[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset + mm2GmSlotElems);
            bmm2GmSlot[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoreOffset + 2 * mm2GmSlotElems);
            workspace = workspace + constInfoXCube.coreNum * 3 * mm2GmSlotElems * sizeof(T);
        }
    }

    __aicore__ inline void InitQSeqLensParser()
    {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            qSeqLensParser.Init(cuSeqLensGmQ, seqUsedGmQ, constInfoXCube.cuSeqLensQSize, constInfoXCube.seqUsedQSize);
        } else {
            qSeqLensParser.Init(seqUsedGmQ, constInfoXCube.seqUsedQSize, constInfoXCube.s1Size);
        }
    }

    __aicore__ inline void InitKvSeqLensParser()
    {
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            kvSeqLensParser.Init(cuSeqLensGmKv, seqUsedGmKv, constInfoXCube.cuSeqLensKVSize, constInfoXCube.seqUsedKvSize);
        } else {
            kvSeqLensParser.Init(seqUsedGmKv, constInfoXCube.seqUsedKvSize, constInfoXCube.s2Size);
        }
    }

     __aicore__ inline void InitConstInfo()
    {
        constInfoXCube.aicIdx = GetBlockIdx();
        constInfoXCube.bSize = this->tilingData->flashAttnBaseParams.bSize;
        constInfoXCube.n2Size = this->tilingData->flashAttnBaseParams.n2Size;
        constInfoXCube.gSize = this->tilingData->flashAttnBaseParams.gSize;
        constInfoXCube.s1Size = this->tilingData->flashAttnBaseParams.s1Size;
        constInfoXCube.s2Size = this->tilingData->flashAttnBaseParams.s2Size;
        constInfoXCube.dSize = this->tilingData->flashAttnBaseParams.dSize;
        constInfoXCube.dSizeV = this->tilingData->flashAttnBaseParams.dSizeV;
        constInfoXCube.cuSeqLensQSize = this->tilingData->flashAttnBaseParams.cuSeqLensQSize;
        constInfoXCube.cuSeqLensKVSize = this->tilingData->flashAttnBaseParams.cuSeqLensKVSize;
        constInfoXCube.seqUsedQSize = this->tilingData->flashAttnBaseParams.seqUsedQSize;
        constInfoXCube.seqUsedKvSize = this->tilingData->flashAttnBaseParams.seqUsedKvSize;
        constInfoXCube.isKvContinuous = this->tilingData->flashAttnBaseParams.isKvContinuous != 0;
        constInfoXCube.coreNum = this->tilingData->flashAttnBaseParams.coreNum;
        
        constInfoXCube.sparseMode = this->tilingData->flashAttnAttenMaskParams.sparseMode;
        constInfoXCube.preTokens = this->tilingData->flashAttnAttenMaskParams.winLefts;
        constInfoXCube.nextTokens = this->tilingData->flashAttnAttenMaskParams.winRights;

        if constexpr (PAGE_ATTENTION) {
            constInfoXCube.maxBlockNumPerBatch = this->tilingData->flashAttnPageAttentionParams.maxBlockNumPerBatch;
            constInfoXCube.blockSize = this->tilingData->flashAttnPageAttentionParams.blockSize;
        }

        constInfoXCube.dBasicBlock = Align64Func((uint16_t)constInfoXCube.dSizeV);
    }

    __aicore__ inline uint32_t GetFAMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return FLASH_ATTN_METADATA_SIZE * FA_AIC_CORE_NUM * sectionIdx + 16U * coreIdx + metaIdx;
    }

    __aicore__ inline void CrossCoreBufferUnInit()
    {
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_0 + AIV0_AIV1_OFFSET);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_1);
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM1_1 + AIV0_AIV1_OFFSET);
        if constexpr (BMM2_TOUB) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_0 + AIV0_AIV1_OFFSET);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_1);
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CC_BMM2_1 + AIV0_AIV1_OFFSET);
        }
    }
    __aicore__ inline void FlashAttention(uint32_t sectionIdx)
    {
        CommonTaskLoop(*this, sectionIdx);
    }

    __aicore__ inline bool ShouldDispatchTask(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        return ((bN2Cur != bN2End_) || (gS1Cur != gS1OEnd_) || (s2Cur != s2OEnd_));
    }

    __aicore__ inline TASK_DEAL_MODE GetTaskDealMode(uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur)
    {
        bool isFirstTask = (bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_) && (s2Cur == s2OStart_);
        uint32_t bIdx = bN2Cur / constInfoXCube.n2Size;
        if (isFirstTask || prevBIdx != bIdx) {
            prevBIdx = bIdx;
            if (constInfoXCube.cuSeqLensKVSize == 0 && constInfoXCube.seqUsedKvSize == 0 && !constInfoXCube.isKvContinuous) {
                actSeqLensKv = SeqLenFromTensorList<LAYOUT>(keyPtr, bIdx);
            } else {
                actSeqLensKv = kvSeqLensParser.GetActualSeqLength(bIdx);
            }
            actSeqLensQ = qSeqLensParser.GetActualSeqLength(bIdx);
        }
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXCube.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;
        if (s2LoopTimes == 0 || gS1LoopTimes == 0) {
            if (gS1Cur == 0 && s2Cur == 0) {
                return TASK_DEAL_MODE::DEAL_ZERO;
            }
            return TASK_DEAL_MODE::SKIP_ZERO;
        }
        if (isFirstTask || bN2Cur != prevBN2Idx || gS1Cur != prevGS1Idx) {
            if constexpr (!HAS_MASK) {
                CalcCurS2StartEndNoSparse(bN2Cur, gS1Cur);
            } else {
                CalcCurS2StartEndWithSparse(bN2Cur, gS1Cur);
            }
            prevBN2Idx = bN2Cur;
            prevGS1Idx = gS1Cur;
        }

        if (curS2Start < curS2End && s2Cur >= curS2End) {
            return TASK_DEAL_MODE::S2_END;
        }

        if (s2Cur < curS2Start || s2Cur >= curS2End) {
            return TASK_DEAL_MODE::SKIP;
        }



        return TASK_DEAL_MODE::CREATE_TASK;
    }

    __aicore__ inline void GetPreNextTokenLeftUp(int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &preTokenLeftUp,
                                                 int64_t &nextTokenLeftUp)
    {
        preTokenLeftUp = constInfoXCube.preTokens;
        nextTokenLeftUp = constInfoXCube.nextTokens;
        fa_base_vector::GetSafeActToken(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp,
                                        constInfoXCube.sparseMode);

        if (constInfoXCube.sparseMode == fa_base_vector::BAND) {
            preTokenLeftUp = static_cast<int64_t>(actSeqLensQ) - static_cast<int64_t>(actSeqLensKv) + preTokenLeftUp;
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ) + nextTokenLeftUp;
        } else if (constInfoXCube.sparseMode == fa_base_vector::RIGHT_DOWN_CAUSAL) {
            nextTokenLeftUp = static_cast<int64_t>(actSeqLensKv) - static_cast<int64_t>(actSeqLensQ);
        }
    }

    __aicore__ inline void CalcCurS2StartEndNoSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        curS2Start = 0U;
        curS2End = (static_cast<uint32_t>(actSeqLensKv) + s2BaseSize - 1) / s2BaseSize;

        if ((bN2Cur == bN2Start_) && (gS1Cur == gS1OStart_)) {
            curS2Start = s2OStart_;
        }

        if ((bN2Cur == bN2End_) && (gS1Cur == gS1OEnd_)) {
            curS2End = s2OEnd_;
        }
    }

    __aicore__ inline void CalcCurS2StartEndWithSparse(uint32_t bN2Cur, uint32_t gS1Cur)
    {
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokenLeftUp, nextTokenLeftUp);

        int64_t s1GFirstToken = static_cast<int64_t>(gS1Cur) * static_cast<int64_t>(mBaseSize);
        int64_t s1GLastToken = AttentionCommon::Min(s1GFirstToken + static_cast<int64_t>(mBaseSize),
                                   static_cast<int64_t>(actSeqLensQ) * static_cast<int64_t>(constInfoXCube.gSize)) -
                               1;
        int64_t s1FirstToken = 0;
        int64_t s1LastToken = 0;
        if constexpr (GetOutUbFormat<LAYOUT>() == UbFormat::S1G) {
            s1FirstToken = static_cast<int64_t>(s1GFirstToken / constInfoXCube.gSize);
            s1LastToken = static_cast<int64_t>(s1GLastToken / constInfoXCube.gSize);
        } else {
            if (s1GFirstToken / static_cast<int64_t>(actSeqLensQ) == s1GLastToken / static_cast<int64_t>(actSeqLensQ)) {
                s1FirstToken = s1GFirstToken % static_cast<int64_t>(actSeqLensQ);
                s1LastToken = s1GLastToken % static_cast<int64_t>(actSeqLensQ);
            } else {
                s1FirstToken = 0;
                s1LastToken = static_cast<int64_t>(actSeqLensQ);
            }
        }

        uint32_t s2StartWithSparse = 0U;
        uint32_t s2EndWithSparse = 0U;
        int64_t s2FirstToken = s1FirstToken - preTokenLeftUp;
        int64_t s2LastToken = s1LastToken + nextTokenLeftUp;
        if (s2FirstToken >= static_cast<int64_t>(actSeqLensKv) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
            curS2Start = 0U;
            curS2End = 0U;
            return;
        }
        s2FirstToken = ClipSInnerToken(s2FirstToken, 0, static_cast<int64_t>(actSeqLensKv - 1));
        s2LastToken = ClipSInnerToken(s2LastToken, 0, static_cast<int64_t>(actSeqLensKv - 1));

        s2StartWithSparse = static_cast<uint32_t>(s2FirstToken) / s2BaseSize;
        s2EndWithSparse = static_cast<uint32_t>(s2LastToken) / s2BaseSize + 1U;

        curS2Start = s2StartWithSparse;
        curS2End = s2EndWithSparse;

        if (bN2Cur == bN2Start_ && gS1Cur == gS1OStart_) {
            curS2Start = AttentionCommon::Max(s2StartWithSparse, s2OStart_);
        }
        if (bN2Cur == bN2End_ && gS1Cur == gS1OEnd_) {
            curS2End = s2OEnd_ > 0U ? AttentionCommon::Min(s2EndWithSparse, s2OEnd_) : s2EndWithSparse;
        }
        return;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXCube &runInfo0 = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];
        RunInfoXCube &runInfoNegN = taskRunInfo[(loop - PRELOAD_N) & PRELOAD_TASK_CACHE_MASK];
        if (runInfo0.isValid) {
            ComputeMm1(runInfo0);
        }
        if (loop >= PRELOAD_N) {
            if (runInfoNegN.isValid) {
                ComputeMm2(runInfoNegN);
                runInfoNegN.isValid = false;
            }
        }
    }

    __aicore__ inline void ComputeMm1(RunInfoXCube &runInfo)
    {
        uint32_t slotIdx = runInfo.loop % 2U;
        auto bmm1Tensor = bmm1UbSlot[slotIdx].template ReinterpretCast<T>();
        IterateBmm1(bmm1Tensor, CC_BMM1_0 + slotIdx, CC_BMM1_0 + slotIdx, runInfo);
    }

    __aicore__ inline void ComputeMm2(RunInfoXCube &runInfo)
    {
        uint32_t outSlot = runInfo.loop % 2U;
        uint32_t l1PIdx  = runInfo.loop % 3U;
        uint32_t l1PCcId = (l1PIdx == 0U) ? CC_L1P_0 : (l1PIdx == 1U) ? CC_L1P_1 : CC_L1P_2;
        auto l1PTensor   = l1PSlot[l1PIdx].template ReinterpretCast<INPUT_T>();
        if constexpr (BMM2_TOUB) {
            auto bmm2Tensor = bmm2UbSlot[outSlot].template ReinterpretCast<T>();
            IterateBmm2(bmm2Tensor, CC_BMM2_0 + outSlot, CC_BMM2_0 + outSlot,
                                  l1PTensor, l1PCcId, runInfo);
        } else {
            IterateBmm2(bmm2GmSlot[runInfo.loop % 3U],
                                  CC_BMM2_0 + (runInfo.loop % 3U), 0U,
                                  l1PTensor, l1PCcId, runInfo);
        }
    }

    __aicore__ inline void CreateTask(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur,
                                      RunInfoXCube taskRunInfo[PRELOAD_TASK_CACHE_SIZE])
    {
        RunInfoXCube &runInfo = taskRunInfo[loop & PRELOAD_TASK_CACHE_MASK];
        CalcParams(loop, bN2Cur, gS1Cur, s2Cur, runInfo);
        runInfo.isValid = true;
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t bN2Cur, uint32_t gS1Cur, uint32_t s2Cur, RunInfoXCube &info)
    {
        info.loop = loop;
        info.bIdx = bN2Cur / constInfoXCube.n2Size;
        info.n2Idx = bN2Cur % constInfoXCube.n2Size;
        info.gS1Idx = gS1Cur * mBaseSize;
        info.s2Idx = s2Cur * s2BaseSize;

        info.actMSize = mBaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXCube.gSize;
        if (((gS1Cur + 1) * mBaseSize) > gS1Size) {
            info.actMSize = gS1Size - gS1Cur * mBaseSize;
        }
        info.actSingleLoopS2Size = s2BaseSize;
        if (((s2Cur + 1) * s2BaseSize) > actSeqLensKv) {
            info.actSingleLoopS2Size = actSeqLensKv - s2Cur * s2BaseSize;
        }
        if (constInfoXCube.isKvContinuous) {
            info.isChangeBatch = false;
        } else if (loop == 0) {
            info.isChangeBatch = true;
        } else {
            info.isChangeBatch = (info.n2Idx == 0 && s2Cur == curS2Start);
        }

        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokenLeftUp(actSeqLensQ, actSeqLensKv, preTokensLeftUp, nextTokensLeftUp);

        info.isFirstS2Loop = ((loop == 0) || (s2Cur == curS2Start));
        info.isLastS2Loop = (s2Cur + 1 == curS2End);
    }

    __aicore__ inline void UpdateAxisInfo(uint32_t &bN2Cur, uint32_t &gS1Cur,
                                          uint32_t &s2Cur)
    {
        uint64_t s2LoopTimes = (actSeqLensKv + s2BaseSize - 1) / s2BaseSize;
        uint64_t gS1Size = actSeqLensQ * constInfoXCube.gSize;
        uint64_t gS1LoopTimes = (gS1Size + mBaseSize - 1) / mBaseSize;

        if (s2Cur + 1 < s2LoopTimes) {
            s2Cur++;
            return;
        }



        s2Cur = 0;
        if (gS1Cur + 1 < gS1LoopTimes) {
            gS1Cur++;
            return;
        }



        gS1Cur = 0;
        bN2Cur++;
    }

    __aicore__ inline void GetFASectionInfo(uint32_t sectionIdx)
    {
        bN2Start_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_BN2_START_INDEX, sectionIdx));
        gS1OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_M_START_INDEX, sectionIdx));
        s2OStart_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_S2_START_INDEX, sectionIdx));
        bN2End_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_BN2_END_INDEX, sectionIdx));
        gS1OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_M_END_INDEX, sectionIdx));
        s2OEnd_ = faMetaDataGm.GetValue(GetFAMetaDataIndex(constInfoXCube.aicIdx, FLASH_ATTN_S2_END_INDEX, sectionIdx));
    }

    __aicore__ inline void Process()
    {
        if (constInfoXCube.aicIdx < constInfoXCube.coreNum) {
            InitBuffers();
            AllocEventID();
            for (uint32_t sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
                FlashAttention(sectionIdx);
            }
            FreeEventID();
            CrossCoreBufferUnInit();
        }
    }
}; // FlashAttentionNoQuantGqaKernel

} // namespace BaseApi

#endif // FLASH_ATTN_GQA_CUBE_H_
