/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_block_cube_noquant_gqa.h
 * \brief
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "fa_static_buffer.h"  // 自包含的Buffer/BufferManager/BuffersPolicy，替代common下的buff文件
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"         // PFA_CFG_ROW_MAJOR_UB
#include "../../../common/op_kernel/memory_copy_arch35.h" // SeqLenFromTensorList
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
// using namespace AttentionCommon;

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
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false>
class CubeBlockBase {
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

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    using L0AType = typename L0ABuffSel<INPUT_T>::Type;
    using L0BType = typename L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    __aicore__ inline CubeBlockBase(ConstInfoX &constInfo){};
};

template <typename INPUT_T, typename T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, uint8_t KvLayoutType = 0, bool useDn = false,
          bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockCube : public CubeBlockBase<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType,
                                                   dVTemplateType, KvLayoutType, useDn, bmm2Write2Ub, splitD> {
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

    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout, KvLayoutType, PAGE_ATTENTION>();

    using ROPE_T = INPUT_T;
    using Q_T = INPUT_T;
    using KV_T = INPUT_T;
    using MM_T = T;

    // L0A/L0B类型保留给matmul使用
    using L0AType = typename L0ABuffSel<INPUT_T>::Type;
    using L0BType = typename L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::NO_QUANT>;

    // ====== L1/L0C 缓冲区布局常量 ======
    static constexpr bool L1Q_IS_SINGLE = (dBaseSize > 256);
    static constexpr bool L1KV_IS_SINGLE = (s2BaseSize == 256 && dBaseSize > 128);
    static constexpr bool L0C_IS_4BUFF = (mBaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
                                          mBaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4);
    static constexpr uint32_t L0C_SLOT_COUNT = L0C_IS_4BUFF ? 4 : 2;

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

    static constexpr ActualSeqLensMode KV_PARSER_MODE = GetKvActSeqMode<LAYOUT, PAGE_ATTENTION>();
    using KvSeqParserType = typename std::conditional<IS_TND_LAYOUT, ActualSeqLensParser<KV_PARSER_MODE, int32_t, true>,
                                                      ActualSeqLensParser<KV_PARSER_MODE, int32_t>>::type;

    using FaGmTensorQ = FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_NEEDS_WZH>;
    using FaGmTensorKV = FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_NEEDS_WZH>;

    /* =====================GM变量(with layout)==================== */
    FaGmTensorQ queryGm;
    FaGmTensorKV keyGm;
    FaGmTensorKV valueGm;
    FaGmTensorQ queryRopeGm;
    FaGmTensorKV keyRopeGm;
    GlobalTensor<int32_t> blockTableGm;

    QSeqParserType *qSeqParserPtr = nullptr;
    KvSeqParserType *kvSeqParserPtr = nullptr;

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

    const ConstInfoX &constInfo;

    /*============================================================================== */
    __aicore__ inline FANoQuantGqaBlockCube(ConstInfoX &constInfo)
        : CubeBlockBase<INPUT_T, T, layout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType,
                        useDn, bmm2Write2Ub, splitD>(constInfo),
          constInfo(constInfo){};

    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable,
                                         QSeqParserType &qParser, KvSeqParserType &kvParser)
    {
        this->qSeqParserPtr = &qParser;
        this->kvSeqParserPtr = &kvParser;
        InitCubeInput(query, key, value, blockTable);
    }

    __aicore__ inline void InitBuffers()
    {
        // ─────────────────────────────────────────────────────────────────────────
        // 与 VEC 侧保持一致：InitBuffers 仅负责缓冲区地址初始化，不插入任何同步原语。
        // 流水预置 SetFlag 统一由 AllocEventID() 完成。
        //
        // L1 内存布局（从 offset=0 起）:
        //   [0 .. 3*mBaseSize*s2BaseSize*sizeof(Q_T)) : l1PBuffers (3buff, InitMMResBuf 分配)
        //   [FA_L1_CUBE_BASE ...) : l1QBuffers / l1KBuffers / l1VBuffers
        // ─────────────────────────────────────────────────────────────────────────
        if constexpr (dBaseSize > 256) {
            // L1Q = Single (dBaseSize>256), L1KV: Single if s2==256&&d>128, else DB
            constexpr uint32_t mm1LeftSize  = mBaseSize * dBaseSize * sizeof(Q_T);
            constexpr uint32_t mm1RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            constexpr uint32_t mm2RightSize = s2BaseSize * l1BaseD * sizeof(KV_T);
            constexpr uint32_t l1QOff       = FA_L1_CUBE_BASE;
            constexpr uint32_t l1KOff       = l1QOff + mm1LeftSize;

            l1QSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1QOff, mm1LeftSize);

            if constexpr (s2BaseSize == 256 && dBaseSize > 128) {
                // Single + Single + Single
                constexpr uint32_t l1VOff = l1KOff + mm1RightSize;
                l1KSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1KOff, mm1RightSize);
                l1VSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1VOff, mm2RightSize);
            } else {
                // Single + DB + DB
                constexpr uint32_t l1VOff = l1KOff + 2U * mm1RightSize;
                l1KSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1KOff, mm1RightSize);
                l1KSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1KOff + mm1RightSize, mm1RightSize);
                l1VSlot[0] = LocalTensor<uint8_t>(TPosition::A1, l1VOff, mm2RightSize);
                l1VSlot[1] = LocalTensor<uint8_t>(TPosition::A1, l1VOff + mm2RightSize, mm2RightSize);
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
        if constexpr (L0C_IS_4BUFF) {
            constexpr uint32_t l0cSlotSize = (uint32_t)(L0C_SIZE / NUM_4) * (uint32_t)KB_TO_BYTES;
            l0CSlot[0] = LocalTensor<uint8_t>(TPosition::C1, 0U, l0cSlotSize);
            l0CSlot[1] = LocalTensor<uint8_t>(TPosition::C1, l0cSlotSize, l0cSlotSize);
            l0CSlot[2] = LocalTensor<uint8_t>(TPosition::C1, 2U * l0cSlotSize, l0cSlotSize);
            l0CSlot[3] = LocalTensor<uint8_t>(TPosition::C1, 3U * l0cSlotSize, l0cSlotSize);
        } else {
            constexpr uint32_t l0cSlotSize = (uint32_t)(L0C_SIZE / NUM_2) * (uint32_t)KB_TO_BYTES;
            l0CSlot[0] = LocalTensor<uint8_t>(TPosition::C1, 0U, l0cSlotSize);
            l0CSlot[1] = LocalTensor<uint8_t>(TPosition::C1, l0cSlotSize, l0cSlotSize);
        }
    }

    __aicore__ inline void InitCubeInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                         __gm__ uint8_t *blockTable)
    {
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }

        InitQBuffer(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size, constInfo.dSize, queryGm,
                    query);

        keyPtr = key;
        valuePtr = value;
        if (constInfo.isKvContinuous) {
            InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, constInfo.dSize,
                         keyGm, key);
            InitKVBuffer(constInfo.bSize, constInfo.s2Size, constInfo.n2Size, constInfo.blockSize, constInfo.dSizeV,
                         valueGm, value);
        }
    }

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, FaGmTensorQ &qGmTensor, __gm__ uint8_t *gm)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ Q_T *)gm);
        if constexpr (Q_NEEDS_WZH) {
            qGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, *this->qSeqParserPtr);
        } else {
            qGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, *this->qSeqParserPtr);
        }
    }

    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize, uint32_t n2Size,
                                        uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensorKV &kvGmTensor,
                                        __gm__ uint8_t *gm)
    {
        kvGmTensor.gmTensor.SetGlobalBuffer((__gm__ KV_T *)gm);

        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, n2Size, kvSeqSize, headDim);
            kvGmTensor.offsetCalculator.Init(*this->kvSeqParserPtr);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(n2Size, headDim, *this->kvSeqParserPtr);
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
                                          RunInfoX &runInfo)
    {
        uint32_t nopeDealSize = dRealSize;

        uint32_t dstStride = (runInfo.actMSize + 15) >> 4 << 4;
        if (nopeDealSize > 0) {
            FaL1Tensor<Q_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmCoord gmCoord{.bIdx = runInfo.bIdx,
                            .n2Idx = runInfo.n2Idx,
                            .gS1Idx = runInfo.gS1Idx,
                            .dIdx = dOffset,
                            .gS1DealSize = runInfo.actMSize,
                            .dDealSize = nopeDealSize};
            copyQueryGmToL1(l1Tensor, queryGm, gmCoord);
        }
    }

    __aicore__ inline void CopyQueryTile(const LocalTensor<Q_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        CopyQuerySlice(dstTensor, 0, dSize, runInfo);
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyKeySlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                        RunInfoX &runInfo)
    {
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 15) >> 4 << 4;

        uint32_t nopeDealSize = dRealSize;
        uint32_t ropeDealSize = 0;

        if (nopeDealSize > 0) {
            FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor, .rowCount = dstStride};

            GmKvCoord gmCoord{.bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
                              .n2Idx = runInfo.n2Idx,
                              .s2Idx = runInfo.s2Idx,
                              .dIdx = dOffset,
                              .s2DealSize = runInfo.actSingleLoopS2Size,
                              .dDealSize = nopeDealSize};
            copyKvGmToL1(l1Tensor, keyGm, gmCoord);
        }
    }

    // copy key with full s2 and split D
    __aicore__ inline void CopyValueSlice(const LocalTensor<KV_T> &dstTensor, uint32_t dOffset, uint32_t dRealSize,
                                          RunInfoX &runInfo)
    {
        FaL1Tensor<KV_T, L1Format::NZ> l1Tensor{.tensor = dstTensor,
                                                .rowCount = AttentionCommon::Align(runInfo.actSingleLoopS2Size, 16U)};

        GmKvCoord gmCoord{.bIdx = constInfo.isKvContinuous ? runInfo.bIdx : 0,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = dOffset,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dRealSize};
        copyKvGmToL1(l1Tensor, valueGm, gmCoord);
    }

    // 全量拷贝
    __aicore__ inline void CopyKeyTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        uint32_t dSize = constInfo.dSize;
        CopyKeySlice(dstTensor, 0, dSize, runInfo);
    }

    __aicore__ inline void CopyValueTile(const LocalTensor<KV_T> &dstTensor, RunInfoX &runInfo)
    {
        CopyValueSlice(dstTensor, 0, constInfo.dSizeV, runInfo);
    }

    __aicore__ inline void UpdateKey(uint32_t bIdx)
    {
        ListTensorDesc keyListTensorDesc((__gm__ void *)(this->keyPtr));
        __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(this->keyPtr, bIdx);
        keyGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)key_);
        keyGm.offsetCalculator.Init(0, constInfo.n2Size, s2Size, constInfo.dSize);
    }

    __aicore__ inline void UpdateValue(uint32_t bIdx)
    {
        ListTensorDesc valueListTensorDesc((__gm__ void *)(this->valuePtr));
        __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);
        uint64_t s2Size = SeqLenFromTensorList<LAYOUT>(valuePtr, bIdx);
        valueGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)value_);
        valueGm.offsetCalculator.Init(0, constInfo.n2Size, s2Size, constInfo.dSize);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm1(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
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

    __aicore__ inline void FixpipeMm1(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
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
    __aicore__ inline void IterateBmm1NdL0Split(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
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
                                    (uint32_t)(constInfo.dSize), false, true);
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
    __aicore__ inline void IterateBmm1NdL1SplitK(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
    {
        constexpr uint32_t baseK = l1BaseD;
        uint32_t kLoops = (constInfo.dSize + baseK - 1) / baseK;

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
                realK = constInfo.dSize - k * baseK;
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
    __aicore__ inline void IterateBmm1DnSplitK(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
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
                                    (uint32_t)(constInfo.dSize), false, true);

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

    __aicore__ inline void FixpipeMm1Dn(const LocalTensor<T> &dstTensor, const LocalTensor<T> &l0C, RunInfoX &runInfo)
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
    __aicore__ inline void IterateBmm1Dn(OutTensorT &outputTensor, uint32_t ccId0, uint32_t ccId1, RunInfoX &runInfo)
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
                                    (uint32_t)(constInfo.dSize), false, true);

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
                                        LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
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
                                                LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
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
        uint32_t nLoops = ((uint32_t)constInfo.dSizeV + baseN - 1) / baseN; // 尾块处理
        uint32_t realN = baseN;
        for (uint32_t n = 0; n < nLoops; ++n) {
            if (n == nLoops - 1) {
                uint32_t tailSize = (uint32_t)constInfo.dSizeV % baseN;
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
                                              RunInfoX &runInfo)
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
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV;
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(dstTensor, l0C, fixpipeParams);
    }

    template<typename OutTensorT>
    __aicore__ inline void IterateBmm2l0Split(OutTensorT &outputTensor, uint32_t outCcId0, uint32_t outCcId1,
                                               LocalTensor<Q_T> &l1PTensor, uint32_t l1PCcId, RunInfoX &runInfo)
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
            (uint32_t)constInfo.dSizeV,            // singleN 128
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

        FixpipeMm2PartialN(outputTensor, l0CTensor, constInfo.dSizeV, runInfo);

        SetFlag<HardEvent::FIX_M>(L0CEvt(cIdx)); // 释放
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(outCcId0 + AIV0_AIV1_OFFSET);
    }

}; // FANoQuantGqaBlockCube
} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_BLOCK_CUBE_H_
