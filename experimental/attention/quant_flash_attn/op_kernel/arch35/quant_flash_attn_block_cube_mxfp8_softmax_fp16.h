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
 * \file quant_flash_attn_block_cube_mxfp8_softmax_fp16.h
 * \brief MxFP8 Softmax FP16 Cube Block（AIC 侧 L1/L0 静态 Tensor 分配 + C1/F1/C2/F2 计算接口）
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_

#include "kernel_operator.h"
#include "quant_flash_attn_common_def.h"
#include "../../../common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../../common/op_kernel/memcopy/gm_coord.h"
#include "../../../common/op_kernel/memcopy/parser.h"

using namespace AscendC;

#include "../../../common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "../../../common/op_kernel/memcopy/fa_l1_tensor.h"

namespace QFA_KERNEL {

template <typename QUANT_T, typename SCALE_T>
class QuantFlashAttnBlockCubeMxfp8SoftmaxFp16 {
public:
    using MM_OUT_T = float;
    using SOFTMAX_T = half;

    static constexpr uint32_t mBaseSize  = S1_BASE_SIZE;   // 256
    static constexpr uint32_t s2BaseSize = S2_BASE_SIZE;   // 256
    static constexpr uint32_t dBaseSize  = D_BASE_SIZE;    // 128
    static constexpr uint32_t dVBaseSize = DV_BASE_SIZE;   // 128

private:
    static constexpr uint32_t S2_SPLIT_SIZE = S2_SUB_LOOP_SIZE; // 128, C1 subLoop M 轴
    static constexpr uint32_t MM2_SUB_K_SIZE = 128U;             // 128, C2 subK 轴
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;          // MxFP8 量化组大小（每 32 元素共享 scale）

    // ===== L1 私有区尺寸常量 =====
    static constexpr uint32_t L1_Q_SINGLE_SLOT_SIZE      = 128 * 256;   // 32KB
    static constexpr uint32_t L1_Q_BUFCNT                = 2;
    static constexpr uint32_t L1_QSCALE_SINGLE_SLOT_SIZE = 1024;        // 1KB
    static constexpr uint32_t L1_QSCALE_BUFCNT           = 2;
    static constexpr uint32_t L1_K_SINGLE_SLOT_SIZE      = 128 * 128;   // 16KB
    static constexpr uint32_t L1_K_BUFCNT                = 2;
    static constexpr uint32_t L1_KSCALE_SINGLE_SLOT_SIZE = 512;         // 512B
    static constexpr uint32_t L1_KSCALE_BUFCNT           = 2;
    static constexpr uint32_t L1_V_SINGLE_SLOT_SIZE      = 256 * 128;   // 32KB
    static constexpr uint32_t L1_V_BUFCNT                = 2;
    static constexpr uint32_t L1_VSCALE_SINGLE_SLOT_SIZE = 1024;        // 1KB
    static constexpr uint32_t L1_VSCALE_BUFCNT           = 2;
    static constexpr uint32_t L1_ONEFILL_V_SIZE          = 144 * 128;   // 18KB
    static constexpr uint32_t L1_ONEFILL_VSCALE_SIZE     = 576;          // 576B
    static constexpr uint16_t ONE_FILL_BLOCK_CNT = L1_ONEFILL_V_SIZE / 32;           // 576
    static constexpr uint16_t ONE_FILL_SCALE_BLOCK_CNT = L1_ONEFILL_VSCALE_SIZE / 32; // 18
    static constexpr uint16_t FP8_ONE_U16   = 0x3838U;  // fp8 e4m3 1.0, byte pair
    static constexpr uint16_t E8M0_ONE_U16  = 0x7F7FU;  // e8m0 2^0, byte pair

    // ===== L1 私有区 BASE（从共享区末尾链式推导）=====
    static constexpr uint32_t L1_Q_BASE = L1_SHARED_REGION_SIZE;
    static constexpr uint32_t L1_QSCALE_BASE = L1_Q_BASE + L1_Q_BUFCNT * L1_Q_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_K_BASE = L1_QSCALE_BASE + L1_QSCALE_BUFCNT * L1_QSCALE_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_KSCALE_BASE = L1_K_BASE + L1_K_BUFCNT * L1_K_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_V_BASE = L1_KSCALE_BASE + L1_KSCALE_BUFCNT * L1_KSCALE_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_VSCALE_BASE = L1_V_BASE + L1_V_BUFCNT * L1_V_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_ONEFILL_V_BASE = L1_VSCALE_BASE + L1_VSCALE_BUFCNT * L1_VSCALE_SINGLE_SLOT_SIZE;
    static constexpr uint32_t L1_ONEFILL_VSCALE_BASE = L1_ONEFILL_V_BASE + L1_ONEFILL_V_SIZE;

    // ===== L0 尺寸常量 =====
    static constexpr uint32_t L0A_MM1_SIZE = 2 * 128 * 128;             // 32KB, K 双缓冲
    static constexpr uint32_t MM2_L0A_M  = dVBaseSize + ROWSUM_ROWS;    // 128+16 = 144
    static constexpr uint32_t MM2_L0A_K  = MM2_SUB_K_SIZE;               // 128
    static constexpr uint32_t L0A_MM2_SIZE = MM2_L0A_M * MM2_L0A_K;     // 18KB, V_ext
    static constexpr uint32_t L0B_Q_SIZE   = 128 * 256;                 // 32KB, Q 常驻
    static constexpr uint32_t L0B_P_SIZE   = 128 * 128;                 // 16KB, P 单块（C2 串行无关缓冲）
    static constexpr uint32_t L0C_SIZE     = 2 * 128 * 256;             // 256KB, C1+C2 ring

    // ===== L0A BASE =====
    static constexpr uint32_t L0A_MM1_BASE = 0;
    static constexpr uint32_t L0A_MM2_BASE = L0A_MM1_BASE + L0A_MM1_SIZE;

    // ===== L0B BASE =====
    static constexpr uint32_t L0B_Q_BASE = 0;
    static constexpr uint32_t L0B_P_BASE = L0B_Q_BASE + L0B_Q_SIZE;

    using QSeqParser = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;
    using KvSeqParser = ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, int32_t>;

    const ConstInfo &constInfo_;

    CopyQueryGmToL1<QUANT_T, GmFormat::BNGSD, L1Format::NZ> copyQueryGmToL1_;
    CopyQueryScaleGmToL1<SCALE_T, GmFormat::BNGSD> copyQueryScaleGmToL1_; 
    CopyKvGmToL1<QUANT_T, GmFormat::BNSD> copyKvGmToL1_;
    CopyKeyScaleGmToL1<SCALE_T, GmFormat::BNSD> copyKeyScaleGmToL1_;

    // ===== GM 输入（BNSD 单模板，直接 GlobalTensor）=====
    FaGmTensor<QUANT_T, GmFormat::BNGSD, int32_t> queryGm_;
    FaGmTensor<SCALE_T, GmFormat::BNGSD, int32_t> queryScaleGm_;
    FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t> keyGm_;
    FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t> keyScaleGm_;
    FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t> valueGm_;
    FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t> valueScaleGm_;

    // ===== L1 Buffer =====
    // --- 共享区（P/pScale，AIC/AIV 同偏移，常量来源 common_def.h）---
    LocalTensor<QUANT_T> pL1V0Tensor_;
    LocalTensor<QUANT_T> pL1V1Tensor_;
    LocalTensor<SCALE_T> pScaleL1V0Tensor_;
    LocalTensor<SCALE_T> pScaleL1V1Tensor_;

    // --- 私有区（Q/K/V 及 scale）---
    LocalTensor<QUANT_T>  qL1Tensor_;
    LocalTensor<SCALE_T> qScaleL1Tensor_;
    LocalTensor<QUANT_T>  kL1Tensor_;
    LocalTensor<SCALE_T> kScaleL1Tensor_;
    LocalTensor<QUANT_T>  vL1Tensor_;
    LocalTensor<SCALE_T> vScaleL1Tensor_;

    // --- 刷 1 常驻（独立 L1 buffer，一次 Fill 全程只读）---
    LocalTensor<QUANT_T>  oneFillVL1Tensor_;
    LocalTensor<SCALE_T> oneFillVScaleL1Tensor_;

    // ===== L0 Buffer =====
    LocalTensor<QUANT_T>  mm1L0ATensor_;  // L0A, K 左矩阵双缓冲 [128,128]×2
    LocalTensor<QUANT_T>  mm2L0ATensor_;  // L0A, V_ext [144,128]（前 128 行 V + 后 16 行全 1）
    LocalTensor<QUANT_T>  qL0BTensor_;    // L0B, Q 常驻槽 [128,256]
    LocalTensor<QUANT_T>  pL0BTensor_;    // L0B, P 单块 [128,128]（C2 串行无 buffering）
    LocalTensor<MM_OUT_T> mmL0CTensor_;   // L0C, C1[128,256]+C2[144,128] ring 双槽 [128,256]×2

    // ===== UB（Fixpipe 目标，TOTAL ≤ 256KB）=====
    static constexpr uint32_t UB_MM1RES_SIZE = 128U * 128U; // [128,128] half/槽
    static constexpr uint32_t UB_MM1RES_BUFCNT = 2U;        // ring 双缓冲
    LocalTensor<SOFTMAX_T> mm1ResUB_;
    static constexpr uint32_t UB_MM2RES_SIZE = (dBaseSize + 1U) * 256U; // [129,256] fp32 = 132KB
    LocalTensor<MM_OUT_T> mm2ResUB_;

    // ===== Event ID（内部流水同步，SetFlag/WaitFlag 显式编排）=====
    // mte2 ↔ mte1：Q / K / V 的 GM→L1（K/V 独立事件，避免并行在途串扰）
    static constexpr uint32_t Q_EVENT0  = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1  = EVENT_ID3;
    uint32_t qBufId_ = 0;
    static constexpr uint32_t K_EVENT0 = EVENT_ID4;
    static constexpr uint32_t K_EVENT1 = EVENT_ID5;
    static constexpr uint32_t V_EVENT0 = EVENT_ID6;
    static constexpr uint32_t V_EVENT1 = EVENT_ID7;
    uint32_t kBufId_  = 0;
    uint32_t vBufId_  = 0;

    // mte1 ↔ m：Q→L0B / K→L0A（使用 EVENT_ID2~5，Q_L0B=2/3, K_L0A=4/5）
    static constexpr uint32_t Q_L0B_EVENT0  = EVENT_ID2;
    static constexpr uint32_t K_L0A_EVENT0 = EVENT_ID4;
    static constexpr uint32_t K_L0A_EVENT1 = EVENT_ID5;
    uint32_t kL0aBufId_ = 0;

    // C2 L0AB（P L0B / V L0A 单块，事件同步用固定 bufId=0）
    static constexpr uint32_t MM2_L0AB_EVENT0 = EVENT_ID6;
    uint32_t mm2L0abBufId_ = 0;

    // m ↔ fix：L0C（使用 EVENT_ID2~3，C1 ring 双缓冲）
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID3;
    uint32_t l0cBufId_ = 0;

    // ===== Q/K 搬运 helper（BNSD 布局，GM→L1）=====
    __aicore__ inline void InitQBuffer(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                        FaGmTensor<QUANT_T, GmFormat::BNGSD, int32_t> &qGmTensor, __gm__ uint8_t *gm,
                                        QSeqParser &qSeqParser)
    {
        qGmTensor.gmTensor.SetGlobalBuffer((__gm__ QUANT_T *)gm);
        qGmTensor.offsetCalculator.Init(b, n2, g, s1, d, qSeqParser);
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                             FaGmTensor<SCALE_T, GmFormat::BNGSD, int32_t> &qScaleGmTensor,
                                             __gm__ uint8_t *gm, QSeqParser &qSeqParser)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        qScaleGmTensor.offsetCalculator.Init(b, n2, g, s1, d, qSeqParser);
    }

    __aicore__ inline void InitKBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                        FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t> &kVGmTensor,
                                        __gm__ uint8_t *gm, KvSeqParser &kvSeqParser)
    {
        kVGmTensor.gmTensor.SetGlobalBuffer((__gm__ QUANT_T *)gm);
        kVGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                             FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t> &kScaleGmTensor,
                                             __gm__ uint8_t *gm, KvSeqParser &kvSeqParser)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        kScaleGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void InitVBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                        FaGmTensor<QUANT_T, GmFormat::BNSD, int32_t> &vGmTensor,
                                        __gm__ uint8_t *gm, KvSeqParser &kvSeqParser)
    {
        vGmTensor.gmTensor.SetGlobalBuffer((__gm__ QUANT_T *)gm);
        vGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void InitVScaleBuffer(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                             FaGmTensor<SCALE_T, GmFormat::BNSD, int32_t> &vScaleGmTensor,
                                             __gm__ uint8_t *gm, KvSeqParser &kvSeqParser)
    {
        vScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ SCALE_T *)gm);
        vScaleGmTensor.offsetCalculator.Init(b, n2, s2, d, kvSeqParser);
    }

    __aicore__ inline void AllocEventID()
    {
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(K_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(K_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT1);

        SetFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
        SetFlag<HardEvent::M_MTE1>(K_L0A_EVENT0);
        SetFlag<HardEvent::M_MTE1>(K_L0A_EVENT1);

        SetFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(K_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(K_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(K_L0A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(K_L0A_EVENT1);

        WaitFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
    }


public:
    __aicore__ inline explicit QuantFlashAttnBlockCubeMxfp8SoftmaxFp16(ConstInfo &constInfo)
        : constInfo_(constInfo) {}

    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                   __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                   __gm__ uint8_t *dequantScaleValue,
                   QSeqParser &qSeqParser, KvSeqParser &kvSeqParser)
    {
        InitQBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSize,
                    queryGm_, query, qSeqParser);
        InitQScaleBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSize,
                        queryScaleGm_, dequantScaleQuery, qSeqParser);
        InitKBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize,
                    keyGm_, key, kvSeqParser);
        InitKScaleBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize,
                        keyScaleGm_, dequantScaleKey, kvSeqParser);
        InitVBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize,
                    valueGm_, value, kvSeqParser);
        InitVScaleBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.s2Size, constInfo_.dSize,
                        valueScaleGm_, dequantScaleValue, kvSeqParser);
        (void)kvSeqParser;
    }

    __aicore__ inline void InitTensors()
    {
        AllocEventID();

        // ===== L1 共享区 =====
        pL1V0Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_P_V0_BASE,
                                            L1_P_BUFCNT * L1_P_V0_SINGLE_SLOT_SIZE);
        pL1V1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_P_V1_BASE,
                                            L1_P_BUFCNT * L1_P_V1_SINGLE_SLOT_SIZE);
        pScaleL1V0Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V0_BASE,
                                                 L1_P_BUFCNT * L1_PSCALE_V0_SINGLE_SLOT_SIZE);
        pScaleL1V1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_PSCALE_V1_BASE,
                                                 L1_P_BUFCNT * L1_PSCALE_V1_SINGLE_SLOT_SIZE);

        // ===== L1 私有区 =====
        qL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_Q_BASE,
                                          L1_Q_BUFCNT * L1_Q_SINGLE_SLOT_SIZE);
        qScaleL1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_QSCALE_BASE,
                                               L1_QSCALE_BUFCNT * L1_QSCALE_SINGLE_SLOT_SIZE);
        kL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_K_BASE,
                                          L1_K_BUFCNT * L1_K_SINGLE_SLOT_SIZE);
        kScaleL1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_KSCALE_BASE,
                                               L1_KSCALE_BUFCNT * L1_KSCALE_SINGLE_SLOT_SIZE);
        vL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_V_BASE,
                                          L1_V_BUFCNT * L1_V_SINGLE_SLOT_SIZE);
        vScaleL1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_VSCALE_BASE,
                                               L1_VSCALE_BUFCNT * L1_VSCALE_SINGLE_SLOT_SIZE);
        oneFillVL1Tensor_ = LocalTensor<QUANT_T>(TPosition::A1, L1_ONEFILL_V_BASE,
                                                 L1_ONEFILL_V_SIZE);
        oneFillVScaleL1Tensor_ = LocalTensor<SCALE_T>(TPosition::A1, L1_ONEFILL_VSCALE_BASE,
                                                      L1_ONEFILL_VSCALE_SIZE);

        // ===== L0A =====
        mm1L0ATensor_ = LocalTensor<QUANT_T>(TPosition::A2, L0A_MM1_BASE, L0A_MM1_SIZE);
        mm2L0ATensor_ = LocalTensor<QUANT_T>(TPosition::A2, L0A_MM2_BASE, L0A_MM2_SIZE);

        // ===== L0B =====
        qL0BTensor_ = LocalTensor<QUANT_T>(TPosition::B2, L0B_Q_BASE, L0B_Q_SIZE);
        pL0BTensor_ = LocalTensor<QUANT_T>(TPosition::B2, L0B_P_BASE, L0B_P_SIZE);

        // ===== L0C =====
        mmL0CTensor_ = LocalTensor<MM_OUT_T>(TPosition::CO1, 0U, L0C_SIZE);

        mm1ResUB_ = LocalTensor<SOFTMAX_T>(TPosition::VECCALC, 0U, UB_MM1RES_SIZE * UB_MM1RES_BUFCNT);
        mm2ResUB_ = LocalTensor<MM_OUT_T>(TPosition::VECCALC,
                                          UB_MM1RES_SIZE * UB_MM1RES_BUFCNT * sizeof(SOFTMAX_T) / sizeof(MM_OUT_T),
                                          UB_MM2RES_SIZE);

        InitOneFillBuffer();
        InitL0BufferForReduceSum();
    }

    __aicore__ inline void InitOneFillBuffer()
    {
        InitConstValueParams<uint16_t> dataParams(1U, ONE_FILL_BLOCK_CNT, 0U, FP8_ONE_U16);
        Fill(oneFillVL1Tensor_.template ReinterpretCast<uint16_t>(), dataParams);
        PipeBarrier<PIPE_MTE2>();

        InitConstValueParams<uint16_t> scaleParams(1U, ONE_FILL_SCALE_BLOCK_CNT, 0U, E8M0_ONE_U16);
        Fill(oneFillVScaleL1Tensor_.template ReinterpretCast<uint16_t>(), scaleParams);
        PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void InitL0BufferForReduceSum()
    {
        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = 0U;
        dataParams.kStartPosition = 0U;
        dataParams.mStep = MM2_L0A_M / 16U;
        dataParams.kStep = MM2_L0A_K / 32U;
        dataParams.srcStride = dataParams.mStep;
        dataParams.dstStride = dataParams.mStep;
        dataParams.ifTranspose = false;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = MM2_L0A_M / 16U;
        scaleParams.yStep = MM2_L0A_K / 32U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        LoadData(mm2L0ATensor_.template ReinterpretCast<QUANT_T>(),
                 oneFillVL1Tensor_.template ReinterpretCast<QUANT_T>(),
                 oneFillVScaleL1Tensor_,
                 dataParams, scaleParams);
    }

    __aicore__ inline void ReleaseTensors() { FreeEventID(); }

    __aicore__ inline void ComputeMm1(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx, bool isLastSubLoop)
    {
        // ———— Q 常驻载入（行首一次，同一行所有 K subLoop 复用）————
        if (unlikely(runInfo.isFirstS2Loop && subLoopIdx == 0)) {
            WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId_);
            CopyQGmToL1(runInfo);
            CopyQScaleGmToL1(runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId_);
            WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + qBufId_);

            WaitFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
            LoadQToL0B(runInfo);
            SetFlag<HardEvent::MTE1_M>(Q_L0B_EVENT0);
            WaitFlag<HardEvent::MTE1_M>(Q_L0B_EVENT0);
        }

        // ———— K 流式搬运（每 subLoop）————
        WaitFlag<HardEvent::MTE1_MTE2>(K_EVENT0 + kBufId_);
        CopyKGmToL1(runInfo, subLoopIdx);
        CopyKScaleGmToL1(runInfo, subLoopIdx);
        SetFlag<HardEvent::MTE2_MTE1>(K_EVENT0 + kBufId_);
        WaitFlag<HardEvent::MTE2_MTE1>(K_EVENT0 + kBufId_);

        {
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
            WaitFlag<HardEvent::M_MTE1>(K_L0A_EVENT0 + kL0aBufId_);
            LoadKToL0A(runInfo, subLoopIdx);
            SetFlag<HardEvent::MTE1_M>(K_L0A_EVENT0 + kL0aBufId_);
            WaitFlag<HardEvent::MTE1_M>(K_L0A_EVENT0 + kL0aBufId_);

            uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE))
                                    ? runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE
                                    : S2_SPLIT_SIZE;
            MatmulKQ(runInfo, actKSize);
            SetFlag<HardEvent::M_MTE1>(K_L0A_EVENT0 + kL0aBufId_);
            kL0aBufId_ = (kL0aBufId_ + 1U) % 2U;
            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
        }

        SetFlag<HardEvent::MTE1_MTE2>(K_EVENT0 + kBufId_);
        kBufId_ = (kBufId_ + 1U) % 2U;
        if (unlikely(runInfo.isLastS2Loop && isLastSubLoop)) {
            SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qBufId_);
            SetFlag<HardEvent::M_MTE1>(Q_L0B_EVENT0);
            qBufId_ = (qBufId_ + 1U) % 2U;
        }
    }

    __aicore__ inline void FixpipeMm1(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx, uint32_t subBlockId)
    {
        // subBlockId=0 消耗唯一一次 M_FIX，subBlockId=1 不再等待（L0C 未变）
        if (subBlockId == 0U) {
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
        }

        uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE))
                                ? runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE
                                : S2_SPLIT_SIZE;

        uint32_t actVecS1Size = (subBlockId == 0U)
            ? AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE)
            : (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);

        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (actVecS1Size + 7U) >> 3 << 3;
        fixpipeParams.mSize = (actKSize + 1U) >> 1 << 1;
        fixpipeParams.srcStride = ((actKSize + 15U) / 16U) * 16U;
        fixpipeParams.dstStride = actVecS1Size;
        fixpipeParams.quantPre = QuantMode_t::F322F16;
        fixpipeParams.deqScalar = static_cast<uint64_t>(
            *reinterpret_cast<const int32_t *>(&constInfo_.scaleValue));
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.subBlockId = runInfo.loop % 2U;

        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = fixpipeParams.mSize;
        fixpipeParams.params.dstNdStride = fixpipeParams.dstStride;

        uint32_t l0cOffset  = l0cBufId_ * 128U * 256U;
        uint32_t l0cColOff  = subBlockId * S1_SUB_BLOCK_SIZE;
        uint32_t ubRingSlot = runInfo.loop % UB_MM1RES_BUFCNT;
        uint32_t ubBase     = ubRingSlot * UB_MM1RES_SIZE;

        Fixpipe<SOFTMAX_T, MM_OUT_T, CFG_ROW_MAJOR>(
            mm1ResUB_[ubBase],
            mmL0CTensor_[l0cOffset + l0cColOff],
            fixpipeParams);

        // subBlockId=1 做完最后一次 Fixpipe 后释放 L0C 并翻转槽
        if (subBlockId == 1U) {
            SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
            l0cBufId_ = (l0cBufId_ + 1U) % 2U;
        }
    }

    __aicore__ inline void ComputeMm2(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t vCore)
    {
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);

        // ———— V 整行搬运 ————
        WaitFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId_);
        CopyVGmToL1(runInfo);
        CopyVScaleGmToL1(runInfo);
        SetFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId_);
        WaitFlag<HardEvent::MTE2_MTE1>(V_EVENT0 + vBufId_);

        // ———— V/P 分 subK 加载 + MatmulVP 累加 ————
        uint32_t subKLoop = AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, MM2_SUB_K_SIZE);
        bool isFirstSubK = true;
        for (uint32_t subK = 0U; subK < subKLoop; ++subK) {
            uint32_t actSubKSize = (subK + 1U == subKLoop)
                ? runInfo.actSingleLoopS2Size - subK * MM2_SUB_K_SIZE
                : MM2_SUB_K_SIZE;

            WaitFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0);
            if (unlikely(runInfo.isLastS2Loop && actSubKSize < MM2_SUB_K_SIZE)) {
                InitL0BufferForReduceSum();
            }
            LoadPToL0B(runInfo, vCore, subK, actSubKSize);
            LoadVToL0A(runInfo, subK, actSubKSize);
            SetFlag<HardEvent::MTE1_M>(MM2_L0AB_EVENT0);
            WaitFlag<HardEvent::MTE1_M>(MM2_L0AB_EVENT0);
            MatmulVP(runInfo, actSubKSize, isFirstSubK);
            SetFlag<HardEvent::M_MTE1>(MM2_L0AB_EVENT0);
            isFirstSubK = false;
        }

        SetFlag<HardEvent::MTE1_MTE2>(V_EVENT0 + vBufId_);
        vBufId_ = (vBufId_ + 1U) % 2U;

        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);
    }

    __aicore__ inline void FixpipeMm2(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t vCore)
    {
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId_);

        uint32_t l0cOffset = l0cBufId_ * 128U * 256U;

        uint32_t actVecS1Size = (vCore == 0U)
            ? AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE)
            : (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);

        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = (actVecS1Size + 7U) >> 3 << 3;
        fixpipeParams.mSize = dBaseSize + 1U;
        fixpipeParams.srcStride = ((dBaseSize + ROWSUM_ROWS + 15U) / 16U) * 16U;
        fixpipeParams.dstStride = actVecS1Size;
        fixpipeParams.quantPre = QuantMode_t::NoQuant;
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.subBlockId = vCore;
        fixpipeParams.params.ndNum = 1U;
        fixpipeParams.params.srcNdStride = fixpipeParams.mSize;
        fixpipeParams.params.dstNdStride = fixpipeParams.dstStride;

        Fixpipe<MM_OUT_T, MM_OUT_T, CFG_ROW_MAJOR>(mm2ResUB_, mmL0CTensor_[l0cOffset], fixpipeParams);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId_);
        l0cBufId_ = (l0cBufId_ + 1U) % 2U;
    }

    __aicore__ inline void CopyQGmToL1(const RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        uint32_t l1BaseOffset = qBufId_ * L1_Q_SINGLE_SLOT_SIZE;
        uint32_t dstStride = (runInfo.actMSize + 31U) / 32U * 32U;
        FaL1Tensor<QUANT_T, L1Format::NZ> l1Tensor{.tensor = qL1Tensor_[l1BaseOffset], .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.n2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0U,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dBaseSize};
        copyQueryGmToL1_(l1Tensor, queryGm_, gmCoord);
    }

    __aicore__ inline void CopyQScaleGmToL1(const RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        uint32_t l1BaseOffset = qBufId_ * L1_QSCALE_SINGLE_SLOT_SIZE;
        uint32_t dstStride = (runInfo.actMSize + 31U) / 32U * 32U;
        uint32_t dDealSize = dBaseSize / MXFP_GROUP_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = qScaleL1Tensor_[l1BaseOffset], .rowCount = dstStride};

        GmCoordGs1Merge gmCoord{.bIdx = runInfo.bIdx,
                                .n2Idx = runInfo.n2Idx,
                                .gS1Idx = runInfo.gS1Idx,
                                .dIdx = 0U,
                                .gS1DealSize = runInfo.actMSize,
                                .dDealSize = dDealSize};
        copyQueryScaleGmToL1_(l1Tensor, queryScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyKGmToL1(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx)
    {
        uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE))
                                ? runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE
                                : S2_SPLIT_SIZE;
        uint32_t l1BaseOffset = kBufId_ * L1_K_SINGLE_SLOT_SIZE;
        FaL1Tensor<QUANT_T, L1Format::NZ> l1Tensor{.tensor = kL1Tensor_[l1BaseOffset], .rowCount = actKSize};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx + subLoopIdx * S2_SPLIT_SIZE,
                          .dIdx = 0U,
                          .s2DealSize = actKSize,
                          .dDealSize = dBaseSize};
        copyKvGmToL1_(l1Tensor, keyGm_, gmCoord);
    }

    __aicore__ inline void CopyKScaleGmToL1(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx)
    {
        uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE))
                                ? runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE
                                : S2_SPLIT_SIZE;
        uint32_t dDealSize = dBaseSize / MXFP_GROUP_SIZE;
        uint32_t l1BaseOffset = kBufId_ * L1_KSCALE_SINGLE_SLOT_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = kScaleL1Tensor_[l1BaseOffset], .rowCount = actKSize};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx + subLoopIdx * S2_SPLIT_SIZE,
                          .dIdx = 0U,
                          .s2DealSize = actKSize,
                          .dDealSize = dDealSize};
        copyKeyScaleGmToL1_(l1Tensor, keyScaleGm_, gmCoord);
    }

    __aicore__ inline void CopyVGmToL1(const RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        uint32_t l1BaseOffset = vBufId_ * L1_V_SINGLE_SLOT_SIZE;
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 63U) / 64U * 64U;
        FaL1Tensor<QUANT_T, L1Format::NZ> l1Tensor{.tensor = vL1Tensor_[l1BaseOffset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0U,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dBaseSize};
        copyKvGmToL1_(l1Tensor, valueGm_, gmCoord);
    }

    __aicore__ inline void CopyVScaleGmToL1(const RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        uint32_t l1BaseOffset = vBufId_ * L1_VSCALE_SINGLE_SLOT_SIZE;
        uint32_t dstStride = (runInfo.actSingleLoopS2Size + 63U) / 64U * 64U;
        uint32_t dDealSize = dBaseSize / MXFP_GROUP_SIZE;
        FaL1Tensor<SCALE_T, L1Format::NZ> l1Tensor{.tensor = vScaleL1Tensor_[l1BaseOffset], .rowCount = dstStride};

        GmKvCoord gmCoord{.bIdx = runInfo.bIdx,
                          .n2Idx = runInfo.n2Idx,
                          .s2Idx = runInfo.s2Idx,
                          .dIdx = 0U,
                          .s2DealSize = runInfo.actSingleLoopS2Size,
                          .dDealSize = dDealSize};
        copyKeyScaleGmToL1_(l1Tensor, valueScaleGm_, gmCoord);
    }

    __aicore__ inline void LoadQToL0B(const RunInfoMxfp8SoftmaxFp16 &runInfo)
    {
        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = 0U;
        dataParams.kStartPosition = 0U;
        dataParams.mStep = ((runInfo.actMSize + 31U) / 32U * 32U) / 16U;
        dataParams.kStep = dBaseSize / 32U;
        dataParams.srcStride = dataParams.mStep;
        dataParams.dstStride = dataParams.mStep;
        dataParams.ifTranspose = false;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = dataParams.mStep;
        scaleParams.yStep = 1U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t qL1Offset      = qBufId_ * L1_Q_SINGLE_SLOT_SIZE;
        uint32_t qScaleL1Offset = qBufId_ * L1_QSCALE_SINGLE_SLOT_SIZE;
        LoadData(qL0BTensor_, qL1Tensor_[qL1Offset],
                 qScaleL1Tensor_[qScaleL1Offset], dataParams, scaleParams);
    }

    __aicore__ inline void LoadKToL0A(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx)
    {
        uint32_t actKSize = (subLoopIdx + 1 == AttentionCommon::CeilDiv(runInfo.actSingleLoopS2Size, S2_SPLIT_SIZE))
                                ? runInfo.actSingleLoopS2Size - subLoopIdx * S2_SPLIT_SIZE
                                : S2_SPLIT_SIZE;
        uint32_t actKSizeAlign = (actKSize + 15U) / 16U * 16U;

        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = subLoopIdx * (S2_SPLIT_SIZE / 16U);
        dataParams.kStartPosition = 0U;
        dataParams.mStep = actKSizeAlign / 16U;
        dataParams.kStep = dBaseSize / 32U;
        dataParams.srcStride = runInfo.actSingleLoopS2SizeAlign / 16U;
        dataParams.dstStride = actKSizeAlign / 16U;
        dataParams.ifTranspose = false;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = subLoopIdx * (S2_SPLIT_SIZE / 16U);
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = actKSizeAlign / 16U;
        scaleParams.yStep = 1U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t kL1Offset      = kBufId_ * L1_K_SINGLE_SLOT_SIZE;
        uint32_t kScaleL1Offset = kBufId_ * L1_KSCALE_SINGLE_SLOT_SIZE;
        LoadData(mm1L0ATensor_[kL0aBufId_ * 128U * 128U], kL1Tensor_[kL1Offset],
                 kScaleL1Tensor_[kScaleL1Offset], dataParams, scaleParams);
    }

    __aicore__ inline void MatmulKQ(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t actKSize)
    {
        MmadParams mmadParams;
        mmadParams.m = actKSize;
        mmadParams.n = runInfo.actMSizeAlign32;
        mmadParams.k = dBaseSize;
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource = false;
        mmadParams.disableGemv = true;

        uint32_t l0aOffset = kL0aBufId_ * 128U * 128U;
        uint32_t l0cOffset = l0cBufId_ * 128U * 256U;
        Mmad(mmL0CTensor_[l0cOffset], mm1L0ATensor_[l0aOffset], qL0BTensor_, mmadParams);
    }

    __aicore__ inline void LoadPToL0B(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t vCore,
                                        uint32_t subKIdx, uint32_t actSubKSize)
    {
        uint32_t pSlot = runInfo.pSlot;

        LocalTensor<QUANT_T> &pL1Tensor      = (vCore == 0U) ? pL1V0Tensor_  : pL1V1Tensor_;
        LocalTensor<SCALE_T> &pScaleL1Tensor  = (vCore == 0U) ? pScaleL1V0Tensor_ : pScaleL1V1Tensor_;

        uint32_t actVecS1Size = (vCore == 0U)
            ? AttentionCommon::Min(runInfo.actMSize, S1_SUB_BLOCK_SIZE)
            : (runInfo.actMSize > S1_SUB_BLOCK_SIZE ? runInfo.actMSize - S1_SUB_BLOCK_SIZE : 0U);

        uint32_t pL1Offset       = pSlot * 256U * 128U;
        uint32_t pScaleL1Offset  = pSlot * 1024U;

        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = 0U;
        dataParams.kStartPosition = subKIdx * (MM2_SUB_K_SIZE / 32U);
        dataParams.mStep = actVecS1Size / 16U;
        dataParams.kStep = actSubKSize / 32U;
        dataParams.srcStride = runInfo.actSingleLoopS2SizeAlign / 16U;
        dataParams.dstStride = actVecS1Size / 16U;
        dataParams.ifTranspose = true;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = 0U;
        scaleParams.yStartPosition = subKIdx * (MM2_SUB_K_SIZE / 64U);
        scaleParams.xStep = actVecS1Size / 16U;
        scaleParams.yStep = (actSubKSize + 63U) / 64U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t l0bOffset = 0U;
        LoadData(pL0BTensor_[l0bOffset], pL1Tensor[pL1Offset],
                 pScaleL1Tensor[pScaleL1Offset], dataParams, scaleParams);
    }

    __aicore__ inline void LoadVToL0A(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subKIdx,
                                       uint32_t actSubKSize)
    {
        uint32_t actSubKSizeAlign = (actSubKSize + 63U) / 64U * 64U;

        LoadData2DParamsV2 dataParams;
        dataParams.mStartPosition = subKIdx * (MM2_SUB_K_SIZE / 16U);
        dataParams.kStartPosition = 0U;
        dataParams.mStep = actSubKSizeAlign / 16U;
        dataParams.kStep = dBaseSize / 32U;
        dataParams.srcStride = (runInfo.actSingleLoopS2Size + 63U) / 64U * 64U / 16U;
        dataParams.dstStride = (dBaseSize + 15U) / 16U + 1U;
        dataParams.ifTranspose = true;

        LoadData2DMxParams scaleParams;
        scaleParams.xStartPosition = subKIdx * (MM2_SUB_K_SIZE / 16U);
        scaleParams.yStartPosition = 0U;
        scaleParams.xStep = (dBaseSize + 15U) / 16U;
        scaleParams.yStep = (actSubKSizeAlign + 63U) / 64U;
        scaleParams.srcStride = scaleParams.yStep;
        scaleParams.dstStride = scaleParams.yStep;

        uint32_t vL1Offset      = vBufId_ * L1_V_SINGLE_SLOT_SIZE;
        uint32_t vScaleL1Offset = vBufId_ * L1_VSCALE_SINGLE_SLOT_SIZE;
        LoadData(mm2L0ATensor_.template ReinterpretCast<QUANT_T>(), vL1Tensor_[vL1Offset],
                 vScaleL1Tensor_[vScaleL1Offset], dataParams, scaleParams);
    }

    __aicore__ inline void MatmulVP(const RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subKSize,
                                     bool isFirstSubK)
    {
        MmadParams mmadParams;
        mmadParams.m = dVBaseSize + ROWSUM_ROWS;  // 144 = 128(V) + 16(全1 rowsum)
        mmadParams.n = runInfo.actMSizeAlign32;
        mmadParams.k = subKSize;
        mmadParams.cmatrixInitVal = isFirstSubK;
        mmadParams.cmatrixSource   = !isFirstSubK;
        mmadParams.disableGemv = true;

        uint32_t l0cOffset = l0cBufId_ * 128U * 256U;
        uint32_t l0bOffset = 0U;
        Mmad(mmL0CTensor_[l0cOffset], mm2L0ATensor_, pL0BTensor_[l0bOffset], mmadParams);
    }

};
} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
