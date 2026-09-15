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

using namespace AscendC;

namespace QFA_KERNEL {

template <typename QUANT_T, typename SCALE_T>
class QuantFlashAttnBlockCubeMxfp8SoftmaxFp16 {
public:
    using MM_OUT_T = float;

    static constexpr uint32_t mBaseSize  = S1_BASE_SIZE;   // 256
    static constexpr uint32_t s2BaseSize = S2_BASE_SIZE;   // 256
    static constexpr uint32_t dBaseSize  = D_BASE_SIZE;    // 128
    static constexpr uint32_t dVBaseSize = DV_BASE_SIZE;   // 128

private:
    static constexpr uint32_t S2_SPLIT_SIZE = S2_SUB_LOOP_SIZE; // 128, C1 subLoop M 轴
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
    static constexpr uint32_t MM2_L0A_K  = S2_SPLIT_SIZE;               // 128
    static constexpr uint32_t L0A_MM2_SIZE = MM2_L0A_M * MM2_L0A_K;     // 18KB, V_ext
    static constexpr uint32_t L0B_Q_SIZE   = 128 * 256;                 // 32KB, Q 常驻
    static constexpr uint32_t L0B_P_SIZE   = 2 * 128 * 128;             // 32KB, P 双缓冲
    static constexpr uint32_t L0C_SIZE     = 2 * 128 * 256;             // 256KB, C1+C2 ring

    // ===== L0A BASE =====
    static constexpr uint32_t L0A_MM1_BASE = 0;
    static constexpr uint32_t L0A_MM2_BASE = L0A_MM1_BASE + L0A_MM1_SIZE;

    // ===== L0B BASE =====
    static constexpr uint32_t L0B_Q_BASE = 0;
    static constexpr uint32_t L0B_P_BASE = L0B_Q_BASE + L0B_Q_SIZE;

    const ConstInfo &constInfo_;

    // ===== GM 输入（BNSD 单模板，直接 GlobalTensor）=====
    GlobalTensor<QUANT_T> queryGm_;
    GlobalTensor<QUANT_T> keyGm_;
    GlobalTensor<QUANT_T> valueGm_;
    GlobalTensor<SCALE_T> queryScaleGm_;
    GlobalTensor<SCALE_T> keyScaleGm_;
    GlobalTensor<SCALE_T> valueScaleGm_;

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
    LocalTensor<QUANT_T>  pL0BTensor_;    // L0B, P 双缓冲 [128,128]×2
    LocalTensor<MM_OUT_T> mmL0CTensor_;   // L0C, C1[128,256]+C2[144,128] ring 双槽 [128,256]×2

public:
    __aicore__ inline explicit QuantFlashAttnBlockCubeMxfp8SoftmaxFp16(ConstInfo &constInfo)
        : constInfo_(constInfo) {}

    __aicore__ inline void InitInput(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                   __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                   __gm__ uint8_t *dequantScaleValue)
    {
        queryGm_.SetGlobalBuffer((__gm__ QUANT_T *)query);
        keyGm_.SetGlobalBuffer((__gm__ QUANT_T *)key);
        valueGm_.SetGlobalBuffer((__gm__ QUANT_T *)value);
        queryScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)dequantScaleQuery);
        keyScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)dequantScaleKey);
        valueScaleGm_.SetGlobalBuffer((__gm__ SCALE_T *)dequantScaleValue);
    }

    __aicore__ inline void InitTensors()
    {
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

    __aicore__ inline void ReleaseTensors() {}

    __aicore__ inline void ComputeMm1(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx)
    {
#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[C1][AIC%u] loop=%u subLoop=%u\n", constInfo_.aicIdx, runInfo.loop, subLoopIdx);
#endif
    }

    __aicore__ inline void FixpipeMm1(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t subLoopIdx, uint32_t subBlockId)
    {
#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[F1][AIC%u] loop=%u subLoop=%u subBlk=%u\n", constInfo_.aicIdx, runInfo.loop, subLoopIdx,
                        subBlockId);
#endif
    }

    __aicore__ inline void ComputeMm2(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t vCore)
    {
#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[C2][AIC%u] loop=%u vCore=%u\n", constInfo_.aicIdx, runInfo.loop, vCore);
#endif
    }

    __aicore__ inline void FixpipeMm2(RunInfoMxfp8SoftmaxFp16 &runInfo, uint32_t vCore)
    {
#if QFA_SF16_DEBUG_PRINTF
        AscendC::printf("[F2][AIC%u] loop=%u vCore=%u\n", constInfo_.aicIdx, runInfo.loop, vCore);
#endif
    }

};
} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
