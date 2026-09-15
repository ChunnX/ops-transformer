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
};
} // namespace QFA_KERNEL

#endif // QUANT_FLASH_ATTN_BLOCK_CUBE_MXFP8_SOFTMAX_FP16_H_
