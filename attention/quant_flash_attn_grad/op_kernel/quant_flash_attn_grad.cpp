/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_grad_apt.cpp
 * \brief QuantFlashAttnGrad kernel entry (HIFLOAT8 only)
 */

#ifdef KFC_L1_RESERVER_SIZE
#undef KFC_L1_RESERVER_SIZE
#define KFC_L1_RESERVER_SIZE 0
#else
#define KFC_L1_RESERVER_SIZE 0
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

#include "arch35/quant_flash_attn_grad_template_tiling_key.h"
#include "arch35/quant_flash_attn_grad_entry_regbase.h"
#include "arch35/quant_flash_attention_score_grad_tiling_data_regbase.h"

// QuantFlashAttnGrad kernel 入口
// implementation of Kernel function
template <uint8_t HasAttenMask, uint8_t HasSink, uint16_t S1TemplateNum, uint16_t S2TemplateNum, uint8_t DTemplateNum, uint8_t IsNEqual, uint8_t Layout>
__global__ __aicore__ void quant_flash_attn_grad(
    __gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v, __gm__ uint8_t *dout, __gm__ uint8_t *attn_out,
    __gm__ uint8_t *q_descale, __gm__ uint8_t *k_descale, __gm__ uint8_t *v_descale, __gm__ uint8_t *do_descale,
    __gm__ uint8_t *p_scale, __gm__ uint8_t *ds_scale, __gm__ uint8_t *softmax_lse, __gm__ uint8_t *cu_seqlens_q,
    __gm__ uint8_t *cu_seqlens_kv, __gm__ uint8_t *seqused_q, __gm__ uint8_t *seqused_kv,
    __gm__ uint8_t *sinks, __gm__ uint8_t *attn_mask, __gm__ uint8_t *metadata, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
    __gm__ uint8_t *dv, __gm__ uint8_t *dsink, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    using fagTiling = optiling::QuantFag::QuantFlashAttnGradTilingData;
    REGISTER_TILING_DEFAULT(fagTiling);
    // 量化主路径
    QuantRegbaseFAG<HasAttenMask, HasSink, S1TemplateNum, S2TemplateNum, DTemplateNum, IsNEqual, Layout>(
        q, k, v, dout, attn_out, q_descale, k_descale, v_descale, do_descale, p_scale,
        ds_scale, softmax_lse, cu_seqlens_q, cu_seqlens_kv, seqused_q,
        seqused_kv, sinks, attn_mask, metadata, dq, dk, dv, dsink, workspace, tiling_data);
}