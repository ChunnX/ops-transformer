/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON‑INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_flash_attn_grad_entry_rebase.h
 * \brief QuantFlashAttnGrad kernel entry (HIFLOAT8 only)
 */

#ifndef QUANT_FLASH_ATTN_GRAD_ENTRY_REGBASE_H_
#define QUANT_FLASH_ATTN_GRAD_ENTRY_REGBASE_H_

#include "quant_flash_attention_score_grad_common.h"
#include "quant_flash_attention_score_grad_tiling_data_regbase.h"
using namespace commondef;
using namespace AscendC;
using namespace optiling::QuantFag;
#include "quant_flash_attention_score_grad_s1s2_bn2gs1s2_post_regbase.h"
#include "quant_flash_attention_score_grad_presfmg_regbase.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "quant_flash_attention_score_grad_block_vec.h"
#include "quant_flash_attention_score_grad_block_cube.h"
#include "quant_flash_attention_score_grad_kernel.h"

namespace optiling {
namespace QuantFag {

#define INVOKE_QUANT_FAG_REGBASE_IMPL(OUTDTYPE, IS_N_EQUAL, HAS_ATTEN_MASK, HAS_SINK, S1TEMPLATENUM, S2TEMPLATENUM, DTEMPLATENUM, LAYOUT) \
do {                                                                                                           \
    QuantFlashAttentionScoreGradPresfmgRegbase<hifloat8_t, float, OUTDTYPE,                                    \
        DTemplateType::Aligned128,                                                                             \
        LAYOUT> opPre;                                                                                         \
    opPre.Init(dq, dk, dv, dout, attn_out, do_descale, cu_seqlens_kv, user, tilingData, &pipeIn);              \
    opPre.Process();                                                                                           \
    opPre.SyncALLCores();                                                                                      \
    pipeIn.Destroy();                                                                                          \
                                                                                                               \
    TPipe pipeBase;                                                                                            \
    using CubeBlockType = typename std::conditional<                                                           \
        g_coreType == AscendC::AIC,                                                                            \
        QuantFagBaseApi::QuantFAGBlockCube<hifloat8_t, float, OUTDTYPE, HAS_ATTEN_MASK, false, false, false,   \
        false, 0, IS_N_EQUAL, false, false, false,                                                             \
        false, BN2GS1S2, S1TemplateType::Aligned512,                                                           \
        S2TemplateType::Aligned512, DTemplateType::Aligned128>,                                                  \
        QuantFagBaseApi::QuantFAGBlockCubeDummy<hifloat8_t, float, OUTDTYPE, HAS_ATTEN_MASK, false, false, false, \
        false, 0, IS_N_EQUAL, false, false,                                                                    \
        false, false, BN2GS1S2, S1TemplateType::Aligned512,                      \
        S2TemplateType::Aligned512, DTemplateType::Aligned128>>::type;           \
    using VecBlockType = typename std::conditional<                                 \
        g_coreType == AscendC::AIC,                                                     \
        QuantFagBaseApi::QuantFAGBlockVecDummy<hifloat8_t, float, OUTDTYPE, HAS_ATTEN_MASK, false, false, false,    \
        false, 0, IS_N_EQUAL, false, false,       \
        false, false, BN2GS1S2, S1TemplateType::Aligned512,         \
        S2TemplateType::Aligned512, DTemplateType::Aligned128>,         \
        QuantFagBaseApi::QuantFAGBlockVec<hifloat8_t, float, OUTDTYPE, HAS_ATTEN_MASK, false, false, false,       \
        false, 0, IS_N_EQUAL, false, false, false,                 \
        false, BN2GS1S2, S1TemplateType::Aligned512,                     \
        S2TemplateType::Aligned512, DTemplateType::Aligned128>>::type;          \
    QuantFagBaseApi::QuantFlashAttentionScoreGradKernel<CubeBlockType, VecBlockType, LAYOUT> op;      \
    op.Init(k, v, dout, q, attn_out, softmax_lse,          \
        cu_seqlens_q, cu_seqlens_kv, q_descale, k_descale,           \
        v_descale, do_descale, ds_scale, p_scale, dq, dk, dv,            \
        user, tilingData, &pipeBase);        \
    op.Process();         \
    op.SyncALLCores();        \
    pipeBase.Destroy();            \
    TPipe pipePost;              \
    QuantFlashAttentionScoreGradS1S2BNGS1S2PostRegbase<hifloat8_t, float, OUTDTYPE, BN2GS1S2, false,       \
        0, false, false> opPost;               \
    opPost.Init(dq, dk, dv, dsink, user, tilingData, &pipePost);            \
    opPost.Process();              \
} while (0)

// implementation of kernel function
template <uint8_t HasAttenMask, uint8_t HasSink, uint16_t S1TemplateNum, uint16_t S2TemplateNum, uint8_t DTemplateNum, uint8_t IsNEqual, uint8_t Layout>
inline __aicore__ void
QuantRegbaseFAG(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *v, __gm__ uint8_t *dout, __gm__ uint8_t *attn_out,
        __gm__ uint8_t *q_descale, __gm__ uint8_t *k_descale, __gm__ uint8_t *v_descale, __gm__ uint8_t *do_descale,
        __gm__ uint8_t *p_scale, __gm__ uint8_t *ds_scale, __gm__ uint8_t *softmax_lse, __gm__ uint8_t *cu_seqlens_q,
        __gm__ uint8_t *cu_seqlens_kv, __gm__ uint8_t *seqused_q, __gm__ uint8_t *seqused_kv,
        __gm__ uint8_t *sinks, __gm__ uint8_t *attn_mask, __gm__ uint8_t *metadata, __gm__ uint8_t *dq, __gm__ uint8_t *dk,
        __gm__ uint8_t *dv, __gm__ uint8_t *dsink, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipeIn;
    SetMaskNorm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    using fagTiling = QuantFlashAttnGradTilingData;
    const __gm__ fagTiling *__restrict tilingData =
        (const __gm__ fagTiling *__restrict)((__gm__ uint8_t*)tiling_data);

    INVOKE_QUANT_FAG_REGBASE_IMPL(bfloat16_t, IsNEqual, HasAttenMask, HasSink, S1TemplateNum, S2TemplateNum, DTemplateNum, Layout);
}

} // namespace QuantFag
} // namespace optiling

#endif // QUANT_FLASH_ATTN_GRAD_ENTRY_REBASE_H_