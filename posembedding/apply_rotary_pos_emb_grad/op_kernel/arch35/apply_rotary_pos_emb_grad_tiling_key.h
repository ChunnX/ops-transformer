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
 * \file apply_rotary_pos_emb_grad_tiling_key.h
 * \brief apply_rotary_pos_emb_grad tiling key 定义
 * 四维 tiling key: REDUCE_TPL_KEY + DxTilingKey + DcosFlag + Layout
 * BNSD(Layout=2) 仅定义枚举值，不参与任何 SEL，无 kernel 实例化
 */

#ifndef __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define APPLY_ROPE_GRAD_DX_BIT_WIDTH 8
#define APPLY_ROPE_GRAD_LAYOUT_BIT_WIDTH 2

ASCENDC_TPL_ARGS_DECL(
    ApplyRotaryPosEmbGrad, REDUCE_TPL_KEY_DECL(),
    ASCENDC_TPL_UINT_DECL(DxTilingKey, APPLY_ROPE_GRAD_DX_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 201, 206),
    ASCENDC_TPL_UINT_DECL(DcosFlag, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(Layout, APPLY_ROPE_GRAD_LAYOUT_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 0, 0, 3));

// Layout 枚举值:
//   0 = BSND        (1S1D)  — BAB tiling
//   1 = SBND        (S11D/SB1D/BS1D) — AB tiling
//   2 = BNSD        (11SD/B1SD/1SND) — ABA/BA tiling，预留，不参与任何 SEL
//   3 = NO_BROADCAST + BROADCAST_BSN — A/B tiling
//       A(205): dy shape == cos shape, 全同
//       B(206): cos shape == (1,1,1,D), 全广播
// TND (3D) 不是独立 layout，经维度映射 (B=1, S=T, N=N) 后归为上述 4 种之一

// DcosFlag=0: 双路 Q+K 均不需要计算，仅做 dx 回传
// DcosFlag=1: Q 或 K 有 x 输入，需要同步累加 dcos/dsin

ASCENDC_TPL_SEL(
    // ========== Empty / A reduce template ==========
    // BSND + BAB
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    // SBND + AB
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),
    // NO_BROADCAST_AND_BROADCAST_BSN + A/B
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 205, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 3)),

    // ========== A reduce template ==========
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_A(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_A(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_A(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 205, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 3)),

    // ========== ARA_NORMAL reduce template ==========
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 3)),

    // ========== ARA_GROUP reduce template ==========
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 206),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 3)),

    // ========== ARARARARA_NORMAL reduce template ==========
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),

    // ========== ARARARARA_GROUP reduce template ==========
    ASCENDC_TPL_ARGS_SEL(
        REDUCE_TPL_KEY_SEL_ARARARARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)));

#endif // __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__
