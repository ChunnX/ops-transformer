/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * ...
 */

/*!
 * \file apply_rotary_pos_emb_grad_tiling_key.h
 * \brief apply_rotary_pos_emb_grad tiling key
 *
 * 四维 tiling key: REDUCE_TPL_KEY + DxTilingKey(8b) + DcosFlag(1b) + Layout(2b)
 *
 * Layout: 0=BSND(BAB), 1=SBND(AB), 2=BNSD(参数预留), 3=NO_BROADCAST(A)
 * DxTilingKey: 203(BAB), 204(AB), 205(A)
 * A(205): cos == grad_query_embed, N=1, 无广播轴 → DcosFlag=1 退化为 EMPTY
 */
#ifndef __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__

#include "atvoss/reduce/reduce_tiling_key_decl.h"

#define APPLY_ROPE_GRAD_DX_BIT_WIDTH 8
#define APPLY_ROPE_GRAD_LAYOUT_BIT_WIDTH 2

ASCENDC_TPL_ARGS_DECL(
    ApplyRotaryPosEmbGrad, REDUCE_TPL_KEY_DECL(),
    ASCENDC_TPL_UINT_DECL(DxTilingKey, APPLY_ROPE_GRAD_DX_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 1, 203, 205),
    ASCENDC_TPL_UINT_DECL(DcosFlag, 1, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_UINT_DECL(Layout, APPLY_ROPE_GRAD_LAYOUT_BIT_WIDTH, ASCENDC_TPL_UI_RANGE, 0, 0, 3));

// Layout=2(BNSD) 参数预留, 不参与任何 SEL

ASCENDC_TPL_SEL(
    // ================================================================
    // EMPTY — DcosFlag=0 for all; DcosFlag=1 only for A(205) where no broadcast
    // ================================================================
    // BSND + BAB
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    // SBND + AB
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),
    // NO_BROADCAST + A (DcosFlag={0,1}: DcosFlag=1 时无广播轴, 退化到 EMPTY)
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_EMPTY(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 205),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 3)),

    // ================================================================
    // A reduce — DcosFlag=1
    // ================================================================
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_A(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),
    // ================================================================
    // ARA_NORMAL / ARA_GROUP — DcosFlag=1, BAB+AB only
    // ================================================================
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),

    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 204),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 1)),

    // ================================================================
    // ARARARARA — 仅 BSND+BAB
    // ================================================================
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_NORMAL(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)),
    ASCENDC_TPL_ARGS_SEL(REDUCE_TPL_KEY_SEL_ARARARARA_GROUP(),
        ASCENDC_TPL_UINT_SEL(DxTilingKey, ASCENDC_TPL_UI_LIST, 203),
        ASCENDC_TPL_UINT_SEL(DcosFlag, ASCENDC_TPL_UI_LIST, 1),
        ASCENDC_TPL_UINT_SEL(Layout, ASCENDC_TPL_UI_LIST, 0)));

#endif // __APPLY_ROTARY_POS_EMB_GRAD_TILING_KEY_H__
