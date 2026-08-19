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
 * \file ffn_to_attention_v2_tilling_key.h
 * \brief
 */

#define TILINGKEY_COMM_MTE 0
#define TILINGKEY_COMM_URMA 1
 
ASCENDC_TPL_ARGS_DECL(FFNToAttentionV2,
    ASCENDC_TPL_BOOL_DECL(TILINGKEY_RANK_TABLE_MODE, 0, 1),
    ASCENDC_TPL_UINT_DECL(ARCH_TAG, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A5),
    ASCENDC_TPL_UINT_DECL(COMM_MODE, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                          TILINGKEY_COMM_MTE, TILINGKEY_COMM_URMA), );
 
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_RANK_TABLE_MODE, 0),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ..., TILINGKEY_TPL_A5),
                         ASCENDC_TPL_UINT_SEL(COMM_MODE, ..., TILINGKEY_COMM_URMA),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FFNToAttentionV2TilingData)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(TILINGKEY_RANK_TABLE_MODE, 1),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ..., TILINGKEY_TPL_A5),
                         ASCENDC_TPL_UINT_SEL(COMM_MODE, ..., TILINGKEY_COMM_URMA),
                         ASCENDC_TPL_TILING_STRUCT_SEL(FFNToAttentionV2TilingData)));
