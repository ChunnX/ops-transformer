/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * (see LICENSE at root)
 */
/*! * moe_distribute_dispatch_v2_a2_tiling_key.h */
#ifndef MDISPATCH_V2_A2_TILING_KEY_H
#define MDISPATCH_V2_A2_TILING_KEY_H
#include "../moe_distribute_dispatch_v2_tiling_key_decl.h"

namespace Mc2Tiling {
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(TILINGKEY_QUANT_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_QUANT),
                         ASCENDC_TPL_BOOL_SEL(TILINGKEY_SCALE, 0, 1),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_FULLMESH, ASCENDC_TPL_UI_LIST, TILINGKEY_NO_FULLMESH),
                         ASCENDC_TPL_UINT_SEL(TILINGKEY_COMM_MODE, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_MTE,
                                              TILINGKEY_TPL_AICPU),
                         ASCENDC_TPL_UINT_SEL(ARCH_TAG, ASCENDC_TPL_UI_LIST, TILINGKEY_TPL_A2),
                         ASCENDC_TPL_TILING_STRUCT_SEL(MoeDistributeDispatchA2TilingData)),
);
} // namespace Mc2Tiling
#endif
