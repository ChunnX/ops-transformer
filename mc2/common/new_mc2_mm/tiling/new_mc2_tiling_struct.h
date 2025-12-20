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
 * \file new_mc2_tiling_struct.h
 * \brief
 */

#ifndef __NEW_MC2_TILING_STRUCT_H__
#define __NEW_MC2_TILING_STRUCT_H__

#include "register/tilingdata_base.h"
#include "../../../3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_tiling.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(NewMc2MatmulTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, rankDim);
    TILING_DATA_FIELD_DEF(uint32_t, rankM);
    TILING_DATA_FIELD_DEF(uint32_t, rankID);
    TILING_DATA_FIELD_DEF(uint32_t, enableL2Tile);
    TILING_DATA_FIELD_DEF_STRUCT(Mc2BatchMatmulTilingData, bmmTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(NewMc2MatmulTilingDataOp, NewMc2MatmulTilingData);

} // namespace optiling

#endif