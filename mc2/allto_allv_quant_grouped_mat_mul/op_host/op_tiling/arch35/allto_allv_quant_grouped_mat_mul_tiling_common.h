/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_quant_grouped_mat_mul_tiling_common.h
 * \brief V1 inherits V2, fixed to use CCU mode
 */
#ifndef ALLTO_ALLV_QUANT_GROUPED_MATMUL_TILING_COMMON_H
#define ALLTO_ALLV_QUANT_GROUPED_MATMUL_TILING_COMMON_H

#include "mc2/allto_allv_quant_grouped_mat_mul_v2/op_host/op_tiling/arch35/allto_allv_quant_grouped_mat_mul_v2_tiling_common.h"
#include "mc2/allto_allv_quant_grouped_mat_mul_v2/op_kernel/allto_allv_quant_grouped_mat_mul_v2_tiling.h"

namespace optiling {

// V1 inherits V2 Common, reuse all methods except SetHcclTiling
class AlltoAllvQuantGmmTilingCommon : public AlltoAllvQuantGmmV2TilingCommon {
public:
    explicit AlltoAllvQuantGmmTilingCommon(gert::TilingContext *context) : AlltoAllvQuantGmmV2TilingCommon(context){};

protected:
    // Override SetHcclTiling to fix CCU mode (no comm_mode attr in V1 proto)
    ge::graphStatus SetHcclTiling() const override;
};

}  // namespace optiling
#endif  // ALLTO_ALLV_QUANT_GROUPED_MATMUL_TILING_COMMON_H