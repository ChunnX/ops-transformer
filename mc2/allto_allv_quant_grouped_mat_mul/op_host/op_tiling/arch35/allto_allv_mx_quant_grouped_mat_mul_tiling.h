/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_mx_quant_grouped_mat_mul_tiling.h
 * \brief V1 MX inherits V2 MX
 */
#ifndef ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H
#define ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H

#include "mc2/allto_allv_quant_grouped_mat_mul_v2/op_host/op_tiling/arch35/allto_allv_mx_quant_grouped_mat_mul_tiling.h"

namespace optiling {

class AlltoAllvMXQuantGmmTiling : public AlltoAllvMXQuantGmmV2Tiling {
public:
    explicit AlltoAllvMXQuantGmmTiling(gert::TilingContext *context) : AlltoAllvMXQuantGmmV2Tiling(context){};
};

}  // namespace optiling
#endif  // ALLTO_ALLV_MX_QUANT_GROUPED_MATMUL_TILING_H