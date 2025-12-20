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
 * \file mc2_copy_quant_matmul_params.h
 * \brief
 */

#ifndef __NEW_MC2_COPY_QUANT_MATMUL_PARAMS_H__
#define __NEW_MC2_COPY_QUANT_MATMUL_PARAMS_H__

#include "quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {
void NewCopyQuantBatchMatmulParams(DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& quantBatchMatmulParams, 
    Mc2QuantBatchMatmulV3TilingData& quantBmmV3TilingData);
}
#endif