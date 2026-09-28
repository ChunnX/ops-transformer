/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_matmul_swiglu_quant_v2_weight_quant_asw_tiling_data.h
 * \brief
 */

#ifndef __OP_KERNEL_GMMSQ_WEIGHT_QUANT_ASW_TILING_DATA_H__
#define __OP_KERNEL_GMMSQ_WEIGHT_QUANT_ASW_TILING_DATA_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

namespace GMMSQArch35Tiling {
#pragma pack(push, 8)
struct GMMSQWeightQuantAswParams {
    uint8_t groupListType = 0;
    uint8_t reserved[7] = {0};
    uint32_t groupNum = 0;
    uint32_t l0cSize = 0;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct GMMSQWeightQuantAswTilingData {
    GMMSQWeightQuantAswParams gmmWeightQuantAswParam;
    TCubeTiling mmTilingData;
};
#pragma pack(pop)
} // namespace GMMSQArch35Tiling

#endif