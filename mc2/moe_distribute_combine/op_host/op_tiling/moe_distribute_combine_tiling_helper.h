/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_tiling_helper.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H
#define MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "mc2_log.h"
#include "tiling_base/tiling_base.h"
using namespace Ops::Transformer::OpTiling;

namespace optiling {
constexpr uint32_t EXPAND_X_INDEX = 0;
constexpr uint32_t EXPERT_IDS_INDEX = 1;
constexpr uint32_t EXPAND_IDX_INDEX = 2;
constexpr uint32_t EP_SEND_COUNTS_INDEX = 3;
constexpr uint32_t EXPERT_SCALES_INDEX = 4;
constexpr uint32_t TP_SEND_COUNTS_INDEX = 5;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 6;
constexpr uint32_t SHARED_EXPERT_X_INDEX = 11;
constexpr uint32_t OUTPUT_X_INDEX = 0;
constexpr uint32_t ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;

constexpr uint32_t THREE_DIMS = 3U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;

class MoeDistributeCombineTilingHelper {
public:
    static ge::graphStatus TilingCheckMoeDistributeCombine(gert::TilingContext *context, const char *nodeName);

protected:
    static bool CheckTensorDim(gert::TilingContext *context, const char *nodeName);
    static bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName);
    static bool CheckTensorFormat(const gert::TilingContext *context, const char *nodeName);

private:
    inline static bool CheckInputTensorDim(const gert::TilingContext *context, const char *nodeName);
    inline static bool CheckInputSendCountsTensorDim(const gert::TilingContext *context, const char *nodeName);
    inline static bool CheckInputExpertScalesTensorDim(const gert::TilingContext *context, const char *nodeName);
    inline static bool CheckOutputTensorDim(const gert::TilingContext *context, const char *nodeName);
};
} // namespace optiling
#endif // MOE_DISTRIBUTE_COMBINE_TILING_HELPER_H