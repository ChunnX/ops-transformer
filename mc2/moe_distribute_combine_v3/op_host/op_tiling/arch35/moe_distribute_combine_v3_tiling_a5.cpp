/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v3_tiling_a5.cpp
 * \brief
 */

#include "../moe_distribute_combine_v3_tiling_base.h"
#include "../../../../moe_distribute_combine_v2/op_host/op_tiling/arch35/moe_distribute_combine_tiling_arch35.h"
namespace optiling {
ge::graphStatus MoeDistributeCombineV2TilingFuncNew(gert::TilingContext* context, const CombineV2Config& config)
{
    MoeDistributeCombineV2TilingFuncA5 funcA5;
    return funcA5.MoeDistributeCombineV2TilingFuncNew(context, config);
}
}