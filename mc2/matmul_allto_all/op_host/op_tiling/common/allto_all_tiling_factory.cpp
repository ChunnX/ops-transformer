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
 * \file allto_all_tiling_factory.cpp
 * \brief AllToAll Tiling 工厂类实现
 */
#include "allto_all_tiling_factory.h"
#include "allto_all_formulaic_tiling.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"

namespace MC2Tiling {
class MMAlltoAllFitBalanceTiling;
}

namespace MC2Tiling {

CutResult AlltoAllTilingFactory::CreateTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                              TopoType topoType, SocVersion socVersion, NpuArch npuArch)
{
    // 判断是否使用 fit balance tiling（arch35 标卡4P）
    if (mc2tiling::IsStandardCard4P(args.rankDim, npuArch)) {
        OP_LOGD("AlltoAllTilingFactory", "Using fit balance tiling for arch35 standard card 4P");

        // 创建 arch35 的 fit balance tiling 实例
        MMAlltoAllFitBalanceTiling fitBalanceTiling(args, kernelType, topoType, socVersion);
        return fitBalanceTiling.GetTiling();
    }

    // 其他情况使用 formulaic tiling
    OP_LOGD("AlltoAllTilingFactory", "Using formulaic tiling");

    AlltoAllMM formulaicTiling(args, args.rankDim, kernelType, socVersion);
    formulaicTiling.GetTiling();
    return formulaicTiling.tilingM_.cutRes;
}

} // namespace MC2Tiling
