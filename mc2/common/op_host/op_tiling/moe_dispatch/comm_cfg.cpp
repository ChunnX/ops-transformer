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
 * \file comm_cfg.cpp
 * \brief MOE Distribute Dispatch 通信配置实现
 *
 * 注：Mc2CcTilingInner 镜像结构体 + reinterpret_cast 设置 protocal 字段，
 *     是因为框架 Mc2CcTilingConfig 未公开 protocal setter，受限于框架 API。
 */

#include "comm_cfg.h"
#include "defs.h"
#include "tiling/tiling_api.h"
#include "../mc2_tiling_utils.h"

namespace moe_dispatch_comm_cfg {

namespace {
// 镜像 Mc2CcTiling 内部布局（与各算子已有定义保持一致），用于设置 protocal 字段
struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocal; // 拼写与框架一致：0 = UB-MEM, 1 = URMA
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};

// 选择 AllToAll 的 algConfig 字符串
inline std::string PickAllToAllAlgConfig(bool isLayered)
{
    return isLayered ? "AlltoAll=level1:hierarchy" : "AlltoAll=level0:fullmesh;level1:pairwise";
}
} // namespace

// ========== EP AllToAll 配置 ==========
void SetEpAllToAllCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                      const std::string &groupEp, bool isLayered, bool isUrma)
{
    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = PickAllToAllAlgConfig(isLayered);

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTiling);

    if (isUrma) {
        // 框架未公开 protocal setter，按既有算子实现保持一致
        reinterpret_cast<Mc2CcTilingInner *>(&mc2CcTiling)->protocal =
            static_cast<uint8_t>(moe_dispatch_defs::URMA_COMM);
    }
}

// ========== TP AllGather 配置 ==========
void SetTpAllGatherCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                       const std::string &groupTp)
{
    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_GATHER;
    std::string algConfigStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupTp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTiling);
}

// ========== CCU 配置 ==========
void SetCcuCommCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTilingEp,
                   AscendC::tiling::Mc2CcTiling &mc2CcTilingTp, const std::string &groupEp,
                   const std::string &groupTp)
{
    uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV);
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTilingEp);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.GetTiling(mc2CcTilingTp);
}

} // namespace moe_dispatch_comm_cfg
