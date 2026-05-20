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
 * \file comm_cfg.cpp
 * \brief MOE Distribute Dispatch 通信配置实现
 */

#include "comm_cfg.h"
#include "defs.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "../mc2_tiling_utils.h"

namespace moe_dispatch_comm_cfg {

using namespace optiling;

// ========== 内部结构体定义（与框架兼容）==========
struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocal;      // 注意：拼写保持与框架一致
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};

// ========== 版本1：简化接口（dispatch_v2 使用）==========
void SetEpAllToAllCfg(
    AscendC::tiling::Mc2InitTiling& mc2InitTiling,
    AscendC::tiling::Mc2CcTiling& mc2CcTiling,
    const std::string& groupEp,
    bool isLayered) {

    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_TO_ALL;

    // 根据 layered 模式选择算法配置
    std::string algConfigStr;
    if (isLayered) {
        algConfigStr = "AlltoAll=level1:hierarchy";
    } else {
        algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    }

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTiling);
}

// ========== 版本2：完整接口（teardown/setup 使用，支持 URMA）==========
void SetEpAllToAllCfgWithUrma(
    AscendC::tiling::Mc2InitTiling& mc2InitTiling,
    AscendC::tiling::Mc2CcTiling& mc2CcTiling,
    const std::string& groupEp,
    bool isLayered,
    bool isUrma) {

    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_TO_ALL;

    // 根据 layered 模式选择算法配置
    std::string algConfigStr;
    if (isLayered) {
        algConfigStr = "AlltoAll=level1:hierarchy";
    } else {
        algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    }

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTiling);

    // 设置 URMA 标志
    if (isUrma) {
        auto* inner = reinterpret_cast<Mc2CcTilingInner*>(&mc2CcTiling);
        if (inner != nullptr) {
            inner->protocal = static_cast<uint8_t>(moe_dispatch_defs::URMA_COMM);
        }
    }
}

// ========== TP AllGather 配置 ==========
void SetTpAllGatherCfg(
    AscendC::tiling::Mc2InitTiling& mc2InitTiling,
    AscendC::tiling::Mc2CcTiling& mc2CcTiling,
    const std::string& groupTp) {

    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_GATHER;
    std::string algConfigStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupTp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTiling);
}

// ========== CCU 配置 ==========
void SetCcuCommCfg(
    AscendC::tiling::Mc2InitTiling& mc2InitTiling,
    AscendC::tiling::Mc2CcTiling& mc2CcTiling1,
    AscendC::tiling::Mc2CcTiling& mc2CcTiling2,
    const std::string& groupEp,
    const std::string& groupTp) {

    uint32_t opType = static_cast<uint32_t>(mc2tiling::AicpuComType::HCCL_CMD_HALFALLTOALLV);
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::A5_CCU_ENGINE);
    mc2CcTilingConfig.GetTiling(mc2InitTiling);
    mc2CcTilingConfig.GetTiling(mc2CcTiling1);

    mc2CcTilingConfig.SetGroupName(groupTp);
    mc2CcTilingConfig.GetTiling(mc2CcTiling2);
}

// ========== BatchWrite 配置 ==========
// 注：当前未接入，暂用 #if 0 保留
#if 0
void SetBatchWriteCfg(
    gert::TilingContext* ctx,
    const std::string& groupEp,
    uint32_t aivNum,
    uint32_t sdmaUsedStreamPerCore) {

    if (ctx == nullptr) {
        return;
    }

    uint32_t opType = moe_dispatch_defs::OP_TYPE_BATCH_WRITE;
    std::string algConfigStr = "BatchWrite=level0:fullmesh";

    uint32_t reduceType = 0U;
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr, reduceType);
    mc2CcTilingConfig.SetQueueNum(sdmaUsedStreamPerCore);
    mc2CcTilingConfig.SetCommBlockNum(aivNum);

    // TODO: 需指定 GetTilingData<T>() 模板参数
    // auto* tilingData = ctx->GetTilingData<XXX>();
    // if (tilingData != nullptr) {
    //     mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    //     mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    // }
}
#endif

} // namespace moe_dispatch_comm_cfg