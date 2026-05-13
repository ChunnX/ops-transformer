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

namespace moe_dispatch_comm_cfg {

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

void SetEpAllToAllCfg(
    gert::TilingContext* ctx,
    const std::string& groupEp,
    bool isUrma,
    bool isLayered) {

    if (ctx == nullptr) {
        return;
    }

    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_TO_ALL;

    // 根据 layered 模式选择算法配置
    std::string algConfigStr;
    if (isLayered) {
        algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    } else {
        algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    }

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);

    auto* tilingData = ctx->GetTilingData();
    if (tilingData != nullptr) {
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    }

    // 设置 URMA 标志
    if (isUrma) {
        auto* inner = reinterpret_cast<Mc2CcTilingInner*>(&tilingData->mc2CcTiling);
        if (inner != nullptr) {
            inner->protocal = static_cast<uint8_t>(moe_dispatch_defs::URMA_COMM);
        }
    }
}

void SetTpAllGatherCfg(
    gert::TilingContext* ctx,
    const std::string& groupTp) {

    if (ctx == nullptr) {
        return;
    }

    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_GATHER;
    std::string algConfigStr = "AllGather=level0:fullmesh";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupTp, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::AIV_ENGINE);

    auto* tilingData = ctx->GetTilingData();
    if (tilingData != nullptr) {
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling2);
    }
}

void SetCcuCommCfg(
    gert::TilingContext* ctx,
    const std::string& groupEp) {

    if (ctx == nullptr) {
        return;
    }

    // CCU 模式使用 HalfAllToAllV 算子类型
    uint32_t opType = moe_dispatch_defs::OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigStr);
    // CCU 引擎
    mc2CcTilingConfig.SetCommEngine(moe_dispatch_defs::A5_CCU_ENGINE);

    auto* tilingData = ctx->GetTilingData();
    if (tilingData != nullptr) {
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling1);
    }
}

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

    auto* tilingData = ctx->GetTilingData();
    if (tilingData != nullptr) {
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    }
}

} // namespace moe_dispatch_comm_cfg