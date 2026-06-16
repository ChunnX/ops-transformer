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
 * \file comm_cfg.h
 * \brief MOE Distribute Dispatch 通信配置统一接口
 */

#ifndef MOE_DISPATCH_COMM_CFG_H
#define MOE_DISPATCH_COMM_CFG_H

#include <string>
#include "defs.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace moe_dispatch_comm_cfg {

// ========== EP AllToAll 配置 ==========

// 版本1：简化接口（dispatch_v2 使用，不需要 URMA 参数）
void SetEpAllToAllCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                      const std::string &groupEp, bool isLayered);

// 版本2：完整接口（teardown/setup 使用，支持 URMA）
void SetEpAllToAllCfgWithUrma(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                              const std::string &groupEp, bool isLayered, bool isUrma);

// ========== TP AllGather 配置 ==========
// 仅 dispatch_v2 TP=2 场景
void SetTpAllGatherCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                       const std::string &groupTp);

// ========== CCU 配置 ==========
void SetCcuCommCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling1,
                   AscendC::tiling::Mc2CcTiling &mc2CcTiling2, const std::string &groupEp, const std::string &groupTp);

// ========== BatchWrite 配置 ==========
// 注：当前未接入，声明与实现均暂用 #if 0 保留
#if 0
void SetBatchWriteCfg(
    gert::TilingContext* ctx,
    const std::string& groupEp,
    uint32_t aivNum,
    uint32_t sdmaUsedStreamPerCore);
#endif

} // namespace moe_dispatch_comm_cfg

#endif // MOE_DISPATCH_COMM_CFG_H