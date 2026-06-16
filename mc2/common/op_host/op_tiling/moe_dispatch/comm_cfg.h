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
 *
 * 设计原则（architecture_v0.md §5.6）：
 *   提供 SetEpAllToAllCfg / SetTpAllGatherCfg / SetCcuCommCfg 三组单一接口。
 *   公共层不依赖具体算子 tilingData 类型，调用方传入 mc2InitTiling/mc2CcTiling 引用。
 *   isUrma 参数控制 protocal 字段（受限于框架 API：通过镜像结构体设置）。
 */

#ifndef MOE_DISPATCH_COMM_CFG_H
#define MOE_DISPATCH_COMM_CFG_H

#include <string>
#include "defs.h"
#include "tiling/tiling_api.h"

namespace moe_dispatch_comm_cfg {

// ========== EP AllToAll 配置 ==========
// isLayered : 是否使用 hierarchy 算法（true 用 level1:hierarchy；false 用 level0:fullmesh;level1:pairwise）
// isUrma    : 是否启用 URMA 协议（仅 setup/teardown 部分场景需要；v2 默认 false）
void SetEpAllToAllCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                      const std::string &groupEp, bool isLayered, bool isUrma = false);

// ========== TP AllGather 配置 ==========
// 仅 dispatch_v2 TP=2 场景使用
void SetTpAllGatherCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTiling,
                       const std::string &groupTp);

// ========== CCU 配置 ==========
// CCU 场景：同一 Mc2CcTilingConfig 分别 GetTiling 到两个 mc2CcTiling（EP/TP）
void SetCcuCommCfg(AscendC::tiling::Mc2InitTiling &mc2InitTiling, AscendC::tiling::Mc2CcTiling &mc2CcTilingEp,
                   AscendC::tiling::Mc2CcTiling &mc2CcTilingTp, const std::string &groupEp,
                   const std::string &groupTp);

} // namespace moe_dispatch_comm_cfg

#endif // MOE_DISPATCH_COMM_CFG_H
