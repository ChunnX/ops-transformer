/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * [PSEUDOCODE] MOE Dispatch Tiling 重构 - 通信配置统一实现
 * 配套文档：dispatch_analysis_opus_3/architecture_v0.md §5.6
 * 配套伪代码：dispatch_analysis_opus_3/pseudocode_v0.md §1.7
 *
 * 目标：替代原 setup/teardown 通过继承体系（MoeTilingBase → Base → arch 子类）
 * 实现的 SetHcommCfg 逻辑，改为参数化普通函数。
 *
 * arch 差异通过参数传入（isUrma），不依赖继承。
 */

#ifndef MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_COMM_CFG_H
#define MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_COMM_CFG_H

#include "defs.h"
// TODO: include framework headers (TilingContext, graphStatus, hccl::HcclAlgConfig, etc.)

namespace comm_cfg {

using namespace moe_dispatch_defs;

/**
 * 设置 EP AllToAll 通信配置。
 * 覆盖 setup/teardown/v2 的 EP 通信，arch 差异通过 isUrma 区分。
 *
 * @param isUrma true = URMA 通信（A5/arch35），false = SDMA/BatchWrite（A3）
 */
graphStatus SetEpAllToAllCfg(TilingContext* ctx,
                              hccl::HcclAlgConfig& algConfig,
                              bool isUrma);

/**
 * 设置 TP AllGather 通信配置。
 * 仅 v2 TP 场景（tpWorldSize > 1）使用。
 */
graphStatus SetTpAllGatherCfg(TilingContext* ctx);

/**
 * CCU 专用通信配置。
 * 仅 arch35 CCU（dispatch_ccu_tiling.cpp）使用。
 */
graphStatus SetCcuCommCfg(TilingContext* ctx /* TODO: CCU 参数 */);

}  // namespace comm_cfg

#endif  // MC2_COMMON_OP_HOST_OP_TILING_MOE_DISPATCH_COMM_CFG_H
