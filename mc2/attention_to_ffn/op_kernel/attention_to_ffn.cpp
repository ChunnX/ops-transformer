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
 * \file attention_to_ffn.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "attention_to_ffn.h"
#include "attention_to_ffn_tiling.h"
#include "attention_to_ffn_tiling_key.h"

// 引入AscendC算子编程框架命名空间
using namespace AscendC;
// 引入AttentionToFFN算子实现命名空间
using namespace AttentionToFFNImpl;
// 引入MC2 Tiling命名空间
using namespace Mc2Tiling;

/**
 * @brief AttentionToFFN算子入口函数
 * @tparam isQuant       是否启用量化模式
 * @tparam isSync        是否启用同步模式
 * @tparam isActiveMask  是否启用活跃掩码
 * @tparam ArchTag       架构标签，用于区分不同芯片架构
 * @param x               输入张量
 * @param sessionId       会话ID
 * @param microBatchId    微批次ID
 * @param layerId         层ID
 * @param expertIds       专家ID列表
 * @param expertRankTable 专家排列表
 * @param scales          量化缩放因子
 * @param active_mask     活跃掩码
 * @param workspaceGM     工作区内存地址
 * @param tilingGM        Tiling数据内存地址
 */
template<bool isQuant, bool isSync, bool isActiveMask, uint8_t ArchTag>
__global__ __aicore__ void attention_to_ffn(GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId,
    GM_ADDR layerId, GM_ADDR expertIds, GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 注册默认Tiling数据结构
    REGISTER_TILING_DEFAULT(AttentionToFFNTilingData);
    // 针对A3架构注册特定Tiling Key
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A3", AttentionToFFNTilingData);
    // 声明流水线对象，用于任务间同步
    TPipe pipe;
    
    // A3架构分支：获取Tiling数据，初始化并执行算子
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        // 从全局内存中获取Tiling结构化数据
        GET_TILING_DATA_WITH_STRUCT(AttentionToFFNTilingData, tilingData, tilingGM);
        // 实例化AttentionToFFN算子对象
        AttentionToFFN<DTYPE_X, isQuant, isSync, isActiveMask> op;
        // 初始化算子，传入输入输出及流水线、Tiling数据
        op.Init(x, sessionId, microBatchId, layerId, expertIds, expertRankTable, scales, active_mask,
                workspaceGM, &pipe, &tilingData);
        // 执行算子计算流程
        op.Process();
    }
}