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
 * \file urma_p2p_benchmark.cpp
 * \brief URMA P2P基准测试Kernel入口函数（双模式通信）
 *
 * 支持URMA通信模式(CommMode == TILINGKEY_TPL_URAMA)
 * 双模式设计:
 *   - SQE级(P2pUrmaSqeComm): HcclCombinOpParam上下文(框架自动创建)，无需context参数
 *   - Hcomm(P2pHcommComm): Mc2MoeContext上下文(Host端Mc2Context创建)，需要contextGM参数
 * 模板参数: uint8_t CommMode, uint8_t ArchTag
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "urma_p2p_benchmark_tiling_key.h"
#include "urma_p2p_benchmark.h"

using namespace UrmaP2pBenchmarkImpl;
using namespace UrmaP2pBenchmarkTiling;
using namespace AscendC;

template<uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void urma_p2p_benchmark(
    GM_ADDR context, GM_ADDR x, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR performanceInfo, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    // 注册Tiling数据结构，支持默认配置和A5架构
    REGISTER_TILING_DEFAULT(UrmaP2pBenchmarkTilingData);
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A5", UrmaP2pBenchmarkTilingData);
    TPipe pipe;

    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        // 获取Tiling数据
        GET_TILING_DATA_WITH_STRUCT(UrmaP2pBenchmarkTilingData, tilingData, tilingGM);
        // 只支持URMA通信模式
        if constexpr (CommMode == TILINGKEY_TPL_URAMA) {
            UrmaP2pBenchmark<DTYPE_X, DTYPE_Y> op;
            // context: SQE模式下为nullptr(HcclCombinOpParam框架自动创建), Hcomm模式下为Mc2MoeContext GM地址
            op.Init(x, y, receiveCnt, performanceInfo, workspaceGM, context, &pipe, &tilingData);
            op.Process();
        }
    }
}