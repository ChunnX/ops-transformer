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
 * \brief URMA P2P基准测试Kernel入口函数（SQE级通信）
 *
 * 只支持URMA通信模式(CommMode == TILINGKEY_TPL_URAMA)
 * 使用HcclCombinOpParam上下文（通过GetHcclContext内部获取），无需传入context参数
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
    GM_ADDR x, GM_ADDR y, GM_ADDR receiveCnt,
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
            op.Init(x, y, receiveCnt, performanceInfo, workspaceGM, &pipe, &tilingData);
            op.Process();
        }
    }
}