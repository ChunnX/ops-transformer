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
 * \file urma_p2p_benchmark_tiling.h
 * \brief URMA P2P基准测试Tiling数据结构定义
 */
#ifndef URMA_P2P_BENCHMARK_TILING_H
#define URMA_P2P_BENCHMARK_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct UrmaP2pBenchmarkTilingData {
    Mc2InitTiling mc2InitTiling;       // MC2框架初始化数据（通信域配置）
    Mc2CcTiling mc2CcTiling;           // MC2通信引擎配置（含protocal字段）
    int32_t data_size;       // 数据大小（元素数，即hidden size维度）
    int32_t world_size;      // 通信域rank总数
    int32_t max_bs;          // 最大batch size
    int32_t mode;            // 测试模式: 0=overhead+latency, 1=bandwidth
    int32_t aiv_num;         // AIV核总数
    int32_t dst_rank_id;     // P2P通信的目标rank ID（-1表示使用dst_rank_id输入tensor）
    int32_t timing_slots;    // timing slot数量
    uint64_t total_ub_size;  // 总UB大小
    uint64_t total_win_size; // 通信窗口总大小
    int32_t comm_mode;       // 通信模式: 0=SQE级(P2pUrmaSqeComm), 1=Hcomm(WriteNbi+Drain)
};

#endif // URMA_P2P_BENCHMARK_TILING_H