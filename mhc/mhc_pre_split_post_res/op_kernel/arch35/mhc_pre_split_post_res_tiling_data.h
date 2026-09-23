/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_tiling_data.h
 * \brief ascend950（arch35）kernel 侧 tiling data 裸结构定义
 *
 * ascend950 的 APT 编译链路不为 op_host 侧 BEGIN_TILING_DATA_DEF 宏生成的
 * MhcPreSplitPostResTilingData 生成 kernel 侧定义，需在 kernel 源码中显式定义。
 * 字段顺序/类型与 op_host/mhc_pre_split_post_res_tiling.h 中的宏版本保持一致。
 */

#ifndef MHC_PRE_SPLIT_POST_RES_TILING_DATA_H_
#define MHC_PRE_SPLIT_POST_RES_TILING_DATA_H_

// 本头文件仅被 op_kernel（arch35）侧包含，保持自足：仅依赖 C 标准类型
#include <stdint.h>

namespace MhcPreSplitPostRes {

struct MhcPreSplitPostResTilingData {
    int64_t totalLength;     // B*S 或 T，合轴后的token总数
    int64_t n;               // 超连接维度 (固定4)
    int64_t d;               // Dim维度 (白名单 2560/5120/6144/7168)
    int64_t phiRows;         // N^2+N，phi的第一维
    int64_t dimTile;         // 核内D的分块大小，min(D, 2560)
    int64_t dimLoop;         // D需要几个tile
    int64_t coresPerToken;   // 每token使用的核数（1或2）
    int64_t phiRowMid;       // 单核处理的phi行起始偏移（双核模式用）
    int64_t blockDim;        // 实际启动的总核数
    int64_t baseT;           // 每（组）核处理的token数（前coreTAct-1组）
    int64_t tailT;           // 最后一（组）核处理的token数
    int64_t coreTAct;        // T轴实际分组数
    float normEps;           // RmsNorm epsilon
    int64_t reduceWorkSize;  // ReduceSum work tensor大小（float元素数）
    float invK;              // 1.0f/(N*D)，RmsNorm均值系数
};

}  // namespace MhcPreSplitPostRes

#endif  // MHC_PRE_SPLIT_POST_RES_TILING_DATA_H_
