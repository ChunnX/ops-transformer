/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file core_occupancy_first_tiling.h
 * \brief 核利用率优先的通信切分策略：根据 matmul tiling 结果推导 AllToAll/AllGather 通信切分方案
 *
 * 这不是基类，而是一个独立的静态切分策略：以核利用率为第一优先级，通信切分直接跟随
 * matmul 的最优分核结果——每轮通信的 M 跨度等于一个计算波
 * （headMSize = ceil(usedCoreNum / nTile) * baseM），不引入任何时延模型。
 * 当通算耗时不均衡时的进一步配平，见 fit_balance_tiling.h 的 fit-balance 策略。
 *
 * 用法：
 *   1. 先由 QuantMatmulTilingSwat 计算 matmul tiling
 *   2. 将 mm tiling 的 baseM/baseN/usedCoreNum + 算子维度 (m, n, ka) 传入
 *   3. 得到 data 和 scale 两组 CommTilingData
 */

#pragma once

#include <cstdint>
#include "tiling_data_struct_define.h"
#include "matmul_tiling/quant_matmul_tiling_data.h"
#include "../utils/apace_constant.h"

namespace apace {

class CoreOccupancyFirstTiling {
public:
    /*!
     * \brief 计算数据通道的通信 tiling
     *
     * \param m         M_per_rank（单卡输出 M）
     * \param n         N
     * \param ka        per-rank K
     * \param mmTiling  已完成 matmul tiling 的结果（需填充 baseM/baseN/usedCoreNum）
     * \param dataTiling   [out] 数据通信切分
     * \param scaleTiling  [out] Scale 通信切分（nonSplitAxisSize = ka / 32）
     */
    static void GetCommTilingData(uint64_t m, uint64_t n, uint64_t ka, const QuantMatmulTilingData &mmTiling,
                                  CommTilingData &dataTiling, CommTilingData &scaleTiling)
    {
        uint64_t baseM = mmTiling.baseM;
        uint64_t baseN = mmTiling.baseN;
        uint64_t usedCoreNum = mmTiling.usedCoreNum;

        // 防御性零值守卫：baseM/baseN/usedCoreNum 为 0 说明 matmul tiling 异常，
        // 此时无法推导通信切分，直接返回零值 tiling 避免除零。
        if (baseM == 0UL || baseN == 0UL || usedCoreNum == 0UL) {
            dataTiling = {};
            scaleTiling = {};
            return;
        }

        auto nTile = (n + baseN - 1) / baseN;
        uint64_t headMSize = ((usedCoreNum + nTile - 1) / nTile) * baseM;
        if (headMSize == 0UL) {
            dataTiling = {};
            scaleTiling = {};
            return;
        }
        uint64_t headTileCnt = m / headMSize;
        uint64_t tailMSize = m % headMSize;
        uint64_t tailTileCnt = (tailMSize > 0) ? 1 : 0;

        dataTiling.splitAxisTileSize = headMSize;
        dataTiling.splitAxisTileCnt = headTileCnt;
        dataTiling.splitAxisTailSize = tailMSize;
        dataTiling.splitAxisTailCnt = tailTileCnt;
        dataTiling.nonSplitAxisSize = ka;

        scaleTiling = dataTiling;
        scaleTiling.nonSplitAxisSize = ka / MX_GROUP_SIZE;
    }
};

} // namespace apace
