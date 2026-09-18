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
 * 命名空间: global(2026-09-17 裁定——apace/tiling 与 mmtiling/quant_matmul_tiling 一致)。
 *
 * 用法：
 *   统一入口 GetCoreOccupancyFirstTiling()（2026-09-17 §4.21 调用下沉）: op_tiling 只调
 *   该函数, 内部完成 SWAT matmul tiling → 核数守卫 → 通信切分 → fp4 K 轴修正;
 *   分步接口 GetCommTilingData() 保留供仅需通信切分的调用方使用。
 */

#pragma once

#include <cstdlib>
#include <cstdint>
#include "tiling_data_struct_define.h"
#include "matmul_tiling/quant_matmul_tiling.h"
#include "../utils/apace_constant.h"

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
        // 标定辅助开关: MC2_A2A_V2_CALIB_R1=1 强制单轮切分(R=1, splitAxisTileSize=m),
        // 供通信/计算时延标定使用, 默认关闭。fit-balance 侧对该开关同步旁路(见算子侧
        // fit-balance 头的 IsEnabled), 保证标定期不受 fit-balance 决策影响。
        const char *r1Env = getenv("MC2_A2A_V2_CALIB_R1");
        if ((r1Env != nullptr) && (r1Env[0] == '1')) {
            dataTiling.splitAxisTileSize = m;
            dataTiling.splitAxisTileCnt = 1;
            dataTiling.splitAxisTailSize = 0;
            dataTiling.splitAxisTailCnt = 0;
            dataTiling.nonSplitAxisSize = ka;
            scaleTiling = dataTiling;
            scaleTiling.nonSplitAxisSize = ka / MX_GROUP_SIZE;
            return;
        }

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

/*!
 * \brief 统一入口: SWAT matmul tiling → 核数守卫 → 核利用率优先通信切分 → fp4 K 轴修正
 *
 * §4.21 调用下沉(2026-09-17): 原 a2a_v2 op_tiling 中"先 GetMatmulTiling 再核利用率优先
 * GetCommTiling"的调用序列整体下沉本文件, op_tiling 只调本函数(序列逐行对应原实现)。
 *
 * \param platformInfo  平台信息(context_->GetPlatformInfo())
 * \param m/n/ka        per-rank M / N / K(AllToAll 通信 x1 的 K 维 = per-rank K)
 * \param isFp4         x1 为 MXFP4 打包存储(2 fp4/字节): SWAT 按 fp4 推导 baseK 与 L1 布局,
 *                      数据通信 K 轴字节数减半(scale 为 e8m0, 32 元素/字节, 不受影响)
 * \param hasBias       是否带 bias 输入
 * \param[out] mmTiling     matmul tiling 结果(含 usedCoreNum, 调用方校验/SetBlockDim 用)
 * \param[out] dataTiling   数据通道通信切分
 * \param[out] scaleTiling  Scale 通道通信切分
 *
 * 纯计算序列, 不含校验与日志——核数守卫(usedCoreNum>=rankSize 防 kernel 死锁、
 * min(aic,aiv)>=rankSize)与错误日志由调用方 op_tiling 负责(2026-09-17 裁定)。
 */
inline void GetCoreOccupancyFirstTiling(fe::PlatFormInfos *platformInfo, uint64_t m, uint64_t n, uint64_t ka,
                                        bool isFp4, bool hasBias, QuantMatmulTilingData &mmTiling,
                                        CommTilingData &dataTiling, CommTilingData &scaleTiling)
{
    // ---- matmul tiling(SWAT): MXFP4 按 fp4 推导 baseK(256 vs 128)与 L1 布局, 其余按 fp8 ----
    if (isFp4) {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT4_E2M1, mm::DataType::DT_FLOAT4_E2M1, mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(platformInfo);
        tilingEngine.SetBiasInfo(hasBias);
        tilingEngine.EnableBaseMHalving(true);
        tilingEngine.GetTilingData(m, n, ka, false, true, mmTiling);
    } else {
        QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
                              mm::BiasDataType::DT_FLOAT>
            tilingEngine;
        tilingEngine.SetPlatformInfoPtr(platformInfo);
        tilingEngine.SetBiasInfo(hasBias);
        tilingEngine.EnableBaseMHalving(true);
        tilingEngine.GetTilingData(m, n, ka, false, true, mmTiling);
    }

    // ---- 通信切分: 核利用率优先(历史: aiv_urma 路径的 fit-balance tiling 已废弃, 2026-09-15) ----
    CoreOccupancyFirstTiling::GetCommTilingData(m, n, ka, mmTiling, dataTiling, scaleTiling);

    // ---- MXFP4 打包存储(2 fp4/字节): 数据通信的 K 轴字节数需减半 ----
    if (isFp4) {
        dataTiling.nonSplitAxisSize = ka / 2;
    }
}
