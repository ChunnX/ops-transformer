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
 * \file all_to_all_matmul_tiling_data.h
 * \brief 融合算子 tiling——聚合结构体（通信 + 计算切分）+ host 侧构造入口
 *
 * 聚合结构体承自原 kernel/fusions/all_to_all_quant_matmul/ 的
 * all_to_all_mx_quant_matmul_ccu_tiling_data.h；FillAllToAllMatmulTilingData 承自
 * 原 ST main.cpp 的 tiling 构造段（comm 仅 tileM，matmul 由 apace SWAT 引擎按
 * shape 现算）。QuantMatmulTilingData 为 apace 仓内共享件，原位引用不拷入。
 */

#pragma once

#include "ccu_all_to_all.asc"          // CcuCommTilingData（通信契约，本示例通信服务件）
#include "apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_matmul_tiling_data.h" // QuantMatmulTilingData（apace 共享件，原位引用）
#include "apace/tiling/quant_matmul_tiling_swat.h" // host 侧 SWAT tiling 引擎（matmul 切分按 shape 现算）

/*!
 * \file all_to_all_mx_quant_matmul_ccu_tiling_data.h
 * \brief AllToAll MX Quant Matmul（CCU 版）——统一 tiling 聚合（通信 + 计算）
 *
 * 通信侧 CcuCommTilingData 来自 ccu_all_to_all.asc（通信服务契约）；
 * 计算侧 QuantMatmulTilingData 来自 apace 仓内共享件（原位引用）；经 <<<>>> 整体传参。
 */



// 统一 tiling：通信切分 + 计算切分
#pragma pack(push, 8)
struct alignas(8) ccuAllToAllMatmulTilingData {
    CcuCommTilingData comm;
    QuantMatmulTilingData matmul;
};
#pragma pack(pop)

// ---- host 侧 tiling 构造入口（承自原 ST main.cpp 构造段，逐字）----
// comm 仅 tileM（其余由 Validate/PrepareCcuCommResources 填充）；matmul 由 SWAT
// 引擎按 shape 现算（板上验证的配置：stepK/scaleKL1 随 k 自适应——L1 深度搜索把
// A/B K 暂存窗用满，大 K 段相对写死值 −27.8%）
inline void FillAllToAllMatmulTilingData(uint64_t blockRows, uint64_t N, uint64_t K,
                                         uint64_t tileM, ccuAllToAllMatmulTilingData &tilingData)
{
    tilingData.comm.tileM = static_cast<uint32_t>(tileM);
    QuantMatmulTilingSwat<mm::DataType::DT_FLOAT8_E4M3FN, mm::DataType::DT_FLOAT8_E4M3FN,
        mm::BiasDataType::DT_FLOAT> tilingEngine;
    tilingEngine.GetTilingData(blockRows, N, K, tilingData.matmul);
}
