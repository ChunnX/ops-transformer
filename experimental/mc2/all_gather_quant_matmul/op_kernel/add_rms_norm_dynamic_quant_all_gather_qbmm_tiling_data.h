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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_TILING_DATA_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"
// #include "quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

struct AddRmsNormDynamicQuantAllGatherTilingData {
    float epsilon = 0;
    float avgFactor = 0;
    uint32_t M = 0;
    uint32_t Ka = 0;
    uint32_t N = 0;
    uint64_t xSize = 0; // 单卡dynamic quant输出数据大小
    uint64_t xNums = 0; // 单卡dynamic quant输出数据数量
    uint64_t scaleSize = 0; // 单卡dynamic quant输出scale大小
    uint64_t mteBlockBytes = 0; // all gather单块搬运大小
    uint64_t axisKaAlignSize = 0;   // x1 K轴32B对齐后大小
    uint64_t axisKaAlignFloatSize = 0;   // dynamic quant输出scale K轴32B对齐后大小
    uint64_t axisKaAlignInt8Size = 0;   // dynamic quant输出数据K轴32B对齐后大小 名称要改成fp820260509 todo
    uint32_t mteTileK = 0;      // all gather K方向切块数目
    uint32_t mteMSplitNum = 0;  // all gather M方向核数划分
    uint32_t mteKSplitNum = 0;  // all gather K方向核数划分
    uint32_t mteMSplitSize = 0; // all gather 单核负责行数向下取整
    uint32_t mteKSplitSize = 0; // all gather 单核负责列数（保证整除）
    uint32_t mteMSplitLargeBlockNum = 0;  // all gather M方向切分后大块的数目
    uint32_t singleCoreM = 0;
    uint32_t singleCoreK = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t rankSize = 0;
    uint32_t sendCoreNumPerRank = 0;    // 负责一个对端的核数
    uint32_t cvStateRowNum = 0;         // CV状态区的行数
    int64_t residualNormMode = 0;
    bool isOptionalOutput = false;
    bool isSmoothScale = false;
};

struct QbmmParams {
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint32_t needUbBuffer;
};

struct AddRmsNormDynamicQuantAllGatherQbmmInfo {
    AddRmsNormDynamicQuantAllGatherTilingData addRmsNormDynamicQuantAllGatherTilingData;
    QbmmParams qbmmParams;
    //DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams matmulTiling; // todo
    TCubeTiling matmulTiling;
};

// tiling struct待完善
struct AddRmsNormDynamicQuantAllGatherQbmmTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AddRmsNormDynamicQuantAllGatherQbmmInfo tilingInfo;
};

#endif // ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_H