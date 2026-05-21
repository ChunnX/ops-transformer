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
 * \file allto_all_matmul_tiling_data_arch35.h
 * \brief 定义tiling_data
 */
#ifndef ALLTO_ALL_MATMUL_TILING_DATA_H
#define ALLTO_ALL_MATMUL_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../common/op_kernel/mc2_tiling_struct.h"
#include "tensor_matmul_tiling_data.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

struct AlltoAllMatmulTilingInfo {
    uint32_t rankDim;
    uint32_t tileM;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint32_t biasLen;
    uint32_t rankM;
    uint32_t rankN;
    uint32_t rankK;
    uint32_t rankKScaleGrp;
    uint32_t aicCoreNum;
    uint64_t commLen;
    uint64_t permuteLen;
    uint64_t hcclDataType;
    uint64_t x1ScaleOptionalLen;
    uint64_t dynamicExtraSpace;
    uint64_t x1QuantDtype;
};

struct AlltoAllMatmulTilingData {
    Mc2InitTiling mc2InitTiling; // 初始化通信任务配置
    Mc2CcTiling mc2CcTiling;     // 具体每个通信任务的参数配置
    AlltoAllMatmulTilingInfo alltoAllMatmulTilingInfo;
    TensorMatmulTilingData tensorMatmulTileTilingData; // 通算切分头块matmul tiling数据
    TensorMatmulTilingData tensorMatmulTailTilingData; // 通算切分尾块matmul tiling数据
};

// MX量化tiling结构体
struct AlltoAllQuantMatmulTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AlltoAllMatmulTilingInfo alltoAllQuantMatmulTilingInfo;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams mc2QuantMmTileTilingData;
    DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams mc2QuantMmTailTilingData;
    MxTensorMatmulTilingData mxTensorTileTilingData;
    MxTensorMatmulTilingData mxTensorTailTilingData;
};

#endif // ALLTO_ALL_MATMUL_TILING_H
