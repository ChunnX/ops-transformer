/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file upsample_nearest3d_struct.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST3D_STRUCT_H
#define UPSAMPLE_NEAREST3D_STRUCT_H

#include "ascendc/host_api/tiling/template_argument.h"

namespace UpsampleNearest3d {

#define UPSAMPLE_NEAREST3D_TPL_FP16 10
#define UPSAMPLE_NEAREST3D_TPL_FP32 20
#define UPSAMPLE_NEAREST3D_TPL_BF16 30

ASCENDC_TPL_ARGS_DECL(
    UpsampleNearest3d,
    ASCENDC_TPL_DTYPE_DECL(
        D_T_X, UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_BF16, UPSAMPLE_NEAREST3D_TPL_FP32),
    ASCENDC_TPL_DTYPE_DECL(
        D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP16, UPSAMPLE_NEAREST3D_TPL_BF16, UPSAMPLE_NEAREST3D_TPL_FP32), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP16),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP16), ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_BF16),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_BF16), ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(D_T_X, UPSAMPLE_NEAREST3D_TPL_FP32),
        ASCENDC_TPL_DTYPE_SEL(D_T_Y, UPSAMPLE_NEAREST3D_TPL_FP32), ));

struct UpsampleNearest3dTilingData {
    uint8_t dataType;
    int64_t batches;
    int64_t inputShapes[3];
    int64_t outputShapes[3];
    float scaleW;
    float scaleH;
    float scaleD;
    int64_t slideSizeW;
    int64_t tensorSizeW;
    int64_t tensorSizeH;
    int64_t tensorSizeD;
    int64_t slideNumH;
    int64_t slideNumD;
    int64_t eachCoreSlideNum;
    int64_t remainder;
    int64_t tailStartSlideNum;
    int64_t groupCoreNum;
    int64_t inputRow;
    int64_t tailAvergingRow;
    int64_t needCoreNum;
    bool isView1DAndSmallW;
};

} // namespace UpsampleNearest3d
#endif // UPSAMPLE_NEAREST3D_STRUCT_H