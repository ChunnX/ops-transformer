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
 * \file rotary_position_embedding3d_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __ROTARY_POSITION_EMBEDDING3D_TILLING_DATA_H__
#define __ROTARY_POSITION_EMBEDDING3D_TILLING_DATA_H__
 
struct RotaryPositionEmbedding3dTilingData {
    int64_t totalLength;   // 数据总量 B*S*D
    int64_t lastDim;       // 最后一维大小 D（用于查找元素对）
    int64_t tileNum;       // tile 数量
};
#endif
