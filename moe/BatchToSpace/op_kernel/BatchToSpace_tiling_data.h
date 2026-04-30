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
 * \file BatchToSpace_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __BATCHTOSPACE_TILLING_DATA_H__
#define __BATCHTOSPACE_TILLING_DATA_H__
 
struct BatchtospaceTilingData {
    int64_t inShape[4];
    int64_t outShape[4];
    int64_t blockSize;
    int64_t crops[2][2];
    uint64_t totalCount;
    uint64_t perCoreCount;
    uint8_t ubAxis;
    uint32_t ubFactor;
    uint32_t bufferSize;
};
#endif