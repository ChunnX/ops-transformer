/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_tiling_data.h
 * \brief StemOamPrepPagedKV TilingData struct definition (arch35)
 */
#ifndef STEM_OAM_PREP_PAGED_KV_TILING_DATA_H
#define STEM_OAM_PREP_PAGED_KV_TILING_DATA_H

#include "kernel_operator.h"

constexpr uint32_t DIM_QK = 128;
constexpr uint32_t MAX_KV_PADDED = 262144;
constexpr uint32_t KV_INDICES_BATCH_SIZE = 64;
constexpr uint32_t MAX_STEM_BLOCK_SIZE = 256;
constexpr uint32_t MAX_STEM_STRIDE = 64;
constexpr uint32_t MAX_R_VAL = MAX_STEM_BLOCK_SIZE / 16;
constexpr uint32_t MAX_KFLAT_DIM = MAX_STEM_STRIDE * DIM_QK;
constexpr uint32_t DENSE_ROW_ELEM = DIM_QK;
constexpr uint32_t REDUCE_TMP_SIZE = 9216;

struct StemPrepKVTilingData {
    uint32_t batchSize = 0;
    uint32_t numKvHeads = 0;
    uint32_t totalBh = 0;
    uint32_t usedCoreNum = 0;
    uint32_t totalBlocks = 0;
    uint32_t kvBlockSize = 64;
    uint32_t dimQk = DIM_QK;
    uint32_t maxKvBlocks = 0;
    uint32_t stemBlockSize = 128;
    uint32_t stemStride = 16;
    float lambdaMag = 0.3f;
    uint32_t kvLayout = 0;
    uint64_t vFlatOutGM = 0;
};

#endif // STEM_OAM_PREP_PAGED_KV_TILING_DATA_H
