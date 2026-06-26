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
 * \file fia_tiling_data.h
 * \brief
 */

#ifndef FIA_TILING_DATA_H
#define FIA_TILING_DATA_H

namespace optiling {

constexpr uint32_t NPU_AIC_CORE_NUM = 36;
constexpr uint32_t NPU_AIV_CORE_NUM = 72;

constexpr uint32_t FA_METADATA_SIZE = 8;
constexpr uint32_t COMBINE_METADATA_SIZE = 8;

// 索引数组含义
//  FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0;
constexpr uint32_t FA_BN2_START_INDEX = 1;
constexpr uint32_t FA_M_START_INDEX = 2;
constexpr uint32_t FA_S2_START_INDEX = 3;
constexpr uint32_t FA_BN2_END_INDEX = 4;
constexpr uint32_t FA_M_END_INDEX = 5;
constexpr uint32_t FA_S2_END_INDEX = 6;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7;

// FD Metadata Index Definitions
constexpr uint32_t COMBINE_CORE_ENABLE_INDEX = 0;
constexpr uint32_t COMBINE_BN2_IDX_INDEX = 1;
constexpr uint32_t COMBINE_M_IDX_INDEX = 2;
constexpr uint32_t COMBINE_S2_SPLIT_NUM_INDEX = 3;
constexpr uint32_t COMBINE_WORKSPACE_IDX_INDEX = 4;
constexpr uint32_t COMBINE_M_START_INDEX = 5;
constexpr uint32_t COMBINE_M_NUM_INDEX = 6;

struct FaBaseParams {
    uint32_t bSize = 0;
    uint32_t t1Size = 0;
    uint32_t t2Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    uint32_t dSize = 0;
    uint32_t dSizeV = 0;
    uint32_t actualSeqLengthsQSize = 0;
    uint32_t actualSeqLengthsKVSize = 0;
    float softmaxScale = 0.0f;
    uint32_t coreNum = 0;
    uint32_t outputLayout = 0;
};

struct FaAttnMaskParams {
    uint8_t sparseMode = 0;
    int32_t preTokens = 0;
    int32_t nextTokens = 0;
    uint32_t attnMaskBatch = 0;
    uint32_t attnMaskS1Size = 0;
    uint32_t attnMaskS2Size = 0;
    uint8_t isRowInvalidOpen = 0;
    uint8_t isExistRowInvalid = 0;
};

struct FaWorkspaceParams {
    uint32_t accumOutSize = 0;
    uint32_t logSumExpSize = 0;
};

struct FaMetaData {
    uint32_t FAMetadata[NPU_AIC_CORE_NUM][FA_METADATA_SIZE] = {0};
    uint32_t CombineMetadata[NPU_AIV_CORE_NUM][COMBINE_METADATA_SIZE] = {0};
};

class NoQuantTilingArch35 {
public:
    FaBaseParams faBaseParams;
    FaAttnMaskParams faAttnMaskParams;
    FaWorkspaceParams faWorkspaceParams;
};

class FlashAttnTilingData {
public:
    NoQuantTilingArch35 baseTiling;
    FaMetaData faMetaData;
};
} // namespace optiling
#endif