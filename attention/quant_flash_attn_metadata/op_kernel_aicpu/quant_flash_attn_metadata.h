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
 * \file quant_flash_attn_metadata.h
 * \brief
 */

#ifndef QUANT_FLASH_ATTN_METADATA_H
#define QUANT_FLASH_ATTN_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;
constexpr uint32_t FA_META_SIZE = 1024U;
using FA_METADATA_T = uint32_t;

constexpr uint32_t HEAD_METADATA_SIZE = 16U;
constexpr uint32_t FA_METADATA_SIZE = 16U;
constexpr uint32_t FD_METADATA_SIZE = 16U;
constexpr uint32_t QUANT_FAG_METADATA_SIZE = 4096U;

constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;
constexpr uint32_t HEAD_NEED_INIT_OUTPUT_INDEX = 15U;

constexpr uint32_t FA_BN_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

constexpr uint32_t FD_BN_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

// ---- QuantFlashAttnGrad 反向调度数据 (metadata 第二维) ----
// 标量区 [0, 16)
constexpr uint32_t QUANT_FAG_DETER_MAX_NUM_INDEX = 0U;    // 确定性调度总轮数, 即 kernel 的 loop_max
constexpr uint32_t QUANT_FAG_NEED_INIT_OUTPUT_INDEX = 1U; // dq/dk/dv workspace 是否需要清零
constexpr uint32_t QUANT_FAG_BATCH_SIZE_INDEX = 2U;       // batch 数, kernel 反查 batch 的循环上界
constexpr uint32_t QUANT_FAG_SCHEDULE_VALID_INDEX = 3U;   // swizzle 调度是否安全, 0 表示不支持该 shape
constexpr uint32_t QUANT_FAG_AIC_CORE_NUM_INDEX = 4U;     // 生成调度时使用的 AIC 核数, kernel 需校验一致
// 数组区: roundPrefix 有 b+1 项, s1Outer/s2Outer 各 b 项
constexpr uint32_t QUANT_FAG_ROUND_PREFIX_BASE_INDEX = 16U;
constexpr uint32_t QUANT_FAG_S1_OUTER_BASE_INDEX = 256U;
constexpr uint32_t QUANT_FAG_S2_OUTER_BASE_INDEX = 512U;
// swizzle 调度的 batch 上限, 与 FAG 的 TND_SWIZZLE_PREFIX_NUM 对齐
constexpr uint32_t QUANT_FAG_MAX_BATCH_SIZE = 128U;
// 分核基本块大小, 与 kernel 的 CUBE_BASEM / CUBE_BASEN 保持一致
constexpr uint32_t QUANT_FAG_CUBE_BASE_M = 512U;
constexpr uint32_t QUANT_FAG_CUBE_BASE_N = 512U;

namespace detail {
struct FaMetaData {
    uint32_t sectionNum;
    uint32_t *headMedata;
    uint32_t *faMetadata;
    uint32_t *fdMetadata;
    FaMetaData(void *metadataPtr, uint32_t sectionNum)
        : sectionNum(sectionNum),
          headMedata(static_cast<uint32_t *>(metadataPtr)),
          faMetadata(headMedata + HEAD_METADATA_SIZE),
          fdMetadata(faMetadata + sectionNum * AIC_CORE_NUM * FA_METADATA_SIZE)
    {
        headMedata[0] = sectionNum;
    }

    void SetHeadMedata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_SIZE);
        headMedata[metaIdx] = val;
    }

    uint32_t GetHeadMedata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_SIZE);
        return headMedata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        faMetadata[sectionIdx * AIC_CORE_NUM * FA_METADATA_SIZE + aicIdx * FA_METADATA_SIZE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        return faMetadata[AIC_CORE_NUM * FA_METADATA_SIZE * sectionIdx + FA_METADATA_SIZE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_SIZE);
        fdMetadata[AIV_CORE_NUM * FD_METADATA_SIZE * sectionIdx + FD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_SIZE);
        return fdMetadata[AIV_CORE_NUM * FD_METADATA_SIZE * sectionIdx + FD_METADATA_SIZE * aivIdx + metaIdx];
    }
};

struct QuantFAGMetaData {
    uint32_t *data;
    // metadata shape 为 (2, max_schedule_size)，第一维存正向 FA 调度数据，
    // 第二维存反向 QuantFAG 调度数据，偏移 max_schedule_size 个元素到达第二维起始。
    QuantFAGMetaData(void *metadataPtr)
        : data(static_cast<uint32_t *>(metadataPtr) + QUANT_FAG_METADATA_SIZE)
    {}

    void SetDeterMaxRound(uint32_t metaIdx, int64_t val)
    {
        assert(metaIdx < QUANT_FAG_METADATA_SIZE);
        data[metaIdx] = static_cast<uint32_t>(val);
    }

    int64_t GetDeterMaxRound(uint32_t metaIdx)
    {
        assert(metaIdx < QUANT_FAG_METADATA_SIZE);
        return data[metaIdx];
    }

    void SetMetaData(uint32_t metaIdx, int64_t val)
    {
        assert(metaIdx < QUANT_FAG_METADATA_SIZE);
        data[metaIdx] = static_cast<uint32_t>(val);
    }

    int64_t GetMetaData(uint32_t metaIdx)
    {
        assert(metaIdx < QUANT_FAG_METADATA_SIZE);
        return data[metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
