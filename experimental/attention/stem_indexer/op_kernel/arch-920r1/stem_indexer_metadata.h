/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef STEM_INDEXER_ARCH_920R1_METADATA_H
#define STEM_INDEXER_ARCH_920R1_METADATA_H

#include <cstdint>

namespace optiling {

constexpr uint32_t AIC_CORE_NUM = 64U;
using SLI_METADATA_T = uint32_t;

constexpr uint32_t SLI_PER_CORE_STRIDE = 16U;
constexpr uint32_t SLI_METADATA_HEADER_OFFSET = SLI_PER_CORE_STRIDE * sizeof(uint32_t);

constexpr uint32_t SLI_SEC_BN2_START_INDEX = 0U;
constexpr uint32_t SLI_SEC_M_START_INDEX = 1U;
constexpr uint32_t SLI_SEC_S2_START_INDEX = 2U;
constexpr uint32_t SLI_SEC_BN2_END_INDEX = 3U;
constexpr uint32_t SLI_SEC_M_END_INDEX = 4U;
constexpr uint32_t SLI_SEC_S2_END_INDEX = 5U;

#ifdef __CCE_AICORE__

__aicore__ inline uint32_t GetFASectionMetaIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
{
    return SLI_PER_CORE_STRIDE * AIC_CORE_NUM * sectionIdx + SLI_PER_CORE_STRIDE * coreIdx + metaIdx;
}

#endif

} // namespace optiling

#endif // STEM_INDEXER_ARCH_920R1_METADATA_H
