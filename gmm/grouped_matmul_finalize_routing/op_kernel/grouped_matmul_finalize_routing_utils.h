/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing_utils.h
 * \brief
 */

#ifndef __GROUPED_MATMUL_FINALIZE_ROUTING_KERNEL_UTILS_H_
#define __GROUPED_MATMUL_FINALIZE_ROUTING_KERNEL_UTILS_H_

namespace GroupedMatmulFinalizeRouting {
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;

struct MNConfig {
    uint32_t m = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t offsetM = 0;
    uint64_t workSpaceOffset = 0;
};

template<class AT_, class BT_, class CT_, class BiasT_, const auto& MM_CFG = CFG_MDL>
struct MMImplType {
  using AT = AT_;
  using BT = BT_;
  using CT = CT_;
  using BiasT = BiasT_;
  using MT = matmul::MatmulImpl<AT, BT, CT, BiasT, MM_CFG>;
};

struct DataCopy2DDimParams {
    uint32_t dim1;
    uint32_t dim0;
    uint32_t srcDim0;
};

struct MMInitParams {
  GM_ADDR x;
  GM_ADDR weight;
  GM_ADDR bias;
  GM_ADDR group_tokens;
  GM_ADDR scale;          // notice
  GM_ADDR pertoken_scale;
  GM_ADDR offset;
  GM_ADDR logits;
  GM_ADDR token_ranks;
  GM_ADDR residual;
  GM_ADDR y;
  GM_ADDR workspace;
};

struct VectorAtomicParams {
    uint32_t curVecBaseM;
    uint32_t curVecBaseN;
    uint32_t alignBaseN;
    uint32_t offsetM;
    uint64_t mGlobalOffset;
    uint64_t yGmOffset0;
    uint64_t yGmOffset1;
};

} // namespace GroupedMatmulFinalizeRouting
#endif