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
 * \file block_balance_def.h
 * \brief BlockBalance结构体定义
 */

#ifndef BLOCK_BALANCE_DEF_H
#define BLOCK_BALANCE_DEF_H

#include <cstdint>
#include <vector>

namespace load_balance {

#define BLOCK_BALANCE_SUCCESS                     0
#define BLOCK_BALANCE_ERROR_INVALID_PARAM         1

struct BlockBalanceParam {
    uint32_t batchSize { 0U };
    uint32_t kvHeadNum { 0U };
    uint32_t aicCoreNum { 0U };
    uint32_t qBlockMax { 0U };
    const int32_t *sparseSeqLen { nullptr };
};

struct BlockBalanceResult {
    uint32_t usedCoreNum { 0U };
    uint64_t maxBlock { 0U };
    std::vector<uint32_t> bN2End {};
    std::vector<uint32_t> s1End {};
    std::vector<uint32_t> s2End {};
    std::vector<uint32_t> firstFdDataWorkspaceIdx {};

    explicit BlockBalanceResult(uint32_t aicNum)
        : bN2End(aicNum),
          s1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum) {}
};

} // namespace load_balance

#endif
