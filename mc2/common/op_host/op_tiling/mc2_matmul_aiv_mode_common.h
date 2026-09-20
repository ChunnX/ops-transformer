/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_MATMUL_AIV_MODE_COMMON_H
#define MC2_MATMUL_AIV_MODE_COMMON_H

#include <cstdint>
#include <map>
#include <vector>

#include "exe_graph/runtime/tiling_context.h"

namespace mc2tiling {

struct MknRange {
    int32_t mStart;
    int32_t mEnd;
    int32_t kStart;
    int32_t kEnd;
    int32_t nStart;
    int32_t nEnd;

    bool Contains(int32_t m, int32_t k, int32_t n) const;
};

using MknConditionMap = std::map<int32_t, std::vector<MknRange>>;

int32_t GetValueFromMknConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                    const MknConditionMap &conditionMap);

struct AivPlatformInfo {
    uint32_t aivNum;
    uint64_t ubSize;
};

AivPlatformInfo GetAivPlatformInfo(const gert::TilingContext *context);

struct AivScaleDtypeResult {
    bool valid;
    bool isInt64;
};

AivScaleDtypeResult CheckAivMatmulScaleDtype(const gert::Tensor *scale, ge::DataType outputType);

ge::graphStatus ValidateCclBufferSizeIfAvailable(const char *group, const char *opName, uint64_t minBufferSize);

} // namespace mc2tiling

#endif // MC2_MATMUL_AIV_MODE_COMMON_H
