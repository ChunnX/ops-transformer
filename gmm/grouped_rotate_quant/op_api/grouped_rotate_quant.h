/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GROUPED_ROTATE_QUANT_H
#define GROUPED_ROTATE_QUANT_H

#include <cstddef>
#include <cstdint>
#include <tuple>
#include "opdev/op_executor.h"

namespace grq {
constexpr int64_t NO_GROUP = -1;
constexpr int64_t GROUP_M = 0;
constexpr size_t MAX_GROUP_NUM = 1024;
constexpr int64_t MX_SCALE_BLOCK = 64;

struct Params {
    const aclTensorList *x;
    const aclTensorList *rotation;
    const aclTensor *alpha;
    const aclTensor *groupList;
    int64_t splitItem;
    int64_t groupType;
    int64_t groupListType;
    int64_t axis;
    const char *roundMode;
    int64_t scaleAlg;
    double dstTypeMax;
    bool trans;
    const aclIntArray *tuningConfig;
    const aclTensorList *y;
    const aclTensorList *scale;
};
} // namespace grq

namespace l0op {
using GroupedRotateQuantResult = std::tuple<const aclTensorList *, const aclTensorList *>;
GroupedRotateQuantResult GroupedRotateQuant(const grq::Params &params, aclOpExecutor *executor);
} // namespace l0op
#endif // GROUPED_ROTATE_QUANT_H
