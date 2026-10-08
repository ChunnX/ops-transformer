/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GROUPED_ROTATE_QUANT_CHECKER_H
#define GROUPED_ROTATE_QUANT_CHECKER_H

#include "grouped_rotate_quant.h"

namespace grq {
// Validates metadata, then normalizes splitItem, axis and the default round mode.
// Does not read device groupList/alpha values or mutate caller tensor descriptors.
aclnnStatus CheckParams(Params &params);
} // namespace grq
#endif // GROUPED_ROTATE_QUANT_CHECKER_H
