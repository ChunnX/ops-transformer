/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_v3_counting_sort_full_load_unquantized.h
 * \brief A5 CountingSort FullLoad 非量化子类。
 *        expandedX 直接搬运 x（无量化），expandedScale 直接搬运 scale。
 */
#ifndef MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
#define MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H

#include "moe_v3_counting_sort_full_load_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortFullLoadUnquantized : public MoeV3CountingSortFullLoadBase<T> {
public:
    __aicore__ inline MoeV3CountingSortFullLoadUnquantized(){};
};

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_FULL_LOAD_UNQUANTIZED_H
