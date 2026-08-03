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
 * \file moe_v3_counting_sort_cut_origin_static_quant.h
 * \brief A5 CountingSort CutOrigin 静态量子类（占位，主场景由非量化承接）。
 */
#ifndef MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H
#define MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H

#include "moe_v3_counting_sort_cut_origin_base.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;

template <typename T>
class MoeV3CountingSortCutOriginStaticQuant : public MoeV3CountingSortCutOriginBase<T> {
public:
    __aicore__ inline MoeV3CountingSortCutOriginStaticQuant(){};
};

} // namespace MoeInitRoutingV3
#endif // MOE_V3_COUNTING_SORT_CUT_ORIGIN_STATIC_QUANT_H
