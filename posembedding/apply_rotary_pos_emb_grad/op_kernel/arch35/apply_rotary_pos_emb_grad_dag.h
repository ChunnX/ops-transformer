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
 * \file apply_rotary_pos_emb_grad_dag.h
 * \brief
 */

#ifndef __APPLY_ROTARY_POS_EMB_GRAD_DAG_H__
#define __APPLY_ROTARY_POS_EMB_GRAD_DAG_H__

#include "atvoss/util/elems.h"
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/reduce/reduce_operator.h"

namespace ApplyRotaryPosEmbGrad {
using namespace Ops::Base;

// BAB/AB 模板 grad_cos/grad_sin 归约 DAG:
// 输入: workspace 中的部分积 (query*gradQ + key*gradK 的逐元素累加结果, fp32 存储)
// fp32 直接归约(无前置提升 Cast), 跨广播轴(B/N)求和后一次 Cast 回 T 写出
template <typename T>
struct ApplyRotaryPosEmbGradDagFp32In {
    using OpCopyIn = Bind<Vec::CopyIn<float>, Placeholder::In0<float>>;
    using Reduce0 = Bind<Vec::ReduceSumOp<float>, OpCopyIn>;
    using Cast1 = Bind<Vec::Cast<T, float, 1>, Reduce0>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, Cast1>;
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace ApplyRotaryPosEmbGrad

#endif
