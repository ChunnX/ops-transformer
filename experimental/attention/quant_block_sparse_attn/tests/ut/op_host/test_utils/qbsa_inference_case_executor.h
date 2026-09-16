/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_INFERENCE_CASE_EXECUTOR_H
#define QBSA_INFERENCE_CASE_EXECUTOR_H

#include <string>
#include <vector>

#include "../../../../op_host/tiling_common/qbsa_op_contract.h"
#include "infer_shape_case_executor.h"
#include "infer_shape_context_faker.h"
#include "qbsa_host_ut_param.h"

namespace QuantBlockSparseAttnUT {

inline void ExecuteQbsaInferShapeCase(const QuantBlockSparseAttnInferShapeUtParam &param)
{
    using TensorDesc = gert::InfershapeContextPara::TensorDescription;
    using OpAttr = gert::InfershapeContextPara::OpAttr;

    std::vector<TensorDesc> inputTensorDesc({
        param.query,
        param.key,
        param.value,
        param.qDescale,
        param.kDescale,
        param.vDescale,
        param.pScale,
        param.cuSeqlensQ,
        param.cuSeqlensKv,
        param.sequsedQ,
        param.sequsedKv,
        param.sparseIndices,
        param.sparseSeqLen,
        param.blockTable,
        param.attenMask,
        param.metadata,
    });
    std::vector<TensorDesc> outputTensorDesc({
        param.attentionOut,
        param.softmaxLse,
    });
    std::vector<OpAttr> attrs({
        {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
        {"sparse_q_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_q_block_size)},
        {"sparse_kv_block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_kv_block_size)},
        {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
        {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
        {"layout_sparse_indices", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_sparse_indices)},
        {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
        {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_mode)},
        {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
        {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
    });

    gert::InfershapeContextPara para("QuantBlockSparseAttn", inputTensorDesc, outputTensorDesc, attrs);
    ExecuteTestCase(para, param.expectResult, param.expectOutputShape);
}

} // namespace QuantBlockSparseAttnUT

#endif // QBSA_INFERENCE_CASE_EXECUTOR_H
