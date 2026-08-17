/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册stem_indexer算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
//   - schema 入参顺序：必选张量 + 必选标量在前，'*' 之后为可选张量与带默认值的属性
//   - 张量 dtype / 属性默认值对齐 op_host/stem_indexer_def.cpp
TORCH_LIBRARY(custom, m)
{
    m.def("npu_stem_indexer(Tensor qflat, Tensor kflat, Tensor vbias, Tensor q_seq_lens, "
          "Tensor kv_seq_lens, *, "
          "Tensor? num_prompt_tokens=None, Tensor? metadata=None, "
          "Scalar causal=True, Scalar stem_block_size=128, Scalar stem_stride=16, Scalar alpha=1.0, "
          "Scalar initial_blocks=4, Scalar window_size=4, Scalar k_block_num_rate_medium=0.2, "
          "Scalar k_block_num_bias_medium=30, Scalar k_block_num_rate_large=0.1, "
          "Scalar k_block_num_bias_large=30, Scalar topk_score_precision=1) -> (Tensor, Tensor)");
    m.def("npu_stem_indexer_metadata(Tensor q_seq_lens, Tensor kv_seq_lens, Scalar q_heads, "
          "Scalar kv_heads, *, Scalar causal=True, Scalar stem_block_size=128, Scalar dim_qkflat=128, "
          "Scalar window_size=4) -> Tensor");
}

// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
