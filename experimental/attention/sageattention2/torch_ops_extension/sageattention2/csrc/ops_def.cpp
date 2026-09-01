/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(sageattention2, m)
{
    m.def("sage_attention2(Tensor query, Tensor key, Tensor value, *, str tensor_layout='HND', "
          "bool is_causal=False, str qk_quant_gran='per_thread', float sm_scale=0.0, "
          "str pv_accum_dtype='fp32+fp16', bool smooth_k=True, bool smooth_v=False, "
          "bool return_lse=False) -> (Tensor, Tensor)");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}

