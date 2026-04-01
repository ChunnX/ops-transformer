/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

namespace custom {
TORCH_LIBRARY(custom, m)
{
    m.def(R"(npu_fused_infer_attention_score(Tensor query, Tensor key, Tensor value, *, 
                                            Tensor? query_rope=None, Tensor? key_rope=None, 
                                            Tensor? pse_shift=None, Tensor? atten_mask=None, 
                                            Tensor? actual_seq_qlen=None, Tensor? actual_seq_kvlen=None, 
                                            Tensor? block_table=None, Tensor? dequant_scale_query=None, 
                                            Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, 
                                            Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, 
                                            Tensor? dequant_scale_key_rope=None, Tensor? quant_scale_out=None, 
                                            Tensor? quant_offset_out=None, Tensor? learnable_sink=None, 
                                            Tensor? metadata=None,
                                            int num_query_heads=1, int num_key_value_heads=0, 
                                            float softmax_scale=1.0, int pre_tokens=2147483647, 
                                            int next_tokens=2147483647, str input_layout='BSH', 
                                            int sparse_mode=0, int block_size=0, 
                                            int query_quant_mode=0, int key_quant_mode=0, 
                                            int value_quant_mode=0, int inner_precise=1, 
                                            bool return_softmax_lse=False, int? query_dtype=None, 
                                            int? key_dtype=None, int? value_dtype=None, 
                                            int? query_rope_dtype=None, int? key_rope_dtype=None, 
                                            int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, 
                                            int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, 
                                            int? dequant_scale_value_dtype=None, 
                                            int? dequant_scale_key_rope_dtype=None) -> (Tensor, Tensor))"); 

    m.def(R"(npu_fused_infer_attention_score_metadata(int batch_size,
                                                    int query_seq_size,
                                                    int query_head_num,
                                                    int key_head_num,
                                                    int head_dim,
                                                    int block_size,
                                                    int max_block_num_per_batch,
                                                    Tensor actual_seq_lengths_kv=None,
                                                    *,
                                                    str layout_query='BSND') -> Tensor)");
}
    // 通过pybind将c++接口和python接口绑定
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
} // namespace custom


