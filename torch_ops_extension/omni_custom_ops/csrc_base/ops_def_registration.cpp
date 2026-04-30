/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch/library.h>

// 在custom命名空间里注册后续的XXX算子，每次新增自定义aten ir都需先增加定义
// step1, 为新增自定义算子添加定义
TORCH_LIBRARY_FRAGMENT(custom, m) {
    m.def("npu_fused_infer_attention_sink(Tensor query, Tensor key, Tensor value, *, Tensor? query_rope=None,"
        "Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, Tensor? actual_seq_qlen=None,"
        "Tensor? actual_seq_kvlen=None, Tensor? block_table=None, Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None,"
        "Tensor? dequant_offset_key=None, Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None,"
        "Tensor? key_sink=None, Tensor? key_rope_sink=None, Tensor? value_sink=None,"
        "Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? meta_data=None, int num_query_heads=1, int num_key_value_heads=0,"
        "float softmax_scale=1.0, int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout='TND', int sparse_mode=0, int block_size=0,"
        "int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, int inner_precise=0, bool return_softmax_lse=False, int sink_number=0,"
        "int? query_dtype=None, int? key_dtype=None, int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None,"
        "int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, int? dequant_scale_query_dtype=None,"
        "int? dequant_scale_key_dtype=None, int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None) -> (Tensor, Tensor)");

    m.def("_npu_fused_infer_attention_sink_metadata(int num_heads_q, int num_heads_kv, int head_dim_qk, int head_dim_v, *, Tensor? actual_seq_lengths=None,"
        "Tensor? actual_seq_lengths_kv=None, int batch_size=0, int sparse_mode=0, int pre_tokens=2147483647, int next_tokens=2147483647,"
        "str input_layout='TND', str input_layout_kv='TND', int sink_num=0, int k_sink_num=0, int rope_head_dim=0, int block_size=0, int aic_core_num=24, int aiv_core_num=48) -> Tensor");

    m.def("npu_ai_infra_causal_conv1d_fn_add(Tensor x, Tensor weight, Tensor conv_states, *, Tensor? bias=None, Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? initial_state_mode=None, str? activation=None, int? residual_connection=1, int? pad_slot_id=-1) -> (Tensor)");
    m.def("npu_ai_infra_causal_conv1d_update_add(Tensor x, Tensor weight, Tensor conv_states, *, Tensor? bias=None, Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? num_accepted_tokens=None, str? activation=None, int? residual_connection=1, int? pad_slot_id=-1) -> (Tensor)");
    m.def("npu_ai_infra_fused_causal_conv1d(Tensor x, Tensor weight, Tensor conv_states, *, Tensor? query_start_loc=None, Tensor? cache_indices=None, Tensor? initial_state_mode=None, Tensor? bias=None, Tensor? num_accepted_tokens=None, Tensor? num_computed_tokens=None, Tensor? block_idx_first_scheduled_token=None, Tensor? block_idx_last_scheduled_token=None, Tensor? initial_state_idx=None, str? activation=None, int pad_slot_id=-1, int run_mode=0, int? max_query_len=-1, int residual_connection=1, int block_size=128, int conv_mode=1, bool inplace=False) -> Tensor");
    m.def("npu_chunk_gated_delta_rule_recurrence(Tensor initial_state, Tensor kgexp, Tensor value, "
        "Tensor k_cumdecay, Tensor qgexp, Tensor gexp, Tensor actual_seqlens) ->(Tensor, Tensor)");
    m.def("npu_lower_triangular_inverse(Tensor x) -> Tensor");
    m.def("npu_sparse_flash_attention_gqa(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
    "SymInt[]? actual_seq_lengths_query=None, SymInt[]? actual_seq_lengths_kv=None,"
    "Tensor? block_table=None, int num_query_heads=1, int num_key_value_heads=0,"
    "str input_layout='BSH', str input_layout_kv='BSND', int sparse_mode=0, int block_size=0, bool return_softmax_lse=False) -> (Tensor, Tensor)");
    m.def("npu_sparse_flash_attention_gqa.out(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
    "SymInt[]? actual_seq_lengths_query=None, SymInt[]? actual_seq_lengths_kv=None,"
    "Tensor? block_table=None, int num_query_heads=1, int num_key_value_heads=0,"
    "str input_layout='BSH', str input_layout_kv='BSND', int sparse_mode=0, int block_size=0, bool return_softmax_lse=False, Tensor? workspace=None, Tensor[] out) -> (Tensor, Tensor)");
    m.def("npu_sparse_flash_attention_gqa.tensor(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
    "Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None,"
    "Tensor? block_table=None, int num_query_heads=1, int num_key_value_heads=0,"
    "str input_layout='BSH', str input_layout_kv='BSND', int sparse_mode=0, int block_size=0, bool return_softmax_lse=False) -> (Tensor, Tensor)");
    m.def("npu_sparse_flash_attention_gqa_get_max_workspace(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, *,"
    "SymInt[]? actual_seq_lengths_query=None, SymInt[]? actual_seq_lengths_kv=None,"
    "Tensor? block_table=None, int num_query_heads=1, int num_key_value_heads=0,"
    "str input_layout='BSH', str input_layout_kv='BSND', int sparse_mode=0, int block_size=0, bool return_softmax_lse=False) -> Tensor");
    m.def("npu_sparse_flash_attention_gqa_infer_output(Tensor query, Tensor value, *,"
    "Tensor? block_table=None, int num_query_heads=1, int num_key_value_heads=0,"
    "str input_layout='BSH', bool return_softmax_lse=False) -> (Tensor, Tensor)");
    m.def("npu_esa_select_topk(Tensor query, Tensor key, int blk_size, int init_blk_num, int local_blk_num, int topk,"
    "str input_layout, *, SymInt[]? actual_seq_q_len_optional=None, SymInt[]? actual_seq_k_len_optional=None,"
    "SymInt[]? actual_cmp_seq_k_len_optional=None, int compress_blk_size=16) -> Tensor");
    m.def("npu_esa_select_topk.out(Tensor query, Tensor key, int blk_size, int init_blk_num, int local_blk_num, int topk,"
        "str input_layout, *, SymInt[]? actual_seq_q_len_optional=None, SymInt[]? actual_seq_k_len_optional=None,"
        "SymInt[]? actual_cmp_seq_k_len_optional=None, int compress_blk_size=16, Tensor? workspace=None, Tensor out) -> Tensor");
    m.def("_npu_esa_select_topk_get_max_workspace(Tensor query, Tensor key, int blk_size, int init_blk_num, int local_blk_num, int topk,"
        "str input_layout, *, SymInt[]? actual_seq_q_len_optional=None, SymInt[]? actual_seq_k_len_optional=None,"
        "SymInt[]? actual_cmp_seq_k_len_optional=None, int compress_blk_size=16) -> Tensor");
    m.def("_npu_esa_select_topk_infer_output(Tensor query, Tensor key, int init_blk_num, int local_blk_num, int topk,"
        "str input_layout) -> Tensor");
    m.def("npu_ai_infra_scatter_block_update(Tensor input, Tensor indices, Tensor update) -> Tensor");
    m.def("npu_ai_infra_scatter_block_update_(Tensor(a!) input, Tensor indices, Tensor update) -> ()");
    m.def("npu_ai_infra_mhc_pre_split_post_res(Tensor x, Tensor phi, Tensor alpha, Tensor bias, *, float norm_eps=1e-6) -> (Tensor, Tensor)");
    m.def("npu_ai_infra_kv_rmsnorm_rope_cache(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index,"
        " Tensor k_cache, Tensor ckv_cache,"
        " *, Tensor? k_rope_scale=None, Tensor? c_kv_scale=None,"
        " Tensor? k_rope_offset=None, Tensor? c_kv_offset=None, Tensor? v=None,"
        " float epsilon=1e-5, str cache_mode='Norm', str rotary_mode='interleave-half', str quant_mode='static', bool is_output_kv=False, int? ckv_cache_dtype=None) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("npu_ai_infra_kv_rmsnorm_rope_cache_v2(Tensor kv, Tensor gamma, Tensor cos, Tensor sin, Tensor index,"
        " Tensor(a!) k_cache, Tensor(b!) ckv_cache,"
        " *, Tensor? k_rope_scale=None, Tensor? c_kv_scale=None,"
        " Tensor? k_rope_offset=None, Tensor? c_kv_offset=None, Tensor? v=None,"
        " float epsilon=1e-5, str cache_mode='Norm', str rotary_mode='interleave-half', str quant_mode='static', bool is_output_kv=False, int? ckv_cache_dtype=None) -> (Tensor, Tensor)");
    m.def("npu_ai_infra_sparse_flash_attention_pioneer(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, *,"
        "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, "
        "Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, "
        "Tensor? key_sink=None, Tensor? value_sink=None, int sparse_block_size=1, str layout_query='BSND', "
        "str layout_kv='BSND', int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int attention_mode=0, "
        "bool return_softmax_lse=False) -> (Tensor, Tensor, Tensor)");
    m.def("npu_ai_infra_quant_lightning_indexer(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale,"
        "Tensor key_dequant_scale, *, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_key=None,"
        "Tensor? block_table=None, int query_quant_mode=0, int key_quant_mode=0, str layout_query='BSND', str layout_key='BSND',"
        "int sparse_count=2048, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807) -> Tensor");
    m.def("npu_ai_infra_kv_quant_sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices,"
        "float scale_value, int key_quant_mode, int value_quant_mode, *,"
        "Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None,"
        "Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None,"
        "Tensor? key_sink=None, Tensor? value_sink=None,"
        "int sparse_block_size=1, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3,"
        "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, int attention_mode=0, int quant_scale_repo_mode, int tile_size, int rope_head_dim,"
        "int? key_dtype=None, int? value_dtype=None) -> (Tensor)");
}
// 通过pybind将c++接口和python接口绑定，这里绑定的是接口不是算子
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}
