"""Small transfer-test paramset for the DSL QLI V2-compatible test harness."""

import torch


def _case(batch, s1, s2, *, ratio=1, storage="contiguous",
          candidate=2048, candidate_block=8, seed=1):
    pages = (s2 + 127) // 128
    return {
        "batch_size": [batch],
        "q_seq": [s1],
        "k_seq": [s2],
        "q_t_size": [batch * s1],
        "k_t_size": [s2],
        "q_head_num": [32],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128],
        "block_num": [pages],
        "qk_dtype": [torch.uint8],
        "weight_dtype": [torch.float32],
        "dequant_dtype": [torch.uint8],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],
        "cu_seqlens_k": [None],
        "seqused_q": [None],
        "seqused_k": [None],
        "cmp_residual_k": [None],
        "max_seqlen_q": [s1],
        "quant_mode": [1],
        "layout_query": ["TND"],
        "layout_key": ["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange": [[0, 255]],
        "key_datarange": [[0, 255]],
        "weights_datarange": [[0.0, 1.0]],
        "q_scale_datarange": [[125, 129]],
        "k_scale_datarange": [[125, 129]],
        "cmp_ratio": [ratio],
        "return_value": [1],
        "output_idx_offset": [None],
        "candidate_topk_blocks": [candidate],
        "candidate_block_size": [candidate_block],
        "storage_layout": [storage],
        "seed": [seed],
        "run_mode": ["eager"],
    }


TEST_PARAMS = {
    "decode_b8_s2_8k": _case(8, 1, 8192, seed=101),
    "prefill_b1_s1_8_s2_4k": _case(1, 8, 4096, seed=102),
    "cross_01_tiny_s2_5": _case(1, 1, 5, seed=201),
    "cross_02_s2_tail_127": _case(2, 1, 127, seed=202),
    "cross_03_query_tail_s1_3": _case(4, 3, 128, seed=203),
    "cross_04_full_query_tile_s1_4": _case(8, 4, 129, seed=204),
    "cross_05_cmp_ratio_2": _case(16, 1, 1024, ratio=2, seed=205),
    "cross_06_max_batch_32": _case(32, 1, 2048, seed=206),
    "cross_07_multi_query_task": _case(3, 5, 8192, ratio=2, seed=207),
    "cross_08_topk_trunk_merge": _case(1, 1, 16385, seed=208),
    "cross_09_pa_axis0_noncontiguous": _case(
        2, 1, 32768, storage="axis0_noncontiguous", seed=209
    ),
    "cross_10_candidate_disabled": _case(
        4, 1, 8192, candidate=-1, candidate_block=-1, seed=210
    ),
}

ENABLED_PARAMSETS = list(TEST_PARAMS.items())
