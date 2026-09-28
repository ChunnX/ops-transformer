"""QSLI transfer-test parameter sets: decode, prefill, and cross coverage."""

TEST_PARAMS = {
    "cleancode_b1_s1_4_s2_8192": dict(batch_size=[1], query_length=[4], sequence_length=[8192], cmp_ratio=[2], cmp_residual_k=[[0]], storage_mode=["contiguous"], mask_mode=[3], topk=[512], candidate_block_length=[1024], output_idx_offset=[True], seed=[301]),
    "qsli_decode": dict(batch_size=[1], query_length=[1], sequence_length=[8192], cmp_ratio=[1], storage_mode=["contiguous"], mask_mode=[3], topk=[512], candidate_block_length=[1024], output_idx_offset=[True], seed=[101]),
    "qsli_prefill": dict(batch_size=[1], query_length=[128], sequence_length=[1024], cmp_ratio=[1], storage_mode=["contiguous"], mask_mode=[3], topk=[512], candidate_block_length=[128], output_idx_offset=[True], seed=[102]),
    "qsli_cross_b2_r2": dict(batch_size=[2], query_length=[4], sequence_length=[2048], cmp_ratio=[2], cmp_residual_k=[[0, 1]], storage_mode=["contiguous"], mask_mode=[3], topk=[512], candidate_block_length=[256], output_idx_offset=[True], seed=[103]),
    "qsli_cross_b4_mask0": dict(batch_size=[4], query_length=[1], sequence_length=[1024], cmp_ratio=[1], storage_mode=["contiguous"], mask_mode=[0], topk=[512], candidate_block_length=[128], output_idx_offset=[False], seed=[104]),
    "qsli_cross_dim0": dict(batch_size=[1], query_length=[2], sequence_length=[2048], cmp_ratio=[1], storage_mode=["dim0_noncontiguous"], mask_mode=[3], topk=[512], candidate_block_length=[256], output_idx_offset=[True], seed=[105]),
    "qsli_cross_key_stride": dict(batch_size=[1], query_length=[2], sequence_length=[1024], cmp_ratio=[1], storage_mode=["key_dim0_noncontiguous"], mask_mode=[3], topk=[512], candidate_block_length=[128], output_idx_offset=[True], seed=[106]),
    "qsli_cross_scale_stride": dict(batch_size=[1], query_length=[2], sequence_length=[1024], cmp_ratio=[1], storage_mode=["scale_dim0_noncontiguous"], mask_mode=[3], topk=[512], candidate_block_length=[128], output_idx_offset=[True], seed=[107]),
    "qsli_cross_tail5": dict(batch_size=[1], query_length=[1], sequence_length=[5], cmp_ratio=[1], storage_mode=["contiguous"], mask_mode=[3], topk=[8], candidate_block_length=[1], output_idx_offset=[True], seed=[108]),
    "qsli_cross_tail129": dict(batch_size=[1], query_length=[1], sequence_length=[129], cmp_ratio=[1], storage_mode=["contiguous"], mask_mode=[3], topk=[128], candidate_block_length=[17], output_idx_offset=[True], seed=[109]),
    "qsli_cross_candidate17": dict(batch_size=[1], query_length=[1], sequence_length=[512], cmp_ratio=[2], cmp_residual_k=[[0]], storage_mode=["contiguous"], mask_mode=[3], topk=[64], candidate_block_length=[17], output_idx_offset=[True], seed=[110]),
}

ENABLED_PARAMSETS = list(TEST_PARAMS.items())
