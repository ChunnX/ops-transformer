import torch


def _gather_per_token_kscale_one_req_layout_b(
    kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
):
    """Materialize per-token K-scale for one request (Layout B): shape [k_padded, num_head_kv].

    Layout B kScaleCache shape: [total_blocks, H_kv, kv_block_size // 32, 32]
    Padding rows (>= kv_len) get zero.
    """
    device = kscale_fp32.device
    num_head_kv = kscale_fp32.shape[1]
    num_kv_blocks = block_ids.shape[0]

    token_ids = torch.arange(k_padded, device=device, dtype=torch.long)
    valid_mask = token_ids < kv_len
    page_idx = (token_ids // kv_block_size).clamp(max=num_kv_blocks - 1)
    row_in_page = token_ids % kv_block_size
    phys_blocks = block_ids[page_idx].to(torch.long)
    row_chunk = row_in_page // 32
    row_offset = row_in_page % 32
    gathered = kscale_fp32[phys_blocks, :, row_chunk, row_offset]
    gathered = torch.where(
        valid_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered),
    )
    return gathered


def _prep_kv_reference_layout_b(
    kcache_fp8: torch.Tensor,
    vcache_fp8: torch.Tensor,
    kscale_fp32: torch.Tensor,
    vscale: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    kv_block_size: int,
    lambda_mag: float = 0.3,
    stem_block_size: int = 128,
    stem_stride: int = 16,
):
    """Pure PyTorch reference for Layout B (contiguous): [total_blocks, H_kv, kv_block_size, D].

    Layout B differences from Layout A:
    - kcache/vcache: [total_blocks, H_kv, kv_block_size, D] → permute after gather
    - KScaleCache: [total_blocks, H_kv, kv_block_size // 32, 32] → index differently

    Returns:
        kflat: [num_batch, num_head_kv, max_Kb, stem_stride * dim_qk] BF16
        v_bias: [num_batch, num_head_kv, max_Kb] FP32
    """
    device = kcache_fp8.device
    num_batch = kv_seq_lens.shape[0]
    num_head_kv = kcache_fp8.shape[1]
    dim_qk = kcache_fp8.shape[3]
    dim_v = vcache_fp8.shape[3]
    R = stem_block_size // stem_stride

    k_padded_lens = (
        (kv_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    ).to(torch.int32)
    max_k_padded = k_padded_lens.max().item()
    max_Kb = max_k_padded // stem_block_size

    kflat_out = torch.zeros(
        num_batch, num_head_kv, max_Kb, stem_stride * dim_qk,
        dtype=torch.bfloat16, device=device,
    )
    v_bias_out = torch.zeros(
        num_batch, num_head_kv, max_Kb,
        dtype=torch.float32, device=device,
    )

    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        k_padded = k_padded_lens[b].item()
        num_stem_blocks = k_padded // stem_block_size
        k_down_len = k_padded // stem_stride

        num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size
        block_ids = kv_indices[b, :num_kv_blocks]

        # Layout B: [num_kv_blocks, H_kv, kv_block_size, D] → permute to [num_kv_blocks, kv_block_size, H_kv, D]
        k_paged = kcache_fp8[block_ids].float().permute(0, 2, 1, 3).reshape(-1, num_head_kv, dim_qk)
        v_paged = vcache_fp8[block_ids].float().permute(0, 2, 1, 3).reshape(-1, num_head_kv, dim_v)

        k_dense = torch.zeros(k_padded, num_head_kv, dim_qk, dtype=torch.float32, device=device)
        v_dense = torch.zeros(k_padded, num_head_kv, dim_v, dtype=torch.float32, device=device)
        actual_rows = min(kv_len, num_kv_blocks * kv_block_size)
        k_dense[:actual_rows] = k_paged[:actual_rows]
        v_dense[:actual_rows] = v_paged[:actual_rows]

        kscale_per_row = _gather_per_token_kscale_one_req_layout_b(
            kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
        )

        k_scaled = (
            k_dense[:num_stem_blocks * stem_block_size]
            * kscale_per_row[:num_stem_blocks * stem_block_size].unsqueeze(-1)
        )

        for h in range(num_head_kv):
            k_h = k_scaled[:, h, :]
            k_blocks = k_h.reshape(num_stem_blocks, R, stem_stride, dim_qk)
            k_group_sum = k_blocks.sum(dim=1)
            k_group_rev = k_group_sum.flip(1)
            kflat_out[b, h, :num_stem_blocks] = k_group_rev.reshape(
                num_stem_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

            vscale_h = vscale[h].item()
            v_h = v_dense[:num_stem_blocks * stem_block_size, h, :] * vscale_h
            v_rows = v_h.reshape(k_down_len, stem_stride, dim_v)
            norms = torch.sqrt((v_rows ** 2).sum(dim=-1))
            row_ids = torch.arange(
                num_stem_blocks * stem_block_size, device=device,
            ).reshape(k_down_len, stem_stride)
            norms = torch.where(row_ids < kv_len, norms, torch.zeros_like(norms))
            v_norm_down = norms.max(dim=-1).values

            if k_down_len > 0:
                log_vals = torch.log(v_norm_down + 1e-6)
                v_mean = log_vals.mean()
                if k_down_len > 1:
                    v_std = log_vals.std(unbiased=True)
                else:
                    v_std = torch.tensor(0.0, device=device)
                inv_std = 1.0 / (v_std + 1e-6)
                normalized = (log_vals - v_mean) * inv_std
                v_final = lambda_mag * torch.relu(normalized)
                v_final_blocks = v_final[:num_stem_blocks * R].reshape(num_stem_blocks, R)
                v_bias_out[b, h, :num_stem_blocks] = v_final_blocks.mean(dim=1)

    return kflat_out, v_bias_out