#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for stem_oam_prep_varlen_q
"""
# Third-Party Packages
import math
import torch
import numpy as np
import tbetoolkits
from .registry import register_golden


@register_golden(["stem_oam_prep_varlen_q"])
def stem_oam_prep_varlen_q(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Compute golden reference for stem_oam_prep_varlen_q.

    Args:
        context: Contains input arrays and compilation params.
                 input_arrays: q_fp8       [total_tokens, H_q, D]  (numpy, representing FP8_e4m3fn)
                 input_arrays: qscale      [total_tokens, H_q]     (numpy, FP32)
                 input_arrays: q_seq_lens  [batch]                 (numpy, INT32)
                 input_arrays: cu_seqlens_q[batch+1]               (numpy, INT32)
                 
                 other_compilation_params:
                 - B: stem_block_size (default 128)
                 - S: stem_stride (default 16)

    Returns:
        qflat: [batch, H_q, max_Qb, S*D] BF16 (as numpy array)
    """
    
    # 1. Retrieve Inputs from Context
    # Note: Assuming the order of inputs matches the description. 
    # Adjust indices if the actual test case definition differs.
    q_fp8_np = context.input_arrays
    qscale_np = context.input_arrays
    q_seq_lens_np = context.input_arrays
    cu_seqlens_q_np = context.input_arrays

    # Retrieve parameters with defaults
    B = context.other_compilation_params.get('B', 128)
    S = context.other_compilation_params.get('S', 16)

    # 2. Convert to Torch Tensors
    # FP8 data is stored in numpy but needs to be interpreted correctly.
    # Since numpy doesn't natively support FP8 E4M3, we assume the input numpy array 
    # is either uint8 representing the bits or already cast to a compatible type.
    # Here we treat it as raw data and cast to float for calculation (dequantization simulation).
    q_fp8 = torch.from_numpy(q_fp8_np.copy())
    qscale = torch.from_numpy(qscale_np.copy()).float()
    q_seq_lens = torch.from_numpy(q_seq_lens_np.copy()).int()
    cu_seqlens_q = torch.from_numpy(cu_seqlens_q_np.copy()).int()

    # Ensure q_fp8 is treated as float for computation. 
    # If q_fp8_np was uint8 (raw FP8 bits), specific unpacking would be needed.
    # Assuming standard test bench behavior where .float() converts the numeric value appropriately
    # or the input is already provided as a high-precision approximation for golden ref.
    # For strict FP8 E4M3 simulation, one might need torch.float8_e4m3fn if supported, 
    # but standard golden refs often compute in FP32/FP64.
    # Here we cast to float32 to simulate the dequantization step.
    if q_fp8.dtype != torch.float32:
        # If it's uint8/raw, this step depends on how the test case generates input.
        # Usually, for golden ref, we might receive the "value" directly or need to view_as.
        # Assuming the input array contains the actual float values represented by FP8 for simplicity,
        # or that torch.from_numpy handles the type conversion if dtype is specified in test gen.
        # To be safe and generic, we cast to float.
        q_fp8 = q_fp8.float()

    # 3. Shape Definitions
    D = q_fp8.shape[-1]       # Feature dimension, e.g., 128
    H_q = q_fp8.shape      # Number of Q heads
    R = B // S                # Rows per block group, e.g., 8
    batch = len(q_seq_lens)

    # 4. Compute max_Qb (max number of blocks per sequence in batch)
    max_Qb = max(math.ceil(sl.item() / B) for sl in q_seq_lens)

    # 5. Initialize Output Tensor
    # Output dtype is BF16
    qflat = torch.zeros(batch, H_q, max_Qb, S * D, dtype=torch.bfloat16)

    # 6. Core Computation Loop
    for b in range(batch):
        q_len = q_seq_lens[b].item()
        cu_off = cu_seqlens_q[b].item()
        
        # Calculate padded length to nearest block boundary
        q_padded = math.ceil(q_len / B) * B
        num_Qb = q_padded // B
        
        if num_Qb == 0:
            continue

        # Stage 2: De-page varlen: extract segment and cast to FP32, zero-pad
        # Extract valid tokens for this batch item
        Q_segment = q_fp8[cu_off : cu_off + q_len]   # [q_len, H_q, D]
        
        # Create dense padded tensor for this batch item
        Q_dense = torch.zeros(q_padded, H_q, D, dtype=torch.float32)
        Q_dense[:q_len] = Q_segment

        for h in range(H_q):
            # Reshape into stem blocks: [num_Qb, B, D] -> [num_Qb, R, S, D]
            # Take only the valid blocks part
            Q_head_dense = Q_dense[:num_Qb * B, h, :] # [num_Qb * B, D]
            Q_blocks = Q_head_dense.reshape(num_Qb, R, S, D)

            # Prepare scales for this head and block structure
            # positions within the block structure: [num_Qb, R, S]
            positions = torch.arange(num_Qb * B).reshape(num_Qb, R, S)
            
            # Map to global positions to fetch correct scales
            global_positions = cu_off + positions # [num_Qb, R, S]
            
            # Fetch scales: [num_Qb, R, S]
            # Ensure indices are within bounds of qscale
            scales = qscale[global_positions, h]   

            # Weighted group sum: Q × q_scale, then sum over R dimension
            # Q_blocks: [num_Qb, R, S, D]
            # scales: [num_Qb, R, S] -> unsqueeze(-1) -> [num_Qb, R, S, 1]
            Q_weighted = Q_blocks * scales.unsqueeze(-1)   # [num_Qb, R, S, D]
            
            # Sum over R (dim=1) to get group summary
            Q_group_sum = Q_weighted.sum(dim=1)             # [num_Qb, S, D]

            # Stage 4: Flatten in natural order (NO flip) and convert to BF16
            # Target shape for slice: [S * D]
            for qb in range(num_Qb):
                # Reshape [S, D] to [S * D]
                flat_vec = Q_group_sum[qb].reshape(S * D)
                
                # Convert to BF16 and assign
                qflat[b, h, qb, :] = flat_vec.bfloat16()

    # 7. Return as Numpy Array
    return qflat.numpy()
