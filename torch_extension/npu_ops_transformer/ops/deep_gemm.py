# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepGEMM Buffer - High-level wrapper for DeepGEMM-style grouped matmul on NPU.

This module provides a MoeDistributeBuffer-style interface for DeepGEMM users,
enabling them to call DeepGEMM-style grouped matmul operations using CANN's
npu_grouped_matmul underneath.

Usage:
    from npu_ops_transformer.ops import DeepGEMMBuffer

    buffer = DeepGEMMBuffer()
    buffer.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a=(a_quant, a_scale),
        b=(b_quant, b_scale),
        d=output,
        grouped_layout=layout,
        use_psum_layout=False,
    )
"""

import torch
import torch_npu
from typing import Tuple, Optional, List, Union
from npu_ops_transformer.ops.grouped_matmul_v5 import npu_grouped_matmul_v5


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align(x, y):
    return _ceil_div(x, y) * y


# DeepGEMM alignment constant (128)
DEEPGEMM_ALIGNMENT = 128


def _convert_psum_layout_to_group_list_and_offsets(
    grouped_layout: torch.Tensor
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Convert psum_layout to group_list for CANN.

    psum_layout stores cumulative actual_end for each group:
        grouped_layout = [actual_end_0, actual_end_1, ...]

    Returns:
        group_list: actual sizes for each group [m0, m1, ...]
        valid_starts: starting positions of valid data in original tensor
        aligned_starts: starting positions after 128-alignment
    """
    layout_cpu = grouped_layout.cpu().numpy()
    num_groups = len(layout_cpu)

    actual_sizes = []
    valid_starts = []
    aligned_starts = []

    prev_actual_end = 0
    prev_aligned_end = 0

    for i in range(num_groups):
        actual_end = int(layout_cpu[i])
        actual_size = actual_end - prev_actual_end
        actual_sizes.append(actual_size)
        valid_starts.append(prev_actual_end)
        aligned_end = _align(actual_end, DEEPGEMM_ALIGNMENT)
        aligned_starts.append(prev_aligned_end)
        prev_actual_end = actual_end
        prev_aligned_end = aligned_end

    group_list = torch.tensor(actual_sizes, dtype=torch.long, device=grouped_layout.device)

    return group_list, valid_starts, aligned_starts


def _convert_grouped_layout_to_group_list_and_offsets(
    grouped_layout: torch.Tensor
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """
    Convert non-psum grouped_layout to group_list for CANN.

    grouped_layout stores group_id for valid positions, -1 for padding:
        grouped_layout = [0, 0, 0, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, ...]

    Returns:
        group_list: actual sizes for each group [m0, m1, ...]
        valid_starts: starting positions of valid data
        aligned_starts: starting positions after 128-alignment
    """
    layout_cpu = grouped_layout.cpu().numpy()
    m_total = len(layout_cpu)

    group_info = {}

    for i in range(m_total):
        gid = layout_cpu[i]
        if gid >= 0:
            if gid not in group_info:
                group_info[gid] = {'start': i, 'end': i + 1}
            else:
                group_info[gid]['end'] = i + 1

    num_groups = len(group_info)

    actual_sizes = []
    valid_starts = []
    aligned_starts = []

    prev_aligned_end = 0

    for gid in sorted(group_info.keys()):
        info = group_info[gid]
        valid_start = info['start']
        valid_end = info['end']
        actual_size = valid_end - valid_start
        aligned_end = _align(valid_end, DEEPGEMM_ALIGNMENT)

        actual_sizes.append(actual_size)
        valid_starts.append(valid_start)
        aligned_starts.append(prev_aligned_end)

        prev_aligned_end = aligned_end

    group_list = torch.tensor(actual_sizes, dtype=torch.long, device=grouped_layout.device)

    return group_list, valid_starts, aligned_starts


def _remove_padding_from_tensor(
    tensor: torch.Tensor,
    starts: List[int],
    sizes: List[int],
    axis: int = 0
) -> torch.Tensor:
    """
    Remove padding from tensor by extracting valid slices.

    Args:
        tensor: Input tensor with padding
        starts: List of starting indices for each group
        sizes: List of valid sizes for each group
        axis: Axis along which to extract

    Returns:
        Concatenated tensor without padding
    """
    if tensor.dim() == 1:
        slices = [tensor[start:start+size] for start, size in zip(starts, sizes)]
        return torch.cat(slices, dim=0)
    elif tensor.dim() == 2:
        slices = [tensor[start:start+size, :] for start, size in zip(starts, sizes)]
        return torch.cat(slices, dim=0)
    elif tensor.dim() == 3:
        slices = [tensor[:, start:start+size, :] for start, size in zip(starts, sizes)]
        return torch.cat(slices, dim=1)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


def _restore_padding_to_tensor(
    tensor: torch.Tensor,
    aligned_starts: List[int],
    sizes: List[int],
    total_size: int,
    axis: int = 0,
    dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Restore padding to tensor by inserting zero-filled padding.

    Args:
        tensor: Input tensor without padding
        aligned_starts: Starting positions after 128-alignment
        sizes: Valid sizes for each group
        total_size: Total size including padding
        axis: Axis along which to insert padding
        dtype: Output dtype

    Returns:
        Tensor with padding restored
    """
    if tensor.dim() == 1:
        result = torch.zeros(total_size, dtype=dtype or tensor.dtype, device=tensor.device)
        offset = 0
        for aligned_start, size in zip(aligned_starts, sizes):
            result[aligned_start:aligned_start+size] = tensor[offset:offset+size]
            offset += size
        return result
    elif tensor.dim() == 2:
        n = tensor.shape[1]
        result = torch.zeros(total_size, n, dtype=dtype or tensor.dtype, device=tensor.device)
        offset = 0
        for aligned_start, size in zip(aligned_starts, sizes):
            result[aligned_start:aligned_start+size, :] = tensor[offset:offset+size, :]
            offset += size
        return result
    elif tensor.dim() == 3:
        b, m, n = tensor.shape
        result = torch.zeros(b, total_size, n, dtype=dtype or tensor.dtype, device=tensor.device)
        offset = 0
        for aligned_start, size in zip(aligned_starts, sizes):
            result[:, aligned_start:aligned_start+size, :] = tensor[:, offset:offset+size, :]
            offset += size
        return result
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


class DeepGEMMBuffer:
    """
    High-level buffer for DeepGEMM-style grouped matmul operations.

    This class provides a MoeDistributeBuffer-style interface for DeepGEMM users,
    enabling them to call DeepGEMM-style grouped matmul operations using CANN's
    npu_grouped_matmul underneath.

    Key features:
    - Automatic conversion between DeepGEMM's 128-aligned padded format and CANN's format
    - Support for FP8/FP4 quantization patterns
    - Support for both transposed (NT) and non-transposed (NN) matmul
    - Optional bias (bias_add) support

    Example:
        buffer = DeepGEMMBuffer()
        buffer.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a=(a_quant, a_scale),
            b=(b_quant, b_scale),
            d=output,
            grouped_layout=layout,
        )
    """

    def __init__(self, device=None):
        """
        Initialize DeepGEMMBuffer.

        Args:
            device: Optional torch device. If None, uses the current NPU device.
        """
        self.device = device

    def m_grouped_fp8_fp4_gemm_nt_contiguous(
        self,
        a: Tuple[torch.Tensor, torch.Tensor],
        b: Tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        grouped_layout: torch.Tensor,
        recipe: Optional[Tuple[int, int, int]] = None,
        recipe_a: Optional[Tuple[int, int]] = None,
        recipe_b: Optional[Tuple[int, int]] = None,
        compiled_dims: str = "",
        disable_ue8m0_cast: bool = False,
        use_psum_layout: bool = False,
        expected_m_for_psum_layout: Optional[int] = None,
        c: Optional[torch.Tensor] = None,
    ) -> None:
        """
        DeepGEMM-style grouped FP8/FP4 matmul (transposed B, NT).

        This function converts DeepGEMM's 128-aligned padded format to CANN's unpadded format,
        calls npu_grouped_matmul, then restores the padding to the output tensor d.

        Args:
            a: Tuple of (a_quant, a_scale) - quantized activation tensor and its scale.
               Shape: a_quant = (m_total, k), a_scale depends on quantization granularity.
            b: Tuple of (b_quant, b_scale) - quantized weight tensor and its scale.
               Shape: b_quant = (num_groups, n, k) or (num_groups, k, n) depending on layout.
            d: Output tensor with shape (m_total, n). Will be modified in-place.
            grouped_layout: Layout tensor describing group boundaries.
                - If use_psum_layout=True: stores cumulative end positions [end_0, end_1, ...]
                - If use_psum_layout=False: stores group_id for each position, -1 for padding
            recipe: Optional (gran_m, gran_n, gran_k) tuple for quantization recipe.
            recipe_a: Optional (gran_a, scale_type_a) tuple for activation quantization.
            recipe_b: Optional (gran_b, scale_type_b) tuple for weight quantization.
            compiled_dims: Compilation dimensions string (for DeepGEMM compatibility).
            disable_ue8m0_cast: Whether to disable float8_e8m0 cast.
            use_psum_layout: Whether grouped_layout uses psum format.
            expected_m_for_psum_layout: Expected m for psum layout.
            c: Optional bias tensor with shape matching the dequantized output.

        Returns:
            None. The output is written to tensor d in-place.
        """
        a_quant, a_scale = a
        b_quant, b_scale = b

        m_total = a_quant.shape[0]
        k = a_quant.shape[1]
        num_groups = b_quant.shape[0]
        n = b_quant.shape[2] if b_quant.dim() == 3 else b_quant.shape[1]

        # Convert layout to group_list
        if use_psum_layout:
            group_list, valid_starts, aligned_starts = _convert_psum_layout_to_group_list_and_offsets(grouped_layout)
        else:
            group_list, valid_starts, aligned_starts = _convert_grouped_layout_to_group_list_and_offsets(grouped_layout)

        # Remove padding from activation
        a_quant_no_pad = _remove_padding_from_tensor(a_quant, valid_starts, group_list.tolist(), axis=0)
        a_scale_no_pad = _remove_padding_from_tensor(a_scale, valid_starts, group_list.tolist(), axis=0)

        # Prepare weight list
        weight_list = [b_quant]

        # Determine quantization parameters
        use_quant = True
        if recipe_a is not None and recipe_b is not None:
            gran_a = recipe_a[0] if recipe_a else 0
            gran_b = recipe_b[0] if recipe_b else 0
            if gran_a == 0 or gran_b == 0:
                use_quant = False

        if use_quant:
            scale_list = [a_scale_no_pad, b_scale]
            antiquant_scale_list = None
            antiquant_offset_list = None
        else:
            scale_list = None
            antiquant_scale_list = None
            antiquant_offset_list = None

        # Prepare other parameters
        split_item = 3  # IN_SPLIT_OUT_SPLIT
        group_type = 0
        group_list_type = 1
        act_type = 0
        output_dtype = d.dtype
        bias_list = None

        if c is not None:
            c_no_pad = _remove_padding_from_tensor(c, valid_starts, group_list.tolist(), axis=0)
            bias_list = [c_no_pad]

        x_list = [a_quant_no_pad]

        # Call npu_grouped_matmul via torch.ops.npu_ops_transformer
        out = torch.ops.npu_ops_transformer.npu_grouped_matmul_v5(
            x_list,
            weight_list,
            bias=bias_list,
            scale=scale_list,
            offset=None,
            antiquant_scale=antiquant_scale_list,
            antiquant_offset=antiquant_offset_list,
            per_token_scale=None,
            group_list=group_list,
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=split_item,
            group_type=group_type,
            group_list_type=group_list_type,
            act_type=act_type,
            tuning_config=None,
            output_dtype=output_dtype,
        )

        if isinstance(out, list):
            out_tensor = out[0]
        else:
            out_tensor = out

        # Restore padding to output
        output_with_padding = _restore_padding_to_tensor(
            out_tensor,
            aligned_starts,
            group_list.tolist(),
            m_total,
            axis=0,
            dtype=output_dtype
        )

        d.copy_(output_with_padding)

    def m_grouped_fp8_fp4_gemm_nn_contiguous(
        self,
        a: Tuple[torch.Tensor, torch.Tensor],
        b: Tuple[torch.Tensor, torch.Tensor],
        d: torch.Tensor,
        grouped_layout: torch.Tensor,
        recipe: Optional[Tuple[int, int, int]] = None,
        recipe_a: Optional[Tuple[int, int]] = None,
        recipe_b: Optional[Tuple[int, int]] = None,
        compiled_dims: str = "",
        disable_ue8m0_cast: bool = False,
        use_psum_layout: bool = False,
        expected_m_for_psum_layout: Optional[int] = None,
        c: Optional[torch.Tensor] = None,
    ) -> None:
        """
        DeepGEMM-style grouped FP8/FP4 matmul (non-transposed B, NN).

        For NN (non-transposed B), we transpose B before calling NT version.

        Args:
            See m_grouped_fp8_fp4_gemm_nt_contiguous.

        Returns:
            None. The output is written to tensor d in-place.
        """
        b_quant, b_scale = b

        # Transpose B for NN -> NT conversion
        if b_quant.dim() == 3:
            b_transposed = b_quant.transpose(1, 2).contiguous()
            b_transposed_scale = b_scale.transpose(1, 2).contiguous() if b_scale.dim() > 1 else b_scale
        else:
            b_transposed = b_quant.t().contiguous()
            b_transposed_scale = b_scale

        self.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a,
            (b_transposed, b_transposed_scale),
            d,
            grouped_layout,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            compiled_dims=compiled_dims,
            disable_ue8m0_cast=disable_ue8m0_cast,
            use_psum_layout=use_psum_layout,
            expected_m_for_psum_layout=expected_m_for_psum_layout,
            c=c,
        )

    def extract_valid_a_from_padded_quant(
        self,
        a_quant: torch.Tensor,
        grouped_layout: torch.Tensor,
        gran_k: int = 128,
        use_psum_layout: bool = False,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Extract valid (non-padded) tensors from padded quantization tensor.

        Args:
            a_quant: Padded quantization tensor.
            grouped_layout: Layout tensor describing group boundaries.
            gran_k: Alignment granularity (default 128).
            use_psum_layout: Whether using psum layout.

        Returns:
            Tuple of (list of valid tensors for each group, list of valid lengths).
        """
        if use_psum_layout:
            group_list, valid_starts, _ = _convert_psum_layout_to_group_list_and_offsets(grouped_layout)
        else:
            group_list, valid_starts, _ = _convert_grouped_layout_to_group_list_and_offsets(grouped_layout)

        group_sizes = group_list.tolist()

        a_slices = []
        for start, size in zip(valid_starts, group_sizes):
            a_slices.append(a_quant[start:start+size])

        return a_slices, group_sizes
