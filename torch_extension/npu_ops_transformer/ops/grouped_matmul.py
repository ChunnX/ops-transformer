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
High-level wrapper for GroupedMatmul operations.
Provides a user-friendly interface for grouped matrix multiplication on NPU.
"""
import torch
from typing import Optional, List, Tuple
from .grouped_matmul_v5 import npu_grouped_matmul_v5


class GroupedMatmulConfig:
    """Configuration for GroupedMatmul operation."""

    # Activation types
    ACT_NONE = 0
    ACT_RELU = 1
    ACT_GELU_TANH = 2
    ACT_GELU_ERF = 3
    ACT_FAST_GELU = 4
    ACT_SILU = 5

    # Split item modes
    SPLIT_MULTI_TENSOR = 0  # Output as multiple tensors
    SPLIT_SINGLE_TENSOR = 2  # Output as single tensor

    # Group types
    GROUP_NONE = -1  # No grouping
    GROUP_M_AXIS = 0  # Group along M axis
    GROUP_N_AXIS = 1  # Group along N axis (not supported yet)
    GROUP_K_AXIS = 2  # Group along K axis

    # Group list types
    GROUP_LIST_CUMSUM = 0  # Group list contains cumulative sum
    GROUP_LIST_SIZE = 1    # Group list contains size per group


class GroupedMatmul:
    """
    High-level wrapper for GroupedMatmul operations.

    Performs grouped matrix multiplication: y_i = x_i @ weight_i + bias_i
    where i = 1...g (g is the number of groups).

    Supports:
    - M-axis grouping: k_i, n_i are the same across groups, m_i can differ
    - K-axis grouping: m_i, n_i are the same across groups, k_i can differ
    - Quantization: INT8, INT4, FP8 quantization modes
    - Activation functions: ReLU, GELU, SiLU, etc.

    Example:
        >>> gmm = GroupedMatmul()
        >>> x_list = [torch.randn(128, 256).npu() for _ in range(8)]
        >>> weight_list = [torch.randn(256, 512).npu() for _ in range(8)]
        >>> out = gmm(x_list, weight_list)
    """

    def __init__(self):
        self.config = GroupedMatmulConfig()

    def __call__(
        self,
        x: List[torch.Tensor],
        weight: List[torch.Tensor],
        bias: Optional[List[torch.Tensor]] = None,
        scale: Optional[List[torch.Tensor]] = None,
        offset: Optional[List[torch.Tensor]] = None,
        antiquant_scale: Optional[List[torch.Tensor]] = None,
        antiquant_offset: Optional[List[torch.Tensor]] = None,
        per_token_scale: Optional[List[torch.Tensor]] = None,
        group_list: Optional[torch.Tensor] = None,
        split_item: int = GroupedMatmulConfig.SPLIT_MULTI_TENSOR,
        group_type: int = GroupedMatmulConfig.GROUP_NONE,
        group_list_type: int = GroupedMatmulConfig.GROUP_LIST_CUMSUM,
        act_type: int = GroupedMatmulConfig.ACT_NONE,
        tuning_config: Optional[List[int]] = None
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Perform grouped matrix multiplication.

        Args:
            x: List of input tensors [m_i, k_i]
            weight: List of weight tensors [k_i, n_i]
            bias: Optional list of bias tensors [n_i]
            scale: Optional list of quantization scale tensors
            offset: Optional list of quantization offset tensors
            antiquant_scale: Optional list of antiquant scale tensors
            antiquant_offset: Optional list of antiquant offset tensors
            per_token_scale: Optional list of per-token scale tensors
            group_list: Optional group list tensor (cumsum or size per group)
            split_item: Output mode (0/1: multi-tensor, 2/3: single-tensor)
            group_type: Grouping axis (-1: none, 0: M-axis, 2: K-axis)
            group_list_type: Group list type (0: cumsum, 1: size per group)
            act_type: Activation function type (0: none, 1: relu, 2: gelu_tanh, etc.)
            tuning_config: Optional tuning configuration

        Returns:
            Tuple of (output_list, activation_feature_out, dyn_quant_scale_out)
        """
        return torch.ops.npu_ops_transformer.npu_grouped_matmul_v5(
            x, weight, bias, scale, offset,
            antiquant_scale, antiquant_offset, per_token_scale,
            group_list, None, None, None,  # activation inputs (reserved)
            split_item, group_type, group_list_type, act_type,
            tuning_config
        )

    def forward_with_activation(
        self,
        x: List[torch.Tensor],
        weight: List[torch.Tensor],
        bias: Optional[List[torch.Tensor]] = None,
        act_type: int = GroupedMatmulConfig.ACT_RELU,
        **kwargs
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Perform grouped matrix multiplication with activation function.

        Args:
            x: List of input tensors
            weight: List of weight tensors
            bias: Optional list of bias tensors
            act_type: Activation function type
            **kwargs: Additional arguments passed to __call__

        Returns:
            Tuple of (output_list, activation_feature_out, dyn_quant_scale_out)
        """
        return self(x, weight, bias=bias, act_type=act_type, **kwargs)

    def forward_quantized(
        self,
        x: List[torch.Tensor],
        weight: List[torch.Tensor],
        scale: List[torch.Tensor],
        offset: Optional[List[torch.Tensor]] = None,
        per_token_scale: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """
        Perform quantized grouped matrix multiplication.

        Args:
            x: List of quantized input tensors (INT8/INT4)
            weight: List of quantized weight tensors
            scale: List of quantization scale tensors
            offset: Optional list of quantization offset tensors
            per_token_scale: Optional list of per-token scale tensors
            **kwargs: Additional arguments passed to __call__

        Returns:
            Tuple of (output_list, activation_feature_out, dyn_quant_scale_out)
        """
        return self(x, weight, scale=scale, offset=offset,
                   per_token_scale=per_token_scale, **kwargs)

    @staticmethod
    def create_group_list_cumsum(group_sizes: List[int]) -> torch.Tensor:
        """
        Create group list tensor in cumsum format.

        Args:
            group_sizes: List of group sizes [m_1, m_2, ..., m_g]

        Returns:
            Group list tensor containing cumulative sum

        Example:
            >>> group_sizes = [128, 256, 512]
            >>> group_list = GroupedMatmul.create_group_list_cumsum(group_sizes)
            >>> # group_list = [128, 384, 896]
        """
        import numpy as np
        cumsum = np.cumsum(group_sizes)
        return torch.tensor(cumsum, dtype=torch.int64)

    @staticmethod
    def create_group_list_size(group_sizes: List[int]) -> torch.Tensor:
        """
        Create group list tensor in size format.

        Args:
            group_sizes: List of group sizes [m_1, m_2, ..., m_g]

        Returns:
            Group list tensor containing size per group

        Example:
            >>> group_sizes = [128, 256, 512]
            >>> group_list = GroupedMatmul.create_group_list_size(group_sizes)
            >>> # group_list = [128, 256, 512]
        """
        return torch.tensor(group_sizes, dtype=torch.int64)


# Convenience function
def grouped_matmul(
    x: List[torch.Tensor],
    weight: List[torch.Tensor],
    bias: Optional[List[torch.Tensor]] = None,
    **kwargs
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    """
    Convenience function for grouped matrix multiplication.

    Args:
        x: List of input tensors
        weight: List of weight tensors
        bias: Optional list of bias tensors
        **kwargs: Additional arguments (see GroupedMatmul.__call__)

    Returns:
        Tuple of (output_list, activation_feature_out, dyn_quant_scale_out)

    Example:
        >>> x_list = [torch.randn(128, 256).npu() for _ in range(8)]
        >>> weight_list = [torch.randn(256, 512).npu() for _ in range(8)]
        >>> out, _, _ = grouped_matmul(x_list, weight_list)
    """
    gmm = GroupedMatmul()
    return gmm(x, weight, bias, **kwargs)
