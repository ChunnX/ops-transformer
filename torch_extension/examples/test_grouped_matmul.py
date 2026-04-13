#!/usr/bin/env python3
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
Example usage of GroupedMatmul torch extension.
Demonstrates basic usage, quantization, and activation functions.
"""
import torch
import torch_npu
import npu_ops_transformer


def test_basic_grouped_matmul():
    """Test basic grouped matrix multiplication."""
    print("=" * 80)
    print("Test 1: Basic GroupedMatmul")
    print("=" * 80)

    # Create input tensors
    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]

    # Method 1: Using high-level wrapper
    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm(x_list, weight_list)

    print(f"Input: {num_groups} tensors of shape {x_list[0].shape}")
    print(f"Weight: {num_groups} tensors of shape {weight_list[0].shape}")
    print(f"Output: {len(out)} tensors of shape {out[0].shape}")
    print("✓ Basic GroupedMatmul test passed\n")


def test_with_bias_and_activation():
    """Test grouped matmul with bias and activation."""
    print("=" * 80)
    print("Test 2: GroupedMatmul with Bias and ReLU Activation")
    print("=" * 80)

    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]
    bias_list = [torch.randn(512, dtype=torch.float16).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm.forward_with_activation(
        x_list, weight_list, bias_list,
        act_type=npu_ops_transformer.ops.GroupedMatmulConfig.ACT_RELU
    )

    print(f"Input: {num_groups} tensors of shape {x_list[0].shape}")
    print(f"Weight: {num_groups} tensors of shape {weight_list[0].shape}")
    print(f"Bias: {num_groups} tensors of shape {bias_list[0].shape}")
    print(f"Activation: ReLU")
    print(f"Output: {len(out)} tensors of shape {out[0].shape}")
    print("✓ Bias and activation test passed\n")


def test_quantized_mode():
    """Test quantized grouped matmul."""
    print("=" * 80)
    print("Test 3: Quantized GroupedMatmul (INT8)")
    print("=" * 80)

    num_groups = 8
    # INT8 quantized inputs
    x_list = [torch.randint(-128, 127, (128, 256), dtype=torch.int8).npu() for _ in range(num_groups)]
    weight_list = [torch.randint(-128, 127, (256, 512), dtype=torch.int8).npu() for _ in range(num_groups)]
    scale_list = [torch.tensor([0.01], dtype=torch.float32).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm.forward_quantized(
        x_list, weight_list, scale_list
    )

    print(f"Input: {num_groups} INT8 tensors of shape {x_list[0].shape}")
    print(f"Weight: {num_groups} INT8 tensors of shape {weight_list[0].shape}")
    print(f"Scale: {num_groups} tensors")
    print(f"Output: {len(out)} tensors of shape {out[0].shape}")
    print("✓ Quantized mode test passed\n")


def test_with_group_list():
    """Test grouped matmul with group list (M-axis grouping)."""
    print("=" * 80)
    print("Test 4: GroupedMatmul with Group List (M-axis grouping)")
    print("=" * 80)

    num_groups = 8
    group_sizes = [128, 256, 512, 128, 256, 512, 128, 256]

    # Create tensors with different M sizes
    x_list = [torch.randn(m, 256, dtype=torch.float16).npu() for m in group_sizes]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]

    # Create group list (cumsum format)
    group_list = npu_ops_transformer.ops.GroupedMatmul.create_group_list_cumsum(group_sizes)

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm(
        x_list, weight_list,
        group_list=group_list.npu(),
        group_type=npu_ops_transformer.ops.GroupedMatmulConfig.GROUP_M_AXIS,
        group_list_type=npu_ops_transformer.ops.GroupedMatmulConfig.GROUP_LIST_CUMSUM
    )

    print(f"Input: {num_groups} tensors with varying M sizes: {group_sizes}")
    print(f"Weight: {num_groups} tensors of shape {weight_list[0].shape}")
    print(f"Group list (cumsum): {group_list.tolist()}")
    print(f"Output: {len(out)} tensors")
    print("✓ Group list test passed\n")


def test_single_tensor_output():
    """Test grouped matmul with single tensor output."""
    print("=" * 80)
    print("Test 5: GroupedMatmul with Single Tensor Output")
    print("=" * 80)

    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm(
        x_list, weight_list,
        split_item=npu_ops_transformer.ops.GroupedMatmulConfig.SPLIT_SINGLE_TENSOR
    )

    print(f"Input: {num_groups} tensors of shape {x_list[0].shape}")
    print(f"Weight: {num_groups} tensors of shape {weight_list[0].shape}")
    print(f"Output: Single tensor of shape {out[0].shape}")
    print("✓ Single tensor output test passed\n")


def test_convenience_function():
    """Test convenience function."""
    print("=" * 80)
    print("Test 6: Using Convenience Function")
    print("=" * 80)

    num_groups = 4
    x_list = [torch.randn(64, 128, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]

    # Using convenience function
    out, _, _ = npu_ops_transformer.ops.grouped_matmul(x_list, weight_list)

    print(f"Input: {num_groups} tensors of shape {x_list[0].shape}")
    print(f"Weight: {num_groups} tensors of shape {weight_list[0].shape}")
    print(f"Output: {len(out)} tensors of shape {out[0].shape}")
    print("✓ Convenience function test passed\n")


def test_different_activations():
    """Test different activation functions."""
    print("=" * 80)
    print("Test 7: Different Activation Functions")
    print("=" * 80)

    num_groups = 4
    x_list = [torch.randn(64, 128, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    bias_list = [torch.randn(256, dtype=torch.float16).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    config = npu_ops_transformer.ops.GroupedMatmulConfig

    activations = [
        ("None", config.ACT_NONE),
        ("ReLU", config.ACT_RELU),
        ("GELU (tanh)", config.ACT_GELU_TANH),
        ("GELU (erf)", config.ACT_GELU_ERF),
        ("Fast GELU", config.ACT_FAST_GELU),
        ("SiLU", config.ACT_SILU),
    ]

    for act_name, act_type in activations:
        out, _, _ = gmm(x_list, weight_list, bias_list, act_type=act_type)
        print(f"  ✓ {act_name}: Output shape {out[0].shape}")

    print("✓ All activation functions test passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("GroupedMatmul Torch Extension Examples")
    print("=" * 80 + "\n")

    try:
        test_basic_grouped_matmul()
        test_with_bias_and_activation()
        test_quantized_mode()
        test_with_group_list()
        test_single_tensor_output()
        test_convenience_function()
        test_different_activations()

        print("=" * 80)
        print("All tests passed successfully! ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
