# GroupedMatmul Torch Extension

## Overview

This directory contains the PyTorch extension implementation for the GroupedMatmul operator, providing a high-level Python interface to the ACLNN GroupedMatmulV5 API.

## Architecture

The implementation follows the same layered architecture as the MOE operators:

```
grouped_matmul/
├── grouped_matmul.py              # High-level wrapper class
├── grouped_matmul_v5.py           # PyTorch dispatcher layer
├── csrc/
│   └── grouped_matmul_v5.cpp      # C++ ACLNN wrapper
└── graph_convert/
    └── graph_convert_grouped_matmul_v5.py  # TorChair graph converter
```

## Features

- **Grouped Matrix Multiplication**: Performs y_i = x_i @ weight_i + bias_i for i = 1...g
- **M-axis and K-axis Grouping**: Supports different grouping strategies
- **Quantization Support**: INT8, INT4, FP8 quantization modes
- **Activation Functions**: ReLU, GELU (tanh/erf), Fast GELU, SiLU
- **Dual Mode**: Eager mode (PrivateUse1) and graph mode (TorChair)

## Usage

### Basic Usage

```python
import torch
import torch_npu
import npu_ops_transformer

# Create input tensors
x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(8)]
weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(8)]

# Method 1: Using high-level wrapper
gmm = npu_ops_transformer.ops.GroupedMatmul()
out, _, _ = gmm(x_list, weight_list)

# Method 2: Using convenience function
out, _, _ = npu_ops_transformer.ops.grouped_matmul(x_list, weight_list)

# Method 3: Using low-level API
out, _, _ = torch.ops.npu_ops_transformer.npu_grouped_matmul_v5(
    x_list, weight_list
)
```

### With Bias and Activation

```python
bias_list = [torch.randn(512, dtype=torch.float16).npu() for _ in range(8)]

gmm = npu_ops_transformer.ops.GroupedMatmul()
out, _, _ = gmm.forward_with_activation(
    x_list, weight_list, bias_list,
    act_type=npu_ops_transformer.ops.GroupedMatmulConfig.ACT_RELU
)
```

### Quantized Mode

```python
# INT8 quantized inputs
x_list = [torch.randint(-128, 127, (128, 256), dtype=torch.int8).npu() for _ in range(8)]
weight_list = [torch.randint(-128, 127, (256, 512), dtype=torch.int8).npu() for _ in range(8)]
scale_list = [torch.tensor([0.01], dtype=torch.float32).npu() for _ in range(8)]

gmm = npu_ops_transformer.ops.GroupedMatmul()
out, _, _ = gmm.forward_quantized(
    x_list, weight_list, scale_list
)
```

### With Group List

```python
# M-axis grouping with different M sizes per group
group_sizes = [128, 256, 512, 128, 256, 512, 128, 256]
group_list = npu_ops_transformer.ops.GroupedMatmul.create_group_list_cumsum(group_sizes)

gmm = npu_ops_transformer.ops.GroupedMatmul()
out, _, _ = gmm(
    x_list, weight_list,
    group_list=group_list.npu(),
    group_type=npu_ops_transformer.ops.GroupedMatmulConfig.GROUP_M_AXIS,
    group_list_type=npu_ops_transformer.ops.GroupedMatmulConfig.GROUP_LIST_CUMSUM
)
```

## API Reference

### GroupedMatmul Class

High-level wrapper for grouped matrix multiplication.

**Methods:**

- `__call__(x, weight, bias=None, ...)`: Main forward method
- `forward_with_activation(x, weight, bias=None, act_type=ACT_RELU, ...)`: Forward with activation
- `forward_quantized(x, weight, scale, offset=None, ...)`: Quantized forward
- `create_group_list_cumsum(group_sizes)`: Create cumsum group list
- `create_group_list_size(group_sizes)`: Create size-based group list

### GroupedMatmulConfig Class

Configuration constants for GroupedMatmul.

**Activation Types:**
- `ACT_NONE = 0`: No activation
- `ACT_RELU = 1`: ReLU activation
- `ACT_GELU_TANH = 2`: GELU with tanh approximation
- `ACT_GELU_ERF = 3`: GELU with erf
- `ACT_FAST_GELU = 4`: Fast GELU
- `ACT_SILU = 5`: SiLU (Swish) activation

**Split Item Modes:**
- `SPLIT_MULTI_TENSOR = 0`: Output as multiple tensors
- `SPLIT_SINGLE_TENSOR = 2`: Output as single tensor

**Group Types:**
- `GROUP_NONE = -1`: No grouping
- `GROUP_M_AXIS = 0`: Group along M axis
- `GROUP_K_AXIS = 2`: Group along K axis

**Group List Types:**
- `GROUP_LIST_CUMSUM = 0`: Group list contains cumulative sum
- `GROUP_LIST_SIZE = 1`: Group list contains size per group

### Low-level API

```python
torch.ops.npu_ops_transformer.npu_grouped_matmul_v5(
    x: List[Tensor],
    weight: List[Tensor],
    bias: Optional[List[Tensor]] = None,
    scale: Optional[List[Tensor]] = None,
    offset: Optional[List[Tensor]] = None,
    antiquant_scale: Optional[List[Tensor]] = None,
    antiquant_offset: Optional[List[Tensor]] = None,
    per_token_scale: Optional[List[Tensor]] = None,
    group_list: Optional[Tensor] = None,
    activation_input: Optional[List[Tensor]] = None,
    activation_quant_scale: Optional[List[Tensor]] = None,
    activation_quant_offset: Optional[List[Tensor]] = None,
    split_item: int = 0,
    group_type: int = -1,
    group_list_type: int = 0,
    act_type: int = 0,
    tuning_config: Optional[List[int]] = None
) -> Tuple[List[Tensor], Optional[List[Tensor]], Optional[List[Tensor]]]
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| x | List[Tensor] | Input tensor list [m_i, k_i] |
| weight | List[Tensor] | Weight tensor list [k_i, n_i] |
| bias | Optional[List[Tensor]] | Bias tensor list [n_i] |
| scale | Optional[List[Tensor]] | Quantization scale tensors |
| offset | Optional[List[Tensor]] | Quantization offset tensors |
| antiquant_scale | Optional[List[Tensor]] | Antiquant scale tensors |
| antiquant_offset | Optional[List[Tensor]] | Antiquant offset tensors |
| per_token_scale | Optional[List[Tensor]] | Per-token scale tensors |
| group_list | Optional[Tensor] | Group list tensor (cumsum or size) |
| split_item | int | Output mode (0/1: multi, 2/3: single) |
| group_type | int | Grouping axis (-1/0/2) |
| group_list_type | int | Group list type (0: cumsum, 1: size) |
| act_type | int | Activation function type (0-5) |
| tuning_config | Optional[List[int]] | Tuning configuration |

## Supported Data Types

- **Input/Weight**: FLOAT32, FLOAT16, BFLOAT16, INT8, INT4, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8
- **Bias**: FLOAT32, FLOAT16, BFLOAT16, INT32
- **Scale**: FLOAT32, BFLOAT16, UINT64, INT64, FLOAT8_E8M0
- **Output**: FLOAT32, FLOAT16, BFLOAT16, INT8, INT32

## Implementation Details

### Call Stack

```
User Code
  ↓
GroupedMatmul.__call__() (grouped_matmul.py)
  ↓
torch.ops.npu_ops_transformer.npu_grouped_matmul_v5()
  ↓
@impl(AS_LIBRARY, "npu_grouped_matmul_v5", "PrivateUse1")
  ↓
op_module.npu_grouped_matmul_v5() (JIT compiled)
  ↓
npu_grouped_matmul_v5() (grouped_matmul_v5.cpp)
  ↓
ACLNN_CMD(aclnnGroupedMatmulV5, ...)
  ↓
aclnnGroupedMatmulV5GetWorkspaceSize() + aclnnGroupedMatmulV5()
  ↓
NPU Hardware Execution
```

### Key Features

1. **JIT Compilation**: C++ code is compiled on first use via `torch.utils.cpp_extension.load()`
2. **Meta Function**: Supports shape inference for FakeTensor and torch.compile
3. **Zero-Copy**: Direct wrapping of PyTorch storage, no data copy
4. **Graph Mode**: TorChair converter for static graph compilation
5. **Error Handling**: Comprehensive parameter validation

## Performance Tips

1. **Use tuning_config**: Set expected token count per expert for optimal tiling
2. **Choose appropriate split_item**: Multi-tensor (0/1) vs single-tensor (2/3)
3. **Quantization**: Use INT8/INT4 for memory-bound workloads
4. **Group list**: Pre-compute and reuse group_list tensors

## Limitations

- N-axis grouping (group_type=1) is not supported yet
- Maximum 128 groups supported
- activation_input, activation_quant_scale, activation_quant_offset are reserved parameters

## See Also

- [GroupedMatmul Operator Documentation](../../gmm/grouped_matmul/README.md)
- [ACLNN API Reference](../../gmm/grouped_matmul/op_host/op_api/aclnn_grouped_matmul_v5.h)
- [MOE Operators](./deep_ep.py)
