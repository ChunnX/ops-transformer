# GroupedMatmul Torch Extension - Quick Start Guide

## Installation

```bash
cd D:/CANN/master/ops-transformer/torch_extension

# Install dependencies
pip install -r requirements.txt

# Build the wheel package
python -m build --wheel -n

# Install the package
pip install dist/*.whl --force-reinstall --no-deps
```

## Quick Start

### 1. Basic Usage

```python
import torch
import torch_npu
import npu_ops_transformer

# Create input tensors (8 groups)
x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(8)]
weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(8)]

# Perform grouped matmul
gmm = npu_ops_transformer.ops.GroupedMatmul()
out, _, _ = gmm(x_list, weight_list)

print(f"Output: {len(out)} tensors, each with shape {out[0].shape}")
```

### 2. With Activation Function

```python
bias_list = [torch.randn(512, dtype=torch.float16).npu() for _ in range(8)]

out, _, _ = gmm.forward_with_activation(
    x_list, weight_list, bias_list,
    act_type=npu_ops_transformer.ops.GroupedMatmulConfig.ACT_RELU
)
```

### 3. Quantized Mode

```python
# INT8 inputs
x_list = [torch.randint(-128, 127, (128, 256), dtype=torch.int8).npu() for _ in range(8)]
weight_list = [torch.randint(-128, 127, (256, 512), dtype=torch.int8).npu() for _ in range(8)]
scale_list = [torch.tensor([0.01], dtype=torch.float32).npu() for _ in range(8)]

out, _, _ = gmm.forward_quantized(x_list, weight_list, scale_list)
```

## Run Examples

```bash
cd D:/CANN/master/ops-transformer/torch_extension
python examples/test_grouped_matmul.py
```

## File Structure

```
torch_extension/
├── npu_ops_transformer/
│   ├── ops/
│   │   ├── grouped_matmul.py              # High-level API
│   │   ├── grouped_matmul_v5.py           # Dispatcher layer
│   │   ├── grouped_matmul_README.md       # Full documentation
│   │   ├── csrc/
│   │   │   └── grouped_matmul_v5.cpp      # C++ implementation
│   │   └── graph_convert/
│   │       └── graph_convert_grouped_matmul_v5.py
│   └── ...
├── examples/
│   └── test_grouped_matmul.py             # Example code
├── GROUPED_MATMUL_IMPLEMENTATION.md       # Implementation summary
└── ...
```

## API Levels

### Level 1: High-level Wrapper (Recommended)
```python
gmm = npu_ops_transformer.ops.GroupedMatmul()
out, _, _ = gmm(x_list, weight_list)
```

### Level 2: Convenience Function
```python
out, _, _ = npu_ops_transformer.ops.grouped_matmul(x_list, weight_list)
```

### Level 3: Low-level API
```python
out, _, _ = torch.ops.npu_ops_transformer.npu_grouped_matmul_v5(
    x_list, weight_list
)
```

## Configuration Options

```python
config = npu_ops_transformer.ops.GroupedMatmulConfig

# Activation types
config.ACT_NONE          # No activation
config.ACT_RELU          # ReLU
config.ACT_GELU_TANH     # GELU (tanh approximation)
config.ACT_GELU_ERF      # GELU (erf)
config.ACT_FAST_GELU     # Fast GELU
config.ACT_SILU          # SiLU (Swish)

# Output modes
config.SPLIT_MULTI_TENSOR   # Multiple output tensors
config.SPLIT_SINGLE_TENSOR  # Single concatenated output

# Grouping types
config.GROUP_NONE        # No grouping
config.GROUP_M_AXIS      # Group along M axis
config.GROUP_K_AXIS      # Group along K axis
```

## Documentation

- **Full API Documentation**: `npu_ops_transformer/ops/grouped_matmul_README.md`
- **Implementation Details**: `GROUPED_MATMUL_IMPLEMENTATION.md`
- **Architecture Analysis**: `torch_extension_analysis.md`

## Support

For issues or questions, refer to the comprehensive documentation files or check the example code.
