# TurboQuantSparseFlashAttention compatibility interface

The standard interface is `cann_ops_transformer.turbo_quant_sparse_flash_attention`.
This optional Python package forwards the old `custom_ops` entry points to that implementation;
it does not build or load a second native operator. The ACLNN name and argument order are unchanged.

## Build and install

```bash
# First build/install the operator package and the standard wheel as described in ../README.md.
cd attention/turbo_quant_sparse_flash_attention/torch_ops_extension
bash build_and_install.sh
```

## Use

```python
import torch
import custom_ops  # noqa: F401

result = torch.ops.custom.npu_turbo_quant_sparse_flash_attention(...)
```

Importing `custom_ops` also mounts the operation as `torch_npu.npu_turbo_quant_sparse_flash_attention`.

Restart the Python process after upgrading from the old native extension. Do not load both versions
into the same process. New applications should use the standard interface directly.
