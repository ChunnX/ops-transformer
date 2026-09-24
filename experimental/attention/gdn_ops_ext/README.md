# GDN PyTorch extension

This package builds the existing `fused_gdn_gating` and
`chunk_gated_delta_rule_compute_wy` Torch bindings into one independent wheel.
The underlying ACLNN operator packages must be installed first.

## Build and install

Run from an ops-transformer source checkout:

```bash
cd experimental/attention/gdn_ops_ext
bash build_and_install.sh
```

## Verify registration

Run Python outside the source directory:

```python
import torch
import torch_npu
import gdn_ops_ext

assert hasattr(torch.ops.gdn_ops_ext, "fused_gdn_gating")
assert hasattr(torch.ops.gdn_ops_ext, "chunk_gated_delta_rule_compute_wy")
```

The operators are available through:

```python
torch.ops.gdn_ops_ext.fused_gdn_gating
torch.ops.gdn_ops_ext.chunk_gated_delta_rule_compute_wy
```
