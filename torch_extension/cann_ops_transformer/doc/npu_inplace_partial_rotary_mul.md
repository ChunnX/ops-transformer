# npu_inplace_partial_rotary_mul

## Operator Name

`npu_inplace_partial_rotary_mul`

## Schema

```
inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, str rotary_mode, int[2] partial_slice) -> ()
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| x | Tensor | Input tensor to be rotated in-place (4D, BSND layout). Modified in-place. |
| r1 | Tensor | Rotary embedding tensor for cos component. |
| r2 | Tensor | Rotary embedding tensor for sin component. |
| rotary_mode | str | Rotation mode. See supported modes below. |
| partial_slice | int[2] | Start and end indices for partial rotation along the head dimension. |

## Supported Modes

| Mode | Value | Description |
|------|-------|-------------|
| `half` | 0 | Rotate only the first half of each head. |
| `interleave` | 1 | Apply rotation in an interleaved pattern. |
| `quarter` | 2 | Rotate only the first quarter of each head. |
| `interleave-half` | 3 | Apply interleaved-half rotation pattern. |

## Usage Example

```python
import torch
import torch_npu
from cann_ops_transformer.ops import inplace_partial_rotary_mul

x = torch.randn(2, 32, 8, 128, device='npu', dtype=torch.float16)
r1 = torch.randn(2, 32, 8, 64, device='npu', dtype=torch.float16)
r2 = torch.randn(2, 32, 8, 64, device='npu', dtype=torch.float16)

inplace_partial_rotary_mul(x, r1, r2, rotary_mode="half", partial_slice=[0, 64])
```
