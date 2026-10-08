# grouped_matmul_activation_weight_nz

`cann_ops_transformer.grouped_matmul_activation_weight_nz` is the PyTorch wrapper for
`aclnnGroupedMatmulActivationWeightNz`. It supports the Ascend 950PR/Ascend 950DT A4W4 WeightNZ path only.

## Signature

```python
grouped_matmul_activation_weight_nz(
    x, x_scale, group_list, weight, weight_scale, *, bias=None,
    dequant_mode=2, dequant_dtype=0, act_type=0, transpose_weight=False,
    group_list_type=0, tuning_config=None, swiglu_mode=0,
    clamp_limit=0.0, glu_alpha=1.0, glu_bias=0.0) -> Tensor
```

## Constraints

- `x` and the single tensor in `weight` use packed `torch.int8` or `torch.uint8` storage and are interpreted as
  `FLOAT4_E2M1`.
- `x_scale` and the single tensor in `weight_scale` use raw `torch.int8` or `torch.uint8` storage and are interpreted
  as `FLOAT8_E8M0`.
- `weight` is FRACTAL_NZ. `group_list` is `torch.int64`. `bias` must be `None`.
- The logical K and pre-SwiGLU N dimensions must both be divisible by 128. The returned tensor has BF16 dtype and
  shape `(M, N / 2)`.
- `dequant_mode=2`, `dequant_dtype=0`, and `act_type=0` are fixed for this interface. `tuning_config` must be `None`
  or an empty list.

`swiglu_mode=0` performs ordinary SwiGLU without clamp. `swiglu_mode=2` uses the front/back-half clamp behavior;
in that mode `clamp_limit` must be positive, while `glu_alpha` and `glu_bias` configure the activation and gate branch.
