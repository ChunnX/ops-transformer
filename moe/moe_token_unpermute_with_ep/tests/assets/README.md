# Golden reference contract

`golden.py` follows the route-specific TTK plugin layout used by the standard
Sinh template:

- `moe_token_unpermute_with_ep` → raw Kernel spec;
- `aclnnMoeTokenUnpermuteWithEp` → ACLNN spec;
- `MoeTokenUnpermuteWithEp` → GEIR spec.

Kernel and GEIR receive NumPy values at the framework boundary and use only
TTK's `numpy_to_torch_tensor` / `torch_to_numpy_tensor` helpers for conversion.
ACLNN receives Torch tensors directly. Every arithmetic operation is a Torch
primitive (`index_select`, masks, multiply and sum); the computation chain does
not use NumPy and never invokes the tested NPU operator.

The Golden and `third_party` classes are separate implementations. The latter
keeps device placement and has no unrelated device transfer in its timed path.
FP16, BF16 and FP32 are all declared as cross-check L0 because this operator has
no same-name Torch or TensorFlow competitor. Tolerances are FP16/BF16
`rtol=atol=1e-3` and FP32 `rtol=atol=1e-4`.

`test_golden_cpu.py` checks fixed vectors for present/absent probabilities,
non-empty EP ranges, outside-range zero contribution, empty-range fallback and
mixed dtypes before any NPU execution.
