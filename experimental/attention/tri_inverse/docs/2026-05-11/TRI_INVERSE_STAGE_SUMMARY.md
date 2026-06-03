# tri_inverse Stage Summary

Date: 2026-05-11

## Result

`tri_inverse` is an experimental standalone operator for the 64x64
strict-lower triangular inverse stage extracted from `chunk_gated_delta_rule`
Stage1.

Current verified interface:

```text
cc_in  : BF16, ND, [tile_num, 64, 64]
cc_out : BF16, ND, [tile_num, 64, 64]
tile_num = ceil(shape / 64)
```

`cc_in` is the preprocessed CC tile before the kernel-side `Muls(-1.0)`.
It is not raw `KKT`; beta, gamma/decay, and mask are outside this operator.

## Kernel Files

```text
op_kernel/tri_inverse.h      # passing baseline
op_kernel/tri_inverse_mix.h  # MixInverse v0, behavior-equivalent baseline
op_kernel/tri_inverse.cpp    # TRI_INVERSE_USE_MIX_INVERSE include switch
```

Default:

```text
TRI_INVERSE_USE_MIX_INVERSE=1
```

The exported kernel still instantiates `TriInverseOp`; `tri_inverse_mix.h`
aliases that name to `MixInverseOp`.

## Core Flow

The first version keeps the original Stage1 inverse sequence:

```text
Muls(-1.0)
-> InverseAIV(top/bottom 32x32)
-> AttnInverseMMCompute
-> InverseAICProcess
```

Lower-left compose is:

```text
left_down = right_down @ ((-A[32:64, 0:32]) @ top_left)
```

Equivalently:

```text
outLower = bottomInv @ lowerWork @ topInv
lowerWork = -LOW
```

## Golden And Tests

The pytest golden does not call `torch.inverse`. It follows the same 32x32
recurrence as `InverseAIV`:

```text
inv[i, :i] = -A[i, :i] @ inv[:i, :i]
```

Precision coverage:

```text
shape = 64/128/256/512/1024/2048/4096/6144/8192/10240
shape = 65/257/1025/4097/10241
seed = 0/21
```

Latest validation:

```bash
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tri_inverse -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet
cd experimental/attention/tri_inverse/tests/pytest
bash test_run.sh single
```

Result:

```text
30 passed
```

## Profiling

Profiling entry:

```bash
cd experimental/attention/tri_inverse/tests/pytest
bash test_run.sh profile
```

CSV path:

```text
tests/pytest/prof_out/tri_inverse_profile/tri_inverse_profile.csv
```

CSV keeps CANN `kernel_details.csv` field names:

```text
shape,Duration(us),aic_mac_time(us),aiv_vec_time(us),aic_mte2_time(us),aic_mac_ratio
```

## Not In Scope

- raw `KKT` input
- kernel-side beta/gamma/mask preprocessing
- dynamic chunk size
- non-BF16 input/output
- changing host, tiling, golden, PyTorch wrapper, or Python call interface
