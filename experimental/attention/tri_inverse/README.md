# tri_inverse

`tri_inverse` extracts the 64x64 strict-lower triangular inverse stage from
`attention/chunk_gated_delta_rule`.

## Contract

First version:

```text
cc_in  : BF16, ND, [tile_num, 64, 64]
cc_out : BF16, ND, [tile_num, 64, 64]
```

`cc_in` is not raw `KKT`. It is the CC tile after the original Stage1
preprocessing outside this operator: beta, gamma/decay and mask have already
been applied. The input is expected to contain valid strict-lower data, with
diag and upper values set to zero by the caller or test generator.

The kernel keeps the original inverse sequence:

```text
Muls(-1.0) -> InverseAIV(top/bottom 32x32) -> AttnInverseMMCompute -> InverseAICProcess
```

The first version intentionally does not support raw query/key/beta/g inputs,
dynamic chunk sizes, non-BF16 tensors, or inputs that are already negated.

## Kernel Variants

The baseline and MixInverse implementations are physically separated:

```text
op_kernel/tri_inverse.h     : current passing baseline
op_kernel/tri_inverse_mix.h : first MixInverse path, behavior-equivalent to baseline
```

The exported kernel still instantiates `TriInverseOp`; the selected
header is included at compile time in `op_kernel/tri_inverse.cpp`:

```text
TRI_INVERSE_USE_MIX_INVERSE=1 -> include tri_inverse_mix.h
TRI_INVERSE_USE_MIX_INVERSE=0 -> include tri_inverse.h
```

The default is `MixInverseOp`. This first MixInverse version is intentionally a
separate header with the same inverse math and dataflow, so later optimizations
can change one dataflow at a time without mixing with the baseline file.

## Golden

The pytest golden follows the same 32x32 recurrence as `InverseAIV` and does
not call `torch.inverse`:

```text
inv[i, :i] = -A[i, :i] @ inv[:i, :i]
```

The 64x64 result is composed as:

```text
top_left  = inv(A[0:32, 0:32])
right_down = inv(A[32:64, 32:64])
left_down = right_down @ ((-A[32:64, 0:32]) @ top_left)
```

## Build

From the repository root, choose the SoC that matches the visible device:

```bash
bash build.sh --pkg --experimental --soc=ascend910b --ops=tri_inverse -j16
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tri_inverse -j16
```

After installing the generated package, source the custom package environment
before running Python tests:

```bash
build/cann-ops-transformer-custom_linux-aarch64.run --quiet
source ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash
```

## Test

```bash
cd experimental/attention/tri_inverse/tests/pytest
bash test_run.sh single
```

The pytest matrix covers 10 chunk-aligned logical shapes up to `10240` and 5
unaligned logical shapes up to `10241`:

```text
shape = 64/128/256/512/1024/2048/4096/6144/8192/10240
shape = 65/257/1025/4097/10241
seed = 0/21
```

The operator input keeps the compact tile form `[ceil(shape / 64), 64, 64]`;
`shape` is only the user-facing logical coverage used by tests and profiling.

## Profiling

```bash
cd experimental/attention/tri_inverse/tests/pytest
bash test_run.sh profile
```

The profiling summary is written to:

```text
prof_out/tri_inverse_profile/tri_inverse_profile.csv
```

Summary columns keep the CANN `kernel_details.csv` field names:

```text
shape,Duration(us),aic_mac_time(us),aiv_vec_time(us),aic_mte2_time(us),aic_mac_ratio
```

## Consistency Audit

The kernel keeps the original `InverseAIV`, `AttnInverseMMCompute`, and
`InverseAICProcess` statement sequence. The intentional differences are only:

- `cc_in` replaces the original preprocessed Stage1 `attnUbFloat_` source.
- `cc_out` replaces the original Stage1 `attnWsGm_` in-place GM workspace.
- beta, gamma, mask, Stage2, and Stage3 code is removed from this operator.
