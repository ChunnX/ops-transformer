# A8W4 offset tests

CPU reference tests:

```bash
bash run_offset_null.sh
bash run_offset_non_null.sh
```

Add `--run-npu` to invoke `torch_npu.npu_grouped_matmul` as well:

```bash
bash run_offset_null.sh --run-npu
bash run_offset_non_null.sh --run-npu
```

Build and run the native ACLNN cases (one isolated ACL runtime process each):

```bash
bash build_and_run_aclnn.sh null
bash build_and_run_aclnn.sh non-null
```

Build, install and run the custom package against the A8W4 V5 per-group
kernel (`scale=[E,K/256,N]`, `offset=null`):

```bash
A8W4_DEQUANT_MODE=pergroup A8W4_API=V5 \
  A8W4_E=2 A8W4_M=16 A8W4_N=32 A8W4_K=512 \
  bash build_install_run_custom.sh all
```

`A8W4_DEQUANT_MODE` accepts `pergroup` and `perchannel`; `perchannel` is
the default and preserves the previous non-null-offset behavior.

Use the WeightNz ACLNN entry with an NZ weight:

```bash
A8W4_API=WeightNz A8W4_WEIGHT_NZ=1 bash build_install_run_custom.sh run-only
```

For matrix regression, select both the API and weight format:

```bash
A8W4_API=WeightNz A8W4_WEIGHT_FORMAT=NZ bash run_case_matrix.sh
```

Run the previous V5 ND/NZ board calls and the new WeightNz/NZ call together.
The wrapper rejects cards 1/5/6/7 and defaults to card 0:

```bash
A8W4_VISIBLE_DEVICE=0 A8W4_RANDOM_ROUNDS=20 bash run_board_regression.sh
```

Each route writes a separate `summary.tsv`, log directory, and golden directory
under `build_out/a8w4_board_regression`. A run counts as a board result only
after ACL initialization, kernel launch, stream synchronization, D2H, and the
CPU golden comparison all succeed. `build_install_run_custom.sh` checks for the
real driver `libascend_hal.so`; CANN `devlib` compile stubs are intentionally not
used as a runtime fallback.

The null-offset case uses `scale=[E,K/256,N]`. The non-null case uses
`scale=[E,1,N]` and `offset=[E,1,N]`, where offset is the fused
`antiquantOffset * scale` value required by the A8W4 API.

The CPU reference intentionally stores logical INT4 values in INT8 tensors.
Although recent PyTorch releases declare `torch.int4`, CPU conversion and
matmul kernels for that dtype are not implemented. The optional NPU path uses
`torch_npu.npu_convert_weight_to_int4pack`, because A3 requires its interleaved
packing layout rather than a simple low-nibble-first pack.

`bias=[E,N]` in these A8W4 tests is the mandatory offline MSD correction
`reduce_sum(8 * weight * scale, K)`. It cancels the kernel's internal `x - 8`
transformation and is not an application/model bias added to the final result.
