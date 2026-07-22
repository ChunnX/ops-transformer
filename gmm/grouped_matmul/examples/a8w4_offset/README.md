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
