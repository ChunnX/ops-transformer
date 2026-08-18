# QkvRmsNormRopeCacheWithKScale TTK assets

This directory provides a self-contained NumPy golden and TestSpec adapters for
Kernel, ACLNN, and PTA eager execution.  It requires `numpy` and `ml_dtypes`;
ACLNN/PTA additionally require the project Torch/TorchNPU environment.

The four semantic cases cover RoPE `NTD->NTD`, RoPE `TND->TND`, RoPE
`TND->NTD`, and M-RoPE `TND->TND`.  The PTA CSV exercises both the functional
and in-place public interfaces.  The RoPE `TND->TND` case uses physically
non-contiguous cache views on ACLNN and both PTA paths: K/V have `D+8` padded
token strides, K scale has a matching `1+8` padded token stride, and the logical
slot mapping remains non-monotonic and cross-block.

The output precision contract follows TTK's `rtol`/`ptol` interpretation.  BF16
uses `rtol=0.001` with at most `0.001` over-tolerance points; FP8 uses
`rtol=0.01` with at most `0.01` over-tolerance points.  INT8 has maximum
absolute error `<=1`.  Scale FP32 uses `rtol=0.001` with at most `0.001`
over-tolerance points.
`atol=1e-8` is used for these floating comparisons.  Untouched cache slots and
functional input caches remain byte-exact checks.

```bash
TTK=/path/to/ops-test-kit
OP=/path/to/ops-transformer/posembedding/qkv_rms_norm_rope_cache_with_k_scale
ASSETS=$OP/tests/assets
CASES=$OP/tests/st/arch35

cd $TTK
python3 -m ttk kernel -i $CASES/ttk_kernel_qkv_rms_norm_rope_cache_with_k_scale.csv \
  --plugin $ASSETS --validate
python3 -m ttk aclnn -i $CASES/ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale.csv \
  --plugin $ASSETS --validate
PYTHONPATH=$ASSETS:/path/to/ops-transformer/torch_extension:$PYTHONPATH \
python3 -m ttk e2e -i $CASES/ttk_pta_qkv_rms_norm_rope_cache_with_k_scale.csv \
  --plugin $ASSETS --validate
```

Remove `--validate` to execute on Ascend 950.  The custom comparison applies the
precision contract above to the touched outputs, and also checks untouched cache
preservation plus PTA functional input-cache immutability.

CPU golden tests:

```bash
pytest -q $OP/tests/pytest/test_ttk_numpy_golden.py
```

Full-800 random-stride matrix (stride only; no tensor-view offset field):

```bash
python3 $OP/tests/scripts/generate_ttk_full_800_cases.py

python3 -m ttk aclnn \
  -i $CASES/ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_full_800_random_stride.csv \
  --plugin $ASSETS --validate

PYTHONPATH=$ASSETS:/path/to/ops-transformer/torch_extension:$PYTHONPATH \
python3 -m ttk e2e \
  -i $CASES/ttk_pta_qkv_rms_norm_rope_cache_with_k_scale_full_800_random_stride.csv \
  --plugin $ASSETS --process-count 1 --seed 20260817 --proc-no-reuse \
  --output $OP/tests/reports/ttk_pta_full_800_random_stride_result.csv
```

The generator reproduces the previous `8 heads * 4 modes * 25 T` semantic
matrix.  K/V share a randomly selected legal row padding from
`{8,16,24,32}`; scale padding is selected independently from the same set.
Large inputs and Golden/compare work are chunked to keep peak host memory
bounded.
