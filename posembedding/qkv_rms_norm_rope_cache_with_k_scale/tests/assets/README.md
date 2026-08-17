# QkvRmsNormRopeCacheWithKScale TTK assets

This directory provides a self-contained NumPy golden and TestSpec adapters for
Kernel, ACLNN, and PTA eager execution.  It requires `numpy` and `ml_dtypes`;
ACLNN/PTA additionally require the project Torch/TorchNPU environment.

The four semantic cases cover RoPE `NTD->NTD`, RoPE `TND->TND`, RoPE
`TND->NTD`, and M-RoPE `TND->TND`.  The PTA CSV exercises both the functional
and in-place public interfaces.

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

Remove `--validate` to execute on Ascend 950.  The custom comparison checks
quantized exact rates, dequantized accuracy, touched cache rows, untouched cache
preservation, and PTA functional input-cache immutability.

CPU golden tests:

```bash
pytest -q $OP/tests/pytest/test_ttk_numpy_golden.py
```
