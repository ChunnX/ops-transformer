
# Correctness check (pytest)

This directory ships a Python correctness suite for `block_sparse_attention`.
It reuses the workload helpers from the sibling `../../benchmark/util.py`. The
benchmark in `../../benchmark/` only checks correctness at sparsity 0 by default
(the full dense reference mask costs ~42 GB at `S=118 806` and OOMs), so this
pytest suite is the end-to-end correctness gate at smaller sequence lengths.

```bash
# After the kernel + torch_extension are installed (see Prerequisites below).
pytest test_attn.py test_lse.py test_joint.py -v
```

| File | Coverage | Reference |
|---|---|---|
| `test_attn.py` | `attention_out` at `H ∈ {1, 2, 3, 4}`, `S=24000`, `D=128`, BF16, BNSD; two frame settings (no frame and `FrameWidths(2,1,3,1)`); 10 sparsity levels (0.0..0.9). 80 cases total. | `npu_fusion_attention` (PFA): `sparse_mode=0` at sparsity 0, or `sparse_mode=1` with the per-token `pfa_atten_mask` at sparsity > 0. |
| `test_lse.py` | `softmax_lse` output: dense LSE vs FIAS, sparse `H=1` LSE vs FIAS (definitional gap), multi-head sparse LSE vs Python float32 reference, zero-input LSE = `log(S)`, shape/dtype checks (`[B, H, S, 1]` 4-D float32), empty-tensor return when LSE is disabled. Covers bfloat16 + float16. | `npu_fused_infer_attention_score` (FIAS) + a Python float32 implementation (`_lse_reference_sparse`). |
| `test_joint.py` | Joint validation of `attention_out` + `softmax_lse` returned from a single `softmax_lse_flag=True` call: dense (BF16 + FP16) vs FIAS, and sparse `H=1` BF16 (`attention_out` vs PFA, LSE vs Python reference). | FIAS / PFA / Python float32 reference. |

### Tolerances and known gaps

- **`attention_out`**: strictly within `rtol=0.005, atol=0.001` at every tested sparsity.
- **dense `softmax_lse`**: strictly within `rtol=1e-3, atol=1e-4`.
- **sparse `softmax_lse` vs FIAS**: drifts by ~0.06–0.16 at sparsity 0.5 / 0.8 / 0.9 — this is a definitional gap (FIAS `sparse_mode=1` sums over different positions than this kernel). `test_lse.py` records the expected band as `SPARSE_FIAS_LSE_ATOL=0.25`.
- **sparse `softmax_lse` vs Python reference**: strictly within `atol=5e-2` (BF16 matmul accumulation budget). Passing this is what proves the kernel's LSE is correct under its own definition.

## Prerequisites

1. The `block_sparse_attention` kernel + aclnn library must be built and
   installed into `$ASCEND_OPP_PATH/vendors/custom_transformer/op_api/`:
   ```bash
   # adjust --soc to your device: ascend910b (A2), ascend950 (A5), ascend_910_93 (A3)
   bash build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=block_sparse_attention
   ./build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
   ```
2. The `cann_ops_transformer` torch extension must be built + installed
   (it provides the JIT-compiled Python entry point
   `cann_ops_transformer.ops.block_sparse_attention`):
   ```bash
   cd torch_extension
   python3 -m pip install -r requirements.txt
   python3 -m build --wheel -n
   python3 -m pip install dist/*.whl --force-reinstall --no-deps
   ```
   First `import cann_ops_transformer` triggers a one-time ninja JIT compile
   of the C++ wrapper (`ops/csrc/block_sparse_attention.cpp`). See
   [`torch_extension/README.md`](../../../../torch_extension/README.md) for full
   build instructions.
