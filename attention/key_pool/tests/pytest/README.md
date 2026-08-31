# KeyPool Tests

This directory contains the maintained KeyPool validation entry points. The
tests use the public `torch_npu.key_pool` interface and the independent CPU
Golden implementation. Legacy parameter-combination, batch data conversion,
profiling, and ad-hoc JIT loader scripts are intentionally not part of the
release tree.

## Maintained Files

| File | Purpose |
| --- | --- |
| `key_pool_golden.py` | Scalar Golden and independent Torch Oracle |
| `key_pool_test_utils.py` | Deterministic input builders and Golden checks |
| `key_pool_public_loader.py` | Loads only the installed KeyPool public module |
| `test_key_pool_golden.py` | CPU Golden and cache-boundary unit tests |
| `test_key_pool_step01_baseline.py` | Baseline BSH/TH, history, ratio, and block cases |
| `test_key_pool_public_interface.py` | Public signature, schema, Meta, and rejection checks |
| `test_key_pool_public_npu.py` | Direct NPU functional and precision regression |
| `test_key_pool_precision_standalone.py` | Standalone NPU-versus-Golden precision check |
| `test_key_pool_layout_equivalence_npu.py` | BSH/TH and batch-position equivalence |
| `test_key_pool_chunk_equivalence_npu.py` | Single-call versus chunked-call equivalence |
| `test_key_pool_determinism_npu.py` | Repeatability and cache determinism |
| `key_pool_long_sequence_npu.py` | Long-sequence regression, including 32K cases |
| `pytest.ini` | Pytest markers and local test configuration |

## Coverage

The maintained tests cover:

```text
BSH [B, S, H]
TH  [T, H] with cu_seqlens
K/Gate projection
optional LayerNorm
state-cache update and Block Table addressing
Gate + APE pooling
all supported compression ratios
fixed-capacity pooled output and zero tail
non-empty RoPE rejection
```

RoPE remains an interface placeholder. Non-empty `cos`/`sin` inputs must be
rejected explicitly and must not be silently ignored.

The NPU scripts are intended to run after the CANN, custom OPP, and wheel
environments have been loaded. They do not require `torchair`.

Typical commands:

```bash
python3 -m pytest -q test_key_pool_golden.py test_key_pool_step01_baseline.py
python3 -m pytest -q test_key_pool_public_interface.py
python3 test_key_pool_public_npu.py --case all
python3 test_key_pool_public_npu.py --case all --norm-on
python3 key_pool_long_sequence_npu.py --norm-off
python3 key_pool_long_sequence_npu.py --norm-on
```
