# GroupedMatmulQuant Ascend 910B Precision Test Commands

The following commands are intended to be run on an Ascend 910B server. Start in the `ops-transformer` repository
root. Adjust the CANN environment script path if CANN is installed in a non-default location.

## 1. Load the CANN environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('NPU available:', torch.npu.is_available())"
npu-smi info
```

## 2. Build and install the experimental operator package

Use an operator-specific vendor name to avoid colliding with another installed experimental package.

```bash
bash build.sh \
  --pkg \
  --experimental \
  --ops=grouped_matmul_quant \
  --soc=ascend910b \
  --vendor_name=gmm_quant

PACKAGE_PATH=$(find build_out -maxdepth 1 -type f -name 'cann-ops-transformer-gmm_quant_linux*.run' | head -n 1)
test -n "${PACKAGE_PATH}"
chmod +x "${PACKAGE_PATH}"
"${PACKAGE_PATH}" --install-path="${ASCEND_HOME_PATH}/opp"

source "${ASCEND_HOME_PATH}/opp/vendors/gmm_quant_transformer/bin/set_env.bash"
echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
```

## 3. Build and install the PyTorch registration extension

```bash
cd experimental/npu_ops_transformer_ext

python3 -m pip install pytest numpy
python3 -m pip install --no-build-isolation --no-deps -e .

cd ../..
```

Verify that the extension is importable and the operator is registered before running the precision cases:

```bash
python3 -c "import torch, torch_npu, npu_ops_transformer_ext; print('NPU available:', torch.npu.is_available()); print('operator:', torch.ops.npu_ops_transformer_ext.grouped_matmul_quant)"
```

## 4. Run the precision tests

```bash
cd experimental/gmm/grouped_matmul_quant/tests

set -o pipefail
python3 -m pytest -sv test_grouped_matmul_quant.py 2>&1 | tee grouped_matmul_quant_precision.log
```

The expected summary is:

```text
6 passed
```

The four NPU precision cases print `matched_ratio`, `max_abs_error`, and `max_rel_error`. The acceptance criteria are:

| Output type | Minimum matched ratio | Maximum absolute error |
| :--- | :---: | :---: |
| FP16 | 0.99 | 0.1 |
| BF16 | 0.99 | 1.0 |

## 5. Run only selected checks when diagnosing a failure

CPU-only INT4 packing and Golden checks:

```bash
python3 -m pytest -sv test_grouped_matmul_quant.py -k 'not precision'
```

Ascend NPU precision cases only:

```bash
python3 -m pytest -sv test_grouped_matmul_quant.py -k precision
```


j50061995@slave19:~/Projects/ops-transformer/experimental/gmm/grouped_matmul_quant/tests$ python3 -m pytest -sv test_grouped_matmul_quant.py 2>&1 | tee grouped_matmul_quant_precision.log
============================= test session starts ==============================
platform linux -- Python 3.10.12, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
cachedir: .pytest_cache
rootdir: /home/j50061995/Projects/ops-transformer/experimental/gmm/grouped_matmul_quant/tests
plugins: anyio-4.13.0
collecting ... collected 6 items

test_grouped_matmul_quant.py::test_int4_pack_round_trip PASSED
test_grouped_matmul_quant.py::test_golden_nonzero_offset_and_zero_token_expert PASSED
test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_single_tail_split_k] FAILED
test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[bf16_zero_token_expert] FAILED
test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_imbalanced_groups] FAILED
test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_non_split_k] FAILED

=================================== FAILURES ===================================
________ test_grouped_matmul_quant_precision[fp16_single_tail_split_k] _________

case = TestCase(name='fp16_single_tail_split_k', m=17, k=64, n=80, group_list=None, scale_group_size=32, dtype=torch.float16, seed=20260901, int4_boundary_values=True)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_grouped_matmul_quant_precision(case: TestCase):
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is not available")
>       op, op_path = _resolve_operator()

test_grouped_matmul_quant.py:156: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _resolve_operator():
        _import_registration_modules()
        configured = os.getenv("GROUPED_MATMUL_QUANT_OP")
        candidates = [configured] if configured else [
            "ops.vllm.grouped_matmul_quant",
            "ops.vllm_ascend.grouped_matmul_quant",
            "ops._C.grouped_matmul_quant",
            "ops.npu_ops_transformer_ext.grouped_matmul_quant",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return _resolve_attr(candidate), candidate
            except (AttributeError, RuntimeError):
                continue
>       pytest.fail(
            "GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set "
            "GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch)."
        )
E       Failed: GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch).

test_grouped_matmul_quant.py:79: Failed
_________ test_grouped_matmul_quant_precision[bf16_zero_token_expert] __________

case = TestCase(name='bf16_zero_token_expert', m=33, k=128, n=64, group_list=(0, 1, 33), scale_group_size=64, dtype=torch.bfloat16, seed=20260902, int4_boundary_values=False)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_grouped_matmul_quant_precision(case: TestCase):
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is not available")
>       op, op_path = _resolve_operator()

test_grouped_matmul_quant.py:156: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _resolve_operator():
        _import_registration_modules()
        configured = os.getenv("GROUPED_MATMUL_QUANT_OP")
        candidates = [configured] if configured else [
            "ops.vllm.grouped_matmul_quant",
            "ops.vllm_ascend.grouped_matmul_quant",
            "ops._C.grouped_matmul_quant",
            "ops.npu_ops_transformer_ext.grouped_matmul_quant",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return _resolve_attr(candidate), candidate
            except (AttributeError, RuntimeError):
                continue
>       pytest.fail(
            "GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set "
            "GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch)."
        )
E       Failed: GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch).

test_grouped_matmul_quant.py:79: Failed
_________ test_grouped_matmul_quant_precision[fp16_imbalanced_groups] __________

case = TestCase(name='fp16_imbalanced_groups', m=257, k=96, n=48, group_list=(1, 2, 257), scale_group_size=32, dtype=torch.float16, seed=20260903, int4_boundary_values=False)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_grouped_matmul_quant_precision(case: TestCase):
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is not available")
>       op, op_path = _resolve_operator()

test_grouped_matmul_quant.py:156: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _resolve_operator():
        _import_registration_modules()
        configured = os.getenv("GROUPED_MATMUL_QUANT_OP")
        candidates = [configured] if configured else [
            "ops.vllm.grouped_matmul_quant",
            "ops.vllm_ascend.grouped_matmul_quant",
            "ops._C.grouped_matmul_quant",
            "ops.npu_ops_transformer_ext.grouped_matmul_quant",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return _resolve_attr(candidate), candidate
            except (AttributeError, RuntimeError):
                continue
>       pytest.fail(
            "GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set "
            "GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch)."
        )
E       Failed: GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch).

test_grouped_matmul_quant.py:79: Failed
____________ test_grouped_matmul_quant_precision[fp16_non_split_k] _____________

case = TestCase(name='fp16_non_split_k', m=5760, k=32, n=1024, group_list=None, scale_group_size=32, dtype=torch.float16, seed=20260904, int4_boundary_values=False)

    @pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
    def test_grouped_matmul_quant_precision(case: TestCase):
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is not available")
>       op, op_path = _resolve_operator()

test_grouped_matmul_quant.py:156: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    def _resolve_operator():
        _import_registration_modules()
        configured = os.getenv("GROUPED_MATMUL_QUANT_OP")
        candidates = [configured] if configured else [
            "ops.vllm.grouped_matmul_quant",
            "ops.vllm_ascend.grouped_matmul_quant",
            "ops._C.grouped_matmul_quant",
            "ops.npu_ops_transformer_ext.grouped_matmul_quant",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return _resolve_attr(candidate), candidate
            except (AttributeError, RuntimeError):
                continue
>       pytest.fail(
            "GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set "
            "GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch)."
        )
E       Failed: GroupedMatmulQuant torch operator is not registered. Build/import its adapter, or set GROUPED_MATMUL_QUANT_MODULES and GROUPED_MATMUL_QUANT_OP (path relative to torch).

test_grouped_matmul_quant.py:79: Failed
=============================== warnings summary ===============================
../../../../../../.local/lib/python3.10/site-packages/torch/jit/_script.py:362: 14 warnings
  /home/j50061995/.local/lib/python3.10/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

test_grouped_matmul_quant.py:20
  /home/j50061995/Projects/ops-transformer/experimental/gmm/grouped_matmul_quant/tests/test_grouped_matmul_quant.py:20: PytestCollectionWarning: cannot collect test class 'TestCase' because it has a __init__ constructor (from: test_grouped_matmul_quant.py)
    @dataclass(frozen=True)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_single_tail_split_k]
FAILED test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[bf16_zero_token_expert]
FAILED test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_imbalanced_groups]
FAILED test_grouped_matmul_quant.py::test_grouped_matmul_quant_precision[fp16_non_split_k]
=================== 4 failed, 2 passed, 15 warnings in 5.72s ===================