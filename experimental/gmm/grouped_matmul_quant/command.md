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
