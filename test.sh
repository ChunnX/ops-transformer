#!/bin/bash
source /root/data_nvme0n1/jzh/toolkit/9.0.0/cann/set_env.sh
bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer_kl_loss
./build/cann-ops-transformer-custom_linux-aarch64.run
export LD_LIBRARY_PATH=/root/data_nvme0n1/jzh/toolkit/9.0.0/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
cd torch_extension/
# -n: non-isolated build (uses existing environment)
python -m build --wheel -n
python -m pip install dist/*.whl --force-reinstall --no-deps
python test_lightning_indexer_kl_loss.py