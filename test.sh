#!/bin/bash

source /data/jzh/toolkit/cann-9.0.0/cann/set_env.sh || { echo "Error: Failed to source set_env.sh"; exit 1; }
# bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer_kl_loss --op_debug_config ccec_g --op_debug_config "sanitizer" || { echo "Error: Failed to run build.sh"; exit 1; }
bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer_kl_loss --op_debug_config ccec_g || { echo "Error: Failed to run build.sh"; exit 1; }
./build/cann-ops-transformer-custom_linux-aarch64.run || { echo "Error: Failed to run installer"; exit 1; }
export LD_LIBRARY_PATH=/data/jzh/toolkit/cann-9.0.0/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
cd torch_extension/ || { echo "Error: Failed to change directory to torch_extension/"; exit 1; }
python -m build --wheel -n || { echo "Error: Failed to build wheel"; exit 1; }
python -m pip install dist/*.whl --force-reinstall --no-deps || { echo "Error: Failed to install wheel"; exit 1; }
python test_lightning_indexer_kl_loss.py > 1.log || { echo "Error: Failed to run test script"; exit 1; }
# msprof op --output=test --kernel-name=LightningIndexerKlLoss python test_lightning_indexer_kl_loss.py 
# msprof op simulator --soc-version=Ascend910B1 --core-id="0" --kernel-name="LightningIndexerKlLoss" --output=test --timeout=10 python test_lightning_indexer_kl_loss.py
echo "All steps completed successfully."