ops="mhc_pre_sinkhorn"
custom_path="/data/hey/custom"
atk_path="/data/hey/git/gmm-atk-samples/atk_case/RopeWithSinCosCache/"

unset ASCEND_SLOG_PRINT_TO_STDOUT
unset ASCEND_GLOBAL_LOG_LEVEL

export TMPDIR=/home
################### 更改包的路径 ###################
clear;
rm -rf  /data/hey/custom/vendors/
source /data/hey/package/0701/ascend-toolkit/set_env.sh

# --soc=ascend910_93
bash build.sh --pkg --ops=${ops} 
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=${custom_path}
# ./build_out/cann-ops-transformer-custom_linux-aarch64.run 
# cp /data/hey/package/0701/cann-9.1.0/aarch64-linux/lib64/libopapi_transformer.so /data/hey/package/0701/cann-9.1.0/aarch64-linux/lib64/libopapi_transformer.so.bak
# \cp /data/hey/custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so /data/hey/package/0701/cann-9.1.0/aarch64-linux/lib64/libopapi_transformer.so

source /data/hey/package/0701/ascend-toolkit/set_env.sh
source /data/hey/custom/vendors/custom_transformer/bin/set_env.bash
source ${custom_path}/vendors/custom_transformer/bin/set_env.bash

# bash build.sh --ophost_test --ops=mhc_pre_sinkhorn
# bash build.sh --opkernel_test --ops=mhc_pre_sinkhorn
# bash build.sh --run_example mhc_pre_sinkhorn eager cust
# cd ${atk_path}


# AB
# atk node --backend pyaclnn --devices 2 node --backend cpu task -c  qwen3_vl/all_aclnn_rope_with_sin_cos_cache_16_24_24.json --task accuracy -p RopeWithSinCosCache.py -wl [0] -cp

# atk node --backend pyaclnn --devices 2 node --backend cpu task -c all_aclnn_mhc_pre_sinkhorn.json --task accuracy -p MhcPreSinkhorn.py -e 1 

# bash run_atk.sh -r ${atk_case_name}