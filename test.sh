ops="mhc_pre_sinkhorn"
# atk_case_name="MoeGatingTopKSoftmax"

custom_path="/data/hey/heyCustom/"
atk_path="/data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/ST"

###################
rm -rf /data/hey/git/ops-transformer/build
rm -rf /data/hey/heyCustom/vendors/

###########编译########
clear
source /data/hey/package/0403/ascend-toolkit/set_env.sh
bash build.sh --pkg --ops=${ops}

###########安装########
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=${custom_path}
# cp /data/hey/package/0403/cann-9.0.0/arm64-linux/lib64/libopapi_transformer.so /data/hey/package/0403/cann-9.0.0/arm64-linux/lib64/libopapi_transformer.so.bak
\cp /data/hey/heyCustom/vendors/custom_transformer/op_api/lib/libcust_opapi.so  /data/hey/package/0403/cann-9.0.0/arm64-linux/lib64/libopapi_transformer.so
# \cp /data/hey/package/0403/cann/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so  /data/hey/package/0403/cann-9.0.0/arm64-linux/lib64/libopapi_transformer.so

###########环境变量########
source ${custom_path}/vendors/custom_transformer/bin/set_env.bash
source /data/hey/package/0403/ascend-toolkit/set_env.sh
source /data/hey/heyCustom/vendors/custom_transformer/bin/set_env.bash

###########测试执行########
cd ${atk_path}
atk node --backend pyaclnn --devices 6 node --backend cpu task -c *.json --task accuracy -p MhcPreSinkhorn.py -e 1 --save_data output 
sleep 2
python /data/hey/git/compare_all_outputs.py

