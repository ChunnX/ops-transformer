ops="sparse_attn_sharedkv,sparse_attn_sharedkv_metadata"

custom_path="/data/hey/sas_custom/"
atk_path="/data/hey/git/gmm-atk-samples/atk_case/MhcPreSinkhorn"

###################
rm -rf /data/hey/git/ops-transformer/build
rm -rf /data/hey/sas_custom/vendors/

###########编译########
clear
source /data/hey/package/0526/ascend-toolkit/set_env.sh
bash build.sh --pkg --ops=${ops}

###########安装########
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=${custom_path}
# cp /data/hey/package/0526/cann-9.1.0/arm64-linux/lib64/libopapi_transformer.so /data/hey/package/0526/cann-9.1.0/arm64-linux/lib64/libopapi_transformer.so.bak
# \cp /data/hey/heyCustom/vendors/custom_transformer/op_api/lib/libcust_opapi.so  /data/hey/package/0526/cann-9.1.0/arm64-linux/lib64/libopapi_transformer.so
# \cp /data/hey/package/0526/cann/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so  /data/hey/package/0526/cann-9.0.0/arm64-linux/lib64/libopapi_transformer.so

###########环境变量########
source ${custom_path}/vendors/custom_transformer/bin/set_env.bash
source /data/hey/package/0526/ascend-toolkit/set_env.sh
source /data/hey/sas_custom/vendors/custom_transformer/bin/set_env.bash
# bash build.sh --ophost_test --ops=sparse_attn_sharedkv
# bash build.sh --run_example  sparse_attn_sharedkv  eager
###########测试执行########
cd ${atk_path}
# atk node --backend pyaclnn --devices 5 node --backend cpu task -c all_aclnn_mhc_pre_sinkhorn.json --task accuracy -p MhcPreSinkhorn.py -e 1  --save_data output 
# atk node --backend pyaclnn --devices 3 node --backend npu --devices 2  task -c all_aclnn_mhc_pre_sinkhorn.json --task performance_device -p MhcPreSinkhorn.py   -e 1 --save_data profile
# atk node --backend pyaclnn --devices 5 node --backend cpu task -c result/aclnn_mhc_pre_sinkhorn/json/all_aclnn_mhc_pre_sinkhorn.json --task accuracy -p MhcPreSinkhorn.py  -wl [6] --save_data output 
# python /data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/ST/compare_all_outputs.py


