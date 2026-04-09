cd /data/jcy/demo/ops-transformer_1134
bash build.sh --pkg --ops=grouped_matmul
./build_out/cann-ops-transformer-custom_linux-aarch64.run
export LD_LIBRARY_PATH=/data/huangjunzhe/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
# source /data/jcy/custom/vendors/custom_transformer/bin/set_env.bash
cd /data/jcy/atk/gmm-atk-samples/atk_case/GroupedMatmul_A8W4
atk node --backend PYACLNN --devices 1 node --backend CPU task -c GroupedMatmul_A8W4.json --task accuracy -p GroupedMatmul_A8W4.py -wl=[0] 2>&1 | tee log.log
