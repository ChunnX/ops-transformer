conda activate hjz
source /data/huangjunzhe/Ascend/ascend-toolkit/set_env.sh
cd /data/jcy/demo/ops-transformer_1134
bash build.sh --pkg --ops=grouped_matmul
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=/data/jcy/custom
source /data/jcy/custom/vendors/custom_transformer/bin/set_env.bash
cd /data/jcy/demo/atk/gmm-atk-samples/atk_case/GroupedMatmul_A8W4
atk node --backend PYACLNN --devices 2 node --backend CPU task -c GroupedMatmul_A8W4.json --task accuracy -p GroupedMatmul_A8W4.py -wl=[0]
