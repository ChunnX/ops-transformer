cd /data/jcy/new/ops-transformer_1134
bash build.sh --pkg --ops=grouped_matmul 2> log.txt                                                                                                                 
./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=/data/jcy/custom
source /data/jcy/custom/vendors/custom_transformer/bin/set_env.bash
cd /data/jcy/new/ops-transformer_1134/testA8W4
python test.py