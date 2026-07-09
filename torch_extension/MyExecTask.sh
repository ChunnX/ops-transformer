cd ..
bash +x build.sh --pkg --ops=mixed_quant_flash_attn,mixed_quant_flash_attn_metadata --soc=ascend950
cd build_out/
./cann-ops-transformer-custom_linux-x86_64.run --install-path=/home/c00512306/MyInstallDir0701/
source /home/c00512306/MyInstallDir0701/vendors/custom_transformer/bin/set_env.bash
cd ../torch_extension/
python3 test.py > test.log 2>&1