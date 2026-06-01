source /root/miniconda3/bin/activate
conda activate py311_zwh_sas_v2
cd /mnt/z00957671/0513_qsas_pef_debug/bf16_getphyaddr_cube_2/ops-transformer
source /home/z00957671/cann_dev/geneva_0309/cann-9.0.0/bin/setenv.bash
export http_proxy="http://p_atlas:proxy%40123@proxy.huawei.com:8080"
export https_proxy="http://p_atlas:proxy%40123@proxy.huawei.com:8080"
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

bash build.sh --pkg --ops=kv_quant_sparse_attn_sharedkv,kv_quant_sparse_attn_sharedkv_metadata --experimental --soc=ascend950 2>&1 | tee res.log
./build_out/cann-ops-transformer-custom_linux-x86_64.run --install-path=/mnt/z00957671/0513_qsas_pef_debug/bf16_getphyaddr_cube_2/

source /mnt/z00957671/0513_qsas_pef_debug/bf16_getphyaddr_cube_2/vendors/custom_transformer/bin/set_env.bash
cd /mnt/z00957671/0513_qsas_pef_debug/bf16_getphyaddr_cube_2/ops-transformer/experimental/attention/kv_quant_sparse_attn_sharedkv/tests/pytest
msprof python3 -m pytest -rA -s test_kv_quant_sparse_attn_sharedkv_batch.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning 2>&1 | tee res.log