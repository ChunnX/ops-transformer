rm -rf build
rm -rf build_out/
rm -rf output/
bash build.sh --pkg --soc=ascend950 --ops=qbmm_reduce_scatter_add_rms_norm_cast -j16
cd build_out/
./cann-ops-transformer-custom_linux-x86_64.run --force
cd ..
bash build.sh --run_example qbmm_reduce_scatter_add_rms_norm_cast eager cust --vendor_name=custom > log.txt