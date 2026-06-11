rm -rf build
rm -rf ${ASCEND_OPP_PATH}/vendors/*
bash build.sh --pkg --soc=ascend910b --ops=bsa_select_block_mask
bash build_out/*.run