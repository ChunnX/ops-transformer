rm -rf build
rm -rf ${ASCEND_OPP_PATH}/vendors/*
bash build.sh --pkg --soc=ascend910b --ops=bsa_select_block_mask
bash build_out/*.run
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}