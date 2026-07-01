#!/usr/bin/env bash
set -e

platform=${1:-a2}
case "${platform}" in
    a2|A2)
        soc=ascend910b
        ;;
    a5|A5)
        soc=ascend950
        ;;
    *)
        echo "Usage: bash run_build_bsa_mask.sh [a2|a5]"
        echo "  a2: build with --soc=ascend910b (default)"
        echo "  a5: build with --soc=ascend950"
        exit 1
        ;;
esac

rm -rf build
rm -rf ${ASCEND_OPP_PATH}/vendors/*
bash build.sh --pkg --soc=${soc} --ops=bsa_select_block_mask
bash build_out/*.run
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
