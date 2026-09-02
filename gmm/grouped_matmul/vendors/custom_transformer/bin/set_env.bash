#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/data/azyuli/ops-transformer/gmm/grouped_matmul/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/data/azyuli/ops-transformer/gmm/grouped_matmul/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
