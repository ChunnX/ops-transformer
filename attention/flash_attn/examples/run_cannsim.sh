#!/bin/bash
export LD_LIBRARY_PATH=/mnt/workspace/w00644563/ops-transformer_flash_attn/build:/mnt/workspace/w00644563/ops-transformer_flash_attn/packages/vendors/custom_transformer/op_api/lib:/home/developer/Ascend/cann-9.0.0/lib64:$LD_LIBRARY_PATH
./test_flash_attn --B 1 --N1 4 --N2 4 --S1 64 --S2 256 --D 128 --mask 1 --name aclnn_test --dump --verbose