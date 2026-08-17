# MXFP4 BSA ATK 工程

# 清空
rm -rf ./result/op_block_sparse_attention/json/all_block_sparse_attention.json
rm -rf ./atk_output/*
export ATK_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_transformer/op_api/lib/libcust_opapi.so
# 生成用例
atk case -f op_block_sparse_attention.yaml -p generator_blocksparseattention.py

# 执行 (V3 MXFP4)
atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p new_mxfp4_aclnn_blocksparseattention.py -to 7200

# 复检命令
# atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p new_mxfp4_aclnn_blocksparseattention.py -wl [0] -to 7200
