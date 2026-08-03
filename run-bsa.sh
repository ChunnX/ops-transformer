# 一般场景

# 清空
rm -rf ./result/op_block_sparse_attention/json/all_block_sparse_attention.json
rm -rf ./atk_output/*

# 生成
atk case -f op_block_sparse_attention.yaml -p generator_blocksparseattention.py

# 执行
atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p new_fp8_aclnn_blocksparseattention.py -to 120

# atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p aclnn_blocksparseattention.py -mt 50 -to 7200

# result json
# rm -rf ./atk_output/*
# atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p aclnn_blocksparseattention.py -mt 50 -wl [0]


# atk task -c ./all_op_block_sparse_attention_small_3.json -n nodes.yaml -p aclnn_blocksparseattention.py -wl [3]

# 复检命令
# atk node --backend pyaclnn --devices 2 node --backend cpu --name gpu task --task accuracy_lt -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -p aclnn_blocksparseattention.py -wl [19] -to 7200 --loop_nums 200