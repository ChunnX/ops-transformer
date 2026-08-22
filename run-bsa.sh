# 一般场景

# 清空
rm -rf ./result/op_block_sparse_attention
rm -rf ./atk_output/*
rm -rf OPP*


# atk node --backend pyaclnn --devices 1 node --backend cpu --is_compare False task --task performance_device -c result/op_block_sparse_attention/json/all_op_block_sparse_attention.json  -p aclnn_blocksparseattention.py -to 3600 --performance_data 5,5,5


# 生成
atk case -f op_block_sparse_attention.yaml -p generator_blocksparseattention_actseq_aval.py

# 执行
# atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p aclnn_blocksparseattention.py -to 7200 --start=0 --end=20
atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p aclnn_blocksparseattention.py -to 3600


# atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json -n nodes.yaml -p aclnn_blocksparseattention.py -mt 50 -wl [0]
