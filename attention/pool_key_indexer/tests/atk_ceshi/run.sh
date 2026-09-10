# 生成用例
atk case -f pool_key_indexer.yaml -p gdd_gen_exec_pool_key_indexer.py
atk case -f pool_key_indexer_quant.yaml -p gdd_gen_exec_pool_key_indexer.py
atk case -f pool_key_indexer_bound.yaml -p gdd_gen_exec_pool_key_indexer.py

atk case -f pool_key_indexer_un.yaml -p gdd_gen_exec_pool_key_indexer.py
atk case -f pool_key_indexer_nd.yaml -p gdd_gen_exec_pool_key_indexer.py
atk case -f pool_key_indexer_quant_nd.yaml -p gdd_gen_exec_pool_key_indexer.py
atk case -f pool_key_indexer_quant_un.yaml -p gdd_gen_exec_pool_key_indexer.py

#冒烟运行
atk task -n node.yaml -p gdd_exec_pool_key_indexer.py -c result/pool_key_indexer/json/all_pool_key_indexer.json -e 10
