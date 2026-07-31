#!/bin/bash
# 调用 pytest + conftest 跑 test_bsa_mxfp4_golden.py 或用例 ①
# shape 等参数在 conftest.py 的 CLI 参数中指定
# 1. 跑单独用例 
source /srv/workspace/mnt1/work_code/Ascend_20260703/cann/bin/setenv.bash
cd "$(dirname "$0")"
export BSA_CPU_ONLY=0 #npu模式
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 加上-s才能看到日志
python -m pytest test_bsa_mxfp4_golden.py::TestBSAMxfp4Golden::test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128 \
    -v -s \
    --b 1 --n1 1 --n2 1 --s1 128 --s2 512 --qk_d 128 --dump True


# 2.cpu_模式
# source /srv/workspace/mnt1/work_code/Ascend_20260703/cann/bin/setenv.bash
# set -e
# cd "$(dirname "$0")"
# export BSA_CPU_ONLY=1 # cpu模式
# export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# PY="${PYTHON:-python}"
# exec "$PY" -m pytest test_bsa_mxfp4_golden.py::TestBSAMxfp4Golden::test_bsa_mxfp4_bnsd_mha_dense_b1_qn1_kvn1_g1_qs128_kvs128_d128 \
#     -v -s --dump=True "$@"


# 3批量跑小冒烟
# 3.1. 跑整个文件(所有 4 个用例)
# cd /home/w00947241/lwg/bsa_code_0730/ops-transformer_0703/attention/block_sparse_attention/tests/mxfp4_pytest
# source /srv/workspace/mnt1/work_code/Ascend_20260703/cann/bin/setenv.bash
# export BSA_CPU_ONLY=0
# export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# python -m pytest test_bsa_mxfp4_golden.py -v --dump True

# 3.2. 按类批量跑
# python -m pytest "test_bsa_mxfp4_golden.py::TestBSAMxfp4Golden" -v
# 3.3. 指定多个用例(选择其中几个)
# python -m pytest test_bsa_mxfp4_golden.py \
#   -k "dense or gqa" -v --dump True      # 用 -k 表达式筛选