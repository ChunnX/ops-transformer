#!/bin/bash
# 调用 pytest + conftest 跑 test_bsa_mxfp4_golden.py 或用例 ①
# shape 等参数在 conftest.py 的 CLI 参数中指定

cd "$(dirname "$0")"

python -m pytest test_bsa_mxfp4_golden.py::TestBSAMxfp4Golden::test_bsa_mxfp4_bnsd_dense \
    -v \
    --b 2 --n1 8 --n2 8 --s1 128 --s2 128 --qk_d 128
