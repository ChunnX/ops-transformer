#!/usr/bin/env bash
# MXFP4 BSA ATK 工程 (pyaclnn 直调 aclnnBlockSparseAttentionV3)
# 无 torch 封装: 不 import cann_ops_transformer, DUT 只经
# PyAclnnBlockSparseAttentionV3Api 直调两段式 aclnn。

set -eo pipefail   # 不能加 -u: set_env.sh 里有未绑定变量

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---- 环境: CANN + custom 算子, 与 bsa_test.sh 保持一致 ----
source /usr/local/Ascend/cann/set_env.sh
source /home/liwenguihw/work/custom_install/in_0824/vendors/custom_transformer/bin/set_env.bash

# CANN set_env.sh 会把 cann 自带 python site-packages 放到 PYTHONPATH 前部,
# 遮蔽 conda lwg 的 atk / cann_ops_transformer。把 lwg 提到最前。
export PYTHONPATH="$(/home/liwenguihw/.conda/envs/lwg/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):${PYTHONPATH}"
export PATH="/home/liwenguihw/.conda/envs/lwg/bin:${PATH}"

export ASCEND_LAUNCH_BLOCKING=1
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_$(whoami)"
export BSA_CPU_ONLY=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 不设 ASCEND_RT_VISIBLE_DEVICES -> nodes.yaml 的 devices:[0] 即物理 NPU 0
# (若设了 ASCEND_RT_VISIBLE_DEVICES=0, devices:[0] 会报 EE1003 userDevId 越界)
# 注: 设备 2 被 tangkai 的 BSA atk 占用, 本次切到 0 卡(用户指定)

# custom 算子库(最高优先级); 原 /usr/local/Ascend/cann-9.1.0/opp/... 无权限
export ATK_CUSTOM_OPP_PATH=/home/liwenguihw/work/custom_install/in_0824/vendors/custom_transformer/op_api/lib/libcust_opapi.so

# 归档上一轮产物(不再 rm -rf: 上次 Ctrl+C 后 atk_output/ 被 ATK 失败清理整目录删掉,
# rm 又会把旧报告清光 —— 归档到带时间戳目录, 无论成败旧报告都可追溯)
TS="$(date +%Y%m%d-%H%M%S)"
if [ -e ./result ] || [ -e ./atk_output ] || [ -e ./atk.log ]; then
    mkdir -p "./archive_${TS}"
    [ -e ./result ]    && mv ./result    "./archive_${TS}/result"
    [ -e ./atk_output ] && mv ./atk_output "./archive_${TS}/atk_output"
    [ -e ./atk.log ]   && mv ./atk.log   "./archive_${TS}/atk.log"
fi
mkdir -p ./atk_output

# 生成用例
atk case -f op_block_sparse_attention.yaml -p generator_blocksparseattention.py

# 执行: pyaclnn vs CPU golden(负 nice 需特权, 普通用户去掉)
atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json \
  -n nodes.yaml \
  -p function_aclnn_blocksparseattention_v3_mxfp4.py \
  -to 9999 -mt 1

# 复检单卡示例:
# atk task -c ./result/op_block_sparse_attention/json/all_op_block_sparse_attention.json \
#   -n nodes.yaml -p function_aclnn_blocksparseattention_v3_mxfp4.py -wl [0] -to 7200


##
