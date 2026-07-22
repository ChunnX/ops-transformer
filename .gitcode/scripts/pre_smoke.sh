#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set +e

echo "start run test case, please wait ..."
cd /home/taskspace
WORKSPACE=/home/taskspace

if grep -q "scripts/ci/mirror_update_time.txt" ${WORKSPACE}/pr_filelist.txt; then
    echo "need run all ops"
else
    ops=`python3 ${WORKSPACE}/scripts/ci/parse_changed_ops.py ${WORKSPACE}/pr_filelist.txt false`
    if [ "$ops" == "" ];then
        echo "No custom ops need to be tested, skip smoke test."
        exit 0
    fi
fi

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 确定要测试的 ops 列表
# ==============================
declare -a ops
ops=`python3 ${WORKSPACE}/scripts/ci/parse_changed_ops.py ${WORKSPACE}/pr_filelist.txt false` 
echo $ops

# ==============================
# 运行测试主循环
# ==============================
log "start run test case, please wait ..."

arm_package="cann-ops-cv-custom_linux-aarch64.run"
wget -nv https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/${arm_package}
chmod u+x ./${arm_package}
bash ./${arm_package} 2>&1 | tee -a ./run_test.log
CV_OP_CATEGORY_LIST="image objdetect"

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

for op in "${ops[@]}"; do
  echo "Processing: $op"
  mode="eager"
  [ "$op" = "crop_and_resize" ] && mode="graph"
  source /usr/local/Ascend/cann/set_env.sh
  if grep -q "${WORKSPACE}/scripts/ci/mirror_update_time.txt" ${WORKSPACE}/pr_filelist.txt; then
    op=$(bash ${WORKSPACE}/update_image.sh "$ops")
    ops_line=$(echo "$op" | tail -n1)
    IFS=',' read -ra op_array <<< "$ops_line"
    for single_op in "${op_array[@]}"; do
      single_op=$(echo "$single_op" | xargs)
      echo "bash build.sh --run_example "$single_op" eager cust"
      bash build.sh --run_example "$single_op" eager cust  2>&1 | tee -a ./run_test.log
      status=$?
      if [ $status -ne 0 ]; then
        echo "${single_op} example fail"
        exit 1
      fi
    done
  else
    for category in ${CV_OP_CATEGORY_LIST}; do
      if ls "${WORKSPACE}/${category}/${op}/examples"/test_aclnn_* 1> /dev/null 2> /dev/null; then
        bash build.sh --run_example "$op" eager cust  2>&1 | tee -a ./run_test.log
      fi
      if ls "${WORKSPACE}/${category}/${op}/examples"/test_geir_* 1> /dev/null 2> /dev/null; then
        bash build.sh --run_example "$op" graph cust  2>&1 | tee -a ./run_test.log
      fi
    done
  fi
done

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${obs_path}/${slog_name}"; then
  echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/slog.tar.gz"
fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -E '\b(FAIL|errors|fail|failed|error|ERROR:|Error|error:)\b' "./run_test.log" | grep -v "error)"; then
    echo "$date_time : run test case failed"
    exit 1
else
    echo "$date_time : run test case success"
    exit 0
fi
