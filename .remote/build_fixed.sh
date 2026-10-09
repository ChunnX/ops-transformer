#!/bin/bash
# 统一换行 -> 构建改后包（deepseek1009 vendor）
set -u
WORK=/home/wchao00893980/deepseek1009
SRC=$WORK/source
API=$SRC/attention/scatter_pa_kv_cache/op_host/op_api

echo "=== 0. 换行转 LF（本地 Windows 是 CRLF） ==="
sed -i 's/\r$//' $API/aclnn_scatter_pa_kv_cache.cpp $API/scatter_pa_kv_cache.cpp $API/scatter_pa_kv_cache.h
sed -i 's/\r$//' $WORK/test_scatter_pa_kv_cache_paths.py
cd $SRC
echo "--- git status / diff --stat（应只剩本次改动）"
git status --short
git diff --stat
echo "--- 确认新接口与旧名"
grep -c "ScatterPaKvCacheWithAllocOut" $API/scatter_pa_kv_cache.cpp $API/scatter_pa_kv_cache.h $API/aclnn_scatter_pa_kv_cache.cpp
grep -c "OutOfPlace" $API/*.cpp $API/*.h | tr '\n' ' '; echo

echo
echo "=== 1. 构建 ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate wangchao
source /usr/local/Ascend/cann/set_env.sh
echo "python: $(which python)  $(python -V 2>&1)"
echo "开始: $(date +%T)"
bash build.sh --pkg --ops=scatter_pa_kv_cache --soc=ascend910b --vendor_name=deepseek1009 \
  --cann_3rd_lib_path=/home/wchao00893980/gbsa_a2a3_1009/ops-transformer/third_party -j8 \
  > $WORK/build.log 2>&1
rc=$?
echo "$rc" > $WORK/build.exit
echo "结束: $(date +%T)  exit=$rc"

echo
echo "=== 2. 构建结果 ==="
ls -l $SRC/build_out/*.run 2>/dev/null
echo "--- 日志尾部 25 行"
tail -25 $WORK/build.log
echo "--- 日志里的 error/warning 计数"
grep -ci "error" $WORK/build.log || true
grep -ci "warning" $WORK/build.log || true
