#!/bin/bash
# 在 deepseek1009 里搭好工作副本（源码拷贝 + 落盘改动 + 校验）
set -u
WORK=/home/wchao00893980/deepseek1009
SRC_PREV=/home/wchao00893980/scatter1009/source
SRC=$WORK/source

echo "=== 1. 拷贝源码（464M，含 .git，保留提交历史） ==="
if [ -d "$SRC" ]; then
  echo "已存在，跳过拷贝：$SRC"
else
  cp -a "$SRC_PREV" "$SRC"
  echo "拷贝完成"
fi
# 丢掉旧 build（CMake 缓存里是旧绝对路径，跨目录复用不可靠）
rm -rf "$SRC/build" "$SRC/build_out"
du -sh "$SRC"

echo
echo "=== 2. 落盘本次改动（3 个 cpp/h） ==="
API=$SRC/attention/scatter_pa_kv_cache/op_host/op_api
cp -f $WORK/upload/aclnn_scatter_pa_kv_cache.cpp $API/
cp -f $WORK/upload/scatter_pa_kv_cache.cpp $API/
cp -f $WORK/upload/scatter_pa_kv_cache.h $API/
cp -f $WORK/upload/test_scatter_pa_kv_cache_paths.py $WORK/
ls -l $API/ | tail -6

echo
echo "=== 3. git 视角确认改动（应当只有这 3 个文件） ==="
cd "$SRC"
git status --short
echo "--- diff --stat"
git diff --stat
echo "--- 新接口名是否已生效（应能看到 WithAllocOut）"
grep -rn "ScatterPaKvCacheWithAllocOut" $API/ | head -6
echo "--- 旧名是否清除（应为 0）"
grep -rn "OutOfPlace" $API/ | wc -l

echo
echo "=== 4. 关键日志串是否存在于已安装的库里（决定 grep 是否有意义） ==="
LIBS="/usr/local/Ascend/cann-9.1.0-beta.1/aarch64-linux/lib64"
for lib in $(ls $LIBS/libopapi*.so $LIBS/libnnopbase.so 2>/dev/null); do
  n=$(grep -c -a "The input and output point to the same address" "$lib" 2>/dev/null)
  m=$(grep -c -a "Contiguous copy with move align" "$lib" 2>/dev/null)
  r=$(grep -c -a "op executor cache find tensor relation fail" "$lib" 2>/dev/null)
  g=$(grep -c -a "Get op cache key" "$lib" 2>/dev/null)
  a=$(grep -c -a "Add op cache key" "$lib" 2>/dev/null)
  echo "  $(basename $lib): same-address=$n move-align=$m relation-fail=$r get-cache-key=$g add-cache-key=$a"
done

echo
echo "=== 5. 之前装的两个包（A/B 对照用） ==="
ls -d $WORK/../scatter1009/opp-unmodified/vendors/*/ $WORK/../scatter1009/opp-install/vendors/*/ 2>/dev/null
cat $WORK/../scatter1009/opp-unmodified/vendors/*/bin/set_env.bash 2>/dev/null | head -20
