#!/bin/bash
# 在已有的 DEBUG 日志里找 ViewCopy 分支证据；并确认目录规模/包是否还在
D=/home/wchao00893980/scatter1009

echo "=== 1. readable-after-debug.log 里的 ViewCopy 分支打印 ==="
for pat in "same address" "created by the user" "move align" "ViewCopy data addr"; do
  n=$(grep -c "$pat" $D/readable-after-debug.log 2>/dev/null)
  echo "  \"$pat\": $n 次"
done
echo "--- 命中示例（前 3 条，含行号）"
grep -n "same address\|created by the user\|move align" $D/readable-after-debug.log 2>/dev/null | head -3 | cut -c1-220
echo "--- 该日志里 ViewCopy 相关行样例（前 5 条）"
grep -n "ViewCopy" $D/readable-after-debug.log 2>/dev/null | head -5 | cut -c1-220

echo
echo "=== 2. 日志级别/环境确认（日志头部） ==="
head -3 $D/readable-after-debug.log | cut -c1-200
echo "--- 其他日志里的分支关键字"
for f in $D/cache-evidence-small.log $D/msprof-source-before.log $D/msprof-source-after.log $D/test-profile.log $D/test-profile-modified.log; do
  echo "  $f: same-address=$(grep -c 'same address' $f 2>/dev/null) move-align=$(grep -c 'move align' $f 2>/dev/null) ViewCopy=$(grep -c 'ViewCopy' $f 2>/dev/null)"
done

echo
echo "=== 3. 源码目录规模（决定拷贝还是重新 clone） ==="
du -sh $D/source 2>/dev/null
du -sh $D/source/build $D/source/build_out 2>/dev/null
echo "--- git 对象大小"
du -sh $D/source/.git 2>/dev/null
echo "--- git 远端分支/提交"
git -C $D/source log --oneline -1 2>/dev/null
git -C $D/source branch -a 2>/dev/null | head

echo
echo "=== 4. 两个已安装包是否还在 ==="
ls -d $D/opp-install/vendors/*/ $D/opp-unmodified/vendors/*/ 2>/dev/null
for v in $D/opp-install $D/opp-unmodified; do
  echo "--- $v"
  ls $v/vendors/*/op_api/lib/ 2>/dev/null
  cat $v/vendors/*/version.info 2>/dev/null | head -5
done
echo "--- 改前包(op-unmodified)里的 L0 符号"
strings $D/opp-unmodified/vendors/*/op_api/lib/libcust_opapi.so 2>/dev/null | grep -c "ScatterPaKvCacheWithAllocOut"
echo "--- 改后包(opp-install)里的 L0 符号"
strings $D/opp-install/vendors/*/op_api/lib/libcust_opapi.so 2>/dev/null | grep -c "ScatterPaKvCacheWithAllocOut"
echo "--- 改后包(op-install)里的报错串"
strings $D/opp-install/vendors/*/op_api/lib/libcust_opapi.so 2>/dev/null | grep -i "AllocTensor for Scatter" | head -3
echo "--- 改后包(op-install)里是否有旧名残留"
strings $D/opp-install/vendors/*/op_api/lib/libcust_opapi.so 2>/dev/null | grep -c "OutOfPlace"

echo
echo "=== 5. build.sh 的用法（帮助信息） ==="
cd $D/source && sed -n '1,40p' build.sh | grep -n "soc\|vendor\|pkg\|cann_3rd" | head -20
