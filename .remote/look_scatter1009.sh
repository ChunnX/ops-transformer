#!/bin/bash
# 查看 scatter1009（疑似本任务之前的远程工作目录）
D=/home/wchao00893980/scatter1009

echo "=== 顶层 ==="
ls -la $D

echo
echo "=== 子目录 ==="
for x in $D/*/; do
  echo "--- $x"
  ls -la "$x" 2>/dev/null | head -12
done

echo
echo "=== 里面的 git checkout ==="
for g in $D/*/.git; do
  d=$(dirname "$g")
  echo "--- $d"
  git -C "$d" log --oneline -3 2>&1 | head -3
  git -C "$d" remote -v 2>&1 | head -2
  echo "    dirty: $(git -C "$d" status --short 2>/dev/null | wc -l)"
  echo "    当前分支: $(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null)"
done

echo
echo "=== 脚本内容 ==="
for f in $D/*.sh; do
  echo "----- $f"
  head -40 "$f"
done

echo
echo "=== 构建/克隆结果 ==="
for f in $D/*.exit $D/clone.log $D/checkout.log; do
  echo "----- $f"; cat "$f" 2>/dev/null | head -10
done

echo
echo "=== cache-evidence 日志里找关键分支 ==="
for f in $D/cache-evidence.log $D/cache-evidence-small.log; do
  echo "----- $f ($(wc -l < "$f" 2>/dev/null) 行)"
  grep -c "same address" "$f" 2>/dev/null | sed 's/^/  same-address 次数: /'
  grep -c "created by the user" "$f" 2>/dev/null | sed 's/^/  created-by-user 次数: /'
  grep -c "move align" "$f" 2>/dev/null | sed 's/^/  move-align 次数: /'
  grep -c "tensor relation fail" "$f" 2>/dev/null | sed 's/^/  relation-fail 次数: /'
  grep -c "AbandonCache\|abandon" "$f" 2>/dev/null | sed 's/^/  abandon 关键词: /'
  echo "  前 3 行:"; head -3 "$f" 2>/dev/null | cut -c1-160 | sed 's/^/    /'
done

echo
echo "=== 已安装的算子包 ==="
ls -la /usr/local/Ascend/ascend-toolkit/latest/opp/vendors 2>/dev/null | head
find / -maxdepth 6 -name "libcust_opapi*" -o -maxdepth 6 -name "libaclnn*" 2>/dev/null | head -8
