#!/bin/bash
# 读 scatter1009 里已有的结论（只读）
D=/home/wchao00893980/scatter1009

echo "=== install.log（改后包安装位置） ==="
cat $D/install.log

echo
echo "=== build.sh 命令行（来自 build.log 头部） ==="
head -5 $D/build.log

echo
echo "=== test_scatter_readable.py ==="
cat $D/test_scatter_readable.py

echo
echo "=== readable-before/checks.csv（改前） ==="
cat $D/readable-before/checks.csv
echo "--- loaded_libraries.json"
cat $D/readable-before/loaded_libraries.json

echo
echo "=== readable-after-debug/checks.csv（改后） ==="
cat $D/readable-after-debug/checks.csv

echo
echo "=== results-profile/results.json 的关键字段（改后/内置对比） ==="
for f in $D/results-profile/results.json $D/results-profile-modified/results.json \
         $D/results-msprof-builtin/results.json $D/results-msprof-modified/results.json; do
  echo "--- $f"
  python3 - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
m=d.get("metadata",{})
print("  label:",m.get("label"),"| api:",m.get("api"),"| commit:",m.get("commit_under_review") or m.get("git_commit"))
its=d.get("iterations") or d.get("cases") or []
if isinstance(its,dict):
    for k,v in its.items():
        r=v.get("rows",[])
        if r: print(f"  {k}: {len(r)} 轮, host第一轮={r[0]['host_call_us']:.1f}us, 后续中位数={sorted(x['host_call_us'] for x in r[1:])[len(r[1:])//2]:.1f}us")
    continue
host=[x["host_call_us"] for x in its if "host_call_us" in x]
if host:
    later=sorted(host[1:])
    print(f"  {len(host)} 轮: 第一轮={host[0]:.1f}us, 第2..N中位数={later[len(later)//2]:.1f}us")
PY
done

echo
echo "=== msprof 统计（builtin vs modified） ==="
for f in $D/msprof-builtin-20/scatter_msprof_statistics.json $D/msprof-modified-20/scatter_msprof_statistics.json; do
  echo "--- $f"; cat "$f"
done

echo
echo "=== cache-evidence.log 里 contiguous.cpp 相关行（去重） ==="
grep -o "contiguous.cpp:[0-9]*\]\[NNOP\]\[[A-Za-z]*\]" $D/cache-evidence.log | sort | uniq -c | sort -rn | head -20
echo "--- 有没有 AbandonCache / BindWorkspace 之类关键字"
grep -iE "abandon|reuse|cache hit|hit cache|repeat" $D/cache-evidence.log | head -10
echo "--- 日志里的 op 名"
grep -oE "OpName:\[[^]]*\]" $D/cache-evidence.log | sort | uniq -c | sort -rn | head -10

echo
echo "=== bash_history 里与本任务相关的命令（最后 60 条） ==="
tail -60 /root/.bash_history
