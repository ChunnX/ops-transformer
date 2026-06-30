#!/bin/bash
# 回归集：审核后固化。覆盖 退化(topk≥E) + 稀疏(topk<E,部分填充块) + 非1024对齐 h(验 DataCopyPad scale 修复) + 链式。
# 用法: bash regress.sh   （前置：算子已 build_op 装好、test exe 已 build_test 编好）
set -e
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PASS=0; FAIL=0
run() {  # run <op> <bs> <topk> <h> <E> <R> <seed> <tag>
  local op=$1; shift; local tag="${@: -1}"
  local out; out=$(bash run.sh "$op" "$1" "$2" "$3" "$4" "$5" "$6" 2>&1 | grep -E "_TEARDOWN (PASS|FAIL)|x mae" || true)
  if echo "$out" | grep -q PASS; then echo "  [OK ] $op $1 $2 $3 $4 $5 seed$6  ($tag)"; PASS=$((PASS+1));
  else echo "  [BAD] $op $1 $2 $3 $4 $5 seed$6  ($tag)"; echo "$out" | sed 's/^/        /'; FAIL=$((FAIL+1)); fi
}
echo "===== 回归集 ====="
echo "-- 退化场景 topk=8=GE(全密) h=7168 --"
run dispatch 8 8 7168 4 2 1234 "degen"
run combine  8 8 7168 4 2 1234 "degen"
echo "-- 稀疏场景 topk=2<E=4(部分填充块/空洞) h=7168 --"
run dispatch 8 2 7168 4 2 1234 "sparse"
run combine  8 2 7168 4 2 1234 "sparse"
echo "-- 非1024对齐 h=2560(scaleH=80 非32B → 验 DataCopyPad scale 修复) topk=3 --"
run dispatch 8 3 2560 4 2 7 "h2560"
run combine  8 3 2560 4 2 7 "h2560"
echo "-- 非1024对齐 h=1536(scaleH=48 非32B) topk=2 --"
run dispatch 8 2 1536 4 2 99 "h1536"
run combine  8 2 1536 4 2 99 "h1536"
echo "-- 链式(dispatch NPU expand_idx → combine) h=7168 sparse --"
if bash run_chain.sh 8 2 7168 4 2 5 2>&1 | grep -q "COMBINE_TEARDOWN PASS"; then echo "  [OK ] chain 8 2 7168 4 2"; PASS=$((PASS+1)); else echo "  [BAD] chain"; FAIL=$((FAIL+1)); fi
echo "===== 回归汇总: PASS=$PASS FAIL=$FAIL ====="
[ "$FAIL" -eq 0 ] && echo "ALL GREEN" || exit 1
