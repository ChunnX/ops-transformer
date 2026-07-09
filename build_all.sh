#!/bin/bash
# 循环编译 7 个算子 × 3 个 SoC 版本，汇总日志
# Usage: bash build_all.sh [--soc 950,910_93,910b] [--ops dispatch_v2,combine_v2,...]

set -uo pipefail
cd /mnt/workspace/gitCode/gitcode_lijd/ops-transformer

OPS=(moe_distribute_dispatch moe_distribute_dispatch_v2 moe_distribute_dispatch_v3
     moe_distribute_combine moe_distribute_combine_v2 moe_distribute_combine_v3
     moe_distribute_combine_add_rms_norm)

ALL_SOCS=(ascend950 ascend910_93 ascend910b)

LOG_DIR=build_logs_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

SUMMARY_FILE="${LOG_DIR}/summary.txt"
O_COUNT_FILE="${LOG_DIR}/o_counts.txt"

echo "========== 编译汇总 $(date) ==========" > "$SUMMARY_FILE"
echo >> "$SUMMARY_FILE"
echo "# op soc o_count" > "$O_COUNT_FILE"

PASS=0
FAIL=0
SKIP=0
TOTAL=0

# 从 def 文件中解析算子支持的 SoC
get_supported_socs() {
    local op=$1
    local def_file
    def_file=$(find /mnt/workspace/gitCode/gitcode_lijd/ops-transformer/mc2/${op}/op_host -name "${op}_def.cpp" 2>/dev/null | head -1)
    if [[ -z "$def_file" ]]; then
        # 部分 v3 算子可能在 v2 目录中定义
        def_file=$(find /mnt/workspace/gitCode/gitcode_lijd/ops-transformer/mc2/${op} -name "*_def.cpp" 2>/dev/null | head -1)
    fi
    if [[ -z "$def_file" ]]; then
        echo ""
        return
    fi
    grep -oP 'AddConfig\("ascend[^"]+' "$def_file" 2>/dev/null | sed 's/AddConfig("//'
}

check_o_count() {
    local op=$1 soc=$2 runfile=$3 tmpdir
    tmpdir=$(mktemp -d /tmp/opkg_check_XXXXXX)
    bash "$runfile" --noexec --extract="$tmpdir" > /dev/null 2>&1
    local count
    count=$(find "$tmpdir" -name "*.o" -type f 2>/dev/null | wc -l)
    rm -rf "$tmpdir"
    echo "$op  $soc  $count" >> "$O_COUNT_FILE"
    echo "  📦 .o 文件数: $count"
}

for op in "${OPS[@]}"; do
  op_short="${op#moe_distribute_}"
  [[ "$op_short" == combine_add_rms_norm ]] && op_short="arn"

  # 查询 def 支持的 SoC
  supported=($(get_supported_socs "$op"))
  if [[ ${#supported[@]} -eq 0 ]]; then
    echo "[WARN] $op: 未找到 def 文件或未配置 SoC，跳过所有编译"
    for soc in "${ALL_SOCS[@]}"; do
      echo "  ⏭️  SKIP  $op  $soc  (def 中未配置 SoC)" | tee -a "$SUMMARY_FILE"
      ((SKIP++))
    done
    continue
  fi

  for soc in "${ALL_SOCS[@]}"; do
    ((TOTAL++))
    log_file="${LOG_DIR}/${soc}_${op_short}.log"

    # 跳过 def 中不支持的 SoC
    if ! [[ " ${supported[*]} " == *" $soc "* ]]; then
      echo "[$(date +%H:%M:%S)] $op 不支持 $soc，跳过"
      echo "  ⏭️  SKIP  $op  $soc  (def 中未配置)" | tee -a "$SUMMARY_FILE"
      echo "$op  $soc  -" >> "$O_COUNT_FILE"
      ((SKIP++))
      continue
    fi

    echo "[$(date +%H:%M:%S)] 编译 $op for $soc ... 日志: $log_file"

    rm -rf build build_out
    if bash build.sh --pkg --soc="$soc" --ops="$op" > "$log_file" 2>&1; then
      echo "  ✅ PASS  $op  $soc" | tee -a "$SUMMARY_FILE"
      ((PASS++))

      run_files=(build_out/*.run)
      if [ ${#run_files[@]} -eq 1 ] && [ -f "${run_files[0]}" ]; then
        check_o_count "$op" "$soc" "${run_files[0]}"
      else
        echo "  ⚠️  未找到 .run 文件" >> "$SUMMARY_FILE"
        echo "$op  $soc  -" >> "$O_COUNT_FILE"
      fi
    else
      echo "  ❌ FAIL  $op  $soc  (错误摘要见日志末尾)" | tee -a "$SUMMARY_FILE"
      echo "  最后 10 行:" >> "$SUMMARY_FILE"
      grep -n "error:" "$log_file" 2>/dev/null | grep -vi "warning\|note:\|pragma\|In file\|from .*:\|ge_api\|BrokenPipe" | tail -5 >> "$SUMMARY_FILE" || true
      echo >> "$SUMMARY_FILE"
      ((FAIL++))
    fi
    echo
  done
done

echo "==============================================" | tee -a "$SUMMARY_FILE"
echo "累计: $TOTAL  通过: $PASS  失败: $FAIL  跳过: $SKIP" | tee -a "$SUMMARY_FILE"
echo "日志目录: $LOG_DIR" | tee -a "$SUMMARY_FILE"
