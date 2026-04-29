#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

set -e

SCRIPT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_OUT_DIR="${SCRIPT_DIR}/build_out"
LOG_DIR="${SCRIPT_DIR}/mc2/build_logs"
MC2_DIR="${SCRIPT_DIR}/mc2"
TIMING_DATA="${MC2_DIR}/.timing_data.txt"

declare -A SOC_MAP
SOC_MAP["2"]="ascend910b"
SOC_MAP["3"]="ascend910_93"
SOC_MAP["5"]="ascend950"

SOC_VERSION=""
SOC_DESC=""

if [[ $# -ne 1 ]]; then
    echo "使用方法：bash build_mc2_ops.sh -2|-3|-5"
    exit 1
fi

param="$1"
if [[ "$param" == "-2" ]]; then
    SOC_VERSION="ascend910b"
    SOC_DESC="A2"
elif [[ "$param" == "-3" ]]; then
    SOC_VERSION="ascend910_93"
    SOC_DESC="A3"
elif [[ "$param" == "-5" ]]; then
    SOC_VERSION="ascend950"
    SOC_DESC="A5"
else
    echo "错误：支持的参数为 -2, -3, -5"
    exit 1
fi

LOG_DIR="${LOG_DIR}/${SOC_VERSION}"
mkdir -p ${LOG_DIR}
RESULT_FILE="${MC2_DIR}/build_timing_results_${SOC_VERSION}.md"

rm -f ${TIMING_DATA}
touch ${TIMING_DATA}

echo ""
echo "========================================"
echo "芯片型号：${SOC_DESC} (${SOC_VERSION})"
echo "========================================"
echo ""

# 扫描支持的算子
SUPPORTED_OPS=()
for op_dir in ${MC2_DIR}/*/; do
    op_name=$(basename ${op_dir})
    [[ "$op_name" == "3rd" || "$op_name" == "common" || "$op_name" == "tools" ]] && continue
    [[ "$op_name" == "mega_moe" ]] && continue
    
    if ls ${op_dir}/op_host/*.cpp 1>/dev/null 2>&1; then
        if grep -r "${SOC_VERSION}" ${op_dir}/op_host/*.cpp 2>/dev/null | grep -q "AddConfig"; then
            SUPPORTED_OPS+=("$op_name")
        fi
    fi
done

echo "发现 ${#SUPPORTED_OPS[@]} 个支持 ${SOC_DESC} 的算子"
for op in "${SUPPORTED_OPS[@]}"; do echo "  - ${op}"; done
echo ""

# 初始化结果文件
cat > ${RESULT_FILE} << EOF
# MC2算子编译耗时统计（${SOC_DESC}）

## 编译环境
- 芯片：${SOC_DESC} (${SOC_VERSION})
- 模式：--pkg
- 算子数：${#SUPPORTED_OPS[@]}
- 时间：$(date '+%Y-%m-%d %H:%M:%S')

EOF

echo "| 序号 | 算子名称 | 编译耗时 | 状态 | 产物大小 |" >> ${RESULT_FILE}
echo "|------|---------|---------|------|---------|" >> ${RESULT_FILE}

total_start=$(date +%s)
success=0
fail=0
total_time=0

# 编译每个算子
for i in "${!SUPPORTED_OPS[@]}"; do
    op="${SUPPORTED_OPS[$i]}"
    idx=$((i + 1))
    
    echo "[${idx}/${#SUPPORTED_OPS[@]}] 编译：${op}"
    
    rm -rf ${BUILD_DIR} ${BUILD_OUT_DIR}
    start=$(date +%s)
    log="${LOG_DIR}/${op}_build.log"
    
    if bash build.sh --pkg --soc=${SOC_VERSION} --ops=${op} > ${log} 2>&1; then
        end=$(date +%s)
        time=$((end - start))
        min=$((time / 60))
        sec=$((time % 60))
        size=$(du -sh ${BUILD_OUT_DIR}/*.run 2>/dev/null | awk '{print $1}' || echo "N/A")
        
        echo "  ✓ 成功 (${min}分${sec}秒, ${size})"
        echo "| ${idx} | ${op} | ${min}分${sec}秒 | ✓ | ${size} |" >> ${RESULT_FILE}
        echo "${time}|${op}|${min}分${sec}秒|${size}" >> ${TIMING_DATA}
        
        success=$((success + 1))
        total_time=$((total_time + time))
    else
        end=$(date +%s)
        time=$((end - start))
        min=$((time / 60))
        sec=$((time % 60))
        
        echo "  ✗ 失败 (${min}分${sec}秒)"
        echo "| ${idx} | ${op} | ${min}分${sec}秒 | ✗ | N/A |" >> ${RESULT_FILE}
        fail=$((fail + 1))
    fi
done

total_end=$(date +%s)
total=$((total_end - total_start))
total_h=$((total / 3600))
total_m=$((($total % 3600) / 60))
total_s=$((total % 60))

avg=0
avg_m=0
avg_s=0
[[ $success -gt 0 ]] && avg=$((total_time / success)) && avg_m=$((avg / 60)) && avg_s=$((avg % 60))

rate=$(awk "BEGIN {printf \"%.2f\", ($success/${#SUPPORTED_OPS[@]}) * 100}")

# 统计汇总
cat >> ${RESULT_FILE} << EOF

## 统计汇总
- 总算子：${#SUPPORTED_OPS[@]}
- 成功：${success} ✓
- 失败：${fail} ✗
- 成功率：${rate}%
- 总耗时：${total_h}时${total_m}分${total_s}秒
- 平均耗时：${avg_m}分${avg_s}秒

EOF

# 时间排序（正确的排序）
if [[ $success -gt 0 ]]; then
    echo "" >> ${RESULT_FILE}
    echo "## 编译时间排序（从快到慢）" >> ${RESULT_FILE}
    echo "" >> ${RESULT_FILE}
    echo "| 排名 | 算子名称 | 编译耗时 | 产物大小 |" >> ${RESULT_FILE}
    echo "|------|---------|---------|---------|" >> ${RESULT_FILE}
    
    rank=0
    sort -t'|' -k1 -n ${TIMING_DATA} | while IFS='|' read -r sec op time size; do
        rank=$((rank + 1))
        echo "| ${rank} | ${op} | ${time} | ${size} |" >> ${RESULT_FILE}
    done
fi

# 失败列表
if [[ $fail -gt 0 ]]; then
    echo "" >> ${RESULT_FILE}
    echo "## 失败算子" >> ${RESULT_FILE}
    for op in "${SUPPORTED_OPS[@]}"; do
        grep -q "|${op}|" ${TIMING_DATA} || echo "- ${op}" >> ${RESULT_FILE}
    done
fi

rm -f ${TIMING_DATA}

echo ""
echo "========================================"
echo "编译完成 (${SOC_DESC})"
echo "========================================"
echo "成功：${success}/${#SUPPORTED_OPS[@]} (${rate}%)"
echo "耗时：${total_h}时${total_m}分${total_s}秒"
echo "平均：${avg_m}分${avg_s}秒"
echo "结果：${RESULT_FILE}"
echo "========================================"

[[ $fail -gt 0 ]] && exit 1