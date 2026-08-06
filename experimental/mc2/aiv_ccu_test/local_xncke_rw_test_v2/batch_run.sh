#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 用法:
#   bash batch_run.sh cases.csv [cases_result.csv]
#
# 功能:
#   自动编译 + 读取 CSV 用例文件 + 逐行调用 test_local_xncke_rw_v2 + 解析输出 + 回写结果到 CSV。
#   用户只需 source CANN 环境后直接运行本脚本，无需手动编译。
#
# 输入 CSV 格式（首行为表头，字段顺序固定）:
#   mode,res_kind,die_id,warmup_count,loop_count,pattern,rank_id,ranksize
#   2,2,1,10,100,0x5A5A000000000001,0,1
#   0,0,1,10,100,0x5A5A000000000001,0,1
#   ...
#
# 输出 CSV 格式（在输入列后追加结果列）:
#   mode,res_kind,die_id,warmup_count,loop_count,pattern,rank_id,ranksize,
#   warmup_cycles,start_cycle,end_cycle,avg_cycles,avg_ns,verify_result,xn_readback,cke_readback,status
#
# 前置条件:
#   source /usr/local/Ascend/cann/set_env.sh
#   export RANK_TABLE_FILE=/path/to/rank_table.json
# -----------------------------------------------------------------------------------------------------------

set -uo pipefail

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

SOC_VERSION="ascend950pr_9599"
BUILD_TYPE="Release"

if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi
export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
export LD_LIBRARY_PATH=${CURRENT_DIR}/out/lib:${CURRENT_DIR}/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

# ==================== 自动编译 ====================

NEED_BUILD=0
BIN=${CURRENT_DIR}/out/bin/test_local_xncke_rw_v2

if [ ! -x "$BIN" ]; then
    NEED_BUILD=1
else
    # 源码比二进制新则重新编译
    NEWEST_SRC=$(find "$CURRENT_DIR" -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \
        | xargs stat -c '%Y %n' 2>/dev/null | sort -rn | head -1 | awk '{print $1}')
    BIN_MTIME=$(stat -c '%Y' "$BIN" 2>/dev/null || echo 0)
    if [ -n "$NEWEST_SRC" ] && [ "$NEWEST_SRC" -gt "$BIN_MTIME" ]; then
        NEED_BUILD=1
    fi
fi

if [ "$NEED_BUILD" -eq 1 ]; then
    echo "[INFO] 开始编译..."
    rm -rf "$CURRENT_DIR/build"
    mkdir -p "$CURRENT_DIR/build"
    cmake -B "$CURRENT_DIR/build" \
        -DSOC_VERSION=${SOC_VERSION} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/out \
        -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
    cmake --build "$CURRENT_DIR/build" -j
    cmake --install "$CURRENT_DIR/build"
    echo "[INFO] 编译完成"
    echo "==================================================================="
fi

# ==================== 参数解析 ====================

INPUT_CSV=""
OUTPUT_CSV=""
CASE_LINE=0

while [ $# -gt 0 ]; do
    case "$1" in
        -caseid)
            CASE_LINE="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT_CSV" ]; then
                INPUT_CSV="$1"
            elif [ -z "$OUTPUT_CSV" ]; then
                OUTPUT_CSV="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$INPUT_CSV" ]; then
    echo "用法: bash batch_run.sh <input.csv> [output.csv] [-caseid N]"
    echo "      output   默认在 input 文件名后加 _result"
    echo "      -caseid  指定只跑某个 id 的用例（匹配 CSV 中 id 列的值）"
    echo ""
    echo "示例:"
    echo "  bash batch_run.sh cases/cases_example.csv                # 跑全部"
    echo "  bash batch_run.sh cases/cases_example.csv result.csv     # 跑全部，指定输出"
    echo "  bash batch_run.sh cases/cases_example.csv -caseid 3      # 只跑 id=3 的用例"
    echo "  bash batch_run.sh cases/cases_example.csv result.csv -caseid 3  # 只跑 id=3，指定输出"
    exit 1
fi

OUTPUT_CSV="${OUTPUT_CSV:-${INPUT_CSV%.csv}_result.csv}"

if [ ! -f "$INPUT_CSV" ]; then
    echo "[ERROR] 输入文件不存在: $INPUT_CSV"
    exit 1
fi

# ==================== 批量执行 ====================

HEADER="id,mode,res_kind,die_id,warmup_count,loop_count,pattern,rank_id,ranksize,warmup_cycles,start_cycle,end_cycle,avg_cycles,avg_ns,verify_result,xn_readback,cke_readback,status"
echo "$HEADER" > "$OUTPUT_CSV"

total_lines=$(wc -l < "$INPUT_CSV")
case_count=$((total_lines - 1))

if [ -n "$CASE_LINE" ]; then
    echo "[INFO] 只跑 id=$CASE_LINE 的用例（共 $case_count 条）"
else
    echo "[INFO] 共 $case_count 条用例"
fi
echo "  输入: $INPUT_CSV"
echo "  输出: $OUTPUT_CSV"
echo "==================================================================="

current=0

tail -n +2 "$INPUT_CSV" | while IFS=',' read -r id mode res_kind die_id warmup_count loop_count pattern rank_id ranksize; do
    current=$((current + 1))

    # 单跑模式：按 id 匹配
    if [ -n "$CASE_LINE" ] && [ "$id" != "$CASE_LINE" ]; then
        continue
    fi

    echo "[INFO] === 用例 id=$id ($current/$case_count) ==="
    echo "  mode=$mode res_kind=$res_kind die_id=$die_id warmup=$warmup_count loop=$loop_count pattern=$pattern rank=$rank_id/$ranksize"

    output=$("$BIN" \
        --rank_id "$rank_id" \
        --ranksize "$ranksize" \
        --mode "$mode" \
        --res_kind "$res_kind" \
        --warmup "$warmup_count" \
        --die_id "$die_id" \
        --xn_num 1 \
        --xn_index 0 \
        --cke_num 1 \
        --cke_index 0 \
        --loop_count "$loop_count" \
        --pattern "$pattern" 2>&1)
    bin_ret=$?

    # 解析输出
    warmup_cycles=$(echo "$output" | grep "warmupCycle:" | grep -oP '\d+' | tail -1)
    start_cycle=$(echo "$output" | grep "startCycle:" | grep -oP '\d+' | tail -1)
    end_cycle=$(echo "$output" | grep "endCycle:" | grep -oP '\d+' | tail -1)
    avg_ns=$(echo "$output" | grep "平均:" | grep -oP '[\d.]+' | head -1)

    # verify_result: 仅 write_then_read（mode=2）才有数据校验，其他模式为 N/A
    if [ "$mode" = "2" ] || [ "$mode" = "write_then_read" ] || [ "$mode" = "WRITE_THEN_READ" ] \
       || [ "$mode" = "wtr" ] || [ "$mode" = "WTR" ]; then
        verify_str=$(echo "$output" | grep "数据校验:" | grep -oP '(通过|失败)' | head -1)
        if [ "$verify_str" = "通过" ]; then
            verify_result="PASS"
        elif [ "$verify_str" = "失败" ]; then
            verify_result="FAIL"
        else
            verify_result="N/A"
        fi
    else
        verify_result="N/A"
    fi

    # readback 值：仅 read(1) 或 write_then_read(2) 才有 readback 输出
    if [ "$mode" = "1" ] || [ "$mode" = "2" ] \
       || [ "$mode" = "read" ] || [ "$mode" = "READ" ] \
       || [ "$mode" = "write_then_read" ] || [ "$mode" = "WRITE_THEN_READ" ] \
       || [ "$mode" = "wtr" ] || [ "$mode" = "WTR" ]; then
        xn_readback=$(echo "$output" | grep "XN readback:" | grep -oP '0x[0-9a-fA-F]+' | head -1)
        cke_readback=$(echo "$output" | grep "CKE readback:" | grep -oP '0x[0-9a-fA-F]+' | head -1)
    else
        xn_readback=""
        cke_readback=""
    fi

    if [ -n "$start_cycle" ] && [ -n "$end_cycle" ] && [ "$loop_count" -gt 0 ]; then
        elapsed=$((end_cycle - start_cycle))
        avg_cycles=$(awk "BEGIN {printf \"%.2f\", $elapsed / $loop_count}")
    else
        avg_cycles=""
    fi

    if [ "$bin_ret" -eq 0 ]; then
        status="OK"
    else
        status="ERROR"
        echo "[ERROR] 用例执行失败(ret=$bin_ret)，原始输出:"
        echo "$output" | head -20
    fi

    warmup_cycles="${warmup_cycles:-}"
    start_cycle="${start_cycle:-}"
    end_cycle="${end_cycle:-}"
    avg_cycles="${avg_cycles:-}"
    avg_ns="${avg_ns:-}"
    xn_readback="${xn_readback:-}"
    cke_readback="${cke_readback:-}"

    echo "  结果: warmup=$warmup_cycles start=$start_cycle end=$end_cycle avg_cycles=$avg_cycles avg_ns=$avg_ns verify=$verify_result status=$status"
    echo "-------------------------------------------------------------------"

    echo "${id},${mode},${res_kind},${die_id},${warmup_count},${loop_count},${pattern},${rank_id},${ranksize},${warmup_cycles},${start_cycle},${end_cycle},${avg_cycles},${avg_ns},${verify_result},${xn_readback:-},${cke_readback:-},${status}" >> "$OUTPUT_CSV"

done

echo "[INFO] 全部完成，结果已写入: $OUTPUT_CSV"
