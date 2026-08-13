#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 用法:
#   bash batch_run.sh cases.csv [cases_result.csv]
#   bash batch_run.sh cases.csv -caseid N
#   bash batch_run.sh cases.csv cases_result.csv -caseid N
#
# 功能:
#   自动编译 + 读取 CSV 用例文件 + 逐行调用 test_local_ms_rw + 解析输出 + 回写结果到 CSV。
#
# 输入 CSV 格式（首行为表头，字段顺序固定）:
#   id,mode0,warmup,ms_id,loop_count,data_size,rank_id,ranksize
#   3,0,1,0,1000,4096,0,1
#   ...
#
# 输出 CSV 格式（在输入列后追加结果列）:
#   id,mode0,warmup,ms_id,loop_count,data_size,rank_id,ranksize,
#   warmup_cycles,start_cycle,end_cycle,avg_cycles,avg_ns,bandwidth_gbs,verify_result,status
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

# ==================== 参数解析 ====================

INPUT_CSV=""
OUTPUT_CSV=""
CASE_LINE=""

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

# ==================== 自动编译 ====================

NEED_BUILD=0
BIN=${CURRENT_DIR}/out/bin/test_local_ms_rw

if [ ! -x "$BIN" ]; then
    NEED_BUILD=1
else
    NEWEST_SRC=$(find "$CURRENT_DIR" -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \
        | xargs stat -c '%Y %n' 2>/dev/null | sort -rn | head -1 | awk '{print $1}')
    BIN_MTIME=$(stat -c '%Y' "$BIN" 2>/dev/null || echo 0)
    if [ -n "$NEWEST_SRC" ] && [ "$NEWEST_SRC" -gt "$BIN_MTIME" ]; then
        NEED_BUILD=1
    fi
fi

if [ "$NEED_BUILD" -eq 1 ]; then
    if [ -n "$CASE_LINE" ]; then
        # 单跑：编译过程打屏
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
    else
        # 批跑：编译过程静默
        echo "[INFO] 编译中..."
        rm -rf "$CURRENT_DIR/build"
        mkdir -p "$CURRENT_DIR/build"
        cmake -B "$CURRENT_DIR/build" \
            -DSOC_VERSION=${SOC_VERSION} \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
            -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/out \
            -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH} > /dev/null 2>&1
        cmake --build "$CURRENT_DIR/build" -j > /dev/null 2>&1
        cmake --install "$CURRENT_DIR/build" > /dev/null 2>&1
        echo "[INFO] 编译完成"
    fi
fi

echo "==================================================================="
echo "[INFO] 环境信息"
echo "  主机名:       $(hostname)"
echo "  服务器IP:     $(hostname -I 2>/dev/null | awk '{print $1}')"
echo "  操作系统:     $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '"')"
echo "  内核版本:     $(uname -r)"
echo "  架构:         $(uname -m)"
echo "  CANN路径:     ${_ASCEND_INSTALL_PATH}"
echo "  RANK_TABLE:   ${RANK_TABLE_FILE:-未设置}"
echo "  二进制:       ${BIN}"
echo "==================================================================="

# ==================== 批量执行 ====================

HEADER="id,mode0,warmup,ms_id,loop_count,data_size,rank_id,ranksize,warmup_cycles,start_cycle,end_cycle,avg_cycles,avg_ns,bandwidth_gbs,warmup_cycles2,start_cycle2,end_cycle2,avg_ns2,bandwidth_gbs2,verify_result,status"
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

tail -n +2 "$INPUT_CSV" | while IFS=',' read -r id mode0 warmup ms_id loop_count data_size rank_id ranksize || [ -n "$id" ]; do
    current=$((current + 1))

    # 单跑模式：按 id 匹配
    if [ -n "$CASE_LINE" ] && [ "$id" != "$CASE_LINE" ]; then
        continue
    fi

    # 获取 mode0 字符串用于显示，支持字符串和数字两种写法
    case "$mode0" in
        0|write|WRITE) mode0_str="WRITE" ;;
        1|read|READ) mode0_str="READ" ;;
        2|read_write|READ_WRITE|rw|RW) mode0_str="READ_WRITE" ;;
        3|write_then_read|WRITE_THEN_READ|wtr|WTR) mode0_str="WRITE_THEN_READ" ;;
        4|dual_core_rw|DUAL_CORE_RW|dcrw|DCRW) mode0_str="DUAL_CORE_RW" ;;
        *) mode0_str="UNKNOWN($mode0)" ;;
    esac

    echo "[INFO] === 用例 id=$id ($current/$case_count) ==="
    echo "  mode0=$mode0_str warmup=$warmup ms_id=$ms_id loop=$loop_count data_size=$data_size rank=$rank_id/$ranksize"

    if [ -n "$CASE_LINE" ]; then
        # 单跑：直接打屏
        "$BIN" \
            --rank_id "$rank_id" \
            --ranksize "$ranksize" \
            --mode0 "$mode0" \
            --warmup "$warmup" \
            --ms_id "$ms_id" \
            --loop_count "$loop_count" \
            --data_size "$data_size" 2>&1
        bin_ret=$?
        output=""
    else
        # 批跑：捕获到变量，不打屏
        output=$("$BIN" \
            --rank_id "$rank_id" \
            --ranksize "$ranksize" \
            --mode0 "$mode0" \
            --warmup "$warmup" \
            --ms_id "$ms_id" \
            --loop_count "$loop_count" \
            --data_size "$data_size" 2>&1)
        bin_ret=$?
    fi

    # 解析输出
    if [ "$mode0" = "4" ] || [ "$mode0" = "dual_core_rw" ] || [ "$mode0" = "DUAL_CORE_RW" ] \
       || [ "$mode0" = "dcrw" ] || [ "$mode0" = "DCRW" ]; then
        # 双核模式：写核用 startCycle/endCycle，读核用 startCycle2/endCycle2
        warmup_cycles=$(echo "$output" | grep "warmupCycle:" | grep -oP '\d+' | head -1)
        start_cycle=$(echo "$output" | grep "startCycle:" | grep -oP '\d+' | tail -1)
        end_cycle=$(echo "$output" | grep "endCycle:" | grep -oP '\d+' | tail -1)
        warmup_cycles2=$(echo "$output" | grep "warmupCycle:" | grep -oP '\d+' | tail -1)
        start_cycle2=$(echo "$output" | grep "startCycle2:" | grep -oP '\d+' | tail -1)
        end_cycle2=$(echo "$output" | grep "endCycle2:" | grep -oP '\d+' | tail -1)
        avg_ns=$(echo "$output" | grep "平均:" | grep -oP '[\d.]+' | head -1)
        avg_ns2=$(echo "$output" | grep "平均:" | grep -oP '[\d.]+' | tail -1)
        bandwidth_gbs=$(echo "$output" | grep "带宽:" | grep -oP '[\d.]+' | head -1)
        bandwidth_gbs2=$(echo "$output" | grep "带宽:" | grep -oP '[\d.]+' | tail -1)
        verify_result="N/A"
    else
        warmup_cycles=$(echo "$output" | grep "warmupCycle:" | grep -oP '\d+' | tail -1)
        start_cycle=$(echo "$output" | grep "startCycle:" | grep -oP '\d+' | tail -1)
        end_cycle=$(echo "$output" | grep "endCycle:" | grep -oP '\d+' | tail -1)
        avg_ns=$(echo "$output" | grep "平均:" | grep -oP '[\d.]+' | head -1)
        bandwidth_gbs=$(echo "$output" | grep "带宽:" | grep -oP '[\d.]+' | head -1)
        warmup_cycles2=""
        start_cycle2=""
        end_cycle2=""
        avg_ns2=""
        bandwidth_gbs2=""
        # verify_result: 仅 write_then_read 才有数据校验，其他模式为 N/A
        if [ "$mode0" = "3" ] || [ "$mode0" = "write_then_read" ] || [ "$mode0" = "WRITE_THEN_READ" ] \
           || [ "$mode0" = "wtr" ] || [ "$mode0" = "WTR" ]; then
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
    bandwidth_gbs="${bandwidth_gbs:-}"
    warmup_cycles2="${warmup_cycles2:-}"
    start_cycle2="${start_cycle2:-}"
    end_cycle2="${end_cycle2:-}"
    avg_ns2="${avg_ns2:-}"
    bandwidth_gbs2="${bandwidth_gbs2:-}"

    echo "  结果: warmup=$warmup_cycles start=$start_cycle end=$end_cycle avg_cycles=$avg_cycles avg_ns=$avg_ns bw=$bandwidth_gbs"
    if [ -n "$start_cycle2" ]; then
        echo "         read_core: warmup=$warmup_cycles2 start=$start_cycle2 end=$end_cycle2 avg_ns=$avg_ns2 bw=$bandwidth_gbs2"
    fi
    echo "  verify=$verify_result status=$status"
    echo "-------------------------------------------------------------------"

    echo "${id},${mode0},${warmup},${ms_id},${loop_count},${data_size},${rank_id},${ranksize},${warmup_cycles},${start_cycle},${end_cycle},${avg_cycles},${avg_ns},${bandwidth_gbs},${warmup_cycles2},${start_cycle2},${end_cycle2},${avg_ns2},${bandwidth_gbs2},${verify_result},${status}" >> "$OUTPUT_CSV"

done

echo "[INFO] 全部完成，结果已写入: $OUTPUT_CSV"
