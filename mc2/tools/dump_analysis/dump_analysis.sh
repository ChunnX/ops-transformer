#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
# 

# !
#  \file dump_analysis.sh
#  \brief


export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python3
chmod 777 *

# 文件运行使用方法 #####################
# 使用方法
# bash dump_analysis.sh PROFILING_PATH=xxx TARGET_DIR=xxx TOOL_PATH=xxx SOC_VERSION=xxx SP_MOE_NUM=x TP_WORLDSIZE=x SHARE_EXPERT_CARD_COUNT=x SHARE_EXPERT_NUM=x ---正常调用"
# bash dump_analysis.sh -h -----查看参数列表"

function help {
    echo "执行方法"
    echo "bash dump_analysis.sh TARGET_DIR=xxx TOOL_PATH=xxx PROFILING_PATH=xxx SOC_VERSION=xxx SP_MOE_NUM=x TP_WORLDSIZE=x SHARE_EXPERT_CARD_COUNT=x SHARE_EXPERT_NUM=x ---正常调用"
    echo "bash dump_analysis.sh -h -----查看参数列表"
    echo "参数列表"
    echo "TARGET_DIR(选填,分析win区dump数据必填):指向dump数据的路径,例:xxx/xxx/data-dump/"
    echo "TOOL_PATH(选填,分析win区dump数据必填):指向装包路径下tools目录所在的路径,例:xxx/xxx/pkg/8cann-8.x.0/"
    echo "SOC_VERSION(选填,分析win区dump数据必填):所需要分析的数据的芯片版本,910_93 or 950"
    echo "SP_MOE_NUM(选填):所需要分析的数据使用的特殊专家数(SP_MOE_NUM>0),不填默认为0"
    echo "TP_WORLDSIZE(选填):所需要分析的数据使用的TP_WORLDSIZE(TP_WORLDSIZE>1),不填默认为1"
    echo "SHARE_EXPERT_CARD_COUNT(选填):所需要分析的数据输入的共享专家卡数(SHARE_EXPERT_CARD_COUNT>0),不填默认为0"
    echo "SHARE_EXPERT_NUM(选填):所需要分析的数据输入的共享专家数(SHARE_EXPERT_NUM>0),不填默认为0"
    echo "PROFILING_PATH(选填,分析profiling数据必填):profiling数据存放的路径,不填默认在TARGET_DIR指定的路径下查找"
    echo "PLOG_PATH(选填,分析错误日志必填):plog日志存放的路径,不填不进行plog日志分析"
    echo "GRAPH_PATH(选填,分析graph图文件必填):graph图文件存放的路径,不填不进行graph图文件分析"
    exit 0
}
#获取sh脚本的文件路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SOC_VERSION_950="950"
SOC_VERSION_910_93="910_93"

#入参
for arg in "$@"; do
    if [[ "$arg" = "-h" || "$arg" = "-help" ]]; then
        help
    fi
    if [[ "$arg" == TARGET_DIR=* ]]; then
        TARGET_DIR="${arg#*=}"
    fi
    if [[ "$arg" == TOOL_PATH=* ]]; then
        TOOL_PATH="${arg#*=}"
    fi
    if [[ "$arg" == SOC_VERSION=* ]]; then
        SOC_VERSION="${arg#*=}"
    fi
    if [[ "$arg" == SP_MOE_NUM=* ]]; then
        SP_MOE_NUM="${arg#*=}"
    fi
    if [[ "$arg" == TP_WORLDSIZE=* ]]; then
        TP_WORLDSIZE="${arg#*=}"
    fi
    if [[ "$arg" == SHARE_EXPERT_CARD_COUNT=* ]]; then
        SHARE_EXPERT_CARD_COUNT="${arg#*=}"
    fi
    if [[ "$arg" == SHARE_EXPERT_NUM=* ]]; then
        SHARE_EXPERT_NUM="${arg#*=}"
    fi
    if [[ "$arg" == PROFILING_PATH=* ]]; then
        PROFILING_PATH="${arg#*=}"
    fi
    if [[ "$arg" == PLOG_PATH=* ]]; then
        PLOG_PATH="${arg#*=}"
    fi
    if [[ "$arg" == GRAPH_PATH=* ]]; then
        GRAPH_PATH="${arg#*=}"
    fi
    if ! [[ "$arg" =~ ^(-h|-help|TARGET_DIR=|TOOL_PATH=|SP_MOE_NUM=|TP_WORLDSIZE=|SOC_VERSION=|SHARE_EXPERT_CARD_COUNT=|SHARE_EXPERT_NUM=|PROFILING_PATH=|PLOG_PATH=|GRAPH_PATH=) ]]; then
        echo "warning:未知参数 $arg ,使用 -h or -help 查看帮助"
    fi
done
#判断入参
judge=0
#判断TARGET_DIR
if [ ! -n "$TARGET_DIR" ]; then
    echo "warning:TARGET_DIR 参数未输入,不进行dump数据解析"
else
    if [ ! -d "$TARGET_DIR" ]; then 
        echo "warning:未找到该参数对应的路径,TARGET_DIR:$TARGET_DIR,不进行dump数据解析"
    fi
fi
#判断TOOL_PATH
if [ ! -n "$TOOL_PATH" ]; then
    echo "warning:TOOL_PATH 参数未输入"
else
    if [ ! -e "$TOOL_PATH/tools/msaicerr/msaicerr.py" ]; then
        echo "warning:未找到该参数对应的路径,TOOL_PATH:$TOOL_PATH/tools/msaicerr/msaicerr.py"
    fi
fi
#判断SOC_VERSION
if [ ! -n "$SOC_VERSION" ]; then
    echo "warning:SOC_VERSION 参数未输入,不进行dump数据解析"
else
    if [ "$SOC_VERSION" != "$SOC_VERSION_910_93" ] && [ "$SOC_VERSION" != "$SOC_VERSION_950" ]; then
        echo "warning:SOC_VERSION:$SOC_VERSION 为非法输入"
    fi
fi
#判断SP_MOE_NUM,TP_WORLDSIZE
if [ ! -n "$SP_MOE_NUM" ]; then
    SP_MOE_NUM=0
    echo "warning:SP_MOE_NUM 参数未输入,使用默认值 SP_MOE_NUM  = 0"
fi
if [ "$SP_MOE_NUM" -lt 0 ]; then
    echo "error:SP_MOE_NUM:$SP_MOE_NUM should > 0"
    judge=1
fi
if [ ! -n "$TP_WORLDSIZE" ]; then
    TP_WORLDSIZE=1
    echo "warning:TP_WORLDSIZE 参数未输入,使用默认值 TP_WORLDSIZE  = 1"
fi
if [ "$TP_WORLDSIZE" -lt 1 ]; then
    echo "error:TP_WORLDSIZE:$TP_WORLDSIZE should >= 1"
    judge=1
fi
#判断共享专家卡数,共享专家数
if [ ! -n "$SHARE_EXPERT_CARD_COUNT" ]; then
    SHARE_EXPERT_CARD_COUNT=0
    echo "warning:SHARE_EXPERT_CARD_COUNT 参数未输入,使用默认值 SHARE_EXPERT_CARD_COUNT  = 0"
fi
if [ "$SHARE_EXPERT_CARD_COUNT" -lt 0 ]; then
    echo "error:SHARE_EXPERT_CARD_COUNT:$SHARE_EXPERT_CARD_COUNT should > 0"
    judge=1
fi
if [ ! -n "$SHARE_EXPERT_NUM" ]; then
    echo "warning:SHARE_EXPERT_NUM 参数未输入,使用默认值 SHARE_EXPERT_NUM  = 0"
    SHARE_EXPERT_NUM=0
fi
if [ "$SHARE_EXPERT_NUM" -lt 0 ]; then
    echo "error:SHARE_EXPERT_NUM:$SHARE_EXPERT_NUM should > 0"
    judge=1
fi
#判断profiling路径
if [ ! -n "$PROFILING_PATH" ]; then
    PROFILING_PATH=$TARGET_DIR
    echo "warning:PROFILING_PATH 参数未输入,使用默认值 PROFILING_PATH = TARGET_DIR"
fi
#PLOG_PATH
if [ ! -n "$PLOG_PATH" ]; then
    echo "warning:PLOG_PATH 参数未输入,不进行plog日志分析"
else
    if [ ! -d "$PLOG_PATH" ]; then
        echo "warning:未找到该参数对应的路径,PLOG_PATH:$PLOG_PATH,不进行plog日志分析"
    fi
fi
#判断GRAPH_PATH
if [ ! -n "$GRAPH_PATH" ]; then
    echo "warning:GRAPH_PATH 参数未输入,不进行graph图文件分析"
else
    if [ ! -d "$GRAPH_PATH" ]; then
        echo "warning:未找到该参数对应的路径,GRAPH_PATH:$GRAPH_PATH,不进行graph图文件分析"
    fi
fi
if [ "$judge" = "1" ]; then
    help
fi

echo "-----------------------------"
echo "-----------------------------"
echo "dump_path = $TARGET_DIR"
echo "tool_path = $TOOL_PATH/tools/msaicerr/msaicerr.py"
echo "SOC_VERSION = $SOC_VERSION"
echo "SP_MOE_NUM = $SP_MOE_NUM"
echo "TP_WORLDSIZE = $TP_WORLDSIZE"
echo "SHARE_EXPERT_CARD_COUNT = $SHARE_EXPERT_CARD_COUNT"
echo "SHARE_EXPERT_NUM = $SHARE_EXPERT_NUM"
echo "PROFILING_PATH = $PROFILING_PATH"
echo "PLOG_PATH = $PLOG_PATH"
echo "GRAPH_PATH = $GRAPH_PATH"
echo "-----------------------------"
echo "-----------------------------"

#开始解析
file_num=$(ls $TARGET_DIR | wc -l)
#判断输入的共享专家卡数是否超出dump数据对应的卡数
if ls "$TARGET_DIR/1/exception_info."* >/dev/null 2>&1; then
    if [ $SHARE_EXPERT_CARD_COUNT -gt $file_num ]; then
        echo "error:SHARE_EXPERT_CARD_COUNT($SHARE_EXPERT_CARD_COUNT) should <= all_care_num($file_num)"
        exit 1
    fi
else
    if [ $SHARE_EXPERT_CARD_COUNT -gt 1 ]; then
        echo "error:SHARE_EXPERT_CARD_COUNT($SHARE_EXPERT_CARD_COUNT) should <= all_care_num(1)"
        exit 1
    fi
fi

if [ "$SOC_VERSION" = "$SOC_VERSION_910_93" ]; then
    echo "进入 A3 处理流程"
    if ls "$TARGET_DIR/exception_info."* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    elif ls "$TARGET_DIR/1/exception_info."* >/dev/null 2>&1; then
        echo "开始解析多卡dump数据"
        for ((i = 0; i < file_num; i++))
        do
            if ls "$TARGET_DIR$i/exception_info."*.workspace.* >/dev/null 2>&1; then
                echo "开始解析 $i 卡数据"
                python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                echo "$i 卡数据解析完成"
                echo "--------------------------------------------"
            else
                for file_dump in $TARGET_DIR$i/exception_info.*;
                do
                    if [[ -f "$file_dump" ]]; then
                        echo "开始解析 $i 卡数据"
                        python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                        python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                        echo "$i 卡数据解析完成"
                        echo "--------------------------------------------"
                    else
                        echo "error:路径 $TARGET_DIR$i/ 下没有dump数据"
                        echo "--------------------------------------------"
                    fi
                done
            fi
        done
    elif ls "$TARGET_DIR/0/exception_info."* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/0/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/0/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR/0/ 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    else
        echo "error:路径 $TARGET_DIR 下没有dump数据"
    fi
elif [ "$SOC_VERSION" = "$SOC_VERSION_950" ]; then
    echo "进入 A5 处理流程"
    if ls "$TARGET_DIR/mc2_exception_info"* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    elif ls "$TARGET_DIR/1/mc2_exception_info"* >/dev/null 2>&1; then
        echo "开始解析多卡dump数据"
        for ((i = 0; i < file_num; i++))
        do
            if ls "$TARGET_DIR$i/exception_info."*.workspace.* >/dev/null 2>&1; then
                echo "开始解析 $i 卡数据"
                python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                echo "$i 卡数据解析完成"
                echo "--------------------------------------------"
            else
                for file_dump in $TARGET_DIR$i/exception_info.*;
                do
                    if [[ -f "$file_dump" ]]; then
                        echo "开始解析 $i 卡数据"
                        python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                        python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM $file_num $i $TARGET_DIR$i/ $SOC_VERSION
                        echo "$i 卡数据解析完成"
                        echo "--------------------------------------------"
                    else
                        echo "error:路径 $TARGET_DIR$i/ 下没有dump数据"
                        echo "--------------------------------------------"
                    fi
                done
            fi
        done
    elif ls "$TARGET_DIR/0/mc2_exception_info"* >/dev/null 2>&1; then
        echo "开始解析:单卡dump数据"
        if ls "$TARGET_DIR/0/exception_info."*.workspace.* >/dev/null 2>&1; then
            python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
            echo "单卡数据解析完成"
            echo "--------------------------------------------"
        else
            for file_dump in $TARGET_DIR/0/exception_info.*;
            do
                if [[ -f "$file_dump" ]]; then
                    python3 $TOOL_PATH/tools/msaicerr/msaicerr.py -d "$file_dump"
                    python3 $SCRIPT_DIR/dump_analysis.py $SP_MOE_NUM $TP_WORLDSIZE $SHARE_EXPERT_CARD_COUNT $SHARE_EXPERT_NUM 1 0 $TARGET_DIR/0/ $SOC_VERSION
                    echo "单卡数据解析完成"
                    echo "--------------------------------------------"
                else
                    echo "error:路径 $TARGET_DIR/0/ 下没有dump数据"
                    echo "--------------------------------------------"
                fi
            done
        fi
    else
        echo "error:路径 $TARGET_DIR 下没有以mc2_exception_info开头的dump数据"
    fi
fi

#profiling_path判断
if [ ! -d "$PROFILING_PATH" ]; then
    echo "warning:profiling数据指定的路径 $PROFILING_PATH 不存在,不进行profiling数据解析"
else
    folder_count=$(find "$PROFILING_PATH" -maxdepth 1 -type d -regex ".*/.*ascend_pt.*" | wc -l)
    if [ "$folder_count" -eq 0 ]; then
        echo "warning:profiling数据指定的路径 $PROFILING_PATH 下未找到profiling文件,不进行profiling数据解析"
    else
        echo "在指定的路径 $PROFILING_PATH 下找到 $folder_count 张卡的profiling数据"
        python3 profiling_analysis.py $PROFILING_PATH $folder_count
    fi
fi


#graph文件转换
if [[ -z "$GRAPH_PATH" ]]; then
    echo "warning:graph图文件路径未指定,不进行graph图文件转换"
    if [ -d "$PLOG_PATH" ]; then
        echo "========== 开始在 $PLOG_PATH 及其子目录中查找所有包含 .om 的文件的信息 =========="
        result=$(grep -r -I "\.om" "$PLOG_PATH")
        if [ -n "$result" ]; then
            echo "$result"
        else
            echo "warning:$PLOG_PATH下未找到.om文件的信息"
        fi
    fi
else
    # 记录脚本最开始的执行目录（处理完必须回到这里）
    ORIGINAL_DIR=$(pwd)

    # 查找所有 rankx 文件夹
    rank_dirs=$(find "$GRAPH_PATH" -type d -name "*_rank*[0-9]")

    if [ -n "$rank_dirs" ]; then
        for dir in $rank_dirs; do
            # 提取卡号
            rank_num=$(echo "$dir" | grep -oE 'rank[0-9]+' | grep -oE '[0-9]+')
            [ -z "$rank_num" ] && continue

            # 查找 om 文件
            om_file=$(find "$dir" -type f -name "*.om" | head -n 1)

            if [ -z "$om_file" ]; then
                echo "warning:卡$rank_num :$dir 中未找到对应的graph文件(.om文件)"
                continue
            else
                json_file="${om_file%.om}_device${rank_num}.json"
                echo
                echo "开始转换卡$rank_num 的graph图文件:$om_file"
                atc --mode=1 --om="$om_file" --json="$json_file"
                echo "转换后的文件为:$json_file"
                echo
            fi
        done

    else
        # 查找根目录 om
        root_om=$(find "$GRAPH_PATH" -type f -name "*.om" | head -n 1)

        if [ -z "$root_om" ]; then
            echo "warning:路径下没有任何 .om 文件"
        else
            rank_num=0
            json_file="${root_om%.om}_device${rank_num}.json"
            echo
            echo "开始转换卡$rank_num 的graph图文件:$root_om"
            atc --mode=1 --om="$root_om" --json="$json_file"
            echo "转换后的文件为:$json_file"
            echo
        fi
    fi

    # 所有处理完成后，强制回到脚本初始执行路径
    cd "$ORIGINAL_DIR"
fi
[ -z "$PLOG_PATH" ] && PLOG_PATH="NA"
[ -z "$GRAPH_PATH" ] && GRAPH_PATH="NA"
python3 log_graph_analysis.py $PLOG_PATH $GRAPH_PATH