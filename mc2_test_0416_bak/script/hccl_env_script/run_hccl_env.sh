#!/bin/bash

##############################################################################
# 脚本功能：捕获错误并明确打印失败的具体行号和详细信息
# 特性：精确行号标识+多重错误判断+详细错误描述
# 超时处理：120秒超时
# 用法：./run_hccl_env.sh env_file.csv "command"
# 比如：bash run_hccl_env.sh hccl_env.csv "python3 moe_operator_test_v1.py"
##############################################################################

# -------------------------- 配置与参数校验 --------------------------
if [ $# -ne 2 ]; then
    echo -e "\033[31m[ERROR] 用法错误！正确用法: $0 环境变量文件 \"要执行的命令\"\033[0m"
    exit 1
fi

FILE_PATH="$1"
TARGET_CMD="$2"
LOG_DIR="execution_logs"
TIMEOUT_SECONDS=120
mkdir -p "$LOG_DIR" || { echo -e "\033[31m[ERROR] 无法创建日志目录: $LOG_DIR\033[0m"; exit 1; }

# 检查依赖
[ ! -f "$FILE_PATH" ] && { echo -e "\033[31m[ERROR] 文件不存在: $FILE_PATH\033[0m"; exit 1; }
[ ! -r "$FILE_PATH" ] && { echo -e "\033[31m[ERROR] 文件不可读: $FILE_PATH\033[0m"; exit 1; }
[ -z "$TARGET_CMD" ] && { echo -e "\033[31m[ERROR] 命令不能为空\033[0m"; exit 1; }
for cmd in timeout pgrep pkill; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "\033[31m[ERROR] 必需命令 '$cmd' 未找到\033[0m"
        exit 1
    fi
done

# -------------------------- 错误与成功关键词定义 --------------------------
FATAL_ERROR_KEYWORDS=(
    "RuntimeError" "Error" "ERROR" "Fatal" "FATAL"
    "HCCL communicator" "hcclUniqueId" "Connection reset by peer"
    "ERR02005" "dist internal error" "Segmentation fault"
    "Aborted" "Killed" "崩溃" "终止"
)

SUCCESS_KEYWORDS=(
    "测试通过" "success" "SUCCESS" "完成" "结束" "exited normally"
)

# -------------------------- 主要处理逻辑 --------------------------
echo -e "\033[34m[开始处理] 环境变量文件: $FILE_PATH\033[0m"
echo -e "超时设置: $TIMEOUT_SECONDS秒，日志将保存到: $LOG_DIR\n"

LINE_NUM=0
SUCCESS_LINE=0
FAILURE_LINE=0
declare -a FAILURE_DETAILS=()  # 存储所有失败行的详细信息

# 读取文件的每一行
while IFS= read -r line; do
    LINE_NUM=$((LINE_NUM + 1))
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LINE_LOG="$LOG_DIR/line_${LINE_NUM}_${TIMESTAMP}.log"
    CMD_PID=""
    IS_FAILURE=0
    FAILURE_REASON=""
    
    # 初始化日志
    echo "=== 行 $LINE_NUM 处理日志 - $(date) ===" > "$LINE_LOG"
    echo "原始行内容: '$line'" >> "$LINE_LOG"
    echo "----------------------------------------" >> "$LINE_LOG"
    
    # 跳过空行和注释行
    trimmed_line=$(echo "$line" | xargs)
    if [ -z "$trimmed_line" ] || [[ "$trimmed_line" == \#* ]]; then
        echo -e "[行 $LINE_NUM] 空行或注释行，跳过"
        continue
    fi

    # 解析环境变量
    declare -a VALID_VARS=()
    declare -a EXPORTED_VARS=()
    pattern='([a-zA-Z_][a-zA-Z0-9_]*)=([^[:space:]]+)'
    remaining_line="$trimmed_line"
    
    while [[ $remaining_line =~ $pattern ]]; do
        KEY="${BASH_REMATCH[1]}"
        VALUE="${BASH_REMATCH[2]}"
        VALID_VARS+=("$KEY=$VALUE")
        remaining_line="${remaining_line:${#BASH_REMATCH[0]}}"
        remaining_line=$(echo "$remaining_line" | xargs)
    done
    
    # 导出环境变量
    if [ ${#VALID_VARS[@]} -gt 0 ]; then
        echo -e "\n[行 $LINE_NUM] 导出 ${#VALID_VARS[@]} 个环境变量:"
        for var in "${VALID_VARS[@]}"; do
            KEY="${var%%=*}"
            VALUE="${var#*=}"
            export "$var" && echo -e "  ✅ $KEY=$VALUE" || echo -e "  \033[33m导出失败: $var\033[0m"
            EXPORTED_VARS+=("$KEY")
        done
    else
        echo -e "\033[33m[行 $LINE_NUM] 没有有效的环境变量，跳过执行命令\033[0m"
        continue
    fi

    # -------------------------- 执行命令并捕获错误 --------------------------
    echo -e "\n[行 $LINE_NUM] 执行命令（超时: $TIMEOUT_SECONDS秒）: $TARGET_CMD"
    echo "执行命令: $TARGET_CMD" >> "$LINE_LOG"
    echo "环境变量: ${VALID_VARS[*]}" >> "$LINE_LOG"
    echo "开始时间: $(date)" >> "$LINE_LOG"
    echo "----------------------------------------" >> "$LINE_LOG"

    # 执行命令并监控
    (eval "$TARGET_CMD" 2>&1) >> "$LINE_LOG" 2>&1 &
    CMD_PID=$!
    echo "命令PID: $CMD_PID" >> "$LINE_LOG"

    # 超时监控
    (
        sleep $TIMEOUT_SECONDS
        if ps -p $CMD_PID > /dev/null; then
            pkill -P $CMD_PID 2>/dev/null
            kill $CMD_PID 2>/dev/null
            echo "[超时监控] 命令执行超过${TIMEOUT_SECONDS}秒，已强制终止" >> "$LINE_LOG"
        fi
    ) &
    MONITOR_PID=$!

    # 等待命令完成
    wait $CMD_PID 2>/dev/null
    CMD_EXIT_CODE=$?
    kill $MONITOR_PID 2>/dev/null

    # -------------------------- 错误判断逻辑 --------------------------
    # 1. 检查致命关键词
    for keyword in "${FATAL_ERROR_KEYWORDS[@]}"; do
        if grep -qi "$keyword" "$LINE_LOG"; then
            IS_FAILURE=1
            FAILURE_REASON="检测到致命关键词: '$keyword'"
            break
        fi
    done

    # 2. 检查退出码
    if [ $IS_FAILURE -eq 0 ] && [ $CMD_EXIT_CODE -ne 0 ]; then
        IS_FAILURE=1
        FAILURE_REASON="非零退出码: $CMD_EXIT_CODE"
    fi

    # 3. 检查超时
    if [ $IS_FAILURE -eq 0 ] && [ $CMD_EXIT_CODE -eq 124 ]; then
        IS_FAILURE=1
        FAILURE_REASON="执行超时（超过${TIMEOUT_SECONDS}秒）"
    fi

    # 4. 检查成功标志
    if [ $IS_FAILURE -eq 0 ]; then
        HAS_SUCCESS=0
        for keyword in "${SUCCESS_KEYWORDS[@]}"; do
            if grep -qi "$keyword" "$LINE_LOG"; then
                HAS_SUCCESS=1
                break
            fi
        done
        if [ $HAS_SUCCESS -eq 0 ]; then
            IS_FAILURE=1
            FAILURE_REASON="未检测到成功标志，执行不完整"
        fi
    fi

    # -------------------------- 结果处理与行号记录 --------------------------
    if [ $IS_FAILURE -eq 1 ]; then
        FAILURE_LINE=$((FAILURE_LINE + 1))
        echo -e "----------------------------------------"
        # 明确打印失败行号
        echo -e "\033[31m[失败] 行号: $LINE_NUM - $FAILURE_REASON\033[0m"
        
        # 提取关键错误信息
        ERROR_DETAILS=$(grep -E "${FATAL_ERROR_KEYWORDS[*]}" "$LINE_LOG" | head -n 3 | sed 's/^/  /')
        echo -e "错误详情:\n$ERROR_DETAILS"
        echo -e "完整日志: $LINE_LOG"
        
        # 记录失败详情（包含行号）
        FAILURE_DETAILS+=("行号 $LINE_NUM: $FAILURE_REASON (环境变量: '$trimmed_line')")
    else
        SUCCESS_LINE=$((SUCCESS_LINE + 1))
        echo -e "----------------------------------------"
        echo -e "\033[32m[成功] 行号: $LINE_NUM - 命令执行完成\033[0m"
    fi

    echo -e "\n----------------------------------------"
    # 清除环境变量
    for key in "${EXPORTED_VARS[@]}"; do
        unset "$key"
    done
    unset EXPORTED_VARS
done < "$FILE_PATH"

# -------------------------- 汇总报告（突出显示失败行号） --------------------------
echo -e "\033[34m[处理汇总]\033[0m"
echo "========================================"
echo "总行数: $LINE_NUM"
echo "成功行数: $SUCCESS_LINE"
echo "失败行数: $FAILURE_LINE"
echo "========================================"

if [ $FAILURE_LINE -gt 0 ]; then
    echo -e "\n\033[31m[失败行号及原因汇总]\033[0m"
    echo "----------------------------------------"
    for detail in "${FAILURE_DETAILS[@]}"; do
        echo "- $detail"
    done
    echo -e "\n\033[31m[完成] 共 $FAILURE_LINE 行执行失败\033[0m"
    exit 1
else
    echo -e "\033[32m[完成] 所有行执行成功\033[0m"
    exit 0
fi
