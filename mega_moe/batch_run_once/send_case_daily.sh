#!/bin/bash
# ============================================
# 默认配置参数区（可通过命令行覆盖）
# ============================================
DEFAULT_OUTDIR="daily_result"
DEFAULT_EXCEL_FILE="mega_moe_daily_rdv.xlsx"  # 默认 Excel 文件
MODE=0
BATCH_SIZE=100
INTERVAL=30
PYTHON_HELPER="get_sheet_info.py"  # Python 辅助脚本路径

# 默认 Sheet 列表（当不指定 -s 时使用）
DEFAULT_SHEETS=("a8w8_value" "a8w4_value" "a4w4_value")
DEFAULT_START_ROW=2  # 默认开始行（第1行通常是表头）

# ============================================
# 命令行参数变量
# ============================================
RESTART_MODE=0          # 是否强制重新跑（忽略进度）
SPECIFIED_SHEETS=()     # 指定的 Sheet 列表
SPECIFIED_START_ROW=""  # 指定的开始行（优先级最高）
SPECIFIED_END_ROW=""    # 指定的结束行（优先级最高）
CUSTOM_OUTDIR=""        # 自定义输出目录
CUSTOM_EXCEL_FILE=""    # 自定义 Excel 文件
GOLDEN_MODE=0           # 是否强制编译 golden（0=自动判断，1=强制编译）
GOLDEN_ONLY_MODE=0      # 是否只编译 golden（0=正常模式，1=只编译不执行）
SHOW_HELP=0
# ============================================
# 帮助信息
# ============================================
show_help() {
    cat << EOF
用法: $0 [选项]
选项:
    -e, --excel FILE         指定 Excel 文件路径（默认: $DEFAULT_EXCEL_FILE）
    -r, --restart            强制重新跑，删除所有进度文件和日志（从头开始）
    -s, --sheet NAME         指定要跑的 Sheet 名称（可多次使用，如 -s sheet1 -s sheet2）
                             若不指定，则跑默认配置中的所有 Sheets
    --start-row N            指定开始行号（默认: $DEFAULT_START_ROW）
    --end-row N              指定结束行号（强烈建议指定，或使用 Python 辅助脚本自动检测）
    -o, --output DIR         指定输出目录（默认: 根据 Excel 文件名自动生成）
    -b, --batch-size N       指定每批跑多少个用例（默认: $BATCH_SIZE）
    -i, --interval N         指定批次间隔秒数（默认: $INTERVAL）
    -m, --mode N             指定是否复用通信域（默认：$MODE）
    -g, --golden             强制编译 golden（本地执行，不占用 NPU）
                             默认情况下，只有使用非默认 Excel 文件时才编译
    -G, --golden-only        只编译 golden，不提交 qrun 任务（本地执行）
                             必须指定 -s 或使用默认 Sheets，可配合 --start-row/--end-row
    -h, --help               显示此帮助信息

注意:
    - 使用 -g 或非默认 Excel 文件时，每个批次执行前会先编译 golden
    - 使用 -G 时只编译 golden，不提交任何 qrun 任务
    - Golden 编译与 qrun 的行范围完全一致（每批编多少就跑多少）
    - 强烈建议安装 Python 辅助脚本以自动检测行数
    - 如果没有 Python 辅助脚本，建议使用 --end-row 明确指定结束行
    - --start-row 和 --end-row 可以单独使用，也可以组合使用
    - 进度文件会根据 Excel 文件名和输出目录进行区分

默认 Sheets（需要配合 Python 辅助脚本或手动指定行范围）:
EOF
    for sheet in "${DEFAULT_SHEETS[@]}"; do
        echo "    - $sheet"
    done
    cat << EOF

示例:
    # 使用默认 Excel 文件跑所有 Sheets（需要 Python 辅助脚本）
    $0
    
    # 指定 Excel 文件（自动检测所有 Sheet 和行数，先编译 golden）
    $0 -e my_data.xlsx
    
    # 强制编译 golden（即使是默认 Excel 文件）
    $0 -g
    
    # 只编译 golden，不执行（指定 Sheet）
    $0 -G -s a8w8_value
    
    # 只编译 golden，指定 Excel 和 Sheet
    $0 -G -e my_data.xlsx -s Sheet1
    
    # 只编译 golden，指定行范围
    $0 -G -s a8w8_value --start-row 10 --end-row 30
    
    # 只编译 golden，使用默认所有 Sheets
    $0 -G
    
    # 指定 Excel 文件和特定 Sheet（自动检测行数）
    $0 -e my_data.xlsx -s a8w8_value -s a4w4_value
    
    # 强制重跑
    $0 -e data.xlsx -r
    
    # 完整示例
    $0 -e data.xlsx -s Sheet1 -s Sheet2 --start-row 5 --end-row 100 -o output_dir -b 3 -i 60
EOF
}
# ============================================
# 解析命令行参数
# ============================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--excel)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "错误: --excel 需要指定文件路径"
                exit 1
            fi
            CUSTOM_EXCEL_FILE="$2"
            shift 2
            ;;
        -r|--restart)
            RESTART_MODE=1
            shift
            ;;
        -s|--sheet)
            if [[ -z "$2" || "$2" == -* ]]; then
                echo "错误: --sheet 需要指定 Sheet 名称"
                exit 1
            fi
            SPECIFIED_SHEETS+=("$2")
            shift 2
            ;;
        --start-row)
            if [[ -z "$2" || "$2" =~ [^0-9] ]]; then
                echo "错误: --start-row 指定数字"
                exit 1
            fi
            SPECIFIED_START_ROW="$2"
            shift 2
            ;;
        --end-row)
            if [[ -z "$2" || "$2" =~ [^0-9] ]]; then
                echo "错误: --end-row 需要指定数字"
                exit 1
            fi
            SPECIFIED_END_ROW="$2"
            shift 2
            ;;
        -o|--output)
            if [[ -z "$2" ]]; then
                echo "错误: --output 需要指定目录"
                exit 1
            fi
            CUSTOM_OUTDIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            if [[ -z "$2" || "$2" =~ [^0-9] ]]; then
                echo "错误: --batch-size 需要指定数字"
                exit 1
            fi
            BATCH_SIZE="$2"
            shift 2
            ;;
        -i|--interval)
            if [[ -z "$2" || "$2" =~ [^0-9] ]]; then
                echo "错误: --interval 需要指定数字"
                exit 1
            fi
            INTERVAL="$2"
            shift 2
            ;;
        -m|--mode)
            if [[ -z "$2" || ! "$2" =~ ^[01]$ ]]; then
                echo "错误: --mode 需要指定 0 或 1"
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        -g|--golden)
            GOLDEN_MODE=1
            shift
            ;;
        -G|--golden-only)
            GOLDEN_ONLY_MODE=1
            GOLDEN_MODE=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done
# ============================================
# 应用配置
# ============================================
# 设置 Excel 文件
if [[ -n "$CUSTOM_EXCEL_FILE" ]]; then
    EXCEL_FILE="$CUSTOM_EXCEL_FILE"
    echo "📁 使用自定义 Excel 文件: $EXCEL_FILE"
else
    EXCEL_FILE="$DEFAULT_EXCEL_FILE"
    echo "📁 使用默认 Excel 文件: $EXCEL_FILE"
fi

# 检查 Excel 文件是否存在
if [[ ! -f "$EXCEL_FILE" ]]; then
    echo "❌ 错误: Excel 文件不存在: $EXCEL_FILE"
    exit 1
fi

# 生成 Excel 文件的唯一标识（用于进度文件命名）
EXCEL_BASENAME=$(basename "$EXCEL_FILE" .xlsx)
EXCEL_IDENTIFIER=$(echo "$EXCEL_BASENAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g')
echo "🏷️  Excel 标识: $EXCEL_IDENTIFIER"

# 设置输出目录
if [[ -n "$CUSTOM_OUTDIR" ]]; then
    OUTDIR="$CUSTOM_OUTDIR"
else
    # 根据 Excel 文件名生成默认输出目录
    OUTDIR="${EXCEL_IDENTIFIER}_result"
fi
# 创建输出目录
mkdir -p "$OUTDIR"
echo "📂 输出目录: $OUTDIR"

# 设置 golden 编译状态
if [[ $GOLDEN_ONLY_MODE -eq 1 ]]; then
    NEED_GOLDEN=1
    echo "🔧 模式: 只编译 Golden，不执行 qrun"
elif [[ $GOLDEN_MODE -eq 1 ]]; then
    NEED_GOLDEN=1
    echo "🔧 手动指定编译 golden（-g），每个批次执行前将编译 golden（本地执行，不占用 NPU）"
elif [[ -n "$CUSTOM_EXCEL_FILE" ]]; then
    NEED_GOLDEN=1
    echo "🔧 检测到自定义 Excel 文件，每个批次执行前将编译 golden（本地执行，不占用 NPU）"
else
    NEED_GOLDEN=0
    echo "ℹ️  使用默认 Excel 文件，跳过 golden 编译（可用 -g 强制编译，-G 只编译不执行）"
fi

# 设置要跑的 Sheets
if [[ ${#SPECIFIED_SHEETS[@]} -gt 0 ]]; then
    SHEETS_TO_RUN=("${SPECIFIED_SHEETS[@]}")
    READ_PROGRESS=0  # 指定 Sheet 模式：不读取旧进度，但会保存新进度
    echo "模式: 指定 Sheets 模式 [${SHEETS_TO_RUN[*]}]"
else
    # 如果指定了 Excel 文件但没有指定 Sheet，尝试从 Python 获取所有 Sheet
    if [[ -n "$CUSTOM_EXCEL_FILE" ]] && [[ -f "$PYTHON_HELPER" ]]; then
        echo "📊 尝试从 Excel 文件获取所有 Sheet 列表..."
        ALL_SHEETS=$(python3 "$PYTHON_HELPER" "$EXCEL_FILE" --list-sheets 2>/dev/null)
        if [[ $? -eq 0 && -n "$ALL_SHEETS" ]]; then
            # 将输出转换为数组
            IFS=$'\n' read -r -d '' -a SHEETS_TO_RUN < <(echo "$ALL_SHEETS" && printf '\0')
            echo "✅ 从 Excel 获取到 ${#SHEETS_TO_RUN[@]} 个 Sheets: [${SHEETS_TO_RUN[*]}]"
        else
            SHEETS_TO_RUN=("${DEFAULT_SHEETS[@]}")
            echo "⚠️  无法获取 Sheet 列表，使用默认配置: [${SHEETS_TO_RUN[*]}]"
        fi
    else
        SHEETS_TO_RUN=("${DEFAULT_SHEETS[@]}")
        echo "模式: 默认全量模式 [${SHEETS_TO_RUN[*]}]"
    fi
    READ_PROGRESS=1  # 默认模式：读取旧进度，支持断点续跑
fi

# 处理强制重跑模式
if [[ $RESTART_MODE -eq 1 ]]; then
    echo "操作: 清理 Excel [$EXCEL_IDENTIFIER] 的所有进度文件、完成标记和日志（强制重跑）..."
    rm -f "$OUTDIR"/.global_progress_${EXCEL_IDENTIFIER}.txt 
    rm -f "$OUTDIR"/.batch_progress_${EXCEL_IDENTIFIER}_*.txt 
    rm -f "$OUTDIR"/.completion_${EXCEL_IDENTIFIER}_*.txt
    rm -f "$OUTDIR"/mega_moe_${EXCEL_IDENTIFIER}_*.log
    rm -f "$OUTDIR"/golden_build_${EXCEL_IDENTIFIER}_*.log
    READ_PROGRESS=0  # 不读取旧进度
fi

# 使用 Excel 标识符的进度文件
GLOBAL_PROGRESS_FILE="$OUTDIR/.global_progress_${EXCEL_IDENTIFIER}.txt"

# ============================================
# 检查 Python 辅助脚本
# ============================================
PYTHON_AVAILABLE=0
if [[ -f "$PYTHON_HELPER" ]]; then
    if python3 "$PYTHON_HELPER" --help >/dev/null 2>&1; then
        PYTHON_AVAILABLE=1
        echo "✅ Python 辅助脚本可用: $PYTHON_HELPER"
    else
        echo "⚠️  Python 辅助脚本执行失败，将需要手动指定行范围"
    fi
else
    echo "⚠️  未找到 Python 辅助脚本 ($PYTHON_HELPER)"
    echo "   建议: 安装 Python 辅助脚本以自动检测行数，或使用 --start-row 和 --end-row 手动指定"
fi
echo ""

# ============================================
# 确定 Sheet 的行范围（使用 Python 动态获取并验证）
# ============================================
get_sheet_config() {
    local sheet_name=$1
    local start_row=$DEFAULT_START_ROW
    local end_row=""
    local python_args=""
    
    # 构建 Python 脚本参数
    if [[ -n "$SPECIFIED_START_ROW" ]]; then
        python_args="$python_args --start-row $SPECIFIED_START_ROW"
    fi
    if [[ -n "$SPECIFIED_END_ROW" ]]; then
        python_args="$python_args --end-row $SPECIFIED_END_ROW"
    fi
    
    # 尝试使用 Python 动态获取并验证行数
    if [[ $PYTHON_AVAILABLE -eq 1 ]]; then
        echo "📊 正在获取并验证 Sheet [$sheet_name] 的行范围..." >&2
        
        # 调用 Python 脚本获取行数信息（JSON 格式）
        local sheet_info=$(python3 "$PYTHON_HELPER" "$EXCEL_FILE" -s "$sheet_name" $python_args --json 2>/tmp/python_error_$$.log)
        local python_exit_code=$?
        
        if [[ $python_exit_code -eq 0 && -n "$sheet_info" ]]; then
            # 检查是否有错误
            local is_valid=$(echo "$sheet_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('valid', False))" 2>/dev/null)
            
            if [[ "$is_valid" == "True" ]]; then
                # 提取最终的行范围
                start_row=$(echo "$sheet_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['start_row'])" 2>/dev/null)
                end_row=$(echo "$sheet_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['end_row'])" 2>/dev/null)
                local total_rows=$(echo "$sheet_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['total_rows'])" 2>/dev/null)
                local max_row=$(echo "$sheet_info" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['max_row'])" 2>/dev/null)
                
                # 显示警告信息（输出到 stderr）
                local warnings=$(echo "$sheet_info" | python3 -c "
import sys, json
data = json.load(sys.stdin)
warnings = data.get('warnings', [])
for w in warnings:
    print(w)
" 2>/dev/null)
                
                if [[ -n "$warnings" ]]; then
                    echo "$warnings" | while IFS= read -r line; do
                        if [[ -n "$line" ]]; then
                            echo "⚠️  $line" >&2
                        fi
                    done
                fi
                
                echo "✅ Sheet [$sheet_name]: 有效行范围 2-$max_row，最终使用 $start_row - $end_row (共 $total_rows 行数据)" >&2
            else
                # 验证失败，显示错误（输出到 stderr）
                local errors=$(echo "$sheet_info" | python3 -c "
import sys, json
data = json.load(sys.stdin)
errors = data.get('errors', [])
for e in errors:
    print(e)
" 2>/dev/null)
                
                echo "❌ Sheet [$sheet_name] 行范围验证失败:" >&2
                echo "$errors" | while IFS= read -r line; do
                    if [[ -n "$line" ]]; then
                        echo "   $line" >&2
                    fi
                done
                
                # 验证失败，需要手动处理
                end_row=""
            fi
        else
            echo "⚠️  Python 脚本执行失败，回退到手动配置" >&2
            if [[ -f /tmp/python_error_$$.log ]]; then
                local error_msg=$(cat /tmp/python_error_$$.log)
                if [[ -n "$error_msg" ]]; then
                    # 检查是否是 Sheet 不存在的错误
                    if echo "$error_msg" | grep -qi "not found"; then
                        echo "" >&2
                        echo "╔════════════════════════════════════════════════════════╗" >&2
                        echo "║  ❌ 错误: Sheet 不存在！                               ║" >&2
                        echo "║                                                        ║" >&2
                        printf "║  未找到 Sheet: %-41s ║\n" "$sheet_name" >&2
                        echo "║                                                        ║" >&2
                        echo "║  请检查 Sheet 名称是否拼写正确                          ║" >&2
                        echo "║                                                        ║" >&2
                        
                        # 尝试列出可用的 Sheets
                        local available_sheets=$(python3 "$PYTHON_HELPER" "$EXCEL_FILE" --list-sheets 2>/dev/null)
                        if [[ -n "$available_sheets" ]]; then
                            echo "║  可用的 Sheets:                                        ║" >&2
                            echo "$available_sheets" | while IFS= read -r s; do
                                if [[ -n "$s" ]]; then
                                    printf "║    - %-50s ║\n" "$s" >&2
                                fi
                            done
                        fi
                        echo "║                                                        ║" >&2
                        echo "║  提示: 检查是否有多余的字符或拼写错误                   ║" >&2
                        echo "╚════════════════════════════════════════════════════════╝" >&2
                        echo "" >&2
                    else
                        echo "   错误详情: $error_msg" >&2
                    fi
                fi
                rm -f /tmp/python_error_$$.log
            fi
            end_row=""
        fi
    fi
    
    # 如果 Python 获取失败或验证失败，需要手动处理
    if [[ -z "$end_row" ]]; then
        # 先处理用户指定的开始行
        if [[ -n "$SPECIFIED_START_ROW" ]]; then
            echo "ℹ️  使用手动指定的开始行: $SPECIFIED_START_ROW（覆盖默认值 $start_row）" >&2
            start_row="$SPECIFIED_START_ROW"
        else
            echo "ℹ️  使用默认开始行: $start_row" >&2
        fi
        
        # 再检查结束行
        if [[ -z "$SPECIFIED_END_ROW" ]]; then
            echo "❌ 错误: 无法获取 Sheet [$sheet_name] 的结束行" >&2
            echo "   当前设置:" >&2
            echo "   - 开始行: $start_row" >&2
            echo "   解决方法:" >&2
            echo "   1. 安装并使用 Python 辅助脚本以自动检测行数" >&2
            echo "   2. 使用 --end-row 参数手动指定结束行" >&2
            echo "   例如: $0 -s $sheet_name --start-row $start_row --end-row 100" >&2
            exit 1
        else
            end_row="$SPECIFIED_END_ROW"
            echo "ℹ️  使用手动指定的结束行: $end_row" >&2
        fi
    else
        # Python 成功获取了 end_row，再处理用户指定的开始行（如果有）
        if [[ -n "$SPECIFIED_START_ROW" ]]; then
            echo "ℹ️  使用手动指定的开始行: $SPECIFIED_START_ROW（覆盖自动检测值 $start_row）" >&2
            start_row="$SPECIFIED_START_ROW"
        fi
    fi
    
    # 最终验证行范围
    if [[ $start_row -gt $end_row ]]; then
        echo "❌ 错误: 开始行 ($start_row) 不能大于结束行 ($end_row)" >&2
        exit 1
    fi
    
    # 只输出行范围到 stdout（这是返回值）
    echo "$start_row $end_row"
}

# ============================================
# 主程序
# ============================================
echo "========================================"
echo "配置信息:"
echo "  Excel 文件: $EXCEL_FILE"
echo "  Excel 标识: $EXCEL_IDENTIFIER"
echo "  输出目录: $OUTDIR"
echo "  批次大小: $BATCH_SIZE"
echo "  间隔时间: ${INTERVAL}秒"
echo "  运行模式: $MODE"
echo "  读取进度: $([ $READ_PROGRESS -eq 1 ] && echo '是' || echo '否')"
echo "  Python 辅助: $([ $PYTHON_AVAILABLE -eq 1 ] && echo '可用（自动检测行数）' || echo '不可用（需手动指定行范围）')"
echo "  进度文件前缀: ${EXCEL_IDENTIFIER}"
echo "  Golden 编译: $([ $GOLDEN_ONLY_MODE -eq 1 ] && echo '仅编译（不执行 qrun）' || ([ $NEED_GOLDEN -eq 1 ] && echo '需要（按批次编译）' || echo '跳过'))"
echo "========================================"

# 检查是否从特定 Sheet 开始（仅在读取进度模式下）
START_SHEET_INDEX=0
if [[ $READ_PROGRESS -eq 1 && -f "$GLOBAL_PROGRESS_FILE" ]]; then
    START_SHEET_INDEX=$(cat "$GLOBAL_PROGRESS_FILE")
    if [[ "$START_SHEET_INDEX" -ge "${#SHEETS_TO_RUN[@]}" ]]; then
        echo "✅ 所有 Sheet 已处理完成！"
        exit 0
    fi
    echo "📌 断点续跑: 从第 $((START_SHEET_INDEX + 1)) 个 Sheet [${SHEETS_TO_RUN[$START_SHEET_INDEX]}] 开始"
elif [[ $READ_PROGRESS -eq 0 ]]; then
    echo "📌 忽略历史进度: 从头开始运行"
fi

# ============================================
# 遍历 Sheets
# ============================================
for ((sheet_idx=START_SHEET_INDEX; sheet_idx<${#SHEETS_TO_RUN[@]}; sheet_idx++)); do
    SHEET_NAME="${SHEETS_TO_RUN[$sheet_idx]}"
    
    # 获取该 Sheet 的行范围（自动检测并验证）
    read START_ROW END_ROW <<< "$(get_sheet_config "$SHEET_NAME")"
    
    # 使用 Excel 标识符的设置该 Sheet 的日志、进度文件和完成标记
    LOG_FILE="$OUTDIR/mega_moe_${EXCEL_IDENTIFIER}_${SHEET_NAME}.log"
    PROGRESS_FILE="$OUTDIR/.batch_progress_${EXCEL_IDENTIFIER}_${SHEET_NAME}.txt"
    COMPLETION_MARKER="$OUTDIR/.completion_${EXCEL_IDENTIFIER}_${SHEET_NAME}.txt"
    GOLDEN_LOG="$OUTDIR/golden_build_${EXCEL_IDENTIFIER}_${SHEET_NAME}.log"
    
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║  Sheet [$SHEET_NAME] (Excel: $EXCEL_IDENTIFIER)        "
    echo "║  行范围: $START_ROW - $END_ROW                         "
    if [[ $GOLDEN_ONLY_MODE -eq 0 ]]; then
        echo "║  日志: $LOG_FILE                                       "
    else
        echo "║  Golden 日志: $GOLDEN_LOG                              "
    fi
    echo "╚════════════════════════════════════════════════════════╝"
    
    # 检查该Sheet是否已完成（仅在读取进度模式下，且不是 golden-only 模式）
    if [[ $GOLDEN_ONLY_MODE -eq 0 && $READ_PROGRESS -eq 1 && -f "$COMPLETION_MARKER" ]]; then
        echo "✅ Sheet [$SHEET_NAME] 已完成（标记时间: $(cat $COMPLETION_MARKER)），跳过..."
        continue
    fi
    
    # 确定起始行（golden-only 模式总是从头开始）
    if [[ $GOLDEN_ONLY_MODE -eq 1 ]]; then
        current_row=$START_ROW
        echo "🔧 Golden-only 模式: 从第 $current_row 行开始编译"
    elif [[ $READ_PROGRESS -eq 1 && -f "$PROGRESS_FILE" ]]; then
        current_row=$(cat "$PROGRESS_FILE")
        if [[ "$current_row" -lt "$START_ROW" ]] || [[ "$current_row" -gt "$END_ROW" ]]; then
            echo "⚠️  警告: 进度文件行号异常 ($current_row)，重置为 $START_ROW"
            current_row=$START_ROW
        else
            echo "📌 断点续跑: 从第 $current_row 行继续"
        fi
    else
        current_row=$START_ROW
        if [[ $READ_PROGRESS -eq 0 ]]; then
            echo "🆕 全新运行: 从第 $current_row 行开始"
        else
            echo "🆕 新任务: 从第 $current_row 行开始"
        fi
    fi
    
    # 计算总批次数
    total_cases=$(( END_ROW - current_row + 1 ))
    total_batches=$(( (total_cases + BATCH_SIZE - 1) / BATCH_SIZE ))
    current_batch=1
    
    echo "📊 共 $total_cases 条用例，分 $total_batches 批次执行"
    
    # ============================================
    # 内层循环：按批次执行
    # ============================================
    while [[ $current_row -le $END_ROW ]]; do
        batch_end=$(( current_row + BATCH_SIZE - 1 ))
        if [[ $batch_end -gt $END_ROW ]]; then
            batch_end=$END_ROW
        fi
        
        # 批次间隔（除第一批外，golden-only 模式不需要间隔）
        if [[ $GOLDEN_ONLY_MODE -eq 0 && $current_batch -gt 1 ]]; then
            echo "[$SHEET_NAME] ⏳ 等待 ${INTERVAL}秒..."
            sleep $INTERVAL
        fi
        
        # ============================================
        # Golden 编译（需要时，编译当前批次的行范围）
        # ============================================
        if [[ $NEED_GOLDEN -eq 1 ]]; then
            echo "[$SHEET_NAME] 🔨 编译 Golden: 行 $current_row-$batch_end ..."
            python3 mega_moe_once.py -excel "$EXCEL_FILE" -sheet "$SHEET_NAME" -s "$current_row" -e "$batch_end" -m "$MODE" -o "$OUTDIR" -g 2>&1 | tee -a "$GOLDEN_LOG"
            
            if [[ $? -ne 0 ]]; then
                echo "❌ Golden 编译失败 (Sheet: $SHEET_NAME, 行: $current_row-$batch_end)，终止执行"
                echo "   日志: $GOLDEN_LOG"
                exit 1
            fi
            echo "[$SHEET_NAME] ✅ Golden 编译成功"
        fi
        
        # 如果是 golden-only 模式，跳过 qrun
        if [[ $GOLDEN_ONLY_MODE -eq 1 ]]; then
            # 更新计数器
            current_row=$(( batch_end + 1 ))
            current_batch=$(( current_batch + 1 ))
            continue
        fi
        
        echo "----------------------------------------"
        echo "[$SHEET_NAME] 📦 批次 $current_batch/$total_batches | 行 $current_row-$batch_end | $(date '+%H:%M:%S')"
        
        # 执行命令
        qrun -n "MOE_${EXCEL_IDENTIFIER}_${SHEET_NAME}_B${current_batch}" \
            "python3 mega_moe_once.py -excel $EXCEL_FILE -sheet $SHEET_NAME -s $current_row -e $batch_end -m $MODE -o $OUTDIR 2>&1 | tee -a $LOG_FILE"
        
        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo "[$SHEET_NAME] ✅ 成功"
        else
            echo "[$SHEET_NAME] ❌ 失败 (码: $exit_code)，继续下一批"
        fi
        
        # 保存进度（始终执行）
        next_row=$(( batch_end + 1 ))
        echo $next_row > "$PROGRESS_FILE"
        
        # 更新计数器
        current_row=$(( batch_end + 1 ))
        current_batch=$(( current_batch + 1 ))
    done
    
    echo ""
    if [[ $GOLDEN_ONLY_MODE -eq 1 ]]; then
        echo "✅ [$SHEET_NAME] Golden 编译完成！共处理 $total_cases 条用例"
    else
        echo "✅ [$SHEET_NAME] 完成！共处理 $total_cases 条用例"
        
        # 标记该Sheet完成
        echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$COMPLETION_MARKER"
        echo "📝 Sheet [$SHEET_NAME] 已完成并标记"
        
        # 更新全局进度（始终执行）
        next_sheet_idx=$((sheet_idx + 1))
        if [[ $next_sheet_idx -lt ${#SHEETS_TO_RUN[@]} ]]; then
            echo $next_sheet_idx > "$GLOBAL_PROGRESS_FILE"
            echo "📌 全局进度已更新: 下一个 Sheet 索引为 $next_sheet_idx (${SHEETS_TO_RUN[$next_sheet_idx]})"
        else
            # 所有Sheet完成，清理所有进度文件
            echo ""
            echo "════════════════════════════════════════════════════════"
            echo "🎉 所有 Sheet 处理完成！清理 Excel [$EXCEL_IDENTIFIER] 的进度文件..."
            echo "════════════════════════════════════════════════════════"
            rm -f "$GLOBAL_PROGRESS_FILE"
            for sheet in "${SHEETS_TO_RUN[@]}"; do
                rm -f "$OUTDIR/.batch_progress_${EXCEL_IDENTIFIER}_${sheet}.txt"
                rm -f "$OUTDIR/.completion_${EXCEL_IDENTIFIER}_${sheet}.txt"
            done
            echo "🧹 进度文件清理完毕"
        fi
    fi
    
    # Sheet 之间间隔（golden-only 模式不需要间隔）
    if [[ $GOLDEN_ONLY_MODE -eq 0 && $sheet_idx -lt $(( ${#SHEETS_TO_RUN[@]} - 1 )) ]]; then
        echo ""
        echo "⏳ 等待 ${INTERVAL}秒 切换到下一 Sheet..."
        sleep $INTERVAL
    fi
    
done

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║           全部执行完毕！                                ║"
echo "║           Excel: $EXCEL_IDENTIFIER                      ║"
echo "║           时间: $(date '+%Y-%m-%d %H:%M:%S')           ║"
echo "╚════════════════════════════════════════════════════════╝"