#!/bin/bash
# 逐 case 运行 msprof，收集每次生成的随机名 msprof 文件夹，
# 提取 OpBasicInfo.csv 里的 Duration(us)，汇总到一个带时间戳的 Excel。
# 用法：bash run_perf_collect.sh

set -e

cd "$(dirname "$0")"

# 检查依赖
python3 -c "import pandas, openpyxl" 2>/dev/null || {
    echo "Error: pandas and openpyxl are required. Install with: pip install pandas openpyxl"
    exit 1
}

# 创建带时间戳的 Excel
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXCEL_FILE="prof_summary_${TIMESTAMP}.xlsx"

# 初始化 Excel（写入表头）
python3 -c "
import pandas as pd
df = pd.DataFrame(columns=['case_id', 'Duration(us)'])
df.to_excel('$EXCEL_FILE', index=False)
"

# 从 xlsx 读取用例列表
CASES=$(python3 -c "
import pandas as pd
df = pd.read_excel('excel/stem_indexer_cases.xlsx')
for n in df['case_id']:
    print(n)
")

TOTAL=$(echo "$CASES" | wc -l)
CURRENT=0

echo "Found $TOTAL testcases"
echo "Summary Excel: $EXCEL_FILE"
echo "=========================================="

while IFS= read -r name; do
    [ -z "$name" ] && continue
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL] Running: $name"
    echo "------------------------------------------"
    
    # 创建标记文件，用于追踪新创建的目录
    MARKER_FILE=$(mktemp)
    touch "$MARKER_FILE"
    
    # 运行 msprof（失败时不退出）
    set +e
    STEM_INDEXER_CASE_ID="$name" msprof op python3 -m pytest -rA -s "test_stem_indexer_batch.py" -v -m ci  -W ignore::UserWarning -W ignore::DeprecationWarning
    set -e
    
    # 查找标记文件之后创建的新目录
    NEW_DIR=$(find . -maxdepth 1 -type d -newer "$MARKER_FILE" 2>/dev/null | grep -v '^\.$' | head -1 || true)
    
    # 清理标记文件
    rm -f "$MARKER_FILE"
    
    if [ -z "$NEW_DIR" ]; then
        echo "  Warning: No new msprof output directory found"
        DURATION="N/A"
    else
        echo "  Found msprof dir: $NEW_DIR"
        
        # 查找 OpBasicInfo.csv
        CSV_FILE=$(find "$NEW_DIR" -name "OpBasicInfo.csv" 2>/dev/null | head -1 || true)
        
        if [ -z "$CSV_FILE" ]; then
            echo "  Warning: OpBasicInfo.csv not found"
            DURATION="N/A"
        else
            echo "  Found CSV: $CSV_FILE"
            
            # 提取 Duration 并追加到 Excel
            DURATION=$(python3 -c "
import pandas as pd
import sys

csv_path = sys.argv[1]
excel_path = sys.argv[2]
tc_name = sys.argv[3]

try:
    csv_df = pd.read_csv(csv_path)
    if 'Duration(us)' in csv_df.columns:
        duration = csv_df['Duration(us)'].sum()
    else:
        duration = 'N/A'
        print('Warning: Duration(us) column not found', file=sys.stderr)
except Exception as e:
    duration = 'N/A'
    print(f'Error reading CSV: {e}', file=sys.stderr)

try:
    excel_df = pd.read_excel(excel_path)
    new_row = pd.DataFrame([{'case_id': tc_name, 'Duration(us)': duration}])
    excel_df = pd.concat([excel_df, new_row], ignore_index=True)
    excel_df.to_excel(excel_path, index=False)
except Exception as e:
    print(f'Error updating Excel: {e}', file=sys.stderr)

print(duration)
" "$CSV_FILE" "$EXCEL_FILE" "$name")
        fi
    fi
    
    echo "  Duration(us): $DURATION"
done <<< "$CASES"

echo ""
echo "=========================================="
echo "Done. Results saved to: $EXCEL_FILE"
