#!/bin/bash
set -euo pipefail

# 1、Profiler后台单线程上报，降低多进程锁竞争
export PROF_WORKER_NUM=1
# 2、限制Profiler上报队列最大长度，避免埋点堆积触发ACL接口限流报错
export PROF_MAX_QUEUE_LEN=1024
# 3、禁用多进程硬件指标并发上报，等价代码SingleProcess效果，环境变量全局生效
export PROF_AIC_METRICS_SINGLE_PROCESS=1

# 定义脚本需要的必填列（使用完整带括号的列名，统一只在这里维护）
REQUIRED_COLS=(
    "case_name"
    "input_tensor_shape"
    "input_tensor_dtype"
    "input_tensor_range"
    "fetch_indicies"
    "world_size"
)
# 可选列统一在这里管理
OPT_COLS=(
    "num_cpu_bytes"
    "is_graph"
    "is_profiling"
    "use_fp8"
    "sf_dtype"
    "seed"
    "fetch_mode"
)

# 命令行参数解析
xlsx_file=""
sheet_name=""
num_start=""
num_end=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --xlsx)
            xlsx_file="$2"
            shift 2
            ;;
        --sheet)
            sheet_name="$2"
            shift 2
            ;;
        --num)
            # 支持传1个或2个正整数
            if [[ $# -ge 3 && "$2" =~ ^[0-9]+$ && "$3" =~ ^[0-9]+$ ]]; then
                num_start="$2"
                num_end="$3"
                shift 3
            else
                num_start="$2"
                num_end="$2"
                shift 2
            fi
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            echo "用法1: bash run_engram.sh --xlsx engram_new.xlsx --sheet test --num 1"
            echo "用法2: bash run_engram.sh --xlsx engram_new.xlsx --sheet test --num 1 20"
            exit 1
            ;;
    esac
done

# 必选参数校验
if [[ -z "$xlsx_file" || -z "$sheet_name" || -z "$num_start" || -z "$num_end" ]]; then
    echo "[ERROR] 缺少必选参数"
    echo "用法1: bash run_engram.sh --xlsx engram_new.xlsx --sheet test --num 1"
    echo "用法2: bash run_engram.sh --xlsx engram_new.xlsx --sheet test --num 1 20"
    exit 1
fi

# 行号数字校验
if ! [[ "$num_start" =~ ^[1-9][0-9]*$ && "$num_end" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num 必须传入正整数"
    exit 1
fi

# 行号范围校验
if [[ "$num_start" -gt "$num_end" ]]; then
    echo "[ERROR] 起始序号不能大于结束序号"
    exit 1
fi

# 文件存在校验
if [[ ! -f "$xlsx_file" ]]; then
    echo "[ERROR] xlsx文件不存在: $xlsx_file"
    exit 1
fi

# 计算pandas索引（Excel第2行对应index 0）
start_idx=$((num_start - 1))
end_idx=$((num_end - 1))

# 把列名用换行分隔存入环境变量，避免空格分割错乱
export COLS_ENV=$(printf "%s\n" "${REQUIRED_COLS[@]}")
export OPT_COLS_ENV=$(printf "%s\n" "${OPT_COLS[@]}")

read_data() {
    python - <<EOF
import pandas as pd
import sys
import os

# 整数安全转换：NaN/空串/nan/none/纯空格按0处理，其余强转int，非法值报明确错误
def to_int(col, x):
    try:
        if pd.isna(x):
            return 0
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none"):
            return 0
        return int(float(x))
    except Exception:
        raise ValueError(f"[ERROR] 列 {col} 存在非法整数值: {x!r}")

# 按换行分割还原每一个原始列名
cols_list = os.environ["COLS_ENV"].splitlines()
opt_cols = os.environ["OPT_COLS_ENV"].splitlines()

df = pd.read_excel(
    "$xlsx_file",
    sheet_name="$sheet_name",
    header=0
)

# 校验必填列是否全部存在
missing_cols = [col for col in cols_list if col not in df.columns]
if missing_cols:
    print(f"[ERROR] Excel表缺少必填列: {', '.join(missing_cols)}", file=sys.stderr)
    sys.exit(1)

# 校验行号是否超出数据范围
if $end_idx >= len(df):
    print(f"[ERROR] 行号超出范围，表格中共有 {len(df)} 条有效数据", file=sys.stderr)
    sys.exit(1)

int_cols = ["world_size", "num_cpu_bytes", "is_graph", "is_profiling", "use_fp8", "seed"]  # fetch_mode is string, not int

# 读取必填列
df_required = df[cols_list].copy()

# 遍历所有可选列，存在则读取，不存在默认填充0，空值也填充0
for col in opt_cols:
    if col in df.columns:
        df_required[col] = df[col].fillna(0)
    else:
        df_required[col] = 0

# 强制将数值列转为int整数（健壮转换：NaN/空串/nan/none按0，非法值报列名明确的错误）
for col in int_cols:
    if col in df_required.columns:
        df_required[col] = df_required[col].apply(lambda x: to_int(col, x))

# 按行输出参数，用|分隔，顺序：必填列 + 所有可选列
for idx in range($start_idx, $end_idx + 1):
    row = df_required.iloc[idx]
    # sf_dtype归一化: 空/NaN/0/0.0 视为未设置(输出"0", 不传给脚本); 仅真dtype字符串才透传
    sf_val = row["sf_dtype"]
    if pd.isna(sf_val) or str(sf_val).strip() in ("", "0", "0.0"):
        sf_str = "0"
    else:
        sf_str = str(sf_val)
    # fetch_mode 归一化: 空/NaN/0/0.0 视为默认 sliced; 接受 sliced/local/0/1
    fm_val = row["fetch_mode"]
    if pd.isna(fm_val) or str(fm_val).strip() in ("", "0", "0.0"):
        fm_str = "sliced"
    else:
        fm_str = str(fm_val).strip()

    # 去除所有字段中的换行符/<br>标签(替换为空格)，字段内"|"替换为";"，避免 | 分隔的按行解析被截断
    def _clean(v):
        s = str(v).replace(chr(10), ' ').replace(chr(13), '')
        s = s.replace('<br />', ' ').replace('<br/>', ' ').replace('<br>', ' ')
        s = s.replace('|', ';')
        return s

    fields = [
        _clean(row["case_name"]),
        _clean(row["input_tensor_shape"]),
        _clean(row["input_tensor_dtype"]),
        _clean(row["input_tensor_range"]),
        _clean(row["fetch_indicies"]),
        str(int(row["world_size"])),
        str(int(row["num_cpu_bytes"])),
        str(int(row["is_graph"])),
        str(int(row["is_profiling"])),
        str(int(row["use_fp8"])),
        _clean(sf_str),
        str(int(row["seed"])),
        _clean(fm_str)
    ]
    print("|".join(fields))
EOF
}

# 执行读取，捕获错误（python 的错误信息直接走 stderr 透传给用户）
if ! data_output=$(read_data); then
    echo "[ERROR] 读取用例表失败"
    exit 1
fi

# 循环执行用例
total_cases=$((num_end - num_start + 1))
current_case=0
SUCCESS_CASES=()
FAILED_CASES=()
echo "[INFO] 共读取到 $total_cases 条测试用例，开始执行..."
echo "================================================================"

while IFS="|" read -r case_name input_tensor_shape input_tensor_dtype input_tensor_range fetch_indicies world_size num_cpu_bytes is_graph is_profiling use_fp8 sf_dtype seed fetch_mode; do
    current_case=$((current_case + 1))
    # profiling 开启时创建独立临时目录（py 端读取 PROF_TMP_DIR 作为 profiling 输出根目录）
    if [[ "${is_profiling}" -eq 1 ]]; then
        export PROF_TMP_DIR=$(mktemp -d)
        echo "[INFO] 已开启Profiling，当前用例独立输出目录: ${PROF_TMP_DIR}"
    else
        unset PROF_TMP_DIR
        echo "[INFO] 未开启Profiling，不配置Profiler临时目录"
    fi
    # 执行前完整打印读取到的参数
    echo "------------------------------------------------"
    echo "[CASE $current_case/$total_cases] 读取到当前用例参数："
    echo "  case_name:             $case_name"
    echo "  input_tensor_shape:    $input_tensor_shape"
    echo "  input_tensor_dtype:    $input_tensor_dtype"
    echo "  input_tensor_range:    $input_tensor_range"
    echo "  fetch_indicies:        $fetch_indicies"
    echo "  world_size:            $world_size"
    echo "  num_cpu_bytes:         $num_cpu_bytes"
    echo "  is_graph:              $is_graph"
    echo "  is_profiling:          $is_profiling"
    echo "  use_fp8:               $use_fp8"
    echo "  sf_dtype:              $sf_dtype"
    echo "  seed:                  $seed"
    echo "  fetch_mode:            $fetch_mode"
    echo "------------------------------------------------"

    # 组装命令参数
    cmd_args=(
        python engram_test.py
        --case_name "$case_name"
        --input_tensor_shape "$input_tensor_shape"
        --input_tensor_dtype "$input_tensor_dtype"
        --input_tensor_range "$input_tensor_range"
        --fetch_indicies "$fetch_indicies"
        --world_size "$world_size"
        --master_port $((50008 + current_case))
        --prep_cache_dir ./prep_cache
    )
    # 可选参数
    if [[ "$num_cpu_bytes" != "0" ]]; then
        cmd_args+=(--num_cpu_bytes "$num_cpu_bytes")
    fi
    # 可选参数
    if [[ "$is_graph" != "0" ]]; then
        cmd_args+=(--is_graph "$is_graph")
    fi
    if [[ "$is_profiling" != "0" ]]; then
        cmd_args+=(--is_profiling "$is_profiling")
    fi
    if [[ "$use_fp8" != "0" ]]; then
        cmd_args+=(--use_fp8 "$use_fp8")
    fi
    if [[ "$sf_dtype" != "0" && -n "$sf_dtype" ]]; then
        cmd_args+=(--sf_dtype "$sf_dtype")
    fi
    if [[ "$seed" != "0" ]]; then
        cmd_args+=(--seed "$seed")
    fi
    if [[ -n "$fetch_mode" ]]; then
        cmd_args+=(--fetch_mode "$fetch_mode")
    fi

    # 打印即将执行的完整命令
    echo "执行命令："
    echo "${cmd_args[@]}"
    echo "------------------------------------------------"

    # ===== Phase 1: 数据准备 (不排队，写 mmap prep 缓存)，容错模式：失败记录后继续下一条用例 =====
    echo "[PREP] 开始数据准备: $case_name"
    prep_ret=0
    "${cmd_args[@]}" --prep_only || prep_ret=$?
    if [[ $prep_ret -ne 0 ]]; then
        echo "[ERROR] 数据准备失败: $case_name, 返回码: $prep_ret"
        FAILED_CASES+=("${case_name}(prep:${prep_ret})")
        echo "[INFO] 当前测试用例【${case_name}】性能数据存放路径：${PROF_TMP_DIR:-$PWD/profiling_data/${case_name}_profiling}"
        echo "================================================================"
        continue
    fi
    echo "[PREP] 数据准备完成: $case_name"

    # ===== Phase 2: NPU执行 (qrun排队)，容错模式：失败记录后继续下一条用例 =====
    echo "[RUN] 排队等待NPU执行: $case_name"
    exec_ret=0
    qrun --emp q00983190 -n "${case_name}" "${cmd_args[@]}" --run_only || exec_ret=$?
    if [[ $exec_ret -ne 0 ]]; then
        echo "[ERROR] 用例 $case_name 执行失败，返回码: $exec_ret"
        FAILED_CASES+=("${case_name}(run:${exec_ret})")
    else
        echo "[SUCCESS] 用例 $case_name 执行完成"
        SUCCESS_CASES+=("$case_name")
    fi

    echo "------------------------------------------------"

    echo "[INFO] 当前测试用例【${case_name}】性能数据存放路径：${PROF_TMP_DIR:-$PWD/profiling_data/${case_name}_profiling}"
    echo "================================================================"
done <<< "$data_output"

echo "[FINISH] 全部 $total_cases 条测试用例执行完毕"
echo "================================================================"
# prep 缓存(mmap)保留: 供 local/sliced 配对用例与重跑复用, 不随批跑清理
if [[ -d ./prep_cache ]]; then
    echo "[INFO] prep_cache 缓存目录大小: $(du -sh ./prep_cache 2>/dev/null)"
fi
echo "[INFO] prep_cache 缓存保留供 local/sliced 配对与重跑复用, 手动清理用 --purge_prep_cache"
echo "[SUMMARY] 用例执行结果汇总 (SUCCESS: ${#SUCCESS_CASES[@]} / FAILED: ${#FAILED_CASES[@]} / 总数: $total_cases):"
for c in ${SUCCESS_CASES[@]+"${SUCCESS_CASES[@]}"}; do
    echo "  [SUCCESS] $c"
done
for c in ${FAILED_CASES[@]+"${FAILED_CASES[@]}"}; do
    echo "  [FAILED]  $c"
done
if [[ ${#FAILED_CASES[@]} -gt 0 ]]; then
    echo "[ERROR] 存在失败用例，脚本以非零码退出"
    exit 1
fi
echo "[SUCCESS] 全部用例执行成功"
exit 0
