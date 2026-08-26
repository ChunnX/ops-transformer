#!/bin/bash
# 跑 Excel 全部 enable=TRUE 用例的 PTO vs AscendC 纯 kernel 性能（各 10 次）。
#
# 用法（就这一条）：
#   bash run_fag_msprof.sh
#
# 可选：
#   bash run_fag_msprof.sh --filter bn2_multihead
#   bash run_fag_msprof.sh --list
#
# 跑完后看这些文件即可（在 /home/kjc/kjc_tmp/fag_msprof_binary/）：
#   all_cases_10runs.csv     ← 每个用例、两侧、每次运行一行（推荐先看这个）
#   summary_mean.csv         ← 均值汇总
#   <case>/op_summary.csv    ← 该用例唯一一份 msprof op_summary
#   <case>/kernel_10runs.csv ← 该用例 10 次明细
#   <case>/PROF_*/           ← msprof 原始目录（需要时再翻）
set -e
code_path=/home/kjc/ops-transformer
cann_path=/home/kjc/kjc_cann_0819
source $cann_path/cann-9.2.0/set_env.sh
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3


cd $code_path
rm -rf build
bash ${code_path}/build.sh --op_debug_config dump_cce --soc=ascend950 --ops=flash_attn_metadata,flash_attn_grad --pkg
cd $code_path/build
./cann-ops-transformer-custom_linux-x86_64.run --install-path=$cann_path
CANN_HOME="${CANN_HOME:-/home/kjc/kjc_cann_0819}"
REPO="${REPO:-/home/kjc/ops-transformer}"
KERNEL_DIR="${REPO}/attention/flash_attn_grad/op_kernel"
SCRIPT="${KERNEL_DIR}/test_flash_attn_grad-binary-msprof.py"
OUT="${FAG_MSPROF_DIR:-/home/kjc/kjc_tmp/fag_msprof_binary}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-kjc}"
source "${CANN_HOME}/cann-9.2.0/set_env.sh"
source "${CANN_HOME}/vendors/custom_transformer/bin/set_env.bash"

export PYTHONPATH="${REPO}/torch_extension:${CANN_HOME}/cann-9.2.0/python/site-packages"
export PYTHONUNBUFFERED=1
export MSPROF="${MSPROF:-${CANN_HOME}/cann-9.2.0/tools/profiler/bin/msprof}"
export ASCEND_SLOG_PRINT_TO_STDOUT="${ASCEND_SLOG_PRINT_TO_STDOUT:-0}"
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-/home/kjc/kjc_tmp/fag_plog}"
export FAG_MSPROF_DIR="$OUT"
export SQLITE_LIB="${SQLITE_LIB:-/root/.conda/envs/zyj/lib/libsqlite3.so.0}"
[[ -e "$SQLITE_LIB" ]] || export SQLITE_LIB="/root/.conda/envs/kjc/lib/libsqlite3.so.0"

mkdir -p "$OUT" "$ASCEND_PROCESS_LOG_PATH"
find /root/.cache/torch_extensions -name lock -mmin +1 -delete 2>/dev/null || true

cd "$KERNEL_DIR"
# 默认 10 次；外面若传了 --iters 会覆盖。
# 不做 host-warmup；iters>1 时脚本均值会丢掉第一次。
exec python -u "$SCRIPT" --mode perf-suite --iters 10 "$@"
