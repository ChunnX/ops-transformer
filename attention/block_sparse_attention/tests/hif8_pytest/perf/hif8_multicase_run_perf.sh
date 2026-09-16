#!/usr/bin/env bash
# CANN/custom set_env scripts read optional variables before defining them, so
# enable nounset only after both vendor environments have been loaded.
set -eo pipefail

# Local .241 performance comparison.  The same case is profiled in HIF8 and
# non-quantized FP16/BF16 mode; bsa_gen_perf_summary.py compares Task Duration.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CANN=${CANN:-/usr/local/Ascend/cann}
OP=${OP:-bsa}
if [[ "$OP" == "qfa" ]]; then
    CUSTOM=${CUSTOM:-/home/liwenguihw/work/custom_install/qfa_hif8_perf}
else
    # in_0916 = 方案 A (L0 共享槽位 256x128) 的包, 2026-09-16 11:30 装。
    # 旧的 in_0824 是改动前的内核, 直接用默认值跑会得到"看起来正常"的旧数据,
    # 所以这里指向新包; 要对比旧内核时显式传 CUSTOM=.../in_0824。
    CUSTOM=${CUSTOM:-/home/liwenguihw/work/custom_install/in_0916}
fi
PYTHON_BIN=${PYTHON_BIN:-/home/liwenguihw/.conda/envs/lwg/bin}
DEVICE=${DEVICE:-0}
BASE_DTYPE=${BASE_DTYPE:-half}
# 950 ALL_HIGH: selects SMF32_REF32_NOMASK_TRANSMM1 for the non-quantized
# baseline when mask_type=0.  Do not change this for this comparison.
INNER_PRECISE=${INNER_PRECISE:-0}
P_SCALE=${P_SCALE:-32}
WARM_UP=${WARM_UP:-20}
SEED=${SEED:-54}
if [[ "$OP" == "qfa" ]]; then
    PERF_ROOT=${PERF_ROOT:-$SCRIPT_DIR/msprof_cache/qfa_hif8}
    LOG_FILE=${LOG_FILE:-$PERF_ROOT/qfa_hif8.log}
else
    PERF_ROOT=${PERF_ROOT:-$SCRIPT_DIR/msprof_cache/hif8_vs_highprec}
    LOG_FILE=${LOG_FILE:-$PERF_ROOT/hif8_vs_highprec.log}
fi
HIF8_DIR="$PERF_ROOT/hif8_dir"
BASE_DIR="$PERF_ROOT/highprec_${BASE_DTYPE}"

if [[ -f "$CANN/set_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "$CANN/set_env.sh"
fi
if [[ -f "$CUSTOM/vendors/custom_transformer/bin/set_env.bash" ]]; then
    # shellcheck disable=SC1090
    source "$CUSTOM/vendors/custom_transformer/bin/set_env.bash"
fi
set -u
export PATH="$PYTHON_BIN:$PATH"
export PYTHONPATH="$($PYTHON_BIN/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'):${PYTHONPATH:-}"
export BSA_CPU_ONLY=0
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
if [[ "$OP" == "qfa" ]]; then
    export CANN_OPS_TRANSFORMER_LOAD_ONLY=quant_flash_attn,quant_flash_attn_metadata
else
    export CANN_OPS_TRANSFORMER_LOAD_ONLY=block_sparse_attention
fi
export BSA_HIF8_IGNORE_VENDOR_PERM=${BSA_HIF8_IGNORE_VENDOR_PERM:-1}
# If visibility is restricted, DEVICE is the logical ID inside that view.
# Otherwise leave the host's device visibility unchanged and use DEVICE as the
# physical/runtime ID, matching the existing local perf reference.
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES
fi
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$PERF_ROOT/torch_extensions}
mkdir -p "$TORCH_EXTENSIONS_DIR"
if [[ "$OP" == "qfa" ]]; then
    mkdir -p "$PERF_ROOT"
else
    mkdir -p "$PERF_ROOT" "$HIF8_DIR" "$BASE_DIR"
fi

# QFA HIF8 uses the same msprof wrapper but has a separate, minimal runner.
# Keep this branch before the BSA matrix so `OP=qfa bash hif8_multicase_run_perf.sh`
# runs only QFA and does not build/launch the BSA baseline.
if [[ "$OP" == "qfa" ]]; then
    QFA_RUNNER="$SCRIPT_DIR/invoke_npu_qfa_hif8_perf.py"
    QFA_LOG_FILE="$LOG_FILE"
    QFA_Q_SEQS=(${QFA_Q_SEQS:-4096})
    if [[ -n "${QFA_KV_SEQS:-}" ]]; then
        read -r -a QFA_KV_SEQS <<< "$QFA_KV_SEQS"
    else
        QFA_KV_SEQS=("${QFA_Q_SEQS[@]}")
    fi
    QFA_PAIRS=()
    if [[ "${QFA_ALLOW_RECTANGULAR:-0}" == 1 ]]; then
        for q_seq in "${QFA_Q_SEQS[@]}"; do
            for kv_seq in "${QFA_KV_SEQS[@]}"; do
                QFA_PAIRS+=("$q_seq $kv_seq")
            done
        done
    else
        if (( ${#QFA_Q_SEQS[@]} != ${#QFA_KV_SEQS[@]} )); then
            echo "QFA_Q_SEQS and QFA_KV_SEQS must have the same length (or set QFA_ALLOW_RECTANGULAR=1)" >&2
            exit 2
        fi
        for idx in "${!QFA_Q_SEQS[@]}"; do
            QFA_PAIRS+=("${QFA_Q_SEQS[$idx]} ${QFA_KV_SEQS[$idx]}")
        done
    fi
    : > "$QFA_LOG_FILE"

    for pair in "${QFA_PAIRS[@]}"; do
        read -r q_seq kv_seq <<< "$pair"
        case_name="case_B1_qS${q_seq}_kvS${kv_seq}_qN1_kvN1_D128_dev${DEVICE}"
        output_dir="$PERF_ROOT/$case_name"
        mkdir -p "$output_dir"
        echo "[QFA HIF8] $case_name" | tee -a "$QFA_LOG_FILE"
        if command -v msprof >/dev/null 2>&1; then
            msprof op --warm-up="$WARM_UP" --aic-metrics=BasicInfo \
                --application="$PYTHON_BIN/python $QFA_RUNNER $q_seq $kv_seq $DEVICE" \
                --output="$output_dir" >>"$QFA_LOG_FILE" 2>&1
        else
            echo "[QFA HIF8] msprof not found; running unprofiled" | tee -a "$QFA_LOG_FILE"
            "$PYTHON_BIN/python" "$QFA_RUNNER" "$q_seq" "$kv_seq" "$DEVICE" \
                >"$output_dir/run.log" 2>&1
            cat "$output_dir/run.log" >>"$QFA_LOG_FILE"
        fi
    done
    echo "[QFA HIF8] log=$QFA_LOG_FILE output=$PERF_ROOT"
    exit 0
fi

if [[ "${CLEAN:-0}" == 1 ]]; then
    rm -rf "$HIF8_DIR" "$BASE_DIR"
    mkdir -p "$HIF8_DIR" "$BASE_DIR"
fi
: > "$LOG_FILE"

if [[ "${FULL_SWEEP:-0}" == 1 ]]; then
    Q_SEQS=(4096 4320 4608 4928 8422 26352 49408 72000 75348 75600 82800 108000 118800 162000 186300 243000)
    KEEP_RATIOS=(0.5 0.2 0.1)
else
    # One quick local case; set FULL_SWEEP=1 for the reference-sized sweep.
    Q_SEQS=(${Q_SEQS:-4096})
    KEEP_RATIOS=(${KEEP_RATIOS:-1.0})
fi
if [[ -n "${KV_SEQS:-}" ]]; then
    read -r -a KV_SEQS <<< "$KV_SEQS"
else
    KV_SEQS=("${Q_SEQS[@]}")
fi
BATCHES=(${BATCHES:-1})
Q_HEADS=(${Q_HEADS:-1})
KV_HEADS=(${KV_HEADS:-1})
HEAD_DIMS=(${HEAD_DIMS:-128})
BLOCK_QS=(${BLOCK_QS:-128})
BLOCK_KVS=(${BLOCK_KVS:-128})
Q_LAYOUTS=(${Q_LAYOUTS:-BNSD})
KV_LAYOUTS=(${KV_LAYOUTS:-BNSD})
VARIED_LENS=(${VARIED_LENS:-0})
VARIED_SPARSITY=(${VARIED_SPARSITY:-0})
ACTUAL_SEQ_AVAIL=(${ACTUAL_SEQ_AVAIL:-0})

RUNNER="$SCRIPT_DIR/invoke_npu_bsa_hif8_perf.py"
cd "$SCRIPT_DIR"

run_variant() {
  local mode="$1"
  local output_dir="$2"
  local -a run_args=("${args[@]}")
  run_args[15]="$mode"
  if command -v msprof >/dev/null 2>&1; then
    msprof op --warm-up="$WARM_UP" \
      --application="$PYTHON_BIN/python $RUNNER ${run_args[*]}" \
      --output="$output_dir" >>"$LOG_FILE" 2>&1
  else
    # .241 may not expose msprof.  Keep a useful local smoke/perf fallback;
    # 247 should use msprof so the summary is based on Task Duration(us).
    local start_ns end_ns
    start_ns=$(date +%s%N)
    "$PYTHON_BIN/python" "$RUNNER" "${run_args[@]}" \
      >"$output_dir/run.log" 2>&1
    end_ns=$(date +%s%N)
    printf '%s\n' "$(( (end_ns - start_ns) / 1000 ))" >"$output_dir/wall_time_us"
    cat "$output_dir/run.log" >>"$LOG_FILE"
  fi
}

for batch in "${BATCHES[@]}"; do
  for q_seq in "${Q_SEQS[@]}"; do
    for kv_seq in "${KV_SEQS[@]}"; do
      [[ "$q_seq" == "$kv_seq" || "${ALLOW_RECTANGULAR:-0}" == 1 ]] || continue
      for q_heads in "${Q_HEADS[@]}"; do
        for kv_heads in "${KV_HEADS[@]}"; do
          (( q_heads % kv_heads == 0 )) || continue
          for dim in "${HEAD_DIMS[@]}"; do
            for block_q in "${BLOCK_QS[@]}"; do
              for block_kv in "${BLOCK_KVS[@]}"; do
                for keep_ratio in "${KEEP_RATIOS[@]}"; do
                  for q_layout in "${Q_LAYOUTS[@]}"; do
                    for kv_layout in "${KV_LAYOUTS[@]}"; do
                      for varied_len in "${VARIED_LENS[@]}"; do
                        for varied_sparsity in "${VARIED_SPARSITY[@]}"; do
                          for actual_available in "${ACTUAL_SEQ_AVAIL[@]}"; do
                            case_name="case_B${batch}_qS${q_seq}_kvS${kv_seq}_qN${q_heads}_kvN${kv_heads}_D${dim}_sbX${block_q}_sbY${block_kv}_keep${keep_ratio}_innerP${INNER_PRECISE}_qL${q_layout}_kvL${kv_layout}_isVL${varied_len}_isVS${varied_sparsity}_isActS${actual_available}_dev${DEVICE}"
                            case_dir_hif8="$HIF8_DIR/$case_name"
                            case_dir_base="$BASE_DIR/$case_name"
                            mkdir -p "$case_dir_hif8" "$case_dir_base"
                            args=("$batch" "$q_seq" "$kv_seq" "$q_heads" "$kv_heads" "$dim" "$block_q" "$block_kv" "$keep_ratio" "$INNER_PRECISE" "$q_layout" "$kv_layout" "$varied_len" "$varied_sparsity" "$actual_available" "MODE" "$BASE_DTYPE" "$SEED" "$DEVICE" "$P_SCALE")
                            case_info="Case: B=$batch qS=$q_seq kvS=$kv_seq qN=$q_heads kvN=$kv_heads D=$dim block=${block_q}x${block_kv} keep=$keep_ratio innerP=$INNER_PRECISE qL=$q_layout kvL=$kv_layout isVL=$varied_len isVS=$varied_sparsity isActS=$actual_available dev=$DEVICE seed=$SEED"
                            echo "$case_info" | tee -a "$LOG_FILE"
                            echo "[HIF8] $case_name" | tee -a "$LOG_FILE"
                            run_variant hif8 "$case_dir_hif8"
                            echo "[HIGH_PREC $BASE_DTYPE] $case_name" | tee -a "$LOG_FILE"
                            run_variant highprec "$case_dir_base"
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

HIF8_DIR="$HIF8_DIR" BASE_DIR="$BASE_DIR" BASE_DTYPE="$BASE_DTYPE" \
  "$PYTHON_BIN/python" "$SCRIPT_DIR/bsa_gen_perf_summary.py" | tee -a "$LOG_FILE"
