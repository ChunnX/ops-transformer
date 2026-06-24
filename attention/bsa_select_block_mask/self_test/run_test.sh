#!/bin/bash
# MaskKComp self-test: batch end-to-end execution script
# Maintains a parameter case list in bash array, iterates all cases:
#   For each case: gen_data -> gen_golden -> compile -> NPU exec -> compare
#
# Parameter keys: B N S1 S2 D bx by layout sparsity use_seqlen use_q_block_len use_k_block_len
# Usage:
#   ./run_test.sh                  # run all batch cases
#   ./run_test.sh --case 0         # run only case #0
#   ./run_test.sh --case 2 5       # run cases #2 and #5
#   ./run_test.sh --single --B 2 --N 16 ...  # run a single ad-hoc case

# NOTE: No 'set -e' here — test runner must tolerate individual case failures
# and continue running all remaining cases. Only exit 1 at the end if any case failed.

# ============================================================
# ===== Batch Test Case List =====
# Format: "B|N|S1|S2|D|bx|by|layout|sparsity|use_seqlen|use_q_block_len|use_k_block_len"
# Add/modify cases here for batch testing
# ============================================================

TEST_CASES=(
    # Case 0: Basic BNSD, small dims
    # "1|1|128|128|128|128|128|BNSD|0.5|0|0|0"

    # "1|1|1280|256|128|128|128|BNSD|0.5|0|0|0"

    # "1|1|1280|256|128|128|128|BNSD|0.5|0|1|1"

    # "4|10|1280|256|128|128|128|BNSD|0.5|0|1|1"

    # "4|10|258|121|128|128|128|BNSD|0.5|0|1|1"
    # "4|10|128|262144|128|128|128|BNSD|0.5|0|0|0"

    "2|2|39936|17108|128|128|128|BNSD|0.5|0|1|1"

    # "4|10|12054|14220|128|128|128|BNSD|0.5|0|1|1"
    
    # Case 1: Multi-batch
    # "2|8|128|128|128|128|128|BNSD|0.5|0|0|0"
    # # Case 2: Larger seqlen
    # "1|8|1280|256|128|128|128|BNSD|0.5|0|0|0"
    # # Case 3: Multi-batch + larger seqlen
    # "2|8|1280|256|128|128|128|BNSD|0.5|0|0|0"
    # # Case 4: Small block size (64)
    # "1|8|1280|256|128|64|64|BNSD|0.5|0|0|0"
    # # Case 5: Low sparsity (0.3, TopK mode)
    # "1|8|1280|256|128|128|128|BNSD|0.3|0|0|0"
    # # Case 6: High sparsity (0.8, BottomK mode)
    # "1|8|1280|256|128|128|128|BNSD|0.8|0|0|0"
    # # Case 7: With seqlen
    # "1|8|1280|256|128|128|128|BNSD|0.5|1|0|0"
    # # Case 8: With q_block_len
    # "1|8|1280|256|128|64|64|BNSD|0.5|0|1|0"
    # # Case 9: With k_block_len
    # "1|8|1280|256|128|64|64|BNSD|0.5|0|0|1"
    # # Case 10: All optional features enabled
    # "1|8|1280|256|128|64|64|BNSD|0.5|1|1|1"
    # # Case 11: Multi-batch + all features
    # "2|4|1280|256|128|64|64|BNSD|0.3|1|1|1"
)

# ============================================================
# Global variables (set by parse_cli_args / setup_env)
# ============================================================

RUN_MODE="batch"          # batch | single
SELECTED_CASES=()         # specific case indices for batch mode
DEVICE_ID=0
SINGLE_PARAMS=""          # params string for --single mode
SCRIPT_DIR=""
RESULTS_DIR=""
PASS_COUNT=0
FAIL_COUNT=0
FAIL_CASES=()

# ============================================================
# Internal settings (NOT exposed via CLI — modify directly here)
# ============================================================

# NPU execution repeat count for stability verification.
# Each NPU run is compared against golden independently.
# Case passes iff ALL runs pass. Set to 1 for normal single-run mode.
NPU_RUN_COUNT=1

# Generalized test case generation (replaces TEST_CASES when enabled).
# Set USE_GENERALIZED_CASES=0 to randomly generate cases; 0 uses hardcoded TEST_CASES above.
# GENERALIZED_CASE_COUNT: number of random cases to generate.

USE_GENERALIZED_CASES=1
GENERALIZED_CASE_COUNT=100

# ============================================================
# Function: setup_env - Environment initialization
# ============================================================

setup_env() {
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0

    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    cd "$SCRIPT_DIR"

    # RESULTS_DIR="batch_results_$(date +%Y%m%d_%H%M%S)"

    RESULTS_DIR="batch_results"
    mkdir -p "$RESULTS_DIR"
}

# ============================================================
# Function: generate_generalized_cases - Generate random TEST_CASES
# ============================================================
# Constraints:
#   0 < B, N < 40           (1..39)
#   0 < S1, S2 < 1048576    (1..1048575, i.e. < 1*1024*1024)
#   D = 128 (fixed)
#   bx, by: multiples of 64, bx < S1, by < S2
#   layout = BNSD (fixed)
#   sparsity: 0.3 .. 0.98 (host tiling valid range)
#   use_seqlen = 0 (fixed)
#   use_q_block_len, use_k_block_len: randomly 0 or 1

generate_generalized_cases() {
    local count="$GENERALIZED_CASE_COUNT"
    local rand_seed="$RANDOM"

    echo "--- Generating $count generalized test cases (seed=$rand_seed) ---"

    TEST_CASES=()

    local i=0
    for i in $(seq 0 $(($count - 1))); do
        # B: 1..39
        local B=$(( (RANDOM % 5) + 1 ))

        # N: 1..39
        local N=$(( (RANDOM % 3) + 1 ))

        # S1: 1..1048575, bias towards practical sizes (128..2048) with occasional large values
        # 80% chance: small range 128..2048, 20% chance: full range up to 1048575
        local S1
        round=$((RANDOM % 100))
        if [[ $round -lt 5 ]]; then
            S1=$(( ((RANDOM % 800) + 1) * 128 ))
        elif [[ $round -lt 7 ]]; then
            S1=$(( ((RANDOM % 800) + 1) * 128 + (RANDOM % 128) ))
        else
            S1=$(( (RANDOM % 6048575) + 1048576 + ((RANDOM % 20)) * 128))
            # S1=$(( (RANDOM % 20 + 1) * 128 + (RANDOM % 128) ))
        fi

        # S2: same strategy as S1
        local S2
        round=$((RANDOM % 100))
        if [[ $round -lt 5 ]]; then
            S2=$(( ((RANDOM % 800) + 1) * 128 ))
        elif [[ $round -lt 7 ]]; then
            S2=$(( ((RANDOM % 800) + 1) * 128 + (RANDOM % 128) ))
        else
            S2=$(( (RANDOM % 6048575) + 1048576 + ((RANDOM % 20)) * 128))
        fi

        # D: fixed 128
        local D=128
        # bx: multiple of 64, bx < S1
        # Max valid bx = S1 - 1, round down to nearest 64
        # local bx_max=$(( (S1 - 1) / 64 * 64 ))
        # if [ $bx_max -lt 64 ]; then bx_max=64; fi
        # local bx_choices=$(( bx_max / 64 ))
        # local bx=$(( (RANDOM % bx_choices + 1) * 64 ))

        local bx=128

        # by: multiple of 64, by < S2
        # local by_max=$(( (S2 - 1) / 64 * 64 ))
        # if [ $by_max -lt 64 ]; then by_max=64; fi
        # local by_choices=$(( by_max / 64 ))
        # local by=$(( (RANDOM % by_choices + 1) * 64 ))

        local by=128

        # layout: fixed BNSD
        local layout="BNSD"

        # sparsity: 0.30 .. 0.98 (host valid range)
        # Generate as integer 30..98, then format as 0.xx
        local sp_int=$(( (RANDOM % 69) + 30 ))
        local sparsity="0.${sp_int}"

        # use_seqlen: fixed 0
        local use_seqlen=0

        # use_q_block_len: random 0 or 1
        local use_q_block_len=$(( RANDOM % 2 ))

        # use_k_block_len: random 0 or 1
        local use_k_block_len=$(( RANDOM % 2 ))

        local case_str="${B}|${N}|${S1}|${S2}|${D}|${bx}|${by}|${layout}|${sparsity}|${use_seqlen}|${use_q_block_len}|${use_k_block_len}"
        TEST_CASES+=("$case_str")

        echo "  Gen case #${i}: B=$B N=$N S1=$S1 S2=$S2 D=$D bx=$bx by=$by layout=$layout sparsity=$sparsity use_seqlen=$use_seqlen use_q_block_len=$use_q_block_len use_k_block_len=$use_k_block_len"
    done

    echo "--- Generated ${#TEST_CASES[@]} generalized cases ---"
}

# ============================================================
# Function: parse_cli_args - Parse command line arguments
# ============================================================

parse_cli_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --case)
                shift
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    SELECTED_CASES+=("$1")
                    shift
                done
                ;;
            --device)
                DEVICE_ID="$2"; shift 2;;
            --single)
                RUN_MODE="single"; shift;;
            --B|--N|--S1|--S2|--D|--bx|--by|--layout|--sparsity|--use_seqlen|--use_q_block_len|--use_k_block_len)
                SINGLE_PARAMS="$SINGLE_PARAMS $1 $2"; shift 2;;
            *) echo "Unknown option: $1"; exit 1;;
        esac
    done
}

# ============================================================
# Function: parse_case - Parse a case string into global variables
# ============================================================

parse_case() {
    local case_str="$1"
    IFS='|' read -r B N S1 S2 D bx by layout sparsity use_seqlen use_q_block_len use_k_block_len <<< "$case_str"
}

# ============================================================
# Function: compile_binary - Compile C++ binary (shared by all cases)
# ============================================================

compile_binary() {
    echo ""
    echo "--- Compile run_bsa_select_block_mask ---"
    g++ -std=c++17 test_aclnn_bsa_select_block_mask.cpp \
        -I${ASCEND_HOME_PATH}/include \
        -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lnnopbase \
        -L${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib -lcust_opapi \
        -g -o run_bsa_select_block_mask
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "  Compile FAILED: run_bsa_select_block_mask"
        return $rc
    fi
    echo "  Compiled: run_bsa_select_block_mask"
}

# ============================================================
# Function: gen_input_data - Step 1: Generate input data
# ============================================================

gen_input_data() {
    local input_dir="$1"
    echo "--- Step 1: Generate input data ---"
    python3 gen_data.py \
        --B $B --N $N --S1 $S1 --S2 $S2 --D $D --bx $bx --by $by \
        --layout $layout --sparsity $sparsity \
        --use_seqlen $use_seqlen --use_q_block_len $use_q_block_len --use_k_block_len $use_k_block_len \
        --output_dir "$input_dir"
}

# ============================================================
# Function: gen_golden_data - Step 2: Generate golden values
# ============================================================

gen_golden_data() {
    local input_dir="$1"
    local golden_dir="$2"
    echo "--- Step 2: Generate golden ---"
    python3 gen_golden.py --input_dir "$input_dir" --output_dir "$golden_dir"
}

# ============================================================
# Function: exec_npu - Step 3: Execute on NPU
# ============================================================

exec_npu() {
    local input_dir="$1"
    local npu_dir="$2"
    echo "--- Step 3: Execute on NPU ---"
    ./run_bsa_select_block_mask "$input_dir" "$npu_dir" "$DEVICE_ID"
}

# ============================================================
# Function: compare_results - Step 4: Compare NPU vs Golden
# ============================================================

compare_results() {
    local npu_dir="$1"
    local golden_dir="$2"
    local input_dir="$3"
    echo "--- Step 4: Compare NPU vs Golden ---"
    python3 compare.py --npu_dir "$npu_dir" --golden_dir "$golden_dir" --input_dir "$input_dir"
}

# ============================================================
# Function: print_case_header - Print case info header
# ============================================================

print_case_header() {
    local case_idx="$1"
    echo ""
    echo "======================================================"
    echo "  Case #${case_idx}: B=$B N=$N S1=$S1 S2=$S2 D=$D bx=$bx by=$by"
    echo "           layout=$layout sparsity=$sparsity"
    echo "           use_seqlen=$use_seqlen use_q_block_len=$use_q_block_len use_k_block_len=$use_k_block_len"
    echo "======================================================"
}

# ============================================================
# Function: run_single_case - Run one full test case (orchestrates all 4 steps)
# ============================================================

run_single_case() {
    local case_idx="$1"
    local case_str="$2"
    local work_dir="$3"

    parse_case "$case_str"

    print_case_header "$case_idx"

    local case_dir="${work_dir}/case_${case_idx}"
    local input_dir="${case_dir}/input"
    local golden_dir="${case_dir}/golden"
    local npu_base_dir="${case_dir}/npu"

    mkdir -p "$input_dir" "$golden_dir" "$npu_base_dir"

    # ---- Phase 1: gen_data + gen_golden (only once, shared by all NPU runs) ----
    local step_failed=0

    gen_input_data "$input_dir" || { echo "  Case #${case_idx}: Step 1 (gen_data) FAILED"; step_failed=1; }
    if [ $step_failed -eq 0 ]; then
        gen_golden_data "$input_dir" "$golden_dir" || { echo "  Case #${case_idx}: Step 2 (gen_golden) FAILED"; step_failed=1; }
    fi

    if [ $step_failed -ne 0 ]; then
        echo "  Case #${case_idx}: FAIL (data preparation failed)"
        return 1
    fi

    # ---- Phase 2: NPU exec + compare (repeat NPU_RUN_COUNT times for stability) ----
    local all_runs_pass=1
    local run_idx=0
    local fail_run_list=()

    if [ "$NPU_RUN_COUNT" -le 1 ]; then
        # Single run: use default npu dir
        local npu_dir="$npu_base_dir"
        local npu_exec_failed=0
        exec_npu "$input_dir" "$npu_dir" || { echo "  Case #${case_idx}: Step 3 (exec_npu) FAILED"; npu_exec_failed=1; }

        if [ $npu_exec_failed -eq 0 ]; then
            compare_results "$npu_dir" "$golden_dir" "$input_dir" || { all_runs_pass=0; fail_run_list+=("0"); }
        else
            all_runs_pass=0
            fail_run_list+=("exec_failed")
        fi
    else
        # Multiple runs for stability verification
        echo "  --- Stability test: NPU will run $NPU_RUN_COUNT times, each compared against golden ---"
        for run_idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
            if [ "$run_idx" -ge "$NPU_RUN_COUNT" ]; then
                break
            fi

            local npu_dir="${npu_base_dir}/run_${run_idx}"
            mkdir -p "$npu_dir"

            echo "  --- NPU Run #${run_idx}/${NPU_RUN_COUNT} ---"
            local npu_exec_failed=0
            exec_npu "$input_dir" "$npu_dir" || { echo "  Run #${run_idx}: exec_npu FAILED"; npu_exec_failed=1; }

            if [ $npu_exec_failed -eq 0 ]; then
                local this_run_pass=0
                compare_results "$npu_dir" "$golden_dir" "$input_dir" || this_run_pass=1
                if [ $this_run_pass -eq 0 ]; then
                    echo "  Run #${run_idx}: PASS"
                else
                    echo "  Run #${run_idx}: FAIL"
                    all_runs_pass=0
                    fail_run_list+=("$run_idx")
                fi
            else
                all_runs_pass=0
                fail_run_list+=("${run_idx}:exec_failed")
            fi
        done
    fi

    if [ $all_runs_pass -eq 1 ]; then
        echo "  Case #${case_idx}: PASS"
        return 0
    else
        echo "  Case #${case_idx}: FAIL (failed runs: ${fail_run_list[*]})"
        return 1
    fi
}

# ============================================================
# Function: run_single_mode - Execute single ad-hoc mode
# ============================================================

run_single_mode() {
    echo "  Mode: SINGLE (ad-hoc parameters)"

    # Default values
    B=1; N=8; S1=128; S2=128; D=128; bx=128; by=128; layout="BNSD"; sparsity=0.5
    use_seqlen=0; use_q_block_len=0; use_k_block_len=0

    eval $SINGLE_PARAMS

    local case_str="${B}|${N}|${S1}|${S2}|${D}|${bx}|${by}|${layout}|${sparsity}|${use_seqlen}|${use_q_block_len}|${use_k_block_len}"
    run_single_case "adhoc" "$case_str" "$RESULTS_DIR"
    if [ $? -eq 0 ]; then PASS_COUNT=1; else FAIL_COUNT=1; FAIL_CASES+=("adhoc"); fi
}

# ============================================================
# Function: run_batch_selected - Execute selected cases
# ============================================================

run_batch_selected() {
    echo "  Mode: BATCH (selected cases: ${SELECTED_CASES[*]})"

    for idx in "${SELECTED_CASES[@]}"; do
        if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#TEST_CASES[@]}" ]; then
            run_single_case "$idx" "${TEST_CASES[$idx]}" "$RESULTS_DIR"
            if [ $? -eq 0 ]; then
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                FAIL_COUNT=$((FAIL_COUNT + 1))
                FAIL_CASES+=("$idx")
            fi
        else
            echo "  WARNING: case index $idx out of range (0-${#TEST_CASES[@]}-1), skipped"
        fi
    done
}

# ============================================================
# Function: run_batch_all - Execute all cases
# ============================================================

run_batch_all() {
    echo "  Mode: BATCH (all ${#TEST_CASES[@]} cases)"

    for idx in "${!TEST_CASES[@]}"; do
        run_single_case "$idx" "${TEST_CASES[$idx]}" "$RESULTS_DIR"
        if [ $? -eq 0 ]; then
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            FAIL_CASES+=("$idx")
        fi
        # echo ${RESULTS_DIR}/case_${idx}
        rm -rf ${RESULTS_DIR}/case_${idx}
        # rm -rf "/opt/ll/code/ops-transformer_2026/attention/bsa_select_block_mask/self_test/batch_results/"
    done
}

# ============================================================
# Function: print_summary - Print batch test summary to stdout
# ============================================================

print_summary() {
    local total=$((PASS_COUNT + FAIL_COUNT))
    echo ""
    echo "======================================================"
    echo "  Batch Test Summary"
    echo "======================================================"
    echo "  Total cases run:  $total"
    echo "  PASS:             $PASS_COUNT"
    echo "  FAIL:             $FAIL_COUNT"
    if [ ${#FAIL_CASES[@]} -gt 0 ]; then
        echo "  Failed cases:     ${FAIL_CASES[*]}"
    fi
    echo "  Results dir:      $RESULTS_DIR"
    echo "======================================================"
}

# ============================================================
# Function: save_summary - Save summary report to file and cat it
# ============================================================

save_summary() {
    local summary_file="$RESULTS_DIR/batch_summary.txt"
    {
        echo "MaskKComp Batch Test Summary"
        echo "Date: $(date)"
        echo "Total: $((PASS_COUNT + FAIL_COUNT))  PASS: $PASS_COUNT  FAIL: $FAIL_COUNT"
        if [ "$USE_GENERALIZED_CASES" -eq 1 ]; then
            echo "Mode: Generalized (randomly generated $GENERALIZED_CASE_COUNT cases)"
        else
            echo "Mode: Hardcoded TEST_CASES"
        fi
        echo ""
        echo "Case List:"
        for idx in "${!TEST_CASES[@]}"; do
            parse_case "${TEST_CASES[$idx]}"
            local case_status="PASS"
            for fc in "${FAIL_CASES[@]}"; do
                if [ "$fc" = "$idx" ]; then case_status="FAIL"; break; fi
            done
            echo "  #${idx}: [${case_status}] B=$B N=$N S1=$S1 S2=$S2 D=$D bx=$bx by=$by layout=$layout sparsity=$sparsity use_seqlen=$use_seqlen use_q_block_len=$use_q_block_len use_k_block_len=$use_k_block_len"
        done
        echo ""
        if [ ${#FAIL_CASES[@]} -gt 0 ]; then
            echo "Failed cases: ${FAIL_CASES[*]}"
        else
            echo "All cases PASSED!"
        fi
    } > "$summary_file"

    cat "$summary_file"
}

# ============================================================
# Function: print_banner - Print initial banner with env info
# ============================================================

print_banner() {
    echo "=== MaskKComp Batch Self-Test ==="
    echo "  ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
    echo "  Working dir: $SCRIPT_DIR"
    echo "  Device ID: $DEVICE_ID"
    echo "  Total cases defined: ${#TEST_CASES[@]}"
    echo "  NPU run count (stability): $NPU_RUN_COUNT"
    if [ "$USE_GENERALIZED_CASES" -eq 1 ]; then
        echo "  Case mode: GENERALIZED (randomly generated $GENERALIZED_CASE_COUNT cases)"
    else
        echo "  Case mode: HARDCODED TEST_CASES"
    fi
}

# ============================================================
# Function: dispatch_run_mode - Select and execute the correct run mode
# ============================================================

dispatch_run_mode() {
    if [ "$RUN_MODE" = "single" ]; then
        run_single_mode
    elif [ ${#SELECTED_CASES[@]} -gt 0 ]; then
        run_batch_selected
    else
        run_batch_all
    fi
}

# ============================================================
# Function: main - Entry point
# ============================================================

main() {
    parse_cli_args "$@"
    setup_env

    # Generate generalized cases if enabled (replaces TEST_CASES)
    if [ "$USE_GENERALIZED_CASES" -eq 1 ]; then
        generate_generalized_cases
    fi

    print_banner
    compile_binary
    dispatch_run_mode
    print_summary
    save_summary

    if [ $FAIL_COUNT -gt 0 ]; then
        exit 1
    fi
}

main "$@"
