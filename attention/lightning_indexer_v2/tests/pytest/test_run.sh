#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATH1="${SCRIPT_DIR}/excel/*"
PATH2="${SCRIPT_DIR}/pt_path"

LIV2_PT_SAVE_SCRIPT="${SCRIPT_DIR}/batch/lightning_indexer_v2_pt_save.py"
TEST_LIV2_BATCH_SCRIPT="${SCRIPT_DIR}/test_lightning_indexer_v2_batch.py"
REPLACE_PATH_SCRIPT="${SCRIPT_DIR}/batch/replace_path.py"
TEST_LIV2_SINGLE_SCRIPT="${SCRIPT_DIR}/test_lightning_indexer_v2_single.py"


show_help() {
    echo "Usage: $0 <command> [options]"
    echo "Commands:"
    echo "  single [--paramset default|stc|MODULE] [-C CASE[,CASE...]] [-M eager|graph]"
    echo "  batch [eager|graph]"
    echo "  help"
    echo "Examples:"
    echo "  $0 single --paramset stc"
    echo "  $0 single --paramset stc -C META_TRAILING_K_ZERO_41"
    echo "  $0 single --paramset stc -C S2_16_003 -M graph"
    echo "  $0 batch graph"
}


run_single() {
    local paramset_module="$1"
    local case_names="$2"
    local run_mode="$3"
    LIV2_PARAMSET_MODULE="${paramset_module}" \
    LIV2_CASE_NAMES="${case_names}" \
    LIV2_RUN_MODE="${run_mode}" \
        python3 -m pytest -rA -s "${TEST_LIV2_SINGLE_SCRIPT}" -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}


run_batch() {
    local run_mode="${1:-eager}"
    echo "===== Generate LI V2 batch inputs ====="
    # PATH1 intentionally remains unquoted so the shell expands the Excel glob.
    python3 "${LIV2_PT_SAVE_SCRIPT}" ${PATH1} "${PATH2}" || exit 1

    echo "===== Update batch input paths ====="
    python3 "${REPLACE_PATH_SCRIPT}" "${TEST_LIV2_BATCH_SCRIPT}" "${PATH2}" || exit 1

    echo "===== Run LI V2 batch tests (${run_mode}) ====="
    LIV2_RUN_MODE="${run_mode}" \
        python3 -m pytest -rA -s "${TEST_LIV2_BATCH_SCRIPT}" -v -m ci
    local test_status=$?

    if [ -f "${TEST_LIV2_BATCH_SCRIPT}.bak" ]; then
        cp "${TEST_LIV2_BATCH_SCRIPT}.bak" "${TEST_LIV2_BATCH_SCRIPT}"
    fi
    return "${test_status}"
}


if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

command_name="$1"
shift

case "${command_name}" in
    single)
        paramset_module="default"
        case_names=""
        run_mode="eager"
        while [ $# -gt 0 ]; do
            case "$1" in
                --paramset)
                    paramset_module="$2"
                    shift 2
                    ;;
                -C|--cases)
                    case_names="$2"
                    shift 2
                    ;;
                -M|--run-mode)
                    run_mode="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown single option: $1"
                    show_help
                    exit 1
                    ;;
            esac
        done
        if [[ "${run_mode}" != "eager" && "${run_mode}" != "graph" ]]; then
            echo "single run mode must be eager or graph: ${run_mode}"
            exit 1
        fi
        run_single "${paramset_module}" "${case_names}" "${run_mode}"
        ;;
    batch)
        run_mode="${1:-eager}"
        if [[ "${run_mode}" != "eager" && "${run_mode}" != "graph" ]]; then
            echo "batch run mode must be eager or graph: ${run_mode}"
            exit 1
        fi
        run_batch "${run_mode}"
        ;;
    help|-h|--help)
        show_help
        ;;
    *)
        echo "Unknown command: ${command_name}"
        show_help
        exit 1
        ;;
esac
