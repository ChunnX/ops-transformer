#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

TASK_NAME="gmm-v5-noquant-ascend950-msprof-op"
OPERATOR="grouped_matmul"
SOC="ascend950"
CANN_ENV="/usr/local/Ascend/cann-9.2.0/set_env.sh"
TASK_HAS_RUNNER=1

readonly GMM_VENDOR="xwqtju_dev"
readonly GMM_EXAMPLE="test_aclnn_grouped_matmul_v5_noquant.cpp"
readonly GMM_BUILD_MARKER="$PWD/.server_b_gmm_v5_noquant_msprof_build_started"
readonly GMM_PROFILE_EXECUTABLE="$PWD/build/test_aclnn_grouped_matmul"

task_build()
{
    rm -f -- "$GMM_BUILD_MARKER"
    touch "$GMM_BUILD_MARKER"
    bash build.sh \
        --pkg \
        --soc="$SOC" \
        --ops="$OPERATOR" \
        --vendor_name="$GMM_VENDOR" \
        -j1
}

task_select_package()
{
    find "$PWD/build_out" \
        -maxdepth 1 \
        -type f \
        -name '*.run' \
        -newer "$GMM_BUILD_MARKER" \
        -printf '%T@ %p\n' \
        | sort -rn \
        | sed -n '1{s/[^ ]* //;p;}'
}

task_install()
{
    local package_path="$1"
    local install_root="$2"
    chmod u+x "$package_path"
    "$package_path" --install-path="$install_root"
}

task_run()
{
    local install_root="$1"
    local custom_env
    custom_env="$(find "$install_root" -type f -name set_env.bash -print -quit)"
    if [[ -z "$custom_env" ]]; then
        echo "custom operator set_env.bash not found under: $install_root" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    set +u
    source "$custom_env"
    set -u

    bash build.sh \
        --run_example "$OPERATOR" eager cust \
        --vendor_name="$GMM_VENDOR" \
        --soc="$SOC" \
        --example_name="$GMM_EXAMPLE"

    [[ -x "$GMM_PROFILE_EXECUTABLE" ]] || {
        echo "GMM example executable not found: $GMM_PROFILE_EXECUTABLE" >&2
        return 1
    }
    command -v msprof >/dev/null 2>&1 || {
        echo "msprof command not found after loading CANN environment" >&2
        return 1
    }

    local profile_dir="${RUN_DIR:?RUN_DIR is required by the profiling task}/profiling"
    mkdir -p "$profile_dir"
    (
        cd "$profile_dir"
        msprof op "$GMM_PROFILE_EXECUTABLE"
    )

    local opprof_dir
    opprof_dir="$(find "$profile_dir" -mindepth 1 -maxdepth 1 -type d -name 'OPPROF_*' -print -quit)"
    if [[ -z "$opprof_dir" ]]; then
        echo "msprof completed but no OPPROF_* directory was generated under: $profile_dir" >&2
        return 1
    fi
    echo "[PROFILE] msprof op result: $opprof_dir"
}
