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
# Copy this file to scripts/dev_tasks/<task-name>/task.sh and customize it.
# It is sourced by server_b_task.sh and therefore must contain trusted Bash code.

TASK_NAME="<task-name>"
OPERATOR="<operator-name>"
SOC="ascend950"
CANN_ENV="/usr/local/Ascend/cann-9.2.0/set_env.sh"
TASK_HAS_RUNNER=0

task_build()
{
    bash build.sh \
        --pkg \
        --soc="$SOC" \
        --ops="$OPERATOR" \
        --vendor_name=xwqtju_dev \
        -j16
}

task_select_package()
{
    find "$PWD/build_out" -maxdepth 1 -type f -name '*.run' -printf '%T@ %p\n' \
        | sort -rn \
        | head -1 \
        | cut -d' ' -f2-
}

task_install()
{
    local package_path="$1"
    local install_root="$2"
    chmod u+x "$package_path"
    "$package_path" --install-path="$install_root"
}

# When a runner exists, set TASK_HAS_RUNNER=1 and implement this function.
# task_run()
# {
#     local install_root="$1"
#     local custom_env
#     custom_env="$(find "$install_root" -name set_env.bash -print -quit)"
#     [[ -n "$custom_env" ]] || return 1
#     source "$custom_env"
#     bash build.sh --run_example "$OPERATOR" eager cust --vendor_name=xwqtju_dev
# }
