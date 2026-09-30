#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# 判断PR是否需要分布式编译：
#   用法：bash check_distributed_compile.sh <soc_version> <pr_file>
#   1. 调用cmake/scripts/parse_compile_scope.py：按白名单(tests/compile_scope_whitelist.yaml)优先、
#      否则CMake depends反向闭包 + def.cpp AddConfig SoC过滤，从pr_file（变更文件列表，每行一个路径）
#      解析本PR在指定soc上的实际编译算子范围（与PR_COMPILE任务的编译口径一致）
#   2. 不同soc编译耗时不同，长耗时算子字典按soc区分：生效字典 = 全soc通用字典 + 当前soc专属字典
#   3. 编译范围内的算子命中字典，则输出true（PR需要分布式编译），否则输出false
#   返回值：结果通过stdout输出true/false，脚本退出码0表示执行成功，1表示参数/文件/执行错误

set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)

# 全soc通用长耗时算子字典：命中的算子无论什么soc都返回true，其他soc的字典均为其扩展
LONG_COMPILE_OPS_ALL=(
    fused_infer_attention_score
    flash_attention_score
    flash_attention_score_grad
)

# ascend910b长耗时算子字典
LONG_COMPILE_OPS_910B=(
    mla_preprocess_v2
    mla_preprocess
)
# ascend910_93长耗时算子字典（初始沿用910b，后续按实测耗时调整）
LONG_COMPILE_OPS_910_93=(
    mla_preprocess_v2
    mla_preprocess
)
# ascend950
LONG_COMPILE_OPS_950=(
    scatter_pa_kv_cache
    allto_all_matmul
    matmul_all_reduce
    matmul_reduce_scatter_v2
    mla_prolog_v3
    grouped_matmul
    mega_moe
)

usage() {
    echo "Usage: $0 <soc_version> <pr_file>" >&2
}

main() {
    if [ "$#" -ne 2 ]; then
        usage
        exit 1
    fi
    local soc=$1
    local pr_file=$2
    if [ ! -f "$pr_file" ]; then
        echo "pr_file not found: $pr_file" >&2
        exit 1
    fi

    # 受影响算子以parse_compile_scope.py返回值为准（与PR_COMPILE实际编译范围一致）
    local affected_ops
    affected_ops=$(python3 "${REPO_ROOT}/cmake/scripts/parse_compile_scope.py" \
        -f "$pr_file" --soc "$soc") || exit 1
    if [ -z "$affected_ops" ]; then
        echo "false"
        exit 0
    fi

    # 生效字典 = 全soc通用字典 + 当前soc专属字典
    local -a dict=(${LONG_COMPILE_OPS_ALL[@]+"${LONG_COMPILE_OPS_ALL[@]}"})
    case $(echo "$soc" | tr '[:upper:]' '[:lower:]') in
        ascend910b)
            dict+=(${LONG_COMPILE_OPS_910B[@]+"${LONG_COMPILE_OPS_910B[@]}"})
            ;;
        ascend910_93)
            dict+=(${LONG_COMPILE_OPS_910_93[@]+"${LONG_COMPILE_OPS_910_93[@]}"})
            ;;
        ascend950)
            dict+=(${LONG_COMPILE_OPS_950[@]+"${LONG_COMPILE_OPS_950[@]}"})
            ;;
    esac

    local op
    for op in ${affected_ops//;/ }; do
        local d
        for d in "${dict[@]}"; do
            if [ "$op" = "$d" ]; then
                echo "true"
                exit 0
            fi
        done
    done

    echo "false"
    exit 0
}

main "$@"
