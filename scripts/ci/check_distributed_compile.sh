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
#   1. 从pr_file（变更文件列表，每行一个路径）中提取算子目录（路径含 op_host/op_kernel/op_api/op_graph/
#      op_tiling/op_proto/framework 之一，取该目录前的最后一段作为算子名），并去重
#   2. 不同soc编译耗时不同，长耗时算子字典按soc区分：生效字典 = 全soc通用字典 + 当前soc专属字典
#   3. 参考nn仓build.sh的parse_op_dependencies，基于仓内已有的算子依赖声明解析依赖图：各算子
#      op_host/CMakeLists.txt中的set(<op>_depends ...)（与cmake/func.cmake的op_add_depend_directory
#      消费方式一致），对变更算子先做反向依赖展开（依赖变更算子的算子），再做正向依赖展开
#      （这些算子编译所需的算子），获取pr_filelist之外全部受影响的算子
#   4. 变更算子及其依赖展开出的任一算子命中字典，则输出true（PR需要分布式编译），否则输出false
#   返回值：结果通过stdout输出true/false，脚本退出码0表示执行成功，1表示参数/文件错误

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
# ascend950
LONG_COMPILE_OPS_950=(
    scatter_pa_kv_cache
    allto_all_matmul
    matmul_all_reduce
    matmul_reduce_scatter_v2
    mla_prolog_v3
    grouped_matmul
)

declare -A DEP_MAP=()   # 算子 -> 其依赖的算子列表（正向）
declare -A RDEP_MAP=()  # 算子 -> 依赖该算子的算子列表（反向）
declare -A VISITED=()   # 依赖展开的访问标记

usage() {
    echo "Usage: $0 <soc_version> <pr_file>" >&2
}

# 提取算子目录并去重，输出每行一个算子名
extract_ops() {
    local pr_file=$1
    local marker_re='/(op_host|op_kernel|op_api|op_graph|op_tiling|op_proto|framework)/'
    local line before op
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        [[ "$line" =~ $marker_re ]] || continue
        before=${line%%"${BASH_REMATCH[0]}"*}
        op=${before##*/}
        [ -z "$op" ] && continue
        echo "$op"
    done < "$pr_file" | sort -u
}

# 解析依赖目录对应的算子名列表：目录本身含op_host则取目录名，否则取其下一级
# 含op_host的算子目录名（mc2/3rd这类公共依赖目录下有多个算子）
resolve_entry_ops() {
    local entry=$1
    local base="${REPO_ROOT}/${entry}"
    [ -d "$base" ] || base="${REPO_ROOT}/experimental/${entry}"
    [ -d "$base" ] || return 0
    if [ -d "$base/op_host" ]; then
        basename "$base"
        return 0
    fi
    local d
    while IFS= read -r d; do
        basename "$(dirname "$d")"
    done < <(find "$base" -maxdepth 2 -type d -name op_host 2>/dev/null)
}

# 解析仓内op_host/CMakeLists.txt中的set(<op>_depends ...)依赖声明，构建正反向依赖图。
# 变量名支持<op>_depends及其_apt/_a2/_a3变体；依赖项为仓内相对目录，解析为目录下
# 的算子名（与op_add_depend_directory按目录编译依赖算子的行为一致）
parse_op_dependencies() {
    local cmake_file op block tok entry dep_op
    local -A ENTRY_NODES=()
    while IFS= read -r cmake_file; do
        op=$(basename "$(dirname "$(dirname "$cmake_file")")")
        while IFS= read -r block; do
            [ -z "$block" ] && continue
            block=${block#set\(}
            block=${block%%)*}          # 截断右括号及其之后内容
            block=${block##*_depends}   # 去掉变量名
            block=${block//\"*\"/}      # 去掉引号说明串
            for tok in $block; do
                case $tok in
                    CACHE|INTERNAL|PARENT_SCOPE|FORCE) continue ;;
                esac
                entry=$tok
                if [ -z "${ENTRY_NODES[$entry]+x}" ]; then
                    ENTRY_NODES[$entry]=$(resolve_entry_ops "$entry")
                fi
                for dep_op in ${ENTRY_NODES[$entry]}; do
                    [ "$dep_op" = "$op" ] && continue
                    DEP_MAP[$op]="${DEP_MAP[$op]:-} $dep_op"
                    RDEP_MAP[$dep_op]="${RDEP_MAP[$dep_op]:-} $op"
                done
            done
        done < <(grep -zoP 'set\([A-Za-z0-9_]+_depends\s+[^)]*\)' "$cmake_file" 2>/dev/null | tr '\0' '\n')
    done < <(find "${REPO_ROOT}" -path "*/op_host/CMakeLists.txt" \
        -not -path "*/.git/*" -not -path "*/build/*" -not -path "*/build_out/*" 2>/dev/null)
}

# 从起始算子沿邻接表广度优先展开，访问到的算子记录在VISITED
# 参数1：起始算子列表（空格分隔）；参数2：展开方向，rdep=反向（依赖该算子的算子），dep=正向（该算子依赖的算子）
collect_closure() {
    local -a queue=($1)
    local -a next=()
    local op rel rels
    for op in "${queue[@]}"; do
        VISITED[$op]=1
    done
    while [ ${#queue[@]} -gt 0 ]; do
        next=()
        for op in "${queue[@]}"; do
            if [ "$2" = "rdep" ]; then
                rels=${RDEP_MAP[$op]:-}
            else
                rels=${DEP_MAP[$op]:-}
            fi
            for rel in $rels; do
                [ -n "${VISITED[$rel]:-}" ] && continue
                VISITED[$rel]=1
                next+=("$rel")
            done
        done
        queue=("${next[@]}")
    done
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

    local changed_ops
    changed_ops=$(extract_ops "$pr_file")
    if [ -z "$changed_ops" ]; then
        echo "false"
        exit 0
    fi

    # 与nn仓parse_compile_changed_files.py一致：先取反向依赖（受变更影响的算子），
    # 再对反向依赖取正向依赖（这些算子编译所需的算子），得到完整的受影响算子集合
    parse_op_dependencies
    collect_closure "$changed_ops" rdep
    collect_closure "${!VISITED[*]}" dep

    # 生效字典 = 全soc通用字典 + 当前soc专属字典
    local -a dict=(${LONG_COMPILE_OPS_ALL[@]+"${LONG_COMPILE_OPS_ALL[@]}"})
    case $(echo "$soc" | tr '[:upper:]' '[:lower:]') in
        ascend910b)
            dict+=(${LONG_COMPILE_OPS_910B[@]+"${LONG_COMPILE_OPS_910B[@]}"})
            ;;
        ascend950)
            dict+=(${LONG_COMPILE_OPS_950[@]+"${LONG_COMPILE_OPS_950[@]}"})
            ;;
    esac

    local op
    for op in "${!VISITED[@]}"; do
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
