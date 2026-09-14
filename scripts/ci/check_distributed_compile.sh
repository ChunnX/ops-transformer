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
#   2. 与下方按soc配置的长耗时算子字典比较，命中任一算子则输出true（PR需要分布式编译），否则输出false
#   返回值：结果通过stdout输出true/false，脚本退出码0表示执行成功，1表示参数/文件错误

set -u

# 全soc通用长耗时算子字典：命中的算子无论什么soc都返回true，其他soc的字典均为其扩展
LONG_COMPILE_OPS_ALL=(
    fused_infer_attention_score
    flash_attention_score
    flash_attention_score_grad
    matmul_all_reduce
    incre_flash_attention
    norm_rope_concat
)

# ascend910b长耗时算子字典
LONG_COMPILE_OPS_910B=(
    mla_preprocess_v2
    mla_preprocess
    sparse_lightning_indexer_kl_loss_grad
)
# ascend950
LONG_COMPILE_OPS_950=(
    scatter_pa_kv_cache
    allto_all_matmul
    matmul_reduce_scatter_v2
    mla_prolog_v3
    grouped_matmul
    chunk_kda_fwd
    mixed_quant_sparse_flash_mla
    all_gather_matmul_v2
    moe_init_routing_v4
    mod_init_routing_v3
    flash_attn
    sparse_flash_attention_grad
    sparse_flash_mla
    matmul_all_to_all
    moe_distribute_dispatch_v2
    moe_token_permute_with_routing_map_grad
    dense_lightning_indexer_kl_loss_grad
)

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
    while IFS= read -r op; do
        [ -z "$op" ] && continue
        local d
        for d in "${dict[@]}"; do
            if [ "$op" = "$d" ]; then
                echo "true"
                exit 0
            fi
        done
    done < <(extract_ops "$pr_file")

    echo "false"
    exit 0
}

main "$@"
