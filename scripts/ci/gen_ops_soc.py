# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import re
import logging

black_list = ['moe_gather_v2',
              'moe_inplace_index_add',
              'moe_inplace_index_add_with_sorted',
              'moe_masked_scatter']
op_level_list = ['moe_token_permute_with_routing_map',
                 'moe_token_permute_with_routing_map_grad',
                 'moe_token_unpermute_with_routing_map',
                 'moe_distribute_combine_v2',      # 添加 v2
                 'moe_distribute_combine_v3',      # 添加 v3
                 'moe_distribute_dispatch_v2',     # 添加 v2
                 'moe_distribute_dispatch_v3']     # 添加 v3

# 定义算子依赖关系：编译某些算子时会自动包含其他算子
op_dependencies = {
    'moe_distribute_combine_v2': ['moe_distribute_combine_v3'],
    'moe_distribute_dispatch_v2': ['moe_distribute_dispatch_v3'],
}

# obj分组字典,需要分组的算子在字典中对应soc xxxx: obj分组数
group_op_dict = {
    "ascend910b": {"fused_infer_attention_score": 10, 
                   "incre_flash_attention": 5,
                   "prompt_flash_attention": 5},
    "ascend910_93": {"fused_infer_attention_score": 10,
                     "incre_flash_attention": 5,
                     "prompt_flash_attention": 5},
    "ascend950": {"fused_infer_attention_score": 20}
}


def resolve_op_dependencies(op_name_set):
    """
    解析算子依赖关系，返回去重后的算子集合
    规则：如果包含 v2，则自动包含对应的 v3，但不需要单独编译 v3
    """
    resolved_ops = set(op_name_set)
    
    # 检查并添加依赖
    for op_name, dependencies in op_dependencies.items():
        if op_name in resolved_ops:
            # 如果包含了 v2，添加对应的 v3
            resolved_ops.update(dependencies)
    
    # 去重逻辑：如果同时存在 v2 和 v3，保留 v2（因为编译 v2 会自动编译 v3）
    # 但需要在 op_level_list 处理时特殊处理
    
    return resolved_ops


def get_sh_files(gen_dir):
    """获取目录中所有 .sh 文件名（不包含路径）"""
    sh_files = []
    for item in os.listdir(gen_dir):
        item_path = os.path.join(gen_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.sh'):
            sh_files.append(item)
    return sh_files


def parse_opname_from_filename(filename):
    """
    从文件名解析 op_name。
    要求格式：xxx-<opname>-<digits>.sh
    成功返回 op_name，失败返回 None
    """
    parts = filename.split('-')
    if len(parts) < 3:
        return None

    return parts[1]


def count_opnames(sh_filenames):
    """统计每个 op_name 出现的次数"""
    opname_to_count = {}
    for filename in sh_filenames:
        op_name = parse_opname_from_filename(filename)
        if op_name is not None:
            opname_to_count[op_name] = opname_to_count.get(op_name, 0) + 1
    opname_to_count_sorted = dict(sorted(opname_to_count.items()))
    return opname_to_count_sorted


def grouped(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    all_rows = []
    added_op_levels = set()
    special_task = ""
    
    # 先收集所有需要特殊处理的算子
    special_ops_set = set()
    for op_name, count in op_counts.items():
        op_name_real = op_name
        if soc == 'ascend950' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name in ('allto_all_matmul_apt', 'matmul_allto_all_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name_real in black_list:
            continue
        if op_name_real in op_level_list:
            special_ops_set.add(op_name_real)
    
    # 处理依赖关系：如果有 v2，标记 v3 已经被包含
    v2_v3_pairs = [
        ('moe_distribute_combine_v2', 'moe_distribute_combine_v3'),
        ('moe_distribute_dispatch_v2', 'moe_distribute_dispatch_v3')
    ]
    
    for v2_op, v3_op in v2_v3_pairs:
        if v2_op in special_ops_set and v3_op in special_ops_set:
            # 同时存在 v2 和 v3，只保留 v2（v3 会随 v2 自动编译）
            logger.info(f"检测到同时存在 {v2_op} 和 {v3_op}，{v3_op} 将随 {v2_op} 自动编译，跳过单独编译")
            special_ops_set.discard(v3_op)
    
    # 构建 special_task
    special_task = ','.join(sorted(special_ops_set))
    
    for op_name, count in op_counts.items():
        op_name_real = op_name
        if soc == 'ascend950' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name in ('allto_all_matmul_apt', 'matmul_allto_all_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name_real in black_list:
            continue
        if op_name_real in op_level_list:
            # 跳过，已经在 special_task 中处理
            continue
        for i in range(count):
            row_string = f"{op_name_real},{count}-{i}"
            all_rows.append(row_string)
    
    if special_task:
        all_rows.append(special_task)

    for idx, row in enumerate(all_rows):
        result[idx % group_size].append(row)

    return result


def grouped_back(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    added_op_levels = set()
    special_task_parts = []
    current_group_index = 0
    
    # 收集特殊算子
    special_ops_set = set()
    for op_name, count in op_counts.items():
        op_name_real = op_name
        if soc == 'ascend950' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        elif op_name in ('allto_all_matmul_apt', 'matmul_allto_all_apt'):
            op_name_real = op_name.replace('_apt', '')

        if op_name_real in black_list:
            continue
        if op_name_real in op_level_list:
            special_ops_set.add(op_name_real)
    
    # 处理 v2/v3 依赖关系
    v2_v3_pairs = [
        ('moe_distribute_combine_v2', 'moe_distribute_combine_v3'),
        ('moe_distribute_dispatch_v2', 'moe_distribute_dispatch_v3')
    ]
    
    for v2_op, v3_op in v2_v3_pairs:
        if v2_op in special_ops_set and v3_op in special_ops_set:
            logger.info(f"检测到同时存在 {v2_op} 和 {v3_op}，{v3_op} 将随 {v2_op} 自动编译，跳过单独编译")
            special_ops_set.discard(v3_op)
    
    special_task_parts = sorted(special_ops_set)

    for op_name, count in op_counts.items():
        op_name_real = op_name
        if soc == 'ascend950' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        elif op_name in ('allto_all_matmul_apt', 'matmul_allto_all_apt'):
            op_name_real = op_name.replace('_apt', '')

        if op_name_real in black_list:
            continue
        if op_name_real in op_level_list:
            continue
        if count >= group_size:
            for i in range(group_size):
                row_string = f"{op_name_real},{group_size}-{i}"
                result[current_group_index].append(row_string)
                current_group_index = (current_group_index + 1) % group_size
        else:
            for i in range(count):
                row_string = f"{op_name_real},{count}-{i}"
                result[current_group_index].append(row_string)
                current_group_index = (current_group_index + 1) % group_size

    if special_task_parts:
        special_task = ','.join(special_task_parts)
        result[current_group_index].append(special_task)

    return result


def grouped_def(repository_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    init_oplist = []
    op_list = []

    for root, dirs, files in os.walk(repository_path):
        # 过滤掉不需要的目录
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]

        for file in files:
            if file.endswith('_def.cpp'):
                file_path = os.path.join(root, file)
                op_name = file.replace('_def.cpp', '')

                # 提取 AICore 配置
                ai_core_configs = extract_ai_core_configs(file_path)

                if soc in ai_core_configs:
                    init_oplist.append(op_name)
    
    init_oplist.sort()
    
    # 处理特殊算子和依赖关系
    special_ops_set = set()
    regular_ops = []
    
    for op_name in init_oplist:
        if op_name in black_list:
            continue
        if op_name in op_level_list:
            special_ops_set.add(op_name)
            continue
        if op_name in group_op_dict[soc]:
            op_group_size = group_op_dict[soc][op_name]
            if group_size > 1 and op_group_size > group_size:
                op_group_size = group_size
            for i in range(op_group_size):
                row_string = f"{op_name},{op_group_size}-{i}"
                regular_ops.append(row_string)
        else:
            regular_ops.append(op_name)
    
    # 处理 v2/v3 依赖关系
    v2_v3_pairs = [
        ('moe_distribute_combine_v2', 'moe_distribute_combine_v3'),
        ('moe_distribute_dispatch_v2', 'moe_distribute_dispatch_v3')
    ]
    
    for v2_op, v3_op in v2_v3_pairs:
        if v2_op in special_ops_set and v3_op in special_ops_set:
            logger.info(f"检测到同时存在 {v2_op} 和 {v3_op}，{v3_op} 将随 {v2_op} 自动编译，跳过单独编译")
            special_ops_set.discard(v3_op)
    
    # 构建最终的 op_list
    op_list.extend(sorted(regular_ops))
    
    if special_ops_set:
        special_task = ','.join(sorted(special_ops_set))
        op_list.append(special_task)

    for idx, row in enumerate(op_list):
        result[idx % group_size].append(row)

    return result


# 其余函数保持不变（should_skip_directory, parse_foreach_config 等）
# ...（这里省略了其他函数的实现，保持原样）


def should_skip_directory(dir_name):
    """
    判断是否应该跳过该目录
    """
    skip_dirs = {
        'build', 'cmake', 'common', 'docs', 'examples',
        'experimental', 'scripts', 'tests', 'third_party',
        '3rdparty', 'torch_extension', '3rd'
    }
    return dir_name in skip_dirs


def parse_foreach_config(config_str):
    """
    解析 FOREACH_OPDEF 中的配置字符串
    """
    config_mapping = {
        'A2': 'ascend910b',
        '910_93': 'ascend910_93',
        'A5': 'ascend950',
        '910B': 'ascend910b',
        '910B_93': 'ascend910_93',
        '910B_95': 'ascend950',
        '950': 'ascend950',
        '910': 'ascend910',
        '910_55': 'ascend910_55',
    }

    found_configs = []
    config_str_upper = config_str.upper()

    priority_checks = [
        ('A2', 'ascend910b'),
        ('910_93', 'ascend910_93'),
        ('A5', 'ascend950'),
        ('910_55', 'ascend910_55'),
        ('910B', 'ascend910b'),
        ('910B_93', 'ascend910_93'),
        ('910B_95', 'ascend950'),
        ('950', 'ascend950'),
        ('910', 'ascend910'),
    ]

    for key, value in priority_checks:
        if key in config_str_upper and value not in found_configs:
            found_configs.append(value)

    return found_configs


# ... (其他 extract_* 函数保持原样)


def main(repository_path, soc, group_size=1):
    project_path = os.path.abspath(repository_path)
    gen_path = os.path.abspath(os.path.join(project_path, "build", "binary", soc, "gen"))
    if group_size > 1:
        op_data = grouped_back(gen_path, soc, group_size)
    else:
        op_data = grouped_def(project_path, soc, group_size)
    return op_data


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    temp = main(file_path, "ascend910b", 1)
    size = sum(len(sublist) for sublist in temp)
    logger.info(f"Group dicts: {temp}")
    logger.info(f"Total task size: {size}")
