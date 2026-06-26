# -*- coding: utf-8 -*-
"""
MC2 算子测试选择过滤器模块
"""

from .utils import setup_logging, read_file_lines, load_dependency_json, normalize_path
from .op_step1_whitelist import matches_whitelist, glob_to_regex, filter_whitelist
from .op_step2_ignored import is_markdown_file, is_in_tests_directory, filter_ignored
from .op_step3_non_mc2 import is_in_mc2, filter_non_mc2
from .op_step4_common_3rd import is_in_common, is_in_3rd, get_3rd_subdir, filter_common_3rd
# [已禁用] THIRD_PARTY_OP_MAPPING 导出 - common/3rd 变更现在直接触发全量测试
# 如需恢复特殊处理逻辑，取消以下注释:
# from .op_step4_common_3rd import THIRD_PARTY_OP_MAPPING
from .op_step5_extractor import extract_op_name, extract_operators
from .op_step6_dependency import build_reverse_dependency_graph, find_dependent_ops
from .op_step7_output import transform_to_product_grouped, write_output_json, write_soc_json, get_all_ops_with_products, get_ops_info_by_names, load_arch_restrictions, filter_ops_by_arch_and_soc, adjust_supported_products_by_arch, load_whitelist_ops, resolve_whitelist_ops, filter_ops_by_whitelist
from .op_step0_mc2_analyzer import analyze_mc2_and_generate_json, analyze_all_operators