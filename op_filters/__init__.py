# -*- coding: utf-8 -*-
"""
MC2 算子测试选择过滤器模块
"""

from .utils import setup_logging, read_file_lines, load_dependency_json, normalize_path
from .op_whitelist_step1 import matches_whitelist, glob_to_regex, filter_whitelist
from .op_ignored_step2 import is_markdown_file, is_in_tests_directory, filter_ignored
from .op_non_mc2_step3 import is_in_mc2, filter_non_mc2
from .op_common_3rd_step4 import is_in_common, is_in_3rd, get_3rd_subdir, filter_common_3rd, THIRD_PARTY_OP_MAPPING
from .op_extractor_step5 import extract_op_name, extract_operators
from .op_dependency_step6 import build_reverse_dependency_graph, find_dependent_ops
from .op_output_step7 import transform_to_product_grouped, write_output_json, get_all_ops_with_products
from .mc2_analyzer import analyze_mc2_and_generate_json, analyze_all_operators