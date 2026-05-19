# -*- coding: utf-8 -*-
"""
MC2 算子测试选择过滤器模块
"""

from .utils import setup_logging, read_file_lines, load_dependency_json, normalize_path
from .whitelist_filter import matches_whitelist, glob_to_regex, filter_whitelist
from .ignored_filter import is_markdown_file, is_in_tests_directory, filter_ignored
from .non_mc2_filter import is_in_mc2, filter_non_mc2
from .common_3rd_filter import is_in_common_or_3rd, filter_common_3rd
from .operator_extractor import extract_op_name, extract_operators
from .dependency_search import build_reverse_dependency_graph, find_dependent_ops
from .output_formatter import transform_to_product_grouped, write_output_json, get_all_ops_with_products
from .mc2_analyzer import analyze_mc2_and_generate_json, analyze_all_operators