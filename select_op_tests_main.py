#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC2 算子测试选择主脚本

根据 Git PR 变更文件列表、白名单和算子依赖关系，确定需要执行的算子测试用例。

该脚本实现了以下功能：
1. 检测 mc2 目录外的变更，触发全量测试
2. 白名单过滤
3. common/3rd 目录变更触发全量测试
4. 忽略 .md 文件和 tests 目录的变更
5. 基于依赖关系选择需要测试的算子

使用方法：
    # 使用默认参数（推荐）
    python select_op_tests_main.py --pr changed_files.txt
    
    # 自定义参数
    python select_op_tests_main.py \
        --pr changed_files.txt \
        --whitelist custom_whitelist.txt \
        --dependency-json custom_result.json \
        --mc2 /path/to/mc2 \
        --output-json custom_op_tests.json \
        --output-log custom_output.log
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from op_filters.utils import setup_logging, read_file_lines, load_dependency_json, normalize_path
from op_filters.whitelist_filter import filter_whitelist
from op_filters.ignored_filter import filter_ignored
from op_filters.non_mc2_filter import filter_non_mc2
from op_filters.common_3rd_filter import filter_common_3rd
from op_filters.operator_extractor import extract_operators
from op_filters.dependency_search import build_reverse_dependency_graph, find_dependent_ops
from op_filters.output_formatter import write_output_json, get_all_ops_with_products, get_ops_info_by_names
from op_filters.mc2_analyzer import analyze_mc2_and_generate_json


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_WHITELIST = str(SCRIPT_DIR / "whitelist.txt")
DEFAULT_DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
DEFAULT_MC2 = str(SCRIPT_DIR / "mc2")
DEFAULT_OUTPUT_JSON = str(SCRIPT_DIR / "op_tests.json")
DEFAULT_OUTPUT_LOG = str(SCRIPT_DIR / "output.log")


def main() -> int:
    """
    主函数，执行主要的处理逻辑。

    Returns:
        退出码，0 表示成功，非 0 表示失败
    """
    parser = argparse.ArgumentParser(
        description="根据 PR 变更文件、白名单和依赖关系，确定需要执行的算子测试用例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --whitelist        {DEFAULT_WHITELIST}
  --dependency-json  {DEFAULT_DEPENDENCY_JSON}
  --mc2              {DEFAULT_MC2}
  --output-json      {DEFAULT_OUTPUT_JSON}
  --output-log       {DEFAULT_OUTPUT_LOG}

示例:
  # 使用默认参数
  python select_op_tests_main.py --pr changed_files.txt

  # 自定义白名单文件
  python select_op_tests_main.py --pr changed_files.txt --whitelist custom_whitelist.txt

  # 自定义所有参数
  python select_op_tests_main.py --pr changed_files.txt --whitelist custom.txt --output-json result.json
        """
    )
    parser.add_argument(
        "--pr",
        required=True,
        help="变更文件列表文件路径"
    )
    parser.add_argument(
        "--whitelist",
        default=DEFAULT_WHITELIST,
        help=f"白名单文件路径 (默认: {DEFAULT_WHITELIST})"
    )
    parser.add_argument(
        "--dependency-json",
        default=DEFAULT_DEPENDENCY_JSON,
        help=f"依赖关系 JSON 文件路径 (默认: {DEFAULT_DEPENDENCY_JSON})"
    )
    parser.add_argument(
        "--mc2",
        default=DEFAULT_MC2,
        help=f"mc2 目录的路径 (默认: {DEFAULT_MC2})"
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help=f"输出 JSON 文件路径 (默认: {DEFAULT_OUTPUT_JSON})"
    )
    parser.add_argument(
        "--output-log",
        default=DEFAULT_OUTPUT_LOG,
        help=f"输出日志文件路径 (默认: {DEFAULT_OUTPUT_LOG})"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_log)
    
    try:
        logger.info("=" * 60)
        logger.info("开始处理算子测试选择")
        logger.info("=" * 60)
        
        logger.info(f"PR: {args.pr}")
        logger.info(f"白名单文件: {args.whitelist}")
        logger.info(f"依赖关系文件: {args.dependency_json}")
        logger.info(f"mc2 目录: {args.mc2}")
        logger.info(f"输出 JSON: {args.output_json}")
        logger.info(f"输出日志: {args.output_log}")
        
        analyze_mc2_and_generate_json(args.mc2, args.dependency_json, logger)
        
        changed_files = read_file_lines(args.pr, logger)
        logger.info(f"读取到 {len(changed_files)} 个变更文件")
        for i, f in enumerate(changed_files, 1):
            logger.debug(f"  变更文件 {i}: {f}")
        
        whitelist_paths = read_file_lines(args.whitelist, logger)
        logger.info(f"读取到 {len(whitelist_paths)} 个白名单路径")
        for i, p in enumerate(whitelist_paths, 1):
            logger.debug(f"  白名单 {i}: {p}")
        
        dependency_data = load_dependency_json(args.dependency_json, logger)
        
        mc2_path = normalize_path(args.mc2)
        
        # Step 1: 白名单过滤
        whitelist_result = filter_whitelist(changed_files, whitelist_paths, logger)
        filtered_files = whitelist_result["filtered_files"]
        
        if not filtered_files:
            logger.info("=" * 60)
            logger.info("白名单过滤后无剩余变更文件，无需执行任何测试")
            write_output_json([], args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（无测试模式）")
            logger.info("=" * 60)
            return 0
        
        # Step 2: 忽略 .md 文件和 tests 目录
        ignored_result = filter_ignored(filtered_files, mc2_path, logger)
        remaining_files = ignored_result["remaining_files"]
        
        if not remaining_files:
            logger.info("=" * 60)
            logger.info("忽略规则过滤后无剩余变更文件，无需执行任何测试")
            write_output_json([], args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（无测试模式）")
            logger.info("=" * 60)
            return 0
        
        # Step 3: 检测非 mc2 目录文件
        non_mc2_result = filter_non_mc2(remaining_files, mc2_path, dependency_data, logger)
        if non_mc2_result["should_full_test"]:
            all_ops = get_all_ops_with_products(dependency_data, logger)
            write_output_json(all_ops, args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（全量测试模式）")
            logger.info("=" * 60)
            return 0
        
        # Step 4: 检测 common/3rd 目录文件
        common_3rd_result = filter_common_3rd(remaining_files, mc2_path, dependency_data, logger)
        if common_3rd_result["should_full_test"]:
            all_ops = get_all_ops_with_products(dependency_data, logger)
            write_output_json(all_ops, args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（全量测试模式）")
            logger.info("=" * 60)
            return 0
        
        other_files = common_3rd_result["other_files"]
        
        # Step 5: 提取算子名称
        extraction_result = extract_operators(other_files, logger)
        changed_ops = extraction_result["changed_ops"]
        
        if not changed_ops:
            logger.warning("没有成功提取到任何算子名称，输出空列表")
            write_output_json([], args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（无测试模式）")
            logger.info("=" * 60)
            return 0
        
        # Step 6: 构建反向依赖图
        logger.info("=" * 60)
        logger.info("开始构建反向依赖图")
        reverse_graph = build_reverse_dependency_graph(dependency_data, logger)
        
        # Step 7: 反向依赖搜索
        logger.info("=" * 60)
        logger.info("开始反向依赖搜索")
        ops_to_run = find_dependent_ops(changed_ops, reverse_graph, logger)
        
        # Step 8: 生成最终算子列表
        logger.info("=" * 60)
        logger.info("生成最终算子列表")
        final_ops = get_ops_info_by_names(ops_to_run, dependency_data, logger)
        
        logger.info(f"最终需要测试的算子数量: {len(final_ops)}")
        for op_info in final_ops:
            logger.info(
                f"  算子: {op_info['op_name']}, "
                f"支持芯片: {op_info['supported_products']}"
            )
        
        write_output_json(final_ops, args.output_json, logger)
        
        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中发生未捕获的异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())