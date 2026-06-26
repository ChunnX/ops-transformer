#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC2 算子测试选择主脚本

根据 Git PR 变更文件列表、白名单和算子依赖关系，确定需要执行的算子测试用例。

该脚本实现了以下功能：
1. 检测 mc2 目录外的变更，触发全量测试
2. 白名单过滤
3. common 目录变更触发全量测试
4. 特殊 3rd 子目录变更添加特定算子
5. 其他 3rd 目录变更触发全量测试
6. 忽略 .md 文件和 tests 目录的变更
7. 基于依赖关系选择需要测试的算子
8. 根据 SOC 类型过滤 arch 限制的算子

使用方法：
    # 使用默认参数（推荐）
    python select_op_tests_main.py --pr changed_files.txt
    
    # 指定 SOC 类型
    python select_op_tests_main.py --pr changed_files.txt --soc ascend910b ascend910_93
    
    # 自定义参数
    python select_op_tests_main.py \
        --pr changed_files.txt \
        --whitelist custom_whitelist.txt \
        --dependency-json custom_result.json \
        --mc2 /path/to/mc2 \
        --output-json custom_op_tests.json \
        --output-log custom_output.log \
        --soc ascend910b ascend910_93
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import setup_logging, read_file_lines, load_dependency_json, normalize_path
    from op_filters.op_step1_whitelist import filter_whitelist
    from op_filters.op_step2_ignored import filter_ignored
    from op_filters.op_step3_non_mc2 import filter_non_mc2
    from op_filters.op_step4_common_3rd import filter_common_3rd
    from op_filters.op_step5_extractor import extract_operators, filter_ops_by_soc
    from op_filters.op_step6_dependency import build_reverse_dependency_graph, find_dependent_ops
    from op_filters.op_step7_output import write_output_json, get_all_ops_with_products, get_ops_info_by_names, adjust_supported_products_by_arch, write_soc_json, load_whitelist_ops, resolve_whitelist_ops, filter_ops_by_whitelist
    from op_filters.op_step0_mc2_analyzer import analyze_mc2_and_generate_json
else:
    from .utils import setup_logging, read_file_lines, load_dependency_json, normalize_path
    from .op_step1_whitelist import filter_whitelist
    from .op_step2_ignored import filter_ignored
    from .op_step3_non_mc2 import filter_non_mc2
    from .op_step4_common_3rd import filter_common_3rd
    from .op_step5_extractor import extract_operators, filter_ops_by_soc
    from .op_step6_dependency import build_reverse_dependency_graph, find_dependent_ops
    from .op_step7_output import write_output_json, get_all_ops_with_products, get_ops_info_by_names, adjust_supported_products_by_arch, write_soc_json, load_whitelist_ops, resolve_whitelist_ops, filter_ops_by_whitelist
    from .op_step0_mc2_analyzer import analyze_mc2_and_generate_json


DEFAULT_WHITELIST = str(SCRIPT_DIR / "whitelist.txt")
DEFAULT_WHITELIST_OPS = str(SCRIPT_DIR / "whitelist_ops.txt")
DEFAULT_DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
DEFAULT_MC2 = str(PROJECT_DIR / "mc2")
DEFAULT_OUTPUT_JSON = str(WORK_DIR / "op_tests.json")
DEFAULT_OUTPUT_LOG = str(WORK_DIR / "output.log")
DEFAULT_SPECIAL_3RD_OPS_FILE = str(WORK_DIR / "special_3rd_ops.txt")
DEFAULT_SPECIAL_COMMON_OPS_FILE = str(WORK_DIR / "special_common_ops.txt")


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
        "--whitelist-ops",
        default=DEFAULT_WHITELIST_OPS,
        help=f"算子白名单文件路径 (默认: {DEFAULT_WHITELIST_OPS})"
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
    parser.add_argument(
        "--soc",
        nargs="*",
        choices=["ascend910b", "ascend910_93", "ascend950"],
        help="SOC 类型列表，用于过滤 arch 限制的算子和最终输出"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.output_log)
    
    soc_types = set(args.soc) if args.soc else set()
    
    try:
        logger.info("=" * 60)
        logger.info("开始处理算子测试选择")
        logger.info("=" * 60)
        
        logger.info(f"PR: {args.pr}")
        logger.info(f"白名单文件: {args.whitelist}")
        logger.info(f"算子白名单文件: {args.whitelist_ops}")
        logger.info(f"mc2 目录: {args.mc2}")
        logger.info(f"输出 JSON: {args.output_json}")
        logger.info(f"输出日志: {args.output_log}")
        if soc_types:
            logger.info(f"SOC 类型: {sorted(soc_types)}")
        
        whitelist_sections = load_whitelist_ops(args.whitelist_ops, logger)
        whitelist_ops = resolve_whitelist_ops(whitelist_sections, soc_types, logger)
        
        changed_files = read_file_lines(args.pr, logger)
        logger.info(f"读取到 {len(changed_files)} 个变更文件:")
        for f in changed_files:
            logger.info(f"  变更文件: {f}")
        
        whitelist_paths = read_file_lines(args.whitelist, logger)
        logger.info(f"读取到 {len(whitelist_paths)} 个白名单路径:")
        for p in whitelist_paths:
            logger.info(f"  白名单: {p}")
        
        logger.info("=" * 60)
        logger.info(f"动态生成依赖关系文件: {args.dependency_json}")
        dependency_data = analyze_mc2_and_generate_json(args.mc2, args.dependency_json, logger)
        logger.info(f"依赖关系生成完成，共 {len(dependency_data)} 个算子")
        
        mc2_path = normalize_path(args.mc2)
        
        whitelist_result = filter_whitelist(changed_files, whitelist_paths, logger)
        filtered_files = whitelist_result["filtered_files"]
        logger.info(f"白名单过滤后剩余 {len(filtered_files)} 个变更文件:")
        for f in filtered_files:
            logger.info(f"  剩余文件: {f}")
        
        mc2_external_files = [f for f in filtered_files
                              if not (normalize_path(f).startswith("mc2/")
                                      or normalize_path(f).startswith(mc2_path + "/")
                                      or normalize_path(f) == mc2_path)]
        if mc2_external_files:
            logger.info("=" * 60)
            logger.info("检测到 mc2 外部文件变更（非 mc2/ 开头），触发全量测试")
            logger.info(f"mc2 外部文件数量: {len(mc2_external_files)}")
            for f in mc2_external_files:
                logger.info(f"  mc2 外部文件: {f}")
            all_ops = get_all_ops_with_products(dependency_data, logger)
            all_ops = adjust_supported_products_by_arch(all_ops, {}, soc_types, logger)
            all_ops = filter_ops_by_whitelist(all_ops, whitelist_ops, logger)
            write_output_json(all_ops, args.output_json, logger)
            if soc_types:
                soc_name = sorted(soc_types)[0]
                output_dir = str(Path(args.output_json).parent)
                write_soc_json(all_ops, soc_name, output_dir, logger)
            logger.info("=" * 60)
            logger.info("处理完成（全量测试模式 - mc2 外部文件）")
            logger.info("=" * 60)
            return 0
        
        if not filtered_files:
            logger.info("=" * 60)
            logger.info("白名单过滤后无剩余变更文件，无需执行任何测试")
            write_output_json([], args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（无测试模式）")
            logger.info("=" * 60)
            return 0
        
        ignored_result = filter_ignored(filtered_files, mc2_path, logger)
        remaining_files = ignored_result["remaining_files"]
        logger.info(f"忽略规则过滤后剩余 {len(remaining_files)} 个变更文件:")
        for f in remaining_files:
            logger.info(f"  剩余文件: {f}")
        
        if not remaining_files:
            logger.info("=" * 60)
            logger.info("忽略规则过滤后无剩余变更文件，无需执行任何测试")
            write_output_json([], args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（无测试模式）")
            logger.info("=" * 60)
            return 0
        
        non_mc2_result = filter_non_mc2(remaining_files, mc2_path, dependency_data, logger)
        if non_mc2_result["should_full_test"]:
            all_ops = get_all_ops_with_products(dependency_data, logger)
            all_ops = adjust_supported_products_by_arch(all_ops, {}, soc_types, logger)
            all_ops = filter_ops_by_whitelist(all_ops, whitelist_ops, logger)
            write_output_json(all_ops, args.output_json, logger)
            if soc_types:
                soc_name = sorted(soc_types)[0]
                output_dir = str(Path(args.output_json).parent)
                write_soc_json(all_ops, soc_name, output_dir, logger)
            logger.info("=" * 60)
            logger.info("处理完成（全量测试模式）")
            logger.info("=" * 60)
            return 0
        
        common_3rd_result = filter_common_3rd(remaining_files, mc2_path, dependency_data, logger)
        if common_3rd_result["should_full_test"]:
            all_ops = get_all_ops_with_products(dependency_data, logger)
            all_ops = adjust_supported_products_by_arch(all_ops, {}, soc_types, logger)
            all_ops = filter_ops_by_whitelist(all_ops, whitelist_ops, logger)
            write_output_json(all_ops, args.output_json, logger)
            if soc_types:
                soc_name = sorted(soc_types)[0]
                output_dir = str(Path(args.output_json).parent)
                write_soc_json(all_ops, soc_name, output_dir, logger)
            logger.info("=" * 60)
            logger.info("处理完成（全量测试模式）")
            logger.info("=" * 60)
            return 0
        
        other_files = common_3rd_result["other_files"]
        # [已禁用] 特殊处理相关字段，common/3rd 变更现在直接触发全量测试
        # special_3rd_ops = common_3rd_result["special_3rd_ops"]
        # special_common_ops = common_3rd_result["special_common_ops"]
        special_3rd_ops = set()
        special_common_ops = set()
        logger.info(f"common/3rd过滤后剩余 {len(other_files)} 个变更文件:")
        for f in other_files:
            logger.info(f"  算子文件: {f}")
        
        extraction_result = extract_operators(other_files, logger)
        changed_ops = extraction_result["changed_ops"]
        op_arch_restrictions = extraction_result.get("op_arch_restrictions", {})
        logger.info(f"提取到 {len(changed_ops)} 个算子:")
        for op in sorted(changed_ops):
            logger.info(f"  算子: {op}")
        if op_arch_restrictions:
            logger.info(f"arch 限制信息:")
            for op, arch in op_arch_restrictions.items():
                logger.info(f"  {op}: {arch}")
        
        if soc_types and op_arch_restrictions:
            logger.info("=" * 60)
            logger.info("根据 SOC 类型过滤 arch 限制的算子")
            changed_ops = filter_ops_by_soc(changed_ops, op_arch_restrictions, soc_types, logger)
        
        if not changed_ops and not special_3rd_ops and not special_common_ops:
            logger.warning("没有成功提取到任何算子名称，输出空列表")
            write_output_json([], args.output_json, logger)
            logger.info("=" * 60)
            logger.info("处理完成（无测试模式）")
            logger.info("=" * 60)
            return 0
        
        logger.info("=" * 60)
        logger.info("开始构建反向依赖图")
        reverse_graph = build_reverse_dependency_graph(dependency_data, logger)
        
        logger.info("=" * 60)
        logger.info("开始反向依赖搜索")
        ops_to_run = find_dependent_ops(changed_ops, reverse_graph, logger)
        
        if special_3rd_ops:
            logger.info("=" * 60)
            # [已禁用] 合并特殊 3rd 子目录关联算子，common/3rd 变更现在直接触发全量测试
            # 如需恢复，取消注释下方代码
            # logger.info("合并特殊 3rd 子目录关联算子（不做反向依赖处理）")
            # ops_to_run = ops_to_run.union(special_3rd_ops)
            # logger.info(f"合并后算子数量: {len(ops_to_run)}")
            pass
        
        if special_common_ops:
            logger.info("=" * 60)
            # [已禁用] 合并特殊 common 文件关联算子，common/3rd 变更现在直接触发全量测试
            # 如需恢复，取消注释下方代码
            # logger.info("合并特殊 common 文件关联算子（不做反向依赖处理）")
            # ops_to_run = ops_to_run.union(special_common_ops)
            # logger.info(f"合并后算子数量: {len(ops_to_run)}")
            pass
        
        logger.info("=" * 60)
        logger.info("生成最终算子列表")
        final_ops = get_ops_info_by_names(ops_to_run, dependency_data, logger)
        
        final_ops = adjust_supported_products_by_arch(final_ops, op_arch_restrictions, soc_types, logger)
        final_ops = filter_ops_by_whitelist(final_ops, whitelist_ops, logger)
        
        logger.info(f"最终需要测试的算子数量: {len(final_ops)}")
        for op_info in final_ops:
            logger.info(
                f"  算子: {op_info['op_name']}, "
                f"支持芯片: {op_info['supported_products']}"
            )
        
        write_output_json(final_ops, args.output_json, logger)
        if soc_types:
            soc_name = sorted(soc_types)[0]
            output_dir = str(Path(args.output_json).parent)
            write_soc_json(final_ops, soc_name, output_dir, logger)
        
        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"处理过程中发生未捕获的异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())