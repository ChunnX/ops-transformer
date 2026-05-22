# -*- coding: utf-8 -*-
"""
算子名称提取模块

可作为独立脚本运行，用于从文件路径中提取算子名称。

使用方法：
    python op_filters/op_step5_extractor.py [--input INPUT] [--output OUTPUT] [--arch-restrictions ARCH_JSON] [--log LOG] [--soc SOC_TYPES]
    
    输出:
      --output: 算子名称列表文件
      --arch-restrictions: arch 限制信息 JSON 文件 (供 step7 使用)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
ARCH_RESTRICTIONS_FILE = str(WORK_DIR / "arch_restrictions.json")
LOG_FILE = str(WORK_DIR / "step5.log")

ARCH22_SUPPORTED_PRODUCTS = {"ascend910b", "ascend910_93"}
ARCH35_SUPPORTED_PRODUCTS = {"ascend910b", "ascend950"}

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import normalize_path, setup_logging, read_file_lines
else:
    from .utils import normalize_path, setup_logging, read_file_lines


def extract_op_name(file_path: str, logger: logging.Logger) -> str:
    """
    从文件路径中提取算子名称。

    提取规则：
    1. mc2/op_name/... -> op_name
    2. attention/op_name/... -> op_name
    3. moe/op_name/... -> op_name
    4. gmm/op_name/... -> op_name

    支持相对路径和绝对路径。

    Args:
        file_path: 文件路径
        logger: 日志记录器

    Returns:
        算子名称，如果无法提取则返回空字符串
    """
    normalized_file = normalize_path(file_path)
    
    known_prefixes = ["mc2/", "attention/", "moe/", "gmm/"]
    
    for prefix in known_prefixes:
        if normalized_file.startswith(prefix):
            relative_path = normalized_file[len(prefix):]
            parts = relative_path.split("/")
            if parts and parts[0]:
                return parts[0]
        
        prefix_with_slash = "/" + prefix
        if prefix_with_slash in normalized_file:
            idx = normalized_file.index(prefix_with_slash) + len(prefix_with_slash)
            remaining = normalized_file[idx:]
            parts = remaining.split("/")
            if parts and parts[0]:
                return parts[0]
    
    logger.warning(f"无法从路径提取算子名称: {file_path}")
    return ""


def detect_arch_type(file_path: str, logger: logging.Logger) -> str:
    """
    检测文件路径是否属于 arch22 或 arch35 路径。

    路径模式：
    - mc2/算子名称/op_host/op_tiling/arch22 -> "arch22"
    - mc2/算子名称/op_host/op_tiling/arch35 -> "arch35"

    Args:
        file_path: 文件路径
        logger: 日志记录器

    Returns:
        arch类型 ("arch22", "arch35") 或空字符串
    """
    normalized_file = normalize_path(file_path)
    
    arch_patterns = [
        "mc2/",
        "/mc2/"
    ]
    
    for pattern in arch_patterns:
        if pattern in normalized_file:
            if normalized_file.startswith(pattern):
                idx = len(pattern)
            else:
                idx = normalized_file.index(pattern) + len(pattern)
            
            remaining = normalized_file[idx:]
            parts = remaining.split("/")
            
            if len(parts) >= 5:
                if parts[1] == "op_host" and parts[2] == "op_tiling":
                    if parts[3] == "arch22":
                        logger.debug(f"检测到 arch22 路径: {file_path}")
                        return "arch22"
                    elif parts[3] == "arch35":
                        logger.debug(f"检测到 arch35 路径: {file_path}")
                        return "arch35"
    
    return ""


def extract_operators(file_paths: List[str], logger: logging.Logger) -> dict:
    """
    从文件列表中提取算子名称。

    Args:
        file_paths: 变更文件列表
        logger: 日志记录器

    Returns:
        包含 changed_ops、op_arch_restrictions 和 extraction_errors 的字典
    """
    changed_ops: Set[str] = set()
    op_arch_restrictions: Dict[str, str] = {}
    extraction_errors: List[str] = []
    
    for file_path in file_paths:
        op_name = extract_op_name(file_path, logger)
        if op_name:
            changed_ops.add(op_name)
            arch_type = detect_arch_type(file_path, logger)
            if arch_type:
                if op_name in op_arch_restrictions:
                    existing_arch = op_arch_restrictions[op_name]
                    if existing_arch and existing_arch != arch_type:
                        logger.info(f"算子 {op_name} 检测到多种 arch 路径 ({existing_arch}, {arch_type})，移除 arch 限制")
                        op_arch_restrictions[op_name] = ""
                else:
                    op_arch_restrictions[op_name] = arch_type
                    logger.info(f"算子 {op_name} 受 arch 限制: {arch_type}")
            logger.info(f"从文件提取算子名称: {file_path} -> {op_name}")
        else:
            extraction_errors.append(file_path)
            logger.warning(f"无法提取算子名称: {file_path}")
    
    if extraction_errors:
        logger.warning(f"共有 {len(extraction_errors)} 个文件无法提取算子名称")
    
    logger.info(f"提取到 {len(changed_ops)} 个直接变更的算子: {sorted(changed_ops)}")
    
    arch_restricted_ops = {k: v for k, v in op_arch_restrictions.items() if v}
    if arch_restricted_ops:
        logger.info(f"受 arch 限制的算子: {arch_restricted_ops}")
    
    mixed_arch_ops = [k for k, v in op_arch_restrictions.items() if not v]
    if mixed_arch_ops:
        logger.info(f"涉及多种 arch 的算子（无限制）: {mixed_arch_ops}")
    
    return {
        "changed_ops": changed_ops,
        "op_arch_restrictions": op_arch_restrictions,
        "extraction_errors": extraction_errors
    }


def filter_ops_by_soc(
    changed_ops: Set[str],
    op_arch_restrictions: Dict[str, str],
    soc_types: Set[str],
    logger: logging.Logger
) -> Set[str]:
    """
    根据 SOC 类型过滤算子。

    Args:
        changed_ops: 算子名称集合
        op_arch_restrictions: 算子的 arch 限制字典（空字符串表示无限制）
        soc_types: SOC 类型集合
        logger: 日志记录器

    Returns:
        过滤后的算子集合
    """
    filtered_ops: Set[str] = set()
    ignored_ops: List[str] = []
    
    for op_name in changed_ops:
        arch_type = op_arch_restrictions.get(op_name, "")
        
        if not arch_type:
            filtered_ops.add(op_name)
        elif arch_type == "arch22":
            if soc_types & ARCH22_SUPPORTED_PRODUCTS:
                filtered_ops.add(op_name)
            else:
                ignored_ops.append(op_name)
                logger.info(f"算子 {op_name} (arch22) 被 SOC {soc_types} 过滤")
        elif arch_type == "arch35":
            if soc_types & ARCH35_SUPPORTED_PRODUCTS:
                filtered_ops.add(op_name)
            else:
                ignored_ops.append(op_name)
                logger.info(f"算子 {op_name} (arch35) 被 SOC {soc_types} 过滤")
        else:
            filtered_ops.add(op_name)
    
    if ignored_ops:
        logger.info("=" * 60)
        logger.info(f"因 arch 限制与 SOC 不匹配，忽略 {len(ignored_ops)} 个算子")
        for op in ignored_ops:
            arch_type = op_arch_restrictions.get(op)
            logger.info(f"  忽略算子: {op} (arch: {arch_type}, soc: {soc_types})")
    
    return filtered_ops


def main() -> int:
    """
    算子名称提取独立执行入口。
    """
    parser = argparse.ArgumentParser(
        description="从变更文件列表中提取算子名称",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --input            {INPUT_FILE}
  --output           {OUTPUT_FILE}
  --arch-restrictions {ARCH_RESTRICTIONS_FILE}
  --log              {LOG_FILE}

示例:
  # 使用默认参数
  python op_filters/op_step5_extractor.py

  # 指定 SOC 类型过滤
  python op_filters/op_step5_extractor.py --soc ascend910b ascend910_93

  # 自定义输入输出文件
  python op_filters/op_step5_extractor.py --input custom_input.txt --output custom_output.txt
        """
    )
    parser.add_argument(
        "--input",
        default=INPUT_FILE,
        help=f"输入文件路径 (默认: {INPUT_FILE})"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help=f"输出文件路径 (默认: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--arch-restrictions",
        default=ARCH_RESTRICTIONS_FILE,
        help=f"arch 限制信息输出文件路径 (默认: {ARCH_RESTRICTIONS_FILE})"
    )
    parser.add_argument(
        "--log",
        default=LOG_FILE,
        help=f"日志文件路径 (默认: {LOG_FILE})"
    )
    parser.add_argument(
        "--soc",
        nargs="*",
        choices=["ascend910b", "ascend910_93", "ascend950"],
        help="SOC 类型列表，用于过滤 arch 限制的算子"
    )
    
    args = parser.parse_args()
    
    if args.log:
        logger = setup_logging(args.log)
    else:
        logger = logging.getLogger("operator_extractor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")
        logger.info(f"arch 限制输出文件: {args.arch_restrictions}")
        if args.soc:
            logger.info(f"SOC 类型: {args.soc}")
        
        input_files = read_file_lines(args.input, logger)
        
        logger.info(f"读取到 {len(input_files)} 个变更文件:")
        for f in input_files:
            logger.info(f"  输入文件: {f}")
        
        result = extract_operators(input_files, logger)
        
        logger.info(f"提取到 {len(result['changed_ops'])} 个算子:")
        for op in sorted(result["changed_ops"]):
            logger.info(f"  算子: {op}")
        if result["op_arch_restrictions"]:
            logger.info(f"arch 限制信息:")
            for op, arch in result["op_arch_restrictions"].items():
                logger.info(f"  {op}: {arch}")
        
        soc_types = set(args.soc) if args.soc else set()
        
        if soc_types and result["op_arch_restrictions"]:
            logger.info("=" * 60)
            logger.info("根据 SOC 类型过滤算子")
            filtered_ops = filter_ops_by_soc(
                result["changed_ops"],
                result["op_arch_restrictions"],
                soc_types,
                logger
            )
        else:
            filtered_ops = result["changed_ops"]
        
        if args.output:
            output_dir = Path(args.output).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                for op_name in sorted(filtered_ops):
                    f.write(op_name + "\n")
            logger.info(f"结果已写入: {args.output}")
        
        if args.arch_restrictions and result["op_arch_restrictions"]:
            arch_output_dir = Path(args.arch_restrictions).parent
            if arch_output_dir and not arch_output_dir.exists():
                arch_output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.arch_restrictions, "w", encoding="utf-8") as f:
                json.dump(result["op_arch_restrictions"], f, ensure_ascii=False, indent=2)
            logger.info(f"arch 限制信息已写入: {args.arch_restrictions}")
        else:
            print("提取到的算子名称:")
            for op_name in sorted(filtered_ops):
                print(op_name)
        
        print(f"\n原始提取: {len(result['changed_ops'])} 个算子")
        print(f"过滤后: {len(filtered_ops)} 个算子")
        print(f"提取失败: {len(result['extraction_errors'])} 个文件")
        
        if result["extraction_errors"]:
            print("\n提取失败的文件:")
            for file_path in result["extraction_errors"]:
                print(f"  {file_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())