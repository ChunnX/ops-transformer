# -*- coding: utf-8 -*-
"""
算子名称提取模块

可作为独立脚本运行，用于从文件路径中提取算子名称。

使用方法：
    修改脚本顶部的 INPUT_FILE、OUTPUT_FILE、LOG_FILE 变量
    然后直接运行: python op_filters/operator_extractor.py
"""

import logging
import sys
from pathlib import Path
from typing import List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
OUTPUT_FILE = str(WORK_DIR / "operators.txt")
LOG_FILE = str(WORK_DIR / "step5.log")

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


def extract_operators(file_paths: List[str], logger: logging.Logger) -> dict:
    """
    从文件列表中提取算子名称。

    Args:
        file_paths: 变更文件列表
        logger: 日志记录器

    Returns:
        包含 changed_ops 和 extraction_errors 的字典
    """
    changed_ops: Set[str] = set()
    extraction_errors: List[str] = []
    
    for file_path in file_paths:
        op_name = extract_op_name(file_path, logger)
        if op_name:
            changed_ops.add(op_name)
            logger.info(f"从文件提取算子名称: {file_path} -> {op_name}")
        else:
            extraction_errors.append(file_path)
            logger.warning(f"无法提取算子名称: {file_path}")
    
    if extraction_errors:
        logger.warning(f"共有 {len(extraction_errors)} 个文件无法提取算子名称")
    
    logger.info(f"提取到 {len(changed_ops)} 个直接变更的算子: {sorted(changed_ops)}")
    
    return {
        "changed_ops": changed_ops,
        "extraction_errors": extraction_errors
    }


def main() -> int:
    """
    算子名称提取独立执行入口。
    """
    if LOG_FILE:
        logger = setup_logging(LOG_FILE)
    else:
        logger = logging.getLogger("operator_extractor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {INPUT_FILE}")
        logger.info(f"输出文件: {OUTPUT_FILE}")
        
        input_files = read_file_lines(INPUT_FILE, logger)
        
        logger.info(f"读取到 {len(input_files)} 个输入文件")
        
        result = extract_operators(input_files, logger)
        
        if OUTPUT_FILE:
            output_dir = Path(OUTPUT_FILE).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for op_name in sorted(result["changed_ops"]):
                    f.write(op_name + "\n")
            logger.info(f"结果已写入: {OUTPUT_FILE}")
        else:
            print("提取到的算子名称:")
            for op_name in sorted(result["changed_ops"]):
                print(op_name)
        
        print(f"\n成功提取: {len(result['changed_ops'])} 个算子")
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