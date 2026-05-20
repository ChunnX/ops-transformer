# -*- coding: utf-8 -*-
"""
忽略文件过滤模块（.md 文件和 tests 目录）

可作为独立脚本运行，用于忽略 .md 文件和 tests 目录中的文件。

使用方法：
    修改脚本顶部的 INPUT_FILE、MC2_DIR、OUTPUT_FILE、LOG_FILE 变量
    然后直接运行: python op_filters/ignored_filter.py
"""

import logging
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
MC2_DIR = str(PROJECT_DIR / "mc2")
OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
LOG_FILE = str(WORK_DIR / "step2.log")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import normalize_path, setup_logging, read_file_lines
else:
    from .utils import normalize_path, setup_logging, read_file_lines


def is_markdown_file(file_path: str) -> bool:
    """
    判断文件是否为 Markdown 文件（.md 后缀）。

    Args:
        file_path: 文件路径

    Returns:
        是否为 Markdown 文件
    """
    normalized_file = normalize_path(file_path)
    return normalized_file.lower().endswith(".md")


def is_in_tests_directory(file_path: str, mc2_path: str) -> bool:
    """
    判断文件是否在 mc2/<算子名称>/tests 目录下。

    支持相对路径、Windows绝对路径和Linux绝对路径。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        是否在 tests 目录下
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    if normalized_file.startswith(normalized_mc2 + "/"):
        relative_path = normalized_file[len(normalized_mc2) + 1:]
        parts = relative_path.split("/")
        if len(parts) >= 2 and parts[1] == "tests":
            return True
    
    if normalized_file.startswith("mc2/"):
        relative_path = normalized_file[4:]
        parts = relative_path.split("/")
        if len(parts) >= 2 and parts[1] == "tests":
            return True
    
    if "/mc2/" in normalized_file:
        idx = normalized_file.index("/mc2/") + 5
        remaining = normalized_file[idx:]
        parts = remaining.split("/")
        if len(parts) >= 2 and parts[1] == "tests":
            return True
    
    return False


def filter_ignored(file_paths: List[str], mc2_path: str, logger: logging.Logger) -> dict:
    """
    忽略 .md 文件和 tests 目录。

    Args:
        file_paths: 变更文件列表
        mc2_path: mc2 目录路径
        logger: 日志记录器

    Returns:
        包含 ignored_files 和 remaining_files 的字典
    """
    ignored_files = []
    remaining_files = []
    
    for file_path in file_paths:
        if is_markdown_file(file_path):
            ignored_files.append(file_path)
            logger.debug(f"忽略 Markdown 文件: {file_path}")
        elif is_in_tests_directory(file_path, mc2_path):
            ignored_files.append(file_path)
            logger.debug(f"忽略 tests 目录文件: {file_path}")
        else:
            remaining_files.append(file_path)
    
    if ignored_files:
        logger.info("=" * 60)
        logger.info(f"忽略 {len(ignored_files)} 个文件（.md 文件或 tests 目录）")
        for f in ignored_files:
            logger.info(f"  忽略文件: {f}")
    
    logger.info(f"忽略规则过滤后剩余 {len(remaining_files)} 个变更文件")
    
    return {
        "ignored_files": ignored_files,
        "remaining_files": remaining_files
    }


def main() -> int:
    """
    忽略文件过滤独立执行入口。
    """
    if LOG_FILE:
        logger = setup_logging(LOG_FILE)
    else:
        logger = logging.getLogger("ignored_filter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {INPUT_FILE}")
        logger.info(f"mc2 目录: {MC2_DIR}")
        logger.info(f"输出文件: {OUTPUT_FILE}")
        
        input_files = read_file_lines(INPUT_FILE, logger)
        
        logger.info(f"读取到 {len(input_files)} 个输入文件")
        
        result = filter_ignored(input_files, MC2_DIR, logger)
        
        if OUTPUT_FILE:
            output_dir = Path(OUTPUT_FILE).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for file_path in result["remaining_files"]:
                    f.write(file_path + "\n")
            logger.info(f"结果已写入: {OUTPUT_FILE}")
        else:
            print("过滤后的文件列表:")
            for file_path in result["remaining_files"]:
                print(file_path)
        
        print(f"\n忽略文件: {len(result['ignored_files'])} 个")
        print(f"剩余文件: {len(result['remaining_files'])} 个")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())