# -*- coding: utf-8 -*-
"""
忽略文件过滤模块（.md 文件和 tests 目录）
"""

import logging
from typing import List

from .utils import normalize_path


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