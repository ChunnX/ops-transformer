# -*- coding: utf-8 -*-
"""
非 mc2 目录检测模块
"""

import logging
from typing import Dict, List

from .utils import normalize_path


def is_in_mc2(file_path: str, mc2_path: str) -> bool:
    """
    判断文件是否在 mc2 目录下。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        是否在 mc2 目录下
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    if normalized_file.startswith(normalized_mc2 + "/") or normalized_file == normalized_mc2:
        return True
    
    if normalized_file.startswith("mc2/") or "/mc2/" in normalized_file:
        return True
    
    return False


def filter_non_mc2(file_paths: List[str], mc2_path: str, dependency_data: Dict, logger: logging.Logger) -> dict:
    """
    检测非 mc2 目录下的变更文件。

    Args:
        file_paths: 变更文件列表
        mc2_path: mc2 目录路径
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        包含 non_mc2_files、mc2_files 和 should_full_test 的字典
    """
    non_mc2_files = []
    mc2_files = []
    
    for file_path in file_paths:
        if not is_in_mc2(file_path, mc2_path):
            non_mc2_files.append(file_path)
        else:
            mc2_files.append(file_path)
    
    should_full_test = False
    if non_mc2_files:
        logger.info("=" * 60)
        logger.info("检测到非 mc2 目录下的变更文件，触发全量测试")
        logger.info(f"非 mc2 文件数量: {len(non_mc2_files)}")
        for f in non_mc2_files:
            logger.info(f"  非 mc2 文件: {f}")
        should_full_test = True
    else:
        logger.info("所有变更文件均在 mc2 目录下")
    
    return {
        "non_mc2_files": non_mc2_files,
        "mc2_files": mc2_files,
        "should_full_test": should_full_test
    }