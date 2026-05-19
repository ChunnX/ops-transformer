# -*- coding: utf-8 -*-
"""
common/3rd 目录检测模块
"""

import logging
from typing import Dict, List

from .utils import normalize_path


def is_in_common_or_3rd(file_path: str, mc2_path: str) -> bool:
    """
    判断文件是否在 mc2/common/ 或 mc2/3rd/ 目录下。

    支持相对路径、Windows绝对路径和Linux绝对路径。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        是否在 common 或 3rd 目录下
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    common_prefix_abs = normalized_mc2 + "/common"
    third_prefix_abs = normalized_mc2 + "/3rd"
    common_prefix_rel = "mc2/common"
    third_prefix_rel = "mc2/3rd"
    
    if normalized_file.startswith(common_prefix_abs + "/") or normalized_file == common_prefix_abs:
        return True
    if normalized_file.startswith(third_prefix_abs + "/") or normalized_file == third_prefix_abs:
        return True
    if normalized_file.startswith(common_prefix_rel + "/") or normalized_file == common_prefix_rel:
        return True
    if normalized_file.startswith(third_prefix_rel + "/") or normalized_file == third_prefix_rel:
        return True
    
    if "/mc2/common/" in normalized_file:
        return True
    if "/mc2/3rd/" in normalized_file:
        return True
    
    return False


def filter_common_3rd(file_paths: List[str], mc2_path: str, dependency_data: Dict, logger: logging.Logger) -> dict:
    """
    检测 common/3rd 目录下的变更文件。

    Args:
        file_paths: 变更文件列表
        mc2_path: mc2 目录路径
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        包含 common_3rd_files、other_files 和 should_full_test 的字典
    """
    common_3rd_files = []
    other_files = []
    
    for file_path in file_paths:
        if is_in_common_or_3rd(file_path, mc2_path):
            common_3rd_files.append(file_path)
        else:
            other_files.append(file_path)
    
    should_full_test = False
    if common_3rd_files:
        logger.info("=" * 60)
        logger.info("检测到 common/3rd 目录下的变更文件，触发全量测试")
        logger.info(f"common/3rd 文件数量: {len(common_3rd_files)}")
        for f in common_3rd_files:
            logger.info(f"  common/3rd 文件: {f}")
        should_full_test = True
    else:
        logger.info("变更文件不在 common/3rd 目录下")
    
    return {
        "common_3rd_files": common_3rd_files,
        "other_files": other_files,
        "should_full_test": should_full_test
    }