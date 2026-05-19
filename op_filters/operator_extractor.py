# -*- coding: utf-8 -*-
"""
算子名称提取模块
"""

import logging
from typing import List, Set

from .utils import normalize_path


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