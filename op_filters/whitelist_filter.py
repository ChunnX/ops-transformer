# -*- coding: utf-8 -*-
"""
白名单过滤模块
"""

import logging
import re
from typing import List

from .utils import normalize_path


def glob_to_regex(pattern: str) -> str:
    """
    将通配符模式转换为正则表达式。
    
    支持的通配符：
    - `**` 匹配任意多层目录（包括零层）
    - `*` 匹配单层目录（不包含路径分隔符）
    
    Args:
        pattern: 通配符模式
    
    Returns:
        正则表达式字符串
    """
    original_pattern = pattern
    ends_with_slash = pattern.rstrip().endswith('/')
    
    pattern = normalize_path(pattern)
    regex = re.escape(pattern)
    
    regex = regex.replace(r'\*\*', '(?:[^/]+/)*[^/]*')
    regex = regex.replace(r'\*', '[^/]+')
    
    if ends_with_slash:
        regex = regex + '/.+'
    elif '**' in pattern or '*' in pattern:
        if not original_pattern.rstrip().endswith('**') and not original_pattern.rstrip().endswith('*'):
            regex += '(/.*)?$'
    else:
        regex += '(/.*)?$'
    
    if not regex.startswith('^'):
        regex = '^' + regex
    
    return regex


def matches_whitelist(file_path: str, whitelist_paths: List[str], logger: logging.Logger) -> bool:
    """
    判断文件是否匹配白名单（支持通配符）。

    支持的匹配模式：
    1. 精确匹配: `mc2/opA/file.py`
    2. 前缀匹配: `mc2/opA/` 或 `mc2/opA`（匹配该目录下所有文件）
    3. 子路径匹配: 当文件路径为绝对路径时，检查路径中是否包含白名单路径作为子路径
    4. 通配符匹配:
       - `*` 匹配单层目录（如 `mc2/*/tests` 匹配 `mc2/任意算子/tests/...`）
       - `**` 匹配多层目录（如 `mc2/**/tests` 匹配 `mc2/任意路径/tests/...`）

    Args:
        file_path: 文件路径（可以是相对路径或绝对路径）
        whitelist_paths: 白名单路径列表（通常为相对路径）
        logger: 日志记录器

    Returns:
        是否匹配白名单
    """
    normalized_file = normalize_path(file_path)
    
    for whitelist_path in whitelist_paths:
        if not whitelist_path.strip():
            continue
            
        normalized_whitelist = normalize_path(whitelist_path)
        whitelist_stripped = normalized_whitelist.rstrip('/')
        
        if normalized_file == normalized_whitelist or normalized_file == whitelist_stripped:
            logger.debug(f"文件 {file_path} 精确匹配白名单: {whitelist_path}")
            return True
        
        if '*' in whitelist_path:
            regex_pattern = glob_to_regex(whitelist_path)
            try:
                if re.match(regex_pattern, normalized_file):
                    logger.debug(f"文件 {file_path} 通配符匹配白名单: {whitelist_path} (正则: {regex_pattern})")
                    return True
                whitelist_regex = '(?:^|/)' + regex_pattern.lstrip('^')
                if re.search(whitelist_regex, normalized_file):
                    logger.debug(f"文件 {file_path} 通配符子路径匹配白名单: {whitelist_path}")
                    return True
            except re.error as e:
                logger.warning(f"白名单模式正则转换失败: {whitelist_path}, 错误: {e}")
                continue
        else:
            if normalized_file.startswith(whitelist_stripped + "/"):
                logger.debug(f"文件 {file_path} 前缀匹配白名单: {whitelist_path}")
                return True
            
            whitelist_with_slash = '/' + whitelist_stripped + '/'
            if whitelist_with_slash in normalized_file:
                logger.debug(f"文件 {file_path} 子路径匹配白名单: {whitelist_path}")
                return True
            
            if normalized_file.endswith('/' + whitelist_stripped):
                logger.debug(f"文件 {file_path} 结尾匹配白名单: {whitelist_path}")
                return True
    
    return False


def filter_whitelist(file_paths: List[str], whitelist_paths: List[str], logger: logging.Logger) -> dict:
    """
    白名单过滤。

    Args:
        file_paths: 变更文件列表
        whitelist_paths: 白名单路径列表
        logger: 日志记录器

    Returns:
        包含 whitelist_hits 和 filtered_files 的字典
    """
    whitelist_hits = []
    filtered_files = []
    
    for file_path in file_paths:
        if matches_whitelist(file_path, whitelist_paths, logger):
            whitelist_hits.append(file_path)
        else:
            filtered_files.append(file_path)
    
    if whitelist_hits:
        logger.info("=" * 60)
        logger.info(f"白名单命中 {len(whitelist_hits)} 个文件，已过滤")
        for f in whitelist_hits:
            logger.info(f"  白名单命中: {f}")
    
    logger.info(f"白名单过滤后剩余 {len(filtered_files)} 个变更文件")
    
    return {
        "whitelist_hits": whitelist_hits,
        "filtered_files": filtered_files
    }