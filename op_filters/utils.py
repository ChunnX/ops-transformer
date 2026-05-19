# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import json
import logging
import os
from typing import Dict, List


def setup_logging(log_file_path: str) -> logging.Logger:
    """
    配置日志系统，同时输出到控制台和文件。

    Args:
        log_file_path: 日志文件路径

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger("op_test_selector")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    output_dir = os.path.dirname(log_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def read_file_lines(file_path: str, logger: logging.Logger) -> List[str]:
    """
    读取文件的所有非空行。

    Args:
        file_path: 文件路径
        logger: 日志记录器

    Returns:
        非空行列表（去除首尾空白）
    """
    lines = []
    encodings = ["utf-8-sig", "utf-16", "utf-8", "gbk"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        lines.append(stripped)
            logger.debug(f"成功读取文件: {file_path}, 共 {len(lines)} 行, 编码: {encoding}")
            return lines
        except UnicodeDecodeError:
            lines = []
            continue
        except Exception as e:
            logger.error(f"读取文件失败: {file_path}, 错误: {e}")
            raise
    raise UnicodeDecodeError(f"无法用支持的编码读取文件: {file_path}")


def load_dependency_json(file_path: str, logger: logging.Logger) -> Dict:
    """
    加载依赖关系 JSON 文件。

    Args:
        file_path: JSON 文件路径
        logger: 日志记录器

    Returns:
        依赖关系字典
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"成功加载依赖关系文件: {file_path}, 共 {len(data)} 个算子")
        return data
    except Exception as e:
        logger.error(f"加载依赖关系 JSON 失败: {file_path}, 错误: {e}")
        raise


def normalize_path(path: str) -> str:
    """
    规范化路径，转换为标准格式。

    Args:
        path: 原始路径

    Returns:
        规范化后的路径
    """
    return os.path.normpath(path).replace("\\", "/")