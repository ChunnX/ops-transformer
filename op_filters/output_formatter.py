# -*- coding: utf-8 -*-
"""
输出格式化模块
"""

import json
import logging
import os
from typing import Dict, List, Set


def transform_to_product_grouped(data: List[Dict]) -> Dict[str, List[str]]:
    """
    将算子列表转换为按产品类型分组的字典。

    Args:
        data: 算子信息列表，每个元素包含 op_name 和 supported_products

    Returns:
        按产品类型分组的字典，key 为产品类型，value 为算子名称列表
    """
    result: Dict[str, List[str]] = {}
    for op_info in data:
        op_name = op_info["op_name"]
        supported_products = op_info.get("supported_products", [])
        for product in supported_products:
            if product not in result:
                result[product] = []
            result[product].append(op_name)
    
    for product in result:
        result[product] = sorted(result[product])
    
    return result


def get_all_ops_with_products(dependency_data: Dict, logger: logging.Logger) -> List[Dict]:
    """
    获取所有算子及其支持的芯片类型。

    Args:
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        算子信息列表，每个元素包含 op_name 和 supported_products
    """
    result: List[Dict] = []
    for op_name, op_info in dependency_data.items():
        supported_products = op_info.get("supported_products", [])
        result.append({
            "op_name": op_name,
            "supported_products": supported_products
        })
        warnings = op_info.get("warnings", [])
        if warnings:
            logger.warning(f"算子 {op_name} 有警告: {warnings}")
    
    logger.info(f"共获取 {len(result)} 个算子的信息")
    return result


def get_ops_info_by_names(
    op_names: Set[str],
    dependency_data: Dict,
    logger: logging.Logger
) -> List[Dict]:
    """
    根据算子名称集合获取算子信息。

    Args:
        op_names: 算子名称集合
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        算子信息列表
    """
    result: List[Dict] = []
    for op_name in sorted(op_names):
        if op_name not in dependency_data:
            logger.warning(f"算子 {op_name} 不在依赖关系 JSON 中，将被忽略")
            continue
        
        op_info = dependency_data[op_name]
        supported_products = op_info.get("supported_products", [])
        result.append({
            "op_name": op_name,
            "supported_products": supported_products
        })
        
        warnings = op_info.get("warnings", [])
        if warnings:
            logger.warning(f"算子 {op_name} 有警告: {warnings}")
    
    return result


def write_output_json(data: List[Dict], output_path: str, logger: logging.Logger) -> None:
    """
    写入输出 JSON 文件，按产品类型分组。

    Args:
        data: 算子信息列表，每个元素包含 op_name 和 supported_products
        output_path: 输出文件路径
        logger: 日志记录器
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        grouped_data = transform_to_product_grouped(data)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(grouped_data, f, ensure_ascii=False, indent=2)
        logger.info(f"成功写入输出 JSON 文件: {output_path}")
    except Exception as e:
        logger.error(f"写入输出 JSON 文件失败: {output_path}, 错误: {e}")
        raise