# -*- coding: utf-8 -*-
"""
输出格式化模块

可作为独立脚本运行，用于将算子列表转换为按产品类型分组的 JSON 格式输出。

使用方法：
    修改脚本顶部的 INPUT_FILE、DEPENDENCY_JSON、OUTPUT_JSON、LOG_FILE、FULL_TEST 变量
    然后直接运行: python op_filters/output_formatter.py
    
    如果 FULL_TEST 为 True，将生成全量测试列表（忽略 INPUT_FILE）
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
WORK_DIR = Path.cwd()

INPUT_FILE = str(WORK_DIR / "operators.txt")
SPECIAL_3RD_OPS_FILE = str(WORK_DIR / "special_3rd_ops.txt")
DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
OUTPUT_JSON = str(WORK_DIR / "op_tests.json")
LOG_FILE = str(WORK_DIR / "step7.log")
FULL_TEST = False

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import setup_logging, read_file_lines, load_dependency_json
else:
    from .utils import setup_logging, read_file_lines, load_dependency_json


def transform_to_product_grouped(data: List[Dict]) -> Dict[str, List[str]]:
    """
    将算子列表转换为按产品类型分组的字典。

    Args:
        data: 箔子信息列表，每个元素包含 op_name 和 supported_products

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
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        grouped_data = transform_to_product_grouped(data)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(grouped_data, f, ensure_ascii=False, indent=2)
        logger.info(f"成功写入输出 JSON 文件: {output_path}")
    except Exception as e:
        logger.error(f"写入输出 JSON 文件失败: {output_path}, 错误: {e}")
        raise


def main() -> int:
    """
    输出格式化独立执行入口。
    """
    if LOG_FILE:
        logger = setup_logging(LOG_FILE)
    else:
        logger = logging.getLogger("output_formatter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"依赖关系文件: {DEPENDENCY_JSON}")
        logger.info(f"输出文件: {OUTPUT_JSON}")
        logger.info(f"全量测试模式: {FULL_TEST}")
        
        dependency_data = load_dependency_json(DEPENDENCY_JSON, logger)
        
        if FULL_TEST:
            logger.info("生成全量测试列表")
            final_ops = get_all_ops_with_products(dependency_data, logger)
        else:
            all_ops_set: Set[str] = set()
            
            if Path(INPUT_FILE).exists():
                logger.info(f"输入文件: {INPUT_FILE}")
                input_ops = read_file_lines(INPUT_FILE, logger)
                logger.info(f"读取到 {len(input_ops)} 个算子名称")
                all_ops_set.update(input_ops)
            
            if Path(SPECIAL_3RD_OPS_FILE).exists():
                logger.info(f"特殊 3rd 算子文件: {SPECIAL_3RD_OPS_FILE}")
                special_ops = read_file_lines(SPECIAL_3RD_OPS_FILE, logger)
                logger.info(f"读取到 {len(special_ops)} 个特殊 3rd 算子")
                all_ops_set.update(special_ops)
            
            if not all_ops_set:
                logger.warning("没有读取到任何算子名称，输出空列表")
                final_ops = []
            else:
                logger.info(f"合并后共 {len(all_ops_set)} 个算子")
                final_ops = get_ops_info_by_names(all_ops_set, dependency_data, logger)
        
        write_output_json(final_ops, OUTPUT_JSON, logger)
        
        print(f"成功生成测试列表: {len(final_ops)} 个算子")
        print(f"输出文件: {OUTPUT_JSON}")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())