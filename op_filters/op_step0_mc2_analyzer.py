# -*- coding: utf-8 -*-
"""
MC2 算子库依赖关系与产品支持分析模块
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


SUPPORTED_PRODUCT_TYPES = [
    "ascend910b",
    "ascend910_93",
    "ascend950",
]

EXCLUDED_DIRS = {'3rd', 'common', 'tools'}


def find_mc2_directory(base_path: Path = None) -> Path:
    """
    查找 mc2 目录位置。

    Args:
        base_path: 基础搜索路径，默认为当前工作目录

    Returns:
        mc2 目录的 Path 对象

    Raises:
        FileNotFoundError: 如果找不到 mc2 目录
    """
    if base_path is None:
        base_path = Path.cwd()
    
    mc2_path = base_path / "mc2"
    if mc2_path.exists() and mc2_path.is_dir():
        return mc2_path
    
    raise FileNotFoundError(
        f"无法在 {base_path} 找到 mc2 目录。"
    )


def get_operator_directories(mc2_path: Path) -> List[Path]:
    """
    获取 mc2 目录下所有算子子目录。

    Args:
        mc2_path: mc2 目录路径

    Returns:
        算子目录路径列表
    """
    operator_dirs = []
    for item in mc2_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name not in EXCLUDED_DIRS:
            operator_dirs.append(item)
    return sorted(operator_dirs, key=lambda x: x.name)


def extract_dependencies_from_cmake(cmake_file: Path) -> Tuple[List[str], List[str]]:
    """
    从 CMakeLists.txt 文件中提取依赖关系。

    Args:
        cmake_file: CMakeLists.txt 文件路径

    Returns:
        (依赖列表, 警告信息列表)
    """
    dependencies = []
    warnings = []
    
    if not cmake_file.exists():
        warnings.append(f"未找到文件: {cmake_file}")
        return dependencies, warnings
    
    try:
        with open(cmake_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith('#'):
                code_lines.append(line)
        
        content = ''.join(code_lines)
        
        pattern = r'set\s*\(\s*(?:\$\{CURRENT_DIR_NAME\}|[\w]+)_depends\s+(.*?)\s+PARENT_SCOPE\s*\)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for match in matches:
            deps = match.split()
            for dep in deps:
                dep = dep.strip()
                if dep and not dep.startswith('#'):
                    normalized_dep = dep.lower()
                    if dep.startswith("mc2/"):
                        normalized_dep = dep[4:].lower()
                    if normalized_dep in EXCLUDED_DIRS:
                        continue
                    if any(normalized_dep.startswith(f"{excluded}/") or normalized_dep.startswith(f"{excluded}_") for excluded in EXCLUDED_DIRS):
                        continue
                    if dep.startswith("mc2/"):
                        dep = dep[4:]
                    dependencies.append(dep)
        
        if 'depends' in content and 'PARENT_SCOPE' not in content:
            warnings.append(f"发现依赖声明但格式不符合预期（缺少 PARENT_SCOPE）: {cmake_file}")
            
    except UnicodeDecodeError:
        warnings.append(f"文件编码错误，无法读取: {cmake_file}")
    except Exception as e:
        warnings.append(f"解析文件时出错 {cmake_file}: {str(e)}")
    
    return dependencies, warnings


def find_def_file(operator_dir: Path) -> Path:
    """
    查找算子目录下的 _def.cpp 文件。

    Args:
        operator_dir: 算子目录路径

    Returns:
        _def.cpp 文件路径，如果未找到则返回 None
    """
    op_host_dir = operator_dir / "op_host"
    if not op_host_dir.exists():
        return None
    
    for file in op_host_dir.iterdir():
        if file.is_file() and file.name.endswith('_def.cpp'):
            return file
    
    return None


def extract_products_from_def(def_file: Path) -> Tuple[List[str], List[str]]:
    """
    从 _def.cpp 文件中提取支持的产品类型。

    Args:
        def_file: _def.cpp 文件路径

    Returns:
        (产品类型列表, 警告信息列表)
    """
    products = []
    warnings = []
    
    if def_file is None:
        warnings.append("未找到 _def.cpp 文件")
        return products, warnings
    
    if not def_file.exists():
        warnings.append(f"未找到文件: {def_file}")
        return products, warnings
    
    try:
        with open(def_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pattern = r'\.AddConfig\s*\(\s*"([^"]+)"\s*,'
        matches = re.findall(pattern, content)
        
        for product in matches:
            if product in SUPPORTED_PRODUCT_TYPES and product not in products:
                products.append(product)
                
    except UnicodeDecodeError:
        warnings.append(f"文件编码错误，无法读取: {def_file}")
    except Exception as e:
        warnings.append(f"解析文件时出错 {def_file}: {str(e)}")
    
    return products, warnings


def analyze_operator(operator_dir: Path) -> Dict:
    """
    分析单个算子的依赖关系和产品支持。

    Args:
        operator_dir: 算子目录路径

    Returns:
        包含依赖关系和产品支持的字典
    """
    operator_name = operator_dir.name
    result = {
        "operator_name": operator_name,
        "dependencies": [],
        "supported_products": [],
        "warnings": []
    }
    
    cmake_file = operator_dir / "op_host" / "CMakeLists.txt"
    deps, deps_warnings = extract_dependencies_from_cmake(cmake_file)
    seen = set()
    unique_deps = []
    for dep in deps:
        if dep not in seen:
            seen.add(dep)
            unique_deps.append(dep)
    result["dependencies"] = unique_deps
    result["warnings"].extend(deps_warnings)
    
    def_file = find_def_file(operator_dir)
    products, products_warnings = extract_products_from_def(def_file)
    result["supported_products"] = products
    result["warnings"].extend(products_warnings)
    
    return result


def analyze_all_operators(mc2_path: Path) -> Dict[str, Dict]:
    """
    分析所有算子。

    Args:
        mc2_path: mc2 目录路径

    Returns:
        所有算子分析结果的字典
    """
    results = {}
    operator_dirs = get_operator_directories(mc2_path)
    
    for operator_dir in operator_dirs:
        analysis = analyze_operator(operator_dir)
        operator_name = analysis["operator_name"]
        results[operator_name] = {
            "dependencies": analysis["dependencies"],
            "supported_products": analysis["supported_products"],
            "warnings": analysis["warnings"]
        }
    
    return results


def save_dependency_json(results: Dict[str, Dict], output_file: str, logger: logging.Logger = None):
    """
    将分析结果保存为 JSON 文件。

    Args:
        results: 分析结果字典
        output_file: 输出文件路径
        logger: 日志记录器（可选）
    """
    try:
        output_path = Path(output_file)
        output_dir = output_path.parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if logger:
            logger.info(f"依赖关系 JSON 已保存到: {output_file}")
    except Exception as e:
        if logger:
            logger.error(f"保存依赖关系 JSON 失败: {e}")
        raise


def analyze_mc2_and_generate_json(mc2_path: str, output_json: str, logger: logging.Logger = None) -> Dict[str, Dict]:
    """
    分析 mc2 算子并生成依赖关系 JSON 文件。

    Args:
        mc2_path: mc2 目录路径
        output_json: 输出 JSON 文件路径
        logger: 日志记录器（可选）

    Returns:
        分析结果字典
    """
    mc2_dir = Path(mc2_path)
    if not mc2_dir.exists():
        error_msg = f"mc2 目录不存在: {mc2_path}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    if logger:
        logger.info(f"开始分析 mc2 目录: {mc2_path}")
    
    results = analyze_all_operators(mc2_dir)
    
    if logger:
        logger.info(f"分析完成，共 {len(results)} 个算子")
    
    save_dependency_json(results, output_json, logger)
    
    return results


def main() -> int:
    """
    独立执行入口，生成依赖关系 JSON 文件。
    """
    import argparse
    
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
    
    DEFAULT_MC2 = str(PROJECT_DIR / "mc2")
    DEFAULT_OUTPUT = str(SCRIPT_DIR / "result.json")
    
    parser = argparse.ArgumentParser(
        description="MC2 算子依赖关系分析工具 - 生成 result.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --mc2    {DEFAULT_MC2}
  --output {DEFAULT_OUTPUT}

示例:
  # 使用默认参数
  python op_filters/op_step0_mc2_analyzer.py
  
  # 自定义 mc2 目录
  python op_filters/op_step0_mc2_analyzer.py --mc2 /path/to/mc2
  
  # 自定义输出路径
  python op_filters/op_step0_mc2_analyzer.py --output custom_result.json
        """
    )
    parser.add_argument(
        "--mc2",
        default=DEFAULT_MC2,
        help=f"mc2 目录路径 (默认: {DEFAULT_MC2})"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出 JSON 文件路径 (默认: {DEFAULT_OUTPUT})"
    )
    
    args = parser.parse_args()
    
    logger = logging.getLogger("mc2_analyzer")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    
    try:
        logger.info("=" * 60)
        logger.info("MC2 算子依赖关系分析")
        logger.info("=" * 60)
        logger.info(f"mc2 目录: {args.mc2}")
        logger.info(f"输出文件: {args.output}")
        
        results = analyze_mc2_and_generate_json(args.mc2, args.output, logger)
        
        logger.info("=" * 60)
        logger.info(f"生成完成，共 {len(results)} 个算子")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())