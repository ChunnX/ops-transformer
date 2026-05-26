# -*- coding: utf-8 -*-
"""
输出格式化模块

可作为独立脚本运行，用于将算子列表转换为按产品类型分组的 JSON 格式输出。

使用方法：
    python op_filters/op_step7_output.py [--input INPUT] [--dependency-json JSON] [--output-json JSON] [--log LOG] [--full-test] [--soc SOC_TYPES]
    
    如果使用 --full-test，将生成全量测试列表（忽略输入文件）
    如果使用 --soc，将根据 SOC 类型过滤 arch 限制的算子和产品支持
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

DEFAULT_INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_SPECIAL_3RD_OPS_FILE = str(WORK_DIR / "special_3rd_ops.txt")
DEFAULT_ARCH_RESTRICTIONS_FILE = str(WORK_DIR / "arch_restrictions.json")
DEFAULT_WHITELIST_OPS_FILE = str(SCRIPT_DIR / "whitelist_ops.txt")
DEFAULT_DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
DEFAULT_OUTPUT_JSON = str(WORK_DIR / "op_tests.json")
DEFAULT_LOG_FILE = str(WORK_DIR / "step7.log")

ARCH22_SUPPORTED_PRODUCTS = {"ascend910b", "ascend910_93"}
ARCH35_SUPPORTED_PRODUCTS = {"ascend910b", "ascend950"}

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


def load_whitelist_ops(file_path: str, logger: logging.Logger) -> Dict[str, Set[str]]:
    """
    加载算子白名单文件，按 section 返回屏蔽算子名称字典。

    文件格式：
    - # 开头为注释
    - [soc_name] 为 section 头，该 section 下的算子仅对指定 SOC 屏蔽
    - 无 section 头的算子为全局屏蔽（所有 SOC）

    Args:
        file_path: 算子白名单文件路径
        logger: 日志记录器

    Returns:
        字典，key 为 section 名（"global" 或 SOC 名），value 为算子名称集合
    """
    if not Path(file_path).exists():
        logger.info(f"算子白名单文件不存在: {file_path}")
        return {}
    sections: Dict[str, Set[str]] = {"global": set()}
    current_section = "global"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if stripped.startswith('[') and stripped.endswith(']'):
                    current_section = stripped[1:-1]
                    if current_section not in sections:
                        sections[current_section] = set()
                    continue
                sections[current_section].add(stripped)
        for section_name, ops in sections.items():
            if ops:
                logger.info(f"白名单 section [{section_name}]: {sorted(ops)} ({len(ops)} 个算子)")
        logger.info(f"成功加载算子白名单文件: {file_path}, 共 {len(sections)} 个 section")
        return sections
    except Exception as e:
        logger.warning(f"加载算子白名单文件失败: {file_path}, 错误: {e}")
        return {}


def resolve_whitelist_ops(sections: Dict[str, Set[str]], soc_types: Set[str], logger: logging.Logger) -> Set[str]:
    """
    根据当前 SOC 类型合并白名单算子集合。

    合并规则：
    - global section 中的算子始终屏蔽
    - 匹配当前 SOC 类型的 section 中的算子也屏蔽

    Args:
        sections: 白名单 section 字典
        soc_types: 当前 SOC 类型集合（为空时只屏蔽 global）
        logger: 日志记录器

    Returns:
        合并后的屏蔽算子名称集合
    """
    whitelist_ops = set(sections.get("global", set()))
    if soc_types:
        for soc_name in soc_types:
            if soc_name in sections:
                whitelist_ops.update(sections[soc_name])
                logger.info(f"合并 SOC 白名单 [{soc_name}]: {sorted(sections[soc_name])} ({len(sections[soc_name])} 个算子)")
    if whitelist_ops:
        logger.info(f"最终白名单屏蔽算子: {sorted(whitelist_ops)} ({len(whitelist_ops)} 个)")
    else:
        logger.info("无白名单屏蔽算子")
    return whitelist_ops


def filter_ops_by_whitelist(data: List[Dict], whitelist_ops: Set[str], logger: logging.Logger) -> List[Dict]:
    """
    根据算子白名单过滤，移除白名单中的算子。

    Args:
        data: 算子信息列表
        whitelist_ops: 需屏蔽的算子名称集合
        logger: 日志记录器

    Returns:
        过滤后的算子信息列表
    """
    if not whitelist_ops:
        return data
    removed_ops = []
    filtered = []
    for op_info in data:
        if op_info["op_name"] in whitelist_ops:
            removed_ops.append(op_info["op_name"])
        else:
            filtered.append(op_info)
    if removed_ops:
        logger.info("=" * 60)
        logger.info(f"算子白名单屏蔽: 移除 {sorted(removed_ops)} (原因: 在 whitelist_ops.txt 中)")
        logger.info(f"屏蔽前 {len(data)} 个算子, 屏蔽后 {len(filtered)} 个算子")
    return filtered


def load_arch_restrictions(file_path: str, logger: logging.Logger) -> Dict:
    """
    加载 arch 限制信息 JSON 文件。

    Args:
        file_path: JSON 文件路径
        logger: 日志记录器

    Returns:
        arch 限制字典，key 为算子名称，value 为 arch 类型 ("arch22" 或 "arch35")
    """
    try:
        if not Path(file_path).exists():
            logger.info(f"arch 限制文件不存在: {file_path}")
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"成功加载 arch 限制文件: {file_path}, 共 {len(data)} 个算子")
        return data
    except Exception as e:
        logger.warning(f"加载 arch 限制文件失败: {file_path}, 错误: {e}")
        return {}


def filter_ops_by_arch_and_soc(
    op_names: Set[str],
    arch_restrictions: Dict[str, str],
    soc_types: Set[str],
    logger: logging.Logger
) -> Set[str]:
    """
    根据 arch 限制和 SOC 类型过滤算子。

    Args:
        op_names: 算子名称集合
        arch_restrictions: arch 限制字典
        soc_types: SOC 类型集合
        logger: 日志记录器

    Returns:
        过滤后的算子集合
    """
    if not soc_types:
        return op_names
    
    if not arch_restrictions:
        return op_names
    
    filtered_ops: Set[str] = set()
    ignored_ops: List[str] = []
    
    for op_name in op_names:
        arch_type = arch_restrictions.get(op_name, "")
        
        if not arch_type:
            filtered_ops.add(op_name)
        elif arch_type == "arch22":
            if soc_types & ARCH22_SUPPORTED_PRODUCTS:
                filtered_ops.add(op_name)
            else:
                ignored_ops.append(op_name)
                logger.info(f"算子 {op_name} (arch22) 被 SOC {soc_types} 过滤")
        elif arch_type == "arch35":
            if soc_types & ARCH35_SUPPORTED_PRODUCTS:
                filtered_ops.add(op_name)
            else:
                ignored_ops.append(op_name)
                logger.info(f"算子 {op_name} (arch35) 被 SOC {soc_types} 过滤")
        else:
            filtered_ops.add(op_name)
    
    if ignored_ops:
        logger.info("=" * 60)
        logger.info(f"因 arch 限制与 SOC 不匹配，忽略 {len(ignored_ops)} 个算子")
        for op in ignored_ops:
            arch_type = arch_restrictions.get(op)
            logger.info(f"  忽略算子: {op} (arch: {arch_type}, soc: {soc_types})")
    
    return filtered_ops


def adjust_supported_products_by_arch(
    final_ops: List[Dict],
    arch_restrictions: Dict[str, str],
    soc_types: Set[str],
    logger: logging.Logger
) -> List[Dict]:
    """
    根据 arch 限制裁剪算子的 supported_products。

    无论是否有 --soc 参数，arch 限制的算子的 supported_products 都应该被裁剪：
    - arch22 算子: 只保留 ascend910b 和 ascend910_93
    - arch35 算子: 只保留 ascend910b 和 ascend950

    如果同时指定了 --soc，则还要进一步裁剪为与 soc_types 交集。

    Args:
        final_ops: 算子信息列表
        arch_restrictions: arch 限制字典
        soc_types: SOC 类型集合（可能为空）
        logger: 日志记录器

    Returns:
        裁剪后的算子信息列表
    """
    if not arch_restrictions:
        if soc_types:
            return _filter_by_soc(final_ops, soc_types, logger)
        return final_ops
    
    adjusted_ops: List[Dict] = []
    arch_adjusted_count = 0
    
    for op_info in final_ops:
        op_name = op_info["op_name"]
        supported_products = op_info.get("supported_products", [])
        arch_type = arch_restrictions.get(op_name, "")
        
        if arch_type == "arch22":
            allowed = ARCH22_SUPPORTED_PRODUCTS
            adjusted = sorted([p for p in supported_products if p in allowed])
            if adjusted != supported_products:
                arch_adjusted_count += 1
                logger.debug(f"算子 {op_name} (arch22) supported_products 从 {supported_products} 裁剪为 {adjusted}")
            supported_products = adjusted
        elif arch_type == "arch35":
            allowed = ARCH35_SUPPORTED_PRODUCTS
            adjusted = sorted([p for p in supported_products if p in allowed])
            if adjusted != supported_products:
                arch_adjusted_count += 1
                logger.debug(f"算子 {op_name} (arch35) supported_products 从 {supported_products} 裁剪为 {adjusted}")
            supported_products = adjusted
        
        if soc_types:
            filtered = sorted([p for p in supported_products if p in soc_types])
            if not filtered:
                logger.info(f"  算子 {op_name} 支持的产品 {supported_products} 不包含 {soc_types}，已过滤")
                continue
            supported_products = filtered
        
        adjusted_ops.append({
            "op_name": op_name,
            "supported_products": supported_products
        })
    
    if arch_adjusted_count:
        logger.info("=" * 60)
        logger.info(f"根据 arch 限制裁剪了 {arch_adjusted_count} 个算子的 supported_products")
    
    if soc_types and len(adjusted_ops) != len(final_ops):
        logger.info("=" * 60)
        logger.info(f"根据 SOC 类型 {sorted(soc_types)} 过滤后剩余 {len(adjusted_ops)} 个算子")
    
    return adjusted_ops


def _filter_by_soc(
    final_ops: List[Dict],
    soc_types: Set[str],
    logger: logging.Logger
) -> List[Dict]:
    """
    仅根据 SOC 类型过滤算子（无 arch 限制信息时使用）。

    Args:
        final_ops: 算子信息列表
        soc_types: SOC 类型集合
        logger: 日志记录器

    Returns:
        过滤后的算子信息列表
    """
    filtered_final_ops: List[Dict] = []
    for op_info in final_ops:
        supported_products = set(op_info.get("supported_products", []))
        if supported_products & soc_types:
            filtered_final_ops.append({
                "op_name": op_info["op_name"],
                "supported_products": sorted([p for p in op_info.get("supported_products", []) if p in soc_types])
            })
        else:
            logger.info(f"  算子 {op_info['op_name']} 支持的产品 {supported_products} 不包含 {soc_types}，已过滤")
    
    logger.info("=" * 60)
    logger.info(f"根据 SOC 类型 {sorted(soc_types)} 过滤后剩余 {len(filtered_final_ops)} 个算子")
    
    return filtered_final_ops


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


def write_soc_json(data: List[Dict], soc_name: str, output_dir: str, logger: logging.Logger) -> str:
    """
    写入适配 SOC 类型的算子列表文件，每行一个算子名称。

    Args:
        data: 算子信息列表（已经按 soc 裁剪过 supported_products）
        soc_name: SOC 类型名称（如 ascend910b）
        output_dir: 输出目录
        logger: 日志记录器

    Returns:
        输出文件路径
    """
    try:
        op_names = sorted([op_info["op_name"] for op_info in data])
        
        if soc_name == "ascend950":
            moe_distribute_names = [n for n in op_names if n.startswith("moe_distribute")]
            if moe_distribute_names:
                op_names = [n for n in op_names if not n.startswith("moe_distribute")] + ["moe_distribute"]
                op_names.sort()
                logger.info(f"ascend950 SOC: 合并 {moe_distribute_names} 为单个 moe_distribute")
        
        output_path = str(Path(output_dir) / "matched_keywords.txt")
        output_path_obj = Path(output_path)
        parent_dir = output_path_obj.parent
        if parent_dir and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for op_name in op_names:
                f.write(op_name + "\n")
        logger.info(f"成功写入 SOC 适配文件: {output_path} ({len(op_names)} 个算子)")
        
        return output_path
    except Exception as e:
        logger.error(f"写入 SOC 适配文件失败: {e}")
        raise


def main() -> int:
    """
    输出格式化独立执行入口。
    """
    parser = argparse.ArgumentParser(
        description="输出格式化模块 - 将算子列表转换为按产品类型分组的 JSON 格式输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --input            {DEFAULT_INPUT_FILE}
  --special-3rd-ops  {DEFAULT_SPECIAL_3RD_OPS_FILE}
  --arch-restrictions {DEFAULT_ARCH_RESTRICTIONS_FILE}
  --dependency-json  {DEFAULT_DEPENDENCY_JSON}
  --output-json      {DEFAULT_OUTPUT_JSON}
  --log              {DEFAULT_LOG_FILE}
  --full-test        False

示例:
  # 使用默认参数
  python op_filters/op_step7_output.py
  
  # 全量测试模式
  python op_filters/op_step7_output.py --full-test
  
  # 指定 SOC 类型过滤
  python op_filters/op_step7_output.py --soc ascend910b ascend910_93
  
  # 自定义输入文件
  python op_filters/op_step7_output.py --input my_operators.txt
  
  # 自定义所有参数
  python op_filters/op_step7_output.py --input input.txt --dependency-json deps.json --output-json output.json --log step7.log
        """
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"输入文件路径 (默认: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--special-3rd-ops",
        default=DEFAULT_SPECIAL_3RD_OPS_FILE,
        help=f"特殊 3rd 算子文件路径 (默认: {DEFAULT_SPECIAL_3RD_OPS_FILE})"
    )
    parser.add_argument(
        "--arch-restrictions",
        default=DEFAULT_ARCH_RESTRICTIONS_FILE,
        help=f"arch 限制信息文件路径 (默认: {DEFAULT_ARCH_RESTRICTIONS_FILE})"
    )
    parser.add_argument(
        "--whitelist-ops",
        default=DEFAULT_WHITELIST_OPS_FILE,
        help=f"算子白名单文件路径 (默认: {DEFAULT_WHITELIST_OPS_FILE})"
    )
    parser.add_argument(
        "--dependency-json",
        default=DEFAULT_DEPENDENCY_JSON,
        help=f"依赖关系 JSON 文件路径 (默认: {DEFAULT_DEPENDENCY_JSON})"
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help=f"输出 JSON 文件路径 (默认: {DEFAULT_OUTPUT_JSON})"
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_FILE,
        help=f"日志文件路径 (默认: {DEFAULT_LOG_FILE})"
    )
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="生成全量测试列表（忽略输入文件）"
    )
    parser.add_argument(
        "--soc",
        nargs="*",
        choices=["ascend910b", "ascend910_93", "ascend950"],
        help="SOC 类型列表，用于过滤 arch 限制的算子和最终输出"
    )
    
    args = parser.parse_args()
    
    if args.log:
        logger = setup_logging(args.log)
    else:
        logger = logging.getLogger("output_formatter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        soc_types = set(args.soc) if args.soc else set()
        
        logger.info(f"依赖关系文件: {args.dependency_json}")
        logger.info(f"输出文件: {args.output_json}")
        logger.info(f"全量测试模式: {args.full_test}")
        if soc_types:
            logger.info(f"SOC 类型: {sorted(soc_types)}")
        
        dependency_data = load_dependency_json(args.dependency_json, logger)
        arch_restrictions = load_arch_restrictions(args.arch_restrictions, logger)
        whitelist_sections = load_whitelist_ops(args.whitelist_ops, logger)
        whitelist_ops = resolve_whitelist_ops(whitelist_sections, soc_types, logger)
        
        if args.full_test:
            logger.info("生成全量测试列表")
            all_op_names = set(dependency_data.keys())
            if soc_types and arch_restrictions:
                logger.info("=" * 60)
                logger.info("根据 SOC 类型过滤 arch 限制的算子")
                all_op_names = filter_ops_by_arch_and_soc(all_op_names, arch_restrictions, soc_types, logger)
            final_ops = get_ops_info_by_names(all_op_names, dependency_data, logger)
        else:
            all_ops_set: Set[str] = set()
            
            if Path(args.input).exists():
                logger.info(f"输入文件: {args.input}")
                input_ops = read_file_lines(args.input, logger)
                logger.info(f"读取到 {len(input_ops)} 个算子名称:")
                for op in input_ops:
                    logger.info(f"  算子: {op}")
                all_ops_set.update(input_ops)
            
            if Path(args.special_3rd_ops).exists():
                logger.info(f"特殊 3rd 算子文件: {args.special_3rd_ops}")
                special_ops = read_file_lines(args.special_3rd_ops, logger)
                logger.info(f"读取到 {len(special_ops)} 个特殊 3rd 算子:")
                for op in special_ops:
                    logger.info(f"  3rd算子: {op}")
                all_ops_set.update(special_ops)
            
            if not all_ops_set:
                logger.warning("没有读取到任何算子名称，输出空列表")
                final_ops = []
            else:
                logger.info(f"合并后共 {len(all_ops_set)} 个算子:")
                for op in sorted(all_ops_set):
                    logger.info(f"  合入算子: {op}")
                
                if soc_types and arch_restrictions:
                    logger.info("=" * 60)
                    logger.info("根据 SOC 类型过滤 arch 限制的算子")
                    all_ops_set = filter_ops_by_arch_and_soc(all_ops_set, arch_restrictions, soc_types, logger)
                
                final_ops = get_ops_info_by_names(all_ops_set, dependency_data, logger)
        
        final_ops = adjust_supported_products_by_arch(final_ops, arch_restrictions, soc_types, logger)
        final_ops = filter_ops_by_whitelist(final_ops, whitelist_ops, logger)
        
        write_output_json(final_ops, args.output_json, logger)
        
        if soc_types:
            soc_name = sorted(soc_types)[0]
            output_dir = str(Path(args.output_json).parent)
            soc_output_path = write_soc_json(final_ops, soc_name, output_dir, logger)
        
        print(f"成功生成测试列表: {len(final_ops)} 个算子")
        print(f"输出文件: {args.output_json}")
        if soc_types:
            print(f"SOC 适配文件: {soc_output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())