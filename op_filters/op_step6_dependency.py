# -*- coding: utf-8 -*-
"""
反向依赖搜索模块

可作为独立脚本运行，用于根据变更的算子查找所有依赖它的算子。

使用方法：
    python op_filters/op_step6_dependency.py [--input INPUT] [--dependency-json JSON] [--output OUTPUT] [--log LOG]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

DEFAULT_INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
DEFAULT_OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_LOG_FILE = str(WORK_DIR / "step6.log")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import setup_logging, read_file_lines, load_dependency_json
else:
    from .utils import setup_logging, read_file_lines, load_dependency_json


def build_reverse_dependency_graph(dependency_data: Dict, logger: logging.Logger) -> Dict[str, Set[str]]:
    """
    构建反向依赖图。

    反向依赖：如果算子 A 的 dependencies 中包含算子 B，则在反向依赖图中，
    B -> A（即 B 的变更会影响 A）。

    Args:
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        反向依赖图字典，key 为算子名称，value 为依赖该算子的算子集合
    """
    reverse_graph: Dict[str, Set[str]] = {}
    
    for op_name in dependency_data.keys():
        reverse_graph[op_name] = set()
    
    for op_name, op_info in dependency_data.items():
        dependencies = op_info.get("dependencies", [])
        for dep_op in dependencies:
            if dep_op in reverse_graph:
                reverse_graph[dep_op].add(op_name)
            else:
                logger.warning(
                    f"算子 {op_name} 依赖的算子 {dep_op} 不在依赖关系中，将被忽略"
                )
    
    logger.info(f"构建反向依赖图完成，共 {len(reverse_graph)} 个算子")
    return reverse_graph


def find_dependent_ops(
    changed_ops: Set[str],
    reverse_graph: Dict[str, Set[str]],
    logger: logging.Logger
) -> Set[str]:
    """
    根据反向依赖图，找到所有受影响的算子。

    只从初始变更的算子（changed_ops）传播一次，不继续传播被纳入测试的算子。
    这样可以避免被间接纳入测试的算子进一步传播影响。

    Args:
        changed_ops: 直接变更的算子集合
        reverse_graph: 反向依赖图
        logger: 日志记录器

    Returns:
        所有需要测试的算子集合（包括 changed_ops 自身及其直接依赖者）
    """
    ops_to_run = set(changed_ops)
    visited = set(changed_ops)
    
    expansion_log: List[str] = []
    
    for current_op in sorted(changed_ops):
        if current_op not in reverse_graph:
            logger.warning(f"算子 {current_op} 不在依赖关系中，跳过")
            continue
        
        dependents = reverse_graph.get(current_op, set())
        new_dependents = dependents - visited
        
        logger.info(f"算子 {current_op} 的反向依赖: {sorted(dependents)}")
        if new_dependents:
            logger.info(f"算子 {current_op} 新增反向依赖(去重后): {sorted(new_dependents)}")
        else:
            logger.info(f"算子 {current_op} 的反向依赖已全部包含在前序算子的扩展中")
        
        expansion_log.append(
            f"从 {current_op} 扩展到: {sorted(new_dependents)}"
        )
        
        for dep_op in new_dependents:
            visited.add(dep_op)
            ops_to_run.add(dep_op)
    
    for log_entry in expansion_log:
        logger.info(f"反向依赖搜索: {log_entry}")
    
    logger.info(f"反向依赖搜索完成，共 {len(ops_to_run)} 个算子需要测试")
    return ops_to_run


def main() -> int:
    """
    反向依赖搜索独立执行入口。
    """
    parser = argparse.ArgumentParser(
        description="反向依赖搜索模块 - 根据变更的算子查找所有依赖它的算子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --input           {DEFAULT_INPUT_FILE}
  --dependency-json {DEFAULT_DEPENDENCY_JSON}
  --output          {DEFAULT_OUTPUT_FILE}
  --log             {DEFAULT_LOG_FILE}

示例:
  # 使用默认参数
  python op_filters/op_step6_dependency.py
  
  # 自定义输入文件
  python op_filters/op_step6_dependency.py --input my_operators.txt
  
  # 自定义所有参数
  python op_filters/op_step6_dependency.py --input input.txt --dependency-json deps.json --output output.txt --log step6.log
        """
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"输入文件路径 (默认: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--dependency-json",
        default=DEFAULT_DEPENDENCY_JSON,
        help=f"依赖关系 JSON 文件路径 (默认: {DEFAULT_DEPENDENCY_JSON})"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help=f"输出文件路径 (默认: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_FILE,
        help=f"日志文件路径 (默认: {DEFAULT_LOG_FILE})"
    )
    
    args = parser.parse_args()
    
    if args.log:
        logger = setup_logging(args.log)
    else:
        logger = logging.getLogger("dependency_search")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {args.input}")
        logger.info(f"依赖关系文件: {args.dependency_json}")
        logger.info(f"输出文件: {args.output}")
        
        input_ops = read_file_lines(args.input, logger)
        dependency_data = load_dependency_json(args.dependency_json, logger)
        
        logger.info(f"读取到 {len(input_ops)} 个变更算子:")
        for op in input_ops:
            logger.info(f"  变更算子: {op}")
        
        changed_ops = set(input_ops)
        
        reverse_graph = build_reverse_dependency_graph(dependency_data, logger)
        
        ops_to_run = find_dependent_ops(changed_ops, reverse_graph, logger)
        
        logger.info(f"反向依赖搜索后共 {len(ops_to_run)} 个算子需要测试:")
        for op in sorted(ops_to_run):
            logger.info(f"  测试算子: {op}")
        
        if args.output:
            output_dir = Path(args.output).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                for op_name in sorted(ops_to_run):
                    f.write(op_name + "\n")
            logger.info(f"结果已写入: {args.output}")
        else:
            print("需要测试的算子列表:")
            for op_name in sorted(ops_to_run):
                print(op_name)
        
        print(f"\n输入变更算子: {len(changed_ops)} 个")
        print(f"需要测试算子: {len(ops_to_run)} 个")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())