# -*- coding: utf-8 -*-
"""
非 mc2 目录检测模块

可作为独立脚本运行，用于检测非 mc2 目录下的变更文件。

使用方法：
    python op_filters/op_step3_non_mc2.py [--input INPUT] [--mc2 MC2_DIR] [--dependency-json JSON] [--output OUTPUT] [--log LOG]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

DEFAULT_INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_MC2_DIR = str(PROJECT_DIR / "mc2")
DEFAULT_DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
DEFAULT_OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_LOG_FILE = str(WORK_DIR / "step3.log")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import normalize_path, setup_logging, read_file_lines, load_dependency_json
else:
    from .utils import normalize_path, setup_logging, read_file_lines, load_dependency_json


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


def main() -> int:
    """
    非 mc2 目录检测独立执行入口。
    """
    parser = argparse.ArgumentParser(
        description="非 mc2 目录检测模块 - 检测非 mc2 目录下的变更文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --input           {DEFAULT_INPUT_FILE}
  --mc2             {DEFAULT_MC2_DIR}
  --dependency-json {DEFAULT_DEPENDENCY_JSON}
  --output          {DEFAULT_OUTPUT_FILE}
  --log             {DEFAULT_LOG_FILE}

示例:
  # 使用默认参数
  python op_filters/op_step3_non_mc2.py
  
  # 自定义输入文件
  python op_filters/op_step3_non_mc2.py --input my_files.txt
  
  # 自定义所有参数
  python op_filters/op_step3_non_mc2.py --input input.txt --mc2 /path/to/mc2 --dependency-json deps.json --output output.txt --log step3.log
        """
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"输入文件路径 (默认: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--mc2",
        default=DEFAULT_MC2_DIR,
        help=f"mc2 目录路径 (默认: {DEFAULT_MC2_DIR})"
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
        logger = logging.getLogger("non_mc2_filter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {args.input}")
        logger.info(f"mc2 目录: {args.mc2}")
        logger.info(f"依赖关系文件: {args.dependency_json}")
        logger.info(f"输出文件: {args.output}")
        
        input_files = read_file_lines(args.input, logger)
        dependency_data = load_dependency_json(args.dependency_json, logger)
        
        logger.info(f"读取到 {len(input_files)} 个变更文件:")
        for f in input_files:
            logger.info(f"  输入文件: {f}")
        
        result = filter_non_mc2(input_files, args.mc2, dependency_data, logger)
        
        logger.info(f"非mc2过滤后剩余 {len(result['mc2_files'])} 个变更文件:")
        for f in result["mc2_files"]:
            logger.info(f"  mc2文件: {f}")
        
        output_files = result["mc2_files"]
        if args.output:
            output_dir = Path(args.output).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                for file_path in output_files:
                    f.write(file_path + "\n")
            logger.info(f"结果已写入: {args.output}")
        else:
            print("mc2 目录下的文件列表:")
            for file_path in output_files:
                print(file_path)
        
        print(f"\n非 mc2 文件: {len(result['non_mc2_files'])} 个")
        print(f"mc2 文件: {len(result['mc2_files'])} 个")
        print(f"是否需要全量测试: {result['should_full_test']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())