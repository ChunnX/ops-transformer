# -*- coding: utf-8 -*-
"""
非 mc2 目录检测模块

可作为独立脚本运行，用于检测非 mc2 目录下的变更文件。

使用方法：
    修改脚本顶部的 INPUT_FILE、MC2_DIR、DEPENDENCY_JSON、OUTPUT_FILE、LOG_FILE 变量
    然后直接运行: python op_filters/non_mc2_filter.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
WORK_DIR = Path.cwd()

INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
MC2_DIR = str(PROJECT_DIR / "mc2")
DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
LOG_FILE = str(WORK_DIR / "step3.log")

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
    if LOG_FILE:
        logger = setup_logging(LOG_FILE)
    else:
        logger = logging.getLogger("non_mc2_filter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {INPUT_FILE}")
        logger.info(f"mc2 目录: {MC2_DIR}")
        logger.info(f"依赖关系文件: {DEPENDENCY_JSON}")
        logger.info(f"输出文件: {OUTPUT_FILE}")
        
        input_files = read_file_lines(INPUT_FILE, logger)
        dependency_data = load_dependency_json(DEPENDENCY_JSON, logger)
        
        logger.info(f"读取到 {len(input_files)} 个输入文件")
        
        result = filter_non_mc2(input_files, MC2_DIR, dependency_data, logger)
        
        output_files = result["mc2_files"]
        if OUTPUT_FILE:
            output_dir = Path(OUTPUT_FILE).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for file_path in output_files:
                    f.write(file_path + "\n")
            logger.info(f"结果已写入: {OUTPUT_FILE}")
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