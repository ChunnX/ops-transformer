# -*- coding: utf-8 -*-
"""
白名单过滤模块

可作为独立脚本运行，用于过滤白名单中的文件路径。

使用方法：
    修改脚本顶部的 INPUT_FILE、WHITELIST_FILE、OUTPUT_FILE、LOG_FILE 变量
    然后直接运行: python op_filters/op_step1_whitelist.py
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "mc2").exists() else SCRIPT_DIR.parent / "ops-transformer"
WORK_DIR = Path.cwd()

DEFAULT_INPUT_FILE = str(SCRIPT_DIR / "changed_files.txt")
DEFAULT_WHITELIST_FILE = str(SCRIPT_DIR / "whitelist.txt")
DEFAULT_OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_LOG_FILE = str(WORK_DIR / "step1.log")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import normalize_path, setup_logging, read_file_lines
else:
    from .utils import normalize_path, setup_logging, read_file_lines


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


def main() -> int:
    """
    白名单过滤独立执行入口。
    """
    parser = argparse.ArgumentParser(
        description="白名单过滤模块 - 过滤白名单中的文件路径",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --input      {DEFAULT_INPUT_FILE}
  --whitelist  {DEFAULT_WHITELIST_FILE}
  --output     {DEFAULT_OUTPUT_FILE}
  --log        {DEFAULT_LOG_FILE}

示例:
  # 使用默认参数
  python op_filters/op_step1_whitelist.py
  
  # 自定义输入文件
  python op_filters/op_step1_whitelist.py --input my_changed_files.txt
  
  # 自定义所有参数
  python op_filters/op_step1_whitelist.py --input input.txt --whitelist whitelist.txt --output output.txt --log step1.log
        """
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=f"输入文件路径 (默认: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--whitelist",
        default=DEFAULT_WHITELIST_FILE,
        help=f"白名单文件路径 (默认: {DEFAULT_WHITELIST_FILE})"
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
        logger = logging.getLogger("whitelist_filter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {args.input}")
        logger.info(f"白名单文件: {args.whitelist}")
        logger.info(f"输出文件: {args.output}")
        
        input_files = read_file_lines(args.input, logger)
        whitelist_paths = read_file_lines(args.whitelist, logger)
        
        logger.info(f"读取到 {len(input_files)} 个变更文件:")
        for f in input_files:
            logger.info(f"  变更文件: {f}")
        logger.info(f"读取到 {len(whitelist_paths)} 个白名单路径:")
        for p in whitelist_paths:
            logger.info(f"  白名单: {p}")
        
        result = filter_whitelist(input_files, whitelist_paths, logger)
        
        logger.info(f"白名单过滤后剩余 {len(result['filtered_files'])} 个变更文件:")
        for f in result["filtered_files"]:
            logger.info(f"  剩余文件: {f}")
        
        if args.output:
            output_dir = Path(args.output).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                for file_path in result["filtered_files"]:
                    f.write(file_path + "\n")
            logger.info(f"结果已写入: {args.output}")
        else:
            print("过滤后的文件列表:")
            for file_path in result["filtered_files"]:
                print(file_path)
        
        print(f"\n白名单命中: {len(result['whitelist_hits'])} 个文件")
        print(f"过滤后剩余: {len(result['filtered_files'])} 个文件")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())