# -*- coding: utf-8 -*-
"""
common/3rd 目录检测模块

可作为独立脚本运行，用于检测 common/3rd 目录下的变更文件。

使用方法：
    修改脚本顶部的 INPUT_FILE、MC2_DIR、DEPENDENCY_JSON、OUTPUT_FILE、LOG_FILE 变量
    然后直接运行: python op_filters/op_common_3rd_step4.py
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
WORK_DIR = Path.cwd()

INPUT_FILE = str(WORK_DIR / "filtered_files.txt")
MC2_DIR = str(PROJECT_DIR / "mc2")
DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
OPS_OUTPUT_FILE = str(WORK_DIR / "special_3rd_ops.txt")
LOG_FILE = str(WORK_DIR / "step4.log")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import normalize_path, setup_logging, read_file_lines, load_dependency_json
else:
    from .utils import normalize_path, setup_logging, read_file_lines, load_dependency_json


THIRD_PARTY_OP_MAPPING: Dict[str, List[str]] = {
    "quant_batch_matmul_v3": [
        "all_gather_matmul_v2",
        "allto_all_matmul",
        "matmul_all_reduce",
        "matmul_allto_all",
        "matmul_reduce_scatter_v2"
    ],
    "template_linear_algebra": [
        "all_gather_matmul_v2",
        "allto_all_matmul",
        "matmul_allto_all",
        "matmul_reduce_scatter_v2"
    ],
    "mat_mul_v3": [
        "allto_all_all_gather_batch_mat_mul",
        "batch_mat_mul_reduce_scatter_allto_all"
    ],
    "grouped_matmul": [
        "allto_allv_quant_grouped_mat_mul",
        "quant_grouped_mat_mul_allto_all"
    ],
    "weight_quant_batch_matmul_v2": [
        "all_gather_matmul_v2",
        "matmul_all_reduce"
    ],
    "rms_norm": [
        "matmul_all_reduce_add_rms_norm",
        "moe_distribute_combine_add_rms_norm"
    ],
    "norm_common": [
        "matmul_all_reduce_add_rms_norm",
        "moe_distribute_combine_add_rms_norm"
    ],
    "add_rms_norm": [
        "matmul_all_reduce_add_rms_norm"
    ]
}


def is_in_common(file_path: str, mc2_path: str) -> bool:
    """
    判断文件是否在 mc2/common/ 目录下。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        是否在 common 目录下
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    common_prefix_abs = normalized_mc2 + "/common"
    common_prefix_rel = "mc2/common"
    
    if normalized_file.startswith(common_prefix_abs + "/") or normalized_file == common_prefix_abs:
        return True
    if normalized_file.startswith(common_prefix_rel + "/") or normalized_file == common_prefix_rel:
        return True
    if "/mc2/common/" in normalized_file:
        return True
    
    return False


def is_in_3rd(file_path: str, mc2_path: str) -> bool:
    """
    判断文件是否在 mc2/3rd/ 目录下。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        是否在 3rd 目录下
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    third_prefix_abs = normalized_mc2 + "/3rd"
    third_prefix_rel = "mc2/3rd"
    
    if normalized_file.startswith(third_prefix_abs + "/") or normalized_file == third_prefix_abs:
        return True
    if normalized_file.startswith(third_prefix_rel + "/") or normalized_file == third_prefix_rel:
        return True
    if "/mc2/3rd/" in normalized_file:
        return True
    
    return False


def get_3rd_subdir(file_path: str, mc2_path: str) -> str:
    """
    获取文件所在的 3rd 子目录名称。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        3rd 子目录名称，如果不在 3rd 目录下则返回空字符串
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    third_prefix_abs = normalized_mc2 + "/3rd/"
    third_prefix_rel = "mc2/3rd/"
    
    if normalized_file.startswith(third_prefix_abs):
        relative_path = normalized_file[len(third_prefix_abs):]
        parts = relative_path.split("/")
        if parts and parts[0]:
            return parts[0]
    
    if normalized_file.startswith(third_prefix_rel):
        relative_path = normalized_file[len(third_prefix_rel):]
        parts = relative_path.split("/")
        if parts and parts[0]:
            return parts[0]
    
    if "/mc2/3rd/" in normalized_file:
        idx = normalized_file.index("/mc2/3rd/") + len("/mc2/3rd/")
        remaining = normalized_file[idx:]
        parts = remaining.split("/")
        if parts and parts[0]:
            return parts[0]
    
    return ""


def filter_common_3rd(file_paths: List[str], mc2_path: str, dependency_data: Dict, logger: logging.Logger) -> dict:
    """
    检测 common/3rd 目录下的变更文件。

    Args:
        file_paths: 变更文件列表
        mc2_path: mc2 目录路径
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        包含 common_files、special_3rd_files、special_3rd_ops、other_3rd_files、
        other_files、should_full_test 的字典
    """
    common_files = []
    special_3rd_files = []
    special_3rd_ops: Set[str] = set()
    other_3rd_files = []
    other_files = []
    
    for file_path in file_paths:
        if is_in_common(file_path, mc2_path):
            common_files.append(file_path)
        elif is_in_3rd(file_path, mc2_path):
            subdir = get_3rd_subdir(file_path, mc2_path)
            if subdir in THIRD_PARTY_OP_MAPPING:
                special_3rd_files.append(file_path)
                special_3rd_ops.update(THIRD_PARTY_OP_MAPPING[subdir])
                logger.info(f"特殊 3rd 子目录 {subdir} 文件: {file_path}，关联算子: {THIRD_PARTY_OP_MAPPING[subdir]}")
            else:
                other_3rd_files.append(file_path)
        else:
            other_files.append(file_path)
    
    should_full_test = False
    
    if common_files:
        logger.info("=" * 60)
        logger.info("检测到 common 目录下的变更文件，触发全量测试")
        logger.info(f"common 文件数量: {len(common_files)}")
        for f in common_files:
            logger.info(f"  common 文件: {f}")
        should_full_test = True
    
    if other_3rd_files:
        logger.info("=" * 60)
        logger.info("检测到其他 3rd 子目录下的变更文件，触发全量测试")
        logger.info(f"其他 3rd 文件数量: {len(other_3rd_files)}")
        for f in other_3rd_files:
            subdir = get_3rd_subdir(f, mc2_path)
            logger.info(f"  3rd/{subdir} 文件: {f}")
        should_full_test = True
    
    if special_3rd_files:
        logger.info("=" * 60)
        logger.info(f"检测到特殊 3rd 子目录下的变更文件，添加特定算子")
        logger.info(f"特殊 3rd 文件数量: {len(special_3rd_files)}")
        logger.info(f"关联算子: {sorted(special_3rd_ops)}")
    
    if not common_files and not other_3rd_files and not special_3rd_files:
        logger.info("变更文件不在 common/3rd 目录下")
    
    return {
        "common_files": common_files,
        "special_3rd_files": special_3rd_files,
        "special_3rd_ops": special_3rd_ops,
        "other_3rd_files": other_3rd_files,
        "other_files": other_files,
        "should_full_test": should_full_test
    }


def main() -> int:
    """
    common/3rd 目录检测独立执行入口。
    """
    if LOG_FILE:
        logger = setup_logging(LOG_FILE)
    else:
        logger = logging.getLogger("common_3rd_filter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {INPUT_FILE}")
        logger.info(f"mc2 目录: {MC2_DIR}")
        logger.info(f"依赖关系文件: {DEPENDENCY_JSON}")
        logger.info(f"输出文件: {OUTPUT_FILE}")
        logger.info(f"特殊算子输出文件: {OPS_OUTPUT_FILE}")
        
        input_files = read_file_lines(INPUT_FILE, logger)
        dependency_data = load_dependency_json(DEPENDENCY_JSON, logger)
        
        logger.info(f"读取到 {len(input_files)} 个输入文件")
        
        result = filter_common_3rd(input_files, MC2_DIR, dependency_data, logger)
        
        output_files = result["other_files"]
        if OUTPUT_FILE:
            output_dir = Path(OUTPUT_FILE).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for file_path in output_files:
                    f.write(file_path + "\n")
            logger.info(f"结果已写入: {OUTPUT_FILE}")
        else:
            print("非 common/3rd 目录下的文件列表:")
            for file_path in output_files:
                print(file_path)
        
        if OPS_OUTPUT_FILE and result["special_3rd_ops"]:
            output_dir = Path(OPS_OUTPUT_FILE).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(OPS_OUTPUT_FILE, "w", encoding="utf-8") as f:
                for op_name in sorted(result["special_3rd_ops"]):
                    f.write(op_name + "\n")
            logger.info(f"特殊 3rd 算子已写入: {OPS_OUTPUT_FILE}")
        
        print(f"\ncommon 文件: {len(result['common_files'])} 个")
        print(f"特殊 3rd 文件: {len(result['special_3rd_files'])} 个")
        print(f"其他 3rd 文件: {len(result['other_3rd_files'])} 个")
        print(f"其他文件: {len(result['other_files'])} 个")
        print(f"特殊 3rd 关联算子: {len(result['special_3rd_ops'])} 个")
        if result["special_3rd_ops"]:
            print(f"  算子列表: {sorted(result['special_3rd_ops'])}")
        print(f"是否需要全量测试: {result['should_full_test']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())