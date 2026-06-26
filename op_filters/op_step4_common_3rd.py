# -*- coding: utf-8 -*-
"""
common/3rd 目录检测模块

可作为独立脚本运行，用于检测 common/3rd 目录下的变更文件。

使用方法：
    python op_filters/op_step4_common_3rd.py [--input INPUT] [--mc2 MC2_DIR] [--dependency-json JSON] [--output OUTPUT] [--special-3rd-ops FILE] [--special-common-ops FILE] [--log LOG]
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
DEFAULT_MC2_DIR = str(PROJECT_DIR / "mc2")
DEFAULT_DEPENDENCY_JSON = str(SCRIPT_DIR / "result.json")
DEFAULT_OUTPUT_FILE = str(WORK_DIR / "filtered_files.txt")
DEFAULT_SPECIAL_3RD_OPS_FILE = str(WORK_DIR / "special_3rd_ops.txt")
DEFAULT_SPECIAL_COMMON_OPS_FILE = str(WORK_DIR / "special_common_ops.txt")
DEFAULT_LOG_FILE = str(WORK_DIR / "step4.log")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from op_filters.utils import normalize_path, setup_logging, read_file_lines, load_dependency_json
else:
    from .utils import normalize_path, setup_logging, read_file_lines, load_dependency_json


# [已禁用] 3rd 子目录特殊处理映射 - common/3rd 变更现在直接触发全量测试
# 如需恢复特殊处理逻辑，取消以下注释即可
# THIRD_PARTY_OP_MAPPING: Dict[str, List[str]] = {
#     "quant_batch_matmul_v3": [
#         "all_gather_matmul_v2",
#         "allto_all_matmul",
#         "matmul_all_reduce",
#         "matmul_allto_all",
#         "matmul_reduce_scatter_v2"
#     ],
#     "template_linear_algebra": [
#         "all_gather_matmul_v2",
#         "allto_all_matmul",
#         "matmul_allto_all",
#         "matmul_reduce_scatter_v2"
#     ],
#     "batch_mat_mul_v3": [
#         "allto_all_all_gather_batch_mat_mul",
#         "batch_mat_mul_reduce_scatter_allto_all"
#     ],
#     "grouped_matmul": [
#         "allto_allv_quant_grouped_mat_mul",
#         "quant_grouped_mat_mul_allto_all"
#     ],
#     "weight_quant_batch_matmul_v2": [
#         "all_gather_matmul_v2",
#         "matmul_all_reduce"
#     ],
#     "rms_norm": [
#         "matmul_all_reduce_add_rms_norm",
#         "moe_distribute_combine_add_rms_norm"
#     ],
#     "norm_common": [
#         "matmul_all_reduce_add_rms_norm",
#         "moe_distribute_combine_add_rms_norm"
#     ],
#     "add_rms_norm": [
#         "matmul_all_reduce_add_rms_norm"
#     ]
# }
THIRD_PARTY_OP_MAPPING: Dict[str, List[str]] = {}

# [已禁用] common 文件特殊处理映射 - common/3rd 变更现在直接触发全量测试
# 如需恢复特殊处理逻辑，取消以下注释即可
# COMMON_FILE_OP_MAPPING: Dict[str, List[str]] = {
#     "op_api/mc2_aclnn_util.h": [
#         "all_gather_matmul",
#         "all_gather_matmul_v2",
#         "matmul_all_reduce",
#         "matmul_reduce_scatter_v2",
#         "matmul_reduce_scatter"
#     ],
#     "op_api/mc2_context.h": [
#         "moe_distribute_combine_v2",
#         "moe_distribute_dispatch_v2"
#     ],
#     "op_graph/matmul_all_reduce_gen_task_ops_utils.h": [
#         "inplace_matmul_all_reduce_add_rms_norm",
#         "matmul_all_reduce",
#         "matmul_all_reduce_add_rms_norm"
#     ],
#     "op_host/op_tiling/mc2_fit_based_balance_tiling.cpp": [
#         "all_gather_matmul_v2",
#         "matmul_all_reduce",
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_host/op_tiling/mc2_tiling_struct.h": [
#         "all_gather_matmul_v2",
#         "allto_allv_grouped_mat_mul",
#         "allto_allv_quant_grouped_mat_mul",
#         "batch_mat_mul_reduce_scatter_allto_all"
#     ],
#     "op_host/op_tiling/mc2_tiling_common_var.h": [
#         "allto_all_all_gather_batch_mat_mul",
#         "batch_mat_mul_reduce_scatter_allto_all",
#         "grouped_mat_mul_allto_allv"
#     ],
#     "op_host/op_tiling/mc2_calc_num_blocks.h": [
#         "allto_allv_grouped_mat_mul",
#         "grouped_mat_mul_allto_allv"
#     ],
#     "op_host/op_tiling/one_calc_two_comm_tiling.h": [
#         "allto_all_all_gather_batch_mat_mul",
#         "batch_mat_mul_reduce_scatter_allto_all"
#     ],
#     "op_kernel/mc2_moe_context.h": [
#         "moe_distribute_combine_add_rms_norm",
#         "moe_distribute_combine_v2",
#         "moe_distribute_combine_v3",
#         "moe_distribute_dispatch_v2",
#         "moe_distribute_dispatch_v3"
#     ],
#     "op_kernel/mc2_mat_mul_asw_block.h": [
#         "all_gather_matmul_v2",
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_kernel/mc2_mat_mul_asw_kernel.h": [
#         "all_gather_matmul_v2",
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_kernel/mc2_matmul_compute.h": [
#         "all_gather_matmul",
#         "all_gather_matmul_v2",
#         "matmul_reduce_scatter"
#     ],
#     "op_kernel/mc2_nd_to_nz.h": [
#         "all_gather_matmul",
#         "all_gather_matmul_v2",
#         "matmul_reduce_scatter"
#     ],
#     "op_kernel/mc2_quant_batch_matmul.h": [
#         "all_gather_matmul_v2",
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_kernel/qbmm_mix_perblock_noncontiguous.h": [
#         "all_gather_matmul_v2",
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_kernel/reduce_sum_cast_fp32.h": [
#         "matmul_all_reduce",
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_kernel/gm_ub_gm_copy.h": [
#         "matmul_all_reduce"
#     ],
#     "op_kernel/mc2_common_def.h": [
#         "matmul_reduce_scatter_v2"
#     ],
#     "op_kernel/all_gather_matmul_tiling.h": [
#         "all_gather_matmul",
#         "all_gather_matmul_v2"
#     ],
#     "op_kernel/mc2_templates/scheduler/pipeline_builder.h": [
#         "allto_all_matmul",
#         "matmul_allto_all"
#     ],
#     "op_kernel/mc2_templates/computation/math/mc2_vec_transpose.h": [
#         "allto_all_matmul"
#     ],
#     "op_kernel/mc2_templates/computation/matmul/mc2_fp_matmul.h": [
#         "allto_all_matmul"
#     ],
#     "op_kernel/mc2_templates/communication/comm_stage.h": [
#         "allto_all_matmul"
#     ],
#     "utils/context_util.h": [
#         "all_gather_matmul",
#         "allto_allv_grouped_mat_mul",
#         "allto_allv_quant_grouped_mat_mul",
#         "matmul_reduce_scatter",
#         "quant_grouped_mat_mul_allto_allv"
#     ],
#     "utils/mc2_moe_utils.h": [
#         "allto_all_all_gather_batch_mat_mul",
#         "batch_mat_mul_reduce_scatter_allto_all",
#         "grouped_mat_mul_allto_allv",
#         "quant_grouped_mat_mul_allto_allv"
#     ],
#     "utils/context_transfer.h": [
#         "inplace_matmul_all_reduce_add_rms_norm",
#         "matmul_all_reduce",
#         "matmul_all_reduce_add_rms_norm"
#     ],
#     "utils/mc2_exception_dump.h": [
#         "moe_distribute_combine_v2",
#         "moe_distribute_combine_v3",
#         "moe_distribute_dispatch_v2",
#         "moe_distribute_dispatch_v3"
#     ],
#     "utils/ops_utils.h": [
#         "all_gather_matmul",
#         "matmul_reduce_scatter",
#         "matmul_reduce_scatter_v2"
#     ],
#     "utils/matmul_util.h": [
#         "moe_distribute_combine_teardown"
#     ]
# }
COMMON_FILE_OP_MAPPING: Dict[str, List[str]] = {}


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


def get_common_relative_path(file_path: str, mc2_path: str) -> str:
    """
    获取文件在 mc2/common/ 目录下的相对路径。

    Args:
        file_path: 文件路径
        mc2_path: mc2 目录路径

    Returns:
        common 目录下的相对路径，如果不在 common 目录下则返回空字符串
    """
    normalized_file = normalize_path(file_path)
    normalized_mc2 = normalize_path(mc2_path)
    
    common_prefix_abs = normalized_mc2 + "/common/"
    common_prefix_rel = "mc2/common/"
    
    if normalized_file.startswith(common_prefix_abs):
        return normalized_file[len(common_prefix_abs):]
    
    if normalized_file.startswith(common_prefix_rel):
        return normalized_file[len(common_prefix_rel):]
    
    if "/mc2/common/" in normalized_file:
        idx = normalized_file.index("/mc2/common/") + len("/mc2/common/")
        return normalized_file[idx:]
    
    return ""


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

    当前逻辑: common 或 3rd 目录下任何文件变更都直接触发全量测试，
    不再做特殊文件/子目录到关联算子的映射处理。

    Args:
        file_paths: 变更文件列表
        mc2_path: mc2 目录路径
        dependency_data: 依赖关系数据
        logger: 日志记录器

    Returns:
        包含 common_files、3rd_files、other_files、should_full_test 的字典
    """
    common_files = []
    third_files = []
    other_files = []
    
    for file_path in file_paths:
        if is_in_common(file_path, mc2_path):
            common_files.append(file_path)
        elif is_in_3rd(file_path, mc2_path):
            third_files.append(file_path)
        else:
            other_files.append(file_path)
    
    should_full_test = False
    
    if common_files:
        logger.info("=" * 60)
        logger.info("[触发全量测试] 原因: common 目录下有文件变更")
        logger.info(f"触发路径 (共 {len(common_files)} 个):")
        for f in common_files:
            logger.info(f"  -> {f}")
        should_full_test = True
    
    if third_files:
        logger.info("=" * 60)
        logger.info("[触发全量测试] 原因: 3rd 目录下有文件变更")
        logger.info(f"触发路径 (共 {len(third_files)} 个):")
        for f in third_files:
            subdir = get_3rd_subdir(f, mc2_path)
            logger.info(f"  -> {f} (子目录: 3rd/{subdir})")
        should_full_test = True
    
    if not common_files and not third_files:
        logger.info("变更文件不在 common/3rd 目录下")
    
    # [已禁用] 以下为特殊处理逻辑，如需恢复取消注释并替换上方简单分类逻辑
    # special_common_files = []
    # special_common_ops: Set[str] = set()
    # other_common_files = []
    # special_3rd_files = []
    # special_3rd_ops: Set[str] = set()
    # other_3rd_files = []
    # other_files = []
    # for file_path in file_paths:
    #     if is_in_common(file_path, mc2_path):
    #         common_rel_path = get_common_relative_path(file_path, mc2_path)
    #         if common_rel_path in COMMON_FILE_OP_MAPPING:
    #             special_common_files.append(file_path)
    #             special_common_ops.update(COMMON_FILE_OP_MAPPING[common_rel_path])
    #         else:
    #             other_common_files.append(file_path)
    #     elif is_in_3rd(file_path, mc2_path):
    #         subdir = get_3rd_subdir(file_path, mc2_path)
    #         if subdir in THIRD_PARTY_OP_MAPPING:
    #             special_3rd_files.append(file_path)
    #             special_3rd_ops.update(THIRD_PARTY_OP_MAPPING[subdir])
    #         else:
    #             other_3rd_files.append(file_path)
    #     else:
    #         other_files.append(file_path)
    # should_full_test = False
    # if other_common_files:
    #     should_full_test = True
    # if other_3rd_files:
    #     should_full_test = True
    
    return {
        "common_files": common_files,
        "3rd_files": third_files,
        "other_files": other_files,
        # [已禁用] 特殊处理相关字段，如需恢复取消注释
        # "other_common_files": other_common_files,
        # "special_common_files": special_common_files,
        # "special_common_ops": special_common_ops,
        # "special_3rd_files": special_3rd_files,
        # "special_3rd_ops": special_3rd_ops,
        # "other_3rd_files": other_3rd_files,
        "should_full_test": should_full_test
    }


def main() -> int:
    """
    common/3rd 目录检测独立执行入口。
    """
    parser = argparse.ArgumentParser(
        description="common/3rd 目录检测模块 - 检测 common/3rd 目录下的变更文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认参数值:
  --input               {DEFAULT_INPUT_FILE}
  --mc2                 {DEFAULT_MC2_DIR}
  --dependency-json     {DEFAULT_DEPENDENCY_JSON}
  --output              {DEFAULT_OUTPUT_FILE}
  --special-3rd-ops     {DEFAULT_SPECIAL_3RD_OPS_FILE}
  --special-common-ops  {DEFAULT_SPECIAL_COMMON_OPS_FILE}
  --log                 {DEFAULT_LOG_FILE}

示例:
  # 使用默认参数
  python op_filters/op_step4_common_3rd.py
  
  # 自定义输入文件
  python op_filters/op_step4_common_3rd.py --input my_files.txt
  
  # 自定义所有参数
  python op_filters/op_step4_common_3rd.py --input input.txt --mc2 /path/to/mc2 --dependency-json deps.json --output output.txt --log step4.log
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
        "--special-3rd-ops",
        default=DEFAULT_SPECIAL_3RD_OPS_FILE,
        help=f"特殊 3rd 算子输出文件路径 (默认: {DEFAULT_SPECIAL_3RD_OPS_FILE})"
    )
    parser.add_argument(
        "--special-common-ops",
        default=DEFAULT_SPECIAL_COMMON_OPS_FILE,
        help=f"特殊 common 算子输出文件路径 (默认: {DEFAULT_SPECIAL_COMMON_OPS_FILE})"
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
        logger = logging.getLogger("common_3rd_filter")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    
    try:
        logger.info(f"输入文件: {args.input}")
        logger.info(f"mc2 目录: {args.mc2}")
        logger.info(f"依赖关系文件: {args.dependency_json}")
        logger.info(f"输出文件: {args.output}")
        logger.info(f"特殊 3rd 算子输出文件: {args.special_3rd_ops}")
        logger.info(f"特殊 common 算子输出文件: {args.special_common_ops}")
        
        input_files = read_file_lines(args.input, logger)
        dependency_data = load_dependency_json(args.dependency_json, logger)
        
        logger.info(f"读取到 {len(input_files)} 个变更文件:")
        for f in input_files:
            logger.info(f"  输入文件: {f}")
        
        result = filter_common_3rd(input_files, args.mc2, dependency_data, logger)
        
        logger.info(f"common/3rd过滤后剩余 {len(result['other_files'])} 个变更文件:")
        for f in result["other_files"]:
            logger.info(f"  算子文件: {f}")
        
        output_files = result["other_files"]
        if args.output:
            output_dir = Path(args.output).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                for file_path in output_files:
                    f.write(file_path + "\n")
            logger.info(f"结果已写入: {args.output}")
        else:
            print("非 common/3rd 目录下的文件列表:")
            for file_path in output_files:
                print(file_path)
        
        if args.special_3rd_ops and result["special_3rd_ops"]:
            output_dir = Path(args.special_3rd_ops).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.special_3rd_ops, "w", encoding="utf-8") as f:
                for op_name in sorted(result["special_3rd_ops"]):
                    f.write(op_name + "\n")
            logger.info(f"特殊 3rd 算子已写入: {args.special_3rd_ops}")
        
        if args.special_common_ops and result["special_common_ops"]:
            output_dir = Path(args.special_common_ops).parent
            if output_dir and not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.special_common_ops, "w", encoding="utf-8") as f:
                for op_name in sorted(result["special_common_ops"]):
                    f.write(op_name + "\n")
            logger.info(f"特殊 common 算子已写入: {args.special_common_ops}")
        
        print(f"\n其他 common 文件: {len(result['other_common_files'])} 个")
        print(f"特殊 common 文件: {len(result['special_common_files'])} 个")
        print(f"特殊 3rd 文件: {len(result['special_3rd_files'])} 个")
        print(f"其他 3rd 文件: {len(result['other_3rd_files'])} 个")
        print(f"其他文件: {len(result['other_files'])} 个")
        print(f"特殊 common 关联算子: {len(result['special_common_ops'])} 个")
        if result["special_common_ops"]:
            print(f"  算子列表: {sorted(result['special_common_ops'])}")
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