#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
从变更文件清单中解析出变更的算子列表.

仅实现变更算子解析流程的第一步: 解析变更文件 -> 算子集合.
"""

import argparse
import logging
import os

NEW_OPS_PATH = ["mc2", "attention", "ffn", "gmm", "moe", "mhc", "posembedding"]


class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map


BlackList = {}


def extract_operator_name(file_path):
    path_parts = file_path.lstrip("/").split("/")
    domain, operator_name = _get_domain_and_op(path_parts)
    if domain is None:
        return ""

    # 黑名单或 common 路径时，返回默认名称（空或 attention 特殊值）
    if operator_name in BlackList or operator_name == "common":
        return _get_default_name(domain)

    # domain 不在 NEW_OPS_PATH 时，返回默认名称（空或 attention 特殊值）
    if domain not in NEW_OPS_PATH:
        return _get_default_name(domain)

    # 其他情况返回 operator_name
    return operator_name


def _get_domain_and_op(path_parts):
    """从路径部分提取域和算子名"""
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    return None, None


def _get_default_name(domain):
    """根据域返回默认名称（目前只有 attention 特殊处理）"""
    if domain == "attention":
        return "nsa_compress_attention_infer"
    return ""


def get_operator_info_from_ci(changed_file_info_from_ci):
    """
    解析变更文件清单, 返回变更算子信息
    :param changed_file_info_from_ci: 变更文件清单文件
    :return: None or OperatorChangeInfo
    """

    def is_skippable_file(line):
        ext = os.path.splitext(line)[-1].lower()
        return ext in (".md",)

    def process_line(line, operators_set, files_map):
        """处理单行：提取算子名并更新集合和映射"""
        line = line.strip()
        if is_skippable_file(line):
            return
        operator_name = extract_operator_name(line)
        if operator_name:
            operators_set.add(operator_name)
            if operator_name not in files_map:
                files_map[operator_name] = []
            files_map[operator_name].append(line)

    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error(
            "[ERROR] change file is not exist, can not get file change info in this pull request."
        )
        return None

    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        changed_operators = set()
        operator_file_map = {}

        for line in lines:
            process_line(line, changed_operators, operator_file_map)

    return OperatorChangeInfo(
        changed_operators=sorted(changed_operators), operator_file_map=operator_file_map
    )


def get_changed_ops_list(changed_file_info_from_ci, soc=""):
    """仅执行流程第一步: 解析变更文件得到算子列表, soc 参数为后续流程预留"""
    ops_change_info = get_operator_info_from_ci(changed_file_info_from_ci)
    if not ops_change_info:
        logging.info("[INFO] not found ops change info.")
        return ""

    if not ops_change_info.changed_operators:
        logging.info("[INFO] not found any changed ops.")
        return ""

    return ";".join(ops_change_info.changed_operators)


def main():
    ps = argparse.ArgumentParser(
        description="Get changed ops from changed files", epilog="Best Regards!"
    )
    ps.add_argument(
        "changed_file",
        type=str,
        help="changed files desc file, one repo-relative path per line",
    )
    ps.add_argument(
        "-s",
        "--soc",
        type=str,
        required=False,
        default="",
        help="soc version, e.g. ascend910b/ascend950 (reserved, not used in step 1)",
    )
    args = ps.parse_args()
    print(get_changed_ops_list(args.changed_file, args.soc))


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
