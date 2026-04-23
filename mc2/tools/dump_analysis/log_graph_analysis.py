#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
# 

# !
#  \file dump_analysis.py
#  \brief

import os
import sys
import csv
import ast
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )


def get_log_path(log_path_func):
    log_path_list_func = []
    for dir_path, _, filenames in os.walk(log_path_func):
        for filename in filenames:
            if filename.lower().endswith(".log"):
                full_path = os.path.abspath(os.path.join(dir_path, filename))
                log_path_list_func.append(full_path)
    return log_path_list_func


def get_graph_path(graph_path_func):
    graph_path_dict_func = {}
    pattern = re.compile('graph_.*_device(\d+)\.json$', re.I)
    for file in Path(graph_path_func).rglob("graph_*.json"):
        match = pattern.match(file.name)
        if match:
            num = match.group(1)
            graph_path_dict_func[num] = str(file.absolute())
    return graph_path_dict_func


def get_graph_info(graph_path_dict_func):
    graph_info_func = {}
    pattern = re.compile(r'"name"\s*:\s*["\']MoeDistribute(.*?)["\']')
    
    # 遍历每个文件路径
    for idx, file_path in graph_path_dict_func.items():
        current_matches = []
        # 安全读取文件
        try:
            file = Path(file_path)
            if not file.exists():
                logging.warning("%s文件不存在", file_path)
                continue
                
            # 读取文本（不用json.load，避免JSON结构复杂/嵌套深导致漏匹配）
            content = file.read_text(encoding='utf-8')
            
            # 查找所有匹配项
            matches = pattern.findall(content)
            current_matches = matches
            
        except Exception as e:
            logging.error("处理文件%s出错: %s", file_path, str(e))
        
        # 存入结果
        graph_info_func[idx] = current_matches
    
    return graph_info_func


def compare_graph_info(graph_info_func):
    card_ids = list(graph_info_func.keys())
    for i in range(len(card_ids)):
        card_a = card_ids[i]
        list_a = graph_info_func[card_a]
        
        for j in range(i + 1, len(card_ids)):
            card_b = card_ids[j]
            list_b = graph_info_func[card_b]
            
            logging.info("开始对比 卡%s <-> 卡%s 的算子执行顺序、次数",i ,j)
            
            if list_a == list_b:
                logging.info("卡%s <-> 卡%s完全一致(顺序+内容+次数相同)",i ,j)
            elif sorted(list_a) == sorted(list_b):
                logging.error("卡%s <-> 卡%s的算子执行顺序不同,但执行次数相同",i ,j)
            else:
                logging.error("卡%s <-> 卡%s的算子执行次数不同",i ,j)


def get_plog_error(log_path_list_func):
    result = {
        "error1": [],
        "error2": [],
        "error3": [],
        "error4": []
    }

    # 遍历所有日志文件
    for log_path in log_path_list_func:
        try:
            # 读取日志（通用编码，避免乱码）
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            # 逐行匹配
            for line in lines:
                line = line.strip()  # 去掉换行/空格
                if not line:
                    continue
                # 必须包含 [ERROR] 才继续判断
                if "[ERROR]" not in line:
                    continue

                # 按规则分类
                if "0x800000" in line:
                    result["error1"].append(line)
                if "GE" in line:
                    result["error2"].append(line)
                if "Aicore kernel execute failed" in line:
                    result["error3"].append(line)
                if "Error happened, origin_op_name" in line:
                    result["error4"].append(line)

        except Exception as e:
            logging.error("读取文件失败 %s,原因：%s", log_path, str(e))

    return result


def print_error_result(result):
    # error3 正则
    patterns_error3 = {
        "device_id": re.compile(r'device_id=(\w+)'),
        "stream_id": re.compile(r'stream_id=(\w+)'),
        "report_stream_id": re.compile(r'report_stream_id=(\w+)'),
        "task_id": re.compile(r'task_id=(\w+)'),
        "kernel_name": re.compile(r'fault kernel_name=(\w+)')
    }

    # error4 正则
    patterns_error4 = {
        "origin_op_name": re.compile(r'origin_op_name\s+\[([^\]]+)\]'),
        "task_id": re.compile(r'task_id\s*(\d+)'),
        "stream_id": re.compile(r'stream_id\s*(\d+)')
    }

    # error1 处理
    error1_count = len(result.get("error1", []))
    if error1_count > 0:
        logging.info("检测到 (总线错误:0x800000错误,请进一步查看1520日志):%d 个", error1_count)

    # error2 处理
    error2_count = len(result.get("error2", []))
    if error2_count > 0:
        logging.info(
            "检测到 GE错误:%d个,请前往该网址,搜索GE Errors查询对应报错信息及解决方法: "
            "https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/index/index.html",
            error2_count
        )

    # error3 处理
    error3_list = result.get("error3", [])
    error3_count = len(error3_list)
    error3_dict_func = {}

    if error3_count > 0:
        logging.info("检测到 Aicore kernel execute failed类报错:%d个,下列为前10条报错的具体信息", error3_count)
        
        for idx, line in enumerate(error3_list[:10], 1):
            # 批量提取字段
            fields = {}
            for key, pattern in patterns_error3.items():
                match = pattern.search(line)
                fields[key] = match.group(1) if match else "NA"

            # 打印日志
            logging.error(
                "device_id=%s,stream_id=%s,report_id=%s,task_id=%s,报错算子:%s",
                fields["device_id"],
                fields["stream_id"],
                fields["report_stream_id"],
                fields["task_id"],
                fields["kernel_name"]
            )

            # 统计算子信息
            dev_id = fields["device_id"]
            if dev_id not in error3_dict_func:
                error3_dict_func[dev_id] = []
            error3_dict_func[dev_id].append(fields["kernel_name"])

    # error4 处理
    error4_list = result.get("error4", [])
    error4_count = len(error4_list)

    if error4_count > 0:
        logging.info("检测到 Error happened, origin_op_name类报错:%d个,下列为前10条报错的具体信息", error4_count)
        
        for idx, line in enumerate(error4_list[:10], 1):
            # 批量提取字段
            fields = {}
            for key, pattern in patterns_error4.items():
                match = pattern.search(line)
                fields[key] = match.group(1) if match else "NA"

            # 打印日志
            logging.error(
                "stream_id=%s,task_id=%s,报错算子及执行次数:%s",
                fields["stream_id"],
                fields["task_id"],
                fields["origin_op_name"]
            )
    return error3_dict_func


def check_error3_kernel(error3_dict_func):
    # 允许的算子白名单
    allowed_kernels = {"dispatch", "combine"}
    logging.info("开始检测是否所有卡都在dispatch or combine 算子上执行失败")
    has_error = False
    for device_id, kernel_list in error3_dict_func.items():
        for kernel in kernel_list:
            if kernel == "NA":
                continue
            # 只要 kernel 包含 白名单任意一个关键字，就合法
            if not any(key in kernel for key in allowed_kernels):
                logging.error("卡%s没有在 dispatch 或 combine 算子上执行失败，在了%s算子上执行失败", device_id, kernel)
                has_error = True  
    if not has_error:
        logging.info("所有卡均在 dispatch or combine 算子上执行失败")


def dict_to_csv(error_dict_func):
    name_mapping = {
        "error1": "总线错误",
        "error2": "GE Errors",
        "error3": "Aicore kernel execute failed类报错",
        "error4": "Error happened, origin_op_name类报错"
    }

    # 过滤空列表 + 映射名称
    filtered_rows = []
    for key, err_list in error_dict_func.items():
        if err_list:  # 只保留非空列表
            display_name = name_mapping.get(key, key)  # 找不到映射就用原名
            # 每条错误单独一行，两列：错误类型 + 错误信息
            for msg in err_list:
                filtered_rows.append([display_name, msg])

    # 全部为空：打印提示
    if not filtered_rows:
        logging.info("所有错误类型均为空,不写入errors_plog.csv")
        return

    # 写入CSV
    with open("errors_plog.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["错误类型", "错误信息"])
        writer.writerows(filtered_rows)
    logging.info("所有检测到的plog日志中的错误类型均写入errors_plog.csv")


if __name__ == "__main__":
    log_path = sys.argv[1]
    graph_path = sys.argv[2]
    log_path_list = []
    graph_path_dict = {}
    graph_info = {}
    error3_dict = {}
    if log_path != "NA":
        logging.info("开始进行错误日志分析")
        log_path_list = get_log_path(log_path)
        error_dict = get_plog_error(log_path_list)
        error3_dict = print_error_result(error_dict)
        check_error3_kernel(error3_dict)
        dict_to_csv(error_dict)
        logging.info("错误日志分析完成")
    else:
        logging.warning("未输入plog路径,不进行错误日志分析")
    if graph_path != "NA":
        logging.info("开始进行graph图文件分析")
        graph_path_dict = get_graph_path(graph_path)
        for num, path in graph_path_dict.items():
            logging.info("卡%s graph文件:%s",num, path)
        graph_info = get_graph_info(graph_path_dict)
        compare_graph_info(graph_info)
        logging.info("graph图文件分析完成")
    else:
        logging.warning("未输入graph图文件路径,不进行graph图文件分析")