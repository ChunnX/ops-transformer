#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file dump_moe_ep_analysis.py
#  \brief
#

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
from openpyxl import Workbook

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# Metadata 区固定大�?64KiB, 有效结构�?<=512B, 剩余保留
METADATA_SIZE = 64 * 1024
# Metadata �?10 �?uint32 字段�? 按结构体偏移顺序排列
METADATA_FIELD_NAMES = [
    "layoutVersion",
    "nmt",
    "topK",
    "hidden",
    "epWorldSize",
    "localExpertNum",
    "aivNum",
    "networkMode",
    "serverNum",
    "rankNumPerServer",
]
# Metadata �?7 个固�?Region 的名�?
REGION_COUNT = 7
REGION_NAMES = [
    "PER_CORE_DIAG",
    "COUNT_NOTIFY",
    "EXPERT_COUNT",
    "DISPATCH_SLOT_STATE",
    "COMBINE_TOKEN_STATE",
    "COMBINE_STATE_TAIL",
    "HYBRID_SCALEOUT_STATUS",
]
# Region 名称�?dump 文件后缀的映�? 每个 Region 单独输出一�?.bin 文件
REGION_FILE_SUFFIX = {
    "METADATA": "metadata.bin",
    "PER_CORE_DIAG": "per_core_diag.bin",
    "COUNT_NOTIFY": "count_notify.bin",
    "EXPERT_COUNT": "expert_count.bin",
    "DISPATCH_SLOT_STATE": "dispatch_slot_state.bin",
    "COMBINE_TOKEN_STATE": "combine_token_state.bin",
    "COMBINE_STATE_TAIL": "combine_state_tail.bin",
    "HYBRID_SCALEOUT_STATUS": "hybrid_scaleout_status.bin",
}


# 单卡分析结果, 封装 analyze_single_card 的返回�?
@dataclass
class CardAnalysisResult:
    bs: int = 0
    core_count: int = 0
    per_core_data: list = field(default_factory=list)
    count_notify_data: list = field(default_factory=list)
    expert_count_data: list = field(default_factory=list)
    combine_token_state_data: list = field(default_factory=list)


# 全卡分析结果, 封装 save_to_excel 的参�?
@dataclass
class AllAnalysisResult:
    all_metadata: dict = field(default_factory=dict)
    all_per_core: list = field(default_factory=list)
    all_count_notify: list = field(default_factory=list)
    all_expert_count: list = field(default_factory=list)
    all_combine_token_state: list = field(default_factory=list)
    card_count: int = 0


# 在卡目录中按后缀查找 Region 对应�?.bin 文件, 读取全部数据返回 uint8 数组
def read_region_file(target_path: str, suffix: str):
    for filename in os.listdir(target_path):
        if filename.endswith(suffix):
            file_path = os.path.join(target_path, filename)
            with open(file_path, "rb") as f:
                return np.frombuffer(f.read(), dtype=np.uint8)
    logging.warning("未找�?%s 文件", suffix)
    return np.array(0, dtype=np.uint8)


# �?.metadata.bin 文件解析 Metadata
# 有效结构体布局: 10 �?uint32 字段 + 7 �?MoeEpDumpRegion(offset, size), 每个 region �?2 �?uint32
def parse_metadata(target_path: str):
    metadata_dict = {name: 0 for name in METADATA_FIELD_NAMES}
    for name in REGION_NAMES:
        metadata_dict[name] = [0, 0]
    raw = read_region_file(target_path, REGION_FILE_SUFFIX["METADATA"])
    if raw.shape[0] < 512:
        logging.warning("metadata.bin 文件大小不足 512 字节, 无法解析 Metadata")
        return metadata_dict, False
    metadata_raw = raw[:512].view(np.uint32)
    # 解析 10 �?uint32 字段
    for i, name in enumerate(METADATA_FIELD_NAMES):
        metadata_dict[name] = int(metadata_raw[i])
    # 解析 7 �?Region �?offset �?size
    for i, name in enumerate(REGION_NAMES):
        offset = int(metadata_raw[10 + i * 2])
        size = int(metadata_raw[10 + i * 2 + 1])
        metadata_dict[name] = [offset, size]
    return metadata_dict, True


# 遍历所有卡目录, 收集每张卡的 Metadata, 返回全局字典(每字段值为按卡序排列的数组)
def collect_all_metadata(card_dirs: list):
    all_metadata = {name: [] for name in METADATA_FIELD_NAMES + REGION_NAMES}
    for card_num, card_path in enumerate(card_dirs):
        if not os.path.isdir(card_path):
            logging.error("�?d 路径不存�? %s, 跳过该卡", card_num, card_path)
            continue
        metadata_dict, ok = parse_metadata(card_path)
        if not ok:
            logging.error("�?d Metadata 解析失败, 使用默认值填�?, card_num)
            for name in METADATA_FIELD_NAMES:
                all_metadata[name].append(0)
            for name in REGION_NAMES:
                all_metadata[name].append([0, 0])
            continue
        for name in METADATA_FIELD_NAMES + REGION_NAMES:
            all_metadata[name].append(metadata_dict[name])
    return all_metadata


# 打印单卡 Metadata
def print_card_metadata(card_num: int, card_metadata: dict):
    for name in METADATA_FIELD_NAMES + REGION_NAMES:
        logging.info("�?d %s: %s", card_num, name, card_metadata.get(name))


# �?exception_info 解析后的 input.0.bin 中读�?context 数据(int32)
# context �?MoeEpDispatch/Combine 的第一个输�? 包含 HCCL 通信上下文信�?
def get_context(target_path: str):
    target_file_num = "input.0.bin"
    for filename in os.listdir(os.path.join(target_path)):
        if target_file_num in filename:
            file_path = os.path.join(target_path, filename)
            context = np.fromfile(file_path, dtype=np.int32)
            logging.info("context dtype: %s, size: %d", context.dtype, context.shape[0])
            logging.info("context 数据:\n%s", context)
            return context
    logging.warning("未找�?%s 文件, 无法获取 context 数据", target_file_num)
    return None


# �?input.2.bin 读取 topk_idx(int32), �?Metadata 中的 topK reshape �?(bs, k)
# 返回 reshape 后的数组和计算出�?bs
def get_topk_idx(target_path: str, topk: int, global_expert_num: int):
    target_file_num = "input.2.bin"
    for filename in os.listdir(os.path.join(target_path)):
        if target_file_num in filename:
            file_path = os.path.join(target_path, filename)
            topk_idx = np.fromfile(file_path, dtype=np.int32)
            bs = topk_idx.shape[0] // topk if topk > 0 else 0
            if topk > 0 and bs * topk != topk_idx.shape[0]:
                logging.warning(
                    "topk_idx 数据大小 %d 不能�?topK %d 整除, 截断�?%d",
                    topk_idx.shape[0],
                    topk,
                    bs,
                )
                topk_idx = topk_idx[: bs * topk]
            return topk_idx.reshape(bs, topk), bs
    logging.warning("未找�?%s 文件, 无法获取 topk_idx 数据", target_file_num)
    return None, 0


# 校验 topk_idx 值范�? 每个值应 >= 0 �?< global_expert_num, 异常项逐个打印
def check_topk_idx(topk_idx_reshape, global_expert_num: int):
    mask = (topk_idx_reshape < 0) | (topk_idx_reshape >= global_expert_num)
    if np.any(mask):
        row_func, col_func = np.where(mask)
        logging.error(
            "检测到topk_idx中有%d个异常输�? 值应 >= 0 �?< 全局专家�?d",
            len(row_func),
            global_expert_num,
        )
        for r, c in zip(row_func, col_func):
            logging.error(
                "topk_idx下标[%d,%d]的�?%d超范�?, r, c, topk_idx_reshape[r][c]
            )
    else:
        logging.info("未检测到topk_idx中有超范围的异常输入")


# 分析 Per-core 诊断�? �?.per_core_diag.bin 读取
# 布局: 100 个核 × 512B/�? 每核前两�?uint32 �?op_count(执行次数) �?run_pos(执行位置)
# 核数判断: 当某核的 op_count �?run_pos 全为 0 �? 认为该核未使�? 之前的核数即使用核数
def analyze_per_core(target_path: str):
    raw_data = read_region_file(target_path, REGION_FILE_SUFFIX["PER_CORE_DIAG"])
    if raw_data.shape[0] == 0:
        logging.warning("Per-core 诊断区大小为0")
        return 0, []
    max_cores = raw_data.shape[0] // 512
    op_counts = []
    run_poses = []
    core_count = max_cores
    for i in range(max_cores):
        block = raw_data[i * 512 : (i + 1) * 512].view(np.uint32)
        op_count = int(block[0])
        run_pos = int(block[1])
        if op_count == 0 and run_pos == 0:
            core_count = i
            break
        op_counts.append(op_count)
        run_poses.append(run_pos)
    return core_count, list(zip(op_counts, run_poses))


# 打印 Per-core 分析结果
def print_per_core(core_count: int, per_core_data: list):
    logging.info("使用核数: %d", core_count)
    for i in range(core_count):
        op_count, run_pos = per_core_data[i]
        logging.info("�?d: op_count=%d, 执行位置=%d", i, op_count, run_pos)


# 分析 Count Notify �? �?.count_notify.bin 读取
# 布局: epWorldSize �?512B �? 每块第一�?uint32 为该�?rank 发送的 token 计数
def analyze_count_notify(target_path: str, ep_world_size: int):
    raw_data = read_region_file(target_path, REGION_FILE_SUFFIX["COUNT_NOTIFY"])
    if raw_data.shape[0] == 0:
        logging.warning("Count Notify 区大小为0")
        return []
    counts = []
    for i in range(ep_world_size):
        if (i + 1) * 512 > raw_data.shape[0]:
            logging.warning("Count Notify 数据不足 %d �?512B �? 实际只有 %d �?, ep_world_size, i)
            break
        block = raw_data[i * 512 : (i + 1) * 512].view(np.uint32)
        counts.append(int(block[0]))
    logging.info("Count Notify 数据: %s", counts)
    return counts


# 分析 Expert Count �? �?.expert_count.bin 读取
# 布局: epWorldSize �?512B �? 每块包含 localExpertNum �?uint32, 为该 rank 每个专家收到�?token 计数
def analyze_expert_count(target_path: str, ep_world_size: int, local_expert_num: int):
    raw_data = read_region_file(target_path, REGION_FILE_SUFFIX["EXPERT_COUNT"])
    if raw_data.shape[0] == 0:
        logging.warning("Expert Count 区大小为0")
        return []
    expert_counts = []
    for i in range(ep_world_size):
        if (i + 1) * 512 > raw_data.shape[0]:
            logging.warning("Expert Count 数据不足 %d �?512B �? 实际只有 %d �?, ep_world_size, i)
            break
        block = raw_data[i * 512 : (i + 1) * 512].view(np.uint32)
        per_rank = [int(block[j]) for j in range(local_expert_num)]
        expert_counts.append(per_rank)
    logging.info("Expert Count 数据: %s", expert_counts)
    return expert_counts


# 分析 Combine Token State �? �?.combine_token_state.bin 读取
# 布局: N*K �?512B �?N=nmt, K=topK), 每块第一�?uint32 为该 (token, topk) 的回传完成状�?
def analyze_combine_token_state(target_path: str, nmt: int, topk: int):
    raw_data = read_region_file(target_path, REGION_FILE_SUFFIX["COMBINE_TOKEN_STATE"])
    if raw_data.shape[0] == 0:
        logging.warning("Combine Token State 区大小为0")
        return []
    total = nmt * topk
    states = []
    for i in range(total):
        if (i + 1) * 512 > raw_data.shape[0]:
            logging.warning("Combine Token State 数据不足 %d �?512B �? 实际只有 %d �?, total, i)
            break
        block = raw_data[i * 512 : (i + 1) * 512].view(np.uint32)
        states.append(int(block[0]))
    logging.info("Combine Token State 数据: %s", states)
    return states


# 单卡分析主函�? 依次获取 context、topk_idx, 解析�?Region .bin 文件
# 返回 CardAnalysisResult 封装的单卡分析结�?
def analyze_single_card(
    card_path: str, card_num: int, card_metadata: dict, global_expert_num: int
):
    logging.info("开始分析卡%d数据, dump数据路径: %s", card_num, card_path)
    print_card_metadata(card_num, card_metadata)
    logging.info("全局专家�? %d", global_expert_num)

    get_context(card_path)

    topk = card_metadata.get("topK", 0) if card_metadata else 0
    topk_idx, bs = get_topk_idx(card_path, topk, global_expert_num)
    if topk_idx is not None:
        logging.info("topk_idx bs: %d, k: %d", bs, topk)
        check_topk_idx(topk_idx, global_expert_num)

    ep_world_size = card_metadata.get("epWorldSize", 0) if card_metadata else 0
    local_expert_num = card_metadata.get("localExpertNum", 0) if card_metadata else 0
    nmt = card_metadata.get("nmt", 0) if card_metadata else 0

    core_count, per_core_data = analyze_per_core(card_path)
    print_per_core(core_count, per_core_data)
    count_notify_data = analyze_count_notify(card_path, ep_world_size)
    expert_count_data = analyze_expert_count(card_path, ep_world_size, local_expert_num)
    combine_token_state_data = analyze_combine_token_state(card_path, nmt, topk)

    return CardAnalysisResult(
        bs=bs,
        core_count=core_count,
        per_core_data=per_core_data,
        count_notify_data=count_notify_data,
        expert_count_data=expert_count_data,
        combine_token_state_data=combine_token_state_data,
    )


# �?metadata sheet: 每卡一�? 包含 10 �?uint32 字段 + 7 �?Region[offset,size] + bs + core_count
def write_metadata_sheet(wb, all_metadata: dict, card_count: int):
    ws = wb.active
    ws.title = "metadata"
    all_field_names = METADATA_FIELD_NAMES + REGION_NAMES + ["bs", "core_count"]
    ws.append(["card_num"] + all_field_names)
    for card_num in range(card_count):
        row = [card_num]
        for name in all_field_names:
            row.append(
                all_metadata[name][card_num]
                if card_num < len(all_metadata[name])
                else None
            )
        ws.append(row)


# �?per_card_per_core sheet: 每核一�? 包含 card_num, core_count, core_id, op_count, run_pos
def write_per_core_sheet(wb, all_per_core: list):
    ws = wb.create_sheet("per_card_per_core")
    ws.append(["card_num", "core_count", "core_id", "op_count", "run_pos"])
    for card_num, per_core_data in enumerate(all_per_core):
        core_count = len(per_core_data)
        for core_id, (op_count, run_pos) in enumerate(per_core_data):
            ws.append([card_num, core_count, core_id, op_count, run_pos])


# �?count_notify sheet: 每卡�?rank 一�? �?epWorldSize 便于核对行数
def write_count_notify_sheet(wb, all_count_notify: list, all_metadata: dict):
    ws = wb.create_sheet("count_notify")
    ws.append(["card_num", "epWorldSize", "rank_id", "count"])
    for card_num, count_notify_data in enumerate(all_count_notify):
        ep_ws = (
            all_metadata["epWorldSize"][card_num]
            if card_num < len(all_metadata["epWorldSize"])
            else 0
        )
        for rank_id, count in enumerate(count_notify_data):
            ws.append([card_num, ep_ws, rank_id, count])


# �?expert_count sheet: 每卡�?rank �?expert 一�? �?localExpertNum 便于核对行数
def write_expert_count_sheet(wb, all_expert_count: list, all_metadata: dict):
    ws = wb.create_sheet("expert_count")
    ws.append(["card_num", "localExpertNum", "rank_id", "expert_id", "count"])
    for card_num, expert_count_data in enumerate(all_expert_count):
        local_expert_num = (
            all_metadata["localExpertNum"][card_num]
            if card_num < len(all_metadata["localExpertNum"])
            else 0
        )
        for rank_id, per_rank in enumerate(expert_count_data):
            for expert_id, count in enumerate(per_rank):
                ws.append([card_num, local_expert_num, rank_id, expert_id, count])


# �?combine_token_state sheet: 每卡�?(token,topk) 一�? idx �?0~N*K-1 的序�?
def write_combine_token_state_sheet(wb, all_combine_token_state: list):
    ws = wb.create_sheet("combine_token_state")
    ws.append(["card_num", "idx", "token_state"])
    for card_num, combine_token_state_data in enumerate(all_combine_token_state):
        for idx, state in enumerate(combine_token_state_data):
            ws.append([card_num, idx, state])


# 将所有分析结果写�?Excel: 5 �?sheet(metadata/per_card_per_core/count_notify/expert_count/combine_token_state)
def save_to_excel(result: AllAnalysisResult):
    wb = Workbook()
    write_metadata_sheet(wb, result.all_metadata, result.card_count)
    write_per_core_sheet(wb, result.all_per_core)
    write_count_notify_sheet(wb, result.all_count_notify, result.all_metadata)
    write_expert_count_sheet(wb, result.all_expert_count, result.all_metadata)
    write_combine_token_state_sheet(wb, result.all_combine_token_state)
    wb.save("moe_ep_dump_analysis_result.xlsx")
    logging.info("全卡分析结果已归档至 moe_ep_dump_analysis_result.xlsx")


# ===== 主流�?=====
if __name__ == "__main__":
    # 1. 解析参数, 构建 card_dirs 列表
    if len(sys.argv) < 4:
        logging.error(
            "用法: python %s <dump_data_path> <file_num> <is_multi>", sys.argv[0]
        )
        sys.exit(1)

    target_dir = sys.argv[1]
    file_num = int(sys.argv[2])
    is_multi = int(sys.argv[3])
    if not os.path.isdir(target_dir):
        logging.error("路径不存�? %s", target_dir)
        sys.exit(1)

    logging.info("开始解�?dump 数据, 总卡�?%d, is_multi:%d, dump数据路径: %s", file_num, is_multi, target_dir)

    if is_multi == 1:
        card_dirs = [os.path.join(target_dir, str(i)) for i in range(file_num)]
    else:
        card_dirs = [target_dir]

    # 2. 收集所有卡 Metadata, 计算全局专家�?
    all_metadata = collect_all_metadata(card_dirs)
    logging.info("所有卡 Metadata 收集完成")

    global_expert_num = sum(all_metadata["localExpertNum"])
    logging.info("全局专家�? %d", global_expert_num)

    # 3. 逐卡执行单卡分析, 收集结果
    all_bs = []
    all_core_count = []
    all_per_core = []
    all_count_notify = []
    all_expert_count = []
    all_combine_token_state = []

    for card_num, card_path in enumerate(card_dirs):
        logging.info("--------------------------------------------")
        if not os.path.isdir(card_path):
            logging.error("�?d 路径不存�? %s, 跳过该卡", card_num, card_path)
            all_bs.append(0)
            all_core_count.append(0)
            all_per_core.append([])
            all_count_notify.append([])
            all_expert_count.append([])
            all_combine_token_state.append([])
            continue
        card_metadata = {
            name: all_metadata[name][card_num]
            if card_num < len(all_metadata[name])
            else None
            for name in METADATA_FIELD_NAMES + REGION_NAMES
        }
        card_result = analyze_single_card(
            card_path, card_num, card_metadata, global_expert_num
        )
        all_bs.append(card_result.bs)
        all_core_count.append(card_result.core_count)
        all_per_core.append(card_result.per_core_data)
        all_count_notify.append(card_result.count_notify_data)
        all_expert_count.append(card_result.expert_count_data)
        all_combine_token_state.append(card_result.combine_token_state_data)

    all_metadata["bs"] = all_bs
    all_metadata["core_count"] = all_core_count

    # 4. 写入 Excel
    all_result = AllAnalysisResult(
        all_metadata=all_metadata,
        all_per_core=all_per_core,
        all_count_notify=all_count_notify,
        all_expert_count=all_expert_count,
        all_combine_token_state=all_combine_token_state,
        card_count=len(card_dirs),
    )
    save_to_excel(all_result)

    logging.info("--------------------------------------------")
    logging.info("所有卡数据解析完成, 多卡分析待实�?)
