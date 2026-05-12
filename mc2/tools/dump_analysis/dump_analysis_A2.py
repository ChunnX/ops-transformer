#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file dump_analysis_A2.py
#  \brief

import os
import ast
import csv
import logging
from typing import List
from collections import Counter
import pandas as pd
import numpy as np

SOC_VERSION_910B = "910B"
logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

UINT32_PER_BLOCK = 8
SERVER_RANK_NUM = 8

PARENT_PATH = "result_A2/"
DUMP_FILE_PREFIX = "win_dump_input_serverid_"
DUMP_FILE_NAME = DUMP_FILE_PREFIX + "{}.csv"
ERROR_INFO_FILE_NAME = "errorinfo_{}.csv"


#获取各核的islayer值
def get_islayer_ep(arr_func: np.ndarray, core_num_func: int, d_c: str):
    per_core = 128
    islayer = []
    for i in range(core_num_func):
        islayer.append(arr_func[11 + (i * per_core)])
    logging.info("[A2] 1.1 %s各核islayer:%s\n", d_c, islayer)
    islayer_count = Counter(islayer)
    max_islayer = max(islayer_count, key=islayer_count.get)
    diff_islayer = [idx for idx, val in enumerate(islayer) if val != max_islayer]
    if diff_islayer != []:
        logging.error("[A2] 1.1 %s有如下下标的核的islayer与其他核不相等%s,共%d个核\n",
                      d_c, diff_islayer, len(diff_islayer))
    return islayer, max_islayer


#获取发送和接收信息
def get_send_and_recv_info(arr_func: np.ndarray, core_num_func: int,
                           islayer: int, ep_rankid: int, d_c: str, ep_worldsize: int):
    raw_send_or_recv, fullmesh_num_tokens = get_raw_send_or_recv_info(
        arr_func, core_num_func, islayer, ep_worldsize)
    inner_info = []
    unfinish_info, unfinish_info_inner = [[], []], [[], []]
    if islayer == 0:
        send_info, recv_info, unfinish_info = get_fullmesh_send_and_recv_info(
            raw_send_or_recv, core_num_func, ep_rankid, d_c, ep_worldsize, fullmesh_num_tokens)
    elif islayer == 1:
        send_info, recv_info, unfinish_info = get_hierarchy_send_and_recv_info(
            raw_send_or_recv, ep_rankid, d_c, ep_worldsize)
        if d_c == "dispatch":
            inner_info, unfinish_info_inner = get_hierarchy_dis_inner_send_and_recv_info(
                raw_send_or_recv, ep_rankid, d_c, ep_worldsize)
    else:
        logging.error("[A2] 1.1 %s的islayer为%d,不支持分析\n", d_c, islayer)
    return send_info, recv_info, inner_info, [unfinish_info, unfinish_info_inner]


#获取原始发送或接收信息
def get_raw_send_or_recv_info(
        arr_func: np.ndarray, core_num_func: int, islayer: int, ep_worldsize: int):
    per_core = 128
    raw_send_or_recv = []
    for i in range(core_num_func):
        start_pos = 14 + (i * per_core)
        end_pos = (i + 1) * per_core
        raw_send_or_recv.append(arr_func[start_pos:end_pos])
    fullmesh_num_tokens = []
    if islayer == 0:
        start_pos = (core_num_func + 1) * per_core
        end_pos = start_pos + ep_worldsize * 2
        fullmesh_num_tokens = arr_func[start_pos:end_pos]
    return raw_send_or_recv, fullmesh_num_tokens


#获取fullmesh模式下的发送和接收信息
def get_fullmesh_send_and_recv_info(raw_send_or_recv: list, core_num_func: int, ep_rankid: int,
                                    d_c: str, ep_worldsize: int, fullmesh_num_tokens: list):
    has_send_ep = [0] * ep_worldsize
    has_recv_ep = [0] * ep_worldsize
    unfinish_send_ep = []
    unfinish_recv_ep = []
    uint32_bits = 32

    def get_single_core_info(cur_local_rank_id: int,
                             send_flag: np.ndarray) -> int:
        offset = cur_local_rank_id // uint32_bits
        bit_pos = cur_local_rank_id % uint32_bits
        return (send_flag[offset] >> bit_pos) & 0x1
    for i in range(core_num_func):
        cur_send_start_ep_rankid = raw_send_or_recv[i][0]
        cur_send_end_ep_rankid = raw_send_or_recv[i][1]
        cur_recv_start_ep_rankid = raw_send_or_recv[i][2]
        cur_recv_end_ep_rankid = raw_send_or_recv[i][3]
        cur_send_rank_num = cur_send_end_ep_rankid - cur_send_start_ep_rankid
        cur_recv_rank_num = cur_recv_end_ep_rankid - cur_recv_start_ep_rankid
        send_need_uint32_num = (cur_send_rank_num - 1) // uint32_bits + 1
        send_flag = raw_send_or_recv[i][4:4 + send_need_uint32_num]
        recv_flag = raw_send_or_recv[i][4 + send_need_uint32_num:]
        if cur_send_rank_num > 0:
            for local_rank in range(cur_send_rank_num):
                send_ep_rankid = local_rank + cur_send_start_ep_rankid
                has_send_ep[send_ep_rankid] = get_single_core_info(
                    local_rank, send_flag)
                if has_send_ep[send_ep_rankid] == 0:
                    unfinish_send_ep.append(send_ep_rankid)
            for local_rank in range(cur_recv_rank_num):
                recv_ep_rankid = local_rank + cur_recv_start_ep_rankid
                has_recv_ep[recv_ep_rankid] = get_single_core_info(
                    local_rank, recv_flag)
                if has_recv_ep[recv_ep_rankid] == 0:
                    unfinish_recv_ep.append(recv_ep_rankid)
    if unfinish_send_ep != []:
        logging.error(
            "[A2] 1.1 %s EP%d 未完成发送的EP:%s\n",
            d_c,
            ep_rankid,
            unfinish_send_ep)
    if unfinish_recv_ep != []:
        logging.error(
            "[A2] 1.1 %s EP%d 未完成接收的EP:%s\n",
            d_c,
            ep_rankid,
            unfinish_recv_ep)
    send_num_tokens_ep = fullmesh_num_tokens[0:ep_worldsize].tolist()
    recv_num_tokens_ep = fullmesh_num_tokens[ep_worldsize:].tolist()
    return [has_send_ep, send_num_tokens_ep], [has_recv_ep, recv_num_tokens_ep], [
        unfinish_send_ep, unfinish_recv_ep]


#获取hierarchy模式下的发送和接收信息
def get_hierarchy_send_and_recv_info(
        raw_send_or_recv: list, ep_rankid: int, d_c: str, ep_worldsize: int):
    server_num = ep_worldsize // SERVER_RANK_NUM
    has_send_ep = [0] * server_num
    has_recv_ep = [0] * server_num
    send_num_tokens_ep = [0] * server_num
    recv_num_tokens_ep = [0] * server_num
    unfinish_send_ep = []
    unfinish_recv_ep = []
    cur_server_id = ep_rankid // SERVER_RANK_NUM
    for i in range(server_num):
        if raw_send_or_recv[i][0] > 0:
            has_send_ep[raw_send_or_recv[i][0] - 1] = 1
        send_num_tokens_ep[i] = raw_send_or_recv[i][1]
        if raw_send_or_recv[i][2] > 0:
            has_recv_ep[raw_send_or_recv[i][2] - 1] = 1
        recv_num_tokens_ep[i] = raw_send_or_recv[i][3]
    for i in range(server_num):
        if has_send_ep[i] == 0 and i != cur_server_id:
            unfinish_send_ep.append(i)
        if has_recv_ep[i] == 0 and i != cur_server_id:
            unfinish_recv_ep.append(i)
    if unfinish_send_ep != []:
        logging.error("[A2] 1.1 %s EP%d 未完成发送的server:%s\n",
                      d_c, ep_rankid, unfinish_send_ep)
    if unfinish_recv_ep != []:
        logging.error("[A2] 1.1 %s EP%d 未完成接收的server:%s\n",
                      d_c, ep_rankid, unfinish_recv_ep)

    logging.info("[A2] 1.1 %s EP%d has send rank:%s\n", d_c, ep_rankid, has_send_ep)
    logging.info("[A2] 1.1 %s EP%d has recv rank:%s\n", d_c, ep_rankid, has_recv_ep)
    logging.info("[A2] 1.1 %s EP%d send token num:%s\n",
                 d_c, ep_rankid, send_num_tokens_ep)
    logging.info("[A2] 1.1 %s EP%d recv token num:%s\n",
                 d_c, ep_rankid, recv_num_tokens_ep)
    return [has_send_ep, send_num_tokens_ep], [has_recv_ep, recv_num_tokens_ep], [
        unfinish_send_ep, unfinish_recv_ep]


#获取hierarchy模式下dispatch的inner发送和接收信息
def get_hierarchy_dis_inner_send_and_recv_info(
        raw_send_or_recv: list, ep_rankid: int, d_c: str, ep_worldsize: int):
    server_num = ep_worldsize // SERVER_RANK_NUM
    has_send_ep = [0] * server_num
    has_recv_ep = [0] * server_num
    unfinish_send_ep = []
    unfinish_recv_ep = []
    cur_server_id = ep_rankid // SERVER_RANK_NUM
    for i in range(server_num):
        if raw_send_or_recv[i][4] > 0:
            has_send_ep[raw_send_or_recv[i][4] - 1] = 1
        if raw_send_or_recv[i][5] > 0:
            has_recv_ep[raw_send_or_recv[i][5] - 1] = 1
    for i in range(server_num):
        if has_send_ep[i] == 0 and i != cur_server_id:
            unfinish_send_ep.append(i)
        if has_recv_ep[i] == 0 and i != cur_server_id:
            unfinish_recv_ep.append(i)
    if unfinish_send_ep != []:
        logging.error("[A2] 1.1 %s EP%d inner 未完成发送的server:%s\n",
                      d_c, ep_rankid, unfinish_send_ep)
    if unfinish_recv_ep != []:
        logging.error("[A2] 1.1 %s EP%d inner 未完成接收的server:%s\n",
                      d_c, ep_rankid, unfinish_recv_ep)
    logging.info(
        "[A2] 1.1 %s EP%d inner has_send_ep:%s\n",
        d_c,
        ep_rankid,
        has_send_ep)
    logging.info(
        "[A2] 1.1 %s EP%d inner has_recv_ep:%s\n",
        d_c,
        ep_rankid,
        has_recv_ep)
    return [has_send_ep, has_recv_ep], [unfinish_send_ep, unfinish_recv_ep]


#获取算子的dump信息
def get_dumpinfo(target_path: str, bsk: int,
                         moe_expert_num: int, islayer: int, d_c: str):
    expandidx_input, epsendcnt_input = None, None
    if d_c == 'combine':
        expandidx_input = get_dump_expandidx(
            target_path, bsk) if islayer == 0 else []
        epsendcnt_input = get_dump_epsendcnt(target_path, moe_expert_num)
    xactivemask_input = get_dump_xactivemask(target_path, d_c)
    return expandidx_input, xactivemask_input, epsendcnt_input

# 从input.2.bin中获取expandidx
def get_dump_expandidx(target_path: str, bsk: int):
    expandidx_input = []
    for filename_func in os.listdir(os.path.join(target_path)):
        if 'input.2.bin' in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            expandidx_input = np.fromfile(
                file_path_func, dtype=np.int32).tolist()
            expandidx_input = expandidx_input[:bsk]
            return expandidx_input
    return expandidx_input


# 从input.6.bin中获取xactivemask
def get_dump_xactivemask(target_path: str, d_c: str):
    xactivemask_input = []
    target_file_name = 'input.3.bin' if d_c == "dispatch" else 'input.6.bin'
    for filename_func in os.listdir(os.path.join(target_path)):
        if target_file_name in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            xactivemask_input = np.fromfile(
                file_path_func, dtype=np.uint8).tolist()
            if max(xactivemask_input) == 0:
                xactivemask_input = []
            break
    return xactivemask_input


#从input.3.bin中获取epsendcnt
def get_dump_epsendcnt(target_path: str, moe_expert_num: int):
    epsendcnt_input = []
    for filename_func in os.listdir(os.path.join(target_path)):
        if 'input.3.bin' in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            epsendcnt_input = np.fromfile(file_path_func, dtype=np.int32)[
                :moe_expert_num].tolist()
            return epsendcnt_input
    return epsendcnt_input


#获取窗口状态
def get_win_state(win_status: np.ndarray, d_c: str,
                  moe_expert_num: int, ep_worldsize: int, islayer: int):
    if d_c == "dispatch":
        return get_dis_win_state(
            win_status, moe_expert_num, ep_worldsize, islayer)
    return get_com_win_state(win_status, ep_worldsize, islayer)


#获取dispatch算子的窗口状态
def get_dis_win_state(dis_win_status: np.ndarray,
                      moe_expert_num: int, ep_worldsize: int, islayer: int):
    if islayer == 0:
        return get_fullmesh_dis_win_state(
            dis_win_status, moe_expert_num, ep_worldsize)
    else:
        return get_hierarchy_dis_win_status(dis_win_status, ep_worldsize)


#获取combine算子的窗口状态
def get_com_win_state(com_win_status: np.ndarray,
                      ep_worldsize: int, islayer: int):
    if islayer == 0:
        return get_fullmesh_com_win_state(com_win_status, ep_worldsize)
    else:
        return get_hierarchy_com_win_status(com_win_status, ep_worldsize)


#获取hierarchy模式下dispatch算子的窗口状态
def get_hierarchy_dis_win_status(
        dis_win_status: np.ndarray, ep_worldsize: int):
    server_num = ep_worldsize // SERVER_RANK_NUM
    token_win_status = []
    inner_win_status = []
    sync_win_status = []
    per_server = 128
    dis_win_status = dis_win_status.view(np.int32)
    for i in range(server_num):
        token_win_status.append(dis_win_status[per_server * i])
        inner_win_status.append(
            dis_win_status[per_server * i + per_server * (server_num + 1)])
    sync_offset = per_server * (server_num + 1) * 2
    for i in range(SERVER_RANK_NUM):
        sync_win_status.append(
            dis_win_status[sync_offset + UINT32_PER_BLOCK * i])
    return [token_win_status, sync_win_status, inner_win_status]


#获取hierarchy模式下combine算子的窗口状态
def get_hierarchy_com_win_status(
        com_win_status: np.ndarray, ep_worldsize: int):
    server_num = ep_worldsize // SERVER_RANK_NUM
    token_win_status = []
    sync_win_status = []
    per_server = 128
    com_win_status = com_win_status.view(np.int32)
    for i in range(server_num):
        token_win_status.append(com_win_status[per_server * i])
    sync_offset = per_server * (server_num + 1)
    for i in range(SERVER_RANK_NUM):
        sync_win_status.append(
            com_win_status[sync_offset + UINT32_PER_BLOCK * i])
    return [token_win_status, sync_win_status]


#获取fullmesh模式下dispatch算子的窗口状态
def get_fullmesh_dis_win_state(
        dis_win_status: np.ndarray, moe_expert_num: int, ep_worldsize: int):
    local_moe_expert_num = moe_expert_num // ep_worldsize
    status_entry_count = (local_moe_expert_num // UINT32_PER_BLOCK + 1) * UINT32_PER_BLOCK
    token_win_status = []
    arrive_win_status = []
    dis_win_status = dis_win_status.view(np.int32)
    for i in range(ep_worldsize):
        offset = i * status_entry_count
        token_win_status.append(
            dis_win_status[offset: offset + local_moe_expert_num].tolist())
        arrive_win_status.append(
            hex(dis_win_status[offset + status_entry_count - 1].view(np.uint32).item()))
    return [token_win_status, arrive_win_status]


#获取fullmesh模式下combine算子的窗口状态
def get_fullmesh_com_win_state(com_win_status: np.ndarray, ep_worldsize: int):
    token_win_status = []
    arrive_win_status = []
    per_ep = UINT32_PER_BLOCK * 2
    com_win_status = com_win_status.view(np.uint32)
    for i in range(ep_worldsize):
        offset = i * per_ep
        token_win_status.append(com_win_status[offset])
        offset += UINT32_PER_BLOCK
        arrive_win_status.append(hex(com_win_status[offset]))
    return [token_win_status, arrive_win_status]


#获取设备数量
def get_device_count():
    try:
        import torch
        return torch.npu.device_count()
    except Exception:
        pass
    arch = os.uname().machine
    if arch == 'aarch64':
        return 8
    elif arch == 'x86_64':
        return 16
    else:
        logging.error(f"[A2] 未知架构:{arch}，默认返回8个设备")
        return 8


#保存dump数据到CSV文件
def save_dump_data(
    ep_rankid: int, ep_worldsize: int, moe_expert_num: int, bs: int, k: int, islayer: int, d_c: str, buffer_id: int,
    *,
    dis_run_cnts=0, com_run_cnts=0,
    dis_send_info=None, dis_recv_info=None,
    dis_inner_info=None, dis_unfinish_info=None,
    com_send_info=None, com_recv_info=None,
    dis_win_state=None, com_win_state=None,
    com_unfinish_info=None,
    expertids=None, expandidx_input=None,
    xactivemask_input=None, epsendcnt_input=None,
):
    os.makedirs(PARENT_PATH, exist_ok=True)
    device_count = get_device_count()
    server_id = ep_rankid // device_count
    local_rank_id = ep_rankid % device_count
    file_name = os.path.join(PARENT_PATH, DUMP_FILE_NAME.format(server_id))
    file_exists = os.path.exists(file_name)
    if expertids is None:
        expertids = np.array([], dtype=np.int32)
    dump_input_info = pd.DataFrame([[ep_rankid, ep_worldsize, moe_expert_num, bs, k, islayer, d_c, buffer_id,
        dis_run_cnts, com_run_cnts,
        dis_send_info, dis_recv_info,
        dis_inner_info, dis_unfinish_info,
        com_send_info, com_recv_info,
        com_unfinish_info,
        dis_win_state, com_win_state,
        expertids.tolist(), expandidx_input, xactivemask_input,
        epsendcnt_input]],
        columns=['ep_rankid', 'ep_worldsize', 'moe_expert_num', 'bs', 'k',
                '是否分层', '当前报错算子', '当前buffer_id',
                'dispatch执行次数', 'combine执行次数',
                'dispatch是否发送及token数', 'dispatch是否接收及token数',
                'dispatchInner表发送接收情况', 'dispatch未完成行为',
                'combine是否发送及token数', 'combine是否接收及token数',
                'combine未完成行为',
                'dispatch接收状态',
                'combine接收状态',
                'expertids', 'expandidx_input', 'xactivemask_input',
                'epsendcnt_input'],
        index=[local_rank_id])
    dump_input_info.to_csv(file_name, index=True, mode='a',
                           header=not file_exists, encoding="gbk")
    logging.info(f"[A2] EP{ep_rankid} dump info data save to {file_name}")


#保存窗口数据到CSV文件
def save_win_data(all_unfinish_info, islayer: int,
                  ep_rankid: int, ep_worldsize: int, d_c: str,
                  min_run_coreid, run_info):
    device_count = get_device_count()
    server_id = ep_rankid // device_count
    local_rank_id = ep_rankid % device_count

    os.makedirs(PARENT_PATH, exist_ok=True)
    file_name = os.path.join(PARENT_PATH,
                             ERROR_INFO_FILE_NAME.format(server_id))
    file_exists = os.path.exists(file_name)
    unfinish_info, unfinish_info_inner = all_unfinish_info
    unfinish_send_ep, unfinish_recv_ep = unfinish_info
    unfinish_send_ep_inner, unfinish_recv_ep_inner = unfinish_info_inner
    columns = ['ep_rankid', 'ep_worldsize', '是否分层', '出错的算子名', '最早出问题的核', '最早出问题的核执行位置',
               '未发送token到这些rank,hierarchy为serverId',
               '未接收这些rank的token,hierarchy为serverId',
               '未发送inner到这些serverId,hierarchy-dispatch有效',
               '未接收这些serverId的inner,hierarchy-dispatch有效']
    save_datas = [ep_rankid, ep_worldsize, islayer, d_c, min_run_coreid, run_info[min_run_coreid[0]],
                  unfinish_send_ep, unfinish_recv_ep, unfinish_send_ep_inner, unfinish_recv_ep_inner]
    core_num = len(run_info)
    for coreid in range(core_num):
        columns.append(f"核{coreid}执行位置")
        save_datas.append(run_info[coreid])

    dump_input_info = pd.DataFrame(
        [save_datas],
        columns=columns,
        index=[local_rank_id])
    dump_input_info.to_csv(file_name, index=True, mode='a',
                           header=not file_exists, encoding="gbk")
    logging.info(f"[A2] EP{ep_rankid} win info data save to {file_name}")


#获取运行位置排序
def get_runpos_sort(next_pos):
    num = len(next_pos)
    pos_sort = [0] * num
    cnt = 0
    cur_pos = 0
    while next_pos[cur_pos] < num:
        pos_sort[cur_pos] = cnt
        cnt += 1
        cur_pos = next_pos[cur_pos]
    pos_sort[cur_pos] = cnt
    return pos_sort


#获取运行位置分析的基础信息
def get_runpos_analysis_basic(islayer: int, d_c: str):
    # 以下内容根据mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_a2_constant.h定义
    pos_common_mapping = ['UNSTART', 'INIT', 'FINISH']
    if islayer == 0:
        if d_c == "dispatch":
            pos_diff_mapping = [
                'REORDERTOKEN', 'SEND',
                'WAIT_DISPATCH', 'GET_STATUS_CUM_SUM',
                'LOCAL_WINDOW_COPY']
        else:
            pos_diff_mapping = ['ALLTOALLDISPATCH', 'WAIT_DISPATCH']
    else:
        if d_c == "dispatch":
            pos_diff_mapping = [
                'REORDERTOKEN', 'INTER_FINISH',
                'INTRA_FINISH', 'IPC2OUT']
        else:
            pos_diff_mapping = [
                'GM2IPC', 'WAIT2IPC',
                'INTER_SEND', 'INTER_RECV']

    pos_mapping = pos_common_mapping + pos_diff_mapping
    next_pos = [1, 3, len(pos_mapping)]
    next_tmp_pos = next_pos[1] + 1
    for _ in range(3, len(pos_mapping) - 1):
        next_pos.append(next_tmp_pos)
        next_tmp_pos += 1
    next_pos.append(2)
    pos_position = get_runpos_sort(next_pos)
    return pos_mapping, next_pos, pos_position


#分析运行位置
def analyze_runpos(runpos: List[int], islayer: int, d_c: str):
    pos_mapping, next_pos, pos_position = get_runpos_analysis_basic(
        islayer, d_c)
    core_num = len(runpos)
    run_info = [""] * core_num
    min_run_coreid = []
    min_run_pos = len(pos_mapping)
    for core_id in range(core_num):
        cur_run_info = pos_mapping[runpos[core_id]]
        next_run_info = pos_mapping[next_pos[runpos[core_id]]
                                    ] if next_pos[runpos[core_id]] < len(pos_mapping) else ""
        run_info[core_id] = f"{cur_run_info} -> {next_run_info}"
        if min_run_pos > pos_position[runpos[core_id]]:
            min_run_coreid = [core_id]
            min_run_pos = pos_position[runpos[core_id]]
        elif min_run_pos == pos_position[runpos[core_id]]:
            min_run_coreid.append(core_id)
    logging.error("[A2] 最早出问题的核:%s, 对应代码区域:%s", min_run_coreid,
                  run_info[min_run_coreid[0]])
    return min_run_coreid, run_info


#分析单卡数据
def analyze_single_card(arr_func_dis: np.ndarray, arr_func_com: np.ndarray, core_num_func: int,
                        ep_rankid: int, d_c: str, ep_worldsize: int, runpos: List[int],
                        moe_expert_num: int, bs: int, k: int, buffer_id: int,
                        target_path: str,
                        expertids: np.ndarray,
                        dis_status: np.ndarray, com_status: np.ndarray,
                        *,
                        dis_run_cnts=0, com_run_cnts=0):
    if d_c == "dispatch":
        arr_func = arr_func_dis
    else:
        arr_func = arr_func_com
    _, islayer = get_islayer_ep(arr_func, core_num_func, d_c)
    dis_send_info, dis_recv_info, dis_inner_info, dis_unfinish_info = get_send_and_recv_info(
        arr_func_dis, core_num_func, islayer, ep_rankid, "dispatch", ep_worldsize)
    dis_window_status = get_win_state(
        dis_status, "dispatch",
        moe_expert_num, ep_worldsize, islayer)
    if d_c == "combine":
        com_send_info, com_recv_info, _, com_unfinish_info = get_send_and_recv_info(
            arr_func_com, core_num_func, islayer, ep_rankid, d_c, ep_worldsize)
        com_window_status = get_win_state(
            com_status, d_c, moe_expert_num, ep_worldsize, islayer)
    else:
        com_send_info, com_recv_info, com_unfinish_info = None, None, None
        com_window_status = None
    expandidx_input, xactivemask_input, epsendcnt_input = get_dumpinfo(
        target_path, bs * k, moe_expert_num, islayer, d_c)

    min_run_coreid, run_info = analyze_runpos(runpos, islayer, d_c)
    save_dump_data(
        ep_rankid, ep_worldsize, moe_expert_num, bs, k, islayer, d_c, buffer_id,
        dis_run_cnts=dis_run_cnts, com_run_cnts=com_run_cnts,
        dis_send_info=dis_send_info, dis_recv_info=dis_recv_info,
        dis_inner_info=dis_inner_info, dis_unfinish_info=dis_unfinish_info,
        com_send_info=com_send_info, com_recv_info=com_recv_info,
        dis_win_state=dis_window_status, com_win_state=com_window_status,
        com_unfinish_info=com_unfinish_info,
        expertids=expertids, expandidx_input=expandidx_input,
        xactivemask_input=xactivemask_input, epsendcnt_input=epsendcnt_input,
    )
    if d_c == "dispatch":
        all_unfinish_info = dis_unfinish_info
    else:
        all_unfinish_info = com_unfinish_info
    save_win_data(all_unfinish_info, islayer, ep_rankid, ep_worldsize, d_c,
                  min_run_coreid, run_info)


#判断是否为A2架构
def is_A2(soc_version):
    return soc_version == SOC_VERSION_910B

#以下所有函数用于集群分析
def isinvalid_token(pos, xactivemask, k, expid, moe_expert_num):
    if expid >= moe_expert_num or expid < 0:
        return True
    tokenid = pos // k
    # logging.info(f"xactivemask:{xactivemask}")
    if xactivemask is not None and xactivemask.size > 0:
        if xactivemask.ndim == 1:
            return xactivemask[tokenid] == 0
        return xactivemask[pos] == 0
    return False


def check_single_card_expandidx(expandidx_input, expertids, xactivemask, moe_expert_num, bs, k):
    expertids_flatten = np.array(expertids, dtype=np.int32).reshape(-1)
    bsk = min(bs * k, expertids_flatten.size)
    expandidx_input = np.array(expandidx_input, dtype=np.int32).reshape(-1)[:bsk]
    expandidx_golden = np.zeros(bsk, dtype=np.int32)
    if xactivemask is not None:
        xactivemask = np.array(xactivemask, dtype=np.int32)
    count_dict = {}
    for i in range(expertids_flatten.size):
        expid = expertids_flatten[i].item()
        if isinvalid_token(i, xactivemask, k, expid, moe_expert_num):
            continue
        count_dict[expid] = count_dict.get(expid, -1) + 1
        expandidx_golden[i] = count_dict[expid]
    return np.array_equal(expandidx_golden, expandidx_input)


def check_expandidx(expandidx_input_func, expertids_func, xactivemask_func, bs, k, moe_expert_num):
    error_rankid_info = []
    for ep_rankid in sorted(expandidx_input_func.keys(), key=lambda x: int(x)):
        if not check_single_card_expandidx(expandidx_input_func[ep_rankid], expertids_func[ep_rankid],
            xactivemask_func[ep_rankid], moe_expert_num=moe_expert_num, bs=bs, k=k):
            error_rankid_info.append(ep_rankid)
        else:
            logging.info(f"[A2][Fullmesh] Combine EP{ep_rankid} Input ExpandIdxInput is right.")
    if len(error_rankid_info) > 0:
        logging.error(f"[A2][Fullmesh] Combine Input ExpandIdxInput is wrong, cards: {error_rankid_info}")

    return error_rankid_info


def check_epsendcnt(epsendcnt_input_func, expertids_func, xactivemask_func, ep_worldsize, bs, k, moe_expert_num, islayer):
    error_rankid_info = []
    has_cards = len(epsendcnt_input_func.keys())
    if has_cards != ep_worldsize:
        logging.error(f"[A2] Combine EP send count check failed: expected {ep_worldsize} ranks"\
                f"but get {has_cards} ranks, layer: {islayer}")
        logging.error(f"[A2] Only EP{list(epsendcnt_input_func.keys())} have data.")
        return error_rankid_info
    epsendcnt_golden = np.zeros((ep_worldsize, moe_expert_num), dtype=np.int32)
    for ep_rankid in sorted(epsendcnt_input_func.keys(), key=lambda x: int(x)):
        expert_ids = np.array(expertids_func[ep_rankid], dtype=np.int32).flatten()
        bsk = min(bs * k, expert_ids.size)
        if not islayer and xactivemask_func[ep_rankid] is not None:
            xactivemask = np.array(xactivemask_func[ep_rankid], dtype=np.uint8)
            if xactivemask.ndim == 1:
                xactivemask = xactivemask[:bs, None].repeat(k, axis=-1).flatten()
            else:
                xactivemask = xactivemask.flatten()[:bsk]
            mask = (xactivemask == 1) & (expert_ids < moe_expert_num) & (expert_ids >= 0)
            active_indices = np.where(mask)[0]
            expert_ids = expert_ids[active_indices]
        epsendcnt_golden[ep_rankid] = np.bincount(expert_ids, minlength=moe_expert_num)
    epsendcnt_golden = np.cumsum(epsendcnt_golden.T.reshape(ep_worldsize, moe_expert_num), axis=-1, dtype=np.int32)

    for ep_rankid in sorted(epsendcnt_input_func.keys(), key=lambda x: int(x)):
        epsendcnt = np.array(epsendcnt_input_func[ep_rankid], dtype=np.int32)
        if not np.array_equal(epsendcnt, epsendcnt_golden[ep_rankid]):
            logging.error(
                f"[A2][Fullmesh] Combine EP send count mismatch: "
                f"golden={epsendcnt_golden[ep_rankid]}, dump={epsendcnt}"
            )
            error_rankid_info.append(ep_rankid)
    return error_rankid_info

def cal_recv_tokennum(expertids, xactivemask, ep_worldsize, moe_expert_num, local_moe_expert_num, k):
    if xactivemask is not None:
        xactivemask = np.array(xactivemask, dtype=np.int32)
    expertids_flatten = np.array(expertids, dtype=np.int32).reshape(-1)
    count_dict = [0] * ep_worldsize
    for i in range(expertids_flatten.size):
        expid = expertids_flatten[i].item()
        target_ep_rankid = expid // local_moe_expert_num
        if isinvalid_token(i, xactivemask, k, expid, moe_expert_num):
            continue
        count_dict[target_ep_rankid] += 1
    return count_dict


def check_fullmesh_com_win(com_win_state_func, expertids_func, xactivemask_func, unfinish_recv_ep_func,
    ep_worldsize, moe_expert_num, k):
    error_rankid_info = []
    local_moe_expert_num = moe_expert_num // ep_worldsize
    for ep_rankid in sorted(com_win_state_func.keys(), key=lambda x: int(x)):
        com_win_state = com_win_state_func[ep_rankid]
        token_nums_golden = cal_recv_tokennum(expertids_func[ep_rankid],
            xactivemask_func[ep_rankid], ep_worldsize, moe_expert_num, local_moe_expert_num, k)
        token_nums_golden = np.array(token_nums_golden, dtype=np.int32)
        token_nums_dump = np.array(com_win_state[0], dtype=np.int32)
        if not np.array_equal(token_nums_dump, token_nums_golden):
            logging.error(
                f"[A2][Fullmesh] Combine Win State token nums mismatch: "
                f"golden={token_nums_golden}, dump={token_nums_dump}"
            )
            diff_ep_rankid = np.where(token_nums_golden != token_nums_dump)[0]
            logging.error(
                f"[A2][Fullmesh] Combine Win State token nums mismatch at "
                f"ep_rankid={diff_ep_rankid}"
            )
            error_rankid_info.append((ep_rankid, diff_ep_rankid))
        unfinish_recv_ep = unfinish_recv_ep_func[ep_rankid][0][1]
        if len(unfinish_recv_ep) > 0:
            logging.error(
                f"[A2][Fullmesh] EP{ep_rankid} Combine Win State has unfinished recv eps: {unfinish_recv_ep}"
            )
            logging.error(
                f"[A2][Fullmesh] EP{ep_rankid} Combine Win State target flag in dump: "
                f"{np.array(com_win_state[0])[unfinish_recv_ep]}"
            )
            if np.array_equal(token_nums_dump[unfinish_recv_ep], token_nums_golden[unfinish_recv_ep]):
                logging.error(f"[A2][Fullmesh] EP{ep_rankid} Combine Win State token nums is right, maybe flag is wrong.")
            else:
                logging.error(
                    f"[A2][Fullmesh] EP{ep_rankid} Combine Win State token nums in dump: "
                    f"{token_nums_dump[unfinish_recv_ep]}"
                )
                logging.error(
                    f"[A2][Fullmesh] EP{ep_rankid} Combine Win State token nums golden: "
                    f"{token_nums_golden[unfinish_recv_ep]}"
                )
    return error_rankid_info


def check_com(expandidx_input_func, epsendcnt_input_func, expertids_func, xactivemask_func, com_win_state_func,
    unfinish_recv_ep_func, ep_worldsize, bs, k, moe_expert_num, islayer):
    check_epsendcnt(epsendcnt_input_func, expertids_func, xactivemask_func, ep_worldsize, bs, k, moe_expert_num, islayer)
    if not islayer:
        check_expandidx(expandidx_input_func, expertids_func, xactivemask_func, bs, k, moe_expert_num)
        check_fullmesh_com_win(com_win_state_func, expertids_func, xactivemask_func, unfinish_recv_ep_func,
            ep_worldsize, moe_expert_num, k)


def cal_dis_win_status_golden(expertids_func, xactivemask_func, ep_worldsize, moe_expert_num, bs, k):
    dis_win_status_golden = [None] * ep_worldsize
    local_moe_expert_num = moe_expert_num // ep_worldsize
    for ep_rankid in sorted(expertids_func.keys(), key=lambda x: int(x)):
        status_golden = np.zeros((ep_worldsize, local_moe_expert_num), dtype=np.int32)
        expertids_flatten = np.array(expertids_func[ep_rankid], dtype=np.int32).reshape(-1)
        bsk = min(bs * k, expertids_flatten.size)
        expertids_flatten = expertids_flatten[:bsk]
        xactivemask = xactivemask_func[ep_rankid]
        if xactivemask is not None:
            xactivemask = np.array(xactivemask, dtype=np.int32)
        for expidx, expid in enumerate(expertids_flatten):
            if not isinvalid_token(expidx, xactivemask, k, expid, moe_expert_num):
                status_golden[expid // local_moe_expert_num, expid % local_moe_expert_num] += 1
        dis_win_status_golden[int(ep_rankid)] = status_golden
    return dis_win_status_golden


def check_single_card_dis_win(dis_win_state, dis_win_state_flag, dis_win_status_golden, dis_unfinish_info, ep_rankid):
    error_rankid_info = []
    unfinish_recv_ep = dis_unfinish_info[1]
    if len(unfinish_recv_ep) > 0:
        logging.info(f'[A2][Fullmesh] Dispatch has\'t recv token from EP {unfinish_recv_ep}, according to ADump.')
    for target_rankid, win_state in enumerate(dis_win_state):
        if dis_win_status_golden[target_rankid] is None:
            logging.info(f'[A2][Fullmesh] Dispatch has no EP {target_rankid} expertids.')
            continue
        win_state_golden = dis_win_status_golden[target_rankid][int(ep_rankid)]
        local_moe_expert_num = win_state_golden.shape[-1]
        win_state = np.array(win_state, dtype=np.int32)[:local_moe_expert_num]
        if not np.array_equal(win_state, win_state_golden):
            logging.info(f'[A2][Fullmesh] Dispatch EP{ep_rankid} recv wrong status from EP{target_rankid}')
            logging.info(f'[A2][Fullmesh] Win status golden: {win_state_golden.tolist()}')
            logging.info(f'[A2][Fullmesh] Win status current: {win_state.tolist()}')
            error_rankid_info.append(target_rankid)
        else:
            if dis_win_state_flag[target_rankid] == '0xffffffff':
                logging.info(f'[A2][Fullmesh] Current EP{ep_rankid} from EP{target_rankid} Win status flag is right: {dis_win_state_flag[target_rankid]}.')
                if target_rankid in unfinish_recv_ep:
                    logging.info(f'[A2]But the token flag maybe wrong.')
            else:
                logging.info(f'[A2][Fullmesh] Current EP{ep_rankid} from EP{target_rankid} Win status flag is wrong: {dis_win_state_flag[target_rankid]}')
    return error_rankid_info


def check_fullmesh_dis_win(dis_win_state_func, expertids_func, xactivemask_func, unfinish_recv_ep_func, ep_worldsize, moe_expert_num, bs, k):
    dis_win_status_golden = cal_dis_win_status_golden(expertids_func, xactivemask_func,
        ep_worldsize, moe_expert_num, bs, k)
    error_rankid_infos = []
    for ep_rankid in sorted(dis_win_state_func.keys(), key=lambda x: int(x)):
        error_rankid_info = check_single_card_dis_win(dis_win_state_func[ep_rankid][0],
            dis_win_state_func[ep_rankid][1],
            dis_win_status_golden, unfinish_recv_ep_func[ep_rankid][0], ep_rankid)
        error_rankid_infos.append((ep_rankid, error_rankid_info))
    return error_rankid_infos


def check_dis(dis_win_state_func, expertids_func, xactivemask_func, unfinish_recv_ep_func, ep_worldsize, moe_expert_num, bs, k):
    error_rankid_infos = check_fullmesh_dis_win(dis_win_state_func, expertids_func,
        xactivemask_func, unfinish_recv_ep_func, ep_worldsize, moe_expert_num, bs, k)
    if len(error_rankid_infos) > 0:
        logging.error(f"[A2][Fullmesh] Dispatch Win State status mismatch: {error_rankid_infos}")


def analysis_all_cards(parent_path=PARENT_PATH):
    dis_win_state_func, com_win_state_func = {}, {}
    dis_unfinish_info_func = {}
    expertids_func, xactivemask_func = {}, {}
    unfinish_ep_func = {}
    expandidx_input_func, epsendcnt_input_func = {}, {}
    ep_worldsize, moe_expert_num = None, None
    bs, k, islayer = None, None, None
    d_c = None
    filename = None
    for filename in os.listdir(parent_path):
        if filename.startswith(DUMP_FILE_PREFIX) and filename.endswith('.csv'):
            with open(os.path.join(parent_path, filename), 'r', encoding="gbk") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ep_rankid = row['ep_rankid'].strip()
                    dis_win_state_func[ep_rankid] = ast.literal_eval(row['dispatch接收状态'])
                    d_c = row['当前报错算子'].strip()
                    expertids_func[ep_rankid] = ast.literal_eval(row['expertids'])
                    xactivemask_func[ep_rankid] = ast.literal_eval(row['xactivemask_input'])
                    if d_c == "combine":
                        expandidx_input_func[ep_rankid] = ast.literal_eval(row['expandidx_input'])
                        epsendcnt_input_func[ep_rankid] = ast.literal_eval(row['epsendcnt_input'])
                        unfinish_ep_func[ep_rankid] = ast.literal_eval(row['combine未完成行为'])
                        com_win_state_func[ep_rankid] = ast.literal_eval(row['combine接收状态'])
                    else:
                        unfinish_ep_func[ep_rankid] = ast.literal_eval(row['dispatch未完成行为'])
                    ep_worldsize = int(row['ep_worldsize'])
                    moe_expert_num = int(row['moe_expert_num'])
                    bs = int(row['bs'])
                    k = int(row['k'])
                    islayer = int(row['是否分层'])
    logging.info(f"[A2] Start to analysis {filename}")
    logging.info(f"[A2] ep_worldsize: {ep_worldsize}, moe_expert_num: {moe_expert_num}, bs: {bs}, k: {k}, islayer: {islayer}, d_c: {d_c}")
    if d_c is not None:
        if islayer == 0:
            if d_c == "dispatch":
                check_dis(dis_win_state_func, expertids_func, xactivemask_func, unfinish_ep_func, ep_worldsize, moe_expert_num, bs, k)
            else:
                check_com(expandidx_input_func, epsendcnt_input_func, expertids_func, xactivemask_func, com_win_state_func,
                    unfinish_ep_func, ep_worldsize, bs, k, moe_expert_num, islayer)
    logging.info(f"[A2] Analysis {filename} done.")

if __name__ == "__main__":
    analysis_all_cards(parent_path=PARENT_PATH)