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
import pandas as pd
import numpy as np
import logging
from collections import Counter
from typing import List
SOC_VERSION_910B = "910B"
logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

UINT32_PER_BLOCK = 8
SERVER_RANK_NUM = 8

PARENT_PATH = "result_A2/"
DUMP_FILE_NAME = "win_dump_input_serverid_{}.csv"
ERROR_INFO_FILE_NAME = "errorinfo_{}.csv"


def get_islayer_ep(arr_func: np.ndarray, core_num_func: int, d_c: str):
    per_core = 128
    islayer = []
    for i in range(core_num_func):
        islayer.append(arr_func[11 + (i * per_core)])
    logging.info("1.1 %s各核islayer:%s\n", d_c, islayer)
    islayer_count = Counter(islayer)
    max_islayer = max(islayer_count, key=islayer_count.get)
    diff_islayer = [idx for idx, val in enumerate(
        islayer) if val != max_islayer]
    if diff_islayer != []:
        logging.error("1.1 %s有如下下标的核的islayer与其他核不相等%s,共%d个核\n",
                      d_c, diff_islayer, len(diff_islayer))
    return islayer, max_islayer


def get_send_and_recv_info(arr_func: np.ndarray, core_num_func: int,
                           islayer: int, ep_rankid: int, d_c: str, ep_worldsize: int):
    raw_send_or_recv, fullmesh_num_tokens = get_raw_send_or_recv_info(
        arr_func, core_num_func, islayer, ep_worldsize)
    inner_has_send_ep, inner_has_recv_ep = [], []
    unfinish_info, unfinish_info_inner = [], []
    if islayer == 0:
        has_send_ep, has_recv_ep, send_num_tokens_ep, recv_num_tokens_ep, unfinish_info = get_fullmesh_send_and_recv_info(
            raw_send_or_recv, core_num_func, ep_rankid, d_c, ep_worldsize, fullmesh_num_tokens)
    elif islayer == 1:
        has_send_ep, has_recv_ep, send_num_tokens_ep, recv_num_tokens_ep, unfinish_info = get_hierarchy_send_and_recv_info(
            raw_send_or_recv, ep_rankid, d_c, ep_worldsize)
        if d_c == "dispatch":
            inner_has_send_ep, inner_has_recv_ep, unfinish_info_inner = get_hierarchy_dis_inner_send_and_recv_info(
                raw_send_or_recv, ep_rankid, d_c, ep_worldsize)
    else:
        logging.error("1.1 %s的islayer为%d,不支持分析\n", d_c, islayer)
    return [has_send_ep, send_num_tokens_ep], [has_recv_ep, recv_num_tokens_ep], \
        [inner_has_send_ep, inner_has_recv_ep], [
            unfinish_info, unfinish_info_inner]


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


def get_fullmesh_send_and_recv_info(raw_send_or_recv: list, core_num_func: int, ep_rankid: int,
                                    d_c: str, ep_worldsize: int, fullmesh_num_tokens: list):
    has_send_ep = [0] * ep_worldsize
    has_recv_ep = [0] * ep_worldsize
    unfinish_send_ep = []
    unfinish_recv_ep = []
    UINT32_BITS = 32

    def get_single_core_info(cur_local_rank_id: int,
                             send_flag: np.ndarray) -> int:
        offset = cur_local_rank_id // UINT32_BITS
        bit_pos = cur_local_rank_id % UINT32_BITS
        return (send_flag[offset] >> bit_pos) & 0x1
    for i in range(core_num_func):
        cur_send_start_ep_rankid = raw_send_or_recv[i][0]
        cur_send_end_ep_rankid = raw_send_or_recv[i][1]
        cur_recv_start_ep_rankid = raw_send_or_recv[i][2]
        cur_recv_end_ep_rankid = raw_send_or_recv[i][3]
        cur_send_rank_num = cur_send_end_ep_rankid - cur_send_start_ep_rankid
        cur_recv_rank_num = cur_recv_end_ep_rankid - cur_recv_start_ep_rankid
        send_need_uint32_num = (cur_send_rank_num - 1) // UINT32_BITS + 1
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
            "1.1 %s EP%d 未完成发送的EP:%s\n",
            d_c,
            ep_rankid,
            unfinish_send_ep)
    if unfinish_recv_ep != []:
        logging.error(
            "1.1 %s EP%d 未完成接收的EP:%s\n",
            d_c,
            ep_rankid,
            unfinish_recv_ep)
    send_num_tokens_ep = fullmesh_num_tokens[0:ep_worldsize].tolist()
    recv_num_tokens_ep = fullmesh_num_tokens[ep_worldsize:].tolist()
    return has_send_ep, has_recv_ep, send_num_tokens_ep, recv_num_tokens_ep, [
        unfinish_send_ep, unfinish_recv_ep]


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
        logging.error("1.1 %s EP%d 未完成发送的server:%s\n",
                      d_c, ep_rankid, unfinish_send_ep)
    if unfinish_recv_ep != []:
        logging.error("1.1 %s EP%d 未完成接收的server:%s\n",
                      d_c, ep_rankid, unfinish_recv_ep)

    logging.info("1.1 %s EP%d has_send_ep:%s\n", d_c, ep_rankid, has_send_ep)
    logging.info("1.1 %s EP%d has_recv_ep:%s\n", d_c, ep_rankid, has_recv_ep)
    logging.info("1.1 %s EP%d send_num_tokens_ep:%s\n",
                 d_c, ep_rankid, send_num_tokens_ep)
    logging.info("1.1 %s EP%d recv_num_tokens_ep:%s\n",
                 d_c, ep_rankid, recv_num_tokens_ep)
    return has_send_ep, has_recv_ep, send_num_tokens_ep, recv_num_tokens_ep, [
        unfinish_send_ep, unfinish_recv_ep]


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
        logging.error("1.1 %s EP%d inner 未完成发送的server:%s\n",
                      d_c, ep_rankid, unfinish_send_ep)
    if unfinish_recv_ep != []:
        logging.error("1.1 %s EP%d inner 未完成接收的server:%s\n",
                      d_c, ep_rankid, unfinish_recv_ep)
    logging.info(
        "1.1 %s EP%d inner has_send_ep:%s\n",
        d_c,
        ep_rankid,
        has_send_ep)
    logging.info(
        "1.1 %s EP%d inner has_recv_ep:%s\n",
        d_c,
        ep_rankid,
        has_recv_ep)
    return has_send_ep, has_recv_ep, [unfinish_send_ep, unfinish_recv_ep]


def get_combine_dumpinfo(target_path: str, bsk: int,
                         moe_expert_num: int, islayer: int):
    expandidx_input = get_dump_expandidx(
        target_path, bsk) if islayer == 0 else []
    xactivemask_input = get_dump_xactivemask(target_path)
    epsendcnt_input = get_dump_epsendcnt(target_path, moe_expert_num)
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
def get_dump_xactivemask(target_path: str):
    xactivemask_input = []
    for filename_func in os.listdir(os.path.join(target_path)):
        if 'input.6.bin' in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            xactivemask_input = np.fromfile(
                file_path_func, dtype=np.int8).tolist()
            return xactivemask_input
    return xactivemask_input

# 从input.3.bin中获取epsendcnt


def get_dump_epsendcnt(target_path: str, moe_expert_num: int):
    epsendcnt_input = []
    for filename_func in os.listdir(os.path.join(target_path)):
        if 'input.3.bin' in filename_func:
            file_path_func = os.path.join(target_path, filename_func)
            epsendcnt_input = np.fromfile(file_path_func, dtype=np.int32)[
                :moe_expert_num].tolist()
            return epsendcnt_input
    return epsendcnt_input


def get_win_state(win_status: np.ndarray, d_c: str,
                  moe_expert_num: int, ep_worldsize: int, islayer: int):
    if d_c == "dispatch":
        return get_dis_win_state(
            win_status, moe_expert_num, ep_worldsize, islayer)
    return get_com_win_state(win_status, ep_worldsize, islayer)


def get_dis_win_state(dis_win_status: np.ndarray,
                      moe_expert_num: int, ep_worldsize: int, islayer: int):
    if islayer == 0:
        return get_fullmesh_dis_win_state(
            dis_win_status, moe_expert_num, ep_worldsize)
    else:
        return get_hierarchy_dis_win_status(dis_win_status, ep_worldsize)


def get_com_win_state(com_win_status: np.ndarray,
                      ep_worldsize: int, islayer: int):
    if islayer == 0:
        return get_fullmesh_com_win_state(com_win_status, ep_worldsize)
    else:
        return get_hierarchy_com_win_status(com_win_status, ep_worldsize)


def get_hierarchy_dis_win_status(
        dis_win_status: np.ndarray, ep_worldsize: int):
    server_num = ep_worldsize // SERVER_RANK_NUM
    token_win_status = []
    inner_win_status = []
    sync_win_status = []
    per_server = 128
    for i in range(server_num):
        token_win_status.append(dis_win_status[per_server * i])
        inner_win_status.append(
            dis_win_status[per_server * i + per_server * (server_num + 1)])
    sync_offset = per_server * (server_num + 1) * 2
    UINT32_PER_BLOCK = 32
    for i in range(SERVER_RANK_NUM):
        sync_win_status.append(
            dis_win_status[sync_offset + UINT32_PER_BLOCK * i])
    return [token_win_status, sync_win_status, inner_win_status]


def get_hierarchy_com_win_status(
        com_win_status: np.ndarray, ep_worldsize: int):
    server_num = ep_worldsize // SERVER_RANK_NUM
    token_win_status = []
    sync_win_status = []
    per_server = 128
    for i in range(server_num):
        token_win_status.append(com_win_status[per_server * i])
    sync_offset = per_server * (server_num + 1)
    for i in range(SERVER_RANK_NUM):
        sync_win_status.append(
            com_win_status[sync_offset + UINT32_PER_BLOCK * i])
    return [token_win_status, sync_win_status]


def get_fullmesh_dis_win_state(
        dis_win_status: np.ndarray, moe_expert_num: int, ep_worldsize: int):
    local_moe_expert_num = moe_expert_num // ep_worldsize
    status_entry_count = ((local_moe_expert_num + 1 - 1) //
                          UINT32_PER_BLOCK + 1) * UINT32_PER_BLOCK
    token_win_status = []
    arrive_win_status = []
    for i in range(ep_worldsize):
        offset = i * status_entry_count
        token_win_status.append(
            dis_win_status[offset: offset + local_moe_expert_num])
        arrive_win_status.append(
            dis_win_status[offset + status_entry_count - 1])
    return [token_win_status, arrive_win_status]


def get_fullmesh_com_win_state(com_win_status: np.ndarray, ep_worldsize: int):
    token_win_status = []
    arrive_win_status = []
    per_ep = UINT32_PER_BLOCK * 2
    for i in range(ep_worldsize):
        offset = i * per_ep
        token_win_status.append(com_win_status[offset])
        offset += UINT32_PER_BLOCK
        arrive_win_status.append(com_win_status[offset])
    return [token_win_status, arrive_win_status]


def get_device_count():
    try:
        import torch
        return torch.npu.device_count()
    except BaseException:
        pass
    import os
    arch = os.uname().machine
    if arch == 'aarch64':
        return 8
    elif arch == 'x86_64':
        return 16
    else:
        logging.error(f"未知架构:{arch}，默认返回8个设备")
        return 8


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
    dump_input_info = pd.DataFrame([[ep_rankid, ep_worldsize, moe_expert_num, bs, k, islayer, d_c, buffer_id,
                                     dis_run_cnts, com_run_cnts,
                                     dis_send_info, dis_recv_info,
                                     dis_inner_info, dis_unfinish_info,
                                     com_send_info, com_recv_info,
                                     com_unfinish_info,
                                     dis_win_state, com_win_state,
                                     expertids.tolist(), expandidx_input, xactivemask_input,
                                     epsendcnt_input]],
                                   columns=['ep_rankid', 'ep_worldsize', 'moe_expert_num', 'bs', 'k', '是否分层', '当前报错算子', '当前buffer_id',
                                            'dispatch执行次数', 'combine执行次数',
                                            'dispatch是否发送及token数', 'dispatch是否接收及token数',
                                            'dispatchInner表发送接收情况', 'dispatch未完成行为,对应前三列',
                                            'combine是否发送及token数', 'combine是否接收及token数',
                                            'combine未完成行为,对应前二列',
                                            'dispatch接收状态,hierarchy为arrivedflag-syncflag-innerflag,fullmesh为statusEntry-firstflag',
                                            'combine接收状态,hierarchy为arrivedflag-syncflag,fullmesh为tokennum-arrivedflag',
                                            'expertids', 'expandidx_input', 'xactivemask_input',
                                            'epsendcnt_input'],
                                   index=[local_rank_id])
    dump_input_info.to_csv(file_name, index=True, mode='a',
                           header=not file_exists, encoding="gbk")


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
    columns = ['ep_rankid', 'ep_worldsize', '是否分层', '出错的算子名', '最早出问题的核', '最早出问题的核执行位置'
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


def get_runpos_analysis_basic(islayer: int, d_c: str):
    # 以下内容根据mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_a2_constant.h定义
    NEXT_POS = [1, 3, len(POS_MAPPING)]
    POS_COMMON_MAPPING = ['UNSTART', 'INIT', 'FINISH']
    if islayer == 0:
        if d_c == "dispatch":
            POS_DIFF_MAPPING = [
                'REORDERTOKEN', 'SEND',
                'WAIT_DISPATCH', 'GET_STATUS_CUM_SUM',
                'LOCAL_WINDOW_COPY']
        else:
            POS_DIFF_MAPPING = ['ALLTOALLDISPATCH', 'WAIT_DISPATCH']
    else:
        if d_c == "dispatch":
            POS_DIFF_MAPPING = [
                'REORDERTOKEN', 'INTER_FINISH',
                'INTRA_FINISH', 'IPC2OUT']
        else:
            POS_DIFF_MAPPING = [
                'GM2IPC', 'IPC2OUT',
                'INTER_SEND', 'INTER_RECV']

    POS_MAPPING = POS_COMMON_MAPPING + POS_DIFF_MAPPING
    next_pos = NEXT_POS[1] + 1
    for _ in range(3, len(POS_MAPPING)):
        NEXT_POS.append(next_pos)
        next_pos += 1
    NEXT_POS.append(2)
    POS_POSITION = get_runpos_sort(NEXT_POS)
    return POS_MAPPING, NEXT_POS, POS_POSITION


def analyze_runpos(runpos: List[int], islayer: int, d_c: str):
    POS_MAPPING, NEXT_POS, POS_POSITION = get_runpos_analysis_basic(
        islayer, d_c)
    core_num = len(runpos)
    run_info = [""] * core_num
    min_run_coreid = []
    min_run_pos = len(POS_MAPPING)
    for core_id in range(core_num):
        cur_run_info = POS_MAPPING[runpos[core_id]]
        next_run_info = POS_MAPPING[NEXT_POS[runpos[core_id]]
                                    ] if NEXT_POS[runpos[core_id]] < len(POS_MAPPING) else ""
        run_info[core_id] = f"{cur_run_info} -> {next_run_info}"
        if min_run_pos > POS_POSITION[runpos[core_id]]:
            min_run_coreid = [core_id]
            min_run_pos = POS_POSITION[runpos[core_id]]
        elif min_run_pos == POS_POSITION[runpos[core_id]]:
            min_run_coreid.append(core_id)
    logging.error("最早出问题的核:%s, 对应代码区域:%s", min_run_coreid,
                  run_info[min_run_coreid[0]])
    return min_run_coreid, run_info


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
    islayer = get_islayer_ep(arr_func, core_num_func, d_c)
    dis_send_info, dis_recv_info, dis_inner_info, dis_unfinish_info = get_send_and_recv_info(
        arr_func_dis, core_num_func, islayer, ep_rankid, "dispatch", ep_worldsize)
    dis_window_status = get_win_state(
        dis_status, "dispatch",
        moe_expert_num, ep_worldsize, islayer)
    if d_c == "combine":
        com_send_info, com_recv_info, _, com_unfinish_info = get_send_and_recv_info(
            arr_func_dis, core_num_func, islayer, ep_rankid, d_c, ep_worldsize)
        com_window_status = get_win_state(
            com_status, d_c, moe_expert_num, ep_worldsize, islayer)
        expandidx_input, xactivemask_input, epsendcnt_input = None, None, None
    else:
        com_send_info, com_recv_info, com_unfinish_info = None, None, None
        com_window_status = None
        expandidx_input, xactivemask_input, epsendcnt_input = get_combine_dumpinfo(
            target_path, bs * k, moe_expert_num, islayer)

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
    save_win_data()
    if d_c == "dispatch":
        all_unfinish_info = dis_unfinish_info
    else:
        all_unfinish_info = com_unfinish_info
    save_win_data(all_unfinish_info, islayer, ep_rankid, ep_worldsize, d_c,
                  min_run_coreid, run_info)

def isA2(soc_version):
    return soc_version == SOC_VERSION_A2

