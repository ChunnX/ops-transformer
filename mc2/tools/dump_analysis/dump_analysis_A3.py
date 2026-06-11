#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MegaMoe A3 Adump Analysis Tool
Parses adump binary data from exception dump and outputs analysis CSV.
"""

import struct
import sys
import os
import csv
import argparse
from dataclasses import dataclass

ADUMP_MAGIC = 0x1234567890abcdef
ADUMP_RESERVED_SIZE = 32 * 1024
ADUMP_PER_CORE_SIZE = 256

STAGE_NAMES = {
    0: "INIT",
    1: "APPLY_XACTIVATE_MASK",
    2: "INIT_ROUTING",
    3: "CROSS_RANK_SYNC_ALLGATHER_PUSH",
    4: "CROSS_RANK_SYNC_ALLGATHER_WAIT",
    5: "GET_CUMSUM",
    6: "NOTIFY_AIC_GMM1",
    7: "PULL_TOKENS",
    8: "EPILOGUE1_WAIT_GMM1",
    9: "EPILOGUE1_PROCESS",
    10: "EPILOGUE1_NOTIFY_GMM2",
    11: "COMBINE_WAIT_GMM2",
    12: "COMBINE_PROCESS",
    13: "RESET",
    14: "CROSS_RANK_SYNC2",
    15: "UNPERMUTE",
}

HEADER_FMT = '<Q Q I I I Q 20Q'
HEADER_SIZE = struct.calcsize(HEADER_FMT)

SHARED_FIELD_FMT = '<I I I I I I I I I I I I I I I I I I I I f'
SHARED_FIELD_SIZE = struct.calcsize(SHARED_FIELD_FMT)

BASIC_INFO_FMT = '<I I Q Q'
BASIC_INFO_SIZE = struct.calcsize(BASIC_INFO_FMT)

STAGE_INFO_FMTS = {
    1: '<I I I I',
    2: '<I I I I',
    3: '<I I',
    4: '<I I I',
    5: '<I',
    6: '<I',
    7: '<I I I I I I I I',
    8: '<I I',
    9: '<I I I I',
    10: '<I',
    11: '<I I',
    12: '<I I I I I I I',
    13: '<I',
    14: '<I I I I',
    15: '<I I I I',
}

STAGE_INFO_NAMES = {
    1: ['totalElements', 'startIdx', 'endIdx', 'hasXActiveMask'],
    2: ['isQuantRouting', 'm', 'topK', 'expertNum'],
    3: ['targetRankId', 'pushCompleted'],
    4: ['targetRankId', 'waitCompleted', 'checkIdx'],
    5: ['isCore0'],
    6: ['syncgmm1Idx'],
    7: ['groupIdx', 'expertPerRank', 'targetRankId', 'rowSrc', 'prevGroupSum', 'currentM', 'rowStart', 'rows'],
    8: ['syncIdx', 'isW4A8'],
    9: ['rowStartThisCore', 'curRowNum', 'syncIdx', 'isW4A8'],
    10: ['syncIdx'],
    11: ['groupIdx', 'flagId'],
    12: ['groupIdx', 'expertPerRank', 'targetRankId', 'isCombineV1', 'srcRowOffset', 'dataRows', 'dstRowOffset'],
    13: ['resetCoreIdx'],
    14: ['targetRankId', 'syncCompleted', 'isV2', 'syncCount'],
    15: ['totalTokens', 'n2', 'topK', 'coreNumUsed'],
}


@dataclass
class AdumpHeader:
    magic: int
    exec_times: int
    version: int
    op_type: int
    stage_num: int
    time_out_cycles: int


@dataclass
class SharedField:
    ccl_buffer_size: int
    world_size: int
    rank_id: int
    aiv_num: int
    m: int
    k: int
    n: int
    is_transpose_w1: int
    is_transpose_w2: int
    has_bias1: int
    has_bias2: int
    expert_num_per_rank: int
    top_k: int
    max_recv_token_num: int
    dispatch_quant_mode: int
    dispatch_quant_out_dtype: int
    combine_quant_mode: int
    comm_alg_code: int
    num_max_tokens_per_rank: int
    activation_code: int
    activation_clamp: float


@dataclass
class BasicInfo:
    aiv_id: int
    current_stage_id: int
    enter_time_stamp: int
    exit_time_stamp: int


def parse_header(data, offset=0):
    fields = struct.unpack_from(HEADER_FMT, data, offset)
    return AdumpHeader(
        magic=fields[0],
        exec_times=fields[1],
        version=fields[2],
        op_type=fields[3],
        stage_num=fields[4],
        time_out_cycles=fields[5],
    ), HEADER_SIZE


def parse_shared_field(data, offset=0):
    fields = struct.unpack_from(SHARED_FIELD_FMT, data, offset)
    return SharedField(
        ccl_buffer_size=fields[0], world_size=fields[1], rank_id=fields[2], aiv_num=fields[3],
        m=fields[4], k=fields[5], n=fields[6],
        is_transpose_w1=fields[7], is_transpose_w2=fields[8],
        has_bias1=fields[9], has_bias2=fields[10],
        expert_num_per_rank=fields[11], top_k=fields[12],
        max_recv_token_num=fields[13],
        dispatch_quant_mode=fields[14], dispatch_quant_out_dtype=fields[15],
        combine_quant_mode=fields[16], comm_alg_code=fields[17],
        num_max_tokens_per_rank=fields[18], activation_code=fields[19],
        activation_clamp=fields[20],
    ), SHARED_FIELD_SIZE


def parse_basic_info(data, offset=0):
    fields = struct.unpack_from(BASIC_INFO_FMT, data, offset)
    return BasicInfo(
        aiv_id=fields[0],
        current_stage_id=fields[1],
        enter_time_stamp=fields[2],
        exit_time_stamp=fields[3],
    ), BASIC_INFO_SIZE


def parse_stage_info(data, offset, stage_id):
    if stage_id not in STAGE_INFO_FMTS:
        return {}, 0
    fmt = STAGE_INFO_FMTS[stage_id]
    names = STAGE_INFO_NAMES[stage_id]
    size = struct.calcsize(fmt)
    fields = struct.unpack_from(fmt, data, offset)
    return dict(zip(names, fields)), size


def parse_adump_bin(bin_path):
    with open(bin_path, 'rb') as f:
        data = f.read()

    if len(data) < HEADER_SIZE:
        print(f"Error: File too small ({len(data)} bytes), expected at least {HEADER_SIZE}")
        return None, None, None

    header, _ = parse_header(data, 0)
    if header.magic != ADUMP_MAGIC:
        print(f"Error: Invalid magic number 0x{header.magic:016x}, expected 0x{ADUMP_MAGIC:016x}")
        return None, None, None

    shared_field, _ = parse_shared_field(data, HEADER_SIZE)

    per_core_offset = HEADER_SIZE + SHARED_FIELD_SIZE
    cores = []
    for i in range(shared_field.aiv_num):
        core_offset = per_core_offset + i * ADUMP_PER_CORE_SIZE
        if core_offset + BASIC_INFO_SIZE > len(data):
            break
        basic_info, basic_size = parse_basic_info(data, core_offset)
        stage_info, _ = parse_stage_info(data, core_offset + basic_size, basic_info.current_stage_id)
        cores.append((basic_info, stage_info))

    return header, shared_field, cores


def dump_to_csv(header, shared_field, cores, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rankId', 'aivId', 'currentStageId', 'currentStageName',
                         'enterTimeStamp', 'exitTimeStamp', 'stageDuration',
                         'worldSize', 'expertNumPerRank', 'topK', 'm', 'k', 'n'])
        for basic_info, stage_info in cores:
            stage_name = STAGE_NAMES.get(basic_info.current_stage_id, f"UNKNOWN_{basic_info.current_stage_id}")
            duration = basic_info.exit_time_stamp - basic_info.enter_time_stamp if basic_info.exit_time_stamp >= basic_info.enter_time_stamp else 0
            row = [shared_field.rank_id, basic_info.aiv_id, basic_info.current_stage_id, stage_name,
                   basic_info.enter_time_stamp, basic_info.exit_time_stamp, duration,
                   shared_field.world_size, shared_field.expert_num_per_rank, shared_field.top_k,
                   shared_field.m, shared_field.k, shared_field.n]
            for key, val in stage_info.items():
                row.append(val)
            writer.writerow(row)


def diagnose_hang(cores, shared_field):
    diagnoses = []
    hang_stages = {}
    for basic_info, stage_info in cores:
        stage_id = basic_info.current_stage_id
        stage_name = STAGE_NAMES.get(stage_id, f"UNKNOWN_{stage_id}")
        if stage_id not in (0, 15):
            if stage_id not in hang_stages:
                hang_stages[stage_id] = []
            hang_stages[stage_id].append((basic_info, stage_info))

    if not hang_stages:
        return [{"rankId": shared_field.rank_id, "diagnosis": "No hang detected - all cores in INIT or completed stages"}]

    for stage_id, core_list in sorted(hang_stages.items()):
        stage_name = STAGE_NAMES.get(stage_id, f"UNKNOWN_{stage_id}")
        for basic_info, stage_info in core_list:
            diagnosis = ""
            if stage_id == 4:
                target_rank = stage_info.get('targetRankId', '?')
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: waiting for rank {target_rank}. Check if rank {target_rank} completed ALLGATHER_PUSH."
            elif stage_id == 8:
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: waiting for AIC GMM1. Check if AIC is stuck or deadlocked."
            elif stage_id == 11:
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: waiting for AIC GMM2. Check if AIC is stuck or deadlocked."
            elif stage_id == 14:
                target_rank = stage_info.get('targetRankId', '?')
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: waiting for rank {target_rank}. Check if rank {target_rank} completed Combine."
            elif stage_id == 3:
                target_rank = stage_info.get('targetRankId', '?')
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: pushing to rank {target_rank}. Check peermem write."
            elif stage_id == 7:
                target_rank = stage_info.get('targetRankId', '?')
                group_idx = stage_info.get('groupIdx', '?')
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: pulling from rank {target_rank}, expert group {group_idx}."
            elif stage_id == 12:
                target_rank = stage_info.get('targetRankId', '?')
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}: writing to rank {target_rank} peermem."
            else:
                diagnosis = f"Core {basic_info.aiv_id} stuck in {stage_name}."
            diagnoses.append({"rankId": shared_field.rank_id, "aivId": basic_info.aiv_id,
                              "stageId": stage_id, "stageName": stage_name, "diagnosis": diagnosis})

    return diagnoses


def diagnose_to_csv(diagnoses, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rankId', 'aivId', 'stageId', 'stageName', 'diagnosis'])
        writer.writeheader()
        writer.writerows(diagnoses)


def main():
    parser = argparse.ArgumentParser(description='MegaMoe A3 Adump Analysis Tool')
    parser.add_argument('input', help='Path to adump bin file or directory containing bin files')
    parser.add_argument('-o', '--output', default='result_A3', help='Output directory (default: result_A3)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    input_path = args.input
    if os.path.isdir(input_path):
        bin_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.bin')]
    else:
        bin_files = [input_path]

    for bin_path in bin_files:
        print(f"Processing: {bin_path}")
        header, shared_field, cores = parse_adump_bin(bin_path)
        if header is None:
            continue

        print(f"  Magic: 0x{header.magic:016x}")
        print(f"  ExecTimes: {header.exec_times}")
        print(f"  RankId: {shared_field.rank_id}, WorldSize: {shared_field.world_size}")
        print(f"  AIV cores: {shared_field.aiv_num}")
        print(f"  M={shared_field.m}, K={shared_field.k}, N={shared_field.n}")
        print(f"  ExpertPerRank: {shared_field.expert_num_per_rank}, TopK: {shared_field.top_k}")

        base_name = os.path.splitext(os.path.basename(bin_path))[0]
        csv_path = os.path.join(args.output, f"dump_data_{base_name}.csv")
        dump_to_csv(header, shared_field, cores, csv_path)
        print(f"  Dump data written to: {csv_path}")

        diagnoses = diagnose_hang(cores, shared_field)
        diag_path = os.path.join(args.output, f"diagnosis_{base_name}.csv")
        diagnose_to_csv(diagnoses, diag_path)
        print(f"  Diagnosis written to: {diag_path}")

        if diagnoses:
            print(f"  === Diagnosis Summary ===")
            for d in diagnoses:
                print(f"    Rank {d['rankId']} Core {d.get('aivId', '?')}: {d['diagnosis']}")


if __name__ == '__main__':
    main()
