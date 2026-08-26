#!/usr/bin/env python3
"""
AllGatherMatmulV2 TilingData / Workspace / RuntimeInfo dump 解析脚本

用法:
  # 解析 tiling data (自动检测量化/非量化)
  python parse_all_gather_matmul_v2_tiling.py tiling_data_*.bin

  # 解析 args_info
  python parse_all_gather_matmul_v2_tiling.py args_info_*.bin --args

  # 解析 WorkspaceLayoutInfo
  python parse_all_gather_matmul_v2_tiling.py tiling_data_*.bin --ws-layout

  # 解析 workspace 段文件
  python parse_all_gather_matmul_v2_tiling.py workspace_seg*_type12_*.bin --workspace --aic-core-num N

  # JSON 输出
  python parse_all_gather_matmul_v2_tiling.py tiling_data_*.bin --json
"""

import argparse
import json
import logging
import os
import struct
import sys

logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

# ====== 结构体大小常量 ======
MC2_INIT_TILING_SIZE = 64
MC2_CC_TILING_SIZE = 512
RCS_TILING_SIZE = 128  # 16*u32(64) + 3*u64(24) + 9*u32(36) + 4(pad to align 8) = 128
ALL_GATHER_MATMUL_TILING_DATA_V2_SIZE = 1640   # 64+512+128+4+4+4+3*236+8+208
ALL_GATHER_MATMUL_TILING_DATA_FP8_SIZE = 2016  # 64+512+128+4+4+4+3*360+4(pad)+8+208
TILING_DUMP_SIZE = 2048

# ====== Workspace 结构体大小常量 ======
WORKSPACE_SEG_INFO_SIZE = 24
MAX_WORKSPACE_SEGMENTS = 8
WORKSPACE_LAYOUT_INFO_SIZE = 16 + MAX_WORKSPACE_SEGMENTS * WORKSPACE_SEG_INFO_SIZE  # 208
RUNTIME_INFO_PER_CORE_SIZE = 512
RUNTIME_INFO_CORE_MAX = 128
RUNTIME_INFO_TOTAL_SIZE = RUNTIME_INFO_CORE_MAX * RUNTIME_INFO_PER_CORE_SIZE  # 65536
RUNTIME_INFO_MAGIC = 0x5A5A5A5A

# ====== 偏移量常量 ======
OFFSET_MC2_INIT_TILING = 0
OFFSET_MC2_CC_TILING = MC2_INIT_TILING_SIZE  # 64
OFFSET_RCS_TILING = MC2_INIT_TILING_SIZE + MC2_CC_TILING_SIZE  # 576
OFFSET_QUANT_MODE = OFFSET_RCS_TILING + RCS_TILING_SIZE + 4 + 4  # 576 + 120 + 4 + 4 = 704
OFFSET_WS_LAYOUT_V2 = ALL_GATHER_MATMUL_TILING_DATA_V2_SIZE - WORKSPACE_LAYOUT_INFO_SIZE  # 1424
OFFSET_WS_LAYOUT_FP8 = ALL_GATHER_MATMUL_TILING_DATA_FP8_SIZE - WORKSPACE_LAYOUT_INFO_SIZE  # 1796

# args_info_*.bin 中 kernel 参数索引
ARGS_INDEX_NAMES = [
    "aGM",            # [0]
    "bGM",            # [1]
    "biasGM",         # [2]
    "scaleInv1",      # [3]
    "scaleInv2",      # [4]
    "scale",          # [5]
    "cGM",            # [6]
    "gatherOut",      # [7]
    "amax",           # [8]
    "workspaceGM",    # [9]
    "tilingGM",       # [10]
]

# PosTag 映射 (AllGatherMatmulV2 算子)
POS_TAG_MAP_AGMMV2 = {
    0: "COMM_BEFORE",
    1: "COMP_LOCAL_BEFORE",
    2: "COMP_GATHER_BEFORE",
    3: "FINALIZE_BEFORE",
}

# RuntimePhase 映射
RUNTIME_PHASE_MAP = {
    0: "COMM_INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "COMM_FINALIZE",
}

# Workspace 段类型映射
WS_SEG_TYPE_MAP = {
    0: "LIB_API",
    1: "ND2NZ",
    2: "GATHER",
    3: "GATHER_SCALE1",
    4: "GATHER_SCALE",
    5: "COMM_OUT",
    6: "PERMUTE_OUT",
    7: "BIAS",
    8: "MM_RESULT",
    9: "RECV_BUF",
    10: "COMM_INT8",
    11: "DYNAMIC_QUANT",
    12: "RUNTIME_INFO",
    255: "RESERVED",
}


def safe_str(raw: bytes) -> str:
    idx = raw.find(b"\x00")
    if idx >= 0:
        raw = raw[:idx]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def parse_mc2_init_tiling(data: bytes, offset: int) -> dict:
    if offset + MC2_INIT_TILING_SIZE > len(data):
        return {"error": "data too short"}
    return {"size": MC2_INIT_TILING_SIZE}


def parse_mc2_cc_tiling(data: bytes, offset: int) -> dict:
    if offset + MC2_CC_TILING_SIZE > len(data):
        return {"error": "data too short"}
    skip_local, skip_buffer, step_size, version = struct.unpack_from("<4B", data, offset)
    comm_engine = struct.unpack_from("<B", data, offset + 13)[0]
    src_data_type = struct.unpack_from("<B", data, offset + 14)[0]
    dst_data_type = struct.unpack_from("<B", data, offset + 15)[0]
    group_name = safe_str(data[offset + 16: offset + 16 + 128])
    alg_config = safe_str(data[offset + 144: offset + 144 + 128])
    op_type, reduce_type = struct.unpack_from("<2I", data, offset + 272)
    return {
        "skipLocalRankCopy": skip_local,
        "skipBufferWindowCopy": skip_buffer,
        "stepSize": step_size,
        "version": version,
        "commEngine": comm_engine,
        "srcDataType": src_data_type,
        "dstDataType": dst_data_type,
        "groupName": group_name,
        "algConfig": alg_config,
        "opType": op_type,
        "reduceType": reduce_type,
    }


def parse_rcs_tiling(data: bytes, offset: int) -> dict:
    if offset + RCS_TILING_SIZE > len(data):
        return {"error": "data too short for RCSTiling"}
    u32_fields = struct.unpack_from("<16I", data, offset)
    u64_fields = struct.unpack_from("<3Q", data, offset + 64)
    u32_tail = struct.unpack_from("<9I", data, offset + 88)
    return {
        "rankDim": u32_fields[0],
        "rankID": u32_fields[1],
        "commtype": u32_fields[2],
        "subtype": u32_fields[3],
        "tileCnt": u32_fields[4],
        "tailM": u32_fields[5],
        "tailCnt": u32_fields[6],
        "biasLen": u32_fields[7],
        "isAdd": u32_fields[8],
        "rankM": u32_fields[9],
        "rankN": u32_fields[10],
        "rankK": u32_fields[11],
        "gatherIndex": u32_fields[12],
        "isTransposeA": u32_fields[13],
        "isTransposeB": u32_fields[14],
        "storageGather": u32_fields[15],
        "nd2NzWorkLen": u64_fields[0],
        "cToFloatLen": u64_fields[1],
        "gatherLen": u64_fields[2],
        "workspaceAddr4": u32_tail[0],
        "aicCoreNum": u32_tail[1],
        "needUbBuffer": u32_tail[2],
        "addX3UbCnt": u32_tail[3],
        "commWorkSpaceSize": u32_tail[4],
        "isInputCommQuantScale": u32_tail[5],
        "dataType": u32_tail[6],
        "commInt8WorkSpace": u32_tail[7],
        "dynamicQuantTempBuffSize": u32_tail[8],
    }


def detect_quant_mode(data: bytes) -> bool:
    if len(data) < OFFSET_QUANT_MODE + 4:
        return False
    quant_mode = struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0]
    return quant_mode != 0


def parse_non_quant_tiling_data(data: bytes) -> dict:
    return {
        "mc2InitTiling": parse_mc2_init_tiling(data, 0),
        "mc2CcTiling": parse_mc2_cc_tiling(data, MC2_INIT_TILING_SIZE),
        "param": parse_rcs_tiling(data, OFFSET_RCS_TILING),
        "dataType": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE)[0],
        "debugMode": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE + 4)[0],
        "quantMode": struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0],
        "tilingKey": struct.unpack_from("<Q", data, OFFSET_WS_LAYOUT_V2 - 8)[0],
    }


def parse_quant_tiling_data(data: bytes) -> dict:
    return {
        "mc2InitTiling": parse_mc2_init_tiling(data, 0),
        "mc2CcTiling": parse_mc2_cc_tiling(data, MC2_INIT_TILING_SIZE),
        "param": parse_rcs_tiling(data, OFFSET_RCS_TILING),
        "dataType": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE)[0],
        "debugMode": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE + 4)[0],
        "quantMode": struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0],
        "tilingKey": struct.unpack_from("<Q", data, OFFSET_WS_LAYOUT_FP8 - 8)[0],
    }


def parse_workspace_seg_info(data: bytes, offset: int) -> dict:
    if offset + WORKSPACE_SEG_INFO_SIZE > len(data):
        return {"error": "data too short"}
    seg_offset, seg_size, seg_type = struct.unpack_from("<QQB", data, offset)
    return {
        "offset": seg_offset,
        "size": seg_size,
        "type": seg_type,
        "typeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})"),
    }


def parse_workspace_layout_info(data: bytes, offset: int) -> dict:
    if offset + WORKSPACE_LAYOUT_INFO_SIZE > len(data):
        return {"error": f"data too short for WorkspaceLayoutInfo"}
    total_size, seg_count, reserved = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for i in range(min(seg_count, MAX_WORKSPACE_SEGMENTS)):
        seg_offset = offset + 16 + i * WORKSPACE_SEG_INFO_SIZE
        segments.append(parse_workspace_seg_info(data, seg_offset))
    return {
        "totalSize": total_size,
        "segCount": seg_count,
        "segments": segments,
    }


def parse_runtime_info_per_core(data: bytes, offset: int) -> dict:
    if offset + RUNTIME_INFO_PER_CORE_SIZE > len(data):
        return {"error": "data too short"}
    magic_num, core_id = struct.unpack_from("<IH", data, offset)
    exec_turn, exec_position, comm_phase, commit_count, wait_count = struct.unpack_from("<5B", data, offset + 6)
    return {
        "magicNum": magic_num,
        "magicValid": magic_num == RUNTIME_INFO_MAGIC,
        "coreId": core_id,
        "execTurn": exec_turn,
        "execPosition": exec_position,
        "execPositionName": POS_TAG_MAP_AGMMV2.get(exec_position, f"UNKNOWN({exec_position})"),
        "commPhase": comm_phase,
        "commPhaseName": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "commCommitCount": commit_count,
        "commWaitCount": wait_count,
    }


def parse_runtime_info_segment(data: bytes, aic_core_num=None) -> dict:
    max_slots = len(data) // RUNTIME_INFO_PER_CORE_SIZE
    slots = []
    active_count = 0
    for i in range(max_slots):
        offset = i * RUNTIME_INFO_PER_CORE_SIZE
        slot = parse_runtime_info_per_core(data, offset)
        slot["slotIdx"] = i
        if slot.get("magicValid", False):
            active_count += 1
            if aic_core_num is not None:
                slot["coreType"] = "C核" if slot["coreId"] < aic_core_num else "V核"
            else:
                slot["coreType"] = "未知"
            slots.append(slot)
        else:
            slots.append(slot)
    return {
        "segmentSize": len(data),
        "maxSlots": max_slots,
        "activeSlots": active_count,
        "aicCoreNum": aic_core_num,
        "slots": slots,
    }


def parse_workspace_segment_file(data: bytes, seg_type: int, aic_core_num=None) -> dict:
    type_name = WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})")
    result = {
        "segmentType": seg_type,
        "segmentTypeName": type_name,
        "fileSize": len(data),
    }
    if seg_type == 12:
        rt_info = parse_runtime_info_segment(data, aic_core_num)
        result["runtimeInfo"] = rt_info
    else:
        result["hexPreview"] = data[:256].hex()
        result["note"] = f"段类型 {type_name}，原始二进制数据。"
    return result


def detect_workspace_seg_from_filename(filename: str):
    import re
    basename = os.path.basename(filename)
    m = re.match(r"workspace_seg(\d+)_type(\d+)_", basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def parse_args_info(data: bytes) -> dict:
    num_valid_args = len(ARGS_INDEX_NAMES)
    num_ptrs = min(num_valid_args, len(data) // 8)
    ptrs = struct.unpack_from(f"<{num_ptrs}Q", data, 0)
    args = {}
    for i in range(num_ptrs):
        name = ARGS_INDEX_NAMES[i]
        args[f"[{i}] {name}"] = f"0x{ptrs[i]:016x}"
    args["_dumpFileSize"] = len(data)
    args["_validArgCount"] = num_ptrs
    return args


def print_section(title: str):
    logging.info("=" * 70)
    logging.info("  %s", title)
    logging.info("=" * 70)


def print_kv(title: str, d: dict, indent: int = 0):
    prefix = "  " * indent
    logging.info("%s--- %s ---", prefix, title)
    for k, v in d.items():
        if isinstance(v, dict):
            print_kv(k, v, indent + 1)
        elif isinstance(v, list):
            logging.info("%s  %-30s: %s", prefix, k, v)
        else:
            logging.info("%s  %-30s: %s", prefix, k, v)


def print_tiling_data(result: dict, is_quant: bool):
    print_section("Mc2InitTiling (HCCL Init, 64B)")
    print_kv("mc2InitTiling", result["mc2InitTiling"])

    print_section("Mc2CcTiling (HCCL CC, 512B)")
    print_kv("mc2CcTiling", result["mc2CcTiling"])

    print_section(f"RCSTiling ({RCS_TILING_SIZE}B)  [{'QUANT' if is_quant else 'NON-QUANT'}]")
    print_kv("param", result["param"])

    print_section("Extra Fields")
    print_kv("extra", {
        "dataType": result["dataType"],
        "debugMode": result["debugMode"],
        "quantMode": result["quantMode"],
        "tilingKey": result["tilingKey"],
    })


def print_args_info(result: dict):
    print_section("Args Info (kernel GM pointer)")
    dump_file_size = result.pop("_dumpFileSize", 0)
    valid_arg_count = result.pop("_validArgCount", 0)
    for name, addr in result.items():
        if name.startswith("_"):
            continue
        logging.info("  %-30s: %s", name, addr)
    logging.info("")
    logging.info("  有效 args 数量: %d", valid_arg_count)
    logging.info("  dump 文件大小: %d bytes", dump_file_size)


def print_workspace_layout_info(ws_layout: dict):
    if "error" in ws_layout:
        logging.error("  %s", ws_layout["error"])
        return
    print_section("WorkspaceLayoutInfo (208B)")
    logging.info("  %-30s: %s bytes", "totalSize", ws_layout["totalSize"])
    logging.info("  %-30s: %s", "segCount", ws_layout["segCount"])
    logging.info("")
    logging.info("  %-4s  %-12s  %-12s  %-6s  %s", "Idx", "Offset", "Size", "Type", "TypeName")
    logging.info("  " + "-" * 66)
    for i, seg in enumerate(ws_layout["segments"]):
        if "error" in seg:
            logging.error("  [%d]  %s", i, seg["error"])
            continue
        logging.info("  [%d]  0x%010x  %10d  %-6d  %s",
                     i, seg["offset"], seg["size"], seg["type"], seg["typeName"])


def print_runtime_info_segment(rt_info: dict):
    print_section("RuntimeInfo Segment (type=12, 64KB)")
    logging.info("  %-30s: %d bytes", "segmentSize", rt_info["segmentSize"])
    logging.info("  %-30s: %d / %d", "activeSlots (magic matched)", rt_info["activeSlots"], rt_info["maxSlots"])
    if rt_info.get("aicCoreNum") is None:
        logging.warning("  未提供 --aic-core-num，C/V 核分类不可靠（显示为'未知'）。")
    else:
        logging.info("  %-30s: %d (coreId < %d 为 C核, 否则为 V核)",
                     "aicCoreNum", rt_info["aicCoreNum"], rt_info["aicCoreNum"])
    logging.info("")
    logging.info("  --- 仅显示 magicNum 校验通过的 slot (0x%08X) ---", RUNTIME_INFO_MAGIC)
    active_slots = [s for s in rt_info["slots"] if s.get("magicValid", False)]
    if not active_slots:
        logging.warning("  无有效 slot，可能 runtime info 未被写入。")
    else:
        for slot in active_slots:
            core_type = slot.get("coreType", "未知")
            logging.info("  Core %-4d (%s):  turn=%d, pos=%d (%s), phase=%s, commit=%d, wait=%d",
                         slot["coreId"], core_type,
                         slot["execTurn"], slot["execPosition"], slot["execPositionName"],
                         slot["commPhaseName"],
                         slot["commCommitCount"], slot["commWaitCount"])


def print_workspace_segment(result: dict):
    seg_type = result["segmentType"]
    type_name = result["segmentTypeName"]
    print_section(f"Workspace Segment (type={seg_type}/{type_name}, {result['fileSize']}B)")
    if seg_type == 12:
        print_runtime_info_segment(result["runtimeInfo"])
    else:
        logging.info("  段类型: %d (%s)", seg_type, type_name)
        logging.info("  文件大小: %d bytes", result["fileSize"])
        logging.info("  Hex 预览 (前 256 字节):")
        hex_str = result.get("hexPreview", "")
        for i in range(0, min(len(hex_str), 512), 64):
            offset_hex = f"{i // 2:04x}"
            chunk = hex_str[i:i + 64]
            formatted = " ".join(chunk[j:j + 2] for j in range(0, len(chunk), 2))
            logging.info("    %s: %s", offset_hex, formatted)


def main():
    parser = argparse.ArgumentParser(
        description="AllGatherMatmulV2 TilingData / Workspace / RuntimeInfo dump 解析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bin_file", help="dump 产生的 bin 文件路径")
    parser.add_argument("--args", action="store_true", help="解析 args_info_*.bin")
    parser.add_argument("--ws-layout", action="store_true", help="解析 WorkspaceLayoutInfo")
    parser.add_argument("--workspace", action="store_true", help="解析 workspace 段文件")
    parser.add_argument("--aic-core-num", type=int, metavar="N", default=None,
                        help="AIC 核数，用于区分 C/V 核")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()

    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    if args.workspace:
        seg_info = detect_workspace_seg_from_filename(args.bin_file)
        if seg_info is None:
            logging.error("文件名不匹配 workspace_seg{idx}_type{type}_* 模式")
            sys.exit(1)
        seg_idx, seg_type = seg_info
        logging.info("检测到 workspace 段文件: segIdx=%d, segType=%d (%s)",
                     seg_idx, seg_type, WS_SEG_TYPE_MAP.get(seg_type, "UNKNOWN"))
        if args.aic_core_num is not None:
            logging.info("使用 --aic-core-num=%d 区分 C/V 核", args.aic_core_num)
        result = parse_workspace_segment_file(data, seg_type, args.aic_core_num)
        result["segIdx"] = seg_idx
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            print_workspace_segment(result)
        return

    if args.args:
        result = parse_args_info(data)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_args_info(result)
        return

    if args.ws_layout:
        is_quant = detect_quant_mode(data)
        ws_offset = OFFSET_WS_LAYOUT_FP8 if is_quant else OFFSET_WS_LAYOUT_V2
        mode_str = "量化" if is_quant else "非量化"
        logging.info("解析 WorkspaceLayoutInfo (%s模式, offset=%d)", mode_str, ws_offset)
        ws_layout = parse_workspace_layout_info(data, ws_offset)
        if "error" in ws_layout:
            logging.error("解析失败: %s", ws_layout["error"])
            sys.exit(1)
        if args.json:
            print(json.dumps(ws_layout, indent=2, ensure_ascii=False, default=str))
        else:
            print_workspace_layout_info(ws_layout)
        return

    # 默认: tiling_data 解析
    is_quant = detect_quant_mode(data)
    mode_str = "量化" if is_quant else "非量化"
    logging.info("解析模式: %s (quantMode=%d)", mode_str, struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0])

    if is_quant:
        result = parse_quant_tiling_data(data)
        logging.info("结构体 AllGatherMatmulTilingDataFp8: %dB", ALL_GATHER_MATMUL_TILING_DATA_FP8_SIZE)
    else:
        result = parse_non_quant_tiling_data(data)
        logging.info("结构体 AllGatherMatmulTilingDataV2: %dB", ALL_GATHER_MATMUL_TILING_DATA_V2_SIZE)

    # 尝试解析尾部 WorkspaceLayoutInfo
    ws_offset = OFFSET_WS_LAYOUT_FP8 if is_quant else OFFSET_WS_LAYOUT_V2
    if len(data) >= ws_offset + WORKSPACE_LAYOUT_INFO_SIZE:
        ws_layout = parse_workspace_layout_info(data, ws_offset)
        if "error" not in ws_layout and ws_layout.get("segCount", 0) > 0:
            result["workspaceLayout"] = ws_layout
            logging.info("检测到 WorkspaceLayoutInfo (offset=%d, segCount=%d)", ws_offset, ws_layout["segCount"])

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print_tiling_data(result, is_quant)
        if "workspaceLayout" in result:
            print_workspace_layout_info(result["workspaceLayout"])


if __name__ == "__main__":
    main()
