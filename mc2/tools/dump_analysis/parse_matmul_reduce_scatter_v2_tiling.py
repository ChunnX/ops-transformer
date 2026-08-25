#!/usr/bin/env python3
"""
MatmulReduceScatterV2 TilingData / Workspace / RuntimeInfo dump 解析脚本

用法:
  python parse_matmul_reduce_scatter_v2_tiling.py tiling_data_*.bin
  python parse_matmul_reduce_scatter_v2_tiling.py args_info_*.bin --args
  python parse_matmul_reduce_scatter_v2_tiling.py tiling_data_*.bin --ws-layout
  python parse_matmul_reduce_scatter_v2_tiling.py workspace_seg*_type12_*.bin --workspace --aic-core-num N
  python parse_matmul_reduce_scatter_v2_tiling.py tiling_data_*.bin --json
"""

import argparse
import json
import logging
import os
import struct
import sys

logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

MC2_INIT_TILING_SIZE = 64
MC2_CC_TILING_SIZE = 512
RCS_TILING_SIZE = 120
MRSV2_TILING_DATA_SIZE = 1396
QUANT_MRSV2_TILING_DATA_SIZE = 1644
TILING_DUMP_SIZE = 2048

WORKSPACE_SEG_INFO_SIZE = 24
MAX_WORKSPACE_SEGMENTS = 8
WORKSPACE_LAYOUT_INFO_SIZE = 16 + MAX_WORKSPACE_SEGMENTS * WORKSPACE_SEG_INFO_SIZE
RUNTIME_INFO_PER_CORE_SIZE = 512
RUNTIME_INFO_CORE_MAX = 128
RUNTIME_INFO_TOTAL_SIZE = RUNTIME_INFO_CORE_MAX * RUNTIME_INFO_PER_CORE_SIZE
RUNTIME_INFO_MAGIC = 0x5A5A5A5A

OFFSET_RCS_TILING = MC2_INIT_TILING_SIZE + MC2_CC_TILING_SIZE  # 576
OFFSET_QUANT_MODE = OFFSET_RCS_TILING + RCS_TILING_SIZE + 4 + 4  # 704
OFFSET_WS_LAYOUT_NON_QUANT = MRSV2_TILING_DATA_SIZE - WORKSPACE_LAYOUT_INFO_SIZE  # 1188
OFFSET_WS_LAYOUT_QUANT = QUANT_MRSV2_TILING_DATA_SIZE - WORKSPACE_LAYOUT_INFO_SIZE  # 1436

ARGS_INDEX_NAMES = [
    "aGM",            # [0]
    "bGM",            # [1]
    "biasGM",         # [2]
    "x1ScaleGM",      # [3]
    "x2ScaleGM",      # [4]
    "quantScaleGM",   # [5]
    "cGM",            # [6]
    "amaxOutGM",      # [7]
    "workspaceGM",    # [8]
    "tilingGM",       # [9]
]

POS_TAG_MAP_MRSV2 = {
    0: "COMM_BEFORE",
    1: "COMP_MATMUL_BEFORE",
    2: "REDUCE_BEFORE",
    3: "FINALIZE_BEFORE",
}

RUNTIME_PHASE_MAP = {
    0: "COMM_INIT",
    1: "COMM_PREPARE",
    2: "COMM_COMMIT",
    3: "COMM_WAIT",
    4: "COMM_FINALIZE",
}

WS_SEG_TYPE_MAP = {
    0: "LIB_API", 1: "ND2NZ", 2: "GATHER", 3: "GATHER_SCALE1", 4: "GATHER_SCALE",
    5: "COMM_OUT", 6: "PERMUTE_OUT", 7: "BIAS", 8: "MM_RESULT", 9: "RECV_BUF",
    10: "COMM_INT8", 11: "DYNAMIC_QUANT", 12: "RUNTIME_INFO", 255: "RESERVED",
}


def safe_str(raw: bytes) -> str:
    idx = raw.find(b"\x00")
    if idx >= 0:
        raw = raw[:idx]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def parse_mc2_init_tiling(data, offset):
    return {"size": MC2_INIT_TILING_SIZE} if offset + MC2_INIT_TILING_SIZE <= len(data) else {"error": "too short"}


def parse_mc2_cc_tiling(data, offset):
    if offset + MC2_CC_TILING_SIZE > len(data):
        return {"error": "too short"}
    skip_local, skip_buffer, step_size, version = struct.unpack_from("<4B", data, offset)
    comm_engine = struct.unpack_from("<B", data, offset + 13)[0]
    group_name = safe_str(data[offset + 16: offset + 16 + 128])
    alg_config = safe_str(data[offset + 144: offset + 144 + 128])
    op_type, reduce_type = struct.unpack_from("<2I", data, offset + 272)
    return {
        "skipLocalRankCopy": skip_local, "skipBufferWindowCopy": skip_buffer,
        "stepSize": step_size, "version": version, "commEngine": comm_engine,
        "groupName": group_name, "algConfig": alg_config, "opType": op_type, "reduceType": reduce_type,
    }


def parse_rcs_tiling(data, offset):
    if offset + RCS_TILING_SIZE > len(data):
        return {"error": "too short"}
    u32_head = struct.unpack_from("<16I", data, offset)
    u64_mid = struct.unpack_from("<3Q", data, offset + 64)
    u32_tail = struct.unpack_from("<8I", data, offset + 88)
    return {
        "rankDim": u32_head[0], "rankID": u32_head[1], "tileCnt": u32_head[4],
        "tailM": u32_head[5], "tailCnt": u32_head[6], "rankM": u32_head[9],
        "rankN": u32_head[10], "rankK": u32_head[11], "aicCoreNum": u32_tail[1],
        "gatherLen": u64_mid[2],
    }


def detect_quant_mode(data):
    if len(data) < OFFSET_QUANT_MODE + 4:
        return False
    return struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0] != 0


def parse_non_quant_tiling_data(data):
    return {
        "mc2InitTiling": parse_mc2_init_tiling(data, 0),
        "mc2CcTiling": parse_mc2_cc_tiling(data, MC2_INIT_TILING_SIZE),
        "param": parse_rcs_tiling(data, OFFSET_RCS_TILING),
        "dataType": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE)[0],
        "debugMode": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE + 4)[0],
        "quantMode": struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0],
        "tilingKey": struct.unpack_from("<Q", data, OFFSET_WS_LAYOUT_NON_QUANT - 8)[0],
    }


def parse_quant_tiling_data(data):
    return {
        "mc2InitTiling": parse_mc2_init_tiling(data, 0),
        "mc2CcTiling": parse_mc2_cc_tiling(data, MC2_INIT_TILING_SIZE),
        "param": parse_rcs_tiling(data, OFFSET_RCS_TILING),
        "dataType": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE)[0],
        "debugMode": struct.unpack_from("<I", data, OFFSET_RCS_TILING + RCS_TILING_SIZE + 4)[0],
        "quantMode": struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0],
        "tilingKey": struct.unpack_from("<Q", data, OFFSET_WS_LAYOUT_QUANT - 8)[0],
    }


def parse_workspace_seg_info(data, offset):
    if offset + WORKSPACE_SEG_INFO_SIZE > len(data):
        return {"error": "too short"}
    seg_offset, seg_size, seg_type = struct.unpack_from("<QQB", data, offset)
    return {"offset": seg_offset, "size": seg_size, "type": seg_type,
            "typeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})")}


def parse_workspace_layout_info(data, offset):
    if offset + WORKSPACE_LAYOUT_INFO_SIZE > len(data):
        return {"error": "too short"}
    total_size, seg_count, _ = struct.unpack_from("<Q2I", data, offset)
    segments = []
    for i in range(min(seg_count, MAX_WORKSPACE_SEGMENTS)):
        segments.append(parse_workspace_seg_info(data, offset + 16 + i * WORKSPACE_SEG_INFO_SIZE))
    return {"totalSize": total_size, "segCount": seg_count, "segments": segments}


def parse_runtime_info_per_core(data, offset):
    if offset + RUNTIME_INFO_PER_CORE_SIZE > len(data):
        return {"error": "too short"}
    magic_num, core_id = struct.unpack_from("<IH", data, offset)
    exec_turn, exec_position, comm_phase, commit_count, wait_count = struct.unpack_from("<5B", data, offset + 6)
    return {
        "magicNum": magic_num, "magicValid": magic_num == RUNTIME_INFO_MAGIC, "coreId": core_id,
        "execTurn": exec_turn, "execPosition": exec_position,
        "execPositionName": POS_TAG_MAP_MRSV2.get(exec_position, f"UNKNOWN({exec_position})"),
        "commPhase": comm_phase, "commPhaseName": RUNTIME_PHASE_MAP.get(comm_phase, f"UNKNOWN({comm_phase})"),
        "commCommitCount": commit_count, "commWaitCount": wait_count,
    }


def parse_runtime_info_segment(data, aic_core_num=None):
    max_slots = len(data) // RUNTIME_INFO_PER_CORE_SIZE
    slots = []
    active_count = 0
    for i in range(max_slots):
        slot = parse_runtime_info_per_core(data, i * RUNTIME_INFO_PER_CORE_SIZE)
        slot["slotIdx"] = i
        if slot.get("magicValid", False):
            active_count += 1
            slot["coreType"] = "C核" if aic_core_num is not None and slot["coreId"] < aic_core_num else ("V核" if aic_core_num is not None else "未知")
            slots.append(slot)
        else:
            slots.append(slot)
    return {"segmentSize": len(data), "maxSlots": max_slots, "activeSlots": active_count,
            "aicCoreNum": aic_core_num, "slots": slots}


def parse_workspace_segment_file(data, seg_type, aic_core_num=None):
    result = {"segmentType": seg_type,
              "segmentTypeName": WS_SEG_TYPE_MAP.get(seg_type, f"UNKNOWN({seg_type})"),
              "fileSize": len(data)}
    if seg_type == 12:
        result["runtimeInfo"] = parse_runtime_info_segment(data, aic_core_num)
    else:
        result["hexPreview"] = data[:256].hex()
    return result


def detect_workspace_seg_from_filename(filename):
    import re
    m = re.match(r"workspace_seg(\d+)_type(\d+)_", os.path.basename(filename))
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_args_info(data):
    num_ptrs = min(len(ARGS_INDEX_NAMES), len(data) // 8)
    ptrs = struct.unpack_from(f"<{num_ptrs}Q", data, 0)
    args = {f"[{i}] {ARGS_INDEX_NAMES[i]}": f"0x{ptrs[i]:016x}" for i in range(num_ptrs)}
    args["_dumpFileSize"] = len(data)
    args["_validArgCount"] = num_ptrs
    return args


def print_section(title):
    logging.info("=" * 70)
    logging.info("  %s", title)
    logging.info("=" * 70)


def print_kv(title, d, indent=0):
    prefix = "  " * indent
    logging.info("%s--- %s ---", prefix, title)
    for k, v in d.items():
        if isinstance(v, dict):
            print_kv(k, v, indent + 1)
        else:
            logging.info("%s  %-30s: %s", prefix, k, v)


def print_tiling_data(result, is_quant):
    print_section("Mc2InitTiling (HCCL Init, 64B)")
    print_kv("mc2InitTiling", result["mc2InitTiling"])
    print_section("Mc2CcTiling (HCCL CC, 512B)")
    print_kv("mc2CcTiling", result["mc2CcTiling"])
    print_section(f"RCSTiling ({RCS_TILING_SIZE}B)  [{'QUANT' if is_quant else 'NON-QUANT'}]")
    print_kv("param", result["param"])
    print_section("Extra Fields")
    print_kv("extra", {"dataType": result["dataType"], "debugMode": result["debugMode"],
                        "quantMode": result["quantMode"], "tilingKey": result["tilingKey"]})


def print_runtime_info_segment(rt_info):
    print_section("RuntimeInfo Segment (type=12, 64KB)")
    logging.info("  %-30s: %d bytes", "segmentSize", rt_info["segmentSize"])
    logging.info("  %-30s: %d / %d", "activeSlots", rt_info["activeSlots"], rt_info["maxSlots"])
    if rt_info.get("aicCoreNum") is None:
        logging.warning("  未提供 --aic-core-num，C/V 核分类不可靠。")
    else:
        logging.info("  %-30s: %d", "aicCoreNum", rt_info["aicCoreNum"])
    logging.info("")
    logging.info("  --- 仅显示 magicNum 校验通过的 slot (0x%08X) ---", RUNTIME_INFO_MAGIC)
    active = [s for s in rt_info["slots"] if s.get("magicValid", False)]
    if not active:
        logging.warning("  无有效 slot。")
    else:
        for s in active:
            logging.info("  Core %-4d (%s):  turn=%d, pos=%d (%s), phase=%s, commit=%d, wait=%d",
                         s["coreId"], s.get("coreType", "未知"),
                         s["execTurn"], s["execPosition"], s["execPositionName"],
                         s["commPhaseName"], s["commCommitCount"], s["commWaitCount"])


def print_workspace_segment(result):
    seg_type = result["segmentType"]
    print_section(f"Workspace Segment (type={seg_type}/{result['segmentTypeName']}, {result['fileSize']}B)")
    if seg_type == 12:
        print_runtime_info_segment(result["runtimeInfo"])
    else:
        logging.info("  Hex: %s", result.get("hexPreview", "")[:128])


def main():
    parser = argparse.ArgumentParser(description="MatmulReduceScatterV2 dump 解析脚本")
    parser.add_argument("bin_file", help="bin 文件路径")
    parser.add_argument("--args", action="store_true")
    parser.add_argument("--ws-layout", action="store_true")
    parser.add_argument("--workspace", action="store_true")
    parser.add_argument("--aic-core-num", type=int, metavar="N", default=None)
    parser.add_argument("--json", action="store_true")
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
            print_section("Args Info")
            for k, v in result.items():
                if not k.startswith("_"):
                    logging.info("  %-30s: %s", k, v)
        return

    if args.ws_layout:
        is_quant = detect_quant_mode(data)
        ws_offset = OFFSET_WS_LAYOUT_QUANT if is_quant else OFFSET_WS_LAYOUT_NON_QUANT
        ws_layout = parse_workspace_layout_info(data, ws_offset)
        if args.json:
            print(json.dumps(ws_layout, indent=2, ensure_ascii=False, default=str))
        else:
            print_section("WorkspaceLayoutInfo")
            print_kv("layout", ws_layout)
        return

    is_quant = detect_quant_mode(data)
    logging.info("解析模式: %s (quantMode=%d)", "量化" if is_quant else "非量化",
                 struct.unpack_from("<I", data, OFFSET_QUANT_MODE)[0])
    result = parse_quant_tiling_data(data) if is_quant else parse_non_quant_tiling_data(data)
    ws_offset = OFFSET_WS_LAYOUT_QUANT if is_quant else OFFSET_WS_LAYOUT_NON_QUANT
    if len(data) >= ws_offset + WORKSPACE_LAYOUT_INFO_SIZE:
        ws_layout = parse_workspace_layout_info(data, ws_offset)
        if "error" not in ws_layout and ws_layout.get("segCount", 0) > 0:
            result["workspaceLayout"] = ws_layout
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print_tiling_data(result, is_quant)
        if "workspaceLayout" in result:
            print_section("WorkspaceLayoutInfo")
            print_kv("layout", result["workspaceLayout"])


if __name__ == "__main__":
    main()
