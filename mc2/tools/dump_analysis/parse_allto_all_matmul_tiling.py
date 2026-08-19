#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.

# !
#  \file parse_allto_all_matmul_tiling.py
#  \brief AlltoAllMatmul TilingData 解析脚本。
#         解析异常 dump 产生的 tiling_data_*.bin 或 args_info_*.bin 文件。
#
# 用法:
#   python parse_allto_all_matmul_tiling.py <bin_file> [--quant] [--args] [--json]
#
# 模式:
#   默认      解析 tiling_data_*.bin（非量化 AlltoAllMatmulTilingData, 1136B）
#   --quant   解析 tiling_data_*.bin（量化 AlltoAllQuantMatmulTilingData, 1384B）
#   --args    解析 args_info_*.bin（16 个 GM 地址指针, 256B）
#   --json    以 JSON 格式输出（默认为人类可读表格）

import sys
import os
import struct
import json
import argparse
import logging

logging.basicConfig(level=logging.NOTSET, format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

# ====== 结构体大小常量（与 arch35 C++ 定义保持一致）======
MC2_INIT_TILING_SIZE = 64
MC2_CC_TILING_SIZE = 512
ALLTO_ALL_MATMUL_TILING_INFO_SIZE = 96  # 10*u32(40) + quantMode(4) + reserved(4) + 6*u64(48) = 96
TCUBE_TILING_SIZE = 200
MC2_MM_V3_TILING_DATA_SIZE = 236  # TCubeTiling(200) + 9*uint32(36)
MC2_QUANT_BMM_TILING_PARAMS_SIZE = 360  # params(128) + TCubeTiling(200) + L2cache(24) + slidingWindow(8)

ALLTO_ALL_MATMUL_TILING_DATA_SIZE = 1144  # 非量化: 64+512+96+236+236
ALLTO_ALL_QUANT_MATMUL_TILING_DATA_SIZE = 1392  # 量化: 64+512+96+360+360
TILING_DUMP_SIZE = 2048  # dump 固定大小

# 量化模式枚举（x1QuantMode / x2QuantMode，与 op_proto.h 保持一致）
# x1QuantMode: 0=非量化, 7=动态pertoken量化(KC), 6=MX量化
# x2QuantMode: 0=非量化, 2=perchannel量化(KC), 6=MX量化
X1_QUANT_MODE_MAP = {
    0: "NON_QUANT",
    7: "KC_DYNAMIC_PERTOKEN",
    6: "MX_QUANT",
}
X2_QUANT_MODE_MAP = {
    0: "NON_QUANT",
    2: "KC_PERCHANNEL",
    6: "MX_QUANT",
}

# ====== 偏移量常量 ======
OFFSET_MC2_INIT_TILING = 0
OFFSET_MC2_CC_TILING = MC2_INIT_TILING_SIZE  # 64
OFFSET_TILING_INFO = MC2_INIT_TILING_SIZE + MC2_CC_TILING_SIZE  # 576
OFFSET_TILE_MM_TILING = OFFSET_TILING_INFO + ALLTO_ALL_MATMUL_TILING_INFO_SIZE  # 672
OFFSET_TAIL_MM_TILING_NON_QUANT = OFFSET_TILE_MM_TILING + MC2_MM_V3_TILING_DATA_SIZE  # 908
OFFSET_TILE_QUANT_MM_TILING = OFFSET_TILE_MM_TILING  # 672
OFFSET_TAIL_QUANT_MM_TILING = OFFSET_TILE_MM_TILING + MC2_QUANT_BMM_TILING_PARAMS_SIZE  # 1032

# args_info_*.bin 中 kernel 参数索引
ARGS_INDEX_NAMES = [
    "hccl_context",  # [0] runtime 注入
    "x1",            # [1]
    "x2",            # [2]
    "bias",          # [3]
    "x1_scale",      # [4]
    "x2_scale",      # [5]
    "comm_scale",    # [6]
    "x1_offset",     # [7]
    "x2_offset",     # [8]
    "y",             # [9]
    "all2all_out",   # [10]
    "workspaceGM",   # [11]
    "tilingGM",      # [12]
]

# HCCL 数据类型枚举
HCCL_DATA_TYPE_MAP = {
    0: "HCCL_DATA_TYPE_INT8",
    1: "HCCL_DATA_TYPE_INT16",
    2: "HCCL_DATA_TYPE_INT32",
    3: "HCCL_DATA_TYPE_INT64",
    4: "HCCL_DATA_TYPE_FLOAT16",
    5: "HCCL_DATA_TYPE_FLOAT32",
    6: "HCCL_DATA_TYPE_INT128",
    7: "HCCL_DATA_TYPE_FLOAT64",
    8: "HCCL_DATA_TYPE_BF16",
    9: "HCCL_DATA_TYPE_INT20",
    10: "HCCL_DATA_TYPE_UINT8",
    11: "HCCL_DATA_TYPE_UINT16",
    12: "HCCL_DATA_TYPE_UINT32",
    13: "HCCL_DATA_TYPE_UINT64",
    14: "HCCL_DATA_TYPE_FLOAT8_E5M2",
    15: "HCCL_DATA_TYPE_FLOAT8_E4M3FN",
    16: "HCCL_DATA_TYPE_BFLOAT16",
    17: "HCCL_DATA_TYPE_INT4",
    18: "HCCL_DATA_TYPE_UINT4",
    19: "HCCL_DATA_TYPE_FLOAT4_E2M1",
}


def safe_str(raw: bytes) -> str:
    """将 C 风格 char[] 转为 Python 字符串，截断于第一个 '\\0'。"""
    idx = raw.find(b"\x00")
    if idx >= 0:
        raw = raw[:idx]
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("latin-1", errors="replace")


def parse_mc2_init_tiling_inner(data: bytes, offset: int) -> dict:
    """解析 Mc2InitTilingInner（64B），实际布局见 CANN hccl_tiling_msg.h。"""
    if offset + MC2_INIT_TILING_SIZE > len(data):
        return {"error": "data too short for Mc2InitTiling"}
    version, mc2_hcomm_cnt = struct.unpack_from("<2I", data, offset)
    offsets = struct.unpack_from("<8I", data, offset + 8)
    debug_mode, prepare_position = struct.unpack_from("<2B", data, offset + 40)
    queue_num, comm_block_num = struct.unpack_from("<2H", data, offset + 42)
    dev_type = struct.unpack_from("<B", data, offset + 46)[0]
    return {
        "version": version,
        "mc2HcommCnt": mc2_hcomm_cnt,
        "offset": list(offsets),
        "debugMode": debug_mode,
        "preparePosition": prepare_position,
        "queueNum": queue_num,
        "commBlockNum": comm_block_num,
        "devType": dev_type,
    }


def parse_mc2_cc_tiling_inner(data: bytes, offset: int) -> dict:
    """解析 Mc2CcTilingInner（512B），实际布局见 CANN hccl_tiling_msg.h。"""
    if offset + MC2_CC_TILING_SIZE > len(data):
        return {"error": "data too short for Mc2CcTiling"}
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


def parse_allto_all_matmul_tiling_info(data: bytes, offset: int) -> dict:
    """解析 AlltoAllMatmulTilingInfo（96B）。"""
    if offset + ALLTO_ALL_MATMUL_TILING_INFO_SIZE > len(data):
        return {"error": "data too short for AlltoAllMatmulTilingInfo"}
    fields = struct.unpack_from("<12I 6Q", data, offset)
    x1_qm = fields[10]
    x2_qm = fields[11]
    hccl_dt = fields[14]
    return {
        "rankDim": fields[0],
        "tileM": fields[1],
        "tileCnt": fields[2],
        "tailM": fields[3],
        "tailCnt": fields[4],
        "biasLen": fields[5],
        "rankM": fields[6],
        "rankN": fields[7],
        "rankK": fields[8],
        "aicCoreNum": fields[9],
        "x1QuantMode": x1_qm,
        "x1QuantModeName": X1_QUANT_MODE_MAP.get(x1_qm, f"UNKNOWN({x1_qm})"),
        "x2QuantMode": x2_qm,
        "x2QuantModeName": X2_QUANT_MODE_MAP.get(x2_qm, f"UNKNOWN({x2_qm})"),
        "commLen": fields[12],
        "permuteLen": fields[13],
        "hcclDataType": hccl_dt,
        "hcclDataTypeName": HCCL_DATA_TYPE_MAP.get(hccl_dt, f"UNKNOWN({hccl_dt})"),
        "x1ScaleOptionalLen": fields[15],
        "dynamicExtraSpace": fields[16],
        "x1QuantDtype": fields[17],
    }


TCUBE_TILING_FIELDS = [
    "usedCoreNum", "M", "N", "Ka", "Kb", "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM", "stepN", "isBias",
    "transLength", "iterateOrder", "shareMode", "shareL1Size", "shareL0CSize",
    "shareUbSize", "batchM", "batchN", "singleBatchM", "singleBatchN", "stepKa",
    "stepKb", "depthAL1CacheUB", "depthBL1CacheUB", "dbL0A", "dbL0B", "dbL0C",
    "ALayoutInfoB", "ALayoutInfoS", "ALayoutInfoN", "ALayoutInfoG", "ALayoutInfoD",
    "BLayoutInfoB", "BLayoutInfoS", "BLayoutInfoN", "BLayoutInfoG", "BLayoutInfoD",
    "CLayoutInfoB", "CLayoutInfoS1", "CLayoutInfoN", "CLayoutInfoG", "CLayoutInfoS2",
    "BatchNum", "mxTypePara",
]

MC2_MM_V3_EXT_FIELDS = [
    "mTailCnt", "nTailCnt", "kTailCnt", "mBaseTailSplitCnt", "nBaseTailSplitCnt",
    "mTailMain", "nTailMain", "isHf32", "aswWindowLen",
]


def parse_tcube_tiling(data: bytes, offset: int) -> dict:
    """解析 TCubeTiling（200B = 50 × int32）。"""
    if offset + TCUBE_TILING_SIZE > len(data):
        return {"error": "data too short for TCubeTiling"}
    values = struct.unpack_from(f"<{len(TCUBE_TILING_FIELDS)}i", data, offset)
    return dict(zip(TCUBE_TILING_FIELDS, list(values)))


def parse_mc2_mm_v3_tiling_data(data: bytes, offset: int) -> dict:
    """解析 Mc2MatMulV3TilingData（236B = TCubeTiling + 9 × uint32）。"""
    if offset + MC2_MM_V3_TILING_DATA_SIZE > len(data):
        return {"error": "data too short for Mc2MatMulV3TilingData"}
    tcube = parse_tcube_tiling(data, offset)
    ext_values = struct.unpack_from(f"<{len(MC2_MM_V3_EXT_FIELDS)}I", data, offset + TCUBE_TILING_SIZE)
    ext = dict(zip(MC2_MM_V3_EXT_FIELDS, list(ext_values)))
    return {"tCubeTiling": tcube, **ext}


def parse_mc2_quant_bmm_data_params(data: bytes, offset: int) -> dict:
    """解析 Mc2QuantBatchMatmulV3DataParams（128B = 32 × uint32）。"""
    quant_fields = [
        "batchA", "batchB", "batchC", "batchA1", "batchA2", "batchA3", "batchA4",
        "batchB1", "batchB2", "batchB3", "batchB4", "batchC1", "batchC2", "batchC3",
        "batchC4", "singleCoreBatch", "isPerTensor", "isPertoken", "isDoubleScale",
        "biasThreeDim", "ubCalcM", "ubCalcN", "needUbBuffer", "realSingleCoreM",
        "realSingleCoreN", "biasDtype", "ubSize", "isMClash", "isNClash",
        "groupSizeM", "groupSizeN", "groupSizeK",
    ]
    if offset + 128 > len(data):
        return {"error": "data too short for Mc2QuantBatchMatmulV3DataParams"}
    values = struct.unpack_from(f"<{len(quant_fields)}I", data, offset)
    return dict(zip(quant_fields, list(values)))


def parse_mc2_l2cache_tile_params(data: bytes, offset: int) -> dict:
    """解析 Mc2L2cacheTileParams（24B = 6 × uint32）。"""
    l2_fields = ["mTileCntL2", "nTileCntL2", "mTileBlock", "nTileBlock", "calOrder", "isBasicTiling"]
    if offset + 24 > len(data):
        return {"error": "data too short for Mc2L2cacheTileParams"}
    values = struct.unpack_from(f"<{len(l2_fields)}I", data, offset)
    return dict(zip(l2_fields, list(values)))


def parse_mc2_sliding_window_params(data: bytes, offset: int) -> dict:
    """解析 Mc2SlidingWindowParams（8B = 2 × uint32）。"""
    if offset + 8 > len(data):
        return {"error": "data too short for Mc2SlidingWindowParams"}
    m_tail_tile, n_tail_tile = struct.unpack_from("<2I", data, offset)
    return {"mTailTile": m_tail_tile, "nTailTile": n_tail_tile}


def parse_mc2_quant_bmm_tiling_data_params(data: bytes, offset: int) -> dict:
    """解析 Mc2QuantBatchMatmulV3TilingDataParams（360B）。"""
    if offset + MC2_QUANT_BMM_TILING_PARAMS_SIZE > len(data):
        return {"error": "data too short for Mc2QuantBatchMatmulV3TilingDataParams"}
    params = parse_mc2_quant_bmm_data_params(data, offset)
    tcube = parse_tcube_tiling(data, offset + 128)
    l2cache = parse_mc2_l2cache_tile_params(data, offset + 328)
    sliding = parse_mc2_sliding_window_params(data, offset + 352)
    return {"params": params, "matmulTiling": tcube, "tileL2cacheTiling": l2cache, "adaptiveSlidingWin": sliding}


def parse_non_quant_tiling_data(data: bytes) -> dict:
    """解析非量化 AlltoAllMatmulTilingData（1136B）。"""
    return {
        "mc2InitTiling": parse_mc2_init_tiling_inner(data, OFFSET_MC2_INIT_TILING),
        "mc2CcTiling": parse_mc2_cc_tiling_inner(data, OFFSET_MC2_CC_TILING),
        "alltoAllMatmulTilingInfo": parse_allto_all_matmul_tiling_info(data, OFFSET_TILING_INFO),
        "mc2MmV3TileTilingData": parse_mc2_mm_v3_tiling_data(data, OFFSET_TILE_MM_TILING),
        "mc2MmV3TailTilingData": parse_mc2_mm_v3_tiling_data(data, OFFSET_TAIL_MM_TILING_NON_QUANT),
    }


def parse_quant_tiling_data(data: bytes) -> dict:
    """解析量化 AlltoAllQuantMatmulTilingData（1384B）。"""
    return {
        "mc2InitTiling": parse_mc2_init_tiling_inner(data, OFFSET_MC2_INIT_TILING),
        "mc2CcTiling": parse_mc2_cc_tiling_inner(data, OFFSET_MC2_CC_TILING),
        "alltoAllQuantMatmulTilingInfo": parse_allto_all_matmul_tiling_info(data, OFFSET_TILING_INFO),
        "mc2QuantMmTileTilingData": parse_mc2_quant_bmm_tiling_data_params(data, OFFSET_TILE_QUANT_MM_TILING),
        "mc2QuantMmTailTilingData": parse_mc2_quant_bmm_tiling_data_params(data, OFFSET_TAIL_QUANT_MM_TILING),
    }


def detect_quant_mode(data: bytes) -> bool:
    """通过 x1QuantMode / x2QuantMode 字段检测量化/非量化模式。

    x1QuantMode / x2QuantMode 位于 AlltoAllMatmulTilingInfo offset=576+40=616, 各 u32:
      x1QuantMode: 0=非量化, 7=KC动态pertoken量化, 6=MX量化
      x2QuantMode: 0=非量化, 2=KC perchannel量化, 6=MX量化
    两者均为 0 时为非量化，否则为量化。
    """
    if len(data) < OFFSET_TILING_INFO + ALLTO_ALL_MATMUL_TILING_INFO_SIZE:
        return False
    info = parse_allto_all_matmul_tiling_info(data, OFFSET_TILING_INFO)
    if "error" in info:
        return False
    return info.get("x1QuantMode", 0) != 0 or info.get("x2QuantMode", 0) != 0


def parse_args_info(data: bytes) -> dict:
    """解析 args_info_*.bin（AlltoAllMatmul 有 13 个有效 args，index 0-12）。"""
    num_valid_args = len(ARGS_INDEX_NAMES)
    num_ptrs = min(num_valid_args, len(data) // 8)
    ptrs = struct.unpack_from(f"<{num_ptrs}Q", data, 0)
    args = {}
    for i in range(num_ptrs):
        name = ARGS_INDEX_NAMES[i]
        args[f"[{i}] {name}"] = f"0x{ptrs[i]:016x}"
    args["_dumpFileSize"] = len(data)
    args["_validArgCount"] = num_ptrs
    args["_note"] = (
        f"dump 代码固定拷贝 min(devArgsLen, 256) 字节，AlltoAllMatmul 有效 args 为 {num_ptrs} 个。"
        f" 实际 devArgsLen 请查看日志中 'Tiling dump start' 行的 devArgsLen 字段。"
    )
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
    """人类可读格式打印 tiling data。"""
    info_key = "alltoAllQuantMatmulTilingInfo" if is_quant else "alltoAllMatmulTilingInfo"
    mm_tile_key = "mc2QuantMmTileTilingData" if is_quant else "mc2MmV3TileTilingData"
    mm_tail_key = "mc2QuantMmTailTilingData" if is_quant else "mc2MmV3TailTilingData"

    print_section("Mc2InitTiling (HCCL Init, 64B)")
    print_kv("mc2InitTilingInner", result["mc2InitTiling"])

    print_section("Mc2CcTiling (HCCL CC, 512B)")
    print_kv("mc2CcTilingInner", result["mc2CcTiling"])

    print_section(f"AlltoAllMatmulTilingInfo (88B)  [{'QUANT' if is_quant else 'NON-QUANT'}]")
    print_kv(info_key, result[info_key])

    print_section("Matmul Tile Tiling (头块)")
    print_kv(mm_tile_key, result[mm_tile_key])

    print_section("Matmul Tail Tiling (尾块)")
    print_kv(mm_tail_key, result[mm_tail_key])


def print_args_info(result: dict):
    """人类可读格式打印 args info。"""
    print_section("Args Info (kernel GM pointer)")
    dump_file_size = result.pop("_dumpFileSize", 0)
    valid_arg_count = result.pop("_validArgCount", 0)
    for name, addr in result.items():
        logging.info("  %-30s: %s", name, addr)
    logging.info("")
    logging.info("  有效 args 数量: %d (index 0-%d)", valid_arg_count, valid_arg_count - 1)
    logging.info("  dump 文件大小: %d bytes", dump_file_size)
    logging.info("  提示: tilingGM 应在 [12] 位置 (offset=96)")
    tiling_gm = result.get("[12] tilingGM", "N/A")
    logging.info("  [12] tilingGM = %s", tiling_gm)


def verify_tiling_data(dump_data: bytes, ref_data: bytes) -> bool:
    """对比 dump 文件与 host 参考文件，逐字节验证。

    dump 文件固定 2048B（含 padding），参考文件为结构体实际大小。
    只对比参考文件长度范围内的字节。
    """
    ref_size = len(ref_data)
    dump_size = len(dump_data)

    logging.info("=" * 70)
    logging.info("  Verify: dump vs host reference")
    logging.info("=" * 70)
    logging.info("  dump 文件: %d bytes (含 padding)", dump_size)
    logging.info("  参考文件: %d bytes (结构体实际大小)", ref_size)

    if dump_size < ref_size:
        logging.error("dump 文件 %d < 参考文件 %d，dump 数据不完整", dump_size, ref_size)
        return False

    mismatch_count = 0
    first_mismatch_offset = -1
    field_ranges = [
        (0, MC2_INIT_TILING_SIZE, "Mc2InitTiling"),
        (MC2_INIT_TILING_SIZE, MC2_INIT_TILING_SIZE + MC2_CC_TILING_SIZE, "Mc2CcTiling"),
        (OFFSET_TILING_INFO, OFFSET_TILING_INFO + ALLTO_ALL_MATMUL_TILING_INFO_SIZE, "AlltoAllMatmulTilingInfo"),
        (OFFSET_TILE_MM_TILING, OFFSET_TILE_MM_TILING + max(MC2_MM_V3_TILING_DATA_SIZE, MC2_QUANT_BMM_TILING_PARAMS_SIZE), "TileMatmulTiling"),
    ]

    for i in range(ref_size):
        if dump_data[i] != ref_data[i]:
            mismatch_count += 1
            if first_mismatch_offset < 0:
                first_mismatch_offset = i

    if mismatch_count == 0:
        logging.info("")
        logging.info("  结果: PASS - 全部 %d 字节完全一致", ref_size)
        logging.info("  dump 文件尾部 %d 字节为 padding（不在对比范围内）", dump_size - ref_size)
        return True

    logging.error("")
    logging.error("  结果: FAIL - %d/%d 字节不一致", mismatch_count, ref_size)
    logging.error("  首个不一致偏移: %d (0x%x)", first_mismatch_offset, first_mismatch_offset)

    for start, end, name in field_ranges:
        if start <= first_mismatch_offset < end:
            logging.error("  所属字段: %s (offset %d-%d)", name, start, end)
            break

    if mismatch_count <= 20:
        logging.info("")
        logging.info("  不一致字节明细:")
        for i in range(ref_size):
            if dump_data[i] != ref_data[i]:
                field_name = "unknown"
                for start, end, name in field_ranges:
                    if start <= i < end:
                        field_name = name
                        break
                logging.error("    offset %4d (0x%04x) [%s]: dump=0x%02x, ref=0x%02x",
                              i, i, field_name, dump_data[i], ref_data[i])
    else:
        logging.warning("  不一致字节超过 20 个，仅显示前 20 个:")
        shown = 0
        for i in range(ref_size):
            if dump_data[i] != ref_data[i]:
                field_name = "unknown"
                for start, end, name in field_ranges:
                    if start <= i < end:
                        field_name = name
                        break
                logging.error("    offset %4d (0x%04x) [%s]: dump=0x%02x, ref=0x%02x",
                              i, i, field_name, dump_data[i], ref_data[i])
                shown += 1
                if shown >= 20:
                    logging.error("    ... (剩余 %d 个省略)", mismatch_count - 20)
                    break
    return False


def main():
    parser = argparse.ArgumentParser(
        description="AlltoAllMatmul TilingData dump 解析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python parse_allto_all_matmul_tiling.py tiling_data_AlltoAllMatmul.0.1.20260819.bin
  python parse_allto_all_matmul_tiling.py args_info_AlltoAllMatmul.0.1.20260819.bin --args
  python parse_allto_all_matmul_tiling.py tiling_data_AlltoAllMatmul.0.1.20260819.bin --json
  python parse_allto_all_matmul_tiling.py tiling_data_*.bin --verify tiling_ref.bin
        """,
    )
    parser.add_argument("bin_file", help="dump 产生的 bin 文件路径")
    parser.add_argument("--args", action="store_true", help="解析 args_info_*.bin（GM 地址指针）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    parser.add_argument("--verify", metavar="REF_FILE", help="对比 host 参考文件（由 MC2_TILING_DUMP_REF_PATH 环境变量生成）")
    args = parser.parse_args()

    if not os.path.isfile(args.bin_file):
        logging.error("文件不存在: %s", args.bin_file)
        sys.exit(1)

    with open(args.bin_file, "rb") as f:
        data = f.read()

    logging.info("读取文件: %s (%d bytes)", args.bin_file, len(data))

    if args.args:
        result = parse_args_info(data)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_args_info(result)
        return

    if args.verify:
        if not os.path.isfile(args.verify):
            logging.error("参考文件不存在: %s", args.verify)
            sys.exit(1)
        with open(args.verify, "rb") as f:
            ref_data = f.read()
        ok = verify_tiling_data(data, ref_data)
        sys.exit(0 if ok else 1)

    is_quant = detect_quant_mode(data)
    info = parse_allto_all_matmul_tiling_info(data, OFFSET_TILING_INFO)
    x1_qm = info.get("x1QuantMode", 0) if "error" not in info else 0
    x2_qm = info.get("x2QuantMode", 0) if "error" not in info else 0
    x1_qm_name = X1_QUANT_MODE_MAP.get(x1_qm, f"UNKNOWN({x1_qm})")
    x2_qm_name = X2_QUANT_MODE_MAP.get(x2_qm, f"UNKNOWN({x2_qm})")
    if is_quant:
        logging.info("解析模式: 量化 (x1QuantMode=%d/%s, x2QuantMode=%d/%s)", x1_qm, x1_qm_name, x2_qm, x2_qm_name)
    else:
        logging.info("解析模式: 非量化 (x1QuantMode=0, x2QuantMode=0)")

    struct_size = ALLTO_ALL_QUANT_MATMUL_TILING_DATA_SIZE if is_quant else ALLTO_ALL_MATMUL_TILING_DATA_SIZE
    if len(data) < struct_size:
        logging.warning(
            "文件大小 %d < 结构体大小 %d，解析可能不完整", len(data), struct_size
        )

    if is_quant:
        result = parse_quant_tiling_data(data)
        logging.info(
            "结构体 AlltoAllQuantMatmulTilingData: %dB, dump 文件 %dB, 尾部 %dB 为 padding",
            struct_size, len(data), len(data) - struct_size,
        )
    else:
        result = parse_non_quant_tiling_data(data)
        logging.info(
            "结构体 AlltoAllMatmulTilingData: %dB, dump 文件 %dB, 尾部 %dB 为 padding",
            struct_size, len(data), len(data) - struct_size,
        )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_tiling_data(result, is_quant)


if __name__ == "__main__":
    main()
