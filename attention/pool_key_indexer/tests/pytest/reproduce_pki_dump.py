#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoolKeyIndexer 单算子复现脚本：从 data-dump 已解析产物反推 cfg，原样复现单算子调用，判定 CRASH/PASS。

用法:
    python3 reproduce_pki_dump.py <dump_dir> <exception_info前缀> <operator.json>

    <dump_dir>          含 input.N.bin / output.N.bin 与 PoolKeyIndexer_*.o/.json 的目录
    <exception_info前缀> exception_info 文件名（去扩展名），如 exception_info.45.7.20260829101541784
    <operator.json>     同 dump 的 PoolKeyIndexer_<hash>.json（提供 dtype 与 opParaSize）

运行前提:
    - source <CANN>/ascend-toolkit/set_env.sh
    - PYTHONPATH 指向 torch_extension 源码（使 torch.ops.cann_ops_transformer.pool_key_indexer 注册）:
      PYTHONPATH=<repo>/torch_extension python3 reproduce_pki_dump.py ...
    - 设备号用 PKI_DEVICE_ID 指定（默认 0）

原理（对应 ascendc-data-dump-analysis skill）:
    从 tiling 段(input.N.bin 中 size==opParaSize 的那个)解码出 batch/分组/序列长度/属性，
    结合 .json 的 dtype，反推 query/pool_key/weights 的 shape 与算子属性，加载精确 dump 数据重放。
    当前自动反推覆盖 TND/TND 布局；BSND/PA_BBND 仅提示需手工补 shape 映射。
"""

import os
import sys
import json
import glob

import numpy as np
import torch
import torch_npu  # noqa: F401
from cann_ops_transformer import pool_key_indexer

HEAD_DIM = 128  # PKI_HEAD_DIM
N2 = 1  # PKI_N2_FIXED
LAYOUT = {0: "BSND", 1: "TND", 2: "PA_BBND"}
DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}


# PoolKeyIndexerTilingData 字段顺序（op_host/pool_key_indexer_tiling.h），含 8 字节对齐 pad
# word0 bSize, word1 gSize, word2 s1Size, word3 (pad), word4+5 s2Size(int64),
# word6 sparseCount, word7 topk, word8 poolSize, word9 usedCoreNum, word10 s1BaseSize,
# word11 mBaseSize, word12 mBaseSizeMax, word13 s2BaseSize, word14 trunkLen, word15 maskMode,
# word16 quantMode, word17 returnValue, word18 layoutQ, word19 layoutK, ...
def decode_tiling(w32):
    s2_lo, s2_hi = int(w32[4]), int(w32[5])
    return {
        "bSize": int(w32[0]),
        "gSize": int(w32[1]),
        "s1Size": int(w32[2]),
        "s2Size": s2_lo | (s2_hi << 32),
        "sparseCount": int(w32[6]),
        "topk": int(w32[7]),
        "poolSize": int(w32[8]),
        "maskMode": int(w32[15]),
        "quantMode": int(w32[16]),
        "returnValue": int(w32[17]),
        "layoutQ": int(w32[18]),
        "layoutK": int(w32[19]),
    }


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    dump_dir, prefix, jsonf = sys.argv[1], sys.argv[2], sys.argv[3]
    device = int(os.environ.get("PKI_DEVICE_ID", "0"))
    torch_npu.npu.set_device(device)
    dev = f"npu:{device}"

    info = json.load(open(jsonf))
    op_para = int(info.get("opParaSize", 0))
    dtype_s = info["supportInfo"]["inputs"][0]["dtype"]
    dtype = DTYPE[dtype_s]

    # 定位 tiling 段：size == opParaSize 的 input.N.bin
    tiling_bin = None
    for f in sorted(glob.glob(os.path.join(dump_dir, f"{prefix}.input.*.bin"))):
        if op_para and os.path.getsize(f) == op_para:
            tiling_bin = f
            break
    if tiling_bin is None:
        print(
            "[repro] 未找到 tiling 段 (size==opParaSize)，请确认已 msaicerr -d 拆包",
            flush=True,
        )
        sys.exit(3)
    t = decode_tiling(np.fromfile(tiling_bin, dtype=np.int32))

    layout_q, layout_k = LAYOUT[t["layoutQ"]], LAYOUT[t["layoutK"]]
    if layout_q != "TND" or layout_k != "TND":
        print(
            f"[repro] 当前仅自动反推 TND/TND，dump 为 {layout_q}/{layout_k}，请手工补 shape 映射",
            flush=True,
        )
        sys.exit(3)

    def ld(idx, shape):
        # fp16/bf16 均 2 字节：按 uint16 读原始位，再 view 成目标 dtype（位级重解释）
        a = np.fromfile(
            os.path.join(dump_dir, f"{prefix}.input.{idx}.bin"), dtype=np.uint16
        )
        n = int(np.prod(shape))
        a = a[:n] if a.size >= n else a
        return torch.from_numpy(a.copy()).view(dtype).view(shape).to(dev)

    # TND: query (T1, N1, D)、pool_key (T2, N2, D)、weights (T1, N1)
    query = ld(0, (t["s1Size"], t["gSize"], HEAD_DIM))
    pool_k = ld(1, (t["s2Size"], N2, HEAD_DIM))
    weights = ld(2, (t["s1Size"], t["gSize"]))

    def ldi64(idx, n):
        a = np.fromfile(
            os.path.join(dump_dir, f"{prefix}.input.{idx}.bin"), dtype=np.int64
        )
        return torch.from_numpy(a[:n].copy()).to(dev)

    ptk = ldi64(3, t["bSize"])  # pool_tail_k
    asq = ldi64(4, t["bSize"])  # actual_seq_q（前缀和）
    ask = ldi64(5, t["bSize"])  # actual_seq_k（前缀和）

    print(
        f"[repro] dev={device} {layout_q}/{layout_k} {dtype_s} bSize={t['bSize']} "
        f"gSize={t['gSize']} s1={t['s1Size']} s2={t['s2Size']} topk={t['topk']} "
        f"ps={t['poolSize']} mask={t['maskMode']} quant={t['quantMode'] - 1} "
        f"ret={bool(t['returnValue'])} tail={ptk.cpu().tolist()} "
        f"asq={asq.cpu().tolist()} ask={ask.cpu().tolist()}",
        flush=True,
    )

    try:
        idx, val = pool_key_indexer(
            query,
            pool_k,
            weights,
            ptk,
            actual_seq_q=asq,
            actual_seq_k=ask,
            block_table=None,
            layout_q=layout_q,
            layout_k=layout_k,
            topk=t["topk"],
            pool_size=t["poolSize"],
            mask_mode=t["maskMode"],
            quant_mode=t["quantMode"] - 1,
            return_value=bool(t["returnValue"]),
        )
        torch.npu.synchronize()
        print(
            f"[repro] OK  indices={tuple(idx.shape)} values={tuple(val.shape)}",
            flush=True,
        )
        print("REPRODUCE_RESULT: PASS")
    except RuntimeError as e:
        print(f"[repro] CRASH (RuntimeError): {str(e)[:400]}", flush=True)
        print("REPRODUCE_RESULT: CRASH")
        sys.exit(2)


if __name__ == "__main__":
    main()
