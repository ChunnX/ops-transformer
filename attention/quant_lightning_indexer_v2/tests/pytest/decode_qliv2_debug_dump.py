#!/usr/bin/env python3
"""Decode QLI v2 uint16 BF16 sortable-key DumpTensor output."""

import argparse
from pathlib import Path

import numpy as np


def decode_bf16_sortable_keys(keys):
    """Reverse FloatToSortableKey<bfloat16_t> and return FP32 values."""
    keys = np.asarray(keys, dtype=np.uint16)
    positive = (keys & np.uint16(0x8000)) != 0
    bf16_bits = np.where(
        positive,
        keys ^ np.uint16(0x8000),
        keys ^ np.uint16(0xFFFF),
    ).astype(np.uint16)
    fp32_bits = bf16_bits.astype(np.uint32) << np.uint32(16)
    return fp32_bits.view(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Decode QLI v2 pre-TopK/TopK uint16 sortable keys dumped by AscendC::DumpTensor.")
    parser.add_argument("input", type=Path, help="DumpTensor .npy file containing uint16 sortable keys")
    parser.add_argument("-o", "--output", type=Path, help="Output .npy path; defaults to <input>.decoded.npy")
    args = parser.parse_args()

    keys = np.load(args.input)
    if keys.dtype != np.uint16:
        if keys.dtype.itemsize != 2:
            raise TypeError(f"expected a 2-byte sortable-key dtype, got {keys.dtype}")
        keys = keys.view(np.uint16)
    decoded = decode_bf16_sortable_keys(keys)
    output = args.output or args.input.with_name(f"{args.input.stem}.decoded.npy")
    np.save(output, decoded)
    print(f"decoded {decoded.size} BF16 score values to {output}")


if __name__ == "__main__":
    main()


# DumpTensor descriptor map used by the temporary kernel instrumentation:
#   100000 + dumped_s2_start + g_idx: MXFP4 QK values for one G and one S2 segment
#   200000: weight row for the target query
#   300000 + s2_start: pre-TopK score sortable keys for that S2 segment
#   400000 / 400001: non-LD final TopK indices / sortable score keys
#   500000 / 500001: raw query / key E8M0 scale bytes
#   500002 / 500003: seqused_q / seqused_k as read from GM
#   510000: target batch block table for PA_BBND
#   520000 + logical_block: raw PA_BBND key scale block
#   600000 / 600001: LD-merged final TopK indices / sortable score keys
#   700000 + act_s1_size / 800000 + act_s2_size: effective lengths seen by NPU
