#!/usr/bin/env python3
"""Decode the focused arch92 BlockMmad set_cond diagnostics."""

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path


COND_RE = re.compile(r"MOV_SPR_XN\s+SPR:COND,\s+XN:[^=]+=0x([0-9a-fA-F]+)")
SOURCE_RE = re.compile(r"^(.*?instr_log\.dump):\d+:")
UINT_BIAS = 1_000_000

PARTIAL_MARKERS = {
    912101: "GMM2 partial K=128",
    912102: "GMM2 partial K=256",
    912103: "GMM2 partial K=384",
    912104: "GMM2 partial K=512",
    912201: "GMM1 original Fixpipe control",
}

DIRECT_MARKERS = {
    912105: "GMM2 direct L0C2UB, quant_pre_rnd=0",
    912106: "GMM2 direct L0C2UB, quant_pre_rnd=1",
    912202: "GMM1 direct L0C2UB, quant_pre_rnd=0",
    912203: "GMM1 direct L0C2UB, quant_pre_rnd=1",
}


def decode_bf16(raw):
    bits = (raw & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def decode_uint(raw):
    if raw < UINT_BIAS:
        raise ValueError(f"expected biased uint, got 0x{raw:x}")
    return raw - UINT_BIAS


def read_streams(paths):
    streams = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            match = COND_RE.search(line)
            if match is None:
                continue
            source_match = SOURCE_RE.match(line)
            source = source_match.group(1) if source_match else str(path)
            streams[source].append(int(match.group(1), 16))
    return streams


def decode_stream(source, values):
    found = 0
    index = 0
    while index < len(values):
        marker = values[index]
        if marker in PARTIAL_MARKERS:
            payload = values[index + 1:index + 23]
            if len(payload) != 22:
                print(f"[{PARTIAL_MARKERS[marker]}] incomplete in {source}")
                break
            block_idx = decode_uint(payload[0])
            completed_k = decode_uint(payload[1])
            decoded = [decode_bf16(value) for value in payload[2:]]
            print(f"\n[{marker}] {PARTIAL_MARKERS[marker]}")
            print(f"source={source}")
            print(f"physical_block={block_idx}, completed_k={completed_k}")
            print(f"bf16={decoded}")
            found += 1
            index += 23
            continue
        if marker in DIRECT_MARKERS:
            payload = values[index + 1:index + 29]
            if len(payload) != 28:
                print(f"[{DIRECT_MARKERS[marker]}] incomplete in {source}")
                break
            metadata = [decode_uint(value) for value in payload[:8]]
            decoded = [decode_bf16(value) for value in payload[8:]]
            print(f"\n[{marker}] {DIRECT_MARKERS[marker]}")
            print(f"source={source}")
            print(
                "physical_block={}, quant_pre_rnd={}, m={}, n={}, n_size={}, "
                "m_size={}, src_stride={}, dst_stride={}".format(*metadata)
            )
            print(f"bf16={decoded}")
            found += 1
            index += 29
            continue
        index += 1
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Raw instruction log or grep output files")
    args = parser.parse_args()
    streams = read_streams(args.logs)
    found = sum(decode_stream(source, values) for source, values in streams.items())
    if found == 0:
        raise SystemExit("No arch92 MMAD diagnostic markers were found")


if __name__ == "__main__":
    main()
