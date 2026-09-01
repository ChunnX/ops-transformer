#!/usr/bin/env python3
"""Decode focused arch92 MegaMoe BlockMmad stage diagnostics."""

import argparse
import re
import struct
from collections import defaultdict
from pathlib import Path


COND_RE = re.compile(r"MOV_SPR_XN\s+SPR:COND,\s+XN:[^=]+=0x([0-9a-fA-F]+)")
SOURCE_RE = re.compile(r"^(.*?instr_log\.dump):\d+:")
UINT_BIAS = 1_000_000

METADATA_FIELDS = ("payload_format", "physical_block", "rank", "expert", "row")
DATA_COUNT = 20
SCALE_COUNT = 8
CONFIG_FIELDS = (
    "problem_m",
    "problem_n",
    "problem_k",
    "scale_k",
    "actual_m",
    "actual_n",
    "actual_k",
    "tile_m",
    "tile_n",
    "k_a_l1",
    "k_b_l1",
    "scale_k_l1",
    "weight_layout",
    "m_loc",
    "n_loc",
    "k_loc",
)
STAGE_SPECS = {
    910000: ("PRE_DISPATCH", "fp8_scale"),
    910001: ("DISPATCH_OUTPUT", "fp8_scale"),
    910002: ("GMM1_OUTPUT", "bf16_pair"),
    910003: ("SWIGLU_OUTPUT", "bf16"),
    910004: ("GMM2_OUTPUT", "bf16"),
    910005: ("COMBINE_OUTPUT", "bf16"),
    910006: ("SWIGLU_QUANT_OUTPUT", "fp8_scale"),
    910007: ("GMM2_INPUT", "fp8_scale"),
    910012: ("GMM2_CONFIG", "config"),
}

def decode_uint(raw):
    if raw < UINT_BIAS:
        raise ValueError(f"expected biased uint, got 0x{raw:x}")
    return raw - UINT_BIAS


def decode_bf16(raw):
    bits = (raw & 0xFFFF) << 16
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def decode_fp8_e4m3fn(raw):
    raw &= 0xFF
    sign = -1.0 if raw & 0x80 else 1.0
    exponent = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    if exponent == 0:
        return sign * (mantissa / 8.0) * (2.0 ** -6)
    if exponent == 0xF and mantissa == 0x7:
        return float("nan")
    return sign * (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))


def decode_fp8_e5m2(raw):
    raw &= 0xFF
    sign = -1.0 if raw & 0x80 else 1.0
    exponent = (raw >> 2) & 0x1F
    mantissa = raw & 0x3
    if exponent == 0:
        return sign * (mantissa / 4.0) * (2.0 ** -14)
    if exponent == 0x1F:
        return sign * float("inf") if mantissa == 0 else float("nan")
    return sign * (1.0 + mantissa / 4.0) * (2.0 ** (exponent - 15))


def decode_fp8(raw, payload_format):
    if payload_format == 4:
        return decode_fp8_e4m3fn(raw)
    if payload_format == 5:
        return decode_fp8_e5m2(raw)
    raise ValueError(f"unsupported FP8 format tag: {payload_format}")


def decode_e8m0(raw):
    raw &= 0xFF
    return float("nan") if raw == 0xFF else 2.0 ** (raw - 127)


def payload_count(kind):
    if kind == "fp8_scale":
        return DATA_COUNT + SCALE_COUNT
    if kind == "bf16_pair":
        return DATA_COUNT * 2
    if kind == "bf16":
        return DATA_COUNT
    if kind == "config":
        return len(CONFIG_FIELDS)
    raise ValueError(f"unsupported stage kind: {kind}")


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


def scan_stream(source, values):
    records = []
    index = 0
    while index < len(values):
        marker = values[index]
        if marker not in STAGE_SPECS:
            index += 1
            continue

        name, kind = STAGE_SPECS[marker]
        metadata_end = index + 1 + len(METADATA_FIELDS)
        record_end = metadata_end + payload_count(kind)
        metadata_raw = values[index + 1:metadata_end]
        payload = values[metadata_end:record_end]
        if len(metadata_raw) != len(METADATA_FIELDS) or len(payload) != payload_count(kind):
            print(f"WARNING: incomplete record {marker}:{name} in {source}")
            break
        try:
            metadata = {
                field: decode_uint(raw)
                for field, raw in zip(METADATA_FIELDS, metadata_raw)
            }
            if kind == "config":
                payload = [decode_uint(raw) for raw in payload]
        except ValueError as error:
            print(f"WARNING: invalid record {marker}:{name} in {source}: {error}")
            index += 1
            continue
        records.append({
            "source": source,
            "marker": marker,
            "name": name,
            "kind": kind,
            "payload": payload,
            **metadata,
        })
        index = record_end
    return records


def print_value_table(label, raw_values, decoder, raw_width):
    print(f"\n{label}:")
    print("index  raw       decoded")
    print("-----  --------  ------------")
    for index, raw in enumerate(raw_values):
        print(f"{index:5d}  0x{raw:0{raw_width}x}  {decoder(raw):.10g}")


def print_config(payload):
    values = dict(zip(CONFIG_FIELDS, payload))
    values["weight_layout"] = {0: "DN", 1: "ZN"}.get(
        values["weight_layout"], f"UNKNOWN({values['weight_layout']})"
    )
    for field in CONFIG_FIELDS:
        print(f"{field:14s}: {values[field]}")


def print_record(record):
    print(f"\n[{record['marker']}] {record['name']}")
    print(f"source={record['source']}")
    print(
        f"physical_block={record['physical_block']} rank={record['rank']} "
        f"expert={record['expert']} row={record['row']}"
    )
    payload = record["payload"]
    if record["kind"] == "fp8_scale":
        payload_format = record["payload_format"]
        print(f"fp8_format={'E4M3FN' if payload_format == 4 else 'E5M2'}")
        print_value_table(
            "data", payload[:DATA_COUNT],
            lambda raw, fmt=payload_format: decode_fp8(raw, fmt), 2
        )
        print_value_table(
            "scale", payload[DATA_COUNT:DATA_COUNT + SCALE_COUNT], decode_e8m0, 2
        )
    elif record["kind"] == "bf16_pair":
        print_value_table("gate", payload[:DATA_COUNT], decode_bf16, 4)
        print_value_table("up", payload[DATA_COUNT:], decode_bf16, 4)
    elif record["kind"] == "bf16":
        print_value_table("data", payload, decode_bf16, 4)
    else:
        print_config(payload)


def print_missing(records):
    found = {record["marker"] for record in records}
    missing = [
        f"{marker}:{STAGE_SPECS[marker][0]}"
        for marker in STAGE_SPECS
        if marker not in found
    ]
    if missing:
        print(f"\nWARNING: missing records: {', '.join(missing)}")
    else:
        print("\nAll focused Arch92 MegaMoe stage records were captured.")


def main():
    parser = argparse.ArgumentParser(
        description="Decode arch92 MegaMoe BlockMmad precision diagnostics"
    )
    parser.add_argument("logs", nargs="+", help="Raw instruction log or grep output files")
    args = parser.parse_args()

    streams = read_streams(args.logs)
    records = [
        record
        for source, values in streams.items()
        for record in scan_stream(source, values)
    ]
    if not records:
        raise SystemExit("No arch92 BlockMmad precision markers were found")

    print("=== Arch92 MegaMoe BlockMmad precision diagnostics ===")
    for record in sorted(records, key=lambda item: (item["marker"], item["source"])):
        print_record(record)
    print_missing(records)


if __name__ == "__main__":
    main()
