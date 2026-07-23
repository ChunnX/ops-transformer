#!/usr/bin/env python3
"""Convert QLI v2 AscendC DumpTensor text output to NumPy files."""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from decode_qliv2_debug_dump import decode_bf16_sortable_keys


HEADER_RE = re.compile(
    r"DumpTensor:\s*desc=(\d+).*?data_type=([^,\s]+).*?dump_size=(\d+)",
    re.IGNORECASE,
)


def _dtype_from_name(name):
    normalized = name.upper()
    if "UINT16" in normalized:
        return np.uint16
    if "INT32" in normalized:
        return np.int32
    if "UINT32" in normalized:
        return np.uint32
    if "UINT8" in normalized:
        return np.uint8
    if "INT8" in normalized:
        return np.int8
    if "FLOAT32" in normalized or normalized == "FLOAT":
        return np.float32
    return np.float32


def parse_dump_records(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(HEADER_RE.finditer(text))
    records = []
    for index, match in enumerate(matches):
        chunk_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[match.end():chunk_end]
        left = chunk.find("[")
        right = chunk.find("]", left + 1)
        if left < 0 or right <= left:
            continue
        payload = re.sub(r"[\[\],]", " ", chunk[left:right + 1])
        values = np.fromstring(payload, sep=" ", dtype=np.float64)
        dtype = _dtype_from_name(match.group(2))
        values = values.astype(dtype, copy=False)
        records.append({
            "desc": int(match.group(1)),
            "dtype_name": match.group(2),
            "dump_size": int(match.group(3)),
            "values": values,
        })
    return records


def _save_record_files(records, output_dir):
    counters = defaultdict(int)
    for record in records:
        desc = record["desc"]
        ordinal = counters[desc]
        counters[desc] += 1
        path = output_dir / f"desc_{desc}_{ordinal}_{record['dtype_name']}.npy"
        np.save(path, record["values"])
        if record["values"].size != record["dump_size"]:
            print(
                f"warning: desc={desc} expected {record['dump_size']} values, "
                f"parsed {record['values'].size}; the per-core dump may be truncated"
            )


def _records_for_desc(records, desc):
    return [record["values"] for record in records if record["desc"] == desc]


def _save_known_tensors(records, output_dir):
    score_records = sorted(
        (record for record in records if 300000 <= record["desc"] < 400000),
        key=lambda record: record["desc"],
    )
    if score_records:
        score_keys = np.concatenate(
            [record["values"].astype(np.uint16, copy=False) for record in score_records])
        np.save(output_dir / "npu_score_keys.npy", score_keys)
        np.save(output_dir / "npu_score.npy", decode_bf16_sortable_keys(score_keys))

    use_ld = bool(_records_for_desc(records, 600000))
    index_desc = 600000 if use_ld else 400000
    score_desc = 600001 if use_ld else 400001
    indices = _records_for_desc(records, index_desc)
    score_keys = _records_for_desc(records, score_desc)
    if indices:
        np.save(output_dir / "npu_topk_index_kernel.npy", indices[-1].astype(np.int32, copy=False))
    if score_keys:
        keys = score_keys[-1].astype(np.uint16, copy=False)
        np.save(output_dir / "npu_topk_score_keys_kernel.npy", keys)
        np.save(output_dir / "npu_topk_score_kernel.npy", decode_bf16_sortable_keys(keys))

    fixed_desc_names = {
        200001: 'npu_weight_bf16_packed',
        500000: "npu_query_scale_raw",
        500001: "npu_key_scale_raw",
        500002: "npu_seqused_q",
        500003: "npu_seqused_k",
        510000: "npu_block_table",
    }
    for desc, name in fixed_desc_names.items():
        tensors = _records_for_desc(records, desc)
        if tensors:
            np.save(output_dir / f"{name}.npy", tensors[-1])

    act_s1_sizes = sorted({record["desc"] - 700000 for record in records
                           if 700000 <= record["desc"] < 800000})
    act_s2_sizes = sorted({record["desc"] - 800000 for record in records
                           if 800000 <= record["desc"] < 900000})
    if act_s1_sizes:
        np.save(output_dir / "npu_act_s1_size.npy", np.asarray(act_s1_sizes, dtype=np.uint32))
    if act_s2_sizes:
        np.save(output_dir / "npu_act_s2_size.npy", np.asarray(act_s2_sizes, dtype=np.uint32))

    prefix_records = [
        record for record in records if 9000000 <= record['desc'] < 10000000]
    if prefix_records:
        # Keep the final occurrence when a log contains repeated records for the same G.
        prefix_by_desc = {record['desc']: record['values'] for record in prefix_records}
        prefix_descs = sorted(prefix_by_desc)
        prefix_desc_begin = prefix_descs[0]
        prefix_s2_begin = prefix_desc_begin - 9000000
        packed_weights = _records_for_desc(records, 200001)
        expected_g = packed_weights[-1].size if packed_weights else len(prefix_descs)
        if len(prefix_descs) != expected_g:
            print(
                f'warning: expected {expected_g} MXFP4 prefix records, got {len(prefix_descs)}; '
                'the dump may be truncated')
        g_indices = np.asarray(
            [desc - prefix_desc_begin for desc in prefix_descs], dtype=np.uint32)
        if not np.array_equal(g_indices, np.arange(len(prefix_descs), dtype=np.uint32)):
            print(f'warning: non-contiguous MXFP4 prefix descriptors: {prefix_descs}')
        prefix_lengths = {prefix_by_desc[desc].size for desc in prefix_descs}
        if len(prefix_lengths) != 1:
            raise RuntimeError(
                f'inconsistent MXFP4 prefix lengths for descriptors {prefix_descs}: {prefix_lengths}')
        prefix = np.stack(
            [prefix_by_desc[desc].astype(np.float32, copy=False) for desc in prefix_descs])
        np.save(output_dir / 'npu_mxfp4_acc_prefix.npy', prefix)
        np.save(
            output_dir / 'npu_mxfp4_acc_prefix_s2_begin.npy',
            np.asarray(prefix_s2_begin, dtype=np.uint32))
        np.save(output_dir / 'npu_mxfp4_acc_prefix_g_indices.npy', g_indices)


def main():
    parser = argparse.ArgumentParser(
        description="Parse QLI v2 DumpTensor console/log output and save every record as .npy.")
    parser.add_argument("input", type=Path, help="Text file containing AscendC DumpTensor output")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("qliv2_npu_dump"),
        help="Directory for parsed NumPy files")
    args = parser.parse_args()

    records = parse_dump_records(args.input)
    if not records:
        raise RuntimeError(f"no DumpTensor records found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_record_files(records, args.output_dir)
    _save_known_tensors(records, args.output_dir)
    print(f"saved {len(records)} DumpTensor records to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
