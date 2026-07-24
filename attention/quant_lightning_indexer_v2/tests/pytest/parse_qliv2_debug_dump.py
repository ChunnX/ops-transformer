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
CUBE_META_PRINT_RE = re.compile(
    r"QLIV2_CUBE_META\s+desc=(\d+)\s+values=([0-9]+(?:,[0-9]+){31})"
)


LD_PARTIAL_META_BASE = 6100000
LD_PARTIAL_SCORE_BASE = 6200000
LD_PARTIAL_INDEX_BASE = 6300000
LD_READ_META_BASE = 6400000
LD_READ_SCORE_BASE = 6500000
LD_READ_INDEX_BASE = 6600000
LD_MERGED_SCORE_BASE = 6700000
LD_MERGED_INDEX_BASE = 6800000
QK_DATA_BASE = 10000000
QK_META_BASE = 100000000
QK_CORE_DESC_STRIDE = 100000
QK_META_WORDS = 16
CUBE_DESC_BASE = 120000000
CUBE_DESC_PAIR_SIZE = 2
CUBE_DATA_DESC_OFFSET = 0
CUBE_META_DESC_OFFSET = 1
CUBE_META_WORDS = 32
CUBE_DESC_RANGE = 500000000
CUBE_STAGE_NAMES = {
    1: 'key_gm_nd',
    2: 'key_l1_nz',
    3: 'query_gm_nd',
    4: 'query_l1_nz',
    5: 'l0c_nz',
}


def _dtype_from_name(name):
    normalized = name.upper()
    if "UINT16" in normalized:
        return np.uint16
    if "UINT32" in normalized:
        return np.uint32
    if "INT32" in normalized:
        return np.int32
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
    for match in CUBE_META_PRINT_RE.finditer(text):
        values = np.fromstring(match.group(2), sep=",", dtype=np.uint64).astype(np.uint32)
        records.append({
            "desc": int(match.group(1)),
            "dtype_name": "PRINT_UINT32",
            "dump_size": CUBE_META_WORDS,
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


def _last_record_map(records, desc_begin, desc_end):
    return {
        record['desc']: record['values']
        for record in records
        if desc_begin <= record['desc'] < desc_end
    }


def _save_qk_tensors(records, output_dir):
    meta_records = [
        record for record in records
        if QK_META_BASE <= record['desc'] < QK_META_BASE + 10000000
    ]
    meta_ordinals = defaultdict(int)
    for meta_record in meta_records:
        ordinal = meta_ordinals[meta_record['desc']]
        meta_ordinals[meta_record['desc']] += 1
        meta = meta_record['values'].astype(np.uint32, copy=False)
        if meta.size < QK_META_WORDS:
            print(
                f"warning: QK metadata desc={meta_record['desc']} has "
                f"{meta.size} values, expected {QK_META_WORDS}")
            continue

        record_id = meta_record['desc'] - QK_META_BASE
        data_desc = QK_DATA_BASE + record_id
        data_records = _records_for_desc(records, data_desc)
        if ordinal >= len(data_records):
            print(
                f"warning: QK metadata desc={meta_record['desc']} ordinal={ordinal} "
                f"has no matching data desc={data_desc}")
            continue

        block_id = int(meta[1])
        ai_core_idx = int(meta[2])
        cur_s2_idx = int(meta[5])
        cur_aiv_s1_idx = int(meta[6])
        local_row = int(meta[7])
        g_idx = int(meta[8])
        qk_dump_begin = int(meta[9])
        qk_dump_len = int(meta[10])
        qk_block_offset = int(meta[11])
        pingpong = int(meta[12])
        qk_group_stride = int(meta[13])
        qk_row_stride = int(meta[14])
        target_s1_idx = int(meta[15])
        qk_base_index = local_row * qk_row_stride + g_idx * qk_group_stride + qk_block_offset

        prefix = (
            f"npu_qk_block{block_id}_aic{ai_core_idx}_s2_{cur_s2_idx}_"
            f"localrow_{local_row}_g_{g_idx}")
        if len(data_records) > 1:
            prefix += f"_{ordinal}"
        np.save(output_dir / f"{prefix}_meta.npy", meta)
        np.save(output_dir / f"{prefix}.npy", data_records[ordinal])
        print(
            f"[QK meta] blockId={block_id} aiCoreIdx={ai_core_idx} pingpong={pingpong} "
            f"curS2Idx={cur_s2_idx} curAivS1Idx={cur_aiv_s1_idx} "
            f"targetS1Idx={target_s1_idx} localRow={local_row} gIdx={g_idx} "
            f"s2=[{qk_dump_begin},{qk_dump_begin + qk_dump_len}) "
            f"qkBaseIndex={qk_base_index} dataDesc={data_desc}")


def _combine_u64(low, high):
    return int(low) | (int(high) << 32)


def _cube_record_to_nd(stage, data, qk_head_dim, dim0, dim0_aligned, logical_s2_size, aux):
    if stage in (1, 3):
        expected = dim0 * qk_head_dim
        if data.size < expected:
            return None
        return data[:expected].reshape(dim0, qk_head_dim)
    if stage in (2, 4):
        c0 = 32
        if qk_head_dim % c0 != 0:
            return None
        expected = (qk_head_dim // c0) * dim0_aligned * c0
        if data.size < expected:
            return None
        nz = data[:expected].reshape(qk_head_dim // c0, dim0_aligned, c0)
        return nz.transpose(1, 0, 2).reshape(dim0_aligned, qk_head_dim)[:dim0]
    if stage == 5:
        c0 = 16
        n_aligned = aux
        if n_aligned == 0 or n_aligned % c0 != 0:
            return None
        expected = (n_aligned // c0) * dim0_aligned * c0
        if data.size < expected:
            return None
        nz = data[:expected].reshape(n_aligned // c0, dim0_aligned, c0)
        return nz.transpose(1, 0, 2).reshape(dim0_aligned, n_aligned)[:dim0, :logical_s2_size]
    return None


def _save_cube_tensors(records, output_dir):
    meta_records = [
        record for record in records
        if CUBE_DESC_BASE <= record['desc'] < CUBE_DESC_BASE + CUBE_DESC_RANGE and
        (record['desc'] - CUBE_DESC_BASE) % CUBE_DESC_PAIR_SIZE == CUBE_META_DESC_OFFSET
    ]
    meta_ordinals = defaultdict(int)
    for meta_record in meta_records:
        ordinal = meta_ordinals[meta_record['desc']]
        meta_ordinals[meta_record['desc']] += 1
        meta = meta_record['values'].astype(np.uint32, copy=False)
        if meta.size < CUBE_META_WORDS:
            print(
                f"warning: cube metadata desc={meta_record['desc']} has "
                f"{meta.size} values, expected {CUBE_META_WORDS}")
            continue

        record_id = (meta_record['desc'] - CUBE_DESC_BASE) // CUBE_DESC_PAIR_SIZE
        data_desc = (CUBE_DESC_BASE + record_id * CUBE_DESC_PAIR_SIZE +
                     CUBE_DATA_DESC_OFFSET)
        data_records = _records_for_desc(records, data_desc)
        if ordinal >= len(data_records):
            print(
                f"warning: cube metadata desc={meta_record['desc']} ordinal={ordinal} "
                f"has no matching data desc={data_desc}")
            continue

        stage = int(meta[1])
        stage_name = CUBE_STAGE_NAMES.get(stage, f'stage_{stage}')
        block_id = int(meta[2])
        b_idx = int(meta[3])
        n2_idx = int(meta[4])
        target_s1 = int(meta[5])
        g_s1_idx = int(meta[6])
        s2_start = int(meta[16])
        s2_size = int(meta[17])
        tensor_key_offset = _combine_u64(meta[8], meta[9])
        tensor_query_offset = _combine_u64(meta[10], meta[11])
        s1g_gm_offset = _combine_u64(meta[12], meta[13])
        s2_gm_offset = _combine_u64(meta[14], meta[15])
        data_element_offset = _combine_u64(meta[18], meta[19])
        data_element_count = _combine_u64(meta[20], meta[21])
        buffer_index = int(meta[22])
        qk_head_dim = int(meta[23])
        dim0 = int(meta[24])
        dim0_aligned = int(meta[25])
        layout = int(meta[26])
        aux = int(meta[27])
        g_size = int(meta[29])
        s1_base_size = int(meta[31])

        prefix = (
            f'npu_cube_{stage_name}_block{block_id}_b{b_idx}_n2_{n2_idx}_'
            f's1_{target_s1}_s2_{s2_start}_{s2_start + s2_size}')
        if len(data_records) > 1:
            prefix += f'_{ordinal}'
        data = data_records[ordinal]
        np.save(output_dir / f'{prefix}_meta.npy', meta)
        np.save(output_dir / f'{prefix}.npy', data)
        data_nd = _cube_record_to_nd(
            stage, data, qk_head_dim, dim0, dim0_aligned, s2_size, aux)
        if data_nd is not None:
            np.save(output_dir / f'{prefix}_nd.npy', data_nd)
            if stage in (3, 4, 5):
                target_m_start = (
                    (target_s1 - g_s1_idx * s1_base_size) * g_size - s1g_gm_offset)
                target_m_end = target_m_start + g_size
                if 0 <= target_m_start and target_m_end <= data_nd.shape[0]:
                    np.save(
                        output_dir / f'{prefix}_target_s1_nd.npy',
                        data_nd[target_m_start:target_m_end])
        else:
            print(
                f'warning: unable to restore cube record dataDesc={data_desc} to ND; '
                f'stage={stage} size={data.size} qkHeadDim={qk_head_dim} '
                f'dim0={dim0} dim0Aligned={dim0_aligned} aux={aux}')
        print(
            f'[cube meta] stage={stage_name} blockId={block_id} bIdx={b_idx} n2Idx={n2_idx} '
            f's1={target_s1} s2=[{s2_start},{s2_start + s2_size}) '
            f'tensorKeyOffset={tensor_key_offset} tensorQueryOffset={tensor_query_offset} '
            f's1gGmOffset={s1g_gm_offset} s2GmOffset={s2_gm_offset} '
            f'dataElementOffset={data_element_offset} dataElementCount={data_element_count} '
            f'buffer={buffer_index} qkHeadDim={qk_head_dim} dim0={dim0} '
            f'dim0Aligned={dim0_aligned} layout={layout} aux={aux} dataDesc={data_desc}')


def _save_ld_tensors(records, output_dir):
    partial_meta = _last_record_map(records, LD_PARTIAL_META_BASE, LD_PARTIAL_META_BASE + 100000)
    if not partial_meta:
        print('[LD verify] no LD partial records; the target row did not enter the FD/LD path')
        return

    partials = {}
    for desc in sorted(partial_meta):
        partial_id = desc - LD_PARTIAL_META_BASE
        meta = partial_meta[desc].astype(np.uint32, copy=False)
        score_records = _records_for_desc(records, LD_PARTIAL_SCORE_BASE + partial_id)
        index_records = _records_for_desc(records, LD_PARTIAL_INDEX_BASE + partial_id)
        if not score_records or not index_records:
            print(f'[LD verify] partial={partial_id} is missing score or index dump')
            continue
        score = score_records[-1].astype(np.uint16, copy=False)
        index = index_records[-1].astype(np.int32, copy=False)
        workspace_idx = int(meta[7])
        absolute_s1 = int(meta[4])
        partials[(workspace_idx, absolute_s1)] = (score, index, meta)
        prefix = output_dir / f'ld_partial_ws{workspace_idx}_s1_{absolute_s1}'
        np.save(prefix.with_name(prefix.name + '_meta.npy'), meta)
        np.save(prefix.with_name(prefix.name + '_score_key.npy'), score)
        np.save(prefix.with_name(prefix.name + '_score.npy'), decode_bf16_sortable_keys(score))
        np.save(prefix.with_name(prefix.name + '_index.npy'), index)

    read_meta = _last_record_map(records, LD_READ_META_BASE, LD_READ_META_BASE + 100000)
    previous_score = None
    previous_index = None
    for desc in sorted(read_meta):
        round_idx = desc - LD_READ_META_BASE
        meta = read_meta[desc].astype(np.uint32, copy=False)
        score_records = _records_for_desc(records, LD_READ_SCORE_BASE + round_idx)
        index_records = _records_for_desc(records, LD_READ_INDEX_BASE + round_idx)
        merged_score_records = _records_for_desc(records, LD_MERGED_SCORE_BASE + round_idx)
        merged_index_records = _records_for_desc(records, LD_MERGED_INDEX_BASE + round_idx)
        if not score_records or not index_records:
            print(f'[LD verify] round={round_idx} is missing read score or index dump')
            continue

        read_score = score_records[-1].astype(np.uint16, copy=False)
        read_index = index_records[-1].astype(np.int32, copy=False)
        workspace_begin = int(meta[5])
        workspace_num = int(meta[6])
        previous_offset = int(meta[8])
        topk_align16 = int(meta[9])
        valid_len = int(meta[10])
        aligned_len = int(meta[11])
        absolute_s1 = int(meta[4])

        expected_score = []
        expected_index = []
        if previous_offset:
            if previous_score is None or previous_index is None:
                print(f'[LD verify] round={round_idx} has a previous-result prefix but no prior merge dump')
            else:
                expected_score.append(previous_score[:previous_offset])
                expected_index.append(previous_index[:previous_offset])
        for workspace_idx in range(workspace_begin, workspace_begin + workspace_num):
            partial = partials.get((workspace_idx, absolute_s1))
            if partial is None:
                print(f'[LD verify] round={round_idx} cannot find workspace={workspace_idx}, s1={absolute_s1}')
                continue
            expected_score.append(partial[0][:topk_align16])
            expected_index.append(partial[1][:topk_align16])

        if expected_score:
            expected_score = np.concatenate(expected_score)
            expected_index = np.concatenate(expected_index)
            compare_len = min(valid_len, expected_score.size, read_score.size)
            score_equal = np.array_equal(read_score[:compare_len], expected_score[:compare_len])
            index_equal = np.array_equal(read_index[:compare_len], expected_index[:compare_len])
            print(
                f'[LD verify] round={round_idx} workspace=[{workspace_begin},'
                f'{workspace_begin + workspace_num}) score_equal={score_equal} index_equal={index_equal}')

        score_tail_ok = np.all(read_score[valid_len:aligned_len] == 0)
        index_tail_ok = np.all(read_index[valid_len:aligned_len] == -1)
        print(
            f'[LD verify] round={round_idx} aligned_tail '
            f'score_zero={score_tail_ok} index_neg1={index_tail_ok}')

        round_prefix = output_dir / f'ld_round_{round_idx}'
        np.save(round_prefix.with_name(round_prefix.name + '_meta.npy'), meta)
        np.save(round_prefix.with_name(round_prefix.name + '_read_score_key.npy'), read_score)
        np.save(round_prefix.with_name(round_prefix.name + '_read_score.npy'),
                decode_bf16_sortable_keys(read_score))
        np.save(round_prefix.with_name(round_prefix.name + '_read_index.npy'), read_index)

        if merged_score_records and merged_index_records:
            previous_score = merged_score_records[-1].astype(np.uint16, copy=False)
            previous_index = merged_index_records[-1].astype(np.int32, copy=False)
            np.save(round_prefix.with_name(round_prefix.name + '_merged_score_key.npy'), previous_score)
            np.save(round_prefix.with_name(round_prefix.name + '_merged_score.npy'),
                    decode_bf16_sortable_keys(previous_score))
            np.save(round_prefix.with_name(round_prefix.name + '_merged_index.npy'), previous_index)


def _save_known_tensors(records, output_dir):
    _save_qk_tensors(records, output_dir)
    _save_cube_tensors(records, output_dir)

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
    if use_ld:
        values = _records_for_desc(records, 600002)
        if values:
            np.save(output_dir / "npu_ld_topk_value_kernel.npy",
                    values[-1].astype(np.float32, copy=False))

    fixed_desc_names = {
        200000: 'npu_weight',
        200001: 'npu_weight_bf16_packed',
        500000: "npu_query_scale_raw",
        500001: "npu_key_scale_raw",
        500002: "npu_seqused_q",
        500003: "npu_seqused_k",
        510000: "npu_block_table",
        590000: 'npu_metadata_raw',
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

    _save_ld_tensors(records, output_dir)


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
