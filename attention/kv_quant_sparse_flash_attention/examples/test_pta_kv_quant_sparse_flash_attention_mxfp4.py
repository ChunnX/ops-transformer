#!/usr/bin/env python3
"""Standalone selectable PTA precision cases for the D400 MXFP4 QSFA path."""

import argparse
from dataclasses import dataclass
import math

import torch
import torch_npu


N2 = 1
NOPE_D = 512
ROPE_D = 64
QUERY_D = NOPE_D + ROPE_D
FP4_BYTES = NOPE_D // 2
SCALE_COUNT = NOPE_D // 32
KEY_D = FP4_BYTES + ROPE_D * 2 + SCALE_COUNT
MAIN_CASE = "bf16_g128_main"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    q_dtype: torch.dtype
    layout_query: str
    layout_kv: str
    g: int
    q_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    sparse_rows: tuple[tuple[int, ...], ...] = ()
    scale_pattern: str = "normal"
    query_pattern: str = "normal"
    block_size: int = 0
    block_table: tuple[tuple[int, ...], ...] = ()
    rtol: float = 2e-2
    atol: float = 5e-2
    bit_exact: bool = False


CASE_LIST = (
    CaseSpec(
        name=MAIN_CASE,
        q_dtype=torch.bfloat16,
        layout_query="BSND",
        layout_kv="BSND",
        g=128,
        q_lengths=(8, 8, 8, 8),
        kv_lengths=(135, 135, 135, 135),
    ),
    CaseSpec(
        name="fp16_g1_tail3",
        q_dtype=torch.float16,
        layout_query="BSND",
        layout_kv="BSND",
        g=1,
        q_lengths=(1,),
        kv_lengths=(3,),
        sparse_rows=((2, 0, 1),),
    ),
    CaseSpec(
        name="bf16_g1_e8m0_edges",
        q_dtype=torch.bfloat16,
        layout_query="BSND",
        layout_kv="BSND",
        g=1,
        q_lengths=(1,),
        kv_lengths=(1,),
        sparse_rows=((0,),),
        scale_pattern="e8m0_edges",
        query_pattern="zero",
        bit_exact=True,
    ),
    CaseSpec(
        name="tnd_bf16_b2_prefix",
        q_dtype=torch.bfloat16,
        layout_query="TND",
        layout_kv="TND",
        g=1,
        q_lengths=(1, 1),
        kv_lengths=(3, 4),
        sparse_rows=((0, 1, 2, -1), (0, 1, 3, 2)),
    ),
    CaseSpec(
        name="pa_bf16_block_table",
        q_dtype=torch.bfloat16,
        layout_query="BSND",
        layout_kv="PA_BSND",
        g=128,
        q_lengths=(1,),
        kv_lengths=(3,),
        sparse_rows=((0, 1, 2, -1),),
        block_size=16,
        block_table=((1,),),
    ),
)
CASES = {case.name: case for case in CASE_LIST}


def pack_fp4_e2m1(fp4_codes: torch.Tensor) -> torch.Tensor:
    if fp4_codes.dtype != torch.uint8 or fp4_codes.shape[-1] != NOPE_D:
        raise ValueError(f"fp4_codes must be uint8 [..., {NOPE_D}]")
    if torch.any(fp4_codes > 0xF):
        raise ValueError("each FP4 E2M1 code must be in [0, 15]")
    low = fp4_codes[..., 0::2].to(torch.int16)
    high = fp4_codes[..., 1::2].to(torch.int16)
    return (low | (high << 4)).to(torch.uint8).view(torch.int8)


def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8).to(torch.int16)
    return torch.stack((raw & 0xF, raw >> 4), dim=-1).flatten(-2)


def make_packed_records(case: CaseSpec, token_ids: torch.Tensor) -> torch.Tensor:
    prefix_shape = token_ids.shape[:-1]
    if case.scale_pattern == "e8m0_edges":
        fp4_codes = torch.ones((*prefix_shape, NOPE_D), dtype=torch.uint8)
        key_rope = torch.zeros((*prefix_shape, ROPE_D), dtype=case.q_dtype)
        edge_scales = (0, 1, 126, 127, 128, 129, 253, 254, 120, 121, 122, 123, 124, 125, 130, 131)
        scale_e8m0 = torch.tensor(edge_scales, dtype=torch.uint8).reshape(
            *((1,) * len(prefix_shape)), SCALE_COUNT
        ).expand(*prefix_shape, SCALE_COUNT)
    else:
        d = torch.arange(NOPE_D, dtype=torch.int32).reshape(*((1,) * len(prefix_shape)), NOPE_D)
        fp4_codes = torch.remainder(token_ids + d, 16).to(torch.uint8)
        rope_d = torch.arange(ROPE_D, dtype=torch.float32).reshape(*((1,) * len(prefix_shape)), ROPE_D)
        key_rope = (
            0.03125 * (token_ids.float() + 1.0) * (torch.remainder(rope_d, 9) - 4.0)
        ).to(case.q_dtype)
        group = torch.arange(SCALE_COUNT, dtype=torch.int32).reshape(*((1,) * len(prefix_shape)), SCALE_COUNT)
        scale_e8m0 = (126 + torch.remainder(token_ids + group, 4)).to(torch.uint8)

    key = torch.cat(
        (pack_fp4_e2m1(fp4_codes), key_rope.view(torch.int8), scale_e8m0.view(torch.int8)),
        dim=-1,
    ).contiguous()
    if key.shape[-1] != KEY_D:
        raise AssertionError(f"unexpected packed key D: {key.shape[-1]}")
    return key


def make_query_bsnd(case: CaseSpec) -> torch.Tensor:
    batch = len(case.q_lengths)
    max_s1 = max(case.q_lengths)
    if case.query_pattern == "zero":
        return torch.zeros((batch, max_s1, case.g, QUERY_D), dtype=case.q_dtype)
    b = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1, 1)
    s1 = torch.arange(max_s1, dtype=torch.float32).reshape(1, max_s1, 1, 1)
    head = torch.arange(case.g, dtype=torch.float32).reshape(1, 1, case.g, 1)
    d = torch.arange(QUERY_D, dtype=torch.float32).reshape(1, 1, 1, QUERY_D)
    query = 0.00390625 * torch.remainder(d * 3 + head * 5 + b * 7 + s1 * 11, 29)
    return (query - 0.00390625 * 14).to(case.q_dtype)


def logical_token_ids(case: CaseSpec, batch_idx: int, count: int) -> torch.Tensor:
    base = sum(case.kv_lengths[:batch_idx]) + batch_idx * 17
    return (base + torch.arange(count, dtype=torch.int32)).reshape(count, N2, 1)


def build_key_value_and_block_table(case: CaseSpec):
    batch = len(case.kv_lengths)
    if case.layout_kv == "BSND":
        max_s2 = max(case.kv_lengths)
        records = [make_packed_records(case, logical_token_ids(case, b, max_s2)) for b in range(batch)]
        key = torch.stack(records)
        value = torch.zeros((batch, max_s2, N2, NOPE_D), dtype=torch.int8)
        return key, value, None

    if case.layout_kv == "TND":
        records = [
            make_packed_records(case, logical_token_ids(case, b, case.kv_lengths[b])) for b in range(batch)
        ]
        key = torch.cat(records, dim=0)
        value = torch.zeros((sum(case.kv_lengths), N2, NOPE_D), dtype=torch.int8)
        return key, value, None

    block_table = torch.tensor(case.block_table, dtype=torch.int32)
    physical_blocks = int(block_table.max()) + 1
    sentinel_ids = (10000 + torch.arange(physical_blocks * case.block_size, dtype=torch.int32)).reshape(
        physical_blocks, case.block_size, N2, 1
    )
    key = make_packed_records(case, sentinel_ids)
    for batch_idx, kv_length in enumerate(case.kv_lengths):
        records = make_packed_records(case, logical_token_ids(case, batch_idx, kv_length))
        for token_idx in range(kv_length):
            logical_block = token_idx // case.block_size
            block_offset = token_idx % case.block_size
            physical_block = int(block_table[batch_idx, logical_block])
            key[physical_block, block_offset] = records[token_idx]
    value = torch.zeros((physical_blocks, case.block_size, N2, NOPE_D), dtype=torch.int8)
    return key, value, block_table


def make_sparse_indices(case: CaseSpec) -> torch.Tensor:
    rows = list(case.sparse_rows)
    if not rows:
        for batch_idx, q_length in enumerate(case.q_lengths):
            rows.extend(tuple(range(case.kv_lengths[batch_idx])) for _ in range(q_length))
    if len(rows) != sum(case.q_lengths):
        raise ValueError(f"{case.name}: sparse row count must equal total query tokens")
    width = max(len(row) for row in rows)
    padded = [tuple(row) + (-1,) * (width - len(row)) for row in rows]
    sparse_tnd = torch.tensor(padded, dtype=torch.int32).reshape(sum(case.q_lengths), N2, width)
    if case.layout_query == "TND":
        return sparse_tnd

    sparse_bsnd = torch.full(
        (len(case.q_lengths), max(case.q_lengths), N2, width), -1, dtype=torch.int32
    )
    offset = 0
    for batch_idx, q_length in enumerate(case.q_lengths):
        sparse_bsnd[batch_idx, :q_length] = sparse_tnd[offset:offset + q_length]
        offset += q_length
    return sparse_bsnd


def cumulative_lengths(lengths: tuple[int, ...]) -> torch.Tensor:
    result = []
    total = 0
    for length in lengths:
        total += length
        result.append(total)
    return torch.tensor(result, dtype=torch.int32)


def build_inputs(case: CaseSpec):
    if len(case.q_lengths) != len(case.kv_lengths):
        raise ValueError(f"{case.name}: q_lengths and kv_lengths must have the same batch count")
    query_bsnd = make_query_bsnd(case)
    if case.layout_query == "TND":
        query = torch.cat(
            [query_bsnd[b, :case.q_lengths[b]] for b in range(len(case.q_lengths))], dim=0
        )
        actual_q = cumulative_lengths(case.q_lengths)
    else:
        query = query_bsnd
        actual_q = torch.tensor(case.q_lengths, dtype=torch.int32)

    key, value, block_table = build_key_value_and_block_table(case)
    actual_kv = (
        cumulative_lengths(case.kv_lengths)
        if case.layout_kv == "TND"
        else torch.tensor(case.kv_lengths, dtype=torch.int32)
    )
    return {
        "query": query,
        "key": key,
        "value": value,
        "sparse_indices": make_sparse_indices(case),
        "key_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "value_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "block_table": block_table,
        "actual_seq_lengths_query": actual_q,
        "actual_seq_lengths_kv": actual_kv,
    }


def decode_packed_key(key: torch.Tensor, q_dtype: torch.dtype) -> torch.Tensor:
    packed_nope = key[..., :FP4_BYTES]
    rope_bytes = key[..., FP4_BYTES:FP4_BYTES + ROPE_D * 2]
    scale_bytes = key[..., FP4_BYTES + ROPE_D * 2:].view(torch.uint8)
    fp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    fp4_values = fp4_lut[unpack_fp4_e2m1(packed_nope).long()]
    scale_values = torch.ldexp(
        torch.ones(scale_bytes.shape, dtype=torch.float32), scale_bytes.to(torch.int32) - 127
    )
    nope = (fp4_values * torch.repeat_interleave(scale_values, 32, dim=-1)).to(torch.bfloat16)
    if q_dtype == torch.float16:
        nope = nope.to(torch.float16)
    rope = rope_bytes.contiguous().view(q_dtype)
    return torch.cat((nope, rope), dim=-1)


def get_query_batch(case: CaseSpec, inputs, batch_idx: int) -> torch.Tensor:
    if case.layout_query == "BSND":
        return inputs["query"][batch_idx, :case.q_lengths[batch_idx]]
    start = sum(case.q_lengths[:batch_idx])
    return inputs["query"][start:start + case.q_lengths[batch_idx]]


def get_sparse_batch(case: CaseSpec, inputs, batch_idx: int) -> torch.Tensor:
    if case.layout_query == "BSND":
        return inputs["sparse_indices"][batch_idx, :case.q_lengths[batch_idx], 0]
    start = sum(case.q_lengths[:batch_idx])
    return inputs["sparse_indices"][start:start + case.q_lengths[batch_idx], 0]


def get_key_batch(case: CaseSpec, inputs, decoded_key: torch.Tensor, batch_idx: int) -> torch.Tensor:
    kv_length = case.kv_lengths[batch_idx]
    if case.layout_kv == "BSND":
        return decoded_key[batch_idx, :kv_length, 0]
    if case.layout_kv == "TND":
        start = sum(case.kv_lengths[:batch_idx])
        return decoded_key[start:start + kv_length, 0]

    rows = []
    for token_idx in range(kv_length):
        logical_block = token_idx // case.block_size
        block_offset = token_idx % case.block_size
        physical_block = int(inputs["block_table"][batch_idx, logical_block])
        rows.append(decoded_key[physical_block, block_offset, 0])
    return torch.stack(rows)


def cube_input_value(tensor: torch.Tensor, q_dtype: torch.dtype) -> torch.Tensor:
    tensor = tensor.float()
    if q_dtype == torch.bfloat16:
        tensor = torch.where(tensor.abs() < 2.0 ** -126, torch.zeros_like(tensor), tensor)
    return tensor


def build_golden(case: CaseSpec, inputs) -> torch.Tensor:
    decoded_key = decode_packed_key(inputs["key"], case.q_dtype)
    if case.layout_query == "TND":
        output = torch.zeros((sum(case.q_lengths), case.g, NOPE_D), dtype=torch.float32)
    else:
        output = torch.zeros(
            (len(case.q_lengths), max(case.q_lengths), case.g, NOPE_D), dtype=torch.float32
        )
    q_offset = 0
    scale = 1.0 / math.sqrt(float(QUERY_D))

    for batch_idx, actual_s1 in enumerate(case.q_lengths):
        actual_s2 = case.kv_lengths[batch_idx]
        query_batch = get_query_batch(case, inputs, batch_idx)
        sparse_batch = get_sparse_batch(case, inputs, batch_idx)
        key_batch = cube_input_value(get_key_batch(case, inputs, decoded_key, batch_idx), case.q_dtype)
        for s1_idx in range(actual_s1):
            causal_limit = actual_s2 - actual_s1 + s1_idx + 1
            sparse_row = sparse_batch[s1_idx]
            valid_count = min(sparse_row.numel(), max(causal_limit, 0))
            selected = []
            for token_idx in sparse_row[:valid_count].tolist():
                if token_idx == -1:
                    break
                if 0 <= token_idx < causal_limit:
                    selected.append(token_idx)
            if not selected:
                continue
            query = cube_input_value(query_batch[s1_idx], case.q_dtype)
            key = key_batch[selected]
            scores = torch.matmul(query, key.transpose(0, 1)) * scale
            probability = torch.softmax(scores, dim=-1).to(case.q_dtype).float()
            result = torch.matmul(probability, key[:, :NOPE_D])
            if case.layout_query == "TND":
                output[q_offset + s1_idx] = result
            else:
                output[batch_idx, s1_idx] = result
        q_offset += actual_s1
    return output


def run_operator(case: CaseSpec, inputs) -> torch.Tensor:
    block_table = None if inputs["block_table"] is None else inputs["block_table"].npu()
    return torch_npu.npu_kv_quant_sparse_flash_attention(
        inputs["query"].npu(),
        inputs["key"].npu(),
        inputs["value"].npu(),
        inputs["sparse_indices"].npu(),
        1.0 / math.sqrt(float(QUERY_D)),
        2,
        2,
        key_dequant_scale=inputs["key_dequant_scale"].npu(),
        value_dequant_scale=inputs["value_dequant_scale"].npu(),
        block_table=block_table,
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"].npu(),
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"].npu(),
        sparse_block_size=1,
        layout_query=case.layout_query,
        layout_kv=case.layout_kv,
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )


def compare(case: CaseSpec, actual: torch.Tensor, expected: torch.Tensor):
    actual_cpu = actual.cpu().contiguous()
    expected_q = expected.to(case.q_dtype).contiguous()
    if tuple(actual_cpu.shape) != tuple(expected_q.shape):
        raise AssertionError(f"output shape={tuple(actual_cpu.shape)}, expected={tuple(expected_q.shape)}")
    if case.bit_exact:
        actual_bits = actual_cpu.view(torch.uint16)
        expected_bits = expected_q.view(torch.uint16)
        mismatch = torch.nonzero(actual_bits != expected_bits)
        if mismatch.numel() != 0:
            flat_idx = int(torch.nonzero(actual_bits.flatten() != expected_bits.flatten())[0])
            raise AssertionError(
                f"bit mismatch at {flat_idx}: actual=0x{int(actual_bits.flatten()[flat_idx]):04x}, "
                f"expected=0x{int(expected_bits.flatten()[flat_idx]):04x}"
            )
        return "bit_exact=true"

    actual_f32 = actual_cpu.float()
    expected_f32 = expected.float()
    close = torch.isclose(actual_f32, expected_f32, rtol=case.rtol, atol=case.atol)
    close_ratio = float(close.float().mean())
    max_abs_error = float((actual_f32 - expected_f32).abs().max())
    if close_ratio < 0.99:
        raise AssertionError(
            f"close_ratio={close_ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
            f"actual[:8]={actual_f32.flatten()[:8].tolist()}, "
            f"expected[:8]={expected_f32.flatten()[:8].tolist()}"
        )
    return f"close_ratio={close_ratio:.6f}, max_abs_error={max_abs_error:.6f}"


def run_case(case: CaseSpec):
    print(
        f"[RUN] {case.name}: q_dtype={case.q_dtype}, layout={case.layout_query}/{case.layout_kv}, "
        f"B={len(case.q_lengths)}, G={case.g}, q_lengths={case.q_lengths}, kv_lengths={case.kv_lengths}",
        flush=True,
    )
    inputs = build_inputs(case)
    expected = build_golden(case, inputs)
    actual = run_operator(case, inputs)
    torch.npu.synchronize()
    result = compare(case, actual, expected)
    print(f"[PASS] {case.name} {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", action="append", choices=("all", *CASES), default=[])
    args = parser.parse_args()

    if args.list:
        for case in CASE_LIST:
            print(
                f"{case.name}: dtype={case.q_dtype}, layout={case.layout_query}/{case.layout_kv}, "
                f"B={len(case.q_lengths)}, G={case.g}, q={case.q_lengths}, kv={case.kv_lengths}"
            )
        return

    selected = list(CASES) if "all" in args.case else (args.case or [MAIN_CASE])
    selected = list(dict.fromkeys(selected))
    torch.npu.set_device(args.device)
    failures = []
    for case_name in selected:
        try:
            run_case(CASES[case_name])
        except Exception as error:
            failures.append(case_name)
            print(f"[FAIL] {case_name}: {error}", flush=True)
    print(f"summary: executed={len(selected)} passed={len(selected) - len(failures)} failed={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
