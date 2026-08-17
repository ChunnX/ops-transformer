"""Precision checker for cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale.

The script builds inputs with ``make_inputs``, computes the CPU numpy golden
with ``golden.py``, runs the op path, and checks that the number of elements
whose absolute error is greater than the configured tolerance stays within the
allowed ratio. The generated cases mirror the torch_extension example: NTD/TND
layouts, current Nq/Nk/Nv combinations, multi-batch history offsets,
non-contiguous slot mappings, long sequences, and d-padded cache views.

Example:
    python3 qkv_rms_norm_rope_cache_with_k_scale_precision.py --list-cases
    python3 qkv_rms_norm_rope_cache_with_k_scale_precision.py
    python3 qkv_rms_norm_rope_cache_with_k_scale_precision.py --case n16_kv2_decode1_ntd --mode inplace

The default asset directory is this script's directory. Override it with
``--assets-dir`` if the assets are elsewhere. The script uses the installed
``cann_ops_transformer`` package by default; pass ``--torch-extension-dir`` only
when a source-tree package import is required.
"""

import argparse
import importlib
import sys
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import torch_npu
except ImportError:
    torch_npu = None


DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent
HEAD_SIZE = 128
DEFAULT_EPSILON = 1e-6
FP8_ATOL = 1.0
FLOAT_ATOL = 1e-2
MAX_BAD_RATIO = 1e-2
INT8_MAX_ABS_DIFF = 1
ROTATION_SEED_OFFSET = 2000003


@dataclass
class CompareResult:
    name: str
    passed: bool
    total: int
    bad_count: int
    bad_ratio: float
    max_abs_diff: float
    first_bad: Optional[Tuple[int, ...]]
    first_actual: Optional[float]
    first_expected: Optional[float]


@dataclass(frozen=True)
class InputCase:
    """Records parameters for ``make_inputs`` plus case metadata.

    ``post_func`` is called after ``make_inputs`` returns, allowing per-case
    modifications of the input dict (e.g. overriding slot_mapping or zeroing
    specific tensor regions).
    """
    case_name: str
    batch_seq_lens: Tuple[int, ...]
    seq_lens: Tuple[int, ...]
    Nq: int
    Nk: int
    Nv: int
    layout_qkv: str
    layout_q_out: str = ""
    slot_pattern: str = "linear"
    D: int = HEAD_SIZE
    block_size: int = 16
    block_num: int = 8
    cache_stride: str = "contiguous"
    cache_pad: int = 0
    seed: int = 0
    post_func: Optional[Callable] = None
    omit_layout_args: bool = False


def resolve_layout_q_out(case):
    return case.layout_q_out or "NTD"


def prepare_torch_extension_import(torch_extension_dir):
    if torch_extension_dir is None:
        return
    torch_extension_dir = Path(torch_extension_dir).expanduser().resolve()
    if torch_extension_dir.is_dir() and str(torch_extension_dir) not in sys.path:
        sys.path.insert(0, str(torch_extension_dir))


def load_golden(assets_dir):
    assets_dir = Path(assets_dir).resolve()
    golden_file = assets_dir / "golden.py"
    if not golden_file.is_file():
        raise FileNotFoundError(f"golden.py must exist under {assets_dir}")

    sys.path.insert(0, str(assets_dir))
    importlib.invalidate_caches()
    return importlib.import_module("golden")


def make_slot_mapping(case, total_tokens, rng):
    capacity = case.block_num * case.block_size
    if total_tokens > capacity:
        raise ValueError(f"{case.case_name}: total tokens exceed cache slot capacity")

    token_ids = np.arange(total_tokens, dtype=np.int64)
    if case.slot_pattern == "linear":
        slots = token_ids
    elif case.slot_pattern == "reverse":
        slots = np.arange(total_tokens - 1, -1, -1, dtype=np.int64)
    elif case.slot_pattern == "stride3":
        slots = token_ids * 3 + 1
    elif case.slot_pattern == "random":
        slots = rng.permutation(capacity)[:total_tokens].astype(np.int64)
    else:
        raise ValueError(f"{case.case_name}: unknown slot pattern {case.slot_pattern}")

    if int(np.max(slots)) >= capacity or np.unique(slots).size != total_tokens:
        raise ValueError(f"{case.case_name}: slot pattern must produce unique slots within cache capacity")
    return slots.astype(np.int32)


def validate_case(case):
    if min(case.Nq, case.Nk, case.Nv) <= 0:
        raise ValueError(f"{case.case_name}: Nq, Nk, and Nv must be positive")
    if case.Nq > 64:
        raise ValueError(f"{case.case_name}: current kernel requires Nq <= 64")
    if case.Nq != 8 * case.Nk:
        raise ValueError(f"{case.case_name}: current kernel requires Nq == 8 * Nk")
    if case.Nk != case.Nv:
        raise ValueError(f"{case.case_name}: current kernel requires Nk == Nv")
    if case.layout_qkv not in ("NTD", "TND"):
        raise ValueError(f"{case.case_name}: layout_qkv must be NTD or TND")
    layout_q_out = resolve_layout_q_out(case)
    if layout_q_out not in ("NTD", "TND"):
        raise ValueError(f"{case.case_name}: layout_q_out must be NTD or TND")
    if case.layout_qkv == "NTD" and layout_q_out == "TND":
        raise ValueError(f"{case.case_name}: layout_qkv=NTD with layout_q_out=TND is not supported")
    if case.D != HEAD_SIZE:
        raise ValueError(f"{case.case_name}: current kernel only supports D={HEAD_SIZE}")
    if len(case.seq_lens) != len(case.batch_seq_lens):
        raise ValueError(f"{case.case_name}: seq_lens length must match batch_seq_lens length")
    if any(seq_len < batch_len for batch_len, seq_len in zip(case.batch_seq_lens, case.seq_lens)):
        raise ValueError(f"{case.case_name}: seq_lens values must be >= current sequence lengths")
    if case.cache_stride not in ("contiguous", "d_pad"):
        raise ValueError(f"{case.case_name}: cache_stride must be contiguous or d_pad")
    if case.cache_stride == "d_pad" and case.cache_pad <= 0:
        raise ValueError(f"{case.case_name}: cache_pad must be positive for d_pad cache stride")


def make_inputs(golden_mod, case):
    """Build one set of numpy inputs for the golden + NPU op.

    The numpy side keeps logical cache arrays for golden calculation. When the
    NPU op is run, ``make_torch_inputs`` recreates the requested cache view
    stride directly on device so d-padded cache cases exercise the current
    op_host stride path.
    """
    validate_case(case)
    rng = np.random.default_rng(case.seed)
    B = len(case.batch_seq_lens)
    T = int(sum(case.batch_seq_lens))
    query_start_loc = np.zeros(B + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(case.batch_seq_lens)
    seq_lens = np.asarray(case.seq_lens, dtype=np.int32)
    max_seq_len = int(np.max(seq_lens))

    H = case.Nq + case.Nk + case.Nv
    qkv_ntd = golden_mod.to_bf16(
        (rng.standard_normal((H, T, case.D)).astype(np.float32) * 0.5)
    )
    qkv = qkv_ntd if case.layout_qkv == "NTD" else np.transpose(qkv_ntd, (1, 0, 2)).copy()
    q_gamma = (rng.random(case.D, dtype=np.float32) + 0.5).astype(np.float32)
    k_gamma = (rng.random(case.D, dtype=np.float32) + 0.5).astype(np.float32)
    cos_sin = rng.standard_normal((max_seq_len, case.D)).astype(np.float32)
    rotation_rng = np.random.default_rng(case.seed + ROTATION_SEED_OFFSET)
    rotation = golden_mod.to_bf16(
        rotation_rng.standard_normal((case.D, case.D)).astype(np.float32) * (case.D ** -0.5)
    )
    v_scale = (rng.random(case.Nv, dtype=np.float32) + 0.5).astype(np.float32)
    slot_mapping = make_slot_mapping(case, T, rng)

    k_cache = golden_mod.cast_to_e4m3fn_sat_rint(
        np.zeros((case.block_num, case.Nk, case.block_size, case.D), dtype=np.float32))
    v_cache = golden_mod.cast_to_e4m3fn_sat_rint(
        np.zeros((case.block_num, case.Nv, case.block_size, case.D), dtype=np.float32))
    k_scale_cache = np.zeros(
        (case.block_num, case.Nk, case.block_size, 1), dtype=np.float32)

    return dict(
        qkv=qkv, q_gamma=q_gamma, k_gamma=k_gamma, cos_sin=cos_sin,
        slot_mapping=slot_mapping, k_cache=k_cache, v_cache=v_cache,
        k_scale_cache=k_scale_cache, query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        rotation=rotation, v_scale=v_scale,
        head_nums=[case.Nq, case.Nk, case.Nv],
        layout_qkv=case.layout_qkv,
        layout_q_out=resolve_layout_q_out(case),
        epsilon=DEFAULT_EPSILON,
        cache_stride=case.cache_stride, cache_pad=case.cache_pad,
        omit_layout_args=case.omit_layout_args,
    )


def build_case(golden_mod, ic):
    inp = make_inputs(golden_mod, ic)
    if ic.post_func is not None:
        ic.post_func(inp)
    return inp


def zero_first_qkv_token(inp):
    if inp.get("layout_qkv", "TND") == "NTD":
        inp["qkv"][:, 0, :] = 0.0
    else:
        inp["qkv"][0, :, :] = 0.0


def max_slot_tokens(case):
    capacity = case.block_num * case.block_size
    if case.slot_pattern == "stride3":
        return (capacity - 2) // 3 + 1
    return capacity


def make_varied_tnd_to_ntd_batch_seq_lens(case, index):
    old_total = sum(case.batch_seq_lens)
    min_total = len(case.batch_seq_lens)
    max_total = max_slot_tokens(case)
    if min_total > max_total:
        raise ValueError(f"{case.case_name}: minimum token count exceeds slot capacity")

    if old_total >= 1024:
        deltas = (17, -29, 53, -97, 127, -191, 257, -389)
    elif old_total >= 96:
        deltas = (5, -7, 11, -13, 17, -19)
    else:
        deltas = (1, 3, 5, 7, -1, -3, 9, -5)

    new_total = old_total
    for offset in range(len(deltas)):
        delta = deltas[(index + offset) % len(deltas)]
        candidate = min(max(old_total + delta, min_total), max_total)
        if candidate != old_total:
            new_total = candidate
            break
    if new_total == old_total and old_total < max_total:
        new_total = old_total + 1
    elif new_total == old_total and old_total > min_total:
        new_total = old_total - 1

    varied = list(case.batch_seq_lens)
    delta = new_total - old_total
    cursor = index % len(varied)
    while delta > 0:
        varied[cursor % len(varied)] += 1
        delta -= 1
        cursor += 1
    while delta < 0:
        pos = cursor % len(varied)
        if varied[pos] > 1:
            varied[pos] -= 1
            delta += 1
        cursor += 1
    return tuple(varied)


def with_tnd_to_ntd_cases(cases):
    extended = list(cases)
    for index, case in enumerate(cases):
        if case.layout_qkv != "TND" or resolve_layout_q_out(case) != "TND":
            continue
        if not case.case_name.endswith("_tnd"):
            raise ValueError(f"{case.case_name}: TND base case name must end with _tnd")

        varied_batch_seq_lens = make_varied_tnd_to_ntd_batch_seq_lens(case, index)
        history_offsets = tuple(
            seq_len - current for current, seq_len in zip(case.batch_seq_lens, case.seq_lens)
        )
        varied_seq_lens = tuple(
            current + offset for current, offset in zip(varied_batch_seq_lens, history_offsets)
        )
        extended.append(replace(
            case,
            case_name=f"{case.case_name[:-4]}_tnd_to_ntd",
            batch_seq_lens=varied_batch_seq_lens,
            seq_lens=varied_seq_lens,
            layout_q_out="NTD",
            seed=case.seed + 100000,
        ))
    return tuple(extended)


def performance_t16384_tnd_to_ntd_cases(seed):
    specs = (
        ("n64_kv8_perf_t16384_dpad_tnd_to_ntd", (64, 8, 8)),
        ("n16_kv2_perf_t16384_dpad_tnd_to_ntd", (16, 2, 2)),
        ("n8_kv1_perf_t16384_dpad_tnd_to_ntd", (8, 1, 1)),
    )
    return tuple(
        InputCase(
            case_name=name,
            batch_seq_lens=(16384,),
            seq_lens=(16384,),
            Nq=heads[0],
            Nk=heads[1],
            Nv=heads[2],
            layout_qkv="TND",
            layout_q_out="NTD",
            slot_pattern="random",
            seed=seed + index,
            block_num=1024,
            cache_stride="d_pad",
            cache_pad=8,
        )
        for index, (name, heads) in enumerate(specs)
    )


def precision_mb20_t16384_cases(seed):
    batch_seq_lens = (17, 257, 1, 1024, 511, 2048, 33, 4096, 128, 777,
                      64, 1536, 999, 2, 301, 819, 1234, 455, 688, 1394)
    history_offsets = (4096, 0, 8192, 512, 17, 2048, 0, 4096, 128, 3000,
                       0, 1024, 4096, 8192, 513, 0, 256, 2048, 1, 4096)
    seq_lens = tuple(seq + offset for seq, offset in zip(batch_seq_lens, history_offsets))
    specs = (
        ("n16_kv2_mb20_t16384_dpad_ntd", "NTD", "NTD"),
        ("n16_kv2_mb20_t16384_dpad_tnd", "TND", "TND"),
        ("n16_kv2_mb20_t16384_dpad_tnd_to_ntd", "TND", "NTD"),
    )
    return tuple(
        InputCase(
            case_name=name,
            batch_seq_lens=batch_seq_lens,
            seq_lens=seq_lens,
            Nq=16,
            Nk=2,
            Nv=2,
            layout_qkv=layout_qkv,
            layout_q_out=layout_q_out,
            slot_pattern="random",
            seed=seed + index,
            block_num=1024,
            cache_stride="d_pad",
            cache_pad=8,
        )
        for index, (name, layout_qkv, layout_q_out) in enumerate(specs)
    )


def default_layout_case(seed):
    return InputCase(
        case_name="n16_kv2_default_layout_tnd_to_ntd",
        batch_seq_lens=(19,),
        seq_lens=(19,),
        Nq=16,
        Nk=2,
        Nv=2,
        layout_qkv="TND",
        layout_q_out="NTD",
        slot_pattern="random",
        seed=seed,
        block_num=8,
        omit_layout_args=True,
    )


def get_input_cases():
    n16_specs = (
        ("decode1", (1,), (1,), "linear"),
        ("prefill8", (8,), (8,), "reverse"),
        ("prefill17", (17,), (17,), "stride3"),
        ("prefill33", (33,), (33,), "random"),
        ("mb_2_7", (2, 7), (2, 7), "linear"),
        ("offset_b0", (17, 1), (8209, 1), "stride3"),
        ("offset_b1", (1, 17), (1, 8209), "reverse"),
        ("offset_multi", (3, 5, 2), (4099, 5, 8194), "random"),
    )
    n8_specs = (
        ("decode1", (1,), (1,), "linear"),
        ("prefill8", (8,), (8,), "reverse"),
        ("prefill17", (17,), (17,), "stride3"),
        ("mb_2_7", (2, 7), (2, 7), "linear"),
        ("offset_b0", (17, 1), (8209, 1), "stride3"),
        ("offset_b1", (1, 17), (1, 8209), "reverse"),
        ("offset_multi", (3, 5, 2), (4099, 5, 8194), "random"),
    )
    n16_large_strided_specs = (
        ("long5379_dpad", (5379,), (5379,), "random", 512),
        ("long_offset_mb3217_2891_dpad", (3217, 2891), (11397, 3589), "random", 512),
        ("large96_dpad", (96,), (96,), "reverse", 32),
        ("large127_dpad", (127,), (127,), "random", 32),
        ("large128_stride3_dpad", (128,), (128,), "stride3", 32),
        ("large_mb32_64_dpad", (32, 64), (32, 64), "linear", 32),
        ("large_mb17_79_3_dpad", (17, 79, 3), (17, 79, 3), "random", 32),
        ("large_offset_multi_dpad", (33, 17, 49), (4129, 17, 8241), "random", 32),
    )
    n8_large_strided_specs = (
        ("long5863_dpad", (5863,), (5863,), "random", 512),
        ("long_offset_mb2753_3401_dpad", (2753, 3401), (10945, 7509), "random", 512),
        ("large96_dpad", (96,), (96,), "reverse", 32),
        ("large127_dpad", (127,), (127,), "random", 32),
        ("large_mb32_64_dpad", (32, 64), (32, 64), "linear", 32),
        ("large_mb17_79_3_dpad", (17, 79, 3), (17, 79, 3), "random", 32),
        ("large_offset_multi_dpad", (33, 17, 49), (4129, 17, 8241), "random", 32),
    )
    n64_specs = (
        ("decode1", (1,), (1,), "linear", 8, "contiguous", 0),
        ("prefill19", (19,), (19,), "reverse", 8, "contiguous", 0),
        ("prefill73", (73,), (73,), "random", 8, "contiguous", 0),
        ("mb5_11_23", (5, 11, 23), (5, 11, 23), "stride3", 8, "contiguous", 0),
        ("offset_b0", (17, 1), (8209, 1), "stride3", 8, "contiguous", 0),
        ("offset_b1", (1, 29), (1, 8221), "reverse", 8, "contiguous", 0),
        ("offset_multi", (7, 13, 5), (4103, 13, 8197), "random", 8, "contiguous", 0),
        ("long5819_dpad", (5819,), (5819,), "random", 512, "d_pad", 8),
    )
    long_16k_specs = (("long16384_dpad", (16384,), (16384,), "random", 1024, "d_pad", 8),)

    cases = []
    seed = 20260623
    for heads_name, specs, heads in (("n16_kv2", n16_specs, (16, 2, 2)), ("n8_kv1", n8_specs, (8, 1, 1))):
        for suffix, batch_seq_lens, seq_lens, slot_pattern in specs:
            for layout in ("NTD", "TND"):
                cases.append(InputCase(
                    case_name=f"{heads_name}_{suffix}_{layout.lower()}",
                    batch_seq_lens=batch_seq_lens,
                    seq_lens=seq_lens,
                    Nq=heads[0], Nk=heads[1], Nv=heads[2],
                    layout_qkv=layout,
                    layout_q_out=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                ))
                seed += 1
    for heads_name, specs, heads in (
            ("n16_kv2", n16_large_strided_specs, (16, 2, 2)),
            ("n8_kv1", n8_large_strided_specs, (8, 1, 1))):
        for suffix, batch_seq_lens, seq_lens, slot_pattern, block_num in specs:
            for layout in ("NTD", "TND"):
                cases.append(InputCase(
                    case_name=f"{heads_name}_{suffix}_{layout.lower()}",
                    batch_seq_lens=batch_seq_lens,
                    seq_lens=seq_lens,
                    Nq=heads[0], Nk=heads[1], Nv=heads[2],
                    layout_qkv=layout,
                    layout_q_out=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride="d_pad",
                    cache_pad=8,
                ))
                seed += 1
    for suffix, batch_seq_lens, seq_lens, slot_pattern, block_num, cache_stride, cache_pad in n64_specs:
        for layout in ("NTD", "TND"):
            cases.append(InputCase(
                case_name=f"n64_kv8_{suffix}_{layout.lower()}",
                batch_seq_lens=batch_seq_lens,
                seq_lens=seq_lens,
                Nq=64, Nk=8, Nv=8,
                layout_qkv=layout,
                layout_q_out=layout,
                slot_pattern=slot_pattern,
                seed=seed,
                block_num=block_num,
                cache_stride=cache_stride,
                cache_pad=cache_pad,
            ))
            seed += 1
    for heads_name, heads in (("n64_kv8", (64, 8, 8)), ("n16_kv2", (16, 2, 2)), ("n8_kv1", (8, 1, 1))):
        for suffix, batch_seq_lens, seq_lens, slot_pattern, block_num, cache_stride, cache_pad in long_16k_specs:
            for layout in ("NTD", "TND"):
                cases.append(InputCase(
                    case_name=f"{heads_name}_{suffix}_{layout.lower()}",
                    batch_seq_lens=batch_seq_lens,
                    seq_lens=seq_lens,
                    Nq=heads[0], Nk=heads[1], Nv=heads[2],
                    layout_qkv=layout,
                    layout_q_out=layout,
                    slot_pattern=slot_pattern,
                    seed=seed,
                    block_num=block_num,
                    cache_stride=cache_stride,
                    cache_pad=cache_pad,
                ))
                seed += 1
    cases = list(with_tnd_to_ntd_cases(cases))
    cases.extend(precision_mb20_t16384_cases(seed))
    seed += 3
    cases.extend(performance_t16384_tnd_to_ntd_cases(seed))
    seed += 3
    cases.append(default_layout_case(seed))
    return tuple(cases)


def expected_outputs(inp, golden_mod):
    golden_kwargs = {k: v for k, v in inp.items()
                     if k not in ("cache_stride", "cache_pad", "omit_layout_args")}
    q_out, q_scale, k_cache, v_cache, k_scale_cache = (
        golden_mod.qkv_rms_norm_rope_cache_with_k_scale_golden(**golden_kwargs)
    )
    return {
        "q_out": to_compare_numpy(q_out),
        "q_scale": to_compare_numpy(q_scale),
        "k_cache": to_compare_numpy(k_cache),
        "v_cache": to_compare_numpy(v_cache),
        "k_scale_cache": to_compare_numpy(k_scale_cache),
    }


def numpy_to_torch(array, device, dtype=None):
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor.to(device)


def cache_torch_dtype(array):
    return torch.int8 if np.asarray(array).dtype == np.int8 else torch.float8_e4m3fn


def storage_numel(shape, stride):
    return int(sum((dim - 1) * step for dim, step in zip(shape, stride)) + 1)


def cache_view_stride(shape, cache_stride, cache_pad):
    _, heads, block_size, dim = shape
    token_stride = dim if cache_stride == "contiguous" else dim + cache_pad
    return (heads * block_size * token_stride, block_size * token_stride, token_stride, 1)


def make_zero_cache_view(shape, dtype, device, cache_stride, cache_pad):
    if cache_stride == "contiguous":
        if dtype == torch.float8_e4m3fn:
            return torch.zeros(shape, dtype=torch.float32, device=device).to(dtype)
        return torch.zeros(shape, dtype=dtype, device=device)

    stride = cache_view_stride(shape, cache_stride, cache_pad)
    numel = storage_numel(shape, stride)
    if dtype == torch.float8_e4m3fn:
        storage = torch.zeros((numel,), dtype=torch.float32, device=device).to(dtype)
    else:
        storage = torch.zeros((numel,), dtype=dtype, device=device)
    return torch.as_strided(storage, size=shape, stride=stride)


def make_torch_inputs(inp, device):
    cache_stride = inp.get("cache_stride", "contiguous")
    cache_pad = int(inp.get("cache_pad", 0))
    k_cache_dtype = cache_torch_dtype(inp["k_cache"])
    v_cache_dtype = cache_torch_dtype(inp["v_cache"])
    return {
        "qkv": numpy_to_torch(inp["qkv"], device, torch.bfloat16),
        "q_gamma": numpy_to_torch(inp["q_gamma"], device, torch.float32),
        "k_gamma": numpy_to_torch(inp["k_gamma"], device, torch.float32),
        "cos_sin": numpy_to_torch(inp["cos_sin"], device, torch.float32),
        "slot_mapping": numpy_to_torch(
            inp["slot_mapping"], device, torch.int32),
        "k_cache": make_zero_cache_view(
            tuple(inp["k_cache"].shape), k_cache_dtype, device, cache_stride, cache_pad),
        "v_cache": make_zero_cache_view(
            tuple(inp["v_cache"].shape), v_cache_dtype, device, cache_stride, cache_pad),
        "k_scale_cache": make_zero_cache_view(
            tuple(inp["k_scale_cache"].shape), torch.float32, device, cache_stride, cache_pad),
        "query_start_loc": numpy_to_torch(inp["query_start_loc"], device, torch.int32),
        "seq_lens": numpy_to_torch(
            inp["seq_lens"], device, torch.int32),
        "head_nums": list(inp["head_nums"]),
        "rotation": numpy_to_torch(
            inp["rotation"], device, torch.bfloat16),
        "v_scale": numpy_to_torch(inp["v_scale"], device, torch.float32),
        "epsilon": float(inp.get("epsilon", 1e-6)),
        "layout_qkv": inp["layout_qkv"],
        "layout_q_out": inp["layout_q_out"],
    }


def tensor_to_numpy(tensor):
    tensor_cpu = tensor.detach().cpu()
    if tensor_cpu.dtype == torch.int8:
        return tensor_cpu.numpy()
    return tensor_cpu.to(torch.float32).numpy()


def to_compare_numpy(array):
    array = np.asarray(array)
    if array.dtype == np.int8:
        return array.copy()
    return array.astype(np.float32)


def run_npu_op(inp, mode, device):
    if torch is None or torch_npu is None:
        raise RuntimeError(
            "torch and torch_npu are required to run this precision checker")

    from cann_ops_transformer.ops import (
        qkv_rms_norm_rope_cache_with_k_scale,
        qkv_rms_norm_rope_cache_with_k_scale_functional,
    )

    t = make_torch_inputs(inp, device)
    layout_kwargs = (
        {}
        if inp.get("omit_layout_args", False)
        else {"layout_qkv": t["layout_qkv"], "layout_q_out": t["layout_q_out"]}
    )
    with torch.no_grad():
        if mode == "inplace":
            q_out, q_scale = qkv_rms_norm_rope_cache_with_k_scale(
                t["qkv"], t["q_gamma"], t["k_gamma"], t["cos_sin"],
                t["slot_mapping"],
                t["k_cache"], t["v_cache"], t["k_scale_cache"],
                t["query_start_loc"], t["seq_lens"], t["head_nums"],
                rotation=t["rotation"],
                v_scale=t["v_scale"],
                epsilon=t["epsilon"],
                **layout_kwargs,
            )
            outputs = (q_out, q_scale,
                       t["k_cache"], t["v_cache"], t["k_scale_cache"])
        elif mode == "functional":
            outputs = qkv_rms_norm_rope_cache_with_k_scale_functional(
                t["qkv"], t["q_gamma"], t["k_gamma"], t["cos_sin"],
                t["slot_mapping"],
                t["k_cache"], t["v_cache"], t["k_scale_cache"],
                t["query_start_loc"], t["seq_lens"], t["head_nums"],
                rotation=t["rotation"],
                v_scale=t["v_scale"],
                epsilon=t["epsilon"],
                **layout_kwargs,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

    if hasattr(torch, "npu"):
        torch.npu.synchronize()

    q_out, q_scale, k_cache, v_cache, k_scale_cache = outputs
    return {
        "q_out": tensor_to_numpy(q_out),
        "q_scale": tensor_to_numpy(q_scale),
        "k_cache": tensor_to_numpy(k_cache),
        "v_cache": tensor_to_numpy(v_cache),
        "k_scale_cache": tensor_to_numpy(k_scale_cache),
    }


def compare_array(name, actual, expected, atol, max_bad_ratio):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return CompareResult(
            name, False, 1, 1, 1.0, float("inf"), None, None, None)

    total = int(actual.size)
    if total == 0:
        return CompareResult(
            name, True, 0, 0, 0.0, 0.0, None, None, None)

    actual_is_int8 = actual.dtype == np.int8
    expected_is_int8 = expected.dtype == np.int8
    if actual_is_int8 != expected_is_int8:
        return CompareResult(
            name, False, total, total, 1.0, float("inf"), None, None, None)

    is_int8 = actual_is_int8 and expected_is_int8
    if is_int8:
        actual = actual.astype(np.int16, copy=False)
        expected = expected.astype(np.int16, copy=False)
    else:
        actual = actual.astype(np.float32)
        expected = expected.astype(np.float32)

    actual_nan = np.isnan(actual)
    expected_nan = np.isnan(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)

    diff = np.zeros(actual.shape, dtype=np.float32)
    diff[finite] = np.abs(actual[finite] - expected[finite])

    inf_mismatch = (
        (np.isinf(actual) | np.isinf(expected)) & (actual != expected))
    threshold = INT8_MAX_ABS_DIFF if is_int8 else atol
    bad = (actual_nan != expected_nan) | inf_mismatch | (finite & (diff > threshold))
    bad_count = int(np.count_nonzero(bad))
    bad_ratio = bad_count / total
    max_abs_diff = float(np.max(diff[finite])) if np.any(finite) else 0.0

    first_bad = None
    first_actual = None
    first_expected = None
    if bad_count:
        first_bad = tuple(int(i) for i in np.argwhere(bad)[0])
        first_actual = float(actual[first_bad])
        first_expected = float(expected[first_bad])

    return CompareResult(
        name=name,
        passed=bad_count == 0 if is_int8 else bad_ratio <= max_bad_ratio,
        total=total,
        bad_count=bad_count,
        bad_ratio=bad_ratio,
        max_abs_diff=max_abs_diff,
        first_bad=first_bad,
        first_actual=first_actual,
        first_expected=first_expected,
    )


def select_touched_cache_rows(outputs, slot_mapping):
    slots = np.asarray(slot_mapping, dtype=np.int64)
    block_size = outputs["k_cache"].shape[2]
    block_ids = slots // block_size
    block_offsets = slots % block_size
    return {
        "k_cache_touched": outputs["k_cache"][block_ids, :, block_offsets, :],
        "v_cache_touched": outputs["v_cache"][block_ids, :, block_offsets, :],
        "k_scale_cache_touched": outputs["k_scale_cache"][block_ids, :, block_offsets, 0],
    }


def compare_outputs(actual, expected, slot_mapping, fp8_atol, float_atol, max_bad_ratio):
    actual_touched = select_touched_cache_rows(actual, slot_mapping)
    expected_touched = select_touched_cache_rows(expected, slot_mapping)
    pairs = [
        ("q_out", actual["q_out"], expected["q_out"], fp8_atol),
        ("k_cache_touched", actual_touched["k_cache_touched"],
         expected_touched["k_cache_touched"], fp8_atol),
        ("v_cache_touched", actual_touched["v_cache_touched"],
         expected_touched["v_cache_touched"], fp8_atol),
        ("q_scale", actual["q_scale"], expected["q_scale"], float_atol),
        ("k_scale_cache_touched", actual_touched["k_scale_cache_touched"],
         expected_touched["k_scale_cache_touched"], float_atol),
    ]
    return [
        compare_array(name, actual_array, expected_array, atol, max_bad_ratio)
        for name, actual_array, expected_array, atol in pairs
    ]


def print_results(case_name, mode, results):
    case_passed = all(result.passed for result in results)
    status = "PASS" if case_passed else "FAIL"
    total = sum(result.total for result in results)
    correct = sum(result.total - result.bad_count for result in results)
    precision_percent = 100.0 if total == 0 else correct / total * 100.0
    print(
        f"[{status}] case={case_name} mode={mode} "
        f"precision={precision_percent:.4f}% correct={correct}/{total}"
    )
    for result in results:
        correct_count = result.total - result.bad_count
        tensor_precision = (
            100.0 if result.total == 0
            else correct_count / result.total * 100.0)
        line = (
            f"    {result.name}: precision={tensor_precision:.4f}% "
            f"correct={correct_count}/{result.total} "
            f"bad={result.bad_count}/{result.total} "
            f"ratio={result.bad_ratio:.6g} "
            f"max_abs_diff={result.max_abs_diff:.6g}"
        )
        if result.first_bad is not None:
            line += (
                f" first_bad={result.first_bad} "
                f"actual={result.first_actual} "
                f"expected={result.first_expected}"
            )
        print(line)
    return case_passed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument(
        "--case", action="append",
        help="case name to run; may be repeated; default: all")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("inplace", "functional", "both"), default="both")
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--torch-extension-dir", type=Path, default=None,
        help="optional path containing the cann_ops_transformer Python package")
    parser.add_argument("--fp8-atol", type=float, default=FP8_ATOL)
    parser.add_argument("--float-atol", type=float, default=FLOAT_ATOL)
    parser.add_argument(
        "--max-bad-ratio", type=float, default=MAX_BAD_RATIO)
    return parser.parse_args()


def main():
    args = parse_args()
    golden_mod = load_golden(args.assets_dir)
    input_cases = get_input_cases()
    case_names = [ic.case_name for ic in input_cases]

    if args.list_cases:
        print("\n".join(case_names))
        return 0

    prepare_torch_extension_import(args.torch_extension_dir)

    if (torch is None
            or not hasattr(torch, "npu")
            or not torch.npu.is_available()):
        raise RuntimeError(
            "NPU device is required to run this precision checker")

    selected = set(args.case or case_names)
    unknown = selected - set(case_names)
    if unknown:
        raise ValueError(
            f"unknown case(s): {sorted(unknown)}; available: {case_names}")

    modes = ["inplace", "functional"] if args.mode == "both" else [args.mode]
    all_passed = True
    for ic in input_cases:
        if ic.case_name not in selected:
            continue
        inp = build_case(golden_mod, ic)
        expected = expected_outputs(inp, golden_mod)
        for mode in modes:
            actual = run_npu_op(inp, mode, args.device)
            results = compare_outputs(
                actual, expected, inp["slot_mapping"],
                args.fp8_atol, args.float_atol, args.max_bad_ratio)
            all_passed = (
                print_results(ic.case_name, mode, results) and all_passed)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
