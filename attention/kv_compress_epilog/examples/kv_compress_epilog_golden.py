import numpy as np
import torch
from en_dtypes import hifloat8


DATA_TYPE_INT_TO_STR = {
    1: "float16",
    27: "bfloat16",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
}


def requantize_compare(golden, output, dtype_str):
    if dtype_str not in ("float8_e5m2", "float8_e4m3fn", "hifloat8"):
        raise ValueError(f"Unsupported dtype_str: {dtype_str}")
    output_for_compare = output.view(torch.int8)
    golden_for_compare = golden.view(torch.int8)
    diff_abs = torch.abs(output_for_compare.view(-1) - golden_for_compare.view(-1))
    diff_indices = torch.where(diff_abs > 1)[0]
    output_flat_float = output.view(-1).float()
    golden_flat_float = golden.view(-1).float()
    both_nan = torch.isnan(output_flat_float) & torch.isnan(golden_flat_float)
    diff_indices = diff_indices[~torch.isin(diff_indices, torch.where(both_nan)[0])]
    return (len(diff_indices) / golden.numel()) <= 0.001


def hifp8_block_quant(kv_compress_cache, x, slot_mapping, scale):
    valid_mask = slot_mapping != -1
    if not np.any(valid_mask):
        return kv_compress_cache.astype(hifloat8)
    x_valid = x[valid_mask].astype(np.float32)
    slots_valid = slot_mapping[valid_mask]
    y_hifp8 = (x_valid * np.float32(scale)).astype(hifloat8)
    new_cache = kv_compress_cache.astype(hifloat8)
    new_cache[slots_valid] = y_hifp8
    return new_cache
