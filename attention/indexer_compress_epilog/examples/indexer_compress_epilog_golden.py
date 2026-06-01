import numpy as np
import torch
from en_dtypes import hifloat8
from ml_dtypes import float8_e4m3fn, float8_e5m2


DATA_TYPE_INT_TO_STR = {
    1: "float16",
    27: "bfloat16",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
}


def get_dtype_range(dt):
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    numpy_dtype = np.dtype(dt)
    if numpy_dtype.kind in "iu":
        numpy_info = np.iinfo(numpy_dtype)
    else:
        numpy_info = np.finfo(numpy_dtype)
    return numpy_info.min, numpy_info.max


def replace_inf_nan_with_zero(x):
    x_cleaned = x.copy()
    mask = np.isinf(x) | np.isnan(x)
    x_cleaned[mask] = 0
    mask_tensor = np.zeros_like(x, dtype=int)
    mask_tensor[mask] = 1
    return x_cleaned, mask_tensor


def block_max_with_padding(x, row_block_size, col_block_size):
    rows, cols = x.shape
    pad_rows = (row_block_size - rows % row_block_size) % row_block_size
    pad_cols = (col_block_size - cols % col_block_size) % col_block_size
    x_padded = np.pad(x, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)
    row_blocks = x_padded.shape[0] // row_block_size
    col_blocks = x_padded.shape[1] // col_block_size
    result = np.zeros((row_blocks, col_blocks))
    for i in range(row_blocks):
        for j in range(col_blocks):
            block = x_padded[i * row_block_size:(i + 1) * row_block_size,
                             j * col_block_size:(j + 1) * col_block_size]
            result[i, j] = np.max(block)
    return result


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


def fast_log2_ceil(x):
    bits_x = x.view(np.uint32)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((np.uint32(1) << np.uint32(23)) - np.uint32(1))
    base = 1 if man_bits != 0 else 0
    return np.array([exp_x - 127 + base], dtype=np.uint32).view(np.int32)[0]


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return np.array([bits_x], dtype=np.int32).view(np.float32)[0]


def fast_round_scale(amax):
    return fast_pow2(fast_log2_ceil(amax))


def act_quant(x, dst_type, round_scale):
    row_block_size = 1
    col_block_size = 128
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    max_value = 0
    if dst_type_str == "float8_e5m2":
        max_value = (2 - pow(2, -2)) * pow(2, 15)
    elif dst_type_str == "float8_e4m3fn":
        max_value = (2 - pow(2, -2)) * pow(2, 8)

    x_cleaned, mask_tensor = replace_inf_nan_with_zero(x)
    block_max = block_max_with_padding(np.abs(x_cleaned), 1, 128)
    scale = block_max.astype(np.float32) / max_value
    scale_shape = scale.shape
    if round_scale:
        scale = scale.reshape(-1)
        for i in range(len(scale)):
            scale[i] = fast_round_scale(scale[i])
    scale = scale.reshape(scale_shape)
    scale_expanded = np.zeros_like(x_cleaned).astype(np.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            scale_expanded[i * row_block_size:(i + 1) * row_block_size,
                           j * col_block_size:(j + 1) * col_block_size] = scale[i, j]
    out_f32 = x_cleaned.astype(np.float32) / scale_expanded
    out_f32[mask_tensor == 1] = x[mask_tensor == 1]
    max_norm = get_dtype_range(dst_type_str)[1]
    np.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)
    output_scale = (scale.view(np.uint32) >> 23).astype(np.uint8)
    round_data = np.nan_to_num(np.round(out_f32, 8), nan=0.0, copy=False)
    if dst_type == 35:
        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        round_data = round_data.astype(float8_e4m3fn, copy=False)
    return round_data, output_scale


def dynamic_block_quant(x, dst_type):
    row_block_size = 1
    col_block_size = 128
    dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
    max_value = 0
    if dst_type_str == "hifloat8":
        max_value = pow(2, 15)
    elif dst_type_str == "float8_e5m2":
        max_value = (2 - pow(2, -2)) * pow(2, 15)
    elif dst_type_str == "float8_e4m3fn":
        max_value = (2 - pow(2, -2)) * pow(2, 8)
    x_cleaned, mask_tensor = replace_inf_nan_with_zero(x)
    scale = block_max_with_padding(np.abs(x_cleaned), 1, 128).astype(np.float32) / max_value
    scale_expanded = np.zeros_like(x_cleaned).astype(np.float32)
    for i in range(scale.shape[0]):
        for j in range(scale.shape[1]):
            scale_expanded[i * row_block_size:(i + 1) * row_block_size,
                           j * col_block_size:(j + 1) * col_block_size] = scale[i, j]
    out_f32 = x_cleaned.astype(np.float32) / scale_expanded
    out_f32[mask_tensor == 1] = x[mask_tensor == 1]
    max_norm = get_dtype_range(dst_type_str)[1]
    np.clip(out_f32, a_min=-max_norm, a_max=max_norm, out=out_f32)
    output_scale = scale.astype("float32")
    round_data = np.nan_to_num(np.round(out_f32, 8), nan=0.0, copy=False)
    if dst_type == 35:
        round_data = round_data.astype(float8_e5m2, copy=False)
    elif dst_type == 36:
        round_data = round_data.astype(float8_e4m3fn, copy=False)
    elif dst_type == 34:
        round_data = round_data.astype(hifloat8, copy=False)
    return round_data, output_scale


def indexer_compress_epilog_golden(indexer_compress_cache, indexer_compress_cache_scale, x,
                                   slot_mapping, dst_type, quant_mode, round_scale):
    index = np.where(slot_mapping != -1)
    x = x[index]
    slot_mapping = slot_mapping[index]
    if quant_mode == 0:
        y, scale = act_quant(x, dst_type, round_scale)
    elif quant_mode == 1:
        y, scale = dynamic_block_quant(x, dst_type)
    else:
        raise ValueError(f"unsupported quant_mode for this golden: {quant_mode}")
    indexer_compress_cache = indexer_compress_cache.astype(hifloat8 if dst_type == 34 else float8_e5m2)
    indexer_compress_cache[slot_mapping] = y
    indexer_compress_cache_scale[slot_mapping] = scale
    return indexer_compress_cache, indexer_compress_cache_scale


def hifp8_block_quant(indexer_compress_cache, x, slot_mapping, scale):
    valid_mask = slot_mapping != -1
    if not np.any(valid_mask):
        return indexer_compress_cache.astype(hifloat8)
    x_valid = x[valid_mask].astype(np.float32)
    slots_valid = slot_mapping[valid_mask]
    y_hifp8 = (x_valid * np.float32(scale)).astype(hifloat8)
    new_cache = indexer_compress_cache.astype(hifloat8)
    new_cache[slots_valid] = y_hifp8
    return new_cache
