from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import torch

try:
    import torch_npu
    from npu_ops_transformer.ops.grouped_matmul_v5 import npu_grouped_matmul_v5
except ImportError:
    torch_npu = None

ALIGNMENT = 128
TensorPair = Tuple[torch.Tensor, torch.Tensor]

def _print_tensor_meta(stage: str, name: str, tensor: torch.Tensor) -> None:
    print(f"[{stage}] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

def _print_quant_tensors(stage: str, a: TensorPair, b: TensorPair) -> None:
    a_quant, a_scale = a
    b_quant, b_scale = b
    _print_tensor_meta(stage, "a_quant", a_quant)
    _print_tensor_meta(stage, "b_quant", b_quant)
    _print_tensor_meta(stage, "a_scale", a_scale)
    _print_tensor_meta(stage, "b_scale", b_scale)

def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align(x: int, alignment: int = ALIGNMENT) -> int:
    return _ceil_div(x, alignment) * alignment

def _default_granularity(recipe: Optional[Tuple[int, int, int]], recipe_x: Optional[Tuple[int, int]]) -> int:
    if recipe_x is not None:
        return int(recipe_x[1])
    if recipe is not None:
        return int(recipe[2])
    return ALIGNMENT

def _analyze_psum_layout(grouped_layout: torch.Tensor, alignment: int) -> Tuple[List[int], List[int], List[int], int]:
    layout = grouped_layout.to("cpu", torch.int64).tolist()
    actual_sizes: List[int] = []
    padded_starts: List[int] = []
    valid_starts: List[int] = []
    prev_actual_end = 0
    padded_cursor = 0
    for actual_end in layout:
        actual_end = int(actual_end)
        actual_size = actual_end - prev_actual_end
        if actual_size < 0:
            raise ValueError("grouped_layout must be non-decreasing in psum layout")
        actual_sizes.append(actual_size)
        valid_starts.append(prev_actual_end)
        padded_starts.append(padded_cursor)
        padded_cursor += _align(actual_size, alignment)
        prev_actual_end = actual_end
    return actual_sizes, valid_starts, padded_starts, padded_cursor

def _analyze_dense_layout(grouped_layout: torch.Tensor) -> Tuple[List[int], List[int], List[int], int]:
    layout = grouped_layout.to("cpu", torch.int64).tolist()
    actual_sizes: List[int] = []
    valid_starts: List[int] = []
    padded_starts: List[int] = []
    group_id = 0
    cursor = 0
    total = len(layout)
    while cursor < total:
        while cursor < total and layout[cursor] == -1:
            cursor += 1
        if cursor >= total:
            break
        start = cursor
        while cursor < total and layout[cursor] == group_id:
            cursor += 1
        actual_end = cursor
        actual_size = actual_end - start
        if actual_size <= 0:
            raise ValueError(f"invalid grouped_layout for group {group_id}")
        actual_sizes.append(actual_size)
        valid_starts.append(start)
        padded_starts.append(start)
        while cursor < total and layout[cursor] == -1:
            cursor += 1
        group_id += 1
    return actual_sizes, valid_starts, padded_starts, total

def _analyze_layout(grouped_layout: torch.Tensor, use_psum_layout: bool, alignment: int) -> Tuple[List[int], List[int], List[int], int]:
    if grouped_layout.dim() != 1:
        raise ValueError("grouped_layout must be 1D")
    if use_psum_layout:
        return _analyze_psum_layout(grouped_layout, alignment)
    return _analyze_dense_layout(grouped_layout)

def _to_group_list(actual_sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(actual_sizes), dtype=torch.int64, device=device)

def _remove_group_padding_rows(tensor: torch.Tensor, valid_starts: Sequence[int], actual_sizes: Sequence[int]) -> torch.Tensor:
    parts = [tensor[start:start + size] for start, size in zip(valid_starts, actual_sizes)]
    return torch.cat(parts, dim=0) if parts else tensor[:0]

def _restore_group_padding_rows(tensor: torch.Tensor, padded_starts: Sequence[int], actual_sizes: Sequence[int], padded_total_rows: int) -> torch.Tensor:
    result = torch.zeros((padded_total_rows, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device)
    src = 0
    for dst, size in zip(padded_starts, actual_sizes):
        result[dst:dst + size] = tensor[src:src + size]
        src += size
    return result

class DeepGEMMBuffer:
    def __init__(self):
        if torch_npu is None:
            raise RuntimeError("torch_npu is not available")

    def m_grouped_fp8_fp4_gemm_nt_contiguous(
        self,
        a: TensorPair,
        b: TensorPair,
        d: torch.Tensor,
        grouped_layout: torch.Tensor,
        recipe: Optional[Tuple[int, int, int]] = None,
        recipe_a: Optional[Tuple[int, int]] = None,
        recipe_b: Optional[Tuple[int, int]] = None,
        compiled_dims: str = "",
        disable_ue8m0_cast: bool = False,
        use_psum_layout: bool = False,
        expected_m_for_psum_layout: Optional[int] = None,
        c: Optional[torch.Tensor] = None,
    ) -> None:
        del compiled_dims, disable_ue8m0_cast, expected_m_for_psum_layout

        a_quant, a_scale = a
        b_quant, b_scale = b
        _print_quant_tensors("m_grouped_fp8_fp4_gemm_nt_contiguous/input", a, b)
        if a_quant.dim() != 2:
            raise ValueError("a[0] must be 2D")
        if b_quant.dim() != 3:
            raise ValueError("b[0] must be 3D with shape [group, n, k]")

        a_quant = a_quant.npu()
        b_quant = b_quant.npu()
        a_scale = a_scale.npu()
        b_scale = b_scale.npu()
        b_quant = b_quant.transpose(1, 2)
        b_scale = b_scale.transpose(1, 2)

        a_alignment = _default_granularity(recipe, recipe_a)
        actual_sizes, valid_starts, padded_starts, padded_total_rows = _analyze_layout(
            grouped_layout, use_psum_layout, a_alignment
        )
        if len(actual_sizes) != int(b_quant.shape[0]):
            raise ValueError("group count from grouped_layout does not match b groups")

        x = _remove_group_padding_rows(a_quant, valid_starts, actual_sizes)
        x_scale = _remove_group_padding_rows(a_scale, valid_starts, actual_sizes)
        weight = b_quant
        weight_scale = b_scale
        _print_tensor_meta("m_grouped_fp8_fp4_gemm_nt_contiguous/compacted", "x", x)
        _print_tensor_meta("m_grouped_fp8_fp4_gemm_nt_contiguous/compacted", "x_scale", x_scale)
        _print_tensor_meta("m_grouped_fp8_fp4_gemm_nt_contiguous/compacted", "weight", weight)
        _print_tensor_meta("m_grouped_fp8_fp4_gemm_nt_contiguous/compacted", "weight_scale", weight_scale)
        group_list = _to_group_list(actual_sizes, a_quant.device)
        c_compact = _remove_group_padding_rows(c.npu(), valid_starts, actual_sizes) if c is not None else None

        out = npu_grouped_matmul_v5(
            [x],
            [weight],
            bias=[c_compact] if c_compact is not None else None,
            scale=[weight_scale],
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[x_scale],
            group_list=group_list,
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=2,
            group_type=0,
            group_list_type=1,
            act_type=0,
            tuning_config=None,
            output_dtype=d.dtype,
        )
        out_compact = out[0] if isinstance(out, list) else out
        out_padded = _restore_group_padding_rows(out_compact.to(d.dtype), padded_starts, actual_sizes, padded_total_rows)
        d.copy_(out_padded)
