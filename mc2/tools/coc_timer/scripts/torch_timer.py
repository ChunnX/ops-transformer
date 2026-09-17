"""
CocTimer Python/Torch 端解析工具。

提供核心函数：
  - get_timer_buffer_size(core_count)          → 计算 timer buffer tensor 所需元素数
  - parse_timer_buffer(tensor, core_count)      → 解析 raw tensor 为结构化 TimerResult
  - write_time_results_raw(filename, result)    → 写入原始时间戳 CSV（start/end_cycle + duration_us）
  - write_time_results_duration(filename, result) → 写入 Duration CSV（仅 duration_us）

用法示例：

    import torch
    from mc2.tools.coc_timer.torch_timer import (
        get_timer_buffer_size, parse_timer_buffer, core_count_for,
        write_time_results_raw, write_time_results_duration,
    )

    # 1. 已知 block 数，计算 core_count
    block_num = 24
    core_count = core_count_for(block_num)

    # 2. 创建 timer buffer tensor
    buf_size = get_timer_buffer_size(core_count)
    timer_buffer = torch.zeros(buf_size, dtype=torch.int64, device="npu")

    # 3. 执行算子，传入 timer_buffer ...

    # 4. 解析结果
    timer_cpu = timer_buffer.cpu()
    timer_result = parse_timer_buffer(timer_cpu, core_count)

    # 5. 写入 CSV 文件
    write_time_results_raw("./trace/raw_timing.csv", timer_result)
    write_time_results_duration("./trace/duration_timing.csv", timer_result)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
import torch


# ============================================================================
# 常量 (与 coc_timer.hpp / coc_timer_host.hpp 保持同步)
# ============================================================================

_K_CORES_PER_GROUP = 3
_K_SUB_GROUP_ID_AIC = 0
_K_CYCLE_TO_US = 50.0
_K_FALLBACK_BLOCK_NUM = 24

_K_FIXED_NAMES = {
    0: "KERNEL_TIMING_IDX",
}
_K_FIXED_TIMING_COUNT = len(_K_FIXED_NAMES)

_K_DYNAMIC_NAMES = {
    0: "AIC",
    1: "AIV_AG",
    2: "AIV_GO",
    3: "AIV_DQ",
}
_K_DYNAMIC_TYPE_COUNT = len(_K_DYNAMIC_NAMES)
_K_MAX_DYNAMIC_ITER = 150

_K_TIMING_PER_CORE = _K_FIXED_TIMING_COUNT * 2 + _K_DYNAMIC_TYPE_COUNT * _K_MAX_DYNAMIC_ITER * 2
_K_TIMING_PER_CORE_ALIGN = math.ceil(_K_TIMING_PER_CORE / 16) * 16
_K_ITER_PER_CORE_ALIGN = math.ceil(_K_DYNAMIC_TYPE_COUNT / 16) * 16

_K_DYNAMIC_TIMING_START = _K_FIXED_TIMING_COUNT


# ============================================================================
# Dataclass
# ============================================================================


@dataclass
class TimerResult:
    core_count: int = 0
    timing_counter_size: int = 0
    dynamic_iter_total_size: int = 0
    base_cycle_data: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    dynamic_actual_iter_per_core: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    dynamic_actual_iter_max: torch.Tensor = field(default_factory=lambda: torch.empty(_K_DYNAMIC_TYPE_COUNT, dtype=torch.int64))
    is_accumulate: list[bool] = field(default_factory=lambda: [False] * _K_DYNAMIC_TYPE_COUNT)


# ============================================================================
# Public API
# ============================================================================


def core_count_for(block_num: int) -> int:
    """根据 block 数计算总核数 (AIC + AIV)。"""
    return block_num * _K_CORES_PER_GROUP


def get_timer_buffer_size(core_count: int) -> int:
    """
    对应 CocTimerHost::GetTotalTimerBufferSize()。

    返回 timer buffer tensor 所需的 int64 元素个数。
    tensor 应通过 `torch.zeros(buf_size, dtype=torch.int64, device="npu")` 创建。

    Args:
        core_count: 总核数 AIC + AIV (= block_num × 3)

    Returns:
        int: int64 元素个数
    """
    return _timing_counter_size(core_count) + _dynamic_iter_total_size(core_count)


def get_timing_counter_size(core_count: int) -> int:
    """对应 CocTimerHost::GetTimingCounterSize()。"""
    return _timing_counter_size(core_count)


def get_dynamic_iter_total_size(core_count: int) -> int:
    """对应 CocTimerHost::GetDynamicIterTotalSize()。"""
    return _dynamic_iter_total_size(core_count)


def parse_timer_buffer(tensor: torch.Tensor, core_count: int) -> TimerResult:
    """
    对应 CocTimerHost::GetTimeResult(void *deviceAddr)。

    解析 raw timer buffer tensor（已在 CPU 侧），返回结构化 TimerResult。

    Args:
        tensor: int64 类型的 timer buffer 数据 (已通过 .cpu() 取出)
        core_count: 总核数 AIC + AIV

    Returns:
        TimerResult: 包含 base_cycle_data, dynamic_actual_iter_per_core 等
    """
    if tensor.dtype != torch.int64:
        tensor = tensor.to(torch.int64)
    data = tensor.contiguous().view(-1)

    timing_size = _timing_counter_size(core_count)
    iter_size = _dynamic_iter_total_size(core_count)
    expected_total = timing_size + iter_size

    if data.numel() < expected_total:
        raise ValueError(
            f"Tensor 元素数 {data.numel()} 小于期望的 {expected_total} "
            f"(timing={timing_size}, iter={iter_size})"
        )

    base_cycle = data[:timing_size].clone()
    dynamic_iter = data[timing_size : timing_size + iter_size].clone()

    iter_max = torch.zeros(_K_DYNAMIC_TYPE_COUNT, dtype=torch.int64)
    is_accumulate = [False] * _K_DYNAMIC_TYPE_COUNT

    for core_id in range(core_count):
        core_offset = core_id * _K_ITER_PER_CORE_ALIGN
        for dtype_idx in range(_K_DYNAMIC_TYPE_COUNT):
            idx = core_offset + dtype_idx
            if idx >= dynamic_iter.numel():
                continue
            val = dynamic_iter[idx].item()
            if val < 0:
                is_accumulate[dtype_idx] = True
                val = -val
                dynamic_iter[idx] = val
            if val > iter_max[dtype_idx]:
                iter_max[dtype_idx] = val

    return TimerResult(
        core_count=core_count,
        timing_counter_size=timing_size,
        dynamic_iter_total_size=iter_size,
        base_cycle_data=base_cycle,
        dynamic_actual_iter_per_core=dynamic_iter,
        dynamic_actual_iter_max=iter_max,
        is_accumulate=is_accumulate,
    )


def to_duration_dataframe(timer_result: TimerResult) -> pd.DataFrame:
    """
    将 TimerResult 转换为 Duration DataFrame（每核一行，列为各阶段的耗时 us）。

    格式与 C++ 端 WriteTimeResultsDuration() 生成的 CSV 一致：
      group_id, core_type, sub_group_id, KERNEL_TIMING_IDX_duration(us),
      AIC_0_duration(us), ..., AIC_<n>_duration(us), ...
    """
    core_count = timer_result.core_count
    timing_names = _build_timing_names(timer_result)
    total_items = len(timing_names)
    cycle_data = timer_result.base_cycle_data
    iter_max = timer_result.dynamic_actual_iter_max

    records = []
    for core_id in range(core_count):
        group_id = core_id // _K_CORES_PER_GROUP
        sub_group_id = core_id % _K_CORES_PER_GROUP
        core_type = "AIC" if sub_group_id == _K_SUB_GROUP_ID_AIC else "AIV"
        sub_id = 0 if core_type == "AIC" else (sub_group_id - 1)

        base_index = _base_index_for_core(group_id, sub_group_id)
        record = {
            "group_id": group_id,
            "core_type": core_type,
            "sub_group_id": sub_id,
        }

        for name_idx, name in enumerate(timing_names):
            element_idx = _element_idx_for_name(name, name_idx, base_index)
            if element_idx < 0 or element_idx + 1 >= timer_result.timing_counter_size:
                record[f"{name}_duration(us)"] = 0.0
            else:
                start = cycle_data[element_idx].item()
                end = cycle_data[element_idx + 1].item()
                duration = _compute_duration_us(start, end)
                record[f"{name}_duration(us)"] = duration

        for dtype_idx in range(_K_DYNAMIC_TYPE_COUNT):
            if dtype_idx < iter_max.numel() and iter_max[dtype_idx].item() <= 0:
                continue
            if dtype_idx not in _K_DYNAMIC_NAMES:
                continue
            idx_iter = core_id * _K_ITER_PER_CORE_ALIGN + dtype_idx
            val = timer_result.dynamic_actual_iter_per_core[idx_iter].item() if idx_iter < timer_result.dynamic_actual_iter_per_core.numel() else 0
            record[f"{_K_DYNAMIC_NAMES[dtype_idx]}_IterCount"] = val

        records.append(record)

    return pd.DataFrame(records)


def write_time_results_raw(filename: str, timer_result: TimerResult) -> bool:
    """
    对应 CocTimerHost::WriteTimeResultsRaw()。

    写入原始时间戳 CSV：每行一个核，包含 start_cycle / end_cycle / duration_us。

    Args:
        filename: 输出 CSV 文件路径
        timer_result: parse_timer_buffer() 返回的 TimerResult

    Returns:
        bool: 写入成功返回 True，失败返回 False
    """
    try:
        import os
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        core_count = timer_result.core_count
        timing_names = _build_timing_names(timer_result)
        total_items = len(timing_names)
        cycle_data = timer_result.base_cycle_data
        iter_max = timer_result.dynamic_actual_iter_max

        with open(filename, "w") as f:
            header_parts = ["group_id", "core_type", "sub_group_id"]
            for name in timing_names:
                header_parts.append(f"{name}_start_cycle")
                header_parts.append(f"{name}_end_cycle")
                header_parts.append(f"{name}_duration_us")
            for dtype_idx in range(_K_DYNAMIC_TYPE_COUNT):
                if dtype_idx < iter_max.numel() and iter_max[dtype_idx].item() <= 0:
                    continue
                if dtype_idx in _K_DYNAMIC_NAMES:
                    header_parts.append(f"{_K_DYNAMIC_NAMES[dtype_idx]}_IterCount")
            f.write(",".join(header_parts) + "\n")

            for core_id in range(core_count):
                group_id = core_id // _K_CORES_PER_GROUP
                sub_group_id = core_id % _K_CORES_PER_GROUP
                core_type = "AIC" if sub_group_id == _K_SUB_GROUP_ID_AIC else "AIV"
                sub_id = 0 if core_type == "AIC" else (sub_group_id - 1)
                base_index = _base_index_for_core(group_id, sub_group_id)

                row_parts = [str(group_id), core_type, str(sub_id)]

                for name_idx, name in enumerate(timing_names):
                    element_idx = _element_idx_for_name(name, name_idx, base_index)
                    if element_idx < 0 or element_idx + 1 >= timer_result.timing_counter_size:
                        row_parts.extend(["0", "0", "0"])
                    else:
                        start = cycle_data[element_idx].item()
                        end = cycle_data[element_idx + 1].item()
                        duration = _compute_duration_us(start, end)
                        row_parts.append(str(start))
                        row_parts.append(str(end))
                        row_parts.append(str(duration))

                for dtype_idx in range(_K_DYNAMIC_TYPE_COUNT):
                    if dtype_idx < iter_max.numel() and iter_max[dtype_idx].item() <= 0:
                        continue
                    idx_iter = core_id * _K_ITER_PER_CORE_ALIGN + dtype_idx
                    val = timer_result.dynamic_actual_iter_per_core[idx_iter].item() if idx_iter < timer_result.dynamic_actual_iter_per_core.numel() else 0
                    row_parts.append(str(val))

                f.write(",".join(row_parts) + "\n")

        return True
    except Exception:
        return False


def write_time_results_duration(filename: str, timer_result: TimerResult) -> bool:
    """
    对应 CocTimerHost::WriteTimeResultsDuration()。

    写入 Duration CSV：每行一个核，仅包含 duration_us（无原始 cycle 值）。

    Args:
        filename: 输出 CSV 文件路径
        timer_result: parse_timer_buffer() 返回的 TimerResult

    Returns:
        bool: 写入成功返回 True，失败返回 False
    """
    try:
        df = to_duration_dataframe(timer_result)
        import os
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        df.to_csv(filename, index=False)
        return True
    except Exception:
        return False


# ============================================================================
# Internal helpers
# ============================================================================


def _timing_counter_size(core_count: int) -> int:
    return _K_TIMING_PER_CORE_ALIGN * core_count


def _dynamic_iter_total_size(core_count: int) -> int:
    return _K_ITER_PER_CORE_ALIGN * core_count


def _base_index_for_core(group_id: int, sub_group_id: int) -> int:
    base = group_id * _K_TIMING_PER_CORE_ALIGN * _K_CORES_PER_GROUP
    if sub_group_id != _K_SUB_GROUP_ID_AIC:
        base += _K_TIMING_PER_CORE_ALIGN * sub_group_id
    return base


def _cycle_to_us(cycles: int) -> float:
    return cycles / _K_CYCLE_TO_US


def _compute_duration_us(start_cycles: int, end_cycles: int) -> float:
    if start_cycles == 0:
        return _cycle_to_us(end_cycles)
    if end_cycles > start_cycles:
        return _cycle_to_us(end_cycles - start_cycles)
    return 0.0


def _get_dynamic_name(type_idx: int) -> str:
    return _K_DYNAMIC_NAMES.get(type_idx, f"UNKNOWN_DYNAMIC_{type_idx}")


def _parse_dynamic_name(name: str) -> tuple[int, int] | None:
    for type_idx, prefix in _K_DYNAMIC_NAMES.items():
        if name == prefix:
            return (type_idx, 0)
    last_underscore = name.rfind("_")
    if last_underscore == -1:
        return None
    prefix = name[:last_underscore]
    iter_str = name[last_underscore + 1:]
    try:
        it = int(iter_str)
    except ValueError:
        return None
    for type_idx, known_prefix in _K_DYNAMIC_NAMES.items():
        if known_prefix == prefix:
            return (type_idx, it)
    return None


def _element_idx_for_name(name: str, name_idx: int, base_index: int) -> int:
    if name_idx < _K_FIXED_TIMING_COUNT:
        return base_index + name_idx * 2

    parsed = _parse_dynamic_name(name)
    if parsed is None:
        return -1
    dtype_idx, it = parsed
    timing_idx = _K_DYNAMIC_TIMING_START + dtype_idx * _K_MAX_DYNAMIC_ITER + it
    return base_index + timing_idx * 2


def _build_timing_names(timer_result: TimerResult) -> list[str]:
    names = []
    for i in range(_K_FIXED_TIMING_COUNT):
        names.append(_K_FIXED_NAMES.get(i, f"UNKNOWN_FIXED_{i}"))

    for dtype_idx in range(_K_DYNAMIC_TYPE_COUNT):
        if dtype_idx not in _K_DYNAMIC_NAMES:
            continue
        actual_max = timer_result.dynamic_actual_iter_max[dtype_idx].item() if dtype_idx < timer_result.dynamic_actual_iter_max.numel() else 0
        if actual_max <= 0:
            continue
        prefix = _K_DYNAMIC_NAMES[dtype_idx]
        if timer_result.is_accumulate[dtype_idx]:
            names.append(prefix)
        else:
            for it in range(actual_max):
                names.append(f"{prefix}_{it}")
    return names