import ctypes
import glob
import os
from typing import List, Optional, Sequence

import numpy
import torch
import torch.nn as nn
from torch import Tensor


def _setup_custom_opp_for_this_task():
    op_api_lib = None
    lib_path = None
    env_opp = (os.environ.get("ASCEND_CUSTOM_OPP_PATH") or "").strip().rstrip(os.sep)
    if env_opp:
        cand = os.path.join(env_opp, "op_api", "lib")
        if os.path.isfile(os.path.join(cand, "libcust_opapi.so")):
            lib_path = env_opp
            op_api_lib = cand
    if op_api_lib is None:
        here = os.path.dirname(os.path.abspath(__file__))
        ws = os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))
        pattern = os.path.join(
            ws,
            "runs",
            "lvl3",
            "Moe",
            "MoeInitRoutingV3",
            "**",
            "packages",
            "vendors",
            "*",
            "op_api",
            "lib",
            "libcust_opapi.so",
        )
        hits = glob.glob(pattern, recursive=True)
        if not hits:
            return
        so = max(hits, key=os.path.getmtime)
        lib_path = os.path.abspath(os.path.join(os.path.dirname(so), "..", ".."))
        op_api_lib = os.path.join(lib_path, "op_api", "lib")
    if lib_path and op_api_lib and os.path.isdir(op_api_lib):
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = lib_path + os.sep
        sep = os.path.pathsep
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = op_api_lib + (sep + prev if prev else "")
        cust_so = os.path.join(op_api_lib, "libcust_opapi.so")
        if os.path.isfile(cust_so):
            try:
                ctypes.CDLL(cust_so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_setup_custom_opp_for_this_task()


def _adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1


def _to_numpy_f32(x: Tensor) -> numpy.ndarray:
    if x.dtype in (torch.bfloat16, torch.float16):
        return x.detach().cpu().float().numpy()
    return x.detach().cpu().numpy()


def _to_numpy_int32(x: Tensor) -> numpy.ndarray:
    return x.detach().cpu().to(torch.int32).numpy().copy()


def _cpu_reference_forward(
    x: Tensor,
    expert_idx: Tensor,
    scale: Optional[Tensor],
    offset: Optional[Tensor],
    active_num: int,
    expert_capacity: int,
    expert_num: int,
    drop_pad_mode: int,
    expert_tokens_num_type: int,
    expert_tokens_num_flag: bool,
    quant_mode: int,
    active_expert_range: Sequence[int],
    row_idx_type: int,
) -> List[Tensor]:
    """CPU 金标准（与 ATK / 原 validation 算法一致）；scale/offset 可为 None。"""
    x_dtype = x.dtype
    expert_start = int(active_expert_range[0])
    expert_end = int(active_expert_range[1])

    scale_np = offset_np = None
    if scale is not None:
        scale_np = numpy.asarray(torch.as_tensor(scale).detach().cpu().float().numpy().copy())
    if offset is not None:
        offset_np = numpy.asarray(torch.as_tensor(offset).detach().cpu().float().numpy().copy())

    x = _to_numpy_f32(x)
    num_rows = x.shape[0]
    h = x.shape[1]
    expert_idx = _to_numpy_int32(expert_idx)
    k = expert_idx.shape[1]

    expert_idx_in = expert_idx.reshape(-1)
    actual_expert_total_num = int(numpy.sum((expert_idx >= expert_start) & (expert_idx < expert_end)))

    expert_idx_in[(expert_idx_in < expert_start)] = numpy.int32(numpy.iinfo(numpy.int32).max)
    sorted_expert_indices = numpy.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]

    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices
    else:
        expanded_row_idx = numpy.ones(num_rows * k).astype(numpy.int32) * -1
        tmp_indices = numpy.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    expert_tokens_count = None
    if not expert_tokens_num_flag:
        pass
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 1:
                expert_tokens_count = numpy.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = numpy.concatenate(
                    [
                        expert_tokens_count,
                        numpy.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(numpy.int64),
                    ]
                )
            elif expert_tokens_num_type == 0:
                expert_tokens_count = numpy.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = numpy.concatenate(
                    [
                        expert_tokens_count,
                        numpy.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(numpy.int64),
                    ]
                )
                expert_tokens_count = numpy.cumsum(expert_tokens_count)
            elif expert_tokens_num_type == 2:
                expert_id, counts = numpy.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                expert_tokens_count = numpy.column_stack((expert_id, counts))
                if expert_tokens_count.shape[0] < expert_num:
                    expert_tokens_count = numpy.concatenate((expert_tokens_count, [[0, 0]]), axis=0)
                expert_tokens_count = numpy.concatenate(
                    [
                        expert_tokens_count,
                        numpy.zeros((expert_num - expert_tokens_count.shape[0]) * 2).reshape(-1, 2).astype(numpy.int64),
                    ]
                )
        else:
            expert_tokens_count = numpy.bincount(sorted_expert_idx[:actual_expert_total_num] - expert_start)
            expert_tokens_count = numpy.concatenate(
                [
                    expert_tokens_count,
                    numpy.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(numpy.int64),
                ]
            )
        expert_tokens_count = expert_tokens_count.astype(numpy.int64)

    expanded_scale = None
    sort_row_tmp = None
    if drop_pad_mode == 0:
        if active_num == 0:
            active_num = actual_expert_total_num
        else:
            active_num = min(active_num, actual_expert_total_num)
        expanded_x = x[sorted_expert_indices[:active_num] // k, :]
        if scale_np is not None and quant_mode == -1:
            expanded_scale = scale_np[sorted_expert_indices[:active_num] // k]
    else:
        _adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

        sort_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=int)
        offset_tmp = 0
        lastExpertId = 0
        for i, val in enumerate(sorted_expert_indices):
            if val != -1:
                if lastExpertId != sorted_expert_idx[i]:
                    offset_tmp = 0
                    lastExpertId = sorted_expert_idx[i]

                sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                offset_tmp = offset_tmp + 1

        expanded_row_idx = numpy.full(sorted_expert_indices.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        expanded_x_mask = numpy.full((expert_num * expert_capacity, h), 1, dtype=int)
        expanded_x = numpy.full((expert_num * expert_capacity, h), 0, dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = numpy.full((h,), 0, dtype=int)

    if quant_mode == -1:
        if scale_np is None or drop_pad_mode == 1:
            expanded_scale = None

    if quant_mode == 0:
        expanded_scale = None
        expanded_x_fp32 = expanded_x.astype(numpy.float32)
        scale_val = scale_np.astype(numpy.float32)
        offset_val = offset_np.astype(numpy.float32)
        scale_rst = expanded_x_fp32 * scale_val[0]
        add_offset = scale_rst + offset_val[0]
        round_data = numpy.rint(add_offset)
        round_data = numpy.clip(round_data, -128, 127)
        expanded_x = round_data.astype(numpy.int8)

    if quant_mode == 1:
        x_final = expanded_x.astype(numpy.float32)
        if scale_np is None:
            x_abs = numpy.abs(x_final)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)
        else:
            scale_use = scale_np.astype(numpy.float32)
            if scale_use.shape[0] == 1:
                x_final = x_final * scale_use
            else:
                if drop_pad_mode == 0:
                    x_final = x_final * scale_use[sorted_expert_idx[:active_num] - expert_start]
                else:
                    for i, val in enumerate(sort_row_tmp):
                        if val != -1:
                            x_final[i] = x_final[i] * scale_use[i // expert_capacity]

            x_abs = numpy.abs(x_final)
            x_max = numpy.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = numpy.round(expanded_x).astype(numpy.int8)

    if drop_pad_mode == 1:
        expanded_x = numpy.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

    if expert_tokens_count is None:
        expert_tokens_count_t = torch.tensor([], dtype=torch.int64)
    else:
        expert_tokens_count_t = torch.from_numpy(expert_tokens_count.astype("int64"))
    # (1,) 占位避免 Bench precision 日志对 numel==0 调用 min 崩溃。
    if expanded_scale is None:
        expanded_scale_t = torch.zeros(1, dtype=torch.float32)
    else:
        expanded_scale_t = torch.from_numpy(expanded_scale.astype("float32")).reshape(-1)

    expanded_row_idx_t = torch.tensor(expanded_row_idx, dtype=torch.int32)
    if quant_mode == -1:
        expanded_x_t = torch.from_numpy(expanded_x).to(x_dtype)
    else:
        expanded_x_t = torch.from_numpy(expanded_x)

    return [expanded_x_t, expanded_row_idx_t, expert_tokens_count_t, expanded_scale_t]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x: Tensor,
        expert_idx: Tensor,
        scale: Optional[Tensor],
        offset: Optional[Tensor],
        active_num: int,
        expert_capacity: int,
        expert_num: int,
        drop_pad_mode: int,
        expert_tokens_num_type: int,
        expert_tokens_num_flag: bool,
        quant_mode: int,
        active_expert_range,
        row_idx_type: int,
    ) -> List[Tensor]:
        return _cpu_reference_forward(
            x,
            expert_idx,
            scale,
            offset,
            int(active_num),
            int(expert_capacity),
            int(expert_num),
            int(drop_pad_mode),
            int(expert_tokens_num_type),
            bool(expert_tokens_num_flag),
            int(quant_mode),
            list(active_expert_range),
            int(row_idx_type),
        )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        x: Tensor,
        expert_idx: Tensor,
        scale: Optional[Tensor],
        offset: Optional[Tensor],
        active_num: int,
        expert_capacity: int,
        expert_num: int,
        drop_pad_mode: int,
        expert_tokens_num_type: int,
        expert_tokens_num_flag: bool,
        quant_mode: int,
        active_expert_range,
        row_idx_type: int,
    ) -> List[Tensor]:
        import kernel_gen_ops

        dev = x.device if x.device.type == "npu" else torch.device("npu", torch.npu.current_device())
        x = x.to(dev)
        expert_idx = expert_idx.to(dev)
        if scale is not None and scale.numel() > 0:
            scale = scale.to(dev)
        else:
            scale = None
        if offset is not None and offset.numel() > 0:
            offset = offset.to(dev)
        else:
            offset = None
        outs = list(
            kernel_gen_ops.moe_init_routing_v3(
                x,
                expert_idx,
                scale,
                offset,
                int(active_num),
                int(expert_capacity),
                int(expert_num),
                int(drop_pad_mode),
                int(expert_tokens_num_type),
                bool(expert_tokens_num_flag),
                int(quant_mode),
                list(active_expert_range),
                int(row_idx_type),
            )
        )
        if len(outs) > 3 and outs[3].numel() == 0:
            outs[3] = torch.zeros(1, dtype=torch.float32, device=outs[0].device)
        return outs
