#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Standalone HCCL cascade worker for MatmulAllReduce."""

import datetime
import os
import sys
import traceback

import numpy as np
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
from torch.distributed import ReduceOp


def _load_inputs_for_rank(input_path, did):
    data = np.load(input_path, allow_pickle=False)
    tensors = {}
    for key in data.files:
        if key.startswith(f"did{did}_t") and not key.endswith("_dtype"):
            idx = int(key.split("_t")[1])
            arr = data[key]
            dtype_key = f"{key}_dtype"
            dtype_str = str(data[dtype_key]) if dtype_key in data.files else "float32"
            if arr.dtype.kind == "V":
                total_bytes = arr.size * arr.dtype.itemsize
                arr = np.frombuffer(arr.tobytes(), dtype=np.uint8, count=total_bytes)
                arr = arr.reshape(arr.shape if arr.ndim > 0 else (1,))
            tensor = torch.from_numpy(arr.copy())
            if dtype_str in ("torch.bfloat16", "bfloat16"):
                tensor = tensor.view(torch.bfloat16)
            elif dtype_str in ("torch.float16", "float16"):
                tensor = tensor.to(torch.float16)
            tensors[idx] = tensor
    return tensors


def main():
    rank, world_size, port, input_path, result_path, error_path, is_trans_b = sys.argv[1:8]
    rank = int(rank)
    world_size = int(world_size)
    is_trans_b = is_trans_b == "True"
    try:
        os.environ["HCCL_EXEC_TIMEOUT"] = "3600"
        os.environ["HCCL_LINK_TIMEOUT"] = "3600"
        os.environ["HCCL_CONNECT_TIMEOUT"] = "3600"
        torch.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
            timeout=datetime.timedelta(seconds=3600),
        )
        tensors = _load_inputs_for_rank(input_path, rank)
        x1 = tensors[0].npu()
        x2 = tensors[1].npu()
        if is_trans_b and x1.shape[-1] != x2.shape[0]:
            x2 = x2.transpose(-1, -2)
        bias = (
            tensors[2].npu()
            if 2 in tensors and tensors[2] is not None and tensors[2].numel() > 0
            else None
        )
        if x1.dtype in (torch.int8, torch.uint8):
            from torch.distributed.distributed_c10d import _get_default_group

            default_group = _get_default_group()
            hcomm_info = default_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
            dequant_scale = tensors[4].npu() if 4 in tensors else None
            x3 = tensors[3].npu() if 3 in tensors else None
            mm_out = torch_npu.npu_mm_all_reduce_base(
                x1,
                x2,
                hcomm_info,
                reduce_op="sum",
                bias=bias,
                x3=x3,
                dequant_scale=dequant_scale,
            )
        else:
            if x2.dtype in (torch.int8, torch.uint8):
                antiquant_scale = (
                    tensors[3].npu()
                    if 3 in tensors and tensors[3] is not None and tensors[3].numel() > 0
                    else None
                )
                antiquant_offset = (
                    tensors[4].npu()
                    if 4 in tensors and tensors[4] is not None and tensors[4].numel() > 0
                    else None
                )
                x3 = (
                    tensors[5].npu()
                    if 5 in tensors and tensors[5] is not None and tensors[5].numel() > 0
                    else None
                )
                hcomm_info = dist.distributed_c10d._get_default_group()._get_backend(
                    torch.device("npu")
                ).get_hccl_comm_name(rank)
                mm_out = torch_npu.npu_mm_all_reduce_base(
                    x1,
                    x2,
                    hcomm_info,
                    reduce_op="sum",
                    bias=bias,
                    antiquant_scale=antiquant_scale,
                    antiquant_offset=antiquant_offset,
                    x3=x3,
                    antiquant_group_size=int(os.environ.get("TTK_ANTIQUNT_GROUP_SIZE", "0")),
                )
            else:
                mm_out = torch.matmul(x1, x2)
            if bias is not None:
                mm_out = mm_out + bias
            dist.all_reduce(mm_out, op=ReduceOp.SUM)
        torch.npu.synchronize()
        rank_file = f"{result_path}.did{rank}.npz"
        np.savez(rank_file, cascade=mm_out.reshape(-1).contiguous().cpu().float().numpy())
        dist.destroy_process_group()
    except Exception:
        with open(error_path, "a") as f:
            f.write(f"=== rank {rank} traceback ===\n{traceback.format_exc()}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
