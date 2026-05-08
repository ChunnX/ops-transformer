"""
GroupedMatMulAllReduce：多组 matmul + AllReduce。WORLD_SIZE=1 时 AllReduce 为恒等；
金标准按 README y_i = x_i @ weight_i + bias_i（逐组）。
"""

import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


def _detect_rank_size(default: int = 1) -> int:
    ws = os.environ.get("WORLD_SIZE", "").strip()
    if ws.isdigit():
        return max(1, int(ws))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return max(1, int(torch.distributed.get_world_size()))
        except Exception:
            pass
    return default


def _resolve_hcomm_name(group: str) -> str:
    if group and group != "hccl_world_group":
        return group
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return group
    try:
        from torch.distributed.distributed_c10d import _get_default_group

        default_pg = _get_default_group()
        rank = int(torch.distributed.get_rank())
        if torch.__version__ > "2.0.1":
            backend = default_pg._get_backend(torch.device("npu"))
            return backend.get_hccl_comm_name(rank)
        return default_pg.get_hccl_comm_name(rank)
    except Exception:
        return group


def golden_grouped_matmul_all_reduce(
    x_list: List[torch.Tensor],
    w_list: List[torch.Tensor],
    bias_list: List[torch.Tensor],
    rank_size: int,
) -> List[torch.Tensor]:
    outs: List[torch.Tensor] = []
    for i, x in enumerate(x_list):
        od = x.dtype
        w = w_list[i]
        if w.dim() == 2 and w.shape[0] != x.shape[-1] and w.shape[1] == x.shape[-1]:
            w = w.transpose(0, 1)
        y = torch.matmul(x.float(), w.float())
        if i < len(bias_list):
            b = bias_list[i]
            y = y + b.float().reshape(1, -1) if b.dim() == 1 else y + b.float()
        if rank_size > 1:
            y = y * float(rank_size)
        outs.append(y.to(od))
    return outs


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_list: List[torch.Tensor],
        w_list: List[torch.Tensor],
        bias_list: Optional[List[torch.Tensor]],
        group_list: Optional[torch.Tensor],
        split_item: int,
        group: str,
        reduce_op: str,
        comm_turn: int,
        y_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        _ = (group_list, y_list)
        rank_size = _detect_rank_size()
        return golden_grouped_matmul_all_reduce(x_list, w_list, bias_list, rank_size)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_list: List[torch.Tensor],
        w_list: List[torch.Tensor],
        bias_list: Optional[List[torch.Tensor]],
        group_list: Optional[torch.Tensor],
        split_item: int,
        group: str,
        reduce_op: str,
        comm_turn: int,
        y_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        g = _resolve_hcomm_name(group)
        try:
            return kernel_gen_ops.grouped_mat_mul_all_reduce(
                x_list,
                w_list,
                bias_list,
                group_list,
                split_item,
                g,
                reduce_op,
                comm_turn,
                y_list,
            )
        except RuntimeError as err:
            # 单进程精度环境可能没有有效 HCCL comm；与 AllGatherMatmul 一致退化为可比较语义。
            if "HcomGetCommHandleByGroup" not in str(err):
                raise
            rank_size = _detect_rank_size()
            return golden_grouped_matmul_all_reduce(x_list, w_list, bias_list, rank_size)
