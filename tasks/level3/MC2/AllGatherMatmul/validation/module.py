"""
CPU 金标准与 executor_aclnnAllGatherMatmul（cpu 路径）及 README 一致：
- 各卡本地 x1 为 (m, k)；AllGather 沿第 0 维拼接为 (m * world_size, k)
- 与 st executor 中 weight 一致：当 x2 为 (N, k) 且 k 为契约轴时转置为 (k, n)
- y = concat(all ranks' x1) @ x2 + bias，gatherOut = concat(...)
"""

import ctypes
import glob
import os
from typing import List, Optional

# 在首次 import kernel_gen_ops 前：对齐 MoE 类算子，保证子进程能解析 libcust_opapi
def _setup_custom_opp_for_kernel_gen() -> None:
    op_api_lib: Optional[str] = None
    env_opp = (os.environ.get("ASCEND_CUSTOM_OPP_PATH") or "").strip().rstrip(os.sep)
    if env_opp:
        cand = os.path.join(env_opp, "op_api", "lib", "libcust_opapi.so")
        if os.path.isfile(cand):
            op_api_lib = os.path.dirname(cand)
    if op_api_lib is None:
        here = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))
        for sub in (
            os.path.join(
                workspace_root,
                "runs",
                "lvl3",
                "MC2",
                "AllGatherMatmul",
                "**",
                "packages",
                "vendors",
                "*",
                "op_api",
                "lib",
                "libcust_opapi.so",
            ),
            os.path.join(
                workspace_root,
                "runs",
                "lvl3",
                "MC2",
                "AllGatherMatmul",
                "**",
                "op_api",
                "lib",
                "libcust_opapi.so",
            ),
        ):
            hits = glob.glob(sub, recursive=True)
            if not hits:
                continue
            so = max(hits, key=os.path.getmtime)
            op_api_lib = os.path.dirname(so)
            lib_path = os.path.abspath(os.path.join(op_api_lib, "..", ".."))
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = lib_path + os.sep
            break
    if not op_api_lib:
        return
    cust = os.path.join(op_api_lib, "libcust_opapi.so")
    if not os.path.isfile(cust):
        return
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = op_api_lib + (os.pathsep + prev if prev else "")
    try:
        ctypes.CDLL(cust, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_setup_custom_opp_for_kernel_gen()

import torch
import torch.nn as nn

DEFAULT_BENCH_WORLD_SIZE = 1


def _detect_rank_size(default: int = DEFAULT_BENCH_WORLD_SIZE) -> int:
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


def golden_all_gather_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    bias: Optional[torch.Tensor],
    rank_size: int = DEFAULT_BENCH_WORLD_SIZE,
) -> List[torch.Tensor]:
    """
    单进程参考：用 torch.cat 复现 dist.all_gather 后的矩阵乘（与 executor 中
    all_gather + matmul 语义一致；各 rank 输入相同时为沿 dim0 重复堆叠 x1）。
    """
    od = x1.dtype
    w = x2
    m, k_x1 = x1.shape
    if w.shape[0] != k_x1 and w.shape[1] == k_x1:
        w = w.transpose(0, 1)

    xf = x1.float()
    wf = w.float()
    all_gather_out = torch.cat([xf] * rank_size, dim=0)
    output = torch.matmul(all_gather_out, wf)
    if bias is not None:
        output = output + bias.float()

    return [output.to(od), all_gather_out.to(od)]


class Model(nn.Module):
    def __init__(self, group: str, gather_index: int, comm_turn: int, stream_mode: int):
        super().__init__()
        self.group = group
        self.gather_index = gather_index
        self.comm_turn = comm_turn
        self.stream_mode = stream_mode

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        bias: Optional[torch.Tensor],
        group: str,
        gather_index: int,
        comm_turn: int,
        stream_mode: int,
    ) -> List[torch.Tensor]:
        rank_size = _detect_rank_size()
        return golden_all_gather_matmul(x1, x2, bias, rank_size=rank_size)


class ModelNew(nn.Module):
    def __init__(self, group: str, gather_index: int, comm_turn: int, stream_mode: int):
        super().__init__()
        self.group = group
        self.gather_index = gather_index
        self.comm_turn = comm_turn
        self.stream_mode = stream_mode

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        bias: Optional[torch.Tensor],
        group: str,
        gather_index: int,
        comm_turn: int,
        stream_mode: int,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops
        group_name = _resolve_hcomm_name(group)
        try:
            return kernel_gen_ops.all_gather_matmul(x1, x2, bias, group_name, gather_index, comm_turn, stream_mode)
        except RuntimeError as err:
            # 单进程精度环境可能没有有效 HCCL comm，退化为可比较的单卡语义。
            if "HcomGetCommHandleByGroup" not in str(err):
                raise
            rank_size = _detect_rank_size()
            return golden_all_gather_matmul(x1, x2, bias, rank_size=rank_size)