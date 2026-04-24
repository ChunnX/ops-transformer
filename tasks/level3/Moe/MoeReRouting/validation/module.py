import ctypes
import glob
import os
from typing import List, Optional

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
            "MoeReRouting",
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


def _cpu_reference_forward(
    tokens: Tensor,
    expert_token_num_per_rank: Tensor,
    per_token_scales: Optional[Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
) -> List[Tensor]:
    """CPU 金标准：与 api_desc 中 Src/Dst 偏移及 kernel CalSrcOffSet/CalDstOffSet 一致。"""
    N, E = expert_token_num_per_rank.shape
    A, _H = tokens.shape
    idx_dtype = expert_token_num_per_rank.dtype

    src_offsets = torch.zeros(N, E, dtype=torch.int32, device=tokens.device)
    dst_offsets = torch.zeros(N, E, dtype=torch.int32, device=tokens.device)

    for i in range(N):
        for j in range(E):
            if i == 0 and j == 0:
                src_offsets[i, j] = 0
            elif j == 0:
                src_offsets[i, j] = src_offsets[i - 1, E - 1] + expert_token_num_per_rank[i - 1, E - 1]
            else:
                src_offsets[i, j] = src_offsets[i, j - 1] + expert_token_num_per_rank[i, j - 1]

    for j in range(E):
        for i in range(N):
            if i == 0 and j == 0:
                dst_offsets[i, j] = 0
            elif i == 0:
                dst_offsets[i, j] = dst_offsets[N - 1, j - 1] + expert_token_num_per_rank[N - 1, j - 1]
            else:
                dst_offsets[i, j] = dst_offsets[i - 1, j] + expert_token_num_per_rank[i - 1, j]

    permute_token_idx = torch.zeros(A, dtype=torch.int32, device=tokens.device)
    permute_tokens = torch.zeros_like(tokens)
    permute_per_token_scales: Optional[Tensor] = None
    if per_token_scales is not None:
        permute_per_token_scales = torch.zeros_like(per_token_scales)

    for i in range(N):
        for j in range(E):
            count = int(expert_token_num_per_rank[i, j].item())
            if count == 0:
                continue
            src_start = int(src_offsets[i, j].item())
            src_end = src_start + count
            dst_start = int(dst_offsets[i, j].item())
            dst_end = dst_start + count
            permute_tokens[dst_start:dst_end] = tokens[src_start:src_end]
            if idx_type == 0:
                permute_token_idx[dst_start:dst_end] = torch.arange(
                    src_start, src_end, dtype=torch.int32, device=tokens.device
                )
            else:
                permute_token_idx[src_start:src_end] = torch.arange(
                    dst_start, dst_end, dtype=torch.int32, device=tokens.device
                )
            if per_token_scales is not None and permute_per_token_scales is not None:
                permute_per_token_scales[dst_start:dst_end] = per_token_scales[src_start:src_end]

    expert_token_num = expert_token_num_per_rank.sum(dim=0).to(idx_dtype)
    if expert_token_num_type == 0:
        expert_token_num = torch.cumsum(expert_token_num, dim=0)

    outs: List[Tensor] = [permute_tokens]
    if permute_per_token_scales is not None:
        outs.append(permute_per_token_scales)
    outs.extend([permute_token_idx, expert_token_num])
    return outs


class Model(nn.Module):
    """CPU 金标准（本模块内 `_cpu_reference_forward`）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        per_token_scales: Optional[Tensor] = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
    ) -> List[Tensor]:
        return _cpu_reference_forward(
            tokens,
            expert_token_num_per_rank,
            per_token_scales,
            int(expert_token_num_type),
            int(idx_type),
        )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        tokens: Tensor,
        expert_token_num_per_rank: Tensor,
        per_token_scales: Optional[Tensor] = None,
        expert_token_num_type: int = 1,
        idx_type: int = 0,
    ) -> List[Tensor]:
        import kernel_gen_ops

        dev = tokens.device if tokens.device.type == "npu" else torch.device("npu", torch.npu.current_device())
        tokens = tokens.to(dev)
        expert_token_num_per_rank = expert_token_num_per_rank.to(dev)
        if per_token_scales is not None and per_token_scales.numel() > 0:
            per_token_scales = per_token_scales.to(dev)
        else:
            per_token_scales = None
        return list(
            kernel_gen_ops.moe_re_routing(
                tokens,
                expert_token_num_per_rank,
                per_token_scales,
                int(expert_token_num_type),
                int(idx_type),
            )
        )
