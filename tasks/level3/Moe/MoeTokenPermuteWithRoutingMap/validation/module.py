import ctypes
import glob
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _setup_custom_opp_for_this_task():
    """Bench 从任务目录加载 module.py；须在 import kernel_gen_ops 前设置 OPP 与 LD_LIBRARY_PATH。"""
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
            "MoeTokenPermuteWithRoutingMap",
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


def _permute_reference(
    tokens: Tensor,
    routing_map: Tensor,
    probs_optional: Optional[Tensor],
    num_out_tokens: int,
    drop_and_pad: bool,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """与 ops-transformer-dev executor 中 permute 一致（routing_map 为 [num_tokens, num_experts]）。"""
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]

    cap = num_out_tokens // num_experts
    if drop_and_pad:
        permuted_probs = None
        routing_t = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = routing_t.sort(dim=-1, descending=True, stable=True)[1]
        sorted_indices = sorted_indices[:, :cap].contiguous()
        sorted_indices = sorted_indices.view(-1)
        if probs_optional is not None:
            probs_t_1d = probs_optional.T.contiguous().view(-1)
            indices_dim0 = torch.arange(num_experts, device=routing_t.device).unsqueeze(-1)
            indices_1d = (indices_dim0 * num_tokens + sorted_indices.view(num_experts, cap)).view(-1)
            permuted_probs = probs_t_1d.index_select(0, indices_1d)
        permuted_input = tokens.index_select(0, sorted_indices)
        return permuted_input, permuted_probs, sorted_indices.to(torch.int32)

    routing_t = routing_map.bool().T.contiguous()
    token_indices = torch.arange(num_tokens, device=routing_t.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_t)
    sorted_indices2 = torch.sort(sorted_indices.float(), stable=True)[1]

    if probs_optional is not None:
        permuted_probs = probs_optional.T.masked_select(routing_t)
    else:
        permuted_probs = None
    permuted_input = tokens.index_select(0, sorted_indices)
    return permuted_input, permuted_probs, sorted_indices2.to(torch.int32)


def _pad_rows(tokens_out: Tensor, num_out: int, hidden: int) -> Tensor:
    k = tokens_out.shape[0]
    if k == num_out:
        return tokens_out
    out = torch.zeros(num_out, hidden, dtype=tokens_out.dtype, device=tokens_out.device)
    out[:k] = tokens_out
    return out


def _pad_1d(x: Tensor, num_out: int, dtype: torch.dtype) -> Tensor:
    k = x.shape[0]
    if k == num_out:
        return x
    out = torch.zeros(num_out, dtype=dtype, device=x.device)
    out[:k] = x
    return out


def _cpu_reference_forward(
    tokens: Tensor,
    routing_map: Tensor,
    probs_optional: Optional[Tensor],
    num_out_tokens: int,
    drop_and_pad: bool,
) -> List[Tensor]:
    """CPU 金标准；输出形状与 NPU / pybind 一致（dim0 = num_out_tokens）。"""
    num_tokens, hidden = tokens.shape
    num_out_tokens = int(num_out_tokens)
    drop_and_pad = bool(drop_and_pad)

    pt, pp, si = _permute_reference(tokens, routing_map, probs_optional, num_out_tokens, drop_and_pad)
    pt = _pad_rows(pt, num_out_tokens, hidden)
    si = _pad_1d(si, num_out_tokens, torch.int32)
    if pp is not None:
        pp = _pad_1d(pp, num_out_tokens, pp.dtype)
        return [pt, pp, si]
    # 与 pybind 一致：无 probs 时返回 numel=0，dtype 与 tokens 一致
    empty_probs = torch.empty(0, dtype=tokens.dtype, device=tokens.device)
    return [pt, empty_probs, si]


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tokens, routing_map, probs_optional, num_out_tokens, drop_and_pad):
        if hasattr(num_out_tokens, "item"):
            num_out_tokens = int(num_out_tokens.item())
        else:
            num_out_tokens = int(num_out_tokens)
        drop_and_pad = bool(drop_and_pad)
        return _cpu_reference_forward(tokens, routing_map, probs_optional, num_out_tokens, drop_and_pad)


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tokens, routing_map, probs_optional, num_out_tokens, drop_and_pad):
        import kernel_gen_ops

        if hasattr(num_out_tokens, "item"):
            num_out_tokens = int(num_out_tokens.item())
        else:
            num_out_tokens = int(num_out_tokens)
        drop_and_pad = bool(drop_and_pad)
        target = tokens.device if tokens.device.type == "npu" else torch.device("npu", torch.npu.current_device())
        tokens = tokens.to(target)
        routing_map = routing_map.to(target)
        if probs_optional is not None and probs_optional.numel() > 0:
            probs_optional = probs_optional.to(target)
        else:
            probs_optional = None
        return list(
            kernel_gen_ops.moe_token_permute_with_routing_map(
                tokens, routing_map, probs_optional, num_out_tokens, drop_and_pad
            )
        )


expert_matcher = {
    "float32": {"rtol": 1.0e-5, "atol": 1.0e-5, "total_atol": 1.0e-4},
    "float16": {"rtol": 1.0e-3, "atol": 1.0e-3, "total_atol": 1.0e-3},
    "bfloat16": {"rtol": 1.0e-2, "atol": 1.0e-2, "total_atol": 1.0e-2},
}


def get_model(data_format="channels_first", **kwargs):
    return Model()


def get_model_new(data_format="channels_first", **kwargs):
    return ModelNew()


def get_matcher(dtype="float32", **kwargs):
    return expert_matcher.get(dtype, expert_matcher["float32"])
