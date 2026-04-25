import ctypes
import glob
import os


def _setup_custom_opp_for_this_task():
    """Bench 从任务目录加载 module.py；须在 import kernel_gen_ops 前设置 OPP 与 dlopen libcust_opapi。"""
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
            "MoeTokenPermuteWithRoutingMapGrad",
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

import torch
import torch.nn as nn


def permute_grad_with_routing_map_drop(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    hidden: int,
    capacity: int,
):
    sorted_values, perm_idx = torch.sort(sorted_indices.float(), stable=True)
    sorted_values = sorted_values.to(torch.int64)
    perm_idx = perm_idx.to(torch.int64)
    output_tokens = torch.zeros(
        (num_tokens, hidden), dtype=torch.float64, device=permuted_tokens.device
    )
    for i in range(sorted_indices.numel()):
        output_tokens[sorted_values[i]] += permuted_tokens[perm_idx[i]]
    return output_tokens.to(dtype=permuted_tokens.dtype)


class Model(nn.Module):
    """CPU 金标准。

    - 非 padded：token 梯度用链式法则 scatter_add（与 permuted 行、routing 一致）；
    - padded：token/probs 与 op_kernel 中 KernelMoeTokenPermuteWithRoutingMap::Process 一致。
    """

    def __init__(self, use_probs: bool = True):
        super().__init__()
        self.use_probs = use_probs

    def forward(
        self,
        permuted_token_output_grad: torch.Tensor,
        permuted_probs_output_grad_optional: torch.Tensor,
        sorted_indices: torch.Tensor,
        routing_map_optional: torch.Tensor,
        experts_num: int,
        tokens_num: int,
        drop_and_pad: bool,
    ):
        num_tokens = int(tokens_num)
        num_experts = int(experts_num)
        drop_and_pad = bool(drop_and_pad)
        hidden_size = permuted_token_output_grad.size(1)
        num_out_tokens = permuted_token_output_grad.size(0)
        dtype = permuted_token_output_grad.dtype

        if drop_and_pad is False:
            rm_t = routing_map_optional.bool().T.contiguous()
            ti = torch.arange(num_tokens, device=permuted_token_output_grad.device, dtype=torch.int64).unsqueeze(0).expand(
                num_experts, -1
            )
            token_ids_flat = ti.masked_select(rm_t)
            acc = torch.zeros(
                (num_tokens, hidden_size), dtype=torch.float32, device=permuted_token_output_grad.device
            )
            acc.scatter_add_(
                0,
                token_ids_flat.unsqueeze(1).expand(-1, hidden_size),
                permuted_token_output_grad.to(torch.float32),
            )
            tokens_grad = acc.to(dtype)
            if self.use_probs:
                probs_grad = torch.zeros((num_experts, num_tokens), dtype=dtype, device=permuted_token_output_grad.device)
                probs_grad.masked_scatter_(
                    routing_map_optional.T.contiguous(),
                    permuted_probs_output_grad_optional.reshape(-1),
                )
                probs_grad = probs_grad.T
                return [tokens_grad, probs_grad]
            return [tokens_grad, torch.tensor([0.0], dtype=dtype, device=permuted_token_output_grad.device)]

        capacity = num_out_tokens // num_experts
        tokens_grad = permute_grad_with_routing_map_drop(
            permuted_token_output_grad,
            sorted_indices,
            num_tokens,
            hidden_size,
            capacity,
        )
        if self.use_probs:
            probs_grad = torch.zeros(
                (num_tokens, num_experts), dtype=dtype, device=permuted_token_output_grad.device
            )
            si = sorted_indices.to(torch.int64)
            flat_probs = permuted_probs_output_grad_optional.reshape(-1)
            for i in range(num_experts * capacity):
                dim0 = i // capacity
                dim1 = si[i]
                probs_grad[dim1, dim0] = flat_probs[i]
            return [tokens_grad, probs_grad]
        return [tokens_grad, torch.tensor([0.0], dtype=dtype, device=permuted_token_output_grad.device)]


class ModelNew(nn.Module):
    def __init__(self, use_probs: bool = True):
        super().__init__()
        self.use_probs = use_probs

    def forward(
        self,
        permuted_token_output_grad: torch.Tensor,
        permuted_probs_output_grad_optional: torch.Tensor,
        sorted_indices: torch.Tensor,
        routing_map_optional: torch.Tensor,
        experts_num: int,
        tokens_num: int,
        drop_and_pad: bool,
    ):
        import kernel_gen_ops

        target = (
            permuted_token_output_grad.device
            if permuted_token_output_grad.device.type == "npu"
            else torch.device("npu", torch.npu.current_device())
        )
        x = permuted_token_output_grad.to(target)
        si = sorted_indices.to(target)
        rm = routing_map_optional.to(target)
        if not self.use_probs:
            raise RuntimeError("NPU 路径当前仅在校验 use_probs=True 的用例上配置 optional 输入。")
        pg = permuted_probs_output_grad_optional.to(target)
        tokens_grad, probs_grad = kernel_gen_ops.moe_token_permute_with_routing_map_grad(
            x, pg, si, rm, int(experts_num), int(tokens_num), bool(drop_and_pad)
        )
        return [tokens_grad, probs_grad]
