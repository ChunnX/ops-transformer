"""moe_token_unpermute_with_routing_map 的 ATK mssanitizer executor（单 NPU 节点）。

与 gdn_docs/ATK内存检测使用说明.md 中 recompute_w_u_fwd 的 executor 结构一致：
run_cpu / run_npu + FunctionApi 注册。
"""

from __future__ import annotations

import json
from typing import Any

import torch
import torch_npu

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _dtype(name: str) -> torch.dtype:
    return DTYPES[str(name).lower()]


def _marker_device(input_data: InputDataset) -> torch.device:
    for item in input_data.kwargs.values():
        if isinstance(item, torch.Tensor):
            return item.device
    return torch.device("npu:0")


def build_inputs(spec: dict[str, Any], device: torch.device):
    """按 spec 构造 pad 模式输入。

    sorted_indices 语义：expert 主序，每 expert 前 capacity 个被路由 token 的行号。
    routing_map[token, expert] 标记 token 是否路由到 expert。
    spec 支持：
      dtype         token dtype（fp32/fp16/bf16）
      probs_dtype   probs dtype（缺省与 token 一致；混精时用 fp32）
    """
    dtype = _dtype(spec["dtype"])
    prob_dtype = _dtype(spec.get("probs_dtype", spec["dtype"]))
    num_tokens = int(spec["num_tokens"])
    num_experts = int(spec["num_experts"])
    capacity = int(spec["capacity"])
    hidden = int(spec["hidden"])
    seed = int(spec.get("seed", 20260817))
    torch.manual_seed(seed)

    num_out = num_experts * capacity
    # 让每个 expert 都有 capacity 个 token 被路由：token i 路由到 expert (i % num_experts)
    routing_map = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    token_idx = torch.arange(num_out, dtype=torch.long) % num_tokens
    expert_idx = torch.arange(num_experts, dtype=torch.long).repeat_interleave(capacity)
    routing_map[token_idx, expert_idx] = True

    # sorted_indices（expert 主序）直接用上面的 token_idx
    sorted_indices = token_idx.to(torch.int32)

    permuted_tokens = (torch.rand(num_out, hidden) * 2 - 1).to(dtype)
    # probs 只保留被路由位置的权重（与 ST executor 一致）
    probs = torch.rand(num_tokens, num_experts, dtype=torch.float32)
    probs = (probs * routing_map.to(probs.dtype)).to(dtype)
    probs = probs.to(prob_dtype)

    restore_shape = [num_tokens, hidden]
    return {
        "permuted_tokens": permuted_tokens.to(device),
        "sorted_indices": sorted_indices.to(device),
        "restore_shape": restore_shape,
        "probs": probs.to(device),
        "routing_map": routing_map.to(device),
        "drop_and_pad": True,
        "prob_dtype": prob_dtype,
        "is_mixed": dtype in (torch.bfloat16, torch.float16) and prob_dtype == torch.float32,
        "num_experts": num_experts,
        "capacity": capacity,
        "num_tokens": num_tokens,
        "hidden": hidden,
        "dtype": spec["dtype"],
    }


def run_cpu(spec: dict[str, Any], high_precision: bool = False):
    """CPU 参考：与 ST 用例 executor 的 unpermute_padded 一致。"""
    inputs = build_inputs(spec, torch.device("cpu"))
    permuted_tokens = inputs["permuted_tokens"]
    sorted_indices = inputs["sorted_indices"]
    probs = inputs["probs"]
    num_tokens, hidden = inputs["restore_shape"]
    num_experts = inputs["num_experts"]
    capacity = inputs["capacity"]
    # 与仓库 ST 用例 unpermute_padded 一致：仅混精（half token + fp32 probs）才用 fp32 计算
    compute_dtype = torch.float32 if inputs["is_mixed"] else permuted_tokens.dtype
    output_dtype = permuted_tokens.dtype

    expert_indices = torch.arange(num_experts).unsqueeze(-1)
    token_indices = sorted_indices.view(num_experts, capacity)
    prob_indices = (expert_indices * num_tokens + token_indices).view(-1).clamp(0, num_experts * num_tokens - 1)
    permuted_probs = probs.T.contiguous().view(-1).index_select(0, prob_indices)
    weighted_tokens = permuted_tokens.to(compute_dtype) * permuted_probs.to(compute_dtype).unsqueeze(-1)

    sorted_values, sorted_order = sorted_indices.sort(dim=-1, stable=True, descending=False)
    output_tokens = torch.zeros([num_tokens, hidden], dtype=compute_dtype)
    output_tokens.index_add_(0, sorted_values, weighted_tokens[sorted_order])

    return (
        output_tokens.to(output_dtype),
        sorted_order.to(sorted_indices.dtype),
        sorted_values.to(sorted_indices.dtype),
        permuted_probs.to(probs.dtype),
    )


def run_npu(spec: dict[str, Any], input_data: InputDataset):
    """NPU DUT：调用 torch_npu 注册的 aclnn op（_ 版本返回 4 个输出）。"""
    device = _marker_device(input_data)
    inputs = build_inputs(spec, device)
    return torch.ops.npu._npu_moe_token_unpermute_with_routing_map(
        inputs["permuted_tokens"],
        inputs["sorted_indices"],
        inputs["restore_shape"],
        probs=inputs["probs"],
        routing_map=inputs["routing_map"],
        drop_and_pad=inputs["drop_and_pad"],
    )


@register("executor_moe_token_unpermute_with_routing_map")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = json.loads(input_data.kwargs["case_spec"])
        if self.device in {"npu", "pyaclnn"}:
            outputs = run_npu(spec, input_data)
        elif self.device == "cpu":
            outputs = run_cpu(spec)
        else:
            raise RuntimeError(f"unsupported device: {self.device!r}")
        return outputs
