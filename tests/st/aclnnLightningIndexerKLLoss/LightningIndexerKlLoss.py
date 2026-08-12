#!/usr/bin/python
# -*- coding: utf-8 -*-
# 双标杆执行器：
#   api_type      = executor_lightning_indexer_kl_loss        -> CPU 标杆（BaseApi）
#   aclnn_api_type = executor_lightning_indexer_kl_loss_aclnn  -> NPU torch 接口（BaseApi）
# torch 接口签名：
#   lightning_indexer_kl_loss(Tensor target_score, Tensor index_probs,
#                             float eps=1e-9, str weight_type='logits') -> Tensor

import sys

import torch

# ATK 内部用 torch.load(weights_only=True) 加载本地数据集，会拒绝 numpy 对象。
# 本地生成的数据集可信，强制 weights_only=False 以避免逐个 allowlist numpy 类型。
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# 确保 torch_extension（cann_ops_transformer 所在目录）可被导入
_TORCH_EXT_DIR = "/data/jzh/ops-transformer/torch_extension"
if _TORCH_EXT_DIR not in sys.path:
    sys.path.insert(0, _TORCH_EXT_DIR)

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def _kl_loss_cpu(target_score: torch.Tensor, index_probs: torch.Tensor,
                 eps: float, weight_type: str) -> torch.Tensor:
    """CPU 标杆，对齐 torch_extension/test_lightning_indexer_kl_loss.py L30-57。

    reduce_target = sum(y, dim=-1, keepdim=True)
    P = y / (reduce_target + eps)
    log_P = log(clamp_min(P, eps))
    log_Y = log(Y + eps)
    log_ratio = log_P - log_Y
    weight = y  if weight_type == 'logits'  else  P
    result = weight * log_ratio
    return sum(result)

    计算精度：输入为 fp64 时用 fp64（供 cv_fused_double_benchmark 高精度侧），
    否则统一 fp32（避免 bf16/fp16 精度损失）。
    """
    orig_dtype = target_score.dtype
    compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
    y = target_score.to(compute_dtype)
    Y = index_probs.to(compute_dtype)
    reduce_target = torch.sum(y, dim=-1, keepdim=True)
    P = y / (reduce_target + eps)
    log_P = torch.log(torch.clamp_min(P, eps))
    log_Y = torch.log(Y + eps)
    log_ratio = log_P - log_Y
    if weight_type == "logits":
        weight = y
    elif weight_type == "probs":
        weight = P
    else:
        raise ValueError("weight_type must be 'logits' or 'probs'")
    result = weight * log_ratio
    loss = torch.sum(result)
    return loss.to(orig_dtype)


def _kl_loss_npu(target_score: torch.Tensor, index_probs: torch.Tensor,
                 eps: float, weight_type: str) -> torch.Tensor:
    """NPU 侧：调用 torch 接口 cann_ops_transformer.ops.lightning_indexer_kl_loss。"""
    from cann_ops_transformer.ops import lightning_indexer_kl_loss
    return lightning_indexer_kl_loss(target_score, index_probs, eps, weight_type)


@register("executor_lightning_indexer_kl_loss")
class LightningIndexerKLLossCpuApi(BaseApi):
    """CPU 标杆执行器。"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        target_score = input_data.kwargs["targetScore"]
        index_probs = input_data.kwargs["indexProbs"]
        eps = input_data.kwargs.get("eps", 1e-9)
        weight_type = input_data.kwargs.get("weightType", "logits")
        return _kl_loss_cpu(target_score, index_probs, eps, weight_type)


@register("executor_lightning_indexer_kl_loss_aclnn")
class LightningIndexerKLLossNpuApi(BaseApi):
    """NPU torch 接口执行器（双标杆 NPU 侧）。

    torch 接口不走 aclnn C API，继承 BaseApi，用 --backend npu 执行。
    """

    def init_by_input_data(self, input_data: InputDataset):
        # tensor 上 NPU
        for k, v in list(input_data.kwargs.items()):
            if isinstance(v, torch.Tensor) and not v.is_npu:
                input_data.kwargs[k] = v.npu()
        # 记录输出 dtype（标量输出，dtype 跟随 target_score）
        if self.task_result.output_info_list:
            ts_dtype = input_data.kwargs["targetScore"].dtype
            self.task_result.output_info_list[0].dtype = str(ts_dtype)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        target_score = input_data.kwargs["targetScore"]
        index_probs = input_data.kwargs["indexProbs"]
        eps = input_data.kwargs.get("eps", 1e-9)
        weight_type = input_data.kwargs.get("weightType", "logits")
        return _kl_loss_npu(target_score, index_probs, eps, weight_type)
