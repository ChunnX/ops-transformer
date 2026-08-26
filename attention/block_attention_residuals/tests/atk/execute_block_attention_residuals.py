"""ATK adapter for the default BlockAttentionResiduals scenario."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PYTEST_ROOT = _REPO_ROOT / "attention" / "block_attention_residuals" / "tests" / "pytest"
if str(_PYTEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTEST_ROOT))

import block_attention_residuals_golden as golden  # noqa: E402
import block_attention_residuals_torch_runner as torch_runner  # noqa: E402


@register("function_block_attention_residuals_atk")
class BlockAttentionResidualsAtk(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        args = input_data.kwargs
        prefix_sum = args["prefix_sum"]
        block_res = args["block_res"]
        proj_weight = args["proj_weight"]
        norm_weight = args["norm_weight"]
        valid_block_num = int(args["valid_block_num"])
        norm_eps = float(args["norm_eps"])
        need_backward = bool(args["need_backward"])
        if need_backward:
            raise ValueError("ATK case is restricted to need_backward=false")

        if self.device == "cpu":
            return golden.apply_attn_res_cust(
                prefix_sum, block_res, proj_weight, norm_weight, norm_eps
            )
        if self.device == "npu":
            op = torch_runner.get_block_attention_residuals_fn()
            output = op(
                prefix_sum.to(f"npu:{self.device_id}"),
                block_res.to(f"npu:{self.device_id}"),
                proj_weight.to(f"npu:{self.device_id}"),
                norm_weight.to(f"npu:{self.device_id}"),
                valid_block_num,
                norm_eps,
                False,
            )
            torch.npu.synchronize()
            return output
        raise RuntimeError(f"Unsupported device: {self.device}")

@register("pyaclnn_block_attention_residuals_atk")
class PyAclnnBlockAttentionResidualsAtk(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_args = []
        output_packages = []

        for name, value in input_data.kwargs.items():
            input_args.extend(self.backend.convert_input_data(value, name=name))

        for index, output_data in enumerate(self.task_result.output_info_list):
            output_packages.extend(self.backend.convert_output_data(output_data, index))

        input_args.extend(output_packages)
        return input_args, output_packages

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnBlockAttentionResidualsGetWorkspaceSize("
            "const aclTensor *prefixSum, const aclTensor *blockRes, "
            "const aclTensor *projWeight, const aclTensor *normWeight, "
            "int64_t validBlockNum, double normEps, bool needBackward, "
            "aclTensor *hiddenStates, aclTensor *invNorm, aclTensor *probs, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )