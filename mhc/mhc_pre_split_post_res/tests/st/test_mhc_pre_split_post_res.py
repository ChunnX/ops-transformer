#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
import logging
import torch
import torch_npu
import omni_custom_ops
from torch_npu.testing.testcase import TestCase, run_tests
import pytest
import numpy as np
import torch._dynamo as dynamo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMhcPreSplit")

dynamo.config.cache_size_limit = 1000

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

EPS: float = 1e-7
GOLDEN_CPU_DTYPE = torch.float64  # cpu golden 升精度的位数

# ===== 精度等级配置 =====
DEFAULT_PRECISION_LEVEL = "L1"

precision_levels_config = {
    "L0": {"MARE": 10.0, "MERE": 2.0, "RMSE": 2.0},
    "L1": {"MARE": 5.0, "MERE": 1.5, "RMSE": 1.5},
    "L2": {"MARE": 2.0, "MERE": 1.2, "RMSE": 1.2},
}

# ===== 小值域配置 =====
small_value_config = {
    torch.float16: {"threshold": 2**-11, "error": 2**-16},
    torch.bfloat16: {"threshold": 2**-8, "error": 2**-16},
    torch.float32: {"threshold": 2**-14, "error": 2**-30},
}

error_thd_config = {
    torch.float16: 2 ** -11,  # 4.88e-4
    torch.bfloat16: 2 ** -8,  # 3.91e-3
    torch.float32: 2 ** -14,  # 6.10e-5
}


@dataclass
class MhcInputTensors:
    x: torch.Tensor
    phi: torch.Tensor
    alpha: torch.Tensor
    bias: torch.Tensor


@dataclass
class MhcOutputTensors:
    h_post: torch.Tensor
    h_res: torch.Tensor


def absolute_error(actual: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    return torch.abs(actual - golden)


def relative_error(actual: torch.Tensor, golden: torch.Tensor) -> torch.Tensor:
    return torch.abs(actual - golden) / (torch.abs(golden) + EPS)


def max_relative_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    re: torch.Tensor = relative_error(actual, golden)
    return re.max().item()


def mean_relative_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    re: torch.Tensor = relative_error(actual, golden)
    return re.mean().item()


def root_mean_squared_error(actual: torch.Tensor, golden: torch.Tensor) -> float:
    diff: torch.Tensor = actual - golden
    mse: torch.Tensor = torch.mean(diff * diff)
    rmse: torch.Tensor = torch.sqrt(mse)
    return rmse.item()


def calc_error_metrics(actual: torch.Tensor, golden: torch.Tensor) -> dict:
    assert actual.shape == golden.shape
    assert golden.dtype == GOLDEN_CPU_DTYPE

    return {
        "MARE": max_relative_error(actual, golden),
        "MERE": mean_relative_error(actual, golden),
        "RMSE": root_mean_squared_error(actual, golden),
    }


def split_normal_and_small_domain(
    actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
):
    threshold = small_value_config[dtype]["threshold"]

    abs_golden = torch.abs(golden)
    normal_mask = abs_golden >= threshold
    small_mask = abs_golden < threshold

    return (
        actual[normal_mask],
        golden[normal_mask],
        actual[small_mask],
        golden[small_mask],
    )


def normal_domain_pass(
    npu_normal: torch.Tensor,
    gpu_normal: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
    precision_level: str = DEFAULT_PRECISION_LEVEL,
) -> bool:
    npu_metrics = calc_error_metrics(npu_normal, golden)
    gpu_metrics = calc_error_metrics(gpu_normal, golden)

    err_thd = error_thd_config[dtype]

    ratio = {k: npu_metrics[k] / max(gpu_metrics[k] + 1e-7, err_thd) for k in npu_metrics}

    limits = precision_levels_config[precision_level]

    logger.info(f"[NormalDomain] ratio: {ratio}")
    logger.info(f"[NormalDomain] required level: {precision_level}")

    for k, limit in limits.items():
        if ratio[k] > limit:
            logger.warning(f"[NormalDomain] FAIL on {k}: {ratio[k]} > {limit}")
            return False

    logger.info("[NormalDomain] PASS")
    return True


def small_domain_pass(
    npu_actual: torch.Tensor,
    gpu_actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
) -> bool:
    cfg = small_value_config[dtype]
    threshold = cfg["threshold"]
    err = cfg["error"]

    def error_count(actual):
        mask = (torch.abs(golden) < threshold) & (torch.abs(actual - golden) > err)
        return int(mask.sum().item())

    npu_cnt = error_count(npu_actual)
    gpu_cnt = error_count(gpu_actual)

    ratio = npu_cnt / max(gpu_cnt, 1)

    logger.info(
        f"[SmallDomain] NPU ErrorCount={npu_cnt}, "
        f"GPU ErrorCount={gpu_cnt}, Ratio={ratio:.3f}"
    )

    return ratio <= 2.0


def precision_check(
    npu_out: torch.Tensor,
    gpu_out: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
    precision_level: str = DEFAULT_PRECISION_LEVEL,
) -> bool:
    (npu_normal, golden_normal, npu_small, golden_small) = (
        split_normal_and_small_domain(npu_out, golden, dtype)
    )

    (gpu_normal, _, gpu_small, _) = split_normal_and_small_domain(
        gpu_out, golden, dtype
    )

    normal_ok = True
    if golden_normal.numel() > 0:
        normal_ok = normal_domain_pass(
            npu_normal, gpu_normal, golden_normal, dtype, precision_level
        )

    small_ok = True
    if golden_small.numel() > 0:
        small_ok = small_domain_pass(npu_small, gpu_small, golden_small, dtype)

    logger.info(f"[Final] normal_pass={normal_ok}, small_pass={small_ok}")
    return normal_ok and small_ok


class MhcPreSplitModel(torch.nn.Module):
    def forward(self, x_npu, phi_npu, alpha_npu, bias_npu, norm_eps=1e-6):
        h_post_npu, h_res_npu = torch.ops.custom.npu_mhc_pre_split_post_res(
            x_npu,
            phi_npu,
            alpha_npu,
            bias_npu,
            norm_eps=norm_eps
        )
        return h_post_npu, h_res_npu


def aclgraph_npugraphex_result(inputs: MhcInputTensors, norm_eps=1e-6) -> MhcOutputTensors:
    model = MhcPreSplitModel().npu()
    model = torch.compile(model, fullgraph=True, backend="eager", dynamic=False)
    g = torch_npu.npu.NPUGraph()
    with torch.npu.graph(g):
        h_post, h_res = model(inputs.x, inputs.phi, inputs.alpha, inputs.bias, norm_eps=norm_eps)
    g.replay()
    torch.npu.synchronize()
    return MhcOutputTensors(h_post=h_post, h_res=h_res)


def mhc_pre_split_golden_npu(
    inputs: MhcInputTensors,
    norm_eps: float = 1e-6,
) -> MhcOutputTensors:
    x2, phi, alpha, bias = inputs.x, inputs.phi, inputs.alpha, inputs.bias
    B, S, N, D = x2.shape
    x_flat = x2.reshape(B, S, N * D).float()
    inv_rms = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    x_hat = torch.matmul(x_flat, phi.float().T) * inv_rms
    x_post, x_comb = torch.split(x_hat, [N, N * N], dim=-1)
    b_post, b_comb = torch.split(bias.float(), [N, N * N], dim=-1)
    h_post = 2.0 * torch.sigmoid(alpha[0] * x_post + b_post)
    h_res = alpha[1] * x_comb.view(B, S, N, N) + b_comb.view(N, N)
    return MhcOutputTensors(h_post=h_post, h_res=h_res)


def mhc_pre_split_golden_cpu(
    inputs: MhcInputTensors,
    norm_eps: float = 1e-6,
) -> MhcOutputTensors:
    x2, phi, alpha, bias = inputs.x, inputs.phi, inputs.alpha, inputs.bias
    B, S, N, D = x2.shape
    x_flat = x2.reshape(B, S, N * D)
    inv_rms = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    x_hat = torch.matmul(x_flat, phi.T) * inv_rms
    x_post, x_comb = torch.split(x_hat, [N, N * N], dim=-1)
    b_post, b_comb = torch.split(bias, [N, N * N], dim=-1)
    h_post = 2.0 * torch.sigmoid(alpha[0] * x_post + b_post)
    h_res = alpha[1] * x_comb.view(B, S, N, N) + b_comb.view(N, N)
    return MhcOutputTensors(h_post=h_post, h_res=h_res)


class TestMhcPreSplitPostRes(TestCase):
    def _generate_input_tensors(self, B, S, N, D, dtype, seed) -> MhcInputTensors:
        np.random.seed(seed)
        phi_rows = N * N + N

        x_np = np.random.uniform(-2, 2, (B, S, N, D)).astype(np.float32)
        phi_np = np.random.uniform(-2, 2, (phi_rows, N * D)).astype(np.float32)
        alpha_np = np.random.uniform(0.5, 2, (2,)).astype(np.float32)
        bias_np = np.random.uniform(-0.5, 0.5, (phi_rows,)).astype(np.float32)

        return MhcInputTensors(
            x=torch.from_numpy(x_np).to(dtype),
            phi=torch.from_numpy(phi_np),
            alpha=torch.from_numpy(alpha_np),
            bias=torch.from_numpy(bias_np)
        )

    def _compute_cpu_golden(self, inputs: MhcInputTensors, norm_eps) -> MhcOutputTensors:
        outputs = mhc_pre_split_golden_cpu(
            MhcInputTensors(
                x=inputs.x.to(GOLDEN_CPU_DTYPE),
                phi=inputs.phi.to(GOLDEN_CPU_DTYPE),
                alpha=inputs.alpha.to(GOLDEN_CPU_DTYPE),
                bias=inputs.bias.to(GOLDEN_CPU_DTYPE)
            ),
            norm_eps=norm_eps
        )
        return MhcOutputTensors(
            h_post=outputs.h_post.to(GOLDEN_CPU_DTYPE),
            h_res=outputs.h_res.to(GOLDEN_CPU_DTYPE)
        )

    def _compute_npu_golden(self, inputs: MhcInputTensors, norm_eps):
        inputs_npu = MhcInputTensors(
            x=inputs.x.npu(),
            phi=inputs.phi.npu(),
            alpha=inputs.alpha.npu(),
            bias=inputs.bias.npu()
        )
        outputs = mhc_pre_split_golden_npu(inputs_npu, norm_eps)
        return (
            inputs_npu,
            MhcOutputTensors(
                h_post=outputs.h_post.cpu().to(GOLDEN_CPU_DTYPE),
                h_res=outputs.h_res.cpu().to(GOLDEN_CPU_DTYPE)
            )
        )

    def _run_npu_custom_op(
        self, inputs_npu: MhcInputTensors, norm_eps, use_aclgraph
    ) -> MhcOutputTensors:
        if use_aclgraph:
            return aclgraph_npugraphex_result(inputs_npu, norm_eps)
        else:
            h_post, h_res = torch.ops.custom.npu_mhc_pre_split_post_res(
                inputs_npu.x, inputs_npu.phi, inputs_npu.alpha, inputs_npu.bias, norm_eps=norm_eps
            )
            return MhcOutputTensors(h_post=h_post, h_res=h_res)

    def _check_output_precision(self, npu_out, npu_golden_out, cpu_golden_out, dtype, name):
        logger.info(f"=== Checking {name} ===")
        result = precision_check(
            npu_out, npu_golden_out, cpu_golden_out, dtype, DEFAULT_PRECISION_LEVEL
        )
        self.assertEqual(result, True)

    def __run_test_bsnd(
        self,
        B: int,
        S: int,
        N: int,
        D: int,
        dtype: torch.dtype,
        seed: int = 42,
        norm_eps: float = 1e-6,
        use_aclgraph=False
    ):
        # 1. 生成输入（封装类）
        inputs = self._generate_input_tensors(B, S, N, D, dtype, seed)

        # 2. 计算CPU金标准（封装类）
        cpu_outputs = self._compute_cpu_golden(inputs, norm_eps)

        # 3. 计算NPU金标准（封装类）
        inputs_npu, npu_golden_outputs = self._compute_npu_golden(inputs, norm_eps)

        # 4. 运行NPU自定义算子（封装类）
        npu_outputs = self._run_npu_custom_op(inputs_npu, norm_eps, use_aclgraph)

        # 5. 数据类型转换
        npu_outputs = MhcOutputTensors(
            h_post=npu_outputs.h_post.cpu().to(GOLDEN_CPU_DTYPE),
            h_res=npu_outputs.h_res.cpu().to(GOLDEN_CPU_DTYPE)
        )

        # 6. 精度校验
        self._check_output_precision(npu_outputs.h_post, npu_golden_outputs.h_post, cpu_outputs.h_post, dtype, "h_post")
        self._check_output_precision(npu_outputs.h_res, npu_golden_outputs.h_res, cpu_outputs.h_res, dtype, "h_res")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_2_2560_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=2, N=4, D=2560, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_3_2560_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=3, N=4, D=2560, dtype=torch.bfloat16, norm_eps=1e-6, use_aclgraph=True)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4_2560_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=4, N=4, D=2560, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_5_2560_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=5, N=4, D=2560, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_2_5120_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=2, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6, use_aclgraph=True)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_3_5120_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=3, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4_5120_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=4, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_5_5120_pangu_l0_000000(self):
        self.__run_test_bsnd(B=1, S=5, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    # ======================== aclgraph静态kernel编译模式测试 ========================

    @pytest.mark.resources(device="npu:*", npus_per_node=1, cann_min_ver="9.1.0")
    def test_mhc_pre_split_post_res_bf16_aclgraph_static_kernel(self):
        """BF16 aclgraph静态kernel编译模式: B=1, S=2, N=4, D=2560"""
        B, S, N, D = 1, 2, 4, 2560
        dtype = torch.bfloat16
        norm_eps = 1e-6

        # 1. 生成输入
        inputs = self._generate_input_tensors(B, S, N, D, dtype, seed=42)

        # 2. 计算CPU金标准
        cpu_outputs = self._compute_cpu_golden(inputs, norm_eps)

        # 3. 计算NPU金标准
        inputs_npu, npu_golden_outputs = self._compute_npu_golden(inputs, norm_eps)

        # 4. 定义静态kernel Model（必须加@torch.inference_mode()）
        class MhcPreSplitStaticModel(torch.nn.Module):
            @torch.inference_mode()
            def forward(self, x, phi, alpha, bias, norm_eps=1e-6):
                return torch.ops.custom.npu_mhc_pre_split_post_res(
                    x, phi, alpha, bias, norm_eps=norm_eps
                )

        # 5. 初始化模型并编译（使用npugraph_ex + static_kernel_compile）
        model = MhcPreSplitStaticModel().to("npu")
        model_compiled = torch.compile(
            model,
            fullgraph=True,
            backend="npugraph_ex",
            dynamic=False,
            options={"force_eager": True, "static_kernel_compile": True}
        )

        # 6. 模型预执行一次，完成静态kernel编译
        model_compiled(
            inputs_npu.x, inputs_npu.phi, inputs_npu.alpha, inputs_npu.bias,
            norm_eps=norm_eps
        )

        # 7. 录制ACLGraph
        g = torch_npu.npu.NPUGraph()
        with torch.npu.graph(g):
            h_post_npu, h_res_npu = model_compiled(
                inputs_npu.x, inputs_npu.phi, inputs_npu.alpha, inputs_npu.bias,
                norm_eps=norm_eps
            )

        # 8. 重放ACLGraph
        g.replay()
        torch.npu.synchronize()

        # 9. 精度校验
        npu_outputs = MhcOutputTensors(
            h_post=h_post_npu.cpu().to(GOLDEN_CPU_DTYPE),
            h_res=h_res_npu.cpu().to(GOLDEN_CPU_DTYPE)
        )
        self._check_output_precision(
            npu_outputs.h_post, npu_golden_outputs.h_post, cpu_outputs.h_post, dtype, "h_post"
        )
        self._check_output_precision(
            npu_outputs.h_res, npu_golden_outputs.h_res, cpu_outputs.h_res, dtype, "h_res"
        )

if __name__ == "__main__":
    run_tests()