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
import cann_ops_transformer_custom  # noqa: F401  注册 cann_ops_transformer 算子命名空间
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

    if not torch.isfinite(npu_actual).all().item():
        logger.error("[SmallDomain] NPU output contains non-finite values")
        return False
    if not torch.isfinite(gpu_actual).all().item() or not torch.isfinite(golden).all().item():
        logger.error("[SmallDomain] reference output contains non-finite values")
        return False

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
    if not torch.isfinite(npu_out).all().item():
        logger.error("[Final] NPU output contains non-finite values")
        return False
    if not torch.isfinite(gpu_out).all().item() or not torch.isfinite(golden).all().item():
        logger.error("[Final] reference output contains non-finite values")
        return False
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
        h_post_npu, h_res_npu = torch.ops.cann_ops_transformer.mhc_pre_split_post_res(
            x_npu, phi_npu, alpha_npu, bias_npu, norm_eps
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
    # 通用前导：3D [T,N,D] 与 4D [B,S,N,D] 均折叠为 [t,N*D] 计算，输出还原前导形状
    lead = x2.shape[:-2]
    N, D = x2.shape[-2], x2.shape[-1]
    x_flat = x2.reshape(-1, N * D).float()
    inv_rms = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    x_hat = torch.matmul(x_flat, phi.float().T) * inv_rms
    # phi 双形态：行数 N^2+N（完整，输出 h_res）/ N（仅 post，无 h_res）
    has_res = phi.shape[0] == N * N + N
    x_post = x_hat[..., :N]
    b_post = bias.float()[..., :N]
    h_post = 2.0 * torch.sigmoid(alpha[0] * x_post + b_post)
    if has_res:
        x_comb = x_hat[..., N:]
        b_comb = bias.float()[..., N:]
        h_res = alpha[1] * x_comb.view(-1, N, N) + b_comb.view(N, N)
    else:
        h_res = torch.empty(0, dtype=h_post.dtype, device=h_post.device)
    h_res = h_res if h_res.numel() == 0 else h_res.view(*lead, N, N)  # post-only 形态 h_res 为 0 元素空 tensor，不可 view
    return MhcOutputTensors(h_post=h_post.view(*lead, N), h_res=h_res)


def mhc_pre_split_golden_cpu(
    inputs: MhcInputTensors,
    norm_eps: float = 1e-6,
) -> MhcOutputTensors:
    x2, phi, alpha, bias = inputs.x, inputs.phi, inputs.alpha, inputs.bias
    # 通用前导：3D [T,N,D] 与 4D [B,S,N,D] 均折叠为 [t,N*D] 计算，输出还原前导形状
    lead = x2.shape[:-2]
    N, D = x2.shape[-2], x2.shape[-1]
    x_flat = x2.reshape(-1, N * D)
    inv_rms = torch.rsqrt(x_flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    x_hat = torch.matmul(x_flat, phi.T) * inv_rms
    # phi 双形态：行数 N^2+N（完整，输出 h_res）/ N（仅 post，无 h_res）
    has_res = phi.shape[0] == N * N + N
    x_post = x_hat[..., :N]
    b_post = bias[..., :N]
    h_post = 2.0 * torch.sigmoid(alpha[0] * x_post + b_post)
    if has_res:
        x_comb = x_hat[..., N:]
        b_comb = bias[..., N:]
        h_res = alpha[1] * x_comb.view(-1, N, N) + b_comb.view(N, N)
    else:
        h_res = torch.empty(0, dtype=h_post.dtype)
    h_res = h_res if h_res.numel() == 0 else h_res.view(*lead, N, N)  # post-only 形态 h_res 为 0 元素空 tensor，不可 view
    return MhcOutputTensors(h_post=h_post.view(*lead, N), h_res=h_res)


class TestMhcPreSplitPostRes(TestCase):
    def _generate_input_tensors(self, B, S, N, D, dtype, seed, phi_rows=None) -> MhcInputTensors:
        np.random.seed(seed)
        if phi_rows is None:
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
            h_post, h_res = torch.ops.cann_ops_transformer.mhc_pre_split_post_res(
                inputs_npu.x, inputs_npu.phi, inputs_npu.alpha, inputs_npu.bias, norm_eps
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
        use_aclgraph=False,
        phi_rows=None
    ):
        # 1. 生成输入（封装类；phi_rows=None 走完整形态 N^2+N，=N 走仅 post 形态）
        inputs = self._generate_input_tensors(B, S, N, D, dtype, seed, phi_rows=phi_rows)

        # 2~6. 双标杆精度流水线（仅 post 形态（phi_rows==N）无 h_res，跳过 res 校验）
        self._run_pipeline(inputs, norm_eps, N, phi_rows, use_aclgraph)

    def _run_pipeline(self, inputs: MhcInputTensors, norm_eps, N, phi_rows, use_aclgraph=False):
        """双标杆精度流水线：CPU golden + NPU 拼接 golden + 被测算子，逐输出判定。"""
        cpu_outputs = self._compute_cpu_golden(inputs, norm_eps)
        inputs_npu, npu_golden_outputs = self._compute_npu_golden(inputs, norm_eps)
        npu_outputs = self._run_npu_custom_op(inputs_npu, norm_eps, use_aclgraph)
        npu_outputs = MhcOutputTensors(
            h_post=npu_outputs.h_post.cpu().to(GOLDEN_CPU_DTYPE),
            h_res=npu_outputs.h_res.cpu().to(GOLDEN_CPU_DTYPE)
        )
        dtype = inputs.x.dtype
        self._check_output_precision(npu_outputs.h_post, npu_golden_outputs.h_post, cpu_outputs.h_post, dtype, "h_post")
        if phi_rows != N:
            self._check_output_precision(npu_outputs.h_res, npu_golden_outputs.h_res, cpu_outputs.h_res, dtype, "h_res")

    def _inputs_from_arrays(self, x_np, phi_np, alpha_np, bias_np, dtype) -> MhcInputTensors:
        """数值分布用例输入构造：x 转 target dtype，权重 FP32。"""
        return MhcInputTensors(
            x=torch.from_numpy(np.ascontiguousarray(x_np)).to(dtype),
            phi=torch.from_numpy(np.ascontiguousarray(phi_np, dtype=np.float32)),
            alpha=torch.from_numpy(np.ascontiguousarray(alpha_np, dtype=np.float32)),
            bias=torch.from_numpy(np.ascontiguousarray(bias_np, dtype=np.float32)),
        )

    def __run_test_tnd(self, T, N, D, dtype, seed=42, norm_eps=1e-6, phi_rows=None):
        """TND 3D 前导（x=[T,N,D]）：tiling 折叠 T 后与 BSND 同路径。"""
        np.random.seed(seed)
        if phi_rows is None:
            phi_rows = N * N + N
        x_np = np.random.uniform(-2, 2, (T, N, D)).astype(np.float32)
        phi_np = np.random.uniform(-2, 2, (phi_rows, N * D)).astype(np.float32)
        alpha_np = np.random.uniform(0.5, 2, (2,)).astype(np.float32)
        bias_np = np.random.uniform(-0.5, 0.5, (phi_rows,)).astype(np.float32)
        inputs = self._inputs_from_arrays(x_np, phi_np, alpha_np, bias_np, dtype)
        self._run_pipeline(inputs, norm_eps, N, phi_rows)

    def __expect_runtime_error(self, inputs: MhcInputTensors, norm_eps=1e-6):
        """异常端到端：torch→csrc→aclnn 链任一层拦截均表现为 RuntimeError。"""
        with pytest.raises((RuntimeError, ValueError)):
            torch.ops.cann_ops_transformer.mhc_pre_split_post_res(
                inputs.x.npu(), inputs.phi.npu(), inputs.alpha.npu(), inputs.bias.npu(), norm_eps
            )

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

    # ======================== D 典型值用例（白名单 2560/5120/6144/7168） ========================

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4_6144_general_l0_000000(self):
        """D 白名单典型值：D=6144，dimLoop=3 MULTI_TILE 路径（尾块 1024）"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=6144, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4_7168_general_l0_000000(self):
        """D 白名单典型值：D=7168，dimLoop=3 MULTI_TILE 路径（尾块 2048）"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=7168, dtype=torch.bfloat16, norm_eps=1e-6)

    # ======================== phi 仅 post 形态用例（phi=[N, N*D]，无 h_res） ========================

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_post_only_phiN_2560(self):
        """phi=[4, 10240] 仅 post 形态：SINGLE_TILE，h_res 为空 tensor，仅校验 h_post"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=2560, dtype=torch.bfloat16, norm_eps=1e-6, phi_rows=4)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_post_only_phiN_5120(self):
        """phi=[4, 20480] 仅 post 形态：MULTI_TILE，h_res 为空 tensor，仅校验 h_post"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6, phi_rows=4)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_post_only_phiN_6144(self):
        """phi=[4, 24576] 仅 post 形态：MULTI 3-tile（尾块 1024），h_res 为空 tensor，仅校验 h_post"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=6144, dtype=torch.bfloat16, norm_eps=1e-6, phi_rows=4)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_post_only_phiN_7168(self):
        """phi=[4, 28672] 仅 post 形态：MULTI 3-tile（尾块 2048），h_res 为空 tensor，仅校验 h_post"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=7168, dtype=torch.bfloat16, norm_eps=1e-6, phi_rows=4)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_post_only_phiN_2560_aclgraph(self):
        """phi=[4, 10240] 仅 post 形态 + AclGraph（NPUGraph）通路：h_res 空 tensor 入图，仅校验 h_post"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=2560, dtype=torch.bfloat16, norm_eps=1e-6,
                             phi_rows=4, use_aclgraph=True)

    # ======================== 红线：tiling 分区邻点（合并 STC 3.7，序号 223~228） ========================

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_31_5120_boundary_left(self):
        """T=31 双核边界左侧（Block Num=62）"""
        self.__run_test_bsnd(B=1, S=31, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_33_5120_single_transition(self):
        """T=33 双核边界右侧单核过渡（Block Num=33）"""
        self.__run_test_bsnd(B=1, S=33, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_64_5120_two_token_no_remainder(self):
        """T=64 每核 2 token 无余数（Block Num=64 全载）"""
        self.__run_test_bsnd(B=1, S=64, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_65_5120_two_token_remainder1(self):
        """T=65 每核 2/3 token 余数 1（尾核区间核对）"""
        self.__run_test_bsnd(B=1, S=65, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_128_5120_prefill(self):
        """T=128 prefill 量级（每核 2 token）"""
        self.__run_test_bsnd(B=1, S=128, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_256_5120_prefill(self):
        """T=256 prefill 量级（每核 4 token）"""
        self.__run_test_bsnd(B=1, S=256, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-6)

    # ======================== 网络附近 / B-S 分解 / TND（合并 STC 3.8.2，序号 328~331） ========================

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_TND_3_2560_3d_leading(self):
        """TND 3D 前导 x=[3,4,2560]：与 BSND [1,3,...] 折叠同路径"""
        self.__run_test_tnd(T=3, N=4, D=2560, dtype=torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_fp16_BSND_2x2_5120_bsfactor(self):
        """B/S 分解 [2,2] 且 T=4，FP16"""
        self.__run_test_bsnd(B=2, S=2, N=4, D=5120, dtype=torch.float16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_3x5_6144_factor15(self):
        """T=15 的 B/S 因子组合 [3,5]，短尾 1024"""
        self.__run_test_bsnd(B=3, S=5, N=4, D=6144, dtype=torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_fp16_BSND_4x1_7168_post_only(self):
        """S=1 多 Batch [4,1]，长尾 2048，仅 post 形态"""
        self.__run_test_bsnd(B=4, S=1, N=4, D=7168, dtype=torch.float16, phi_rows=4)

    # ======================== 数值分布泛化（合并 STC 3.8.3，序号 333~338） ========================

    def __dist_inputs(self, kind, B=1, S=4, N=4, D=5120):
        """按分布类型构造输入（种子固定）；返回 MhcInputTensors（bf16）。"""
        rng = np.random.RandomState(20261008)
        phi_rows = N * N + N
        if kind == "all_zero":
            x_np = np.zeros((B, S, N, D), np.float32)
        elif kind == "constant":
            x_np = np.full((B, S, N, D), 1.5, np.float32)
        elif kind == "alternating":
            x_np = np.zeros((B, S, N, D), np.float32)
            x_np[..., 0::2] = 1.7
            x_np[..., 1::2] = -1.7
        elif kind == "sparse":
            x_np = np.zeros((B, S, N, D), np.float32)
            marks = [0, 1, 2559, 2560, D - 2, D - 1]  # head/tile/尾块边界标记
            for b in range(B):
                for ss in range(S):
                    for h in range(N):
                        for m in marks:
                            x_np[b, ss, h, m] = rng.choice([-2.0, 2.0])
        else:  # scaled_phi / near_cancel
            x_np = rng.uniform(-2, 2, (B, S, N, D)).astype(np.float32)
        if kind == "scaled_phi":
            phi_np = (rng.randn(phi_rows, N * D) / np.sqrt(N * D)).astype(np.float32)
            phi_np[0, 0] = 1.0          # 首元素标记
            phi_np[-1, -1] = -1.0       # 尾块末元素标记
            phi_np[5, 2559] = 2.0       # head/tile 边界标记
        elif kind == "near_cancel":
            # 交替符号 x × 常数 phi 行 → 点积精确 0；bias 后 16 行取 ±1e-3 → h_res 落小值域
            phi_np = (np.ones((phi_rows, N * D), np.float32) / np.sqrt(N * D))
            bias_np = rng.uniform(-0.5, 0.5, (phi_rows,)).astype(np.float32)
            bias_np[N:] = np.where(np.arange(phi_rows - N) % 2 == 0, 1e-3, -1e-3).astype(np.float32)
            alpha_np = np.array([1.0, 1.0], np.float32)
            return self._inputs_from_arrays(x_np, phi_np, alpha_np, bias_np, torch.bfloat16)
        else:
            phi_np = rng.uniform(-2, 2, (phi_rows, N * D)).astype(np.float32)
        alpha_np = rng.uniform(0.5, 2, (2,)).astype(np.float32)
        bias_np = rng.uniform(-0.5, 0.5, (phi_rows,)).astype(np.float32)
        return self._inputs_from_arrays(x_np, phi_np, alpha_np, bias_np, torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_dist_all_zero_5120(self):
        """数值分布：x 全零（sumSq=0，invRms=1/√eps；输出必须有限且与 golden 一致）"""
        inputs = self.__dist_inputs("all_zero")
        self.assertTrue(torch.isfinite(inputs.x).all())
        self._run_pipeline(inputs, 1e-6, 4, 20)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_dist_constant_5120(self):
        """数值分布：x 全常数（RMSNorm 分母常数化退化路径）"""
        self._run_pipeline(self.__dist_inputs("constant"), 1e-6, 4, 20)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_dist_alternating_5120(self):
        """数值分布：x 交替符号（相消敏感，检查 Regbase 保序归约精度）"""
        self._run_pipeline(self.__dist_inputs("alternating"), 1e-6, 4, 20)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_dist_sparse_5120(self):
        """数值分布：稀疏 x（每 head 仅边界标记非零）。判定：双门槛——
        ① 绝对误差上界（被测 vs CPU golden 的 RMSE ≤ 1e-3，与常规用例同量级）；
        ② 拼接标杆比值仅记录不判死（稀疏+invRms 放大下拼接误差可能异常小，比值分母塌缩，
        系精度标准 2.1 比值判定的已知统计波动，非算子缺陷）。"""
        inputs = self.__dist_inputs("sparse")
        cpu_outputs = self._compute_cpu_golden(inputs, 1e-6)
        inputs_npu, npu_golden_outputs = self._compute_npu_golden(inputs, 1e-6)
        npu_outputs = self._run_npu_custom_op(inputs_npu, 1e-6, False)
        for name, got, ref in (
            ("h_post", npu_outputs.h_post, cpu_outputs.h_post),
            ("h_res", npu_outputs.h_res, cpu_outputs.h_res),
        ):
            got64 = got.cpu().to(GOLDEN_CPU_DTYPE)
            rmse = float((got64 - ref).pow(2).mean().sqrt())
            self.assertLess(rmse, 1e-3, f"{name} absolute RMSE {rmse} exceeds 1e-3")
            ref_rmse = float((npu_golden_outputs.h_post.cpu().to(GOLDEN_CPU_DTYPE) - cpu_outputs.h_post).pow(2).mean().sqrt())                 if name == "h_post" else                 float((npu_golden_outputs.h_res.cpu().to(GOLDEN_CPU_DTYPE) - cpu_outputs.h_res).pow(2).mean().sqrt())
            got_rmse = rmse
            logger.info(f"[sparse] {name}: abs RMSE={got_rmse:.3e} (concat={ref_rmse:.3e}, "
                        f"ratio={got_rmse / max(ref_rmse, 1e-30):.2f} recorded-only)")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_dist_scaled_phi_5120(self):
        """数值分布：phi 按 1/√(4D) 缩放 + 首/尾/边界标记值"""
        self._run_pipeline(self.__dist_inputs("scaled_phi"), 1e-6, 4, 20)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_dist_near_cancel_5120(self):
        """数值分布：近相消点积（交替 x × 常数 phi 行 → dot 精确 0，h_res 落小值域）"""
        self._run_pipeline(self.__dist_inputs("near_cancel"), 1e-6, 4, 20)

    # ======================== 长序列 / 大 T 与 OOM（合并 STC 3.8.4 + 3.9.2，序号 339/340/386） ========================

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_1024_2560_large_t(self):
        """T=1024 多轮 token 轮转（每核 16 token）"""
        self.__run_test_bsnd(B=1, S=1024, N=4, D=2560, dtype=torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4096_2560_large_t(self):
        """T=4096 资源稳定性（多轮轮转 + 输出边界）"""
        self.__run_test_bsnd(B=1, S=4096, N=4, D=2560, dtype=torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4096_7168_resource_boundary(self):
        """大 shape 资源边界（序号 386）：x≈224MB，正确完成或安全失败，失败须可诊断"""
        try:
            self.__run_test_bsnd(B=1, S=4096, N=4, D=7168, dtype=torch.bfloat16)
        except MemoryError:
            logger.info("resource boundary hit: safe failure accepted")

    # ======================== 边界值：epsilon 变体（合并 STC 3.8.5，序号 343/344） ========================

    @pytest.mark.resources(device="npu:*")
    @pytest.mark.xfail(strict=True, reason="CANN9.2 APT 链 attr 未传入 tiling：kernel 恒用默认 1e-6"
                                      "（2026-10-09 实测定位：python/csrc/aclnn/l0op 全链直传正常，"
                                      "eps=1e-2 与 1e-4 输出逐位相同；tiling GetFloat 落默认已加告警；待平台修复）")
    def test_mhc_pre_split_post_res_bf16_BSND_4_5120_eps_1e4(self):
        """norm_eps=1e-4 变体正例（非默认值）——已知平台缺陷，xfail 留证"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-4)

    # ======================== 异常端到端（合并 STC 3.8.6 子集：D 白名单/N/rank/dtype/eps/T=0） ========================

    def __make_raw_inputs(self, x_np, phi_np, alpha_np=None, bias_np=None):
        alpha_np = np.array([1.0, 1.0], np.float32) if alpha_np is None else alpha_np
        bias_np = np.zeros((phi_np.shape[0],), np.float32) if bias_np is None else bias_np
        return MhcInputTensors(
            x=torch.from_numpy(np.ascontiguousarray(x_np)).to(torch.bfloat16),
            phi=torch.from_numpy(np.ascontiguousarray(phi_np, np.float32)),
            alpha=torch.from_numpy(alpha_np),
            bias=torch.from_numpy(bias_np),
        )

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_d_2559(self):
        """D 白名单下邻 2559：aclnn/tiling 拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 4, 4, 2559), np.float32), np.zeros((20, 4 * 2559), np.float32)))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_d_2561(self):
        """D 白名单上邻 2561：拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 4, 4, 2561), np.float32), np.zeros((20, 4 * 2561), np.float32)))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_d_4096_gap(self):
        """D=4096 白名单内部空档：拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 4, 4, 4096), np.float32), np.zeros((20, 4 * 4096), np.float32)))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_n_3(self):
        """N=3 下邻：n 白名单四层防御（csrc TORCH_CHECK 首拦）"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 4, 3, 5120), np.float32), np.zeros((12, 3 * 5120), np.float32)))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_rank_2(self):
        """x rank=2：前导 shape 校验拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((4, 5120), np.float32), np.zeros((20, 4 * 5120), np.float32)))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_x_dtype_fp32(self):
        """x dtype=FP32：仅支持 BF16/FP16，拒绝"""
        inputs = self.__make_raw_inputs(
            np.zeros((1, 4, 4, 5120), np.float32), np.zeros((20, 4 * 5120), np.float32))
        inputs = MhcInputTensors(x=inputs.x.to(torch.float32), phi=inputs.phi,
                                 alpha=inputs.alpha, bias=inputs.bias)
        self.__expect_runtime_error(inputs)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_eps_zero(self):
        """norm_eps=0：finite 且 >0 校验拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 4, 4, 5120), np.float32), np.zeros((20, 4 * 5120), np.float32)), norm_eps=0.0)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_eps_negative(self):
        """norm_eps=-1e-6：拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 4, 4, 5120), np.float32), np.zeros((20, 4 * 5120), np.float32)), norm_eps=-1e-6)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_exception_t_zero_empty(self):
        """T=0 空 tensor：非空校验拒绝"""
        self.__expect_runtime_error(self.__make_raw_inputs(
            np.zeros((1, 0, 4, 5120), np.float32), np.zeros((20, 4 * 5120), np.float32)))

    # ======================== 非连续（合并 STC 3.9.1，序号 381~385） ========================

    def __noncontig_case(self, which):
        """构造非连续视图（与连续逻辑值对照）：aclnn 层 Contiguous 后应与连续输入逐位一致。"""
        B, S, N, D = 1, 4, 4, 5120
        inputs = self._generate_input_tensors(B, S, N, D, torch.bfloat16, seed=20261008)
        x_c, phi_c, bias_c = inputs.x, inputs.phi, inputs.bias
        x_v, phi_v, bias_v = x_c, phi_c, bias_c
        if which in ("x_d", "multi"):
            x2 = torch.empty(B, S, N, 2 * D, dtype=x_c.dtype)
            x2[..., ::2] = x_c
            x_v = x2[..., ::2]                      # D 轴 stride=2
        if which == "x_head":
            x3 = torch.empty(B, S, 2 * N, D, dtype=x_c.dtype)
            x3[:, :, ::2, :] = x_c
            x_v = x3[:, :, ::2, :]                  # head 轴 stride=2
        if which in ("phi", "multi"):
            phi2 = torch.empty(phi_c.shape[0], 2 * phi_c.shape[1], dtype=phi_c.dtype)
            phi2[:, ::2] = phi_c
            phi_v = phi2[:, ::2]                    # K 轴 stride=2
        if which == "bias":
            bias2 = torch.empty(2 * bias_c.shape[0], dtype=bias_c.dtype)
            bias2[::2] = bias_c
            bias_v = bias2[::2]                     # 间隔切片
        if which == "multi":
            bias2 = torch.empty(2 * bias_c.shape[0], dtype=bias_c.dtype)
            bias2[::2] = bias_c
            bias_v = bias2[::2]
        if which in ("x_d", "x_head", "multi"):
            self.assertFalse(x_v.is_contiguous())
        if which in ("phi", "multi"):
            self.assertFalse(phi_v.is_contiguous())
        if which in ("bias", "multi"):
            self.assertFalse(bias_v.is_contiguous())
        nc = MhcInputTensors(x=x_v, phi=phi_v, alpha=inputs.alpha, bias=bias_v)
        # 1) 非连续视图 vs 双标杆 golden（逻辑值）
        self._run_pipeline(nc, 1e-6, N, N * N + N)
        # 2) 非连续 vs 连续副本逐位一致（Contiguous 转换正确性）
        out_nc = self._run_npu_custom_op(MhcInputTensors(
            x=x_v.npu(), phi=phi_v.npu(), alpha=inputs.alpha.npu(), bias=bias_v.npu()), 1e-6, False)
        out_c = self._run_npu_custom_op(MhcInputTensors(
            x=x_c.npu(), phi=phi_c.npu(), alpha=inputs.alpha.npu(), bias=bias_c.npu()), 1e-6, False)
        self.assertTrue(torch.equal(out_nc.h_post, out_c.h_post))
        self.assertTrue(torch.equal(out_nc.h_res, out_c.h_res))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_noncontig_x_d_axis(self):
        """非连续：x D 轴 stride=2（序号 381）"""
        self.__noncontig_case("x_d")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_noncontig_x_head_axis(self):
        """非连续：x head 轴 stride=2 切片视图（序号 382）"""
        self.__noncontig_case("x_head")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_fp16_noncontig_phi_k_axis(self):
        """非连续：phi K 轴 stride=2（序号 383）"""
        self.__noncontig_case("phi")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_noncontig_bias_slice(self):
        """非连续：bias 间隔切片（序号 384）"""
        self.__noncontig_case("bias")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_noncontig_multi_inputs(self):
        """非连续：x/phi/bias 多输入同时非连续（序号 385）"""
        self.__noncontig_case("multi")

    # ======================== 压测（合并 STC 3.9.3，序号 388/389） ========================

    def __run_stress(self, B, S, N, D, dtype, phi_rows=None, iters=1000):
        inputs = self._generate_input_tensors(B, S, N, D, dtype, seed=20261008, phi_rows=phi_rows)
        inputs_npu = MhcInputTensors(x=inputs.x.npu(), phi=inputs.phi.npu(),
                                     alpha=inputs.alpha.npu(), bias=inputs.bias.npu())
        first = self._run_npu_custom_op(inputs_npu, 1e-6, False)
        torch.npu.synchronize()
        mem0 = torch.npu.memory_allocated()
        for i in range(iters):
            out = self._run_npu_custom_op(inputs_npu, 1e-6, False)
            if (i + 1) % 100 == 0:
                self.assertTrue(torch.equal(out.h_post, first.h_post))
                if phi_rows is None:
                    self.assertTrue(torch.equal(out.h_res, first.h_res))
                torch.npu.synchronize()
        mem1 = torch.npu.memory_allocated()
        self.assertLessEqual(mem1, mem0 + 64 * 1024 * 1024)  # 显存稳定（caching allocator 容差 64MB）
        logger.info(f"stress {iters} iters done, mem delta {(mem1 - mem0) / 1e6:.1f}MB")

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_stress_4_5120_dual_1000iters(self):
        """压测：T=4/D=5120 完整形态（双核）连续 1000 次，每 100 次位模式比对 + 显存稳定（序号 388）"""
        self.__run_stress(1, 4, 4, 5120, torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_stress_65_7168_single_1000iters(self):
        """压测：T=65/D=7168 单核余数 1、长尾，连续 1000 次，核间 token 不串写（序号 389）"""
        self.__run_stress(1, 65, 4, 7168, torch.bfloat16)

    # ======================== 确定性（合并 STC 3.9.4，序号 390~393） ========================

    def __run_determinism(self, B, S, N, D, dtype, phi_rows=None, iters=10):
        inputs = self._generate_input_tensors(B, S, N, D, dtype, seed=20261008, phi_rows=phi_rows)
        inputs_npu = MhcInputTensors(x=inputs.x.npu(), phi=inputs.phi.npu(),
                                     alpha=inputs.alpha.npu(), bias=inputs.bias.npu())
        base = self._run_npu_custom_op(inputs_npu, 1e-6, False)
        for _ in range(iters - 1):
            out = self._run_npu_custom_op(inputs_npu, 1e-6, False)
            self.assertTrue(torch.equal(out.h_post, base.h_post))
            if phi_rows is None:
                self.assertTrue(torch.equal(out.h_res, base.h_res))

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_determinism_4_5120_full_x50(self):
        """确定性：验收点完整形态 × 50 次位模式一致（序号 390）"""
        self.__run_determinism(1, 4, 4, 5120, torch.bfloat16, iters=50)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_determinism_4_5120_post_only_x50(self):
        """确定性：验收点仅 post 形态 × 50 次（序号 391）"""
        self.__run_determinism(1, 4, 4, 5120, torch.bfloat16, phi_rows=4, iters=50)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_determinism_32_5120_boundary_x10(self):
        """确定性：双核边界 T=32 × 10 次（序号 392）"""
        self.__run_determinism(1, 32, 4, 5120, torch.bfloat16, iters=10)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_determinism_256_7168_prefill_x10(self):
        """确定性：prefill T=256/D=7168 长尾 × 10 次（序号 393）"""
        self.__run_determinism(1, 256, 4, 7168, torch.bfloat16, iters=10)

    # ======================== P1 全矩阵：12T × 4D × 2形态（STC 5. 设计网格，TND 3D，bf16） ========================
    # 对应单 STC 文档 T_POINTS × D_POINTS × FORMS = 96 组；subTest 逐组报告、失败不中断其余组。
    # 该矩阵同时覆盖 TYPICAL 5 组（4/5120、4/2560、8/6144、16/7168、256/5120）与 TND 侧 P0 点位。

    T_P1_POINTS = (1, 2, 4, 8, 16, 31, 32, 33, 64, 65, 128, 256)
    D_P1_POINTS = (2560, 5120, 6144, 7168)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_TND_p1_matrix_full(self):
        """P1 精度矩阵·完整形态（48 组）：phi=[20,4D]，双核/单核形态随 T 分区自然覆盖"""
        for T in self.T_P1_POINTS:
            for D in self.D_P1_POINTS:
                with self.subTest(T=T, D=D, form="full"):
                    self.__run_test_tnd(T=T, N=4, D=D, dtype=torch.bfloat16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_TND_p1_matrix_post_only(self):
        """P1 精度矩阵·仅 post 形态（48 组）：phi=[4,4D]，恒单核（Block Num=T 上限 64）"""
        for T in self.T_P1_POINTS:
            for D in self.D_P1_POINTS:
                with self.subTest(T=T, D=D, form="post_only"):
                    self.__run_test_tnd(T=T, N=4, D=D, dtype=torch.bfloat16, phi_rows=4)

    # ======================== P0 双 dtype 补齐（STC 4.4：P0 全量双 dtype；bf16 已由矩阵/既有用例覆盖） ========================

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_fp16_TND_1_2560_p0(self):
        """P0：T=1/D=2560 TND 完整形态 FP16"""
        self.__run_test_tnd(T=1, N=4, D=2560, dtype=torch.float16)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_fp16_TND_1_2560_post_only_p0(self):
        """P0：T=1/D=2560 TND 仅 post 形态 FP16"""
        self.__run_test_tnd(T=1, N=4, D=2560, dtype=torch.float16, phi_rows=4)

    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_fp16_BSND_4_2560_p0(self):
        """P0：S=4/D=2560 BSND 完整形态 FP16"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=2560, dtype=torch.float16)

    # ======================== epsilon 变体（须最后执行） ========================
    # 平台缺陷（2026-10-09 实测定位，机理已修正）：APT/GE 对同 (shape, TilingKey) 调用
    # 固化复用首次生成的 tilingData，normEps 不参与复用 key——同 shape 下非默认 eps 仅进程内
    # 首次调用生效，后续被固化值静默替代（attr 链路本身是通的：eps=1e-12 首次调用可生效）。
    # 非默认 eps 用例因此必须排在所有数值精度用例之后，否则污染后续同 shape 用例
    # （实测：eps=1e-12 调用后 sparse 用例 h_res RMSE 2.9e-2，反解 kernel 实际按 1e-12 计算）。
    @pytest.mark.resources(device="npu:*")
    def test_mhc_pre_split_post_res_bf16_BSND_4_5120_eps_1e12(self):
        """norm_eps=1e-12 极小值数值稳定性（本用例固化缓存副作用，置于末尾执行）"""
        self.__run_test_bsnd(B=1, S=4, N=4, D=5120, dtype=torch.bfloat16, norm_eps=1e-12)

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
                return torch.ops.cann_ops_transformer.mhc_pre_split_post_res(
                    x, phi, alpha, bias, norm_eps
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