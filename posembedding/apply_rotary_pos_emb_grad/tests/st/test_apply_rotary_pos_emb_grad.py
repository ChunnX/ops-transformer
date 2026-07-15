# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import omni_training_custom_ops
from torch_npu.testing.testcase import TestCase, run_tests
import pytest
torch_npu.npu.config.allow_internal_format = True


EPS: float = 1e-7
GOLDEN_CPU_DTYPE = torch.float64
DEFAULT_PRECISION_LEVEL = "L2"

precision_levels_config = {
    "L0": {"MARE": 10.0, "MERE": 2.0, "RMSE": 2.0},
    "L1": {"MARE": 5.0, "MERE": 1.5, "RMSE": 1.5},
    "L2": {"MARE": 2.0, "MERE": 1.2, "RMSE": 1.2},
}

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

    ratio = {k: npu_metrics[k] / (gpu_metrics[k] + 1e-7) for k in npu_metrics}

    limits = precision_levels_config[precision_level]

    print(f"[NormalDomain] ratio: {ratio}")
    print(f"[NormalDomain] required level: {precision_level}")

    for k, limit in limits.items():
        if ratio[k] > limit:
            print(f"[NormalDomain] FAIL on {k}: {ratio[k]} > {limit}")
            return False

    print("[NormalDomain] PASS")
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

    print(
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

    print(f"[Final] normal_pass={normal_ok}, small_pass={small_ok}")
    return normal_ok and small_ok


# ---------- RoPE 数学参考 (half 模式) ----------

# lastdim可以不一致情况
def single_golden(x, cos, sin):
    dtype = x.dtype
    if dtype == torch.bfloat16:
        x = x.float()
        cos = cos.float()
        sin = sin.float()

    d_rotary = cos.shape[-1]    # 获取cos维度大小
    if x.shape[-1] != d_rotary:
    # 分割x为前d_rotary部分和剩余部分
        x_front = x[..., :d_rotary]
        x_back = x[..., d_rotary:]

        # 处理前d_rotary部分
        split = d_rotary // 2
        x1 = x_front[..., :split]
        x2 = x_front[..., split:]
        x_new = torch.cat((-x2, x1), dim=-1)
        res_front = x_front * cos + x_new * sin

        # 拼接剩余部分
        res = torch.cat((res_front, x_back), dim=-1)
    else:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x_new = torch.cat((-x2, x1), dim=-1)
        res = x * cos + x_new * sin

    return res.to(dtype)

def rope_forward(q, k, cos, sin):
    q_res = single_golden(q, cos, sin)
    k_res = single_golden(k, cos, sin)
    return q_res, k_res


def apply_rotary_pos_emb_gpu(q, k, cos, sin, rotary_mode="half"):
    k_emb_r = torch_npu.npu_rotary_mul(k, cos, sin, rotary_mode=rotary_mode)
    q_emb_r = torch_npu.npu_rotary_mul(q, cos, sin, rotary_mode=rotary_mode)

    return q_emb_r, k_emb_r


GRAD_NAMES = ["dquery", "dkey", "d_cos", "d_sin"]
# ---------- test cases ----------

class TestNpuApplyRotaryPosEmbGrad(TestCase):
    def _run_rotary_test(self, s, b, qn, kn, d, dtype, layout=2, seed=42,
                         precision_level=DEFAULT_PRECISION_LEVEL, cal_dcos=True, autograd=False):
        torch.manual_seed(seed)
        print("======================================")
        # 按 layout 推导 q/k 与 cos/sin 的 shape
        # qn / kn 分别为 q 路 / k 路的 head 维; cos/sin 的 head 维恒为 1, 自动广播到 q/k 的 head 维
        if layout == 2:
            q_shape, k_shape, cos_shape = (s, b, qn, d), (s, b, kn, d), (s, b, 1, d)
        elif layout == 1:
            q_shape, k_shape, cos_shape = (b, s, qn, d), (b, s, kn, d), (b, s, 1, d)
        elif layout == 4:
            q_shape, k_shape, cos_shape = (s * b, qn, d), (s * b, kn, d), (s * b, 1, d)
        else:
            raise ValueError(f"unsupported layout: {layout}")

        # 0. CPU FP64 主数据 (三路同源, 保证对比公平)
        q_base = torch.randn(*q_shape, dtype=torch.float64)
        k_base = torch.randn(*k_shape, dtype=torch.float64)
        cos_base = torch.randn(*cos_shape, dtype=torch.float64)
        sin_base = torch.randn(*cos_shape, dtype=torch.float64)

        # =====================================================================
        # 1. NPU 参考 (gpu_golden): apply_rotary_pos_emb_v2 (npu_rotary_mul) 级联
        #    即 golden_2.py 指定的 golden —— 对 q / k 各调用一次。
        # =====================================================================
        q_r = q_base.to(dtype).npu()
        k_r = k_base.to(dtype).npu()
        cos_r = cos_base.to(dtype).npu()
        sin_r = sin_base.to(dtype).npu()
        q_r.requires_grad_(True)
        k_r.requires_grad_(True)
        cos_r.requires_grad_(cal_dcos)
        sin_r.requires_grad_(cal_dcos)

        q_emb_r, k_emb_r = apply_rotary_pos_emb_gpu(q_r, k_r, cos_r, sin_r, "half")
        q_emb_r.retain_grad()
        k_emb_r.retain_grad()
        loss_r = q_emb_r.sum() + k_emb_r.sum()
        loss_r.backward()
        gpu_golden = [q_r.grad, k_r.grad, cos_r.grad, sin_r.grad]

        grad_q_emb_n = q_emb_r.grad.detach().clone()
        grad_k_emb_n = k_emb_r.grad.detach().clone()
        # =====================================================================
        # 2. CPU FP64 golden (cpu_golden): 纯 torch rotate_half (升精度真值)
        # =====================================================================
        q_g = q_base.clone().requires_grad_(True)
        k_g = k_base.clone().requires_grad_(True)
        cos_g = cos_base.clone().requires_grad_(cal_dcos)
        sin_g = sin_base.clone().requires_grad_(cal_dcos)

        q_emb_g, k_emb_g = rope_forward(q_g, k_g, cos_g, sin_g)
        loss_g = q_emb_g.sum() + k_emb_g.sum()
        loss_g.backward()
        cpu_golden = [q_g.grad, k_g.grad, cos_g.grad, sin_g.grad]

        # =====================================================================
        # 3. 自定义算子 (npu_result): torch.ops.custom.npu_apply_rotary_pos_emb 级联
        # =====================================================================
        q_n = q_base.to(dtype).npu()
        k_n = k_base.to(dtype).npu()
        cos_n = cos_base.to(dtype).npu()
        sin_n = sin_base.to(dtype).npu()
        q_n.requires_grad_(True)
        k_n.requires_grad_(True)
        cos_n.requires_grad_(cal_dcos)
        sin_n.requires_grad_(cal_dcos)

        if autograd:
            q_emb, k_emb = torch.ops.custom.npu_apply_rotary_pos_emb(
                q_n, k_n, cos_n, sin_n, layout=layout, rotary_mode=0
            )
            loss_n = q_emb.sum() + k_emb.sum()
            loss_n.backward()
            torch.npu.synchronize()
            npu_result = [q_n.grad, k_n.grad, cos_n.grad, sin_n.grad]
        else:
            grad_q, grad_k, grad_cos, grad_sin = torch.ops.custom.npu_apply_rotary_pos_emb_backward(
                grad_q_emb_n, grad_k_emb_n, cos_n, sin_n, q_n, k_n, rotary_mode=0, layout=layout
            )
            torch.npu.synchronize()
            npu_result = [grad_q, grad_k, grad_cos, grad_sin]

        # =====================================================================
        # 4. 4 个梯度分别做三路 precision_check
        # =====================================================================
        shape = (tuple(q_base.shape), tuple(k_base.shape))
        results = []
        for name, g_cpu, g_gpu, g_npu in zip(GRAD_NAMES, cpu_golden, gpu_golden, npu_result):
            if g_cpu is not None:
                npu_out_fp64 = g_npu.detach().cpu().to(torch.float64)
                gpu_out_fp64 = g_gpu.detach().cpu().to(torch.float64)
                cpu_out_fp64 = g_cpu.detach()

                print(f"==== {name} ({dtype}, layout={layout}, level={precision_level}, q_shape={shape[0]}, k_shape={shape[1]}) ====")
                ok = precision_check(npu_out_fp64, gpu_out_fp64, cpu_out_fp64, dtype, precision_level)
                results.append((name, ok))

        failed = [name for name, ok in results if not ok]
        self.assertFalse(
            failed,
            f"precision check failed for grads: {failed} ({dtype}, layout={layout}, shape={shape})",
        )

    # ==========================================
    # 网络场景用例 (BF16 / SBND / Level0)
    #   qn / kn 分别为 q 路 / k 路的 head 维; cos/sin head 维 = 1 自动广播
    #   query / key 取前向输入 q_n / k_n (cos/sin requires_grad, 4 路全对比)
    # ==========================================
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_s1536_b5_qn8_kn1_d96_fp32(self):
        # 7B场景: GQA (qn=8, kn=1)
        self._run_rotary_test(1536, 5, 8, 1, 96, torch.float32, layout=1, precision_level="L2", cal_dcos=True, autograd=True)

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_s8192_b6_qn1_kn8_d112_fp32(self):
        # 7B场景: GQA (qn=8, kn=1)
        self._run_rotary_test(8192, 6, 1, 8, 112, torch.float32, layout=2, precision_level="L2", cal_dcos=False, autograd=True)

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_t4096_qn4_kn4_d48_fp16(self):
        # 7B场景: GQA (qn=8, kn=1)
        self._run_rotary_test(4096, 1, 4, 4, 48, torch.float16, layout=4, precision_level="L2", cal_dcos=True, autograd=True)

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_s10000_b7_qn1_kn4_d128_bf16(self):
        # 7B场景: MHA (qn=16, kn=16)
        self._run_rotary_test(10000, 7, 1, 4, 128, torch.bfloat16, layout=1, precision_level="L2", cal_dcos=False, autograd=True)


if __name__ == "__main__":
    run_tests()
