import torch
import torch_npu
import omni_training_custom_ops
import numpy as np
import random
import itertools

from torch_npu.testing.testcase import TestCase, run_tests
import pytest


# golden函数
def generate_tensor_with_distribution(shape, dtype, dist_type, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    if dist_type == 'uniform':
        # 均匀分布 [-5, 5]
        tensor = torch.rand(shape) * 10 - 5  # [0,1) -> [-5, 5)
    elif dist_type == 'normal':
        # 正态分布 able μ∈[-100, 100], σ∈[1, 25]
        mu = torch.rand(1).item() * 200 - 100  # [-100, 100]
        sigma = torch.rand(1).item() * 24 + 1   # [1, 25]
        tensor = torch.randn(shape) * sigma + mu
    elif dist_type == 'outlier':
        # 离群点分布：基于正态分布，随机0.1%的值乘以1000
        tensor = torch.randn(shape)
        # 生成离群点mask (0.1% 的元素)
        outlier_mask = torch.rand(shape) < 0.001
        tensor[outlier_mask] *= 1000
    elif dist_type == 'double_stochastic':
        # 双随机矩阵：n x n 的正矩阵，每行或每列的和都是1，对后两维约束，不满足时退化到标准正态分布
        if shape[-2] != shape[-1]:
            tensor = torch.randn(shape)
        else:
            tensor = torch.randn(shape).exp()
            tensor = sinkhorn_manual(tensor, eps=1e-6, num_iters=20)
    else:
        # 默认标准正态分布
        tensor = torch.randn(shape)

    return tensor.to(dtype)


def sinkhorn_manual(tensor, eps=1e-6, num_iters=20):
    tensor = tensor + eps
    for _ in range(num_iters):
        # 行归一化
        row_sum = tensor.sum(dim=-1, keepdim=True)
        tensor = tensor / (row_sum + eps)
        # 列归一化
        col_sum = tensor.sum(dim=-2, keepdim=True)
        tensor = tensor / (col_sum + eps)
    return tensor


def get_small_value_thresh(dtype_in):
    if dtype_in == torch.float16:
        return 2**-11  # ~4.88e-4
    elif dtype_in == torch.bfloat16:
        return 2**-8   # ~3.91e-3
    else:  # float32
        return 2**-14  # ~6.10e-5


def get_error_thd(dtype_in):
    if dtype_in == torch.float16:
        return 2**-11
    elif dtype_in == torch.bfloat16:
        return 2**-8
    else:  # float32
        return 2**-14


def real_ratio(x, y, error_thd, default_value=1.0):
    if y == 0:
        if x == 0:
            return 1.0
        else:
            return default_value
    else:
        return float(x / max(y, error_thd))


def calc_metrics_with_mask(actual, golden, small_value_thresh):
    # 值域分区: 小值域 vs 非小值域
    mask = torch.abs(golden) > small_value_thresh

    # 非小值域数据
    actual_large = actual[mask]
    golden_large = golden[mask]

    # 计算 inf/nan mask
    inf_nan_mask = ~torch.isfinite(actual_large) | ~torch.isfinite(golden_large)

    if inf_nan_mask.all() or golden_large.numel() == 0:
        # 全是 inf/nan 或没有非小值域数据
        return 0.0, 0.0, 0.0

    # compute_relative_error: diff / (|golden| + 1e-7)
    abs_diff = torch.abs(actual_large - golden_large)
    re = abs_diff / (torch.abs(golden_large) + 1e-7)

    # 排除 inf/nan 后计算 MARE/MERE
    valid_re = re[~inf_nan_mask]
    if valid_re.numel() == 0:
        mare = 0.0
        mere = 0.0
    else:
        mare = valid_re.max().item()
        mere = valid_re.mean().item()

    # compute_root_mean_squared_error: sqrt(sum((a-g)^2) / numel)
    diff = actual_large[~inf_nan_mask] - golden_large[~inf_nan_mask]
    if diff.numel() == 0:
        rmse = 0.0
    else:
        rmse = torch.sqrt(torch.sum(diff * diff) / diff.numel()).item()

    return mare, mere, rmse


def golden_impl_high_precision(grad_output, x, h_res, h_out, h_post):
    high_dtype = torch.float64

    # 转换到 fp64 并启用梯度追踪 (在 CPU 上)
    h_out_hp = h_out.to(high_dtype).requires_grad_(True)
    x_hp = x.to(high_dtype).requires_grad_(True)
    h_post_hp = h_post.to(high_dtype).requires_grad_(True)
    h_res_hp = h_res.to(high_dtype).requires_grad_(True)
    grad_output_hp = grad_output.to(high_dtype)

    # Forward pass
    # Y[i] = h_post[i] * h_out + h_res[i,:].T @ x
    y = hc_post_forward(h_out_hp, x_hp, h_post_hp, h_res_hp)

    # Backward pass using autograd
    y.backward(grad_output_hp)

    # 返回梯度 (保持 fp64 精度)
    return x_hp.grad, h_res_hp.grad, h_out_hp.grad, h_post_hp.grad


def hc_post_forward(h_out, x, h_post, h_res):
    # h_post.unsqueeze(-1) * h_out.unsqueeze(-2): [B,S,n,1] * [B,S,1,D] -> [B,S,n,D]
    y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + torch.sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)
    return y


def npu_small_ops_bench_impl(grad_output_npu, x_npu, h_res_npu, h_out_npu, h_post_npu):
    dtype = grad_output_npu.dtype

    # Convert to f32 and enable gradients (on NPU)
    h_out_f32 = h_out_npu.float().requires_grad_(True)
    x_f32 = x_npu.float().requires_grad_(True)
    h_post_f32 = h_post_npu.float().requires_grad_(True)
    h_res_f32 = h_res_npu.float().requires_grad_(True)

    # Forward pass (works for both BSND and TND due to broadcasting)
    y = hc_post_forward(h_out_f32, x_f32, h_post_f32, h_res_f32)

    # Backward pass using autograd
    grad_output_f32 = grad_output_npu.float()
    y.backward(grad_output_f32)

    # Get gradients and convert back to original dtype
    # NOTE: Convert back to original dtype to simulate GPU behavior where
    # intermediate computation is fp32 but output is stored in original dtype
    grad_h_out = h_out_f32.grad.to(dtype)
    grad_x = x_f32.grad.to(dtype)
    grad_h_post = h_post_f32.grad  # keep fp32 (output dtype)
    grad_h_res = h_res_f32.grad  # keep fp32 (output dtype)

    return grad_x, grad_h_res, grad_h_out, grad_h_post


def npu_forward_impl(x_npu, h_res_npu, h_out_npu, h_post_npu):
    """
    NPU 小算子拼接实现 (Bench)

    支持 BSND (4D) 和 TND (3D) 两种格式:
    - BSND: x (B, S, n, D), h_res (B, S, n, n), h_out (B, S, D), h_post (B, S, n)
    - TND:  x (T, n, D),    h_res (T, n, n),    h_out (T, D),    h_post (T, n)

    公式说明:
    =========
    Forward: Y[i] = h_post[i] * h_out + h_res[i,:].T @ x
             即: Y = h_post.unsqueeze(-1) * h_out.unsqueeze(-2) + h_res.T @ x

    Args:
        x_npu: Input tensor, shape (B, S, n, D) or (T, n, D)
        h_res_npu: Combination weights, shape (B, S, n, n) or (T, n, n), float32
        h_out_npu: Output tensor, shape (B, S, D) or (T, D)
        h_post_npu: Post weights, shape (B, S, n) or (T, n), float32

    Returns:
        output (on NPU)
    """
    dtype = x_npu.dtype

    # Convert to f32 for computation
    h_out_f32 = h_out_npu.float()
    x_f32 = x_npu.float()
    h_post_f32 = h_post_npu.float()
    h_res_f32 = h_res_npu.float()

    # Forward pass (works for both BSND and TND due to broadcasting)
    y = hc_post_forward(h_out_f32, x_f32, h_post_f32, h_res_f32)

    # Convert back to original dtype
    output = y.to(dtype)

    return output


def compare_with_dual_benchmark(name, npu_val, bench_val, golden_val, input_dtype, verbose=True):
    # 提升一档精度作为 golden_dtype
    golden_dtype = torch.float64

    # 统一对齐到 golden_dtype 进行 diff 计算
    npu_val_for_diff = npu_val.to(golden_dtype)
    bench_val_for_diff = bench_val.to(golden_dtype)
    golden_val_for_diff = golden_val.to(golden_dtype)

    # 获取小值域阈值
    small_value_thresh = get_small_value_thresh(input_dtype)

    # 计算 MARE, MERE, RMSE (排除小值域)
    npu_mare, npu_mere, npu_rmse = calc_metrics_with_mask(npu_val_for_diff, golden_val_for_diff, small_value_thresh)
    bench_mare, bench_mere, bench_rmse = calc_metrics_with_mask(bench_val_for_diff, golden_val_for_diff, small_value_thresh)

    # 计算 Ratio (x / max(y, error_thd))
    error_thd = get_error_thd(input_dtype)
    ratio_mare = real_ratio(npu_mare, bench_mare, error_thd)
    ratio_mere = real_ratio(npu_mere, bench_mere, error_thd)
    ratio_rmse = real_ratio(npu_rmse, bench_rmse, error_thd)

    # 判定阈值 (L0 精度等级)
    pass_mare = ratio_mare <= 10.0
    pass_mere = ratio_mere <= 2.0
    pass_rmse = ratio_rmse <= 2.0

    # 判定结果
    passed = pass_mare and pass_mere and pass_rmse
    status = "PASS" if passed else "FAIL"

    # 失败时打印详细 Tensor 信息
    if not passed and verbose:
        # 计算绝对误差
        npu_diff = (npu_val_for_diff - golden_val_for_diff).abs()
        bench_diff = (bench_val_for_diff - golden_val_for_diff).abs()

        # 计算相对误差 (MARE 的来源)
        golden_abs = golden_val_for_diff.abs() + 1e-7
        npu_re = npu_diff / golden_abs
        bench_re = bench_diff / golden_abs

        # 1. MARE - 最大相对误差位置
        npu_max_re_idx = npu_re.argmax().item()
        bench_max_re_idx = bench_re.argmax().item()
        npu_max_re_pos = np.unravel_index(npu_max_re_idx, npu_val.shape)
        bench_max_re_pos = np.unravel_index(bench_max_re_idx, bench_val.shape)

    # 返回详细指标用于失败分析
    metrics = {
        "npu_mare": npu_mare, "npu_mere": npu_mere, "npu_rmse": npu_rmse,
        "bench_mare": bench_mare, "bench_mere": bench_mere, "bench_rmse": bench_rmse,
        "ratio_mare": ratio_mare, "ratio_mere": ratio_mere, "ratio_rmse": ratio_rmse,
        "pass_mare": pass_mare, "pass_mere": pass_mere, "pass_rmse": pass_rmse
    }

    return passed, metrics


def as_strided(tensor, pad_factor=2):
    dim = len(tensor.shape)
    for i in range(0, dim):
        tensor = tensor.repeat_interleave(pad_factor, dim=i)
    slice_indices = tuple([slice(None, None, pad_factor)] * dim)
    tensor = tensor[slice_indices]
    return tensor


class TestCustomMHCPostGrad(TestCase):
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_mhc_post_grad_netshapes(self):
        net_shapes = [[1, 1024, 4, 5120], [1, 2048, 4, 2560], [1, 4096, 4, 2560]]
        for shape in net_shapes:
            x_cpu, h_res_cpu, h_out_cpu, h_post_cpu = self.gen_tensor_bsnd(shape, torch.bfloat16)
            self.impl_test_mhc_post_grad(x_cpu, h_res_cpu, h_out_cpu, h_post_cpu)
            x_cpu, h_res_cpu, h_out_cpu, h_post_cpu = self.gen_tensor_tnd(shape, torch.bfloat16)
            self.impl_test_mhc_post_grad(x_cpu, h_res_cpu, h_out_cpu, h_post_cpu)
            
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_mhc_post_grad_general_shapes(self):
        gen_shapes = [[1, 1024, 4, 5120], [1, 2048, 4, 2560], [1, 4096, 4, 2560]]
        dtypes = [torch.float16, torch.bfloat16]
        dist_types = ["uniform", "normal", "double_stochastic"]
        for shape, dtype, dist_type in itertools.product(gen_shapes, dtypes, dist_types):
            x_cpu, h_res_cpu, h_out_cpu, h_post_cpu = self.gen_tensor_bsnd(shape, dtype, dist_type=dist_type)
            self.impl_test_mhc_post_grad(x_cpu, h_res_cpu, h_out_cpu, h_post_cpu)

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_mhc_post_grad_noncontiguous_tensor(self):
        net_shapes = [[1, 1024, 4, 5120], [1, 2048, 4, 2560], [1, 4096, 4, 2560]]
        for shape in net_shapes:
            x_cpu, h_res_cpu, h_out_cpu, h_post_cpu = self.gen_tensor_bsnd(shape, torch.bfloat16)
            self.impl_test_mhc_post_grad(x_cpu, h_res_cpu, h_out_cpu, h_post_cpu, npu_as_strided=True)
            x_cpu, h_res_cpu, h_out_cpu, h_post_cpu = self.gen_tensor_tnd(shape, torch.bfloat16)
            self.impl_test_mhc_post_grad(x_cpu, h_res_cpu, h_out_cpu, h_post_cpu)

    def gen_tensor_tnd(self, shape, dtype, dist_type="uniform", seed=123):
        B = shape[0]
        S = shape[1]
        T = B * S
        n = shape[2]
        D = shape[3]
        
        x_cpu = generate_tensor_with_distribution((T, n, D), dtype, dist_type, seed=seed)
        # hres 分布需满足双随机矩阵约束
        h_res_cpu = generate_tensor_with_distribution((T, n, n), torch.float32, dist_type="double_stochastic", seed=seed)
        h_out_cpu = generate_tensor_with_distribution((T, D), dtype, dist_type, seed=seed)
        h_post_cpu = generate_tensor_with_distribution((T, n), torch.float32, dist_type, seed=seed)

        return x_cpu, h_res_cpu, h_out_cpu, h_post_cpu

    def gen_tensor_bsnd(self, shape, dtype, dist_type="uniform", seed=123):
        B = shape[0]
        S = shape[1]
        n = shape[2]
        D = shape[3]
        
        x_cpu = generate_tensor_with_distribution((B, S, n, D), dtype, dist_type, seed=seed)
        # hres 分布需满足双随机矩阵约束
        h_res_cpu = generate_tensor_with_distribution((B, S, n, n), torch.float32, dist_type="double_stochastic", seed=seed)
        h_out_cpu = generate_tensor_with_distribution((B, S, D), dtype, dist_type, seed=seed)
        h_post_cpu = generate_tensor_with_distribution((B, S, n), torch.float32, dist_type, seed=seed)

        return x_cpu, h_res_cpu, h_out_cpu, h_post_cpu

    def impl_test_mhc_post_grad(self, x_cpu, h_res_cpu, h_out_cpu, h_post_cpu, npu_as_strided=False):
        dtype = x_cpu.dtype

        x_npu = x_cpu.npu()
        h_res_npu = h_res_cpu.npu()
        h_out_npu = h_out_cpu.npu()
        h_post_npu = h_post_cpu.npu()

        if npu_as_strided:
            x_npu = as_strided(x_npu)
            h_res_npu = as_strided(h_res_npu)
            h_out_npu = as_strided(h_out_npu)
            h_post_npu = as_strided(h_post_npu)

        # 前向生成grad_output
        grad_output_npu = npu_forward_impl(x_npu, h_res_npu, h_out_npu, h_post_npu)
        grad_output_cpu = grad_output_npu.cpu()

        # Golden implementation
        grad_x_golden, grad_h_res_golden, grad_h_out_golden, grad_h_post_golden = golden_impl_high_precision(
            grad_output_cpu, x_cpu, h_res_cpu, h_out_cpu, h_post_cpu
        )

        # Bench implementation
        grad_x_bench, grad_h_res_bench, grad_h_out_bench, grad_h_post_bench = npu_small_ops_bench_impl(
            grad_output_npu, x_npu, h_res_npu, h_out_npu, h_post_npu
        )

        # Move bench results to CPU
        grad_x_bench_cpu = grad_x_bench.cpu()
        grad_h_res_bench_cpu = grad_h_res_bench.cpu()
        grad_h_out_bench_cpu = grad_h_out_bench.cpu()
        grad_h_post_bench_cpu = grad_h_post_bench.cpu()

        # Custom kernel
        grad_x_npu, grad_h_res_npu, grad_h_out_npu, grad_h_post_npu = torch.ops.custom.npu_ai_infra_mhc_post_grad(
            grad_output_npu, x_npu, h_res_npu, h_out_npu, h_post_npu
        )

        # Move NPU results to CPU
        grad_x_npu_cpu = grad_x_npu.cpu()
        grad_h_res_npu_cpu = grad_h_res_npu.cpu()
        grad_h_out_npu_cpu = grad_h_out_npu.cpu()
        grad_h_post_npu_cpu = grad_h_post_npu.cpu()
        
        # compare
        results = []
        results.append(compare_with_dual_benchmark(
            "grad_x", grad_x_npu_cpu.float(), grad_x_bench_cpu.float(), grad_x_golden, dtype))
        results.append(compare_with_dual_benchmark(
            "grad_h_res", grad_h_res_npu_cpu.float(), grad_h_res_bench_cpu.float(), grad_h_res_golden, torch.float32))
        results.append(compare_with_dual_benchmark(
            "grad_h_out", grad_h_out_npu_cpu.float(), grad_h_out_bench_cpu.float(), grad_h_out_golden, dtype))
        results.append(compare_with_dual_benchmark(
            "grad_h_post", grad_h_post_npu_cpu.float(), grad_h_post_bench_cpu.float(), grad_h_post_golden, torch.float32))
        print(results)
        each_passed = [passed for passed, _ in results]
        all_passed = all(each_passed)
        assert all_passed == True
