import torch
import torch_npu
import omni_training_custom_ops
import numpy as np
import random

from torch_npu.testing.testcase import TestCase, run_tests
import pytest

torch.manual_seed(1234)
torch.set_printoptions(threshold=float('inf'))  # 阈值设为无穷大，永不截断
torch.set_printoptions(sci_mode=False, precision=5) 


def softmax(x, dim=-1):
    """计算softmax"""
    return torch.softmax(x, dim=dim)


def sinkhorn_forward_simple(x, num_iters, eps):
    """
    Sinkhorn前向传播（简化版）

    参数:
        x: 输入张量 [T, n, n]
        num_iters: 迭代次数
        eps: 小常数防止除零

    返回:
        output: 输出 [T, n, n]
        norm_list: 归一化输出列表
        sum_list: 求和结果列表
    """
    norm_list = []
    sum_list = []

    # Step 0: Softmax
    prob = softmax(x, dim=-1)
    curr = prob + eps
    norm_list.append(prob.clone())
    sum_list.append(None)

    # Step 1: Initial Col Norm
    col_sum = curr.sum(dim=-2, keepdim=True) + eps
    curr = curr / col_sum
    norm_list.append(curr.clone())
    sum_list.append(col_sum.clone())

    # Step 2: Loop
    for i in range(num_iters - 1):
        # Row Norm
        row_sum = curr.sum(dim=-1, keepdim=True) + eps
        curr = curr / row_sum
        norm_list.append(curr.clone())
        sum_list.append(row_sum.clone())

        # Col Norm
        col_sum = curr.sum(dim=-2, keepdim=True) + eps
        curr = curr / col_sum
        norm_list.append(curr.clone())
        sum_list.append(col_sum.clone())

    return curr, norm_list, sum_list

def sinkhorn_grad_with_saved_tensors(grad_output, norm_out, sum_out, num_iters):
    # 将grad_output从[T, n, n]转换为[n, n, T]以匹配norm_out的格式
    grad_curr = grad_output.permute(1, 2, 0)  # [T, n, n] -> [n, n, T]
    # 共进行num_iters次迭代
    for i in range(num_iters - 1, 0, -1):
        # 列归一化的反向传播
        col_idx = 2 * i + 1
        # grad_curr: [n, n, T], norm_out[col_idx]: [n, n, T]
        # 列归一化是在axis=0方向求和（第一个n维度）
        dot_prod = torch.sum(grad_curr * norm_out[col_idx], dim=0, keepdim=True)  # [1, n, T]
        # sum_out[col_idx]: [n, T], 需要扩展为 [1, n, T]
        grad_curr = (grad_curr - dot_prod) / sum_out[col_idx].unsqueeze(0)

        # 行归一化的反向传播
        row_idx = 2 * i
        # grad_curr: [n, n, T], norm_out[row_idx]: [n, n, T]
        # 行归一化是在axis=1方向求和（第二个n维度）
        dot_prod = torch.sum(grad_curr * norm_out[row_idx], dim=1, keepdim=True)  # [n, 1, T]
        grad_curr = (grad_curr - dot_prod) / sum_out[row_idx].unsqueeze(1)


    # 最后一次迭代：列归一化
    dot_prod = torch.sum(grad_curr * norm_out[1], dim=0, keepdim=True)  # [1, n, T]
    grad_curr = (grad_curr - dot_prod) / sum_out[1].unsqueeze(0)

    dot_prod = torch.sum(grad_curr * norm_out[0], dim=1, keepdim=True)  # [n, 1, T]
    grad_input = norm_out[0] * (grad_curr - dot_prod)  # [n, n, T]
    # print(f"soft max")
    # print(grad_curr)
    # 将结果从[n, n, T]转回[T, n, n]
    grad_input = grad_input.permute(2, 0, 1)  # [n, n, T] -> [T, n, n]

    return grad_input


def convert_to_npu_format(norm_list, sum_list, num_iters):
    """
    将列表格式的中间结果转换为NPU格式

    参数:
        norm_list: 归一化输出列表，每个元素shape为[T, n, n]
        sum_list: 求和结果列表，每个元素shape为[T, 1, n]或[T, n, 1]
        num_iters: 迭代次数

    返回:
        norm_out: [2*num_iters, n, n, T]
        sum_out: [2*num_iters, n, T]
    """
    T, n, _ = norm_list[0].shape

    # 构建norm_out: [2*num_iters, n, n, T]
    norm_out = torch.zeros(2 * num_iters, n, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, norm_tensor in enumerate(norm_list):
        # [T, n, n] -> [n, n, T]
        norm_out[idx] = norm_tensor.permute(1, 2, 0)

    # 构建sum_out: [2*num_iters, n, T]
    sum_out = torch.zeros(2 * num_iters, n, T, dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, sum_tensor in enumerate(sum_list):
        if sum_tensor is None:
            continue
        # [T, 1, n] or [T, n, 1] -> [n, T]
        if sum_tensor.shape[-2] == 1:  # col_sum: [T, 1, n]
            sum_out[idx] = sum_tensor.squeeze(-2).T
        else:  # row_sum: [T, n, 1]
            sum_out[idx] = sum_tensor.squeeze(-1).T

    return norm_out, sum_out


def cmp_out_golden(x_golden, x_out, dtype):
    if dtype == 'int8':
        cmp = np.isclose(x_out.cpu().numpy()[:len(x_golden)], x_golden, atol=0)
    else:
        cmp = np.isclose(x_out.cpu().numpy()[:len(x_golden)], x_golden, rtol=1e-05, atol=1e-05)

    result = np.where(cmp != True)
    for i in range(len(result)):
        if len(result[i]) != 0:
            return False

    return True

class TestCustomSinkhornGrad(TestCase):
    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_sinkhorn_grad_netshape(self):
        for i in range(1):
            t = 2048
            n = 4
            num_iters = 20
            rtol = 1e-5  # 相对误差阈值
            atol = 2**-16  # 绝对误差阈值

            eps = 1e-6
            x = torch.randn(t, n, n, dtype=torch.float32)

            # 前向传播
            y, norm_list, sum_list = sinkhorn_forward_simple(x, num_iters, eps)
            # 构造输出梯度（假设loss = sum(y^2)）
            grad_output = 2 * y
            norm_out, sum_out = convert_to_npu_format(norm_list, sum_list, num_iters)
            grad_input = torch.ops.custom.npu_sinkhorn_grad(grad_output.npu(), norm_out.npu(), sum_out.npu())
            grad_input_golden = sinkhorn_grad_with_saved_tensors(grad_output, norm_out, sum_out, num_iters)

            grad_input_result = cmp_out_golden(grad_input_golden, grad_input, 'float')
            if (not grad_input_result):
                print(f"第{i}次比较失败，t={t}， n={n}，num_iters={num_iters}")
                print("======golden======")
                print(grad_input_golden)
                print("======npu======")
                print(grad_input.cpu())
                # data_compare(grad_input.cpu().to(torch.float32).numpy(), grad_input_golden.to(torch.float32).numpy())
                break
            # self.assertEqual(grad_input_result, True, "norm_out shape不匹配")
            self.assertRtolEqual(grad_input_golden, grad_input,  prec=0.004, prec16=0.004)

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_sinkhorn_grad_random(self):
        for i in range(1):
            # t = 2048
            # n = 4
            # num_iters = 20
            rtol = 1e-5  # 相对误差阈值
            atol = 2**-16  # 绝对误差阈值
            T_options = [4096, 8192, 16384, 32768]
            t = random.choice(T_options)

            if random.random() < 0.9:
                T_options = [4096, 8192, 16384, 32768]
                t = random.choice(T_options)
            else:
                T_options = [131072, 262144, 3072, 5120]
                t = random.choice(T_options)

            if random.random() < 0.8:
                N_options = [4, 6, 8]
                n = random.choice(N_options)
            else:
                n = random.randint(1, 12)

            # 随机选择 num_iter：1 到 100 之间的整数
            num_iters = random.randint(1, 100)
            # 80% 的几率选择 1 / 20 / 100，否则随机选 1~100
            if random.random() < 0.8:
                special_choices = [1, 20, 100]
                num_iters = random.choice(special_choices)
            else:
                num_iters = random.randint(1, 100)

            eps = 1e-6
            x = torch.randn(t, n, n, dtype=torch.float32)

            # 前向传播
            y, norm_list, sum_list = sinkhorn_forward_simple(x, num_iters, eps)
            # 构造输出梯度（假设loss = sum(y^2)）
            grad_output = 2 * y
            norm_out, sum_out = convert_to_npu_format(norm_list, sum_list, num_iters)
            grad_input = torch.ops.custom.npu_sinkhorn_grad(grad_output.npu(), norm_out.npu(), sum_out.npu())
            grad_input_golden = sinkhorn_grad_with_saved_tensors(grad_output, norm_out, sum_out, num_iters)

            grad_input_result = cmp_out_golden(grad_input_golden, grad_input, 'float')
            if (not grad_input_result):
                print(f"第{i}次比较失败，t={t}， n={n}，num_iters={num_iters}")
                print("======golden======")
                print(grad_input_golden)
                print("======npu======")
                print(grad_input.cpu())
                break
            self.assertRtolEqual(grad_input_golden, grad_input,  prec=0.004, prec16=0.004)

    @pytest.mark.resources(device="npu:*", npus_per_node=1)
    def test_sinkhorn_grad_uncontinue(self):
        for i in range(1):
            # t = 2048
            # n = 4
            # num_iters = 20
            rtol = 1e-5  # 相对误差阈值
            atol = 2**-16  # 绝对误差阈值
            T_options = [4096, 8192, 16384, 32768]
            t = random.choice(T_options)

            if random.random() < 0.9:
                T_options = [4096, 8192, 16384, 32768]
                t = random.choice(T_options)
            else:
                T_options = [131072, 262144, 3072, 5120]
                t = random.choice(T_options)

            if random.random() < 0.8:
                N_options = [4, 6, 8]
                n = random.choice(N_options)
            else:
                n = random.randint(1, 12)

            # total_elements = t * n * n
            # 随机选择 num_iter：1 到 100 之间的整数
            num_iters = random.randint(1, 100)
            # 80% 的几率选择 1 / 20 / 100，否则随机选 1~100
            if random.random() < 0.8:
                special_choices = [1, 20, 100]
                num_iters = random.choice(special_choices)
            else:
                num_iters = random.randint(1, 100)

            eps = 1e-6
            x = torch.randn(t, n, n, dtype=torch.float32)
            # 前向传播
            y, norm_list, sum_list = sinkhorn_forward_simple(x, num_iters, eps)
            # 构造输出梯度（假设loss = sum(y^2)）

            norm_out, sum_out = convert_to_npu_format(norm_list, sum_list, num_iters)

            # grad_output = 2 * y
            y = torch.randn(2 * t, 2 * n, 2 * n, dtype=torch.float32).npu()
            y_cpu = y.clone().cpu()
            grad_output = torch.as_strided(y, size=[t, n, n], stride=[8 * n * n, 4 * n, 2])
            grad_output_cpu = torch.as_strided(y_cpu, size=[t, n, n], stride=[8 * n * n, 4 * n, 2])

            norm_npu = torch.randn(4 * num_iters, 2 * t, 2 * n, 2 * n, dtype=torch.float32).npu()
            sum_npu = torch.randn(4 * num_iters, 2 * t, 2 * n, dtype=torch.float32).npu()
            norm_cpu = norm_npu.clone().cpu()
            sum_cpu = sum_npu.clone().cpu()
            norm_out_npu = torch.as_strided(norm_npu, size=[2 * num_iters, n, n, t], stride=[16 * n * n * t,8 * n * t, 4 * t, 2])
            sum_out_npu = torch.as_strided(sum_npu, size=[2 * num_iters, n, t], stride=[8 * n * t, 4 * t, 2])
            norm_out_cpu = torch.as_strided(norm_cpu, size=[2 * num_iters, n, n, t], stride=[16 * n * n * t,8 * n * t, 4 * t, 2])
            sum_out_cpu = torch.as_strided(sum_cpu, size=[2 * num_iters, n, t], stride=[8 * n * t, 4 * t, 2])
            norm_out_npu.copy_(norm_out)
            norm_out_cpu.copy_(norm_out)
            sum_out_npu.copy_(sum_out)
            sum_out_cpu.copy_(sum_out)

            grad_input = torch.ops.custom.npu_sinkhorn_grad(grad_output, norm_out_npu, sum_out_npu)

            grad_input_golden = sinkhorn_grad_with_saved_tensors(grad_output_cpu, norm_out_cpu, sum_out_cpu, num_iters)

            grad_input_result = cmp_out_golden(grad_input_golden, grad_input, 'float')
            if (not grad_input_result):
                print(f"第{i}次比较失败，t={t}， n={n}，num_iters={num_iters}")
                print("======golden======")
                print(grad_input_golden)
                print("======npu======")
                print(grad_input.cpu())
                break
            self.assertRtolEqual(grad_input_golden, grad_input,  prec=0.004, prec16=0.004)