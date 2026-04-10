import torch
import torch_npu
# import omni_training_custom_ops
import numpy as np
import random

from torch_npu.testing.testcase import TestCase, run_tests
import torch.nn.functional as F

torch.manual_seed(1234)
torch.set_printoptions(threshold=float('inf'))  # 阈值设为无穷大，永不截断
torch.set_printoptions(sci_mode=False, precision=5) 

ubSize_ = 248*1024
SIMD_RESERVED_SIZE = 8*1024
MASK_BUFFER = 64
MAX_BUFFER = 256
inputDTypeSize_ = 4
aivNum = 16
MIN_PER_CORE_ELEMENTS = 32
n = 4
align_n = 8
num_iters = 20
T = 2048
ubAvailableSize = ubSize_ - SIMD_RESERVED_SIZE - MASK_BUFFER - MAX_BUFFER # 240*1024 - 64 - 256  245440    norm_out 2,621,440   280,064
corePerLoopMaxTSize = ubAvailableSize // (inputDTypeSize_ * (2 * n * n + 2 * n * align_n + 2 * align_n))  # 32 + 64 + 32

coreTSize_ = (T + aivNum - 1) // aivNum
corePerLoopMaxTSize = min(corePerLoopMaxTSize, coreTSize_)
if coreTSize_ < MIN_PER_CORE_ELEMENTS and T >= MIN_PER_CORE_ELEMENTS:
    coreTSize_ = MIN_PER_CORE_ELEMENTS

needCoreNum_ = (T + coreTSize_ - 1) // coreTSize_ # 16
loop_ = (coreTSize_ + corePerLoopMaxTSize - 1) // corePerLoopMaxTSize # 1

sum_out_global = torch.zeros([2 * num_iters, needCoreNum_, loop_, align_n * corePerLoopMaxTSize], dtype=float)

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
    #  [needCoreNum_, loop_, 2 * num_iters, n, corePerLoopMaxTSize, align_n]
    #  n = n 
    #  align_n = align_n 
    #  T = needCoreNum_ * loop_ * corePerLoopMaxTSize
    grad_curr = grad_output.permute(1, 2, 0).contiguous() 
    norm_out = norm_out.permute(2, 3, 5, 0, 1, 4).contiguous() 
    sum_out = sum_out_global
    # print("norm_out.shape: ",norm_out.shape) # 40, 4, 8, 16, 1, 128
    # print("sum_out.shape: ",sum_out.shape) # 40, 16, 1, 8*128

    sum_out = sum_out.view(2 * num_iters, needCoreNum_, loop_, align_n, corePerLoopMaxTSize).permute(0, 1, 2, 4, 3).contiguous() 
    sum_out = sum_out.flatten(1, 3).permute(0, 2, 1).contiguous()  # T align_n
    norm_out = norm_out.flatten(3, 5) # n align_n T 
    print("norm_out.shape: ",norm_out.shape) # 4 8 2048
    print("sum_out.shape: ",sum_out.shape) # 8 2048
    print("grad_curr.shape: ",grad_curr.shape) # 4 4 2048
    sum_out_unpad = sum_out[:, :n, :]
    norm_out_unpad = norm_out[:, :, :n, :]
    print("norm_out.shape: ",norm_out_unpad.shape) # 4 8 2048
    print("sum_out.shape: ",sum_out_unpad.shape) # 8 2048
    print("grad_curr.shape: ",grad_curr.shape) # 4 4 2048
    # n, T, current_n = grad_curr.shape
    # padding_size = align_n - current_n
    # if padding_size > 0:
    #     grad_curr = F.pad(grad_curr, (0, padding_size), "constant", 0)  # 在最后一个维度上填充

    # 共进行num_iters次迭代
    for i in range(num_iters - 1, 0, -1):
        # 列归一化的反向传播
        col_idx = 2 * i + 1
        # grad_curr: [n, n, T], norm_out[col_idx]: [n, align_n, T]
        # 列归一化是在axis=0方向求和（第一个n维度）
        dot_prod = torch.sum(grad_curr * norm_out_unpad[col_idx], dim=0, keepdim=True)  # [1, n, T]  1, 4, 2048
        # sum_out_unpad[col_idx]: [align_n, T], 需要扩展为 [1, n, T]
        grad_curr = (grad_curr - dot_prod) / sum_out_unpad[col_idx].unsqueeze(0)

        # 行归一化的反向传播
        row_idx = 2 * i
        # grad_curr: [n, n, T], norm_out_unpad[row_idx]: [n, n, T]
        # 行归一化是在axis=1方向求和（第二个n维度）
        dot_prod = torch.sum(grad_curr * norm_out_unpad[row_idx], dim=1, keepdim=True)  # [n, 1, T]
        grad_curr = (grad_curr - dot_prod) / sum_out_unpad[row_idx].unsqueeze(1)


    # 最后一次迭代：列归一化
    dot_prod = torch.sum(grad_curr * norm_out_unpad[1], dim=0, keepdim=True)  # [1, n, T]
    grad_curr = (grad_curr - dot_prod) / sum_out_unpad[1].unsqueeze(0)

    dot_prod = torch.sum(grad_curr * norm_out_unpad[0], dim=1, keepdim=True)  # [n, 1, T]
    grad_input = norm_out_unpad[0] * (grad_curr - dot_prod)  # [n, n, T]
    # print(f"soft max")
    # print(grad_curr)
    # 将结果从[n, n, T]转回[T, n, n]
    grad_input = grad_input.permute(2, 0, 1).contiguous()   # [n, n, T] -> [T, n, n]
    # grad_input = []
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
    T, n, current_n = norm_list[0].shape
    align_n = 8
    padding_size = align_n - current_n

    ubSize_ = 248*1024
    SIMD_RESERVED_SIZE = 8*1024
    MASK_BUFFER = 64
    MAX_BUFFER = 256
    inputDTypeSize_ = 4
    aivNum = 16
    MIN_PER_CORE_ELEMENTS = 32
    ubAvailableSize = ubSize_ - SIMD_RESERVED_SIZE - MASK_BUFFER - MAX_BUFFER # 240*1024 - 64 - 256  245440    norm_out 2,621,440   280,064
    corePerLoopMaxTSize = ubAvailableSize // (inputDTypeSize_ * (2 * n * n + 2 * n * align_n + 2 * align_n))  # 32 + 64 + 32

    coreTSize_ = (T + aivNum - 1) // aivNum
    corePerLoopMaxTSize = min(corePerLoopMaxTSize, coreTSize_)
    if coreTSize_ < MIN_PER_CORE_ELEMENTS and T >= MIN_PER_CORE_ELEMENTS:
        coreTSize_ = MIN_PER_CORE_ELEMENTS

    needCoreNum_ = (T + coreTSize_ - 1) // coreTSize_ # 16
    loop_ = (coreTSize_ + corePerLoopMaxTSize - 1) // corePerLoopMaxTSize # 1

    # 构建norm_out: [2*num_iters, T, n, n] - > [needCoreNum_, loop_, 2 * num_iters, n, corePerLoopMaxTSize, align_n]
    norm_out = torch.zeros([2 * num_iters, needCoreNum_, loop_, n, corePerLoopMaxTSize, align_n], dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, norm_tensor in enumerate(norm_list):
        if padding_size > 0:
            norm_tensor = F.pad(norm_tensor, (0, padding_size), mode='constant', value=0)  # 在最后一个维度上填充[T, n, align_n]
        norm_out[idx] = norm_tensor.view(needCoreNum_, loop_, corePerLoopMaxTSize, n, align_n).permute(0, 1, 3, 2, 4).contiguous() 
    norm_out = norm_out.permute(1, 2, 0, 3, 4, 5).contiguous() 
    # 构建sum_out: [T, 1, n]或[T, n, 1]
    sum_out = torch.zeros([2 * num_iters, needCoreNum_, loop_, align_n * corePerLoopMaxTSize], dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, sum_tensor in enumerate(sum_list):
        if sum_tensor is None:
            continue
        # [T, 1, n] or [T, n, 1] -> [n, T]
        if sum_tensor.shape[-2] == 1:  # col_sum: [T, 1, n]
            if padding_size > 0:
                sum_tensor = F.pad(sum_tensor, (0, padding_size), mode='constant', value=0)  # 在最后一个维度上填充[T, 1, align_n]
            squeezed_tensor = sum_tensor.squeeze(-2)
            T = squeezed_tensor.shape[0]
            view_tensor = squeezed_tensor.view(needCoreNum_, loop_, corePerLoopMaxTSize, align_n)
            view_tensor_golbal = squeezed_tensor.view(needCoreNum_, loop_, corePerLoopMaxTSize, align_n).permute(0, 1, 3, 2).contiguous() 
            sum_out[idx] = view_tensor.flatten(-2)
            sum_out_global[idx] = view_tensor_golbal.flatten(-2)
            # print("col sum_tensor.shape: ", sum_out[idx].shape)
        else:  # row_sum: [T, n, 1]
            if padding_size > 0:
                sum_tensor = F.pad(sum_tensor, (0, 0, 0, padding_size, 0, 0), mode='constant', value=0)  # 在第二维度上填充[T, align_n, 1]
            squeezed_tensor = sum_tensor.squeeze(-1)
            T = squeezed_tensor.shape[0]
            view_tensor = squeezed_tensor.view(needCoreNum_, loop_, corePerLoopMaxTSize, align_n).permute(0, 1, 3, 2).contiguous() 
            sum_out[idx] = view_tensor.flatten(-2)
            sum_out_global[idx] = view_tensor.flatten(-2)
            # print("row sum_tensor.shape: ", sum_out[idx].shape)
    sum_out = sum_out.permute(1, 2, 0, 3).contiguous() 

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

def test_sinkhorn_grad_netshape():
    for i in range(1):
        t = 2048
        n = 4
        num_iters = 20
        rtol = 1e-5  # 相对误差阈值
        atol = 2**-16  # 绝对误差阈值

        eps = 1e-6
        x = torch.randn(t, n, n, dtype=torch.float32)
        align_n = 8
        padding_size = align_n - n

        # 前向传播
        y, norm_list, sum_list = sinkhorn_forward_simple(x, num_iters, eps)
        # y = y.permute(1, 0, 2)  # [T, n, n] -> [n, T, align_n]
        # if padding_size > 0:
        #     y = F.pad(y, (0, padding_size), "constant", 0)  # 在最后一个维度上填充
        # 构造输出梯度（假设loss = sum(y^2)）
        grad_output = 2 * y
        norm_out, sum_out = convert_to_npu_format(norm_list, sum_list, num_iters)
        print("=======================================================")
        print("norm_out.shape: ", norm_out.shape)  # 16, 1, 40, 4, 128, 8
        print("sum_out.shape: ", sum_out.shape)  # 16, 1, 40, 128*8
        print("grad_output.shape: ", grad_output.shape)  # 2048, 4, 4
        print("=======================================================")
        grad_input_golden = sinkhorn_grad_with_saved_tensors(grad_output, norm_out, sum_out, num_iters)
        # grad_output = grad_output.flatten()
        # norm_out = norm_out.flatten()
        # sum_out = sum_out.flatten()
        grad_input = torch.ops.npu.npu_mhc_sinkhorn_backward(grad_output.npu(), norm_out.npu(), sum_out.npu())
        print("grad_input.shape: ", grad_input.shape)
        print(grad_input)
        # print("grad_input_golden.shape: ", grad_input_golden)
        grad_input_result = cmp_out_golden(grad_input_golden, grad_input.cpu(), 'float')
        # if (not grad_input_result):
        #     print(f"第{i}次比较失败，t={t}， n={n}，num_iters={num_iters}")
        #     print("======golden======")
        #     print(grad_input_golden)
        #     print("======npu======")
        #     print(grad_input.cpu())
        #     # data_compare(grad_input.cpu().to(torch.float32).numpy(), grad_input_golden.to(torch.float32).numpy())
        #     break


if __name__ == "__main__":
    test_sinkhorn_grad_netshape()
