import torch
import torch_npu
import numpy as np 
import torch.nn.functional as F

torch.npu.set_device(0)
torch.manual_seed(1234)
torch.set_printoptions(threshold=float('inf'))
torch.set_printoptions(sci_mode=False, precision=5)

ubSize_ = 248 * 1024
SIMD_RESERVED_SIZE = 8 * 1024
MASK_BUFFER = 64
MAX_BUFFER = 256
inputDTypeSize_ = 4
aivNum = 16
MIN_PER_CORE_ELEMENTS = 32
n = 4
align_n = 8
padding_size = align_n - n 
num_iters = 20
T = 2048

def ceildiv(x, y):
    return (x + y - 1) // y

coreTSize_ = ceildiv(T, aivNum) # tNormCore_
needCoreNum_ = ceildiv(T, coreTSize_) # usedCoreNum_
TailCoreTSize_ = T - coreTSize_ * (needCoreNum_ - 1) # tTailCore_

ubAvailableSize = ubSize_ - SIMD_RESERVED_SIZE - MASK_BUFFER - MAX_BUFFER # ubSizeUsed_

corePerLoopMaxTSize = ubAvailableSize // (inputDTypeSize_ * (2 * n * n + 2 * n * align_n + 2 * align_n)) # tUbfactor

corePerLoopMaxTSize = coreTSize_ if coreTSize_ * inputDTypeSize_ <= ubAvailableSize else corePerLoopMaxTSize

corePerLoopMaxTSize = ((corePerLoopMaxTSize + 7) // 8) * 8

tNormCoreLoop_ = ceildiv(coreTSize_, corePerLoopMaxTSize)  # 整核loop_, 整核的整块大小都是corePerLoopMaxTSize
tTailCoreLoop_ = ceildiv(TailCoreTSize_, corePerLoopMaxTSize) # 尾核loop_
loop_ = tNormCoreLoop_

tUbfactorTail_ = coreTSize_ - (tNormCoreLoop_ - 1) * corePerLoopMaxTSize if tNormCoreLoop_ > 1 else 0 # 整核尾块
tUbTailTail_ = TailCoreTSize_ - (tTailCoreLoop_ - 1) * corePerLoopMaxTSize if tTailCoreLoop_ > 1 else 0 # 尾核尾块

# if coreTSize_ < MIN_PER_CORE_ELEMENTS and T >= MIN_PER_CORE_ELEMENTS:
#     coreTSize_ = MIN_PER_CORE_ELEMENTS

# loop_ = (coreTSize_ + corePerLoopMaxTSize - 1) // corePerLoopMaxTSize


sum_out_global_base_base = torch.zeros([2 * num_iters, needCoreNum_, tNormCoreLoop_, align_n * corePerLoopMaxTSize], dtype=float)
sum_out_global_base_tail = torch.zeros([2 * num_iters, needCoreNum_, tNormCoreLoop_, align_n * tUbfactorTail_], dtype=float)
# 整核的shape为[2 * num_iters, needCoreNum_, tNormCoreLoop_, align_n * (corePerLoopMaxTSize + tUbfactorTail_)]
sum_out_global_base = torch.cat([sum_out_global_base_base, sum_out_global_base_tail], dim=-1)

sum_out_global_tail_base = torch.zeros([2 * num_iters, needCoreNum_, tTailCoreLoop_, align_n * corePerLoopMaxTSize], dtype=float)
sum_out_global_tail_tail = torch.zeros([2 * num_iters, needCoreNum_, tTailCoreLoop_, align_n * tUbTailTail_], dtype=float)
# 尾核的shape为[2 * num_iters, needCoreNum_, tTailCoreLoop_, align_n * (corePerLoopMaxTSize + tUbfactorTail_)]
sum_out_global_tail = torch.cat([sum_out_global_tail_base, sum_out_global_tail_tail], dim=-1)

# 整个sum_out的shape为[2 * num_iters, needCoreNum_, (tNormCoreLoop_ + tTailCoreLoop_), align_n * (corePerLoopMaxTSize + tUbfactorTail_)]
sum_out_global = torch.cat([sum_out_global_base, sum_out_global_tail], dim=-2)


def softmax(x, dim = -1):
    return torch.softmax(x, dim = dim)

def sinkhorn_forward_simple(x, num_iters, eps):
    norm_list = []
    sum_list = []

    # Step 0: Softmax
    prob = softmax(x, dim = -1)
    curr = prob + eps
    norm_list.append(prob.clone())
    sum_list.append(None)

    # Step 1: Initail col Norm
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
    grad_curr = grad_output.permute(1, 2, 0).contiguous()
    norm_out = norm_out.permute(2, 3, 5, 0, 1, 4).contiguous()
    sum_out = sum_out_global

    sum_out = sum_out.view(2 * num_iters, needCoreNum_, loop_, align_n, corePerLoopMaxTSize).permute(0, 1, 2, 4, 3).contiguous()
    sum_out = sum_out.flatten(1, 3).permute(0, 2, 1).contiguous()
    norm_out = norm_out.flatten(3, 5)
    sum_out_unpad = sum_out[:, :n, :]
    norm_out_unpad = norm_out[:, :, :n, :]

    for i in range(num_iters - 1, 0, -1):
        col_idx = 2 * i + 1
        dot_prod = torch.sum(grad_curr * norm_out_unpad[col_idx], dim=0, keepdim=True)
        grad_curr = (grad_curr - dot_prod) / sum_out_unpad[col_idx].unsqueeze(0)

        row_idx = 2 * i 

        dot_prod = torch.sum(grad_curr * norm_out_unpad[row_idx], dim=1, keepdim=True)
        grad_curr = (grad_curr - dot_prod) / sum_out_unpad[row_idx].unsqueeze(1)

    dot_prod = torch.sum(grad_curr * norm_out_unpad[1], dim=0, keepdim=True)
    grad_curr = (grad_curr - dot_prod) / sum_out_unpad[1].unsqueeze(0)

    dot_prod = torch.sum(grad_curr * norm_out_unpad[0], dim=1, keepdim=True)
    grad_input = norm_out_unpad[0] * (grad_curr - dot_prod)

    grad_input = grad_input.permute(2, 0, 1).contiguous()

    return grad_input

def convert_to_npu_format(norm_list, sum_list, num_iters):
    # 构建norm_out: [2*num_iters, T, n, n] -> [needCoreNum_, loop_, 2 * num_iters, n, corePerLoopMaxTSize, align_n]
    norm_out = torch.zeros([2 * num_iters, needCoreNum_, loop_, n, corePerLoopMaxTSize, align_n], dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, norm_tensor in enumerate(norm_list):
        if padding_size > 0:
            norm_tensor = F.pad(norm_tensor, (0, padding_size), mode='constant', value=0)
        norm_out[idx] = norm_tensor.view(needCoreNum_, loop_, corePerLoopMaxTSize, n, align_n).permute(0, 1, 3, 2, 4).contiguous()
    norm_out = norm_out.permute(1, 2, 0, 3, 4, 5).contiguous()

    # 构建sum_out: [T, 1, n] 或 [T, n, 1]
    sum_out = torch.zeros([2 * num_iters, needCoreNum_, loop_, align_n * corePerLoopMaxTSize], dtype=norm_list[0].dtype, device=norm_list[0].device)
    for idx, sum_tensor in enumerate(sum_list):
        if sum_tensor is None:
            continue
        if sum_tensor.shape[-2] == 1: # col
            if padding_size > 0:
                sum_tensor = F.pad(sum_tensor, (0, padding_size), mode = 'constant', value=0)
            squeezed_tensor = sum_tensor.squeeze(-2)
            if tNormCoreLoop_ != 0 and corePerLoopMaxTSize != 0:
                start_idx = 0
                end_idx = needCoreNum_ * tNormCoreLoop_* corePerLoopMaxTSize
                sum_out_base_base = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tNormCoreLoop_, corePerLoopMaxTSize, align_n)
            if tNormCoreLoop_ != 0 and tUbfactorTail_ != 0:
                start_idx = needCoreNum_ * tNormCoreLoop_* corePerLoopMaxTSize
                end_idx = needCoreNum_ * tNormCoreLoop_* (corePerLoopMaxTSize + tUbfactorTail_)
                sum_out_base_tail = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tNormCoreLoop_, tUbfactorTail_, align_n)
            if tTailCoreLoop_ != 0 and corePerLoopMaxTSize != 0:
                start_idx = needCoreNum_ * tNormCoreLoop_* (corePerLoopMaxTSize + tUbfactorTail_)
                end_idx = start_idx + needCoreNum_ * tTailCoreLoop_* corePerLoopMaxTSize
                sum_out_tail_base = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tTailCoreLoop_, corePerLoopMaxTSize, align_n)
            if tNormCoreLoop_ != 0 and corePerLoopMaxTSize != 0:
                start_idx = needCoreNum_ * tNormCoreLoop_* (corePerLoopMaxTSize + tUbfactorTail_) + needCoreNum_ * tTailCoreLoop_* corePerLoopMaxTSize
                end_idx = len(squeezed_tensor)
                sum_out_tail_tail = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tTailCoreLoop_, tUbfactorTail_, align_n)

            sum_out_base = torch.cat([sum_out_base_base, sum_out_base_tail], dim=-1)
            sum_out_tail = torch.cat([sum_out_tail_base, sum_out_tail_tail], dim=-1)
            sum_out_tensor = torch.cat([sum_out_base, sum_out_tail], dim=-2)

            sum_out_global[idx] = sum_out_tensor.flatten(-2)
            # sum_out_tensor = sum_out_tensor.permute(0, 1, 3, 2).contiguous()
            sum_out[idx] = sum_out_tensor.flatten(-2)
        else: #row
            if padding_size > 0:
                sum_tensor = F.pad(sum_tensor, (0, 0, 0, padding_size, 0, 0), mode = 'constant', value=0)
            squeezed_tensor = sum_tensor.squeeze(-1)
            if tNormCoreLoop_ != 0 and corePerLoopMaxTSize != 0:
                start_idx = 0
                end_idx = needCoreNum_ * tNormCoreLoop_* corePerLoopMaxTSize
                sum_out_base_base = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tNormCoreLoop_, corePerLoopMaxTSize, align_n)
            if tNormCoreLoop_ != 0 and tUbfactorTail_ != 0:
                start_idx = needCoreNum_ * tNormCoreLoop_* corePerLoopMaxTSize
                end_idx = needCoreNum_ * tNormCoreLoop_* (corePerLoopMaxTSize + tUbfactorTail_)
                sum_out_base_tail = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tNormCoreLoop_, tUbfactorTail_, align_n)
            if tTailCoreLoop_ != 0 and corePerLoopMaxTSize != 0:
                start_idx = needCoreNum_ * tNormCoreLoop_* (corePerLoopMaxTSize + tUbfactorTail_)
                end_idx = start_idx + needCoreNum_ * tTailCoreLoop_* corePerLoopMaxTSize
                sum_out_tail_base = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tTailCoreLoop_, corePerLoopMaxTSize, align_n)
            if tTailCoreLoop_ != 0 and tUbfactorTail_ != 0:
                start_idx = needCoreNum_ * tNormCoreLoop_* (corePerLoopMaxTSize + tUbfactorTail_) + needCoreNum_ * tTailCoreLoop_* corePerLoopMaxTSize
                end_idx = len(squeezed_tensor)
                sum_out_tail_tail = squeezed_tensor[start_idx:end_idx].view(needCoreNum_, tTailCoreLoop_, tUbfactorTail_, align_n)

            sum_out_base = torch.cat([sum_out_base_base, sum_out_base_tail], dim=-1)
            sum_out_tail = torch.cat([sum_out_tail_base, sum_out_tail_tail], dim=-1)
            sum_out_tensor = torch.cat([sum_out_base, sum_out_tail], dim=-2)

            sum_out_global[idx] = sum_out_tensor.flatten(-2)
            sum_out_tensor = sum_out_tensor.permute(0, 1, 3, 2).contiguous()
            sum_out[idx] = sum_out_tensor.flatten(-2)            

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
        rtol = 1e-5
        atol = 2**-16
        eps = 1e-6
        # x = torch.randn(t, n, n, dtype=torch.float32)
        x = torch.full((t, n, n), 0.25, dtype=torch.float32)
        align_n = 8
        padding_size = align_n - n

        y, norm_list, sum_list = sinkhorn_forward_simple(x, num_iters, eps)
        grad_output = 2 * y
        norm_out, sum_out = convert_to_npu_format(norm_list, sum_list, num_iters)

        print("=======================================================")
        print("norm_out.shape: ", norm_out.shape) # 16, 1, 40, 4, 128, 8
        # print("norm_out: ", norm_out)
        print("=======================================================")
        print("sum_out.shape: ", sum_out.shape) # 16, 1, 40, 8*128
        # print("sum_out: ", sum_out)
        print("=======================================================")
        print("grad_output.shape: ", grad_output.shape)
        # print("grad_output: ", grad_output)
        print("=======================================================")

        grad_input_golden = sinkhorn_grad_with_saved_tensors(grad_output, norm_out, sum_out, num_iters)

        # grad_input = torch.ops.npu.npu_mhc_sinkhorn_backward(grad_output.npu(), norm_out.npu(), sum_out.npu())

        # grad_input_result = cmp_out_golden(grad_input_golden, grad_input.cpu(), 'float')
        # print(grad_input_result)

if __name__ == "__main__":
    test_sinkhorn_grad_netshape()
