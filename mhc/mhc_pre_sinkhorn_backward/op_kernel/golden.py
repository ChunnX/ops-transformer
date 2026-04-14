import torch, torch_npu
import triton
import triton.language as tl
import tilelang
import tilelang.language as T
import torch.nn.functional as F

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32], # b * s, mix_hc
        hc_scale: T.Tensor[(3,), T.float32], # 3
        hc_base: T.Tensor[(mix_hc,), T.float32], # mix_hc
        pre: T.Tensor[(n, hc), FP32], # b * s,  4 
        post: T.Tensor[(n, hc), FP32], # b * s,  4
        comb: T.Tensor[(n, hc, hc), FP32], # b * s,  4, 4
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])

            for _ in T.serial(sinkhorn_iters): # 20
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), # b * s * mix_hc / 24, 24
           hc_scale, # 3
           hc_base, # mix_hc
           pre.view(-1, hc_mult), # b * s,  4
           post.view(-1, hc_mult), # b * s,  4
           comb.view(-1, hc_mult, hc_mult)) # b * s,  4, 4
    return pre, post, comb


def hc_split_sinkhorn_torch_reference(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HC-Split Sinkhorn 算子的 PyTorch 原生精度标杆实现
    完全对齐原算子逻辑，用于验证 Triton/TileLang 版本的精度
    
    Args:
        mixes: 输入张量，形状 [b, s, (2+hc_mult)*hc_mult] 即，[b, s, hc_mult + hc_mult + hc_mult*hc_mult] 
        hc_scale: 缩放张量，形状 [3]
        hc_base: 偏置张量，形状 [(2+hc_mult)*hc_mult] 即，[hc_mult + hc_mult + hc_mult*hc_mult] 
        hc_mult: HC维度大小，默认4
        sinkhorn_iters: Sinkhorn迭代次数，默认20
        eps: 防止除零的小常数，默认1e-6
    
    Returns:
        pre: [b, s, hc_mult]，Sigmoid激活+eps
        post: [b, s, hc_mult]，2×Sigmoid激活
        comb: [b, s, hc_mult, hc_mult]，Sinkhorn归一化后的矩阵
    """
    # 0. 输入校验（保证计算合法性）
    assert mixes.dim() == 3, f"mixes must be 3D, got {mixes.dim()}D"
    assert hc_scale.shape == (3,), f"hc_scale must be [3], got {hc_scale.shape}"
    assert hc_base.shape == ((2 + hc_mult) * hc_mult,), \
        f"hc_base must be [{(2 + hc_mult) * hc_mult}], got {hc_base.shape}"
    assert eps > 0, "eps must be positive to avoid division by zero"
    
    # 1. 保存原始形状，用于最终重塑
    b, s, _ = mixes.shape
    # 展平为 [b*s, (2+hc_mult)*hc_mult]（对齐原算子的展平逻辑）
    mixes_flat = mixes.view(-1, (2 + hc_mult) * hc_mult)

    # 2. 计算pre：[b*s, hc_mult]
    pre_slice = mixes_flat[:, :hc_mult]  # 前hc_mult维
    pre_flat = torch.sigmoid(pre_slice * hc_scale[0] + hc_base[:hc_mult]) + eps
    
    # 3. 计算post：[b*s, hc_mult]
    post_slice = mixes_flat[:, hc_mult:2*hc_mult]  # 中间hc_mult维
    post_flat = 2 * torch.sigmoid(post_slice * hc_scale[1] + hc_base[hc_mult:2*hc_mult])

    # 4. 计算comb初始值：[b*s, hc_mult, hc_mult]
    comb_slice = mixes_flat[:, 2*hc_mult:]  # 最后hc_mult×hc_mult维
    comb_init_flat = comb_slice.view(-1, hc_mult, hc_mult)  # 重塑为二维矩阵
    # 线性变换（scale+base）：base需广播到batch维度
    
    comb_init_flat = comb_init_flat * hc_scale[2] + hc_base[2*hc_mult:].view(1, hc_mult, hc_mult)

    # print('comb_init_flat', comb_init_flat)
    # 5. comb初始Softmax（行维度）+ 首次列归一化（对齐原算子）
    comb_flat = comb_init_flat.clone()
    # 按行减最大值（数值稳定，避免exp爆炸）
    row_max = comb_flat.max(dim=-1, keepdim=True).values
    comb_flat = torch.exp(comb_flat - row_max)
    # print('comb_flat', comb_flat)
    # 6. Sinkhorn迭代（交替行/列归一化）
    for _ in range(sinkhorn_iters):
        # 6.1 行归一化
        row_sum = comb_flat.sum(dim=-1, keepdim=True)
        comb_flat = comb_flat / (row_sum + eps)
        # 6.2 列归一化
        col_sum = comb_flat.sum(dim=-2, keepdim=True)
        comb_flat = comb_flat / (col_sum + eps)
    
    # 7. 重塑为原始形状 [b, s, ...]
    pre = pre_flat.view(b, s, hc_mult)
    post = post_flat.view(b, s, hc_mult)
    comb = comb_flat.view(b, s, hc_mult, hc_mult)
    
    return pre, post, comb


def hc_split_sinkhorn_torch_reference_backward(
        grad_pre: torch.Tensor,
        grad_post: torch.Tensor,
        grad_comb: torch.Tensor,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HC-Split Sinkhorn 算子的反向传播

    Args:
        grad_pre: 对pre的梯度，形状 [b, s, hc_mult]
        grad_post: 对post的梯度，形状 [b, s, hc_mult]
        grad_comb: 对comb的梯度，形状 [b, s, hc_mult, hc_mult]
        mixes: 前向传播的输入张量，形状 [b, s, (2+hc_mult)*hc_mult]
        hc_scale: 缩放张量，形状 [3]
        hc_base: 偏置张量，形状 [(2+hc_mult)*hc_mult]
        hc_mult: HC维度大小，默认4
        sinkhorn_iters: Sinkhorn迭代次数，默认20
        eps: 防止除零的小常数，默认1e-6

    Returns:
        grad_mixes: 对mixes的梯度，形状 [b, s, (2+hc_mult)*hc_mult]
        grad_hc_scale: 对hc_scale的梯度，形状 [3]
        grad_hc_base: 对hc_base的梯度，形状 [(2+hc_mult)*hc_mult]
    """
    # 1. 保存原始形状
    b, s, _ = mixes.shape
    total_dim = (2 + hc_mult) * hc_mult

    # 展平输入
    mixes_flat = mixes.view(-1, total_dim)

    # 初始化梯度
    grad_mixes_flat = torch.zeros_like(mixes_flat)
    grad_hc_scale = torch.zeros_like(hc_scale)
    grad_hc_base = torch.zeros_like(hc_base)

    # 2. 计算 pre 部分的梯度
    pre_slice = mixes_flat[:, :hc_mult]
    pre_input = pre_slice * hc_scale[0] + hc_base[:hc_mult]
    sigmoid_val = torch.sigmoid(pre_input)

    grad_pre_flat = grad_pre.view(-1, hc_mult)
    grad_pre_input = grad_pre_flat * (sigmoid_val * (1 - sigmoid_val))

    grad_mixes_flat[:, :hc_mult] = grad_pre_input * hc_scale[0]
    grad_hc_scale[0] = torch.sum(grad_pre_input * pre_slice)
    grad_hc_base[:hc_mult] = torch.sum(grad_pre_input, dim=0)

    # 3. 计算 post 部分的梯度
    post_slice = mixes_flat[:, hc_mult:2 * hc_mult]
    post_input = post_slice * hc_scale[1] + hc_base[hc_mult:2 * hc_mult]
    sigmoid_val_post = torch.sigmoid(post_input)

    grad_post_flat = grad_post.view(-1, hc_mult)
    grad_post_input = grad_post_flat * 2 * (sigmoid_val_post * (1 - sigmoid_val_post))

    grad_mixes_flat[:, hc_mult:2 * hc_mult] = grad_post_input * hc_scale[1]
    grad_hc_scale[1] = torch.sum(grad_post_input * post_slice)
    grad_hc_base[hc_mult:2 * hc_mult] = torch.sum(grad_post_input, dim=0)

    # 4. 计算 comb 部分的梯度
    # 获取 comb 部分的输入切片
    comb_slice_flat = mixes_flat[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)

    # 前向计算：线性变换
    comb_base = hc_base[2 * hc_mult:].view(1, hc_mult, hc_mult)
    comb_init = comb_slice_flat * hc_scale[2] + comb_base
    # print('comb_init', comb_init)
    # 重新进行前向计算，保存中间变量
    # 行内减最大值 + 指数
    row_max = comb_init.max(dim=-1, keepdim=True).values
    exp_comb = torch.exp(comb_init - row_max)

    # 进行Sinkhorn迭代，保存中间结果
    K = exp_comb.clone()
    intermediate_results = []

    for _ in range(sinkhorn_iters):
        # 行归一化
        row_sum = K.sum(dim=-1, keepdim=True)
        K_row = K / (row_sum + eps)

        # 列归一化
        col_sum = K_row.sum(dim=-2, keepdim=True)
        K_col = K_row / (col_sum + eps)

        # 保存中间结果
        intermediate_results.append({
            # 'K_row': K_row,
            'row_sum': row_sum,
            # 'K_col': K_col,
            'col_sum': col_sum
        })
        # 更新K
        K = K_col

    # 反向传播Sinkhorn迭代
    grad_comb_flat = grad_comb.view(-1, hc_mult, hc_mult)
    dK = grad_comb_flat.clone()
    # print('dK memory', dK)
    # print('K memory', K)
    for i in reversed(range(sinkhorn_iters)):
        inter = intermediate_results[i]

        row_sum = inter['row_sum']
        col_sum = inter['col_sum']
        # print('row_sum', row_sum)
        # print('col_sum', col_sum)
        # 复原K_col
        K_col = K * col_sum + eps

        # 反向传播列归一化
        grad_direct = dK / (col_sum + eps)
        d_col_sum = - (dK * K_col / (col_sum + eps) ** 2).sum(dim=-2, keepdim=True)
        dK_row = grad_direct + d_col_sum

        # 复原K_row
        K_row = K_col * row_sum
        K = K_row
        # 反向传播行归一化
        grad_direct_row = dK_row / (row_sum + eps)
        d_row_sum = - (dK_row * K_row / (row_sum + eps) ** 2).sum(dim=-1, keepdim=True)
        dK = grad_direct_row + d_row_sum

    # print('dK ref', dK)
    # 现在 dK 是 exp_comb 的梯度
    d_exp_comb = dK

    # 反向传播指数和减最大值操作
    d_comb_before_exp = d_exp_comb * exp_comb

    # 处理减最大值的梯度
    max_mask = (comb_init == row_max).float()
    max_count = max_mask.sum(dim=-1, keepdim=True)
    row_sum_d_before_exp = d_comb_before_exp.sum(dim=-1, keepdim=True)

    d_comb_init = d_comb_before_exp - (row_sum_d_before_exp * max_mask / max_count)

    # 反向传播线性变换
    grad_comb_slice = d_comb_init * hc_scale[2]
    grad_hc_scale[2] = torch.sum(d_comb_init * comb_slice_flat)
    grad_hc_base_comb = torch.sum(d_comb_init, dim=0)

    # 更新梯度
    grad_mixes_flat[:, 2 * hc_mult:] = grad_comb_slice.view(-1, hc_mult * hc_mult)
    grad_hc_base[2 * hc_mult:] = grad_hc_base_comb.view(-1)

    # 5. 重塑梯度为原始形状
    grad_mixes = grad_mixes_flat.view(b, s, total_dim)

    return grad_mixes, grad_hc_scale, grad_hc_base


@triton.jit
def _hc_split_sinkhorn_kernel_part1(
    # 输入输出张量指针
    mixes_ptr, hc_scale_ptr, hc_base_ptr,
    pre_ptr, post_ptr,
    # 维度参数
    batch_seq_size, hc_mult, sinkhorn_iters,
    # 常量参数
    eps, feat_dim, hc_square,
    # 块大小（编译期常量）
    BLOCK_HC: tl.constexpr,
):
    """
    Triton 核函数：实现 HC-Split Sinkhorn 的核心计算逻辑
    移除所有 CUDA 相关校验，兼容低版本 Triton（无 keepdim 参数）
    """
    # 1. 获取当前线程块处理的 batch_seq 索引（一维网格）
    b_s_idx = tl.program_id(0)
    if b_s_idx >= batch_seq_size:
        return  # 超出范围直接返回，避免越界

    # 2. 计算当前 batch_seq 对应的内存偏移
    mixes_offset = b_s_idx * feat_dim
    pre_offset = b_s_idx * hc_mult
    post_offset = b_s_idx * hc_mult
    comb_offset = b_s_idx * hc_square

    # 3. 加载 hc_scale 和 hc_base（全局内存→寄存器）
    # hc_scale: [3] → 加载3个标量
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)

    # hc_base: 分3段加载（pre/post/comb）
    base_pre = tl.load(hc_base_ptr + tl.arange(0, BLOCK_HC))
    base_post = tl.load(hc_base_ptr + hc_mult + tl.arange(0, BLOCK_HC))

    # 4. 加载当前 batch_seq 的 mixes 数据（全局内存→寄存器）
    mixes_pre = tl.load(mixes_ptr + mixes_offset + tl.arange(0, BLOCK_HC))
    mixes_post = tl.load(mixes_ptr + mixes_offset + hc_mult + tl.arange(0, BLOCK_HC))

    # 5. 计算 Pre: sigmoid(mixes_pre * scale_pre + base_pre) + eps
    pre = tl.sigmoid(mixes_pre * scale_pre + base_pre) + eps
    # 存储 Pre 结果到全局内存
    tl.store(pre_ptr + pre_offset + tl.arange(0, BLOCK_HC), pre)

    # 6. 计算 Post: 2 * sigmoid(mixes_post * scale_post + base_post)
    post = 2 * tl.sigmoid(mixes_post * scale_post + base_post)
    # 存储 Post 结果到全局内存
    tl.store(post_ptr + post_offset + tl.arange(0, BLOCK_HC), post)


@triton.jit
def _hc_split_sinkhorn_kernel_part2(
    # 输入输出张量指针
    mixes_ptr, hc_scale_ptr, hc_base_ptr,
    comb_ptr,
    # 维度参数
    batch_seq_size, hc_mult, sinkhorn_iters,
    # 常量参数
    eps,
    # 块大小（编译期常量）
    BLOCK_HC: tl.constexpr,
    BLOCK_ALIGN: tl.constexpr,
):
    """
    Triton 核函数：实现 HC-Split Sinkhorn 的核心计算逻辑
    移除所有 CUDA 相关校验，兼容低版本 Triton（无 keepdim 参数）
    """
    # 1. 获取当前线程块处理的 batch_seq 索引（一维网格）
    b_s_idx = tl.program_id(0)
    if b_s_idx >= batch_seq_size:
        return  # 超出范围直接返回，避免越界

    # 2. 计算当前 batch_seq 对应的内存偏移
    mixes_offset = b_s_idx * BLOCK_HC * BLOCK_ALIGN
    pre_offset = b_s_idx * hc_mult
    post_offset = b_s_idx * hc_mult
    hc_square = BLOCK_HC*BLOCK_HC
    comb_offset = mixes_offset

    # 3. 加载 hc_scale 和 hc_base（全局内存→寄存器）
    # hc_scale: [3] → 加载3个标量
    scale_comb = tl.load(hc_scale_ptr + 2)

    # hc_base: 分3段加载（pre/post/comb）
    base_comb = tl.load(hc_base_ptr + tl.arange(0, BLOCK_HC*BLOCK_ALIGN))
    base_comb = tl.reshape(base_comb, (BLOCK_HC, BLOCK_ALIGN))

    # 4. 加载当前 batch_seq 的 mixes 数据（全局内存→寄存器）
    mixes_comb = tl.load(mixes_ptr + mixes_offset + tl.arange(0, BLOCK_HC*BLOCK_ALIGN))
    mixes_comb = tl.reshape(mixes_comb, (BLOCK_HC, BLOCK_ALIGN))

    # 7. 计算 Comb 初始值: mixes_comb * scale_comb + base_comb
    comb = mixes_comb * scale_comb + base_comb

    # comb_flat = tl.reshape(comb, (BLOCK_HC*BLOCK_ALIGN,))
    # tl.store(comb_ptr + comb_offset + tl.arange(0, BLOCK_HC*BLOCK_ALIGN), comb_flat)

    # 8. Comb 初始 Softmax（对齐 PyTorch 逻辑，修复 keepdim 问题）
    # 8.1 按行减最大值（数值稳定）
    row_max = tl.max(comb, 1)  # 低版本 Triton 兼容（无 keepdim）
    row_max = tl.reshape(row_max, (BLOCK_HC, 1))  # 模拟 keepdim=True
    comb = tl.exp(comb - row_max)

    mask1 = tl.arange(0, BLOCK_HC*BLOCK_ALIGN) % BLOCK_ALIGN < BLOCK_HC 

    comb = tl.reshape(comb, (BLOCK_HC * BLOCK_ALIGN))
    comb = tl.where(mask1, comb, 0.0)
    comb = tl.reshape(comb, (BLOCK_HC, BLOCK_ALIGN))

    # 9. Sinkhorn 迭代（交替行/列归一化，迭代次数 sinkhorn_iters-1）
    for _ in range(sinkhorn_iters):
        # 9.1 行归一化
        row_sum = tl.sum(comb, axis=1)
        row_sum = tl.reshape(row_sum, (BLOCK_HC, 1))
        comb = comb / (row_sum + eps)
        # 9.2 列归一化
        col_sum = tl.sum(comb, axis=0)
        col_sum = tl.reshape(col_sum, (1, BLOCK_ALIGN))
        comb = comb / (col_sum + eps)

    # 10. 存储 Comb 结果到全局内存（展平后存储）
    comb_flat = tl.reshape(comb, (BLOCK_HC*BLOCK_ALIGN,))
    tl.store(comb_ptr + comb_offset + tl.arange(0, BLOCK_HC*BLOCK_ALIGN), comb_flat)


def hc_split_sinkhorn_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HC-Split Sinkhorn 的 Triton 实现，完全对齐 PyTorch 精度标杆
    移除所有 CUDA 设备校验逻辑
    
    Args:
        mixes: 输入张量，形状 [b, s, (2+hc_mult)*hc_mult]
        hc_scale: 缩放张量，形状 [3]
        hc_base: 偏置张量，形状 [(2+hc_mult)*hc_mult]
        hc_mult: HC维度大小，默认4
        sinkhorn_iters: Sinkhorn迭代次数，默认20
        eps: 防止除零的小常数，默认1e-6
    
    Returns:
        pre: [b, s, hc_mult]
        post: [b, s, hc_mult]
        comb: [b, s, hc_mult, hc_mult]
    """
    # 0. 仅保留核心输入校验（移除所有 CUDA 相关校验）
    assert mixes.dim() == 3, f"mixes must be 3D, got {mixes.dim()}D"
    assert hc_scale.shape == (3,), f"hc_scale must be [3], got {hc_scale.shape}"
    assert hc_base.shape == ((2 + hc_mult) * hc_mult,), \
        f"hc_base must be [{(2 + hc_mult) * hc_mult}], got {hc_base.shape}"
    assert eps > 0, "eps must be positive to avoid division by zero"
    assert sinkhorn_iters >= 1, "sinkhorn_iters must be ≥1"
    assert hc_mult == 4, "当前Triton实现仅支持hc_mult=4"

    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)

    # 1. 维度计算与展平（对齐 PyTorch 标杆的展平逻辑）
    b, s, _ = mixes.shape
    feat_dim = (2 + hc_mult) * hc_mult
    hc_square = hc_mult * hc_mult
    batch_seq_size = b * s  # 合并 batch 和 seq 维度

    # 2. 展平输入张量（[b, s, feat_dim] → [b*s, feat_dim]）
    mixes_flat = mixes.view(-1, feat_dim).contiguous()

    # 3. 创建输出张量（使用输入张量的设备和类型，不强制 CUDA）
    pre_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    comb_flat = torch.empty((batch_seq_size, hc_square), dtype=mixes.dtype, device=mixes.device)

    # 4. 配置 Triton 核函数参数
    BLOCK_ALIGN = 8

    # 5. 启动 Triton 核函数
    _hc_split_sinkhorn_kernel_part1[(batch_seq_size, 1)](
        # 输入输出指针
        mixes_flat, hc_scale, hc_base,
        pre_flat, post_flat,
        # 维度参数
        batch_seq_size, hc_mult, sinkhorn_iters,
        # 常量参数
        eps, feat_dim, hc_square,
        # 块大小
        BLOCK_HC=hc_mult,
    )

    mixes_flat_slice = mixes_flat[:, 2 * hc_mult:]
    mixes_flat_slice = mixes_flat_slice.view(-1, hc_mult, hc_mult)
    # print('mixes_flat', mixes_flat.shape)
    pad_num = BLOCK_ALIGN - hc_mult
    # print('mixes_flat', mixes_flat[0])
    mixes_flat_padded = F.pad(mixes_flat_slice, pad=(0, pad_num), mode="constant", value=0.0)
    hc_base_slice = hc_base[2 * hc_mult:].view(hc_mult, hc_mult)
    hc_base_padded = F.pad(hc_base_slice, pad=(0, pad_num), mode="constant", value=0.0)

    # print('mixes_flat_padded', mixes_flat_padded.shape)
    comb_flat_padded = torch.empty((batch_seq_size, hc_mult*BLOCK_ALIGN), dtype=mixes.dtype, device=mixes.device)
    # 5. 启动 Triton 核函数
    _hc_split_sinkhorn_kernel_part2[(batch_seq_size, 1)](
        # 输入输出指针
        mixes_flat_padded, hc_scale, hc_base_padded,
        comb_flat_padded,
        # 维度参数
        batch_seq_size, hc_mult, sinkhorn_iters,
        # 常量参数
        eps,
        # 块大小
        BLOCK_HC=hc_mult,
        BLOCK_ALIGN=BLOCK_ALIGN
    )

    # 6. 重塑输出张量（展平的逆操作）
    pre = pre_flat.view(b, s, hc_mult)
    post = post_flat.view(b, s, hc_mult)
    comb = comb_flat_padded.view(b, s, hc_mult, BLOCK_ALIGN)
    # print('comb 过程中', comb)
    comb = comb[:,:,:,:hc_mult]

    pre = pre.to(dtype=origin_dtype)
    post = post.to(dtype=origin_dtype)
    comb = comb.to(dtype=origin_dtype)

    return pre, post, comb


@triton.jit
def hc_split_sinkhorn_backward_kernel_part1(
    # 输入梯度指针
    grad_pre_ptr,
    grad_post_ptr,
    grad_comb_ptr,
    # 前向输入指针
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    # 输出梯度指针
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    # 常量参数
    batch_seq_size: tl.constexpr,
    hc_mult: tl.constexpr = 4,
    sinkhorn_iters: tl.constexpr = 20,
    eps: tl.constexpr = 1e-6,
    total_dim: tl.constexpr = (2+4)*4,  # 可通过 (2 + hc_mult) * hc_mult 动态计算
    BLOCK_ALIGN: tl.constexpr = 8,
):
    # 单个线程块处理一个 (b*s) 样本
    idx = tl.program_id(0)
    if idx >= batch_seq_size:
        return
    
    # 定义常量：所有维度完全基于 hc_mult 动态计算（核心修改）
    EPS = eps
    # 移除硬编码的 PRE_DIM/POST_DIM/COMB_DIM，直接使用 hc_mult 计算
    pre_start = 0
    pre_end = hc_mult  # pre切片：0 ~ hc_mult
    post_start = hc_mult
    post_end = 2 * hc_mult  # post切片：hc_mult ~ 2*hc_mult
    comb_start = 2 * hc_mult
    comb_end = (2 + hc_mult) * hc_mult  # comb切片：2*hc_mult ~ (2+hc_mult)*hc_mult
    comb_flat_dim = hc_mult * hc_mult  # comb展平维度：hc_mult*hc_mult
    mixes_offset = idx * comb_end
    # ------------------- 1. Pre 部分梯度计算（用 hc_mult 替代 PRE_DIM） -------------------
    # 加载pre切片：idx*total_dim + [0, hc_mult)
    pre_slice = tl.load(mixes_ptr + idx * total_dim + tl.arange(0, hc_mult))  # [hc_mult,]
    scale_0 = tl.load(hc_scale_ptr + 0)
    base_pre = tl.load(hc_base_ptr + tl.arange(0, hc_mult))  # [hc_mult,]
    
    pre_input = pre_slice * scale_0 + base_pre
    sigmoid_pre = tl.sigmoid(pre_input)
    
    grad_pre = tl.load(grad_pre_ptr + idx * hc_mult + tl.arange(0, hc_mult))  # [hc_mult,]
    sigmoid_deriv = sigmoid_pre * (1.0 - sigmoid_pre)
    grad_pre_input = grad_pre * sigmoid_deriv
    
    # 更新梯度：pre切片范围 [0, hc_mult)
    tl.store(grad_mixes_ptr + idx * total_dim + tl.arange(0, hc_mult), 
             grad_pre_input * scale_0)
    tl.atomic_add(grad_hc_scale_ptr + 0, tl.sum(grad_pre_input * pre_slice))
    tl.atomic_add(grad_hc_base_ptr + tl.arange(0, hc_mult), grad_pre_input)

    # ------------------- 2. Post 部分梯度计算（用 hc_mult 替代 POST_DIM） -------------------
    # 加载post切片：idx*total_dim + [hc_mult, 2*hc_mult)
    post_slice = tl.load(mixes_ptr + idx * total_dim + hc_mult + tl.arange(0, hc_mult))  # [hc_mult,]
    scale_1 = tl.load(hc_scale_ptr + 1)
    base_post = tl.load(hc_base_ptr + hc_mult + tl.arange(0, hc_mult))  # [hc_mult,]
    
    post_input = post_slice * scale_1 + base_post
    sigmoid_post = tl.sigmoid(post_input)
    
    grad_post = tl.load(grad_post_ptr + idx * hc_mult + tl.arange(0, hc_mult))  # [hc_mult,]
    sigmoid_deriv_post = sigmoid_post * (1.0 - sigmoid_post)
    grad_post_input = grad_post * 2.0 * sigmoid_deriv_post
    
    # 更新梯度：post切片范围 [hc_mult, 2*hc_mult)
    tl.store(grad_mixes_ptr + idx * total_dim + hc_mult + tl.arange(0, hc_mult), 
             grad_post_input * scale_1)
    tl.atomic_add(grad_hc_scale_ptr + 1, tl.sum(grad_post_input * post_slice))
    tl.atomic_add(grad_hc_base_ptr + hc_mult + tl.arange(0, hc_mult), grad_post_input)


@triton.jit
def hc_split_sinkhorn_backward_kernel_part2(
    # 输入梯度指针
    grad_pre_ptr,
    grad_post_ptr,
    grad_comb_ptr,
    # 前向输入指针
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    # 输出梯度指针
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    # 常量参数
    batch_seq_size: tl.constexpr,
    hc_mult: tl.constexpr = 4,
    sinkhorn_iters: tl.constexpr = 20,
    eps: tl.constexpr = 1e-6,
    total_dim: tl.constexpr = (2+4)*4,  # 可通过 (2 + hc_mult) * hc_mult 动态计算
    BLOCK_ALIGN: tl.constexpr = 8,
):
    # 单个线程块处理一个 (b*s) 样本
    idx = tl.program_id(0)
    
    # 定义常量：所有维度完全基于 hc_mult 动态计算（核心修改）
    EPS = eps
    # 移除硬编码的 PRE_DIM/POST_DIM/COMB_DIM，直接使用 hc_mult 计算
    pre_start = 0
    pre_end = hc_mult  # pre切片：0 ~ hc_mult
    post_start = hc_mult
    post_end = 2 * hc_mult  # post切片：hc_mult ~ 2*hc_mult
    comb_start = 2 * hc_mult
    comb_end = (2 + hc_mult) * hc_mult  # comb切片：2*hc_mult ~ (2+hc_mult)*hc_mult
    comb_flat_dim = hc_mult * hc_mult  # comb展平维度：hc_mult*hc_mult
    mixes_offset = idx * comb_end

    total_dim = hc_mult * BLOCK_ALIGN
    # ------------------- 3. Comb 部分梯度计算（用 hc_mult 替代 COMB_DIM） -------------------
    # 3.1 加载comb切片：idx*total_dim + [2*hc_mult, (2+hc_mult)*hc_mult)
    comb_slice_flat = tl.load(mixes_ptr + idx * total_dim + tl.arange(0, hc_mult * BLOCK_ALIGN))  # [hc_mult*hc_mult,]
    comb_slice = comb_slice_flat.reshape(hc_mult, BLOCK_ALIGN)  # [hc_mult, hc_mult]
    
    scale_2 = tl.load(hc_scale_ptr + 2)
    base_comb_flat = tl.load(hc_base_ptr + tl.arange(0, hc_mult * BLOCK_ALIGN))  # [hc_mult*hc_mult,]
    base_comb = base_comb_flat.reshape(hc_mult, BLOCK_ALIGN)  # [hc_mult, hc_mult]
    
    # 3.2 前向计算：无 keepdim 适配，维度基于 hc_mult
    comb_init = comb_slice * scale_2 + base_comb  # [hc_mult, hc_mult]

    # 行内减最大值：手动 reshape 模拟 keepdim=True
    row_max_compressed = tl.max(comb_init, axis=1)  # [hc_mult,]
    row_max = row_max_compressed.reshape(hc_mult, 1)  # [hc_mult, 1]
    exp_comb = tl.exp(comb_init - row_max)  # [hc_mult, hc_mult]

    mask1 = tl.arange(0, hc_mult*BLOCK_ALIGN) % BLOCK_ALIGN < hc_mult

    comb = tl.reshape(exp_comb, (hc_mult * BLOCK_ALIGN))
    comb = tl.where(mask1, comb, 0.0)
    exp_comb = tl.reshape(comb, (hc_mult, BLOCK_ALIGN))

    # 初始化K，仅保存 row_sum/col_sum（维度基于 hc_mult）
    K = exp_comb
    row_sum_list = tl.full((sinkhorn_iters, hc_mult, 1), 0.0, dtype=tl.float32)  # [sinkhorn_iters, hc_mult, 1]
    col_sum_list = tl.full((sinkhorn_iters, 1, BLOCK_ALIGN), 0.0, dtype=tl.float32)  # [sinkhorn_iters, 1, hc_mult]

    # 前向Sinkhorn迭代：维度完全基于 hc_mult
    for i in range(sinkhorn_iters):
        # 行归一化：sum 无 keepdim → 手动 reshape
        row_sum_compressed = tl.sum(K, axis=1)  # [hc_mult,]
        row_sum = row_sum_compressed.reshape(hc_mult, 1)  # [hc_mult, 1]
        K_row = K / (row_sum + EPS)
        
        # 列归一化：sum 无 keepdim → 手动 reshape
        col_sum_compressed = tl.sum(K_row, axis=0)  # [hc_mult,]
        col_sum = col_sum_compressed.reshape(1, BLOCK_ALIGN)  # [1, hc_mult]
        K_col = K_row / (col_sum + EPS)
        
        # 保存 row_sum/col_sum
        row_sum_list = tl.insert_slice(
            ful = row_sum_list,
            sub = row_sum[None,:,:],
            offsets = [i,0,0],
            sizes = [1,hc_mult,1],
            strides = [1,1,1],
        )
        col_sum_list = tl.insert_slice(
            ful = col_sum_list,
            sub = col_sum[None,:,:],
            offsets = [i,0,0],
            sizes = [1,1,BLOCK_ALIGN],
            strides = [1,1,1],
        )
        
        K = K_col

    # 3.3 反向传播：实时复原 K_col/K_row，维度基于 hc_mult
    grad_comb_flat = tl.load(grad_comb_ptr + idx * hc_mult * BLOCK_ALIGN + tl.arange(0, hc_mult * BLOCK_ALIGN))  # [hc_mult*hc_mult,]
    dK = grad_comb_flat.reshape(hc_mult, BLOCK_ALIGN)  # [hc_mult, hc_mult]

    # 反向遍历迭代步骤
    for j in range(sinkhorn_iters):
        i = sinkhorn_iters - j - 1

        # row_sum = row_sum_list[i]  # [hc_mult, 1]
        # col_sum = col_sum_list[i]  # [1, hc_mult]
        row_sum = tl.extract_slice(
            row_sum_list,
            [i,0,0],
            [1,hc_mult,1],
            [1,1,1],
        )
        col_sum = tl.extract_slice(
            col_sum_list,
            [i,0,0],
            [1,1,BLOCK_ALIGN],
            [1,1,1],
        )

        row_sum = row_sum.reshape(hc_mult, 1)
        col_sum = col_sum.reshape(1, BLOCK_ALIGN)

        # 步骤1：复原 K_col = K * (col_sum + EPS)
        K_col = K * (col_sum + EPS)

        # 步骤2：反向传播列归一化（无 keepdim 适配）
        grad_direct = dK / (col_sum + EPS)
        d_col_sum_compressed = - tl.sum(dK * K_col / ((col_sum + EPS) * (col_sum + EPS)), axis=-2)  # [hc_mult,]
        d_col_sum = d_col_sum_compressed.reshape(1, BLOCK_ALIGN)  # [1, hc_mult]
        dK_row = grad_direct + d_col_sum
        
        # 步骤3：复原 K_row = K_col * (row_sum + EPS)
        K_row = K_col * (row_sum + EPS)
        
        # 步骤4：更新K为K_row
        K = K_row
        
        # 步骤5：反向传播行归一化（无 keepdim 适配）
        grad_direct_row = dK_row / (row_sum + EPS)
        d_row_sum_compressed = - tl.sum(dK_row * K_row / ((row_sum + EPS) * (row_sum + EPS)), axis=-1)  # [hc_mult,]
        d_row_sum = d_row_sum_compressed.reshape(hc_mult, 1)  # [hc_mult, 1]
        dK = grad_direct_row + d_row_sum
        tmp = tl.reshape(dK, (hc_mult * BLOCK_ALIGN))
        tmp = tl.where(mask1, tmp, 0.0)
        dK = tl.reshape(tmp, (hc_mult, BLOCK_ALIGN))

    # 3.4 反向传播：指数 + 减行最大值（维度基于 hc_mult）
    d_exp_comb = dK
    d_comb_before_exp = d_exp_comb * exp_comb  # [hc_mult, hc_mult]
    
    # 处理减最大值的梯度
    max_mask = tl.where(comb_init == row_max, 1.0, 0.0)  # [hc_mult, hc_mult]
    max_count_compressed = tl.sum(max_mask, axis=-1)  # [hc_mult,]
    max_count = max_count_compressed.reshape(hc_mult, 1) + EPS  # [hc_mult, 1]
    
    row_sum_d_before_exp_compressed = tl.sum(d_comb_before_exp, axis=-1)  # [hc_mult,]
    row_sum_d_before_exp = row_sum_d_before_exp_compressed.reshape(hc_mult, 1)  # [hc_mult, 1]
    
    d_comb_init = d_comb_before_exp - (row_sum_d_before_exp * max_mask / max_count)  # [hc_mult, hc_mult]
    
    # 3.5 反向传播线性变换
    grad_comb_slice = d_comb_init * scale_2  # [hc_mult, hc_mult]
    grad_comb_slice_flat = grad_comb_slice.reshape(hc_mult * BLOCK_ALIGN)  # [hc_mult*hc_mult,]
    
    # 更新梯度：comb切片范围 [2*hc_mult, (2+hc_mult)*hc_mult)
    tl.store(grad_mixes_ptr + idx * hc_mult * BLOCK_ALIGN + tl.arange(0, hc_mult * BLOCK_ALIGN), grad_comb_slice_flat)
    tl.atomic_add(grad_hc_scale_ptr + 2, tl.sum(d_comb_init * comb_slice))
    # 逐元素原子加（维度基于 hc_mult）
    # for i in range(hc_mult * hc_mult):
    tl.atomic_add(grad_hc_base_ptr + tl.arange(0, hc_mult * BLOCK_ALIGN), d_comb_init.reshape(hc_mult * BLOCK_ALIGN))


def hc_split_sinkhorn_triton_backward(
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. 输入校验
    assert hc_mult == 4, "当前Triton实现仅支持hc_mult=4"
    b, s, total_dim = mixes.shape
    batch_seq_size = b * s

    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)

    grad_pre = grad_pre.to(dtype=torch.float32)
    grad_post = grad_post.to(dtype=torch.float32)
    grad_comb = grad_comb.to(dtype=torch.float32)

    # 2. 初始化输出梯度（全0）
    grad_mixes = torch.zeros_like(mixes, device=mixes.device)
    grad_hc_scale = torch.zeros_like(hc_scale, device=hc_scale.device)
    grad_hc_base = torch.zeros_like(hc_base, device=hc_base.device)
    
    row_sum_list = torch.zeros((sinkhorn_iters, hc_mult, 1), device=mixes.device)  # [sinkhorn_iters, hc_mult, 1]
    col_sum_list = torch.zeros((sinkhorn_iters, 1, hc_mult), device=mixes.device)  # [sinkhorn_iters, 1, hc_mult]

    # 3. 展平输入梯度（适配Triton核函数的二维输入）
    grad_pre_flat = grad_pre.reshape(-1, hc_mult)
    grad_post_flat = grad_post.reshape(-1, hc_mult)
    grad_comb_flat = grad_comb.reshape(-1, hc_mult*hc_mult)

    # 4. 配置Triton网格：每个线程块处理一个 (b*s) 样本
    grid = (batch_seq_size,)
    
    # 5. 启动Triton核函数
    hc_split_sinkhorn_backward_kernel_part1[grid](
        # 输入梯度指针
        grad_pre_flat,
        grad_post_flat,
        grad_comb_flat,
        # 前向输入指针
        mixes,
        hc_scale,
        hc_base,
        # 输出梯度指针
        grad_mixes,
        grad_hc_scale,
        grad_hc_base,
        # 常量参数
        batch_seq_size=batch_seq_size,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        total_dim=total_dim
    )

    mixes = mixes.view(-1, (2 + hc_mult) * hc_mult)
    mixes_slice = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)
    hc_base = hc_base.view((2 + hc_mult) * hc_mult)
    hc_base_slice = hc_base[2 * hc_mult:].view(hc_mult, hc_mult)

    mixes_pad = F.pad(mixes_slice, (0, 8 - hc_mult), mode='constant', value=0)
    hc_base_pad = F.pad(hc_base_slice, (0, 8 - hc_mult), mode='constant', value=0)

    grad_mixes_pad = torch.zeros((b, s, hc_mult, 8), dtype=grad_mixes.dtype, device=grad_mixes.device)
    grad_hc_base_pad = torch.zeros((hc_mult, 8), dtype=grad_hc_base.dtype, device=grad_hc_base.device)

    grad_comb_flat = grad_comb_flat.view(-1, hc_mult, hc_mult)

    grad_comb_flat_pad = F.pad(grad_comb_flat, (0, 8 - hc_mult), mode='constant', value=0)

    hc_split_sinkhorn_backward_kernel_part2[grid](
        # 输入梯度指针
        grad_pre_flat,
        grad_post_flat,
        grad_comb_flat_pad,
        # 前向输入指针
        mixes_pad,
        hc_scale,
        hc_base_pad,
        # 输出梯度指针
        grad_mixes_pad,
        grad_hc_scale,
        grad_hc_base_pad,
        # 常量参数
        batch_seq_size=batch_seq_size,
        hc_mult=4,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        total_dim=total_dim
    )

    grad_mixes_slice = grad_mixes_pad[:,:,:,:hc_mult]
    grad_hc_base_slice = grad_hc_base_pad[:,:hc_mult]

    grad_mixes_slice = grad_mixes_slice.reshape(b, s, hc_mult * hc_mult)
    grad_hc_base_slice = grad_hc_base_slice.reshape(hc_mult * hc_mult)

    grad_mixes[:,:,2*hc_mult:] = grad_mixes_slice
    grad_hc_base[2*hc_mult:] = grad_hc_base_slice

    grad_mixes = grad_mixes.to(dtype=origin_dtype)
    grad_hc_scale = grad_hc_scale.to(dtype=origin_dtype)
    grad_hc_base = grad_hc_base.to(dtype=origin_dtype)

    return grad_mixes, grad_hc_scale, grad_hc_base


if __name__ == "__main__":
    torch.manual_seed(42)

    b = 1
    s = 4096

    hc_mult = 4
    hc_sinkhorn_iters = 20
    hc_eps = 1e-6

    mix_hc = (2 + hc_mult) * hc_mult # hc_mult + hc_mult + hc_mult * hc_mult

    mixes = torch.randn(b, s, mix_hc, device="cpu", dtype=torch.float32).npu()
    hc_scale = torch.randn(3, device="cpu", dtype=torch.float32).npu()
    hc_base = torch.randn(mix_hc, device="cpu", dtype=torch.float32).npu()

    mixes.requires_grad = True
    hc_scale.requires_grad = True
    hc_base.requires_grad = True

    print('------------输入描述------------')
    print('mixes.shape', mixes.shape)
    print('hc_scale.shape', hc_scale.shape)
    print('hc_base.shape', hc_base.shape)
    print('hc_mult', hc_mult)
    print('hc_sinkhorn_iters', hc_sinkhorn_iters)
    print('hc_eps', hc_eps)

    # 前向传播
    pre_ref, post_ref, comb_ref = hc_split_sinkhorn_torch_reference(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
    pre_triton, post_triton, comb_triton = hc_split_sinkhorn_triton(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        record_shapes=True,#采集torch op的input shape和input type的开关
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./forward_part"),
    ) as prof:

        pre_ref, post_ref, comb_ref = hc_split_sinkhorn_torch_reference(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
        pre_triton, post_triton, comb_triton = hc_split_sinkhorn_triton(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps)
        prof.step()

    # 比较前向
    print("------------前向： torch小算子 vs triton算子------------")
    print('pre_ref diff', torch.max(torch.abs(pre_ref - pre_triton)))
    print('post_ref diff',torch.max(torch.abs(post_ref - post_triton)))
    print('comb_ref diff', torch.max(torch.abs(comb_ref - comb_triton)))

    # 随机梯度
    grad_pre = torch.randn_like(pre_ref)
    grad_post = torch.randn_like(post_ref)
    grad_comb = torch.randn_like(comb_ref)

    # 反向传播
    grad_mixes_ref, grad_scale_ref, grad_base_ref = hc_split_sinkhorn_torch_reference_backward(
        grad_pre, grad_post, grad_comb,
        mixes.detach(), hc_scale.detach(), hc_base.detach(),
        hc_mult, hc_sinkhorn_iters
    )
    grad_mixes_triton, grad_scale_triton, grad_base_triton = hc_split_sinkhorn_triton_backward(
        grad_pre, grad_post, grad_comb,
        mixes.detach(), hc_scale.detach(), hc_base.detach(),
        hc_mult, hc_sinkhorn_iters
    )

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        experimental_config = torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        record_shapes=True,#采集torch op的input shape和input type的开关
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./backward_part"),
    ) as prof:
        grad_mixes_ref, grad_scale_ref, grad_base_ref = hc_split_sinkhorn_torch_reference_backward(
            grad_pre, grad_post, grad_comb,
            mixes.detach(), hc_scale.detach(), hc_base.detach(),
            hc_mult, hc_sinkhorn_iters
        )
        grad_mixes_triton, grad_scale_triton, grad_base_triton = hc_split_sinkhorn_triton_backward(
            grad_pre, grad_post, grad_comb,
            mixes.detach(), hc_scale.detach(), hc_base.detach(),
            hc_mult, hc_sinkhorn_iters
        )
        prof.step()


    # # print("------------反向： torch小算子 vs triton算子")
    # # print("mixes grad diff:", torch.max(grad_mixes_ref - grad_mixes_triton))
    # # print("scale grad diff:", torch.max(grad_scale_ref - grad_scale_triton))
    # # print("base grad diff:", torch.max(grad_base_ref - grad_base_triton))

    # 自动微分计算梯度
    loss = (pre_ref * grad_pre).sum() + (post_ref * grad_post).sum() + (comb_ref * grad_comb).sum()
    loss.backward()
    print("------------反向： 自动微分 vs triton算子------------")
    print("mixes grad diff:", torch.max(torch.abs(mixes.grad - grad_mixes_triton)))
    print("scale grad diff:", torch.max(torch.abs(hc_scale.grad - grad_scale_triton)))
    print("base grad diff:", torch.max(torch.abs(hc_base.grad - grad_base_triton)))


