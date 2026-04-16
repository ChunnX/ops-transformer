import torch, torch_npu
import torch.nn.functional as F

def mhc_pre_torch(x: torch.tensor, hc_fn: torch.tensor, hc_scale: torch.tensor,
    hc_base: torch.tensor, sinkhorn_iters: int, hc_mult: int, hc_eps: float,
    norm_eps: float):
    """
    input:
        x: torch.tensor, bfloat16, [bs, seq_len, n, c]
        hc_fn: torch.tensor, float32, [n^2 + 2*n, n*c]
        hc_scale: torch.tensor, float32, [3]
        hc_base: torch.tensor, float32, [n^2 + 2*n]
        sinkhorn_iters: scalar, sinkhorn算法迭代次数
        hc_mult: scalar, n = 4
        hc_eps: scalar, 防止除零小常数
        norm_eps: scalar, RMSNorm中使用
    output:
        y: torch.tensor bfloat16, [bs, seq_len, c]
        h_post: torch.tensor float32, [bs, seq_len, n]
        h_res: torch.tensor bfloat32, [bs, seq_len, n, n]
    """

    shape, dtype = x.size(), x.dtype
    x_fp32 = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + norm_eps)
    h_hat2 = F.linear(x_fp32, hc_fn)
    mixes = h_hat2 * rsqrt
    pre, post, comb, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm = hc_split_sinkhorn_torch_reference(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * x_fp32.view(shape), dim=2)
    return y.to(dtype), pre, post, comb, h_hat2, rsqrt, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm

def hc_split_sinkhorn_torch_reference(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
):
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
    # for _ in range(sinkhorn_iters):
    #     # 6.1 行归一化
    #     row_sum = comb_flat.sum(dim=-1, keepdim=True)
    #     comb_flat = comb_flat / (row_sum + eps)
    #     # 6.2 列归一化
    #     col_sum = comb_flat.sum(dim=-2, keepdim=True)
    #     comb_flat = comb_flat / (col_sum + eps)
    
    # return pre, post, comb

    sk_colSum = []
    sk_colNorm = []
    sk_rowSum = []
    sk_rowNorm = []

    for _ in range(sinkhorn_iters):
        # 6.1 行归一化
        row_sum = comb_flat.sum(dim=-1, keepdim=True)
        comb_flat = comb_flat / (row_sum + eps)
        sk_rowNorm.append(comb_flat)
        # 6.2 列归一化
        col_sum = comb_flat.sum(dim=-2, keepdim=True)
        comb_flat = comb_flat / (col_sum + eps)
        
        sk_colSum.append(col_sum)
        sk_rowSum.append(row_sum)
        sk_colNorm.append(comb_flat)

    # 7. 重塑为原始形状 [b, s, ...]
    pre = pre_flat.view(b, s, hc_mult)
    post = post_flat.view(b, s, hc_mult)
    comb = comb_flat.view(b, s, hc_mult, hc_mult)

    return pre, post, comb, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm

def RMSNorm(x, norm_eps):
    x_norm = (x.flatten(2).square().mean(-1, keepdim=True) + norm_eps).sqrt()
    rsqrt = 1.0 / x_norm
    return rsqrt

def RMSNormGrad(x, rsqrt, grad_rsqrt):
    _, _, NC = x.shape
    grad_x = (- (rsqrt ** 3) * x.flatten(2)) * grad_rsqrt / float(NC)
    return grad_x.view(x.shape)

def SigmoidGrad(x, grad_output):
    # x_sigmoid = x.sigmoid()
    x_sigmoid = 1 / (1 + (-x).exp())
    return x_sigmoid * (1 - x_sigmoid) * grad_output

# def Sinkhorn(x, sinkhorn_iters, eps):
#     saved_x = []
#     x = forward_func(x)
#     for _ in range(sinkhorn_iters):
#         saved_x.append(x.clone())
#         # 6.1 行归一化
#         row_sum = x.sum(dim=-1, keepdim=True)
#         x = x / (row_sum + eps)
#         # 6.2 列归一化
#         col_sum = x.sum(dim=-2, keepdim=True)
#         x = x / (col_sum + eps)
#     return x, saved_x

# def SinkhornGrad(x, grad_output, sinkhorn_iters, eps):
#     _, total_saved_x = Sinkhorn(x, sinkhorn_iters, eps)

#     x_grad = grad_output

#     for saved_x in reversed(total_saved_x):
#         row_sum = saved_x.sum(dim=-1, keepdim=True)
#         x_row_normed = saved_x / (row_sum + eps)
#         col_sum = x_row_normed.sum(dim=-2, keepdim=True)
#         grad_x_row_normed = x_grad / (col_sum + eps) - (x_grad * x_row_normed / (col_sum + eps) ** 2).sum(dim=-2, keepdim=True)
#         x_grad = grad_x_row_normed / (row_sum + eps) - (grad_x_row_normed * saved_x / (row_sum + eps) ** 2).sum(dim=-1, keepdim=True)

#     return x_grad

def Sinkhorn(x, sinkhorn_iters, eps):
    sk_colSum = []
    sk_colNorm = []
    sk_rowSum = []
    sk_rowNorm = []

    x = forward_func(x)

    for _ in range(sinkhorn_iters):
        # 6.1 行归一化
        row_sum = x.sum(dim=-1, keepdim=True)
        x = x / (row_sum + eps)
        sk_rowNorm.append(x)
        # 6.2 列归一化
        col_sum = x.sum(dim=-2, keepdim=True)
        x = x / (col_sum + eps)
        
        sk_colSum.append(col_sum)
        sk_rowSum.append(row_sum)
        sk_colNorm.append(x)

    return x, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm

def SinkhornGrad(x, grad_output, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm, sinkhorn_iters, eps):
    # _, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm = Sinkhorn(x, sinkhorn_iters, eps)
    bs, seq_len, n, _ = x.shape
    x_grad = grad_output

    for col_sum, x_col_normed, row_sum, x_row_normed in reversed(list(zip(sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm))):

        col_sum = col_sum.view(bs, seq_len, 1, n)
        x_col_normed = x_col_normed.view(bs, seq_len, n, n)
        row_sum = row_sum.view(bs, seq_len, n, 1)
        x_row_normed = x_row_normed.view(bs, seq_len, n, n)

        grad_x_row_normed = x_grad / (col_sum + eps) - (x_grad * x_col_normed / (col_sum + eps) ** 2).sum(dim=-2, keepdim=True)
        x_grad = grad_x_row_normed / (row_sum + eps) - (grad_x_row_normed * (x_row_normed * (row_sum + eps)) / (row_sum + eps) ** 2).sum(dim=-1, keepdim=True)
        
    return x_grad

def forward_func(x):
    y = (x - x.max(dim=-1, keepdim=True)[0]).exp()
    return y

def grad_func(x, y_grad):
    x_max, idx = x.max(dim=-1, keepdim=True)
    arange_idx = torch.arange(4).to(x.device).int()[None, :]
    mask = idx != arange_idx
    y = (x - x_max).exp()
    grad_max = -(y * mask * y_grad).sum(dim=-1, keepdim=True)
    func_grad = torch.where(mask == 1, y * y_grad, grad_max)
    return func_grad

def mhc_pre_grad(x: torch.tensor, hc_fn: torch.tensor, pre: torch.tensor, grad_y: torch.tensor, grad_post: torch.tensor, grad_comb: torch.tensor,
    hc_scale: torch.tensor, hc_base: torch.tensor, h_hat2: torch.tensor, rsqrt: torch.tensor, sk_colSum: torch.tensor, sk_colNorm: torch.tensor,
    sk_rowSum: torch.tensor, sk_rowNorm: torch.tensor, sinkhorn_iters: int, hc_mult: int, hc_eps: float, norm_eps: float):
    """
    input:
        x: torch.tensor, bfloat16, [bs, seq_len, n, c]
        hc_fn: torch.tensor, float32, [n^2 + 2*n, n*c]
        h_pre: torch.tensor float32, [bs, seq_len, n]
        grad_y: torch.tensor bfloat16, [bs, seq_len, c]
        grad_post: torch.tensor float32, [bs, seq_len, n]
        grad_comb: torch.tensor float32, [bs, seq_len, n, n]
        hc_scale: torch.tensor, float32, [3]
        hc_base: torch.tensor, float32, [n^2 + 2*n]
        h_hat2: torch.tensor, float32, [bs, seq_len, n^2 + 2*n]
        rsqrt: torch.tensor, float32, [bs, seq_len, 1]
        u: torch.tensor, float32, [bs, seq_len, n]
        v: torch.tensor, float32, [bs, seq_len, n]

        sinkhorn_iters: scalar, sinkhorn算法迭代次数
        hc_mult: scalar, n = 4
        hc_eps: scalar, 防止除零小常数
        norm_eps: scalar, RMSNorm中使用
    output:
        grad_x: torch.tensor, bfloat16, [bs, seq_len, n, c]
        grad_hc_fn: torch.tensor, float32, [n^2 + 2*n, n*c]
        grad_hc_scale: torch.tensor, float32, [3]
        grad_hc_base: torch.tensor, float32, [n^2 + 2*n]
    """
    x_shape = x.shape
    grad_y_fp32 = grad_y.float()
    x_fp32 = x.float().clone()
    
    b, s, n, c = x.shape
    # compute grad_x_3
    grad_x_3 = grad_y_fp32[..., None, :] * pre[..., None]    # [bs, seq_len, n, c]
    # compute pre_grad
    grad_pre = (grad_y_fp32[..., None, :] * x_fp32).sum(-1)

    # h_hat2 -> h_hat
    h_hat2_rsqrt = h_hat2 * rsqrt
    h_hat_pre = h_hat2_rsqrt[..., :hc_mult] * hc_scale[0] + hc_base[..., :hc_mult]
    h_hat_post = h_hat2_rsqrt[..., hc_mult:(hc_mult *2)] * hc_scale[1] + hc_base[..., hc_mult:(hc_mult *2)]
    h_hat_res = (h_hat2_rsqrt[..., (hc_mult *2):] * hc_scale[2] + hc_base[..., (hc_mult *2):]).reshape(x_shape[0], x_shape[1], hc_mult, hc_mult)
    
    grad_h_hat_pre = SigmoidGrad(h_hat_pre, grad_pre)
    grad_h_hat_post = 2 * SigmoidGrad(h_hat_post, grad_post)

    grad_h_hat_res = grad_func(h_hat_res, SinkhornGrad(h_hat_res, grad_comb, sk_colSum, sk_colNorm, sk_rowSum, sk_rowNorm, sinkhorn_iters, hc_eps)).flatten(2)
    grad_hc_base = torch.cat([grad_h_hat_pre.sum((0, 1)), grad_h_hat_post.sum((0, 1)), grad_h_hat_res.sum((0, 1)).flatten()])

    # print("hcbase")
    # print(grad_h_hat_pre)
    # print(grad_h_hat_post)

    grad_hc_scale_pre = (h_hat2_rsqrt[..., :hc_mult] * grad_h_hat_pre).sum()
    grad_hc_scale_post = (h_hat2_rsqrt[..., hc_mult:(hc_mult *2)] * grad_h_hat_post).sum()
    grad_hc_scale_res = (h_hat2_rsqrt[..., (hc_mult *2):] * grad_h_hat_res).sum()
    grad_hc_scale = torch.stack([grad_hc_scale_pre, grad_hc_scale_post, grad_hc_scale_res])
    
    # print("hcscale")
    # print(grad_hc_scale_pre)
    # print(grad_hc_scale_post)
    # print(h_hat2_rsqrt[..., :hc_mult] * grad_h_hat_pre)
    # print(h_hat2_rsqrt[..., hc_mult:(hc_mult *2)] * grad_h_hat_post)

    grad_h_hat2_pre = grad_h_hat_pre * hc_scale[0]
    grad_h_hat2_post = grad_h_hat_post * hc_scale[1]
    grad_h_hat2_res = grad_h_hat_res * hc_scale[2]

    grad_h_hat2 = torch.cat([grad_h_hat2_pre, grad_h_hat2_post, grad_h_hat2_res], dim=-1)

    grad_rsqrt = (h_hat2 * grad_h_hat2).sum(-1, keepdim=True)

    # print(grad_rsqrt)
    # print(h_hat2 * grad_h_hat2)
    # print(h_hat2)
    # print((h_hat2 * grad_h_hat2)[..., :8].sum(-1))
    # print((h_hat2 * grad_h_hat2)[..., 8:].sum(-1))

    grad_x_1 = RMSNormGrad(x_fp32.flatten(2), rsqrt, grad_rsqrt)

    grad_h_hat2 = grad_h_hat2 * rsqrt
    grad_x_2 = grad_h_hat2[..., None, :] @ hc_fn
    grad_x = grad_x_3.view(x_shape) + grad_x_1.view(x_shape) + grad_x_2.view(x_shape)
    # grad_x = grad_x_1.view(x_shape) + grad_x_3.view(x_shape)
    # grad_x = grad_x_3.view(x_shape)

    grad_hc_fn = grad_h_hat2.flatten(0, 1).T @ x_fp32.flatten(2).flatten(0, 1)

    debug_info = {
        "grad_x_3": grad_x_3.to(torch.bfloat16),
        "grad_x_1": grad_x_1.to(torch.bfloat16), 
        "grad_x_1 + grad_x_3": (grad_x_3.flatten() + grad_x_1.flatten()).to(torch.bfloat16),
        "grad_pre": grad_pre,
        "grad_h_hat2_pre": grad_h_hat2_pre,
        "grad_h_hat2_post": grad_h_hat2_post
    }

    return grad_x.bfloat16(), grad_hc_fn, grad_hc_scale, grad_hc_base, debug_info
    

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
):
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