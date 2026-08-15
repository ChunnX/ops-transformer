import torch


def _rotate_half(x):
    """构造旋转项: cat(-x_2, x_1, dim=-1)，其中 x_1, x_2 = chunk(x, 2, dim=-1)"""
    x_1, x_2 = x.chunk(2, dim=-1)
    return torch.cat([-x_2, x_1], dim=-1)


def _apply_rotary_pos_emb_forward(query, key, cos, sin):
    """前向传播（保存反向需要的中间变量）

    参数:
        query: [S, B, N, D], float32, 正向输入 query
        key:   [S, B, N, D], float32, 正向输入 key
        cos:   [S, B, 1, D] or [S, 1, 1, D], float32, 旋转余弦
        sin:   [S, B, 1, D] or [S, 1, 1, D], float32, 旋转正弦

    返回:
        q_embed: [S, B, N, D], float32, 旋转后的 query
        k_embed: [S, B, N, D], float32, 旋转后的 key
        cache: dict, 反向传播需要的中间变量
    """
    q_embed = query * cos + _rotate_half(query) * sin
    k_embed = key * cos + _rotate_half(key) * sin

    cache = {
        'query': query,
        'key': key,
        'cos': cos,
        'sin': sin,
        'query_rotate': _rotate_half(query),
        'key_rotate': _rotate_half(key),
    }
    return q_embed, k_embed, cache


def _apply_rotary_pos_emb_backward(cache, d_q_embed, d_k_embed):
    """反向传播函数（half模式）

    参数:
        cache: 前向传播保存的中间变量
        d_q_embed: [S, B, N, D], float32, 上游梯度 ∂L/∂q_embed
        d_k_embed: [S, B, N, D], float32, 上游梯度 ∂L/∂k_embed

    返回:
        d_query: [S, B, N, D], float32, query 的梯度
        d_key:   [S, B, N, D], float32, key 的梯度
        d_cos:   [S, B, 1, D] or [S, 1, 1, D], float32, cos 的梯度
        d_sin:   [S, B, 1, D] or [S, 1, 1, D], float32, sin 的梯度
    """
    # 与 kernel (BatchHalfGradContiguousVF / AccumDcosDsinVF) 精度模型对齐：中间寄存器
    # RegTensor<float>，fp16/bf16 输入 upcast 到 fp32 做乘加累加，输出 downcast 回原 dtype。
    # 纯低精度中间计算会引入 kernel 不存在的额外舍入误差（fp16 ~2^-7、bf16 ~2^-4）。
    out_dt = cache['query'].dtype
    query = cache['query'].float()
    key = cache['key'].float()
    cos = cache['cos'].float()
    sin = cache['sin'].float()
    query_rotate = cache['query_rotate'].float()
    key_rotate = cache['key_rotate'].float()
    d_q = d_q_embed.float()
    d_k = d_k_embed.float()

    # ==========================================
    # Step 1: 切分上游梯度
    # dq_1, dq_2 = chunk(d_q_embed, 2, dim=-1)
    # dk_1, dk_2 = chunk(d_k_embed, 2, dim=-1)
    # ==========================================
    dq_1, dq_2 = d_q.chunk(2, dim=-1)
    dk_1, dk_2 = d_k.chunk(2, dim=-1)

    # ==========================================
    # Step 2: 切分 cos/sin
    # c_1, c_2 = chunk(cos, 2, dim=-1)
    # s_1, s_2 = chunk(sin, 2, dim=-1)
    # ==========================================
    c_1, c_2 = cos.chunk(2, dim=-1)
    s_1, s_2 = sin.chunk(2, dim=-1)

    # ==========================================
    # Step 3: 计算 d_query
    # d_query = cat(c_1 * dq_1 + s_2 * dq_2,  c_2 * dq_2 - s_1 * dq_1, dim=-1)
    # ==========================================
    d_query = torch.cat([c_1 * dq_1 + s_2 * dq_2,
                         c_2 * dq_2 - s_1 * dq_1], dim=-1)

    # ==========================================
    # Step 4: 计算 d_key（对称计算）
    # d_key = cat(c_1 * dk_1 + s_2 * dk_2,  c_2 * dk_2 - s_1 * dk_1, dim=-1)
    # ==========================================
    d_key = torch.cat([c_1 * dk_1 + s_2 * dk_2,
                       c_2 * dk_2 - s_1 * dk_1], dim=-1)

    # ==========================================
    # Step 5: 计算 d_cos
    # d_cos = sum(d_q_embed * query + d_k_embed * key, dims)
    # dims = cos/sin 相对 query/key 的广播维度（SBND layout 下为 dim=1(B) 和 dim=2(N)）
    # ==========================================
    cos_grad = d_q * query + d_k * key
    dims = _get_broadcast_dims(cos, query)
    d_cos = cos_grad.sum(dim=dims, keepdim=True) if dims else cos_grad

    # ==========================================
    # Step 6: 计算 d_sin
    # d_sin = sum(d_q_embed * query_rotate + d_k_embed * key_rotate, dims)
    # ==========================================
    sin_grad = d_q * query_rotate + d_k * key_rotate
    d_sin = sin_grad.sum(dim=dims, keepdim=True) if dims else sin_grad

    return d_query.to(out_dt), d_key.to(out_dt), d_cos.to(out_dt), d_sin.to(out_dt)


def _get_broadcast_dims(cos, query):
    """推导 SBND layout 下 cos/sin 相对 query/key 的广播维度

    SBND layout: [S, B, N, D]
    cos/sin 的 N 维度必须为 1，B 维度可以为 1 或与 query 一致
    """
    dims = []
    for i in range(cos.dim()):
        if cos.size(i) == 1 and query.size(i) > 1:
            dims.append(i)
    return dims


class ApplyRotaryPosEmbGradFunction(torch.autograd.Function):
    """自定义 autograd 函数"""

    @staticmethod
    def forward(ctx, query, key, cos, sin):
        q_embed, k_embed, cache = _apply_rotary_pos_emb_forward(query, key, cos, sin)
        ctx.save_for_backward(
            cache['query'],
            cache['key'],
            cache['cos'],
            cache['sin'],
            cache['query_rotate'],
            cache['key_rotate'],
        )
        return q_embed, k_embed

    @staticmethod
    def backward(ctx, d_q_embed, d_k_embed):
        query, key, cos, sin, query_rotate, key_rotate = ctx.saved_tensors

        cache = {
            'query': query,
            'key': key,
            'cos': cos,
            'sin': sin,
            'query_rotate': query_rotate,
            'key_rotate': key_rotate,
        }
        d_query, d_key, d_cos, d_sin = _apply_rotary_pos_emb_backward(cache, d_q_embed, d_k_embed)

        return d_query, d_key, d_cos, d_sin


def apply_rotary_pos_emb_autograd(query, key, cos, sin):
    """使用自定义 autograd Function 的接口"""
    return ApplyRotaryPosEmbGradFunction.apply(query, key, cos, sin)


def apply_rotary_pos_emb_native(query, key, cos, sin):
    """PyTorch 原生实现（用于 autograd 对比）"""
    q_embed = query * cos + _rotate_half(query) * sin
    k_embed = key * cos + _rotate_half(key) * sin
    return q_embed, k_embed


# ============================================================
# 验证代码
# ============================================================

def _run_single_test(test_name, S, B, N, D, cos_B, dtype, tolerance):
    """运行单个梯度对比测试

    参数:
        test_name: 测试名称
        S, B, N, D: SBND layout 维度
        cos_B: cos/sin 的 B 维度大小（1 表示广播，等于 B 表示不广播）
        dtype: 数据类型
        tolerance: 容差
    """
    print(f"\n--- {test_name}: S={S}, B={B}, N={N}, D={D}, cos_B={cos_B}, dtype={dtype} ---")

    query_data = torch.randn(S, B, N, D, dtype=torch.float32)
    key_data = torch.randn(S, B, N, D, dtype=torch.float32)
    cos_data = torch.randn(S, cos_B, 1, D, dtype=torch.float32)
    sin_data = torch.randn(S, cos_B, 1, D, dtype=torch.float32)

    # 手动实现的 backward
    q_manual = query_data.clone().requires_grad_(True)
    k_manual = key_data.clone().requires_grad_(True)
    c_manual = cos_data.clone().requires_grad_(True)
    s_manual = sin_data.clone().requires_grad_(True)

    q_embed_m, k_embed_m = apply_rotary_pos_emb_autograd(q_manual, k_manual, c_manual, s_manual)
    grad_q_embed = torch.randn_like(q_embed_m)
    grad_k_embed = torch.randn_like(k_embed_m)
    torch.autograd.backward([q_embed_m, k_embed_m], [grad_q_embed, grad_k_embed])

    # PyTorch 自动求导
    q_autograd = query_data.clone().requires_grad_(True)
    k_autograd = key_data.clone().requires_grad_(True)
    c_autograd = cos_data.clone().requires_grad_(True)
    s_autograd = sin_data.clone().requires_grad_(True)

    q_embed_a, k_embed_a = apply_rotary_pos_emb_native(q_autograd, k_autograd, c_autograd, s_autograd)
    torch.autograd.backward([q_embed_a, k_embed_a], [grad_q_embed, grad_k_embed])

    all_pass = True
    for name, g_m, g_a in [("d_query", q_manual.grad, q_autograd.grad),
                            ("d_key", k_manual.grad, k_autograd.grad),
                            ("d_cos", c_manual.grad, c_autograd.grad),
                            ("d_sin", s_manual.grad, s_autograd.grad)]:
        passed = _print_grad_comparison(name, g_m, g_a, tolerance)
        if not passed:
            all_pass = False

    return all_pass


def compare_gradients():
    """对比手动实现的 backward 和 PyTorch 自动求导的梯度

    测试数据基于文档泛化规格:
    - 数据类型: BF16/FP16/FP32
    - layout: SBND
    - S: 8K - 32K
    - B: 1
    - N: 最大64
    - D: 16的倍数，最大值128
    - cos/sin 的 N 必须为 1，B 可以为 1 或与 query 一致
    """
    print("=" * 70)
    print("ApplyRotaryPosEmbGrad 梯度对比测试 (half模式, SBND layout)")
    print("=" * 70)

    torch.manual_seed(42)
    tolerance = 1e-4
    total_pass = 0
    total_fail = 0

    # ================================================================
    # 测试组1: 文档典型 shape（FP32, cos/sin B=1 广播）
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组1: 文档典型 shape (FP32, cos/sin B=1 广播)")
    print("=" * 70)

    typical_shapes = [
        # 文档典型: B=1, N=8/16/32, D=64/80/128, S 适当缩小
        (2048, 1, 8, 64),
        (2048, 1, 16, 80),
        (2048, 1, 32, 128),
        (1024, 1, 8, 64),
        (1024, 1, 16, 80),
        (1024, 1, 32, 128),
        (4096, 1, 8, 64),
        (4096, 1, 16, 128),
    ]

    for i, (S, B, N, D) in enumerate(typical_shapes):
        name = f"典型shape-{i+1}"
        if _run_single_test(name, S, B, N, D, cos_B=1, dtype=torch.float32, tolerance=tolerance):
            total_pass += 1
        else:
            total_fail += 1

    # ================================================================
    # 测试组2: cos/sin B维度与query一致（无广播）
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组2: cos/sin B维度与query一致 (FP32, cos/sin B=B)")
    print("=" * 70)

    no_broadcast_shapes = [
        (2048, 1, 8, 64),
        (2048, 1, 16, 80),
        (2048, 1, 32, 128),
        (1024, 1, 8, 64),
        (1024, 1, 32, 128),
    ]

    for i, (S, B, N, D) in enumerate(no_broadcast_shapes):
        name = f"无广播-{i+1}"
        if _run_single_test(name, S, B, N, D, cos_B=B, dtype=torch.float32, tolerance=tolerance):
            total_pass += 1
        else:
            total_fail += 1

    # ================================================================
    # 测试组3: 多数据类型（BF16/FP16/FP32）
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组3: 多数据类型 (典型 shape)")
    print("=" * 70)

    dtype_shapes = [
        (2048, 1, 8, 64),
        (2048, 1, 32, 128),
        (1024, 1, 16, 80),
    ]

    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        for i, (S, B, N, D) in enumerate(dtype_shapes):
            name = f"{dtype}-{i+1}"
            if _run_single_test(name, S, B, N, D, cos_B=1, dtype=dtype, tolerance=tolerance):
                total_pass += 1
            else:
                total_fail += 1

    # ================================================================
    # 测试组4: N 维度泛化（N=1,2,4,8,16,32,64）
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组4: N 维度泛化 (S=1024, B=1, D=128, FP32)")
    print("=" * 70)

    for N in [1, 2, 4, 8, 16, 32, 64]:
        name = f"N={N}"
        if _run_single_test(name, 1024, 1, N, 128, cos_B=1, dtype=torch.float32, tolerance=tolerance):
            total_pass += 1
        else:
            total_fail += 1

    # ================================================================
    # 测试组5: D 维度泛化（16的倍数, 16~128）
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组5: D 维度泛化 (S=1024, B=1, N=8, FP32)")
    print("=" * 70)

    for D in [16, 32, 48, 64, 80, 96, 112, 128]:
        name = f"D={D}"
        if _run_single_test(name, 1024, 1, 8, D, cos_B=1, dtype=torch.float32, tolerance=tolerance):
            total_pass += 1
        else:
            total_fail += 1

    # ================================================================
    # 测试组6: S 维度泛化
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组6: S 维度泛化 (B=1, N=8, D=128, FP32)")
    print("=" * 70)

    for S in [512, 1024, 2048, 4096, 8192]:
        name = f"S={S}"
        if _run_single_test(name, S, 1, 8, 128, cos_B=1, dtype=torch.float32, tolerance=tolerance):
            total_pass += 1
        else:
            total_fail += 1

    # ================================================================
    # 测试组7: 边界 D=1024（泛化规格上限）
    # ================================================================
    print("\n" + "=" * 70)
    print("测试组7: 边界 D=1024 (S=512, B=1, N=1, FP32)")
    print("=" * 70)

    if _run_single_test("D=1024边界", 512, 1, 1, 1024, cos_B=1, dtype=torch.float32, tolerance=tolerance):
        total_pass += 1
    else:
        total_fail += 1

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"测试汇总: PASS={total_pass}, FAIL={total_fail}")
    print(f"{'=' * 70}")


def _print_grad_comparison(name, grad_manual, grad_autograd, tolerance):
    """打印梯度对比结果"""
    diff = (grad_manual - grad_autograd).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    mse = (diff ** 2).mean().item()

    print(f"\n  {name}:")
    print(f"    shape: {grad_manual.shape}")
    print(f"    最大差异: {max_diff:.8f}")
    print(f"    平均差异: {mean_diff:.8f}")
    print(f"    MSE: {mse:.10f}")

    abs_grad = grad_autograd.abs()
    meaningful_mask = abs_grad > 1e-6
    if meaningful_mask.any():
        rel_error = (diff[meaningful_mask] / abs_grad[meaningful_mask])
        print(f"    最大相对误差: {rel_error.max():.8f}")
        print(f"    平均相对误差: {rel_error.mean():.8f}")

    if max_diff < tolerance:
        print(f"    [PASS] 差异 < {tolerance}")
        return True
    else:
        print(f"    [FAIL] 差异 >= {tolerance}")
        return False


if __name__ == "__main__":
    compare_gradients()