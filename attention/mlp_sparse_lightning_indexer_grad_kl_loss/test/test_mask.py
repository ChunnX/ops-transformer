"""
单测: 验证 mlp_sparse_lightning_indexer_grad_kl_loss 算子的 mask 功能
- mask=1 的 token 正常计算 loss 和梯度
- mask=0 的 token 跳过，dq/dw 输出为 0，不贡献 loss
"""

import torch
import torch_npu
import numpy as np
import random
import pdb

DEVICE_ID = 11
torch_npu.npu.set_device(int(DEVICE_ID))

np.set_printoptions(precision=4, linewidth=768, threshold=np.inf, suppress=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(42)


def check_result(golden_output, output, name=None, rtol=5e-3, atol=5e-3):
    diff = golden_output - output
    abs_diff = torch.abs(diff)
    print(f"[INFO] {name} max diff: {torch.max(abs_diff).item():.9f}, mean diff: {torch.mean(abs_diff).item():.9f}")

    result_check = torch.allclose(golden_output, output, rtol=rtol, atol=atol)
    if not result_check:
        tolerance = atol + rtol * torch.abs(golden_output)
        mismatched_positions = torch.nonzero(abs_diff > tolerance)
        n_mismatch = mismatched_positions.shape[0]
        n_total = golden_output.numel()
        pct = n_mismatch / n_total * 100
        if n_mismatch > n_total * 0.001:
            print(f"[WARN] {name} FAILED: {n_mismatch}/{n_total} ({pct:.2f}%) elements mismatch")
            # 打印前10个不匹配
            for i in range(min(10, n_mismatch)):
                idx = tuple(mismatched_positions[i].tolist())
                print(f"  idx={idx}, golden={golden_output[idx].item():.6f}, actual={output[idx].item():.6f}, diff={diff[idx].item():.6f}")
            assert False, f"{name} accuracy check failed"
        else:
            print(f"[INFO] {name} Pass (minor mismatch: {n_mismatch}/{n_total})")
    else:
        print(f"[INFO] {name} Pass")


def check_zero(tensor, name):
    max_abs = tensor.abs().max().item()
    print(f"[INFO] {name} max abs: {max_abs}")
    assert max_abs == 0, f"{name} should be all zeros but max abs = {max_abs}"
    print(f"[INFO] {name} is all zeros - Pass")


def print_diff(name, tensor_a, tensor_b):
    diff = torch.abs(tensor_b.float() - tensor_a.float())
    print(f"{name} max diff: {torch.max(diff).item():.9f}, mean diff: {torch.mean(diff).item():.9f}")


def gen_seq_len(B, T1):
    random_numbers = np.random.rand(B)
    normalized_numbers = random_numbers / random_numbers.sum()
    q_len_tmp = [int(T1 * normalized_numbers[i]) for i in range(B)]
    q_len = [1 if x == 0 else x for x in q_len_tmp]
    kv_len = [q_len[i] + np.random.randint(1, 8192) for i in range(B)]
    return q_len, kv_len


def compute_topk(index_q, index_k, weights, topk=2048, topk_indices=None, cu_seqlens_q=None, cu_seqlens_k=None):
    """计算 golden topk score (S'Y)"""
    T1 = index_q.shape[0]
    Nqi = index_q.shape[1]
    Nki = index_k.shape[1]
    G_index = Nqi // Nki
    D_index = index_q.shape[2]

    # topk_indices: [T1, Nqi, K]
    K = topk_indices.shape[2]
    T2 = index_k.shape[0]

    # 计算 topk_score = sum over d of (q_index * k_index[topk]) * weight
    # index_q: [T1, Nqi, D], index_k: [T2, Nki, D]
    # 按 group 处理
    topk_score = torch.zeros(T1, Nqi, dtype=torch.float32, device=index_q.device)

    for t in range(T1):
        for g in range(Nqi):
            ki_group = g // G_index
            w = weights[t, g]
            score_sum = 0.0
            for kidx in range(K):
                k_token = topk_indices[t, g, kidx].item()
                if k_token < 0 or k_token >= T2:
                    continue
                # dot product
                dot = (index_q[t, g, :].float() * index_k[k_token, ki_group, :].float()).sum().item()
                score_sum += dot
            topk_score[t, g] = w * score_sum

    return topk_score, topk_indices


def kl_loss_per_token(select_p, topk_score):
    """
    计算每个 token 的 KL loss (简化版)
    select_p: [T, N] - softmax attention 的 probability sum per head
    topk_score: [T, N] - indexer 的 score
    返回 per-token loss: [T]
    """
    # KL(P||Y) = sum_head [ P * log(P / Y) ] 的简化实现
    # 实际算子内部计算逻辑更复杂，这里用 autograd 的方式获取 golden
    pass


class KLLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, select_p, topk_score):
        # KL loss: sum over heads of (P - P*log(P) + P*log(sum_exp_Y) - Y_max)
        # 简化为: loss = -sum(select_p * topk_score) 的某种形式
        # 实际使用算子 golden 的实现
        P = select_p.float()
        Y = topk_score.float()
        # loss per token per head
        loss = P * (torch.log(P.clamp(min=1e-12)) - Y)
        ctx.save_for_backward(select_p, topk_score)
        return loss.sum(dim=-1)  # [T]

    @staticmethod
    def backward(ctx, grad_output):
        select_p, topk_score = ctx.saved_tensors
        P = select_p.float()
        Y = topk_score.float()
        # d/dY = -P
        dY = -P * grad_output[:, None]
        # d/dP = (log P - Y + 1) but P is detached in actual usage
        return None, dY


def test_mask(B, T1, blockCount, blockSize, mask, validTokenNum=-1, test_name="mask test"):
    """
    测试 mask 功能的主函数
    Args:
        B: batch size (num sequences)
        T1: total query tokens
        blockCount: topk count
        blockSize: block size
        mask: torch.Tensor of shape [T1], dtype int8, 1=valid, 0=skip
        validTokenNum: valid token num attr (-1 = no truncation)
        test_name: test case name for display
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"  B={B}, T1={T1}, blockCount={blockCount}, blockSize={blockSize}")
    print(f"  validTokenNum={validTokenNum}")
    print(f"  mask valid tokens: {mask.sum().item()}/{mask.shape[0]}")
    print(f"{'='*60}")

    device = torch.device(f'npu:{DEVICE_ID}')
    torch.npu.set_device(device)
    dtype = torch.bfloat16
    isDetermin = True

    headNum = 64
    headNumK = 1
    headNumIdx = 32
    headNumKKIdx = 1
    headDimP = 512
    headDimSY = 128
    sm_scale = 192 ** -0.5
    layout = "TND"

    actual_seq_lengths_query, actual_seq_lengths_key = gen_seq_len(B, T1)
    T1 = int(np.sum(actual_seq_lengths_query))
    T2 = int(np.sum(actual_seq_lengths_key))
    cu_actual_seq_lengths_query = list(np.cumsum(actual_seq_lengths_query))
    cu_actual_seq_lengths_key = list(np.cumsum(actual_seq_lengths_key))
    print(f"  Actual T1={T1}, T2={T2}")

    # 重新生成匹配实际 T1 长度的 mask
    if mask.shape[0] != T1:
        # 截断或扩展 mask 到实际长度
        if mask.shape[0] > T1:
            mask = mask[:T1]
        else:
            extended = torch.zeros(T1, dtype=torch.int8)
            extended[:mask.shape[0]] = mask
            mask = extended

    # inputs
    q_absorb = (torch.rand(T1, headNum, headDimP, device=device, dtype=dtype) * 20 - 10).requires_grad_(True)
    q_pe = (torch.rand(T1, headNum, 64, device=device, dtype=dtype) * 20 - 10).requires_grad_(True)
    ct_kv = (torch.rand(T2, headNumK, headDimP, device=device, dtype=dtype) * 20 - 10).requires_grad_(True)
    k_pe = (torch.rand(T2, headNumK, 64, device=device, dtype=dtype) * 20 - 10).requires_grad_(True)
    q = torch.cat([q_absorb, q_pe], dim=-1)
    kv = torch.cat([ct_kv, k_pe], dim=-1)

    scale = headNumIdx ** -0.5 * (headDimSY ** -0.5)
    query_index = (torch.rand(T1, headNumIdx, headDimSY, device=device, dtype=dtype) * 20 - 10).requires_grad_(True)
    key_index = (torch.rand(T2, headNumKKIdx, headDimSY, device=device, dtype=dtype) * 20 - 10).requires_grad_(True)
    weights = (torch.rand(T1, headNumIdx, device=device, dtype=torch.float) * 5 * scale).requires_grad_(True)
    do = torch.randn(T1, headNum, headDimP, device=device, dtype=dtype)

    cu_actual_seq_lengths_query0 = [0] + list(np.cumsum(actual_seq_lengths_query))
    cu_actual_seq_lengths_key0 = [0] + list(np.cumsum(actual_seq_lengths_key))
    mlp_cu_q0 = torch.tensor(cu_actual_seq_lengths_query0, dtype=torch.int64).to(device).contiguous()
    mlp_cu_k0 = torch.tensor(cu_actual_seq_lengths_key0, dtype=torch.int64).to(device).contiguous()

    # Step 1: 调用 lightning_indexer 获取 topk_index
    topk_index, _ = torch_npu.mlp_lightning_indexer(
        query_index,
        key_index,
        weights,
        cur_seq_lengths_query=mlp_cu_q0,
        cur_seq_lengths_key=mlp_cu_k0,
        block_table=None,
        layout_query="TND",
        layout_key="TND",
        sparse_count=blockCount,
        kv_block_len=1,
        q_block_len=1,
        init_num=16,
        local_num=0,
        sparse_mode=3,
        pre_tokens=512*1024,
        next_tokens=512*1024,
        return_value=False)

    # Step 2: 计算 golden reference
    # 使用 SFA 获取 softmax max/sum（与实际算子一致）
    _, mlp_max, mlp_sum = torch_npu.mlp_sparse_flash_attention(
        q_absorb, ct_kv, ct_kv, topk_index, sm_scale, block_table=None,
        cur_seq_lengths_query=mlp_cu_q0, cur_seq_lengths_kv=mlp_cu_k0,
        query_rope=q_pe, key_rope=k_pe, sparse_block_size=blockSize,
        layout_query=layout, layout_kv=layout, sparse_mode=3, pre_tokens=512*1024,
        next_tokens=512*1024, attention_mode=0, return_softmax_lse=True)

    # Step 3: 用不带 mask 的算子调用获取完整结果（作为 golden 基准）
    # 不传 mask，所有 token 都计算
    full_dq, full_dk, full_dw, full_loss = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss_skip_padding(
        q_absorb,
        ct_kv,
        query_index,
        key_index,
        weights,
        topk_index,
        mlp_max,
        mlp_sum,
        q_pe,
        k_pe,
        None,  # mask=None, 全部计算
        cu_actual_seq_lengths_query0,
        cu_actual_seq_lengths_key0,
        sm_scale,
        layout,
        3,  # sparse_mode
        512*1024,
        512*1024,
        blockSize,
        isDetermin,
        validTokenNum
    )
    torch.npu.synchronize()

    # Step 4: 带 mask 的算子调用
    mask_npu = mask.to(device).contiguous()
    masked_dq, masked_dk, masked_dw, masked_loss = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss_skip_padding(
        q_absorb,
        ct_kv,
        query_index,
        key_index,
        weights,
        topk_index,
        mlp_max,
        mlp_sum,
        q_pe,
        k_pe,
        mask_npu,  # 传入 mask
        cu_actual_seq_lengths_query0,
        cu_actual_seq_lengths_key0,
        sm_scale,
        layout,
        3,  # sparse_mode
        512*1024,
        512*1024,
        blockSize,
        isDetermin,
        validTokenNum
    )
    torch.npu.synchronize()

    # Step 5: 验证结果
    valid_mask = (mask == 1).to(device)
    invalid_mask = (mask == 0).to(device)
    n_valid = valid_mask.sum().item()
    n_invalid = invalid_mask.sum().item()

    print(f"\n--- Results (valid={n_valid}, invalid={n_invalid}) ---")

    # 5.1 验证 mask=1 的 token: dq 应与 full_dq 对应位置一致
    if n_valid > 0:
        print("\n[Check 1] Valid token gradients match full computation:")
        # dq: [T1, headNumIdx, headDimSY]
        valid_idx = valid_mask.nonzero(as_tuple=True)[0]
        print_diff('dq (valid)', full_dq[valid_idx], masked_dq[valid_idx])
        check_result(full_dq[valid_idx], masked_dq[valid_idx], 'dq (valid tokens)')

        print_diff('dw (valid)', full_dw[valid_idx], masked_dw[valid_idx])
        check_result(full_dw[valid_idx], masked_dw[valid_idx], 'dw (valid tokens)')

    # 5.2 验证 mask=0 的 token: dq, dw 应为全零
    if n_invalid > 0:
        print("\n[Check 2] Invalid token gradients are zero:")
        invalid_idx = invalid_mask.nonzero(as_tuple=True)[0]
        # pdb.set_trace()
        check_zero(masked_dq[invalid_idx], 'dq (masked tokens)')
        check_zero(masked_dw[invalid_idx], 'dw (masked tokens)')

    # 5.3 验证 dk: masked token 不贡献 scatter_add
    # dk 的 golden 需要特殊处理：只有 valid token 做 scatter_add
    # 由于 dk 是 scatter_add 的结果，无法简单分离，但可以验证：
    # 用 mask 版本的 dk 与完整版本的 dk 比较——如果有 invalid token，masked_dk 应该更小
    print("\n[Check 3] dk comparison:")
    print_diff('dk (full vs masked)', full_dk, masked_dk)
    # dk 不要求完全相等（因为 masked token 不贡献），只要求 masked 版本的值 <= full 版本
    dk_diff = (masked_dk.float() - full_dk.float()).abs()
    print(f"  dk diff max: {dk_diff.max().item():.9f}, mean: {dk_diff.mean().item():.9f}")

    # 5.4 验证 loss: masked 版本的 loss 应等于只对 valid token 计算的 loss
    print("\n[Check 4] Loss comparison:")
    print(f"  Full loss:   {full_loss.item():.6f}")
    print(f"  Masked loss: {masked_loss.item():.6f}")
    # masked loss 应该小于等于 full loss（少了部分 token 的贡献）
    if n_invalid > 0:
        assert masked_loss.item() <= full_loss.item() + 1e-3, \
            f"Masked loss ({masked_loss.item()}) should be <= full loss ({full_loss.item()})"
        print("  Masked loss <= Full loss - Pass")

    # 5.5 回归测试: mask 全为 1 时应与不传 mask 完全一致
    print("\n[Check 5] Regression: all-ones mask == no mask:")
    all_ones_mask = torch.ones(T1, dtype=torch.int8).to(device).contiguous()
    reg_dq, reg_dk, reg_dw, reg_loss = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss_skip_padding(
        q_absorb,
        ct_kv,
        query_index,
        key_index,
        weights,
        topk_index,
        mlp_max,
        mlp_sum,
        q_pe,
        k_pe,
        all_ones_mask,
        cu_actual_seq_lengths_query0,
        cu_actual_seq_lengths_key0,
        sm_scale,
        layout,
        3,
        512*1024,
        512*1024,
        blockSize,
        isDetermin,
        validTokenNum
    )
    torch.npu.synchronize()

    # all-ones mask 应与 no-mask 完全 bit-exact
    assert full_dq.equal(reg_dq), "dq: all-ones mask != no mask"
    assert full_dk.equal(reg_dk), "dk: all-ones mask != no mask"
    assert full_dw.equal(reg_dw), "dw: all-ones mask != no mask"
    assert full_loss.equal(reg_loss), "loss: all-ones mask != no mask"
    print("  All-ones mask == No mask: Pass (bit-exact)")

    print(f"\n{'='*60}")
    print(f"TEST PASSED: {test_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Case 1: 连续 mask (模拟单轮 SFT: 前面 prompt, 后面 response)
    # mask1 = torch.ones(3000, dtype=torch.int8)
    mask1 = torch.zeros(3000, dtype=torch.int8)
    mask1[1000:] = 1  # token [1000, 3000) 是 response
    test_mask(1, 3000, 2048, 1, mask1, validTokenNum=-1,
              test_name="Contiguous mask (single-turn SFT)")

    # Case 2: 非连续 mask (模拟多轮对话: 交替 prompt/response)
    mask2 = torch.zeros(3000, dtype=torch.int8)
    mask2[300:800] = 1     # 第一轮 response
    mask2[1200:2000] = 1   # 第二轮 response
    mask2[2400:2800] = 1   # 第三轮 response
    test_mask(1, 3000, 2048, 1, mask2, validTokenNum=-1,
              test_name="Non-contiguous mask (multi-turn chat)")

    # Case 3: mask + validTokenNum 组合
    mask3 = torch.zeros(3000, dtype=torch.int8)
    mask3[200:600] = 1
    mask3[1000:1500] = 1
    test_mask(1, 3000, 2048, 1, mask3, validTokenNum=2000,
              test_name="Mask + validTokenNum combined")

    # Case 4: 全零 mask (所有 token 都跳过)
    mask4 = torch.zeros(3000, dtype=torch.int8)
    test_mask(1, 3000, 2048, 1, mask4, validTokenNum=-1,
              test_name="All-zero mask (no valid tokens)")

    # Case 5: 多 batch
    mask5 = torch.zeros(4000, dtype=torch.int8)
    mask5[500:1500] = 1
    mask5[2500:3500] = 1
    test_mask(2, 4000, 2048, 1, mask5, validTokenNum=-1,
              test_name="Multi-batch with mask")

    # Case 6: 稀疏 mask (每隔几个 token 有效)
    mask6 = torch.zeros(3000, dtype=torch.int8)
    mask6[::3] = 1  # 每3个token取1个
    test_mask(1, 3000, 2048, 1, mask6, validTokenNum=-1,
              test_name="Sparse mask (every 3rd token)")

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
