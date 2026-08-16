import ast
import gc
import math
import os
import random
import sys
import traceback
from pathlib import Path

import torch
import torch_npu
from einops import repeat

# Use the local torch extension implementation instead of torch_npu's built-in API.
TORCH_EXTENSION_ROOT = Path(__file__).resolve().parents[4] / "torch_extension"
if str(TORCH_EXTENSION_ROOT) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION_ROOT))
from cann_ops_transformer.ops import generic_block_sparse_attention_grad
from block_mask_to_idx import blcok_sparse_mask_to_idx

torch.set_printoptions(precision=6)

SEED = 128
DEVICE = os.getenv("BSAG_DEVICE", "npu:6")
DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAME = os.getenv("BSAG_DTYPE", "fp16").lower()
if DTYPE_NAME not in DTYPE_MAP:
    raise ValueError(f"unsupported BSAG_DTYPE: {DTYPE_NAME}")
DTYPE = DTYPE_MAP[DTYPE_NAME]
# BF16 的有效尾数较短，MatMul/Reduce 后单次量化误差可达到 1e-2；
# FP16 继续使用更严格的 1e-3。环境变量仍可显式覆盖默认阈值。
DEFAULT_TOL = "1e-2" if DTYPE in (torch.bfloat16,) else "1e-3"
RTOL = float(os.getenv("BSAG_RTOL", DEFAULT_TOL))
ATOL = float(os.getenv("BSAG_ATOL", DEFAULT_TOL))
CASE_SEPARATOR = "=" * 100
CASE_DETAIL_SEPARATOR = "-" * 100


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)


def print_case_begin(case_id, full_case):
    print(f"\n{CASE_SEPARATOR}", flush=True)
    print(f"CASE_BEGIN id={case_id:03d} dtype={DTYPE_NAME} device={DEVICE}", flush=True)
    print(f"CASE_CONFIG id={case_id:03d} values={full_case!r}", flush=True)
    print(CASE_DETAIL_SEPARATOR, flush=True)


def print_case_end(case_id, status, error=None):
    print(CASE_DETAIL_SEPARATOR, flush=True)
    if status == "PASS":
        print(f"CASE_END id={case_id:03d} status=PASS", flush=True)
    else:
        print(f"CASE_END id={case_id:03d} status=FAIL error={error!r}", flush=True)
    print(CASE_SEPARATOR, flush=True)


seed_everything(SEED)


def compare_tensor(name, golden, out, max_mismatch_samples=20):
    golden = golden.cpu().flatten().to(torch.float32)
    out = out.cpu().flatten().to(torch.float32)
    passed = torch.allclose(golden, out, rtol=RTOL, atol=ATOL)

    abs_diff = torch.abs(out - golden)
    rel_diff = abs_diff / torch.clamp(torch.abs(golden), min=1e-8)
    max_abs_err = abs_diff.max().item() if abs_diff.numel() else 0.0
    max_rel_err = rel_diff.max().item() if rel_diff.numel() else 0.0
    mean_abs_err = abs_diff.mean().item() if abs_diff.numel() else 0.0
    print(
        f"COMPARE tensor={name} status={'PASS' if passed else 'FAIL'} "
        f"max_abs={max_abs_err:.8e} max_rel={max_rel_err:.8e} mean_abs={mean_abs_err:.8e}"
    )

    if not passed:
        rel_diff_max = rel_diff.max()
        rel_diff_max_id = rel_diff.argmax().item()

        abs_diff_max = abs_diff.max()
        diff_max_id = abs_diff.argmax().item()

        print(
            f"COMPARE_DETAIL tensor={name} max_abs index={diff_max_id} diff={abs_diff_max:.8e} "
            f"out={out[diff_max_id]:.8e} golden={golden[diff_max_id]:.8e}"
        )
        print(
            f"COMPARE_DETAIL tensor={name} max_rel index={rel_diff_max_id} diff={rel_diff_max:.8e} "
            f"out={out[rel_diff_max_id]:.8e} golden={golden[rel_diff_max_id]:.8e}"
        )

        equal_mask = torch.isclose(golden, out, rtol=RTOL, atol=ATOL)
        unequal_indices = torch.where(~equal_mask)[0]
        print(
            f"COMPARE_DETAIL tensor={name} mismatch_count={unequal_indices.numel()} "
            f"indices={unequal_indices[:max_mismatch_samples].tolist()}"
        )

    return bool(passed)


def load_test_cases(case_file):
    cases = []
    for line_number, raw_line in enumerate(
        case_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            case = ast.literal_eval(line)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"invalid test case at {case_file}:{line_number}: {line}"
            ) from error
        if not isinstance(case, list) or len(case) != 10:
            raise ValueError(
                f"test case must be a 10-item list at {case_file}:{line_number}: {case!r}"
            )
        if not isinstance(case[9], tuple) or len(case[9]) != 2:
            raise ValueError(
                f"block shape must be a 2-item tuple at {case_file}:{line_number}: {case[9]!r}"
            )
        cases.append(case)
    if not cases:
        raise ValueError(f"no test cases found in {case_file}")
    return cases


def select_test_cases(cases):
    requested_ids = os.getenv("BSAG_CASE_IDS", "").strip()
    if requested_ids:
        selected_ids = {
            int(value) for value in requested_ids.split(",") if value.strip()
        }
    else:
        start = int(os.getenv("BSAG_CASE_START", "1"))
        end = int(os.getenv("BSAG_CASE_END", str(len(cases))))
        selected_ids = set(range(start, end + 1))
    invalid_ids = sorted(
        case_id for case_id in selected_ids if case_id < 1 or case_id > len(cases)
    )
    if invalid_ids:
        raise ValueError(f"case ids out of range 1..{len(cases)}: {invalid_ids}")
    return [(case_id, cases[case_id - 1]) for case_id in sorted(selected_ids)]


def compare_golden(cpu_dq, cpu_dk, cpu_dv, d_query, d_key, d_value):
    results = (
        compare_tensor("dq", cpu_dq, d_query),
        compare_tensor("dk", cpu_dk, d_key),
        compare_tensor("dv", cpu_dv, d_value),
    )
    return all(results)


def attention_ori(query, key, value, scale_value):
    """
    (b, n, s, d)
    """
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    batch_size, head_num, seq_size, head_dim = query.shape
    kv_head_num = key.shape[1]
    if head_num % kv_head_num != 0:
        raise ValueError(
            f"query heads ({head_num}) must be divisible by KV heads ({kv_head_num})"
        )
    if kv_head_num != head_num:
        group_size = head_num // kv_head_num
        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)

    bmm1_res = torch.matmul(query, torch.transpose(key, 2, 3)).to(torch.float32)
    bmm1_res = bmm1_res * scale_value

    softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
    softmax_sub = bmm1_res - softmax_max

    # print("attention_ori")
    # print(f"softmax_sub shape = {softmax_sub.shape}")
    softmax_exp = torch.exp(softmax_sub)
    softmax_sum = softmax_exp.sum(dim=-1, keepdim=True)
    # print(f"softmax_sum shape = {softmax_sum.shape}")
    softmax_out = softmax_exp / softmax_sum
    # print(f"softmax_out shape = {softmax_out.shape}")
    # softmax_out = softmax_out.to(torch.float)
    attention_out = torch.matmul(softmax_out, value)
    # print(f"attention_out shape = {attention_out.shape}")
    softmax_log_max_sum = softmax_max + torch.log(softmax_sum)

    return softmax_log_max_sum, attention_out.to(DTYPE).contiguous()
    # return softmax_max, softmax_sum, softmax_log_max_sum, attention_out


def attention_gqa_tnd(query, key, value, scale_value, cu_seqlens_q, cu_seqlens_k):
    """
    (T, N, D) 格式的 MHA 注意力前向实现
    T: 所有序列拼接后的总 token 数
    N: num_heads
    D: head_dim

    Args:
        query: (T_q, N, D)
        key: (T_k, N, D)  或 (T_k, N_k, D) 对于 GQA/MQA
        value: (T_k, N, D) 或 (T_k, N_k, D)
        scale_value: float, 缩放因子
        cu_seqlens_q: (B+1,) cumulative sequence lengths for query
        cu_seqlens_k: (B+1,) cumulative sequence lengths for key/value

    Returns:
        softmax_log_max_sum: (T_q, N, 1) log-sum-exp 的 log(max) + log(sum) 用于反向
        attention_out: (T_q, N, D) 注意力输出
    """
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    curr_device = query.device
    t_q, num_heads, head_dim = query.shape
    t_k, num_heads_k, _ = key.shape

    batch_size = len(cu_seqlens_q) - 1
    assert len(cu_seqlens_k) - 1 == batch_size, "cu_seqlens_k batch size mismatch"

    if num_heads % num_heads_k != 0:
        raise ValueError(
            f"query heads ({num_heads}) must be divisible by KV heads ({num_heads_k})"
        )
    group_size = num_heads // num_heads_k

    attention_out = torch.zeros_like(query)
    softmax_log_max_sum = torch.zeros(
        (t_q, num_heads, 1), dtype=torch.float32, device=curr_device
    )

    for b in range(batch_size):
        q_start, q_end = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        k_start, k_end = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        q_b = query[q_start:q_end].transpose(0, 1)
        k_b = key[k_start:k_end].transpose(0, 1).repeat_interleave(group_size, dim=0)
        v_b = value[k_start:k_end].transpose(0, 1).repeat_interleave(group_size, dim=0)

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale_value
        softmax_max = scores.max(dim=-1, keepdim=True).values
        softmax_exp = torch.exp(scores - softmax_max)
        softmax_sum = softmax_exp.sum(dim=-1, keepdim=True)
        softmax_out = softmax_exp / softmax_sum
        lse_b = softmax_max + torch.log(softmax_sum)

        attention_out[q_start:q_end] = torch.matmul(softmax_out, v_b).transpose(0, 1)
        softmax_log_max_sum[q_start:q_end] = lse_b.transpose(0, 1)
    return softmax_log_max_sum.contiguous(), attention_out.to(DTYPE).contiguous()


def generate_base_sparsity_mask(
    max_seqlen_q,
    max_seqlen_k,
    m_block_dim,
    n_block_dim,
    batch_size,
    num_blocksparse_heads,
    sparsity_list,
    causal=False,
):
    assert len(sparsity_list) == num_blocksparse_heads

    def round_to_multiple(x, base):
        return (x + base - 1) // base

    nrow, ncol = (
        round_to_multiple(max_seqlen_q, m_block_dim),
        round_to_multiple(max_seqlen_k, n_block_dim),
    )
    base_mask = torch.zeros(
        batch_size, num_blocksparse_heads, nrow, ncol, dtype=torch.uint8
    )
    # print("generate_base_sparsity_mask ========")
    # print(f"base_mask shape = {base_mask.shape}")
    # print(f"sparsity_list = {sparsity_list}")
    for batch in range(batch_size):
        for head_rank in range(num_blocksparse_heads):
            sparsity = sparsity_list[head_rank]
            if not sparsity == 0.0 and not sparsity == 1.0:
                for i in range(nrow):
                    idx = nrow - i - 1
                    if causal:
                        available_col_num = max(0, ncol - i)
                        num_one = max(1, int(sparsity * available_col_num))
                        base_mask[batch][head_rank][
                            idx, torch.randperm(available_col_num)[:num_one]
                        ] = True
                    else:
                        available_col_num = ncol
                        num_one = max(1, int(sparsity * available_col_num))
                        # print(f"num_one = {num_one}")
                        # print(f"available_col_num = {available_col_num}")
                        # print(f"idx = {idx}")
                        base_mask[batch][head_rank][
                            idx, torch.randperm(available_col_num)[:num_one]
                        ] = True
            elif sparsity == 1.0:
                # print("sparsity == 1.0")
                base_mask[batch][head_rank] = torch.ones_like(
                    base_mask[batch][head_rank]
                )

    return base_mask


def tailor_mixedmask_for_test(spanded_base_mixedmask, seqlen_q, seqlen_k):
    if (
        spanded_base_mixedmask.shape[-2] < seqlen_q
        or spanded_base_mixedmask.shape[-1] < seqlen_k
    ):
        raise ValueError(
            "expanded block mask is smaller than the requested sequence shape: "
            f"mask={tuple(spanded_base_mixedmask.shape)}, requested=({seqlen_q}, {seqlen_k})"
        )
    return (
        spanded_base_mixedmask[:, :, :seqlen_q, :seqlen_k].to(torch.bool).contiguous()
    )


def q_k_mad_bnsd(query, key, blockSparseMask, blockShape, batch_idx=-1, headnum_idx=-1):
    """
    batch_idx=-1
    headnum_idx=-1
    (b, n, s, d)
    """
    # print("q_k_mad_bnsd =====================")
    # print("ker shape: ", key.shape)
    # print("query shape: ", query.shape)
    key_t = torch.transpose(key, 2, 3).contiguous()
    # print("key_t shape: ", query.shape)
    # print("blockSparseMask shape: ", blockSparseMask.shape)
    # print("blockShape: ", blockShape)
    batch_size, num_heads, actual_seqlen_q, _ = query.shape
    actual_seqlen_k = key.shape[2]

    # 获取划分的基本块shape 目前设定为（128,128）
    q_block, kv_block = blockShape[0], blockShape[1]

    # 划分之后， Q 和 k 的块数已经尾块size
    # q_row, k_col = blockSparseMask.shape[-2], blockSparseMask.shape[-1]
    # q_tail_block_size = max_seq_q - (q_row - 1 ) * q_block if q_row * q_block != max_seq_q else 0
    # k_tail_block_size = max_seq_kv - (k_col - 1 ) * kv_block if k_col * kv_block != max_seq_q else 0

    # S = torch.zeros((batch_size, head_num, max_seq_q, max_seq_kv), dtype=query.dtype)
    S = torch.einsum("bnqd, bndk->bnqk", query, key_t)
    # print(f"S.shape: {S.shape}")
    selected_mask = blockSparseMask
    if batch_idx != -1:
        selected_mask = selected_mask[batch_idx : batch_idx + 1, ...]
    if headnum_idx != -1:
        selected_mask = selected_mask[:, headnum_idx : headnum_idx + 1, ...]

    full_mask = repeat(
        selected_mask,
        "b n s_m s_n -> b n (s_m d_m) (s_n d_n)",
        d_m=q_block,
        d_n=kv_block,
    )
    # print(f"full_mask.shape = {full_mask.shape}")
    full_mask = tailor_mixedmask_for_test(full_mask, actual_seqlen_q, actual_seqlen_k)
    # print(f"full_mask.shape = {full_mask.shape}")
    neg_full_mask = ~full_mask  # ?
    # print(f"neg_full_mask.shape: {neg_full_mask.shape}")
    # print(f"neg_full_mask.shape: {neg_full_mask.shape}")
    # S = S.mul(full_mask)
    S = S.masked_fill(neg_full_mask, float("-inf"))
    # print(f"S.shape: {S.shape}")
    # print("q_k_mad_bnsd =====================")
    # print("q_k_mad_bnsd end =====================")
    return S


def torch_batch_dot(a, b):
    return torch.matmul(a.to(torch.float), b.to(torch.float))


def block_mask_for_q_chunk(
    block_sparse_mask, block_shape, q_start, q_end, k_seqlen, batch_idx=-1
):
    """Expand only the block-mask rows required by one Q chunk."""
    q_block, kv_block = block_shape
    selected_mask = block_sparse_mask
    if batch_idx != -1:
        selected_mask = selected_mask[batch_idx : batch_idx + 1]

    mask_k = selected_mask.repeat_interleave(kv_block, dim=-1)[..., :k_seqlen]
    q_block_indices = torch.arange(q_start, q_end, device=mask_k.device) // q_block
    return mask_k.index_select(-2, q_block_indices).to(torch.bool)


def convert_sqlen_idx_to_num(sqlen_idx):
    """
    example : max_sqlen = 117
    sqlen_idx = [ 0, 95, 189, 299, 408, 508]
    sqlen_mum = [ 95, 94, 110, 109, 100]
    """
    batch = len(sqlen_idx) - 1
    sqlen_mum = [0] * batch
    # print(sqlen_mum)
    for i in range(batch - 1, -1, -1):
        sqlen_mum[i] = sqlen_idx[i + 1] - sqlen_idx[i]
    # print(sqlen_mum)
    return sqlen_mum


def bsag_gqa_tnd(
    attention_score_grad_all,
    query,
    key,
    value,
    attention_score_all,
    softmaxLse,
    blockSparseMask,
    blockShape,
    actualSeqLength,
    actualSeqLengthsKv,
    scale_value,
    max_seqlen_q,
    max_seqlen_k,
):
    """
    (T. N, D)
    attention_in: (Tq, N1, d) attention_in
    query : (Tq, N1, d)
    key : (Tkv, N2, d)
    value : ((Tkv, N2, d)
    out : (Tq, N, d) dy
    softmaxLse : (Tq, N, d)
    blockShape : (q_block_szie, k_blcok_size) 暂定(128, 128)
    blockSparseMask : (b, n, [q_s / q_block_szie], [kv_s / k_blcok_size] ) 向上取整
    actualSeqLength : list 序号
    actualSeqLengthsKv : list 序号
    """

    t_q, head_num, head_dim = query.shape
    t_kv, kv_head_num, _ = key.shape
    batch_size = len(actualSeqLength) - 1
    query = query.to(torch.float)
    key = key.to(torch.float)
    value = value.to(torch.float)
    attention_in = attention_score_all.to(torch.float)  # O
    out = attention_score_grad_all.to(torch.float)  # dO
    softmaxLse = softmaxLse.to(torch.float)

    # print(f"query shape : {query.shape}")
    # print(f"key shape : {key.shape}")
    # print(f"value shape : {value.shape}")
    # print(f"attention_in shape : {attention_in.shape}")
    # print(f"out shape : {out.shape}")
    # print(f"softmaxLse shape : {softmaxLse.shape}")
    # print(f"blockSparseMask shape : {blockSparseMask.shape}")
    # print(f"blockShape : {blockShape}")
    # print(f"batch_size : {batch_size}")
    # print(f"head_num : {head_num}")
    # print(f"actualSeqLength : {actualSeqLength}")
    # print(f"actualSeqLengthsKv : {actualSeqLengthsKv}")
    # print(f"max_seqlen_q : {max_seqlen_q}")
    # print(f"max_seqlen_k : {max_seqlen_k}")

    q_len_nums = convert_sqlen_idx_to_num(actualSeqLength)
    kv_len_nums = convert_sqlen_idx_to_num(actualSeqLengthsKv)

    def tnd_to_1nsd(tensor, idx_front, idx_back, nums, head_num, head_dim):
        """
        针对不等长场景
        针对batch为1的格式转换
        idx_front: s 序号首 切片用
        idx_back: s 序号尾 切片用
        nums: s value
        """
        return (
            tensor[idx_front:idx_back, ...]
            .reshape(1, nums, head_num, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    softmaxLse = softmaxLse.reshape(t_q, head_num, 1).contiguous()
    # tnd
    dq = torch.zeros_like(query)
    dk = torch.zeros_like(key)
    dv = torch.zeros_like(value)

    # print("for tnd")
    for i in range(batch_size):
        q_s_num = q_len_nums[i]
        kv_s_num = kv_len_nums[i]
        q_temp = tnd_to_1nsd(
            query,
            actualSeqLength[i],
            actualSeqLength[i + 1],
            q_s_num,
            head_num,
            head_dim,
        )
        atten_in_temp = tnd_to_1nsd(
            attention_in,
            actualSeqLength[i],
            actualSeqLength[i + 1],
            q_s_num,
            head_num,
            head_dim,
        )
        out_temp = tnd_to_1nsd(
            out, actualSeqLength[i], actualSeqLength[i + 1], q_s_num, head_num, head_dim
        )
        k_temp = tnd_to_1nsd(
            key,
            actualSeqLengthsKv[i],
            actualSeqLengthsKv[i + 1],
            kv_s_num,
            kv_head_num,
            head_dim,
        )
        v_temp = tnd_to_1nsd(
            value,
            actualSeqLengthsKv[i],
            actualSeqLengthsKv[i + 1],
            kv_s_num,
            kv_head_num,
            head_dim,
        )
        softmaxLse_temp = tnd_to_1nsd(
            softmaxLse, actualSeqLength[i], actualSeqLength[i + 1], q_s_num, head_num, 1
        )
        lse_tmp = softmaxLse_temp[:, :, 0:q_s_num, :]  # (b, n, q_s_max)

        # bnsd b为1
        dq_t, dk_t, dv_t = bsa_gqa_bnsd_cpu(
            out_temp,
            q_temp,
            k_temp,
            v_temp,
            atten_in_temp,
            lse_tmp,
            blockSparseMask,
            blockShape,
            actualSeqLength,
            actualSeqLengthsKv,
            scale_value,
            max_seqlen_q,
            max_seqlen_k,
            i,
        )
        # print(f"dq_t shape : {dq_t.shape}")
        # print(f"dk_t shape : {dk_t.shape}")
        # print(f"dv_t shape : {dv_t.shape}")
        # bsnd b为1
        dq_t = dq_t.permute(0, 2, 1, 3)
        dk_t = dk_t.permute(0, 2, 1, 3)
        dv_t = dv_t.permute(0, 2, 1, 3)
        # print("transpose after")
        # print(f"dq_t shape : {dq_t.shape}")
        # print(f"dk_t shape : {dk_t.shape}")
        # print(f"dv_t shape : {dv_t.shape}")

        # print(f"dq shape : {dq.shape}")
        # print(f"dk shape : {dk.shape}")
        # print(f"dv shape : {dv.shape}")
        dq[actualSeqLength[i] : actualSeqLength[i + 1], ...] = dq_t[0:1, ...]
        dk[actualSeqLengthsKv[i] : actualSeqLengthsKv[i + 1], ...] = dk_t[0:1, ...]
        dv[actualSeqLengthsKv[i] : actualSeqLengthsKv[i + 1], ...] = dv_t[0:1, ...]
    return dq, dk, dv


def bsa_gqa_bnsd_cpu(
    attention_score_grad_all,
    query_all,
    key_all,
    value_all,
    attention_score_all,
    softmax_log_max_sum_all,
    blockSparseMask,
    blockShape,
    actualSeqLength,
    actualSeqLengthsKv,
    scale_value,
    max_seqlen_q,
    max_seqlen_k,
    batch_idx,
):
    """
    (b, n, s, d)
    """
    # value_grad, key_grad, query_grad
    nG = query_all.shape[1] // key_all.shape[1]
    batch, head_num, seq_size, head_dim = query_all.shape

    # Match the K_OUT kernel: accumulate all head/tile contributions in FP32, then cast once in gen_golden.
    query_grad = torch.zeros_like(query_all, dtype=torch.float32)
    key_grad = torch.zeros_like(key_all, dtype=torch.float32)
    value_grad = torch.zeros_like(value_all, dtype=torch.float32)

    for i in range(query_all.shape[1]):
        # q的每连续 ng 个n 与kv的1个n 对应
        g = i // nG
        # print(i)
        # if i != 5:
        #     continue
        attention_score = attention_score_all[:, i : i + 1, :, :].to(torch.float)
        attention_score_grad = attention_score_grad_all[:, i : i + 1, :, :].to(
            torch.float
        )
        softmax_log_max_sum = softmax_log_max_sum_all[:, i : i + 1, :].to(torch.float)
        query = query_all[:, i : i + 1, :, :].to(torch.float)

        key = key_all[:, g : g + 1, :, :].to(torch.float)
        value = value_all[:, g : g + 1, :, :].to(torch.float)

        batch_size, head_num, qSeqlen, head_dim = query.shape
        batch_size, head_num, kSeqlen, head_dim = key.shape

        # print(batch_size, head_num, qSeqlen)
        # print("softmax_log_max_sum shape: ", softmax_log_max_sum.size())
        # print("softmax_log_max_sum_all shape: ", softmax_log_max_sum_all.size())
        softmax_log_max_sum = softmax_log_max_sum.view(
            (batch_size, head_num, qSeqlen, 1)
        )

        # D
        softmax_out_sum = (
            (
                attention_score.to(torch.float).detach()
                * attention_score_grad.to(torch.float).detach()
            )
            .sum(-1, keepdim=True)
            .reshape((batch_size, head_num, qSeqlen, 1))
        )

        # intermediate calculation as fp32
        S = q_k_mad_bnsd(query, key, blockSparseMask, blockShape, batch_idx, i)
        dP = torch_batch_dot(attention_score_grad, torch.transpose(value, 2, 3)).to(
            torch.float
        )

        S = S * scale_value
        p = torch.exp(S - softmax_log_max_sum)

        dS = (dP - softmax_out_sum) * p
        bmm1_res_grad = dS.to(DTYPE)

        query_grad[:, i : i + 1, :, :] = (
            torch_batch_dot(bmm1_res_grad.to(torch.float), key) * scale_value
        )
        key_grad[:, g : g + 1, :, :] += (
            torch_batch_dot(torch.transpose(bmm1_res_grad, 2, 3).to(torch.float), query)
            * scale_value
        )
        value_grad[:, g : g + 1, :, :] += torch_batch_dot(
            torch.transpose(p.to(DTYPE), 2, 3).to(torch.float), attention_score_grad
        )

    return query_grad, key_grad, value_grad


def bsa_gqa_bnsd_cpu_vectorized(
    attention_score_grad_all,
    query_all,
    key_all,
    value_all,
    attention_score_all,
    softmax_log_max_sum_all,
    blockSparseMask,
    blockShape,
    actualSeqLength,
    actualSeqLengthsKv,
    scale_value,
    max_seqlen_q,
    max_seqlen_k,
    batch_idx,
):
    """Vectorized BNSD Golden with bounded Q-dependent temporary tensors."""
    batch, num_heads, q_seqlen, head_dim = query_all.shape
    num_kv_heads = key_all.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"query heads ({num_heads}) must be divisible by KV heads ({num_kv_heads})"
        )
    group_size = num_heads // num_kv_heads
    k_seqlen = key_all.shape[2]
    q_chunk_size = int(os.getenv("BSAG_GOLDEN_Q_CHUNK_SIZE", "512"))
    if q_chunk_size <= 0:
        raise ValueError("BSAG_GOLDEN_Q_CHUNK_SIZE must be a positive integer")

    # Convert once; low-precision casts below intentionally match the kernel reference.
    query = query_all.to(torch.float32)
    key = key_all.to(torch.float32)
    value = value_all.to(torch.float32)
    attention_score = attention_score_all.to(torch.float32)
    attention_score_grad = attention_score_grad_all.to(torch.float32)
    softmax_log_max_sum = softmax_log_max_sum_all.to(torch.float32).reshape(
        batch, num_heads, q_seqlen, 1
    )
    key_expanded = key.repeat_interleave(group_size, dim=1)
    value_expanded = value.repeat_interleave(group_size, dim=1)

    # Match K_OUT FP32 accumulation and bound the Q-dependent temporary tensors.
    query_grad = torch.zeros_like(query)
    key_grad = torch.zeros_like(key)
    value_grad = torch.zeros_like(value)

    for q_start in range(0, q_seqlen, q_chunk_size):
        q_end = min(q_start + q_chunk_size, q_seqlen)
        query_chunk = query[:, :, q_start:q_end, :]
        d_out_chunk = attention_score_grad[:, :, q_start:q_end, :]
        out_chunk = attention_score[:, :, q_start:q_end, :]
        lse_chunk = softmax_log_max_sum[:, :, q_start:q_end, :]

        scores = torch.matmul(query_chunk, key_expanded.transpose(-2, -1))
        mask_chunk = block_mask_for_q_chunk(
            blockSparseMask, blockShape, q_start, q_end, k_seqlen, batch_idx
        )
        scores = scores.masked_fill(~mask_chunk, float("-inf")) * scale_value
        probability = torch.exp(scores - lse_chunk)

        softmax_out_sum = (out_chunk.detach() * d_out_chunk.detach()).sum(
            -1, keepdim=True
        )
        d_probability = torch.matmul(d_out_chunk, value_expanded.transpose(-2, -1))
        d_scores = (d_probability - softmax_out_sum) * probability
        d_scores_low_precision = d_scores.to(DTYPE).to(torch.float32)

        query_grad[:, :, q_start:q_end, :] = (
            torch.matmul(d_scores_low_precision, key_expanded) * scale_value
        )
        key_grad += (
            torch.matmul(d_scores_low_precision.transpose(-2, -1), query_chunk)
            .reshape(batch, num_kv_heads, group_size, k_seqlen, head_dim)
            .sum(dim=2)
            * scale_value
        )
        value_grad += (
            torch.matmul(
                probability.to(DTYPE).to(torch.float32).transpose(-2, -1), d_out_chunk
            )
            .reshape(batch, num_kv_heads, group_size, k_seqlen, head_dim)
            .sum(dim=2)
        )

    return query_grad, key_grad, value_grad


def gen_golden(
    query,
    key,
    value,
    d_out,
    block_sparse_mask,
    attention_out_cpu,
    softmax_lse_cpu,
    block_shape,
    scale_value,
    S1_max,
    S2_max,
    accumulate_seq_length,
    accumulate_seq_length_kv,
    format_type,
):
    query_grad = torch.zeros_like(query, dtype=torch.float32)
    key_grad = torch.zeros_like(key, dtype=torch.float32)
    value_grad = torch.zeros_like(value, dtype=torch.float32)
    layout = format_type.lower()
    if layout == "tnd":
        cpu_dq, cpu_dk, cpu_dv = bsag_gqa_tnd(
            d_out,
            query,
            key,
            value,
            attention_out_cpu,
            softmax_lse_cpu,
            block_sparse_mask,
            block_shape,
            accumulate_seq_length,
            accumulate_seq_length_kv,
            scale_value,
            S1_max,
            S2_max,
        )
    elif layout == "bnsd":
        cpu_dq, cpu_dk, cpu_dv = bsa_gqa_bnsd_cpu(
            d_out,
            query,
            key,
            value,
            attention_out_cpu,
            softmax_lse_cpu,
            block_sparse_mask,
            block_shape,
            accumulate_seq_length,
            accumulate_seq_length_kv,
            scale_value,
            S1_max,
            S2_max,
            -1,
        )
    else:
        raise ValueError(f"unsupported layout: {format_type!r}")

    return cpu_dq.to(query.dtype), cpu_dk.to(query.dtype), cpu_dv.to(query.dtype)


def generate_seqlen(
    max_seqlen_q,  # q 的最大序列长度
    max_seqlen_kv,  # kv 的最大序列长度
    batch_size,  # q kv 的 batch size
):
    """
    根据 max_seqlen 和 batch size accumulate_seq_length
    每个样本的序列长度会在 [1, max_seqlen] 范围内随机生成
    """
    # 1. 随机生成每个样本的序列长度 (范围: 1 到 max_seqlen)
    seqlen_q_list = [random.randint(1, max_seqlen_q) for _ in range(batch_size)]
    seqlen_kv_list = [random.randint(1, max_seqlen_kv) for _ in range(batch_size)]

    max_idx = seqlen_q_list.index(max(seqlen_q_list))
    # max_idx = max(range(batch_size), key=lambda i: seqlen_q_list[i])
    seqlen_q_list[max_idx] = max_seqlen_q

    max_idx = seqlen_kv_list.index(max(seqlen_kv_list))
    # max_idx = max(range(batch_size), key=lambda i: seqlen_kv_list[i])
    seqlen_kv_list[max_idx] = max_seqlen_kv

    # seqlen_q_list = [10 for _ in range(batch_size)]
    # seqlen_kv_list = [10 for _ in range(batch_size)]

    # 2. 计算总序列长度
    total_seqlen_q = sum(seqlen_q_list)
    total_seqlen_kv = sum(seqlen_kv_list)

    actual_q_seq_max = max(seqlen_q_list)
    actual_kv_seq_max = max(seqlen_kv_list)

    # 3. 生成 accumulate_seq_length 索引列表
    accumulate_seq_length = [0]
    for length in seqlen_q_list:
        accumulate_seq_length.append(accumulate_seq_length[-1] + length)

    accumulate_seq_lengths_kv = [0]
    for length in seqlen_kv_list:
        accumulate_seq_lengths_kv.append(accumulate_seq_lengths_kv[-1] + length)

    return (
        accumulate_seq_length,
        accumulate_seq_lengths_kv,
        seqlen_q_list,
        seqlen_kv_list,
        total_seqlen_q,
        total_seqlen_kv,
        actual_q_seq_max,
        actual_kv_seq_max,
    )


def gen_data(params, block_shape):
    """
    Arguments:
    params: [B, N1, S1_max, S2_max, D, num_kv_heads sparsity mha_type format_type]
    block_shape: (block_shape_x, block_shape_y)
    """
    # B, N1, S1_max, S2_max, D，num_kv_heads sparsity mha_type format_type block shape
    B, N, S1, S2, D = params[0], params[1], params[2], params[3], params[4]
    scale_value = 1.0 / math.sqrt(D)
    # scale_value = 1.0
    sparsity = params[6]
    mha_type = params[7]
    format_type = params[8].lower()
    if format_type not in ("tnd", "bnsd"):
        raise ValueError(f"layout must be 'TND' or 'BNSD', got {format_type!r}")
    num_kv_heads = N if mha_type == "mha" else params[5]
    assert N % num_kv_heads == 0

    if format_type == "tnd":
        (
            accumulate_seq_length,
            accumulate_seq_length_kv,
            seqlen_q_list,
            seqlen_kv_list,
            total_seqlen_q,
            total_seqlen_kv,
            actual_q_seq_max,
            actual_kv_seq_max,
        ) = generate_seqlen(S1, S2, B)
        query = torch.randn(total_seqlen_q, N, D, dtype=DTYPE)
        key = torch.randn(total_seqlen_kv, num_kv_heads, D, dtype=DTYPE)
        value = torch.randn(total_seqlen_kv, num_kv_heads, D, dtype=DTYPE)
        d_out = torch.randn(total_seqlen_q, N, D, dtype=DTYPE)
    else:
        seqlen_q_list = [S1] * B
        seqlen_kv_list = [S2] * B
        accumulate_seq_length = [index * S1 for index in range(B + 1)]
        accumulate_seq_length_kv = [index * S2 for index in range(B + 1)]
        total_seqlen_q = B * S1
        total_seqlen_kv = B * S2
        actual_q_seq_max = S1
        actual_kv_seq_max = S2
        query = torch.randn(B, N, S1, D, dtype=DTYPE)
        key = torch.randn(B, num_kv_heads, S2, D, dtype=DTYPE)
        value = torch.randn(B, num_kv_heads, S2, D, dtype=DTYPE)
        d_out = torch.randn(B, N, S1, D, dtype=DTYPE)

    # query = torch.ones(total_seqlen_q, N, D, dtype=DTYPE)
    # key = torch.ones(total_seqlen_kv, num_kv_heads, D,  dtype=DTYPE)
    # key = torch.full((total_seqlen_kv, num_kv_heads, D), 2.0, dtype=DTYPE)
    # value = torch.ones(total_seqlen_kv, num_kv_heads, D, dtype=DTYPE)
    # d_out = torch.full((total_seqlen_q, N, D), 0.1, dtype=DTYPE)
    # d_out = torch.tril(torch.ones((B, N, S1, D), dtype=DTYPE)).permute(0, 2, 1, 3).reshape(total_seqlen_q, N, D).contiguous()
    # print(d_out)

    # 生成mask
    # packed GQA 要求同一 KV head 对应的 group 内所有 Q head 使用完全相同的稀疏 pattern。
    num_blocksparse_heads = num_kv_heads
    sparsity_list = [sparsity] * num_blocksparse_heads
    kv_block_sparse_mask = generate_base_sparsity_mask(
        actual_q_seq_max,
        actual_kv_seq_max,
        block_shape[0],
        block_shape[1],
        B,
        num_blocksparse_heads,
        sparsity_list,
    )
    block_sparse_mask = kv_block_sparse_mask.repeat_interleave(
        N // num_kv_heads, dim=1
    ).contiguous()
    if format_type == "tnd":
        softmax_lse_cpu, attention_out_cpu = attention_gqa_tnd(
            query,
            key,
            value,
            scale_value,
            accumulate_seq_length,
            accumulate_seq_length_kv,
        )
    else:
        softmax_lse_cpu, attention_out_cpu = attention_ori(
            query, key, value, scale_value
        )

    return (
        query,
        key,
        value,
        d_out,
        block_sparse_mask,
        attention_out_cpu,
        softmax_lse_cpu,
        block_shape,
        num_kv_heads,
        scale_value,
        S1,
        S2,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        seqlen_q_list,
        seqlen_kv_list,
    )


def excute_bsag(
    query,
    key,
    value,
    d_out,
    rsvd_block_idx,
    rsvd_block_count,
    attention_out_cpu,
    softmax_lse_cpu,
    block_shape,
    num_kv_heads,
    scale_value,
    S1_max,
    S2_max,
    seqlen_q_list,
    seqlen_kv_list,
    format_type,
):
    query = query.to(DEVICE)
    key = key.to(DEVICE)
    value = value.to(DEVICE)
    d_out = d_out.to(DEVICE)
    rsvd_block_idx = rsvd_block_idx.to(DEVICE)
    rsvd_block_count = rsvd_block_count.to(DEVICE)
    attention_out = attention_out_cpu.to(DEVICE)
    softmax_lse_out = softmax_lse_cpu.to(DEVICE)
    # print("query dtye: ", query.dtype)
    # print("key dtye: ", key.dtype)
    # print("value dtye: ", value.dtype)
    # print("attention_out dtye: ", attention_out.dtype)
    # print("d_out dtye: ", d_out.dtype)
    # print("block_sparse_mask dtye: ", block_sparse_mask.dtype)
    # print("softmax_lse_out dtye: ", softmax_lse_out.dtype)
    # print("query dtye: ", query.dtype)
    # print("key dtye: ", key.dtype)
    # print("value dtye: ", value.dtype)

    d_query, d_key, d_value = generic_block_sparse_attention_grad(
        d_out,
        query,
        key,
        value,
        attention_out,
        softmax_lse_out,
        rsvd_block_idx,
        rsvd_block_count,
        block_shape=block_shape,
        actual_seq_lengths=seqlen_q_list,
        actual_seq_lengths_kv=seqlen_kv_list,
        q_input_layout=format_type.upper(),
        kv_input_layout=format_type.upper(),
        num_key_value_heads=num_kv_heads,
        scale_value=scale_value,
    )

    return d_query, d_key, d_value


def run_test_case(case_id, full_case):
    case_params = list(full_case[:9])
    case_block_shape = tuple(full_case[9])
    format_type = case_params[8].lower()
    # seed_everything(SEED + case_id)
    seed_everything(SEED)

    print_case_begin(case_id, full_case)
    (
        query,
        key,
        value,
        d_out,
        block_sparse_mask,
        attention_out_cpu,
        softmax_lse_cpu,
        block_shape,
        num_kv_heads,
        scale_value,
        S1_max,
        S2_max,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        seqlen_q_list,
        seqlen_kv_list,
    ) = gen_data(case_params, case_block_shape)
    print(
        f"CASE_DATA id={case_id:03d} actual_q={seqlen_q_list} actual_kv={seqlen_kv_list} "
        f"tq={accumulate_seq_length[-1]} tkv={accumulate_seq_length_kv[-1]}",
        flush=True,
    )

    cpu_dq, cpu_dk, cpu_dv = gen_golden(
        query,
        key,
        value,
        d_out,
        block_sparse_mask,
        attention_out_cpu,
        softmax_lse_cpu,
        block_shape,
        scale_value,
        S1_max,
        S2_max,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        format_type,
    )
    print(f"CASE_GOLDEN id={case_id:03d} accumulation_dtype=float32", flush=True)

    query_npu = query.to(DEVICE)
    key_npu = key.to(DEVICE)
    value_npu = value.to(DEVICE)
    d_out_npu = d_out.to(DEVICE)
    rsvd_block_idx_cpu, rsvd_block_count_cpu = blcok_sparse_mask_to_idx(
        block_sparse_mask,
        list(block_shape),
        seqlen_q_list,
        seqlen_kv_list,
        S1_max,
        S2_max,
        "kv2qIdx",
        q_head_num=case_params[1],
        kv_head_num=num_kv_heads,
        is_packed_gqa=True,
    )
    rsvd_block_idx_npu = rsvd_block_idx_cpu.to(DEVICE)
    rsvd_block_count_npu = rsvd_block_count_cpu.to(DEVICE)
    attention_out_npu = attention_out_cpu.to(DEVICE)
    softmax_lse_npu = softmax_lse_cpu.to(DEVICE)

    repetitions = int(os.getenv("BSAG_REPETITIONS", "1"))
    # repetitions=15
    for repetition in range(repetitions):
        d_query, d_key, d_value = excute_bsag(
            query_npu,
            key_npu,
            value_npu,
            d_out_npu,
            rsvd_block_idx_npu,
            rsvd_block_count_npu,
            attention_out_npu,
            softmax_lse_npu,
            block_shape,
            num_kv_heads,
            scale_value,
            S1_max,
            S2_max,
            seqlen_q_list,
            seqlen_kv_list,
            format_type,
        )
        torch.npu.synchronize()
        print(
            f"CASE_NPU id={case_id:03d} repetition={repetition} "
            f"dq_shape={tuple(d_query.shape)} dk_shape={tuple(d_key.shape)} dv_shape={tuple(d_value.shape)} ********************",
            flush=True,
        )
        if not compare_golden(cpu_dq, cpu_dk, cpu_dv, d_query, d_key, d_value):
            raise AssertionError("BSAG accuracy comparison failed")

    print_case_end(case_id, "PASS")


if __name__ == "__main__":
    case_file = Path(
        os.getenv("BSAG_CASE_FILE", Path(__file__).with_name("test_case.txt"))
    )
    all_cases = load_test_cases(case_file)
    selected_cases = select_test_cases(all_cases)
    continue_on_failure = os.getenv("BSAG_CONTINUE_ON_FAILURE", "1") != "0"
    failures = []
    print(
        f"{CASE_SEPARATOR}\nSUITE_BEGIN case_file={case_file} total_cases={len(all_cases)} selected_cases={len(selected_cases)} "
        f"dtype={DTYPE_NAME} rtol={RTOL} atol={ATOL}",
        flush=True,
    )
    print(CASE_SEPARATOR, flush=True)

    for case_id, full_case in selected_cases:
        try:
            run_test_case(case_id, full_case)
        except Exception as error:
            failures.append((case_id, str(error)))
            print(f"CASE_ERROR id={case_id:03d} error={error!r}", flush=True)
            traceback.print_exc(file=sys.stdout)
            print_case_end(case_id, "FAIL", error=str(error))
            if not continue_on_failure:
                raise
        finally:
            gc.collect()
            try:
                torch.npu.empty_cache()
            except Exception:
                pass

    print(
        f"SUITE_END status={'FAIL' if failures else 'PASS'} total={len(selected_cases)} "
        f"passed={len(selected_cases) - len(failures)} failed={len(failures)}",
        flush=True,
    )
    for case_id, error in failures:
        print(f"SUITE_FAILURE id={case_id:03d} error={error}", flush=True)
    print(CASE_SEPARATOR, flush=True)
    if failures:
        raise SystemExit(1)
