import importlib
import importlib.util
import math
from math import ceil
from pathlib import Path

import torch
import torch_npu

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
import sys
sys.path.insert(0, str(_TORCH_EXT))


def _register_sparse_attention_score_op():
    spec = importlib.util.spec_from_file_location(
        "sparse_attention_score",
        _TORCH_EXT / "cann_ops_transformer" / "ops" / "sparse_attention_score.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cache_base = Path.home() / ".cache" / "torch_extensions"
    candidate_dirs = [cache_base / "py311_npu" / "npu_sparse_attention_score"]
    if cache_base.is_dir():
        candidate_dirs.extend(
            sub / "npu_sparse_attention_score"
            for sub in cache_base.iterdir()
            if "npu" in sub.name.lower()
        )

    so_file = None
    for so_dir in candidate_dirs:
        if so_dir.is_dir():
            for f in so_dir.iterdir():
                if f.suffix == ".so":
                    so_file = f
                    break
        if so_file is not None:
            break

    if so_file is None:
        print("[INFO] Pre-compiled .so not found; OpBuilder will compile on first NPU call.")
        return

    ext_spec = importlib.util.spec_from_file_location("npu_sparse_attention_score", so_file)
    ext_mod = importlib.util.module_from_spec(ext_spec)
    ext_spec.loader.exec_module(ext_mod)

    from cann_ops_transformer.op_builder.builder import OpBuilder
    OpBuilder._loaded_ops["npu_sparse_attention_score"] = ext_mod
    print(f"[INFO] Loaded pre-compiled extension: {so_file}")


_register_sparse_attention_score_op()


def generate_block_index_with_causal(query_fp32, key_fp32, q_seqlen, kv_seqlen,
                                     kv_heads, group_size, block_size=128, top_k=16):
    his_seq_len = kv_seqlen - q_seqlen
    total_blocks = ceil(kv_seqlen / block_size)
    select_idx = torch.full((kv_heads, q_seqlen, top_k), -1, dtype=torch.int32)
    select_num_idx = torch.zeros((kv_heads, q_seqlen), dtype=torch.int32)

    for kv_head in range(kv_heads):
        representative_q_head = kv_head * group_size
        k_head = key_fp32[:, kv_head, :]

        for q_token in range(q_seqlen):
            q_vec = query_fp32[q_token, representative_q_head, :]
            causal_bound = his_seq_len + q_token

            scores = torch.matmul(q_vec, k_head[:kv_seqlen, :].transpose(0, 1))

            pooled = torch.full((total_blocks,), -float("inf"), dtype=torch.float32)
            q_block = causal_bound // block_size

            for block_idx in range(total_blocks):
                block_begin = block_idx * block_size
                block_end = min(block_begin + block_size, kv_seqlen)

                if block_idx > q_block:
                    pooled[block_idx] = -float("inf")
                elif block_idx == q_block:
                    pooled[block_idx] = float("inf")
                else:
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end > block_begin:
                        pooled[block_idx] = torch.max(scores[block_begin:effective_end]).item()

            visible_blocks = min(total_blocks, q_block + 1)
            valid_k = min(top_k, visible_blocks)
            select_num_idx[kv_head, q_token] = valid_k
            if valid_k > 0:
                topk_indices = torch.topk(pooled, k=valid_k, largest=True).indices.to(torch.int32)
                select_idx[kv_head, q_token, :valid_k] = topk_indices

    return select_idx, select_num_idx


def generate_block_table(batch, max_blocks_per_batch, shuffle=True):
    total_physical = batch * max_blocks_per_batch
    all_physical_ids = list(range(total_physical))
    if shuffle:
        import random
        rng = random.Random(137)
        rng.shuffle(all_physical_ids)
    block_table = torch.zeros(batch, max_blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = all_physical_ids[b * max_blocks_per_batch + i]
    return block_table


def cpu_sparse_attention_score_fp32(
    query, key, value, select_idx, block_table,
    actual_seq_lengths, actual_seq_lengths_kv,
    num_key_value_heads, select_num_idx=None,
    block_size=128, scale_value=1.0,
):
    select_idx_cpu = select_idx.to(torch.int64)
    block_table_cpu = block_table.to(torch.int64)

    total_q_tokens, q_heads, head_dim = query.shape
    kv_heads = num_key_value_heads
    group_size = q_heads // kv_heads
    top_k = select_idx.shape[2]
    batch = len(actual_seq_lengths)

    output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.float32)
    q_offset = 0
    for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
        kv_seqlen = actual_seq_lengths_kv[batch_idx]
        history_len = kv_seqlen - q_seqlen

        for q_token_in_batch in range(q_seqlen):
            global_q_token = q_offset + q_token_in_batch
            causal_bound = history_len + q_token_in_batch

            for kv_head in range(kv_heads):
                valid_top_k = top_k
                if select_num_idx is not None:
                    valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
                    valid_top_k = min(valid_top_k, top_k)
                if valid_top_k == 0:
                    continue

                q_start = kv_head * group_size
                q_group = query[global_q_token, q_start:q_start + group_size, :].float()

                last_max = torch.full((group_size,), -float("inf"), dtype=torch.float32)
                last_sum = torch.zeros(group_size, dtype=torch.float32)
                o_acc = torch.zeros(group_size, head_dim, dtype=torch.float32)
                is_first = True

                for topk_idx in range(valid_top_k):
                    logical_id = int(select_idx_cpu[kv_head, global_q_token, topk_idx].item())
                    if logical_id < 0:
                        continue
                    block_begin = logical_id * block_size
                    block_end = min(block_begin + block_size, kv_seqlen)
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue

                    physical_id = int(block_table_cpu[batch_idx, logical_id].item())
                    valid_len = effective_end - block_begin
                    k_block = key[physical_id, :valid_len, kv_head, :].float()
                    v_block = value[physical_id, :valid_len, kv_head, :].float()

                    s = torch.matmul(q_group, k_block.t())
                    s_scaled = s * scale_value

                    now_max = s_scaled.max(dim=1).values
                    if not is_first:
                        now_max = torch.max(now_max, last_max)

                    p = torch.exp(s_scaled - now_max.unsqueeze(1))
                    now_sum = p.sum(dim=1)

                    if is_first:
                        last_sum = now_sum
                        last_max = now_max
                    else:
                        correction = torch.exp(last_max - now_max)
                        last_sum = correction * last_sum + now_sum
                        last_max = now_max

                    pv = torch.matmul(p, v_block)
                    if is_first:
                        o_acc = pv
                    else:
                        o_acc = o_acc * correction.unsqueeze(1) + pv

                    is_first = False

                if last_sum.max() > 0:
                    result = o_acc / last_sum.unsqueeze(1)
                    output[global_q_token, q_start:q_start + group_size, :] = result

        q_offset += q_seqlen

    return output


def run_single_case():
    q_seqlen = 1
    kv_seqlen = 256
    q_heads = 8
    kv_heads = 2
    head_dim = 128
    block_size = 128
    top_k = 2
    seed = 42
    batch = 1
    total_blocks = ceil(kv_seqlen / block_size)
    max_blocks_per_batch = total_blocks
    actual_seq_lengths = [q_seqlen] * batch
    actual_seq_lengths_kv = [kv_seqlen] * batch

    torch.manual_seed(seed)
    query_fp32 = torch.rand(q_seqlen, q_heads, head_dim, dtype=torch.float32) * 2 - 1
    total_physical_blocks = total_blocks * batch
    key_fp32 = torch.rand(
        total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
    value_fp32 = torch.rand(
        total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1

    block_table = generate_block_table(batch, max_blocks_per_batch)
    key_logical = torch.zeros(total_blocks * block_size, kv_heads, head_dim, dtype=torch.float32)
    for logical_id in range(total_blocks):
        physical_id = int(block_table[0, logical_id].item())
        key_logical[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
    key_flat = key_logical[:kv_seqlen, :, :]
    group_size = q_heads // kv_heads
    select_idx, select_num_idx = generate_block_index_with_causal(
        query_fp32, key_flat, q_seqlen, kv_seqlen,
        kv_heads, group_size, block_size, top_k)
    scale_value = 1.0 / math.sqrt(head_dim)

    query_bf16 = query_fp32.to(torch.bfloat16)
    key_bf16 = key_fp32.to(torch.bfloat16)
    value_bf16 = value_fp32.to(torch.bfloat16)

    print(f"query: {query_bf16.shape}, key: {key_bf16.shape}, value: {value_bf16.shape}")
    print(f"select_idx: {select_idx.shape}, block_table: {block_table.shape}")

    cpu_out = cpu_sparse_attention_score_fp32(
        query_bf16, key_bf16, value_bf16, select_idx, block_table,
        actual_seq_lengths, actual_seq_lengths_kv,
        num_key_value_heads=kv_heads,
        select_num_idx=select_num_idx,
        block_size=block_size,
        scale_value=scale_value,
    )

    npu_out = torch_npu.npu_sparse_attention_score(
        query_bf16.npu(), key_bf16.npu(), value_bf16.npu(),
        select_idx.npu(), block_table.npu(),
        select_num_idx=select_num_idx.npu(),
        actual_seq_lengths=torch.tensor(actual_seq_lengths, dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor(actual_seq_lengths_kv, dtype=torch.int32).npu(),
        num_key_value_heads=kv_heads,
        scale_value=scale_value,
        block_size=block_size,
        top_k=top_k,
    )

    npu_out_cpu = npu_out.cpu()

    diff = (npu_out_cpu.float() - cpu_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff < 0.02:
        print("[PASS] NPU result matches CPU golden within tolerance.")
    else:
        print("[FAIL] max_diff too large!")
        print("CPU sample:", cpu_out[0, 0, :8].float())
        print("NPU sample:", npu_out_cpu[0, 0, :8].float())


def make_case_multi_batch(batch=2, q_seqlens=None, kv_seqlens=None,
                          q_heads=8, kv_heads=2, head_dim=128, block_size=128,
                          top_k=4, seed=42):
    if q_seqlens is None:
        q_seqlens = [1] * batch
    if kv_seqlens is None:
        kv_seqlens = [256] * batch
    assert len(q_seqlens) == batch and len(kv_seqlens) == batch
    group_size = q_heads // kv_heads
    max_kv_seqlen = max(kv_seqlens)
    max_blocks_per_batch = ceil(max_kv_seqlen / block_size)
    total_q_tokens = sum(q_seqlens)
    total_physical_blocks = max_blocks_per_batch * batch
    torch.manual_seed(seed)
    query_fp32 = torch.rand(total_q_tokens, q_heads, head_dim, dtype=torch.float32) * 2 - 1
    key_fp32 = torch.rand(
        total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
    value_fp32 = torch.rand(
        total_physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1
    block_table = generate_block_table(batch, max_blocks_per_batch)
    select_idx = torch.full((kv_heads, total_q_tokens, top_k), -1, dtype=torch.int32)
    select_num_idx = torch.zeros((kv_heads, total_q_tokens), dtype=torch.int32)
    q_offset = 0
    for b in range(batch):
        q_seqlen_b = q_seqlens[b]
        kv_seqlen_b = kv_seqlens[b]
        total_blocks_b = ceil(kv_seqlen_b / block_size)
        key_logical_b = torch.zeros(
            total_blocks_b * block_size, kv_heads, head_dim, dtype=torch.float32)
        for logical_id in range(total_blocks_b):
            physical_id = int(block_table[b, logical_id].item())
            key_logical_b[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
        key_flat_b = key_logical_b[:kv_seqlen_b, :, :]
        q_for_batch = query_fp32[q_offset:q_offset + q_seqlen_b, :, :]
        batch_select_idx, batch_select_num = generate_block_index_with_causal(
            q_for_batch, key_flat_b, q_seqlen_b, kv_seqlen_b,
            kv_heads, group_size, block_size, top_k)
        select_idx[:, q_offset:q_offset + q_seqlen_b, :] = batch_select_idx
        select_num_idx[:, q_offset:q_offset + q_seqlen_b] = batch_select_num
        q_offset += q_seqlen_b
    scale_value = 1.0 / math.sqrt(head_dim)
    actual_seq_lengths_t = torch.tensor(q_seqlens, dtype=torch.int32)
    actual_seq_lengths_kv_t = torch.tensor(kv_seqlens, dtype=torch.int32)
    return (
        query_fp32.to(torch.bfloat16),
        key_fp32.to(torch.bfloat16),
        value_fp32.to(torch.bfloat16),
        select_idx,
        block_table,
        select_num_idx,
        actual_seq_lengths_t,
        actual_seq_lengths_kv_t,
        kv_heads,
        block_size,
        top_k,
        scale_value,
    )

def run_multi_batch_case(batch=2, q_seqlens=None, kv_seqlens=None,
                         q_heads=8, kv_heads=2, top_k=4, seed=42, head_dim=128, block_size=128):
    case_data = make_case_multi_batch(
        batch=batch, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
        q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
        block_size=block_size, top_k=top_k, seed=seed)
    (query, key, value, select_idx, block_table, select_num_idx,
     actual_seq_lengths, actual_seq_lengths_kv,
     kv_heads, block_size, top_k, scale_value) = case_data

    cpu_out = cpu_sparse_attention_score_fp32(
        query, key, value, select_idx, block_table,
        actual_seq_lengths.tolist(), actual_seq_lengths_kv.tolist(),
        num_key_value_heads=kv_heads,
        select_num_idx=select_num_idx,
        block_size=block_size,
        scale_value=scale_value)

    npu_out = torch_npu.npu_sparse_attention_score(
        query.npu(), key.npu(), value.npu(), select_idx.npu(), block_table.npu(),
        select_num_idx=select_num_idx.npu(),
        actual_seq_lengths=actual_seq_lengths.npu(),
        actual_seq_lengths_kv=actual_seq_lengths_kv.npu(),
        num_key_value_heads=kv_heads,
        scale_value=scale_value,
        block_size=block_size,
        top_k=top_k)

    npu_out_cpu = npu_out.cpu()
    diff = (npu_out_cpu.float() - cpu_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[multi-batch] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    if max_diff < 0.02:
        print("[PASS]")
    else:
        print("[FAIL]")

if __name__ == "__main__":
    print("=== Single batch ===")
    run_single_case()
    print()
    print("=== Multi batch 2batch same seqlen ===")
    run_multi_batch_case(batch=2, q_seqlens=[1, 1], kv_seqlens=[256, 256],
                         q_heads=8, kv_heads=2, top_k=2, seed=42)
    print()
    print("=== Multi batch 2batch diff kvseqlen ===")
    run_multi_batch_case(batch=2, q_seqlens=[1, 1], kv_seqlens=[256, 512],
                         q_heads=8, kv_heads=2, top_k=2, seed=42)
    print()
    print("=== Multi batch 4batch mixed ===")
    run_multi_batch_case(batch=4, q_seqlens=[1, 2, 4, 1], kv_seqlens=[300, 500, 1000, 700],
                         q_heads=8, kv_heads=2, top_k=4, seed=7)