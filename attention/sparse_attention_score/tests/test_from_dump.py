"""
Load dumped tensors from dump.tar.gz and dump_nv.tar.gz, run SparseAttentionScore
on NPU, and compare with NV reference output.

Usage:
    # Extract dump files first:
    #   tar -xzf dump.tar.gz -C ./dump_ascend
    #   tar -xzf dump_nv.tar.gz -C ./dump_nv
    #
    # Then run:
    #   python test_from_dump.py --dump_dir ./dump_ascend/dump --ref_dir ./dump_nv/dump

    # Or if both have same structure:
    #   python test_from_dump.py --dump_dir ./dump/

Parameters (expected from your case):
    qseqlen=1, kvseqlen=133, topk=16, qhead=64, kvhead=4, head_dim=128, block_size=128
"""

import argparse
import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
import torch_npu

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
sys.path.insert(0, str(_TORCH_EXT))


def _register_sparse_attention_score_op():
    spec = importlib.util.spec_from_file_location(
        "sparse_attention_score",
        _TORCH_EXT / "npu_ops_transformer" / "ops" / "sparse_attention_score.py",
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
        print("[INFO] Pre-compiled npu_sparse_attention_score .so not found; "
              "OpBuilder will compile it on first NPU call.")
        return

    ext_spec = importlib.util.spec_from_file_location("npu_sparse_attention_score", so_file)
    ext_mod = importlib.util.module_from_spec(ext_spec)
    ext_spec.loader.exec_module(ext_mod)

    from npu_ops_transformer.op_builder.builder import OpBuilder
    OpBuilder._loaded_ops["npu_sparse_attention_score"] = ext_mod
    print(f"[INFO] Loaded pre-compiled extension: {so_file}")


_register_sparse_attention_score_op()


def load_tensor(fpath):
    """Try loading a single .pt file with fallback strategies."""
    fsize = os.path.getsize(fpath)
    try:
        return torch.load(fpath, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(fpath, map_location='cpu')
    except (RuntimeError, Exception) as e:
        err_msg = str(e).lower()
        if 'central directory' in err_msg or 'not a zip file' in err_msg:
            # Large zip64 file that PyTorchStreamReader cannot handle
            # Try loading via zipfile + io.BytesIO per entry
            import zipfile, io, pickle
            try:
                with zipfile.ZipFile(fpath, 'r', allowZip64=True) as zf:
                    names = zf.namelist()
                    # Find the data.pkl and reconstruct
                    pkl_name = [n for n in names if n.endswith('data.pkl')]
                    if pkl_name:
                        buf = io.BytesIO()
                        # Rebuild a valid small zip in memory
                        with zipfile.ZipFile(buf, 'w') as wz:
                            for name in names:
                                wz.writestr(name, zf.read(name))
                        buf.seek(0)
                        return torch.load(buf, map_location='cpu', weights_only=False)
            except Exception as e2:
                print(f"    zipfile fallback also failed: {e2}")
            # Last resort: try pickle directly
            with open(fpath, 'rb') as f:
                magic = f.read(2)
                f.seek(0)
                if magic == b'\x80\x02' or magic == b'\x80\x05':
                    return pickle.load(f)
            raise
        raise


def load_dump(dump_dir):
    """Load all tensors from a dump directory."""
    tensors = {}
    for fname in sorted(os.listdir(dump_dir)):
        fpath = os.path.join(dump_dir, fname)
        if fname.endswith('.pt'):
            key = fname.replace('_input_tensor.pt', '').replace('_tensor.pt', '').replace('.pt', '')
            try:
                tensors[key] = load_tensor(fpath)
                print(f"  Loaded: {fname} -> key='{key}'")
            except Exception as e:
                fsize = os.path.getsize(fpath)
                print(f"  FAILED: {fname} (size={fsize} bytes): {e}")
        elif fname.endswith('.npy'):
            import numpy as np
            key = fname.replace('_input_tensor.npy', '').replace('_tensor.npy', '')
            tensors[key] = torch.from_numpy(np.load(fpath))
            print(f"  Loaded: {fname} -> key='{key}'")
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_dir', type=str, required=True,
                        help='Path to extracted dump/ directory (Ascend inputs)')
    parser.add_argument('--ref_dir', type=str, default=None,
                        help='Path to extracted dump_nv/ directory (NV reference output)')
    parser.add_argument('--kv_heads', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=16)
    args = parser.parse_args()

    print(f"Loading tensors from: {args.dump_dir}")
    data = load_dump(args.dump_dir)

    # Normalize key names (support both old and new dump naming)
    key_aliases = {
        'key': ['key', 'k_cache'],
        'value': ['value', 'v_cache'],
        'selected_idx': ['selected_idx', 'topk_idx'],
        'block_tables': ['block_tables', 'block_table'],
        'gpu_output': ['gpu_output'],
    }
    for canonical, aliases in key_aliases.items():
        if canonical not in data:
            for alias in aliases:
                if alias in data:
                    data[canonical] = data[alias]
                    break

    required_keys = ['query', 'key', 'value', 'selected_idx', 'block_tables']
    missing = [k for k in required_keys if k not in data]
    if missing:
        print(f"\n[ERROR] Missing required tensors: {missing}")
        print(f"  Available keys: {sorted(data.keys())}")
        print("[ERROR] Cannot proceed. Exiting.")
        return

    for k, v in sorted(data.items()):
        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        if any(x in k for x in ['selected_idx', 'block_table', 'topk_idx']):
            if v.numel() <= 200:
                print(f"    value={v}")
            else:
                print(f"    first 32: {v.flatten()[:32].tolist()}")

    query = data['query']
    key = data['key']
    value = data['value']
    select_idx = data['selected_idx']
    block_table = data['block_tables']

    # KV cache: block_table values will be adjusted by -1 later,
    # so physical IDs directly map to k_cache indices without padding.
    num_physical_blocks = key.shape[0]
    max_physical_id = block_table.max().item()
    print(f"  k_cache physical blocks: {num_physical_blocks}, block_table max_id: {max_physical_id}")
    print(f"  (block_table will be adjusted by -1 before NPU call)")
    print(f"  select_idx unique: {select_idx.unique().tolist()}")
    print(f"  block_table[0, :10]: {block_table[0, :10].tolist()}")

    # =========== Dump Validity Check ===========
    print(f"\n--- Dump Tensor Validity Check ---")
    print(f"  query shape: {list(query.shape)}, dtype={query.dtype}")
    print(f"  key (k_cache) shape: {list(key.shape)}, dtype={key.dtype}")
    print(f"  value (v_cache) shape: {list(value.shape)}, dtype={value.dtype}")
    print(f"  select_idx shape: {list(select_idx.shape)}, dtype={select_idx.dtype}")
    print(f"  block_table shape: {list(block_table.shape)}, dtype={block_table.dtype}")

    num_blocks_in_cache = key.shape[0]
    block_size_in_cache = key.shape[1]
    print(f"\n  KV cache: {num_blocks_in_cache} physical blocks x {block_size_in_cache} tokens/block"
          f" = {num_blocks_in_cache * block_size_in_cache} total KV slots")

    # block_table validity: all referenced physical IDs should be < num_blocks_in_cache
    bt_max = block_table.max().item()
    bt_min = block_table.min().item()
    bt_unique = block_table.unique()
    print(f"  block_table: min_id={bt_min}, max_id={bt_max}, unique_count={bt_unique.numel()}")
    if bt_max >= num_blocks_in_cache:
        print(f"  ERROR: block_table max_id ({bt_max}) >= num physical blocks ({num_blocks_in_cache})!")
        print(f"         Physical block IDs in block_table are out of bounds!")
    else:
        print(f"  OK: all block_table IDs within [0, {num_blocks_in_cache - 1}]")

    # Per-batch block_table info
    batch_count = block_table.shape[0]
    print(f"\n  batch_count (from block_table dim0): {batch_count}")
    for b in range(min(batch_count, 4)):
        row = block_table[b]
        nonzero_count = (row != 0).sum().item()
        row_unique = row.unique()
        print(f"    batch {b}: nonzero={nonzero_count}/{row.shape[0]}, "
              f"ids=[{row[:16].tolist()}{'...' if row.shape[0] > 16 else ''}]")

    # selectIdx validity: logical IDs should be valid block indices
    sel_max = select_idx.max().item()
    sel_min = select_idx.min().item()
    sel_valid = select_idx[select_idx >= 0]
    print(f"\n  select_idx: min={sel_min}, max={sel_max}")
    if sel_valid.numel() > 0:
        print(f"    valid (>=0) entries: {sel_valid.numel()} / {select_idx.numel()}")
        print(f"    valid range: [{sel_valid.min().item()}, {sel_valid.max().item()}]")
        max_logical_block = sel_valid.max().item()
        needed_kv_slots = (max_logical_block + 1) * block_size_in_cache
        print(f"    max logical block = {max_logical_block} -> needs kv_seqlen >= {needed_kv_slots}")
    else:
        print(f"    WARNING: no valid (>=0) entries in select_idx!")

    # selectIdx for first/last tokens
    print(f"\n  selectIdx samples:")
    print(f"    token 0:    {select_idx[:, 0, :].tolist()}")
    if select_idx.shape[1] > 1:
        mid = select_idx.shape[1] // 2
        print(f"    token {mid}: {select_idx[:, mid, :].tolist()}")
    print(f"    token {select_idx.shape[1]-1}: {select_idx[:, -1, :].tolist()}")
    # =========== End Validity Check ===========

    # Adjust block_table: subtract 1 from all values
    block_table = block_table - 1
    print(f"\n  [Adjusted] block_table -= 1, new range: [{block_table.min().item()}, {block_table.max().item()}]")

    q_seqlen = query.shape[0]
    q_heads = query.shape[1]
    head_dim = query.shape[2]
    kv_heads = args.kv_heads
    block_size = args.block_size
    top_k = args.top_k

    # Infer kv_seqlen from block_table
    max_blocks = block_table.shape[1]
    # Assume kvseqlen is passed or inferred
    # For this case: kvseqlen = 133
    kv_seqlen = 2048  # adjust if needed

    scale_value = 1.0 / math.sqrt(head_dim)
    actual_seq_lengths = [q_seqlen]
    actual_seq_lengths_kv = [kv_seqlen]

    # =========== KV Data Distribution Analysis ===========
    print(f"\n--- KV Data Distribution & BF16 Overflow Risk Analysis ---")
    BF16_MAX = 3.3895e+38  # torch.finfo(torch.bfloat16).max
    BF16_TINY = 1.1755e-38  # torch.finfo(torch.bfloat16).tiny (smallest normal)

    for name, tensor in [('query', query), ('key', key), ('value', value)]:
        t_f32 = tensor.float()
        abs_t = t_f32.abs()
        print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
        print(f"    range: [{t_f32.min().item():.6f}, {t_f32.max().item():.6f}]")
        print(f"    abs_max={abs_t.max().item():.6f}, abs_mean={abs_t.mean().item():.6f}, std={t_f32.std().item():.6f}")
        print(f"    nan_count={torch.isnan(t_f32).sum().item()}, inf_count={torch.isinf(t_f32).sum().item()}")

        # BF16 overflow check: abs value > BF16_MAX
        overflow_count = (abs_t > BF16_MAX).sum().item()
        # BF16 underflow check: 0 < abs value < BF16_TINY
        underflow_mask = (abs_t > 0) & (abs_t < BF16_TINY)
        underflow_count = underflow_mask.sum().item()
        if overflow_count > 0:
            print(f"    WARNING: {overflow_count} elements exceed BF16_MAX ({BF16_MAX:.4e})")
        if underflow_count > 0:
            print(f"    INFO: {underflow_count} elements below BF16 smallest normal ({BF16_TINY:.4e})")

    # QK score magnitude estimation
    q_f32 = query.float()
    # Estimate QK dot product magnitude: ||q|| * ||k|| per token
    q_norms = torch.norm(q_f32, dim=-1)  # [q_seqlen, q_heads]
    k_f32 = key.float()  # [num_blocks, block_size, kv_heads, head_dim]
    k_flat = k_f32.reshape(-1, head_dim)
    k_norms = torch.norm(k_flat, dim=-1)  # [num_blocks * block_size]

    max_qk_estimate = (q_norms.max() * k_norms.max() * scale_value).item()
    mean_qk_estimate = (q_norms.mean() * k_norms.mean() * scale_value).item()
    print(f"\n  QK score estimates (after scale={scale_value:.6f}):")
    print(f"    max |QK|_est = {max_qk_estimate:.4f} (||q||_max * ||k||_max * scale)")
    print(f"    mean |QK|_est = {mean_qk_estimate:.4f}")

    # exp(QK) overflow risk: exp(x) overflows bf16 when x > ln(BF16_MAX) ≈ 88.7
    # In practice, after subtracting max, exp(s - max) is in [0, 1], so no overflow in softmax
    # But if scale is wrong or data is extreme, pre-max QK could be large
    EXP_OVERFLOW_THRESHOLD = 88.7
    if max_qk_estimate > EXP_OVERFLOW_THRESHOLD:
        print(f"    WARNING: max QK estimate ({max_qk_estimate:.2f}) > exp overflow threshold ({EXP_OVERFLOW_THRESHOLD})")
        print(f"    This can cause exp() overflow BEFORE online-softmax max subtraction!")
    else:
        print(f"    OK: max QK estimate within safe exp() range")

    # Check if BF16 truncation of QK scores loses significant precision
    # BF16 has 7-bit mantissa, so relative precision is ~2^-7 ≈ 0.0078
    # If QK scores span a large dynamic range within one softmax group,
    # small scores may be completely zeroed after exp(s - max)
    print(f"\n  BF16 precision impact on softmax:")
    print(f"    BF16 relative precision: ~0.0078 (7-bit mantissa)")
    print(f"    If QK score range within one block > ~16, small scores contribute ~0 after softmax")
    print(f"    (exp(-16) ≈ 1.1e-7, negligible in bf16)")

    # =========== CPU Reference (FP32 and BF16) ===========
    # Skipped: CPU golden is too slow for large seqlen. Only compare NPU vs GPU.
    cpu_fp32_out = None
    cpu_bf16_out = None

    if False:  # Set to True to enable CPU golden (slow)
     group_size = q_heads // kv_heads
     def cpu_sparse_attention(q, k_cache, v_cache, sel_idx, blk_tbl, dtype_mode='fp32'):
        """CPU reference: compute sparse attention score.
        Uses per-block online softmax to match hardware behavior.
        """
        # q: [q_seqlen, q_heads, head_dim]
        # k_cache/v_cache: [num_blocks, block_size, kv_heads, head_dim]
        # sel_idx: [kv_heads, q_seqlen, top_k]
        # blk_tbl: [batch, max_blocks]
        if dtype_mode == 'fp32':
            q_w = q.float()
        else:
            q_w = q.bfloat16().float()

        scale_bf16 = torch.tensor(scale_value, dtype=torch.bfloat16).item()
        out = torch.zeros(q_seqlen, q_heads, head_dim, dtype=torch.float32)

        for qh in range(q_heads):
            kvh = qh // group_size
            for qt in range(q_seqlen):
                q_vec = q_w[qt, qh, :]  # [head_dim]

                last_max_fp32 = torch.tensor(-float('inf'), dtype=torch.float32)
                last_sum_fp32 = torch.tensor(0.0, dtype=torch.float32)
                o_acc_fp32 = torch.zeros(head_dim, dtype=torch.float32)
                is_first = True

                for ti in range(top_k):
                    logical_id = sel_idx[kvh, qt, ti].item()
                    if logical_id < 0:
                        continue
                    physical_id = int(blk_tbl[0, logical_id].item())
                    history_len = kv_seqlen - q_seqlen
                    causal_bound = history_len + qt
                    block_begin = logical_id * block_size
                    block_end = min(block_begin + block_size, kv_seqlen)
                    effective_end = min(block_end, causal_bound + 1)
                    if effective_end <= block_begin:
                        continue
                    valid_len = effective_end - block_begin

                    k_blk = k_cache[physical_id, :valid_len, kvh, :]
                    v_blk = v_cache[physical_id, :valid_len, kvh, :]

                    if dtype_mode == 'bf16':
                        k_blk = k_blk.bfloat16().float()
                        v_blk = v_blk.bfloat16().float()
                    else:
                        k_blk = k_blk.float()
                        v_blk = v_blk.float()

                    if dtype_mode == 'fp32':
                        s = torch.matmul(q_vec.unsqueeze(0), k_blk.T).squeeze(0)
                        s_scaled = s * scale_value
                        now_max = s_scaled.max()
                        if not is_first:
                            now_max = torch.max(now_max, last_max_fp32)
                        p = torch.exp(s_scaled - now_max)
                        now_sum = p.sum()
                        if is_first:
                            last_sum_fp32 = now_sum
                            last_max_fp32 = now_max
                        else:
                            correction = torch.exp(last_max_fp32 - now_max)
                            last_sum_fp32 = correction * last_sum_fp32 + now_sum
                            last_max_fp32 = now_max
                        pv = torch.matmul(p.unsqueeze(0), v_blk).squeeze(0)
                        if is_first:
                            o_acc_fp32 = pv
                        else:
                            o_acc_fp32 = o_acc_fp32 * correction + pv
                    else:
                        # BF16 mode: match hardware truncation pattern
                        s = torch.matmul(q_vec.float().unsqueeze(0), k_blk.float().T).squeeze(0)
                        s_bf16 = s.to(torch.bfloat16)
                        s_scaled_bf16 = (s_bf16.float() * scale_bf16).to(torch.bfloat16)
                        now_max_bf16 = s_scaled_bf16.max().to(torch.bfloat16)
                        if not is_first:
                            last_max_bf16 = last_max_fp32.to(torch.bfloat16)
                            now_max_bf16 = torch.max(now_max_bf16, last_max_bf16)
                        diff_bf16 = (s_scaled_bf16.float() - now_max_bf16.float()).to(torch.bfloat16)
                        p_bf16 = torch.exp(diff_bf16.float()).to(torch.bfloat16)
                        now_sum_bf16 = p_bf16.float().sum().to(torch.bfloat16)
                        now_max_fp32 = now_max_bf16.float()
                        if is_first:
                            last_sum_fp32 = now_sum_bf16.float()
                            last_max_fp32 = now_max_fp32
                        else:
                            correction_fp32 = torch.exp(last_max_fp32 - now_max_fp32)
                            last_sum_fp32 = correction_fp32 * last_sum_fp32 + now_sum_bf16.float()
                            last_max_fp32 = now_max_fp32
                        pv = torch.matmul(p_bf16.float().unsqueeze(0), v_blk.float()).squeeze(0)
                        if is_first:
                            o_acc_fp32 = pv
                        else:
                            o_acc_fp32 = o_acc_fp32 * correction_fp32 + pv

                    is_first = False

                if last_sum_fp32 > 0:
                    result = o_acc_fp32 / last_sum_fp32
                    if dtype_mode == 'bf16':
                        out[qt, qh, :] = result.to(torch.bfloat16).float()
                    else:
                        out[qt, qh, :] = result
     return out

     cpu_fp32_out = cpu_sparse_attention(query, key, value, select_idx, block_table, 'fp32')
     cpu_bf16_out = cpu_sparse_attention(query, key, value, select_idx, block_table, 'bf16')

     print(f"  CPU FP32 output range: [{cpu_fp32_out.min().item():.6f}, {cpu_fp32_out.max().item():.6f}]")
     print(f"  CPU BF16 output range: [{cpu_bf16_out.min().item():.6f}, {cpu_bf16_out.max().item():.6f}]")

     if 'gpu_output' in data:
        golden = data['gpu_output'].float()
        diff_fp32 = (cpu_fp32_out - golden).abs()
        diff_bf16 = (cpu_bf16_out - golden).abs()
        print(f"  CPU FP32 vs GPU golden: max={diff_fp32.max().item():.6f}, mean={diff_fp32.mean().item():.6f}")
        print(f"  CPU BF16 vs GPU golden: max={diff_bf16.max().item():.6f}, mean={diff_bf16.mean().item():.6f}")
        diff_fp32_bf16 = (cpu_fp32_out - cpu_bf16_out).abs()
        print(f"  CPU FP32 vs CPU BF16:   max={diff_fp32_bf16.max().item():.6f}, mean={diff_fp32_bf16.mean().item():.6f}")
    # =========== End CPU Reference ===========

    print(f"\nRunning SparseAttentionScore on NPU:")
    print(f"  q_seqlen={q_seqlen}, kv_seqlen={kv_seqlen}")
    print(f"  q_heads={q_heads}, kv_heads={kv_heads}, head_dim={head_dim}")
    print(f"  block_size={block_size}, top_k={top_k}")
    print(f"  scale_value={scale_value}")
    print(f"  select_idx shape={list(select_idx.shape)}")
    print(f"  block_table shape={list(block_table.shape)}")
    print(f"  L1norm(query)={query.float().abs().sum().item():.6f}")
    print(f"  L1norm(key)={key.float().abs().sum().item():.6f}")
    print(f"  L1norm(value)={value.float().abs().sum().item():.6f}")

    # Run on NPU
    npu_out = torch_npu.npu_sparse_attention_score(
        query.npu(), key.npu(), value.npu(),
        select_idx.npu(), block_table.npu(),
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=kv_heads,
        scale_value=scale_value,
        block_size=block_size,
        top_k=top_k,
        inner_precise=4,
    )
    npu_out_cpu = npu_out.cpu()
    print(f"\nNPU output shape={list(npu_out_cpu.shape)}, dtype={npu_out_cpu.dtype}")
    npu_nan = torch.isnan(npu_out_cpu.float()).sum().item()
    npu_inf = torch.isinf(npu_out_cpu.float()).sum().item()
    print(f"  NPU output: nan_count={npu_nan}, inf_count={npu_inf}")
    if npu_nan == 0:
        print(f"  NPU output range: [{npu_out_cpu.float().min().item():.6f}, {npu_out_cpu.float().max().item():.6f}]")

    # =========== Token 0 Detailed Check ===========
    print(f"\n--- Token 0 Detailed Check ---")
    q_token0 = query[0].float()  # [q_heads, head_dim]
    npu_token0 = npu_out_cpu[0].float()  # [q_heads, head_dim]
    print(f"  query[0]:      range=[{q_token0.min().item():.6f}, {q_token0.max().item():.6f}], "
          f"L1={q_token0.abs().sum().item():.6f}")
    print(f"  NPU out[0]:    range=[{npu_token0.min().item():.6f}, {npu_token0.max().item():.6f}], "
          f"L1={npu_token0.abs().sum().item():.6f}")
    if 'gpu_output' in data:
        gpu_token0 = data['gpu_output'][0].float()
        print(f"  GPU out[0]:    range=[{gpu_token0.min().item():.6f}, {gpu_token0.max().item():.6f}], "
              f"L1={gpu_token0.abs().sum().item():.6f}")
        diff_npu_gpu_t0 = (npu_token0 - gpu_token0).abs()
        cos_t0 = torch.nn.functional.cosine_similarity(
            npu_token0.flatten().unsqueeze(0), gpu_token0.flatten().unsqueeze(0)).item()
        print(f"  NPU[0] vs GPU[0]: max_diff={diff_npu_gpu_t0.max().item():.6f}, "
              f"mean_diff={diff_npu_gpu_t0.mean().item():.6f}, cos_sim={cos_t0:.6f}")
        # Check if NPU output[0] looks like all zeros or constant
        if npu_token0.abs().max().item() < 1e-6:
            print(f"  WARNING: NPU token 0 output is nearly all zeros!")
        # Check first few values
        print(f"  NPU out[0][0][:8]:  {npu_token0[0, :8].tolist()}")
        print(f"  GPU out[0][0][:8]:  {gpu_token0[0, :8].tolist()}")
        print(f"  query[0][0][:8]:    {q_token0[0, :8].tolist()}")

    # Also check last token (should have full causal access)
    last_t = npu_out_cpu.shape[0] - 1
    npu_last = npu_out_cpu[last_t].float()
    print(f"\n  NPU out[{last_t}]:  range=[{npu_last.min().item():.6f}, {npu_last.max().item():.6f}], "
          f"L1={npu_last.abs().sum().item():.6f}")
    if 'gpu_output' in data:
        gpu_last = data['gpu_output'][last_t].float()
        print(f"  GPU out[{last_t}]:  range=[{gpu_last.min().item():.6f}, {gpu_last.max().item():.6f}], "
              f"L1={gpu_last.abs().sum().item():.6f}")
        diff_last = (npu_last - gpu_last).abs()
        cos_last = torch.nn.functional.cosine_similarity(
            npu_last.flatten().unsqueeze(0), gpu_last.flatten().unsqueeze(0)).item()
        print(f"  NPU[{last_t}] vs GPU[{last_t}]: max_diff={diff_last.max().item():.6f}, "
              f"mean_diff={diff_last.mean().item():.6f}, cos_sim={cos_last:.6f}")
    print(f"--- End Token 0 Check ---\n")

    # =========== Direct NPU vs GPU Comparison ===========
    if 'gpu_output' in data:
        golden = data['gpu_output']
        golden_nan = torch.isnan(golden.float()).sum().item()
        golden_inf = torch.isinf(golden.float()).sum().item()
        print(f"\nGPU golden shape={list(golden.shape)}, dtype={golden.dtype}")
        print(f"  GPU golden: nan_count={golden_nan}, inf_count={golden_inf}")
        if golden_nan == 0:
            print(f"  GPU golden range: [{golden.float().min().item():.6f}, {golden.float().max().item():.6f}]")

        valid_mask = ~(torch.isnan(npu_out_cpu.float()) | torch.isnan(golden.float()))
        if valid_mask.sum() > 0:
            npu_f = npu_out_cpu.float()
            gld_f = golden.float()
            diff = (npu_f - gld_f).abs()
            diff_valid = diff[valid_mask]
            cos_sim = torch.nn.functional.cosine_similarity(
                npu_f[valid_mask].flatten().unsqueeze(0),
                gld_f[valid_mask].flatten().unsqueeze(0)
            ).item()
            rel_diff = diff_valid / (gld_f.abs()[valid_mask] + 1e-8)

            print(f"\n  NPU vs GPU golden:")
            print(f"    max_abs_diff={diff_valid.max().item():.6f}")
            print(f"    mean_abs_diff={diff_valid.mean().item():.6f}")
            print(f"    max_rel_diff={rel_diff.max().item():.6f}")
            print(f"    mean_rel_diff={rel_diff.mean().item():.6f}")
            print(f"    cosine_similarity={cos_sim:.8f}")
            print(f"    valid elements: {valid_mask.sum().item()} / {valid_mask.numel()}")

            # Per-token breakdown (first/last few tokens)
            if npu_f.dim() == 3:
                token_diffs = diff.amax(dim=(1, 2))
                print(f"\n  Per-token max_abs_diff (first 10):")
                for t in range(min(10, token_diffs.shape[0])):
                    print(f"    token {t}: {token_diffs[t].item():.6f}")
                if token_diffs.shape[0] > 10:
                    print(f"  Per-token max_abs_diff (last 5):")
                    for t in range(max(10, token_diffs.shape[0] - 5), token_diffs.shape[0]):
                        print(f"    token {t}: {token_diffs[t].item():.6f}")

                # Find worst token
                worst_token = token_diffs.argmax().item()
                print(f"\n  Worst token: {worst_token}, max_diff={token_diffs[worst_token].item():.6f}")
        else:
            print("  All elements are NaN, cannot compare with GPU golden.")
    else:
        print("\n  [WARNING] No gpu_output found in dump dir. Cannot compare.")

    # Compare with NV reference if provided
    if args.ref_dir:
        print(f"\nLoading NV reference from: {args.ref_dir}")
        nv_data = load_dump(args.ref_dir)
        for k, v in sorted(nv_data.items()):
            print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")

        if 'attn_output' in nv_data:
            nv_ref = nv_data['attn_output']
            diff_nv = (npu_out_cpu.float() - nv_ref.float()).abs()
            print(f"\n  NPU vs NV: max_diff={diff_nv.max().item():.6f}, mean_diff={diff_nv.mean().item():.6f}")

            # Per-token stats
            if nv_ref.dim() == 3:
                for t in range(min(nv_ref.shape[0], 5)):
                    token_diff = diff_nv[t].max().item()
                    print(f"    token {t}: max_diff={token_diff:.6f}")
        else:
            print("  No attn_output found in NV dump.")

    print("\nDone.")


if __name__ == '__main__':
    main()
