
import math
import numpy as np
import torch
import torch_npu
import shutil
import random
import os
import sys
from ml_dtypes import bfloat16
from dataclasses import dataclass

# test_bsa_mxfp4_golden 位于本脚本上级目录(mxfp4_pytest/), 需加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_bsa_mxfp4_golden import (
    mxfp4_quantize_pack_last,
    _e8m0_qk_to_v3,
    _e8m0_v_to_v3,
    QUANT_MODE_OCP,
)
from cann_ops_transformer.ops import block_sparse_attention_v3

def gen_block_sparse_pattern(q_seqlen_list: list,
        kv_seqlen_list: list,
        s_block_x: int,
        s_block_y: int,
        batch: int,
        num_heads: int,
        sparsity_ratio: float,
        is_varied_sparsity: int
):
    q_s_aligned = max(q_seqlen_list)
    kv_s_aligned = max(kv_seqlen_list)
    q_s_block_num_aligned = (q_s_aligned + s_block_x - 1) // s_block_x
    kv_s_block_num_aligned = (kv_s_aligned + s_block_y - 1) // s_block_y

    block_sparse_mask = np.zeros((batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int8)
    block_sparse_idx = np.ones((batch, num_heads, q_s_block_num_aligned, kv_s_block_num_aligned), dtype=np.int32)
    # sparse idx 无效位置为-1，便于和idx=0做区分
    block_sparse_idx = -block_sparse_idx
    block_sparse_count = np.zeros((batch, num_heads, q_s_block_num_aligned), dtype=np.int32)
    block_sparse_shape = [s_block_x, s_block_y]

    for b in range(batch):
        q_s_block_num_aval = (q_seqlen_list[b] + s_block_x - 1) // s_block_x
        kv_s_block_num_aval = (kv_seqlen_list[b] + s_block_y - 1) // s_block_y
        max_rsvd_kv_s_block_num = max(1, int(kv_s_block_num_aval * sparsity_ratio))
        for n1 in range(num_heads):
            for q_blk_id in range(q_s_block_num_aval):
                cur_row_sparse_mask = np.zeros(kv_s_block_num_aval, dtype=np.int8)
                # 当前rainfusion稀疏算法的原理：每个BN组合的所有X方向分块都采取同一个topK的方式选择Y方向的块
                # is_varied_sparsity是依据上述算法发散的进一步泛化场景：每个BN组合的每个X方向分块的topK的值也不相同
                rsvd_kv_s_block_num = max_rsvd_kv_s_block_num
                if is_varied_sparsity:
                    random.seed(b * num_heads * q_s_block_num_aval + n1 * q_s_block_num_aval + q_blk_id)
                    rsvd_kv_s_block_num = random.randint(1, max_rsvd_kv_s_block_num)
                # 依据随机出的保留块计数生成随机的稀疏pattern以及保留块下标
                cur_row_sparse_idx = sorted(np.random.choice(kv_s_block_num_aval, rsvd_kv_s_block_num, replace=False))
                for idx in cur_row_sparse_idx:
                    cur_row_sparse_mask[idx] = 1
                # 将生成的mask，idx，count填入对应位置
                block_sparse_mask[b, n1, q_blk_id, 0:kv_s_block_num_aval] = cur_row_sparse_mask
                block_sparse_idx[b, n1, q_blk_id, 0:rsvd_kv_s_block_num] = cur_row_sparse_idx
                block_sparse_count[b, n1, q_blk_id] = rsvd_kv_s_block_num
    return block_sparse_mask, block_sparse_idx, block_sparse_count, block_sparse_shape

if __name__ == "__main__":
    batch = int(sys.argv[1])
    q_seqlen = int(sys.argv[2])
    kv_seqlen = int(sys.argv[3])
    num_heads = int(sys.argv[4])
    kv_heads = int(sys.argv[5])
    embedding_size = int(sys.argv[6])
    s_block_x = int(sys.argv[7])  # S矩阵Q方向分块大小
    s_block_y = int(sys.argv[8])  # S矩阵KV方向分块大小
    sparsity_ratio = float(sys.argv[9])   # 舍弃的占比
    inner_prec = int(sys.argv[10])
    q_layout = str(sys.argv[11])
    kv_layout = str(sys.argv[12])
    is_varied_len = int(sys.argv[13])
    is_varied_sparsity = int(sys.argv[14]) # 1 means sparsity ratio varies across different query blocks of each batch and head
    str_dtype = str(sys.argv[15])
    is_act_seqlen_aval = int(sys.argv[16])
    device_id = int(sys.argv[17])

    is_mxfp4 = (str_dtype == "mxfp4")

    if str_dtype == "half":
        dtype = torch.float16
    elif str_dtype == "bf16":
        dtype = torch.bfloat16
    elif str_dtype == "mxfp4":
        dtype = torch.float32  # mxfp4 先用 fp32 造数据, 再量化打包
    else:
        raise ValueError(f"不支持的 dtype: {str_dtype}, 仅支持 half/bf16/mxfp4")

    scale_value = 1.0 / math.sqrt(embedding_size)
    torch_npu.npu.set_device(device_id)

    q_seqlen_list = [q_seqlen] * batch
    kv_seqlen_list = [kv_seqlen] * batch
    q_tokens = q_seqlen * batch
    kv_tokens = kv_seqlen * batch
    q_seqlen_aligned = max(q_seqlen_list) if is_act_seqlen_aval == 1 else q_seqlen
    kv_seqlen_aligned = max(kv_seqlen_list) if is_act_seqlen_aval == 1 else kv_seqlen
    block_shape = [s_block_x, s_block_y]

    block_sparse_mask, _, _, _ = gen_block_sparse_pattern(
        q_seqlen_list, kv_seqlen_list, s_block_x, s_block_y,
        batch, num_heads, 1 - sparsity_ratio, is_varied_sparsity)
    block_sparse_mask = torch.from_numpy(block_sparse_mask)

    if is_mxfp4:
        # ===== MXFP4 路径: fp32 造数据(BNSD) -> 量化打包 -> 按布局重排 -> NPU =====
        assert q_layout in ("BNSD", "BSND"), \
            f"mxfp4 当前仅支持 BNSD/BSND, 当前 q_layout={q_layout}"
        assert embedding_size in (64, 128), \
            f"mxfp4 head_dim 仅支持 64/128, 当前 {embedding_size}"

        # fp32 BNSD 造数据
        q_fp32 = torch.randn(batch, num_heads, q_seqlen_aligned, embedding_size, dtype=torch.float32)
        k_fp32 = torch.randn(batch, kv_heads, kv_seqlen_aligned, embedding_size, dtype=torch.float32)
        v_fp32 = torch.randn(batch, kv_heads, kv_seqlen_aligned, embedding_size, dtype=torch.float32)

        # MXFP4 量化打包 + e8m0 scale 重排 (在 CPU 上完成)
        q_packed, q_e8m0 = mxfp4_quantize_pack_last(q_fp32, quant_axis=-1,
                                                     quant_mode=QUANT_MODE_OCP, log2_cx=0.0)
        k_packed, k_e8m0 = mxfp4_quantize_pack_last(k_fp32, quant_axis=-1,
                                                     quant_mode=QUANT_MODE_OCP, log2_cx=0.0)
        v_packed, v_e8m0 = mxfp4_quantize_pack_last(v_fp32, quant_axis=-2,
                                                     quant_mode=QUANT_MODE_OCP, log2_cx=0.0)
        q_scale = _e8m0_qk_to_v3(q_e8m0)
        k_scale = _e8m0_qk_to_v3(k_e8m0)
        v_scale = _e8m0_v_to_v3(v_e8m0)

        # 按布局重排
        if q_layout == "BSND":
            q_packed = q_packed.permute(0, 2, 1, 3).contiguous()
            k_packed = k_packed.permute(0, 2, 1, 3).contiguous()
            v_packed = v_packed.permute(0, 2, 1, 3).contiguous()
            q_scale = q_scale.permute(0, 2, 1, 3, 4).contiguous()
            k_scale = k_scale.permute(0, 2, 1, 3, 4).contiguous()
            v_scale = v_scale.permute(0, 2, 1, 3, 4).contiguous()

        # 搬到 NPU
        q_packed = q_packed.contiguous().npu()
        k_packed = k_packed.contiguous().npu()
        v_packed = v_packed.contiguous().npu()
        q_scale = q_scale.npu()
        k_scale = k_scale.npu()
        v_scale = v_scale.npu()
        block_sparse_mask = block_sparse_mask.to("npu")

        print(f"block_sparse_attention_v3 start")
        attn, lse = block_sparse_attention_v3(
            q_packed, k_packed, v_packed,
            block_sparse_mask, block_shape,
            q_seqlen_list, kv_seqlen_list,
            num_heads=num_heads, num_key_value_heads=kv_heads,
            q_input_layout=q_layout, kv_input_layout=kv_layout,
            mask_type=0, scale_value=scale_value, inner_precise=4, block_size=0,
            pre_tokens=2147483647, next_tokens=2147483647,
            softmax_lse_flag=False, quant_mode=2, dst_type_max=0,
            q_dequant_scale=q_scale, k_dequant_scale=k_scale, v_dequant_scale=v_scale,
            attention_out_dtype=torch.bfloat16,
        )

    else:
        # ===== Half / BF16 路径 (原有逻辑) =====
        if q_layout == "TND":
            q = torch.randn(q_tokens, num_heads, embedding_size, dtype=dtype)
            k = torch.randn(kv_tokens, kv_heads, embedding_size, dtype=dtype)
            v = torch.randn(kv_tokens, kv_heads, embedding_size, dtype=dtype)
        elif q_layout == "BNSD":
            q = torch.randn(batch, num_heads, q_seqlen_aligned, embedding_size, dtype=dtype)
            k = torch.randn(batch, kv_heads, kv_seqlen_aligned, embedding_size, dtype=dtype)
            v = torch.randn(batch, kv_heads, kv_seqlen_aligned, embedding_size, dtype=dtype)
        elif q_layout == "BSND":
            q = torch.randn(batch, q_seqlen_aligned, num_heads, embedding_size, dtype=dtype)
            k = torch.randn(batch, kv_seqlen_aligned, kv_heads, embedding_size, dtype=dtype)
            v = torch.randn(batch, kv_seqlen_aligned, kv_heads, embedding_size, dtype=dtype)

        q = q.to("npu")
        k = k.to("npu")
        v = v.to("npu")
        block_sparse_mask = block_sparse_mask.to("npu")

        attn, lse = torch_npu.npu_block_sparse_attention(
            q, k, v, block_sparse_mask, block_shape,
            q_input_layout=q_layout, kv_input_layout=kv_layout,
            num_key_value_heads=kv_heads, scale_value=scale_value, inner_precise=inner_prec,
            actual_seq_lengths=q_seqlen_list, actual_seq_lengths_kv=kv_seqlen_list,
            softmax_lse_flag=0)
