import torch
import torch_npu
import numpy as np
import torch.nn as nn
import math


def float_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """
    手动将 float tensor 转换为 E8M0 (float8_e8m0fnu) 格式。

    E8M0 编码规则 (OCP MX):
      - 8 bit 全为指数位，无符号，无尾数
      - bias = 127
      - E=0        -> 0.0
      - E=1~254    -> 2^(E - 127)
      - E=255      -> 保留 (Inf/NaN)
    转换: E = clamp(round(log2(|x|)) + 127, 0, 254)
    """
    x_f32 = x.detach().float()
    x_abs = torch.abs(x_f32)

    # 零值特殊处理
    zero_mask = (x_abs == 0)

    # 避免 log2(0)，用 1.0 填充零位（后续会被覆盖）
    x_safe = torch.where(zero_mask, torch.ones_like(x_abs), x_abs)

    # 计算 biased exponent (使用 floor 向下取整，与原生行为一致)
    log2_val = torch.log2(x_safe)
    biased_exp = torch.floor(log2_val) + 127.0
    biased_exp = torch.clamp(biased_exp, 1.0, 254.0)

    # 零值 -> E=0
    biased_exp = torch.where(zero_mask, torch.zeros_like(biased_exp), biased_exp)

    # 转为 uint8 (E8M0 的原始字节表示)
    raw_bytes = biased_exp.to(torch.uint8)

    # 尝试 reinterpret 为 float8_e8m0fnu dtype
    try:
        return raw_bytes.view(torch.float8_e8m0fnu)
    except (RuntimeError, AttributeError):
        # 如果 dtype 不存在或 view 不支持，返回 uint8
        return raw_bytes


n1 = 64
n2 = 1
d = 128
block_size = 128
layout_key = "PA_BSND"
layout_query = "BSND"
query_quant_mode = 0
key_quant_mode = 0
np.random.seed(0)
# -------------
b = 1
t = None
s1 = 4
s2 = 512
act_seq_q = None
act_seq_k = None
sparse_mode = 3
sparse_count = 2048
max_block_table_num = (s2 + block_size - 1) // block_size
block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
key = torch.tensor(np.random.uniform(1, 1, (b * max_block_table_num, block_size, n2, d))).to(torch.float8_e4m3fn)
key_dequant_scale = torch.tensor(np.random.uniform(1, 11, (b * max_block_table_num, block_size, n2, d//32)))
key_dequant_scale = float_to_e8m0(key_dequant_scale)
query = torch.tensor(np.random.uniform(1, 1, (b, s1, n1, d))).to(torch.float8_e4m3fn)
query_dequant_scale = float_to_e8m0(torch.tensor(np.random.uniform(1, 1, (b, s1, n1, d//32))))
weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.bfloat16)
actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                            if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                            if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)

print(query.shape)
print(key.shape)
print(weights.shape)
print(query_dequant_scale.shape)
print(key_dequant_scale.shape)
print(actual_seq_lengths_query.shape)
print(actual_seq_lengths_key.shape)
print(block_table.shape)
print(query_quant_mode)
print(key_quant_mode)
print(layout_key)
print(layout_query)
print(sparse_count)
print(sparse_mode)

npu_out = torch_npu.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                key_dequant_scale.npu(),
                                                actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                block_table=block_table.npu(),
                                                query_quant_mode=query_quant_mode,
                                                key_quant_mode=key_quant_mode,
                                                layout_query=layout_query,
                                                layout_key=layout_key, sparse_count=sparse_count,
                                                sparse_mode=sparse_mode)

print(npu_out.shape)
print(npu_out)