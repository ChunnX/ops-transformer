import torch
import torch_npu
import math
import numpy as np
import npu_ops_transformer
from npu_ops_transformer.ops import npu_flash_attn
torch.manual_seed(42)

# B = 1
B_list = [1]
numHeads = 1
numKeyValueHeads = 1
# Sq = 1
Sq_list = [128]
Skv = 128
D = 64
# T = 4096
type=torch.float16

for B in B_list:
    for Sq in Sq_list:
        query = torch.randn(B, numHeads, Sq, D, dtype=type).npu()
        key = torch.randn(B, numKeyValueHeads, Skv, D, dtype=type).npu()
        value = torch.randn(B, numKeyValueHeads, Skv, D, dtype=type).npu()
        scale_value = 1/math.sqrt(float(D))

        actual_seq_lengths_kv = [Skv]*B
        attention_mask = torch.tril(torch.ones(2048,2048)).to(torch.bool).npu()

        for _ in range(1):
            out, _ = npu_flash_attn(
                    query, key, value,
                    block_table=None,
                    cu_seqlens_q=None,
                    cu_seqlens_kv=None,
                    seqused_q=None,
                    seqused_kv=None,
                    sinks=None,
                    metadata=None,
                    softmax_scale=scale_value,
                    mask_mode=0,
                    win_left=0,
                    win_right=0,
                    max_seqlen_q=0,
                    max_seqlen_kv=0,
                    layout_q = "BNSD",
                    layout_kv = "BNSD",
                    layout_out = "BNSD",
                    return_softmax_lse=0,
                    deterministic=0)
        out = out.cpu()

        print("************end*************", out, out.shape)