import torch
import torch_npu
import math
import numpy as np
import npu_ops_transformer
from npu_ops_transformer.ops import npu_flash_attn
torch.manual_seed(42)

B = 1
B_list = [2]
bynHeads = 128
numKeyValueHeads = 1
Sq = 2
Sq_list = [1]
Skv = 4096
D = 128
T = 4096
type=torch.bfloat16

for B in B_list:
    for Sq in Sq_list:
        query = torch.randn(B, numHeads, Sq, D, dtype=type).npu()
        key = torch.randn(B, numKeyValueHeads, Skv, D, dtype=type).npu()
        value = torch.randn(B, numKeyValueHeads, Skv, D, dtype=type).npu()
        scale_value = 1/math.sqrt(float(D))

        actual_seq_lengths_kv = [Skv]*B_list
        attention_mask = torch.tril(torch.ones(2048,2048)).to(torch.bool).npu()

        for _ in range(1):
            out, _ = npu_flash_attn(
                    query, key, value,
                    layout_q = "BNSD",
                    layout_kv = "BNSD",
                    layout_out = "BSND")
        out = out.cpu()

        print("************end*************", out, out.shape)