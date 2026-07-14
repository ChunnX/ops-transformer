import torch
import torch_npu
from cann_ops_transformer.ops import chunk_gated_delta_rule
# 构造输入
B, seqlen, nk, nv, dk, dv = (2, 100, 4, 8, 128, 128)
actual_seq_lengths = (torch.ones(B) * seqlen).to("npu").to(torch.int32)
T = int(torch.sum(actual_seq_lengths))
state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16).npu()
query = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
key = torch.rand((T, nk, dk), dtype=torch.bfloat16).npu()
value = torch.rand((T, nv, dv), dtype=torch.bfloat16).npu()
g = torch.rand((T, nv), dtype=torch.float32).npu() * (-1.0)
beta = torch.rand((T, nv), dtype=torch.bfloat16).npu()
 	 
query = torch.nn.functional.normalize(query, p=2, dim=-1)
key = torch.nn.functional.normalize(key, p=2, dim=-1)
scale = dk ** -0.5
# 调用算子
o, final_state = torch.ops.cann_ops_transformer.chunk_gated_delta_rule(
    query, key, value,
    beta=beta,
    initial_state=state,
    actual_seq_lengths=actual_seq_lengths,
    scale=scale,
    g=g)
print(o)
print(o.shape, final_state.shape)
