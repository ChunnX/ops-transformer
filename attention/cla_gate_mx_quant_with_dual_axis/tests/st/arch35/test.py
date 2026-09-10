import torch
import torch_npu
import cann_ops_transformer

device = "npu:0"
dtype = torch.bfloat16

t, n, d = 64, 96, 256
global_out = torch.randn(t, n, d, dtype=dtype, device=device)
local_out = torch.randn(t, n, d, dtype=dtype, device=device)
global_gate_logits = torch.randn(t, n, dtype=dtype, device=device)
local_gate_logits = torch.randn(t, n, dtype=dtype, device=device)

row_data, row_scale, col_data, col_scale = (
    cann_ops_transformer.cla_gate_mx_quant_with_dual_axis(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        round_mode="rint",
        scale_alg=1,
        dst_type=36,
        layout=0,
    )
)

print(row_data.shape)
print(row_scale.shape)
print(col_data.shape)
print(col_scale.shape)
