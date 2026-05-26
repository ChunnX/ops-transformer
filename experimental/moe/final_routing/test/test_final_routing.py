import torch
import torch_npu
import npu_ops_transformer_ext
import logging

logging.basicConfig(level=logging.INFO)

token_num = 2
expert_num = 4
hidden_size = 8
block_dim = 40

input = torch.randint(1, 10, (token_num * expert_num, hidden_size)).to(torch.bfloat16)
token_table = torch.arange(token_num * expert_num, dtype=torch.int32).reshape(token_num, expert_num)
score_table = torch.randint(0, 10, (token_num, expert_num)).to(torch.bfloat16)
output = torch.empty(token_num, hidden_size, dtype=torch.bfloat16)
output_cpu = torch.empty_like(output).float()

input_32 = input.float()
for t in range(token_num):
    for e in range(expert_num):
        input_idx = token_table[t, e].item()
        if input_idx < 0:
            continue
        score = score_table[t, e].item()
        feat = input_32[input_idx]
        output_cpu[t] += feat * score

output_cpu = output_cpu.to(torch.bfloat16)
output_npu = output.npu()
torch.ops.npu_ops_transformer_ext.final_routing(block_dim, input.npu(), token_table.npu(), score_table.npu(), output_npu)

logging.info(f"cpu result vs npu result: {torch.equal(output_cpu, output_npu.cpu())}")