"""
Step1: 生成 CPU golden（combine 级联）
dispatch golden 由 test_dispatch_compare.py 通过 aclnn 对比验证

用法: python tests/gen_cpu_golden.py
输出: tests/golden.pt
"""
import torch, os

BS, H, K, EP, MOE = 55, 1038, 1, 2, 2

inputs = {}
for rank in range(EP):
    torch.manual_seed(42 + rank)
    inputs[rank] = {
        'x': torch.randn(BS, H, dtype=torch.bfloat16),
        'expert_ids': torch.randint(0, MOE, (BS, K), dtype=torch.int32),
        'expert_scales': torch.rand(BS, K, dtype=torch.float32) + 0.5,
    }

# Combine golden: dispatch→combine round-trip, golden[i] = x[i] * scale[i]
golden_combine = {}
for rank in range(EP):
    golden_combine[rank] = (inputs[rank]['x'].float() * inputs[rank]['expert_scales']).to(torch.bfloat16)

out_path = os.path.join(os.path.dirname(__file__), 'golden.pt')
torch.save({'inputs': inputs, 'golden_combine': golden_combine}, out_path)
print(f"Saved: {out_path}")
for r in range(EP):
    print(f"  R{r}: abs={golden_combine[r].float().abs().sum():.2f}")
