"""
脚本3: 加载 golden.pt 和 npu_output.pt，用 v3v4 的 compare_output 对比
用法: python tests/compare_results.py
输入: tests/golden.pt, tests/npu_output.pt
"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pta-moe-test'))
from moe_operator_test_v3v4 import compare_output, ComparedOutput

EP = 2

test_dir = os.path.dirname(__file__)
golden_path = os.path.join(test_dir, 'golden.pt')
npu_path = os.path.join(test_dir, 'npu_output.pt')

assert os.path.exists(golden_path), f"Run gen_cpu_golden.py first! {golden_path}"
assert os.path.exists(npu_path), f"Run run_kernel_launch.py first! {npu_path}"

data = torch.load(golden_path)
golden = data['golden_combine']
npu_output = torch.load(npu_path)

print("=" * 60)
print("Compare: <<<>>> output vs CPU golden")
print("  Using v3v4 compare_output function")
print("=" * 60)

golden_per_card = [golden[r] for r in range(EP)]
npu_per_card = [npu_output[r] for r in range(EP)]

print(f"\nGolden R0 abs: {golden_per_card[0].float().abs().sum():.2f}")
print(f"NPU    R0 abs: {npu_per_card[0].float().abs().sum():.2f}")
print(f"Golden R1 abs: {golden_per_card[1].float().abs().sum():.2f}")
print(f"NPU    R1 abs: {npu_per_card[1].float().abs().sum():.2f}")

print("\n>>> v3v4 compare_output (combine x_out):")
res = compare_output(
    ComparedOutput.X,
    golden_per_card,
    npu_per_card,
    "float32",
    True,
    list(range(EP))
)

print(f"\nResult: {res}")
all_pass = all(res.values())
print(f"{'PASSED' if all_pass else 'FAILED'}")
# compare_output 内部已自动写入 csv_result/moe_distribute.csv
