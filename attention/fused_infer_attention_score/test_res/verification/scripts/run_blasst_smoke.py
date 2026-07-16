#!/usr/bin/env python3
"""
BLASST 快速烟雾测试 — 用 real .pt 数据验证 NPU 算子能跑通
    python run_blasst_smoke.py
    python run_blasst_smoke.py --compare       # 对比 Python BLASST 仿真精度
    python run_blasst_smoke.py --consistency   # NPU 一致性校验（重复运行 N 次）
"""
import sys, os, argparse, math, torch, torch_npu

torch_npu.npu.set_device(5)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--compare", action="store_true", help="对比 Python BLASST 仿真")
parser.add_argument("--consistency", action="store_true", help="NPU 一致性校验")
parser.add_argument("--consistency-runs", type=int, default=3, help="一致性检查重复次数 (默认 3)")
args = parser.parse_args()

# kernel 内硬编码的 sparseLambda (与 BLASST_SPARSE_LAMBDA 一致)
BLASST_LAMBDA = -3.0

# 0. 加载输入
pt = torch.load(os.path.join(os.path.dirname(__file__), "..", "data", "layer_30_1775227380534.pt"))
inputs = pt["inputs"]
Q = inputs["query"].npu().half()
K = inputs["key"].npu().half()
V = inputs["value"].npu().half()
scale = inputs["scale"].item()
num_heads = int(inputs["num_heads"].item())
num_kv_heads = int(inputs["num_kv_heads"].item())
Sq = int(inputs["actual_seq_qlen"].item())
Skv = int(inputs["actual_seq_kvlen"].item())

print(f"Input: B=1, Sq={Sq}, Skv={Skv}, Nq={num_heads}, Nkv={num_kv_heads}, D=128, scale={scale:.4f}")
print(f"BLASST lambda={BLASST_LAMBDA}")

# 1. NPU 算子
print("\n[1] Running NPU FIA...")
def run_npu():
    r, _ = torch_npu.npu_fused_infer_attention_score(
        query=Q, key=K, value=V, atten_mask=None, block_table=None,
        input_layout="TND", block_size=0,
        actual_seq_lengths=[Sq], actual_seq_lengths_kv=[Skv],
        num_key_value_heads=num_kv_heads, num_heads=num_heads,
        scale=scale, sparse_mode=0,
    )
    torch.npu.synchronize()
    return r

result = run_npu()
print(f"    output: {result.shape} max={result.max().item():.4f} min={result.min().item():.4f}")

# 2. 一致性校验
if args.consistency:
    print(f"\n[2] Consistency check ({args.consistency_runs} runs)...")
    results = []
    for i in range(args.consistency_runs):
        r = run_npu()
        results.append(r.float().cpu())
    max_diff = 0.0
    for i in range(1, len(results)):
        diff = (results[i] - results[0]).abs()
        if diff.max().item() > max_diff:
            max_diff = diff.max().item()
    print(f"    max_diff across {args.consistency_runs} runs: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("    PASS (deterministic)")
    else:
        print("    FAIL (non-deterministic)")

# 3. 精度对比 Python BLASST
if args.compare:
    print(f"\n[{2+args.consistency}] Python BLASST simulation...")
    from row_loop_python_blasst import optimized_blasst_sim
    q = Q.float().cpu()
    k = K.float().cpu()
    v = V.float().cpu()
    out_blasst = optimized_blasst_sim(
        q, k, v,
        actual_qlen=torch.tensor(Sq), actual_kvlen=torch.tensor(Skv),
        scale=scale, threshold_log=BLASST_LAMBDA, sparse_mode=0,
    )[0]
    ref = out_blasst.unsqueeze(0)
    diff = (result.float().cpu() - ref.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        result.float().cpu().reshape(-1), ref.float().reshape(-1), dim=0
    )
    print(f"    max_diff={diff.max().item():.6f} cos_sim={cos.item():.6f}")
    print(f"    {'PASS' if cos.item() > 0.99 else 'WARN: cos_sim < 0.99'}")
