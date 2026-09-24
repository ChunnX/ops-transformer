"""对泛化矩阵做 NPU vs CPU 全量功能校验（不依赖 ATK 精度标准）。"""

import json
import sys

sys.path.insert(0, "/data/fangziyang/work_mss_moe_unpermute")

import torch
import torch_npu

from executor_moe_token_unpermute_with_routing_map import build_inputs, run_cpu

MATRIX = "/data/fangziyang/work_mss_moe_unpermute/atk_moe_unpermute_mss_matrix.json"


def check_case(spec, cid):
    inputs = build_inputs(spec, torch.device("npu:0"))
    out_tokens, out_index, permute_token_id, permute_probs = torch.ops.npu._npu_moe_token_unpermute_with_routing_map(
        inputs["permuted_tokens"],
        inputs["sorted_indices"],
        inputs["restore_shape"],
        probs=inputs["probs"],
        routing_map=inputs["routing_map"],
        drop_and_pad=inputs["drop_and_pad"],
    )

    # half 精度：NPU 内部按 fp32 累加后转回 token dtype，参考也按 fp32 计算
    inputs_cpu = build_inputs(spec, torch.device("cpu"))
    pt = inputs_cpu["permuted_tokens"].float()
    si = inputs_cpu["sorted_indices"]
    pr = inputs_cpu["probs"].float()
    num_tokens, hidden = inputs_cpu["restore_shape"]
    num_experts = inputs_cpu["num_experts"]
    capacity = inputs_cpu["capacity"]
    expert_indices = torch.arange(num_experts).unsqueeze(-1)
    token_indices = si.view(num_experts, capacity)
    prob_indices = (expert_indices * num_tokens + token_indices).view(-1).clamp(0, num_experts * num_tokens - 1)
    permuted_probs = pr.T.contiguous().view(-1).index_select(0, prob_indices)
    token_dtype = inputs["permuted_tokens"].dtype
    is_mixed = inputs["is_mixed"]
    if token_dtype in (torch.bfloat16, torch.float16) and not is_mixed:
        # 非混精 half：NPU 先按 token dtype 做乘法（乘积量化），再 fp32 累加
        weighted = (inputs_cpu["permuted_tokens"].to(token_dtype) * permuted_probs.to(token_dtype).unsqueeze(-1)).float()
    else:
        weighted = pt * permuted_probs.unsqueeze(-1)
    sv, so = si.sort(dim=-1, stable=True, descending=False)
    ref_fp32 = torch.zeros([num_tokens, hidden], dtype=torch.float32)
    ref_fp32.index_add_(0, sv, weighted[so])
    ref_tokens = ref_fp32.to(token_dtype)

    ref_index = so.to(si.dtype)
    ref_token_id = sv.to(si.dtype)
    ref_probs = permuted_probs.to(pr.dtype)

    out_tokens = out_tokens.cpu()
    permute_probs = permute_probs.cpu()

    # 1) 确定性输出（permute_probs / out_index / permute_token_id）必须逐位一致
    exact_ok = (
        torch.equal(out_index.cpu(), ref_index)
        and torch.equal(permute_token_id.cpu(), ref_token_id)
        and torch.equal(permute_probs, ref_probs)
    )

    # 2) unpermuted_tokens：按 dtype 给容差（bf16/fp16 累加 ULP 差异）
    diff = (out_tokens.float() - ref_tokens.float()).abs()
    dtype = inputs["permuted_tokens"].dtype
    if dtype == torch.bfloat16 or dtype == torch.float16:
        tol = 0.05
    else:
        tol = 1e-3
    max_diff = diff.max().item()
    tokens_ok = max_diff <= tol

    ok = exact_ok and tokens_ok
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] case {cid:02d} dtype={str(spec['dtype']):5s} "
        f"E={spec['num_experts']:5d} C={spec['capacity']:5d} "
        f"T={spec['num_tokens']:5d} H={spec['hidden']:5d} "
        f"exact={exact_ok} tokens_max_diff={max_diff:.6g}"
    )
    if not ok:
        if not exact_ok:
            print("   exact mismatch:",
                  "index" if not torch.equal(out_index.cpu(), ref_index) else "",
                  "token_id" if not torch.equal(permute_token_id.cpu(), ref_token_id) else "",
                  "probs" if not torch.equal(permute_probs, ref_probs) else "")
    return ok


def main():
    cases = json.load(open(MATRIX))
    passed = 0
    for c in cases:
        spec = json.loads(c["inputs"][0]["range_values"])
        try:
            if check_case(spec, c["id"]):
                passed += 1
        except Exception as e:
            print(f"[FAIL] case {c['id']:02d} exception: {e}")
    print(f"\n==== matrix accuracy: {passed}/{len(cases)} passed ====")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
