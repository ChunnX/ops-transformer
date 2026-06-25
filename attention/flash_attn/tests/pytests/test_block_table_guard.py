"""验证拦截：非 PA 用例传入 block_table 应报错。读取 functional_stc.py 中的用例。"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch, torch_npu
from cann_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata
from core.case_loader import load_case_modules, normalize_params
from core.data import flash_attn_inputs, build_flash_attn_params

torch.npu.set_device(0)

all_cases = load_case_modules(["test_cases.functional_stc"])
non_pa_cases = {k: v for k, v in all_cases.items()
                if not str(v.get("layout_kv", v.get("layout_q", "BNSD"))).startswith("PA_")}

# 只取前 3 条非 PA 用例，够验证即可
candidates = list(non_pa_cases.items())[:3]
if not candidates:
    print("没有非 PA 用例"); sys.exit(1)

for name, raw in candidates:
    p = normalize_params(raw)
    layout = p.get("layout_q", "BNSD")
    layout_kv = p.get("layout_kv", layout)
    print(f"\n{'='*60}")
    print(f"  {name}  layout_kv={layout_kv}  B={p['B']} N1={p['N1']} N2={p.get('N2')}")

    inputs = flash_attn_inputs.generate(p, device=torch.device("npu"),
                                        layout=layout, layout_kv=layout_kv)
    meta_kw, kern_kw, _ = build_flash_attn_params(p, torch.device("npu"), inputs)
    metadata = npu_flash_attn_metadata(**meta_kw)

    # --- 正常调用 (block_table=None) ---
    try:
        out, _ = npu_flash_attn(inputs["q"], inputs["k"], inputs["v"],
                                metadata=metadata, **kern_kw)
        torch.npu.synchronize()
        print(f"  [normal] PASS  out.shape={out.shape}")
    except Exception as e:
        print(f"  [normal] FAIL: {e}")
        continue

    # --- 非法调用：强行塞 block_table ---
    bt = torch.zeros(p["B"], 4, dtype=torch.int32, device="npu")
    kern_kw_bad = dict(kern_kw, block_table=bt)
    try:
        out2, _ = npu_flash_attn(inputs["q"], inputs["k"], inputs["v"],
                                 metadata=metadata, **kern_kw_bad)
        torch.npu.synchronize()
        print(f"  [block_table] NOT REJECTED — 拦截未生效!")
    except Exception as e:
        print(f"  [block_table] REJECTED: {e}")
