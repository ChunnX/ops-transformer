"""编排器 — 串联数据生成 → 后端计算 → 精度对比。"""

import gc
import torch
from typing import Any, Dict, List, Optional

from core.data import InputSpec, flash_attn_inputs
from core.backends.base import Backend
from utils.compare import check_result
from utils.data import trans_bnsd_to_layout


def run_case(params: dict,
             primary: Backend,
             golden: Backend,
             spec: InputSpec = None,
             comparators: List[Backend] = None) -> dict:
    spec = spec or flash_attn_inputs
    comparators = comparators or []
    layout = params.get("input_layout", params.get("layout_q", "BNSD"))
    out_layout = params.get("layout_out", layout)

    # 1. 统一在 CPU 生成数据（目标 layout：PA/TND 等格式不变）
    inputs = spec.generate(params, device=torch.device("cpu"), layout=layout,
                           layout_kv=params.get("layout_kv", layout))
    if str(params.get("layout_kv", "")).startswith("PA_"):
        inputs["block_table"] = params["block_table"]
    elif "block_table" in params and params["block_table"] is not None:
        inputs["block_table"] = params["block_table"]

    # 2. CPU golden 先算（数据已在 CPU，零拷贝）
    print(f"  [golden] CPU 参考计算...")
    golden_out = golden.compute(inputs, params)

    # 3. primary 后算（数据拷贝到 device）
    print(f"  [primary] {primary.name} 计算...")
    dev_inputs = {k: v.to(primary.device) for k, v in inputs.items()}
    primary_out = primary.compute(dev_inputs, params)

    # 4. 精度对比 —— CPU golden 始终 BNSD，统一转 target layout
    cpu_raw = golden_out["out"]
    dev_raw = primary_out["out"].cpu() if primary_out["out"].is_cuda or str(primary_out["out"].device).startswith("npu") else primary_out["out"]

    cpu_out = trans_bnsd_to_layout(cpu_raw, out_layout, **params) if out_layout != "BNSD" else cpu_raw
    dev_out = dev_raw

    print(f"  [compare] CPU vs {primary.name}")
    attn_result = check_result(f"CPU_vs_{primary.name}",
                               cpu_out.float(), dev_out.float(),
                               except_label="CPU", comp_label=primary.name)

    # LSE 对比（仅当用例要求 return_softmax_lse 时才比）
    lse_result = {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                  "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    if params.get("return_softmax_lse", 0) and "lse" in primary_out and "lse" in golden_out:
        dev_lse = primary_out["lse"]
        if isinstance(dev_lse, torch.Tensor):
            dev_lse = dev_lse.cpu().float()
            cpu_lse = golden_out["lse"]
            if isinstance(cpu_lse, torch.Tensor):
                cpu_lse = cpu_lse.float()
                print(f"  [compare] LSE check")
                lse_result = check_result("LSE", cpu_lse, dev_lse,
                                          except_label="CPU_lse", comp_label=f"{primary.name}_lse")

    # 5. 额外对比
    for comp in comparators:
        comp_inputs = {k: v.to(comp.device) for k, v in inputs.items()}
        comp_out = comp.compute(comp_inputs, params)
        comp_cpu = comp_out["out"].cpu()
        print(f"  [compare] {primary.name} vs {comp.name}")
        _ = check_result(f"{primary.name}_vs_{comp.name}",
                         dev_out.float(), comp_cpu.float(),
                         except_label=primary.name, comp_label=comp.name)

    ret = {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}

    del inputs, dev_inputs, primary_out, golden_out
    del cpu_raw, dev_raw, cpu_out, dev_out
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()

    return ret


def _to_stats(val) -> dict:
    if isinstance(val, dict):
        return val
    if val:
        return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    return {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
            "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}
