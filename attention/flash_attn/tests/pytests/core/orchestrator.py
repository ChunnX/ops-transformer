"""编排器 — 串联数据生成 → 后端计算 → 精度对比。"""

import torch
from typing import Any, Dict, List, Optional

from core.data import InputSpec, flash_attn_inputs
from core.backends.base import Backend
from utils.compare import check_result


def run_case(params: dict,
             primary: Backend,
             golden: Backend,
             spec: InputSpec = None,
             comparators: List[Backend] = None) -> dict:
    """标准测试流程。

    1. 在 primary.device 上生成数据
    2. 拷贝 CPU → golden 计算
    3. primary 计算
    4. 精度对比 (primary vs golden)
    5. (可选) comparator 计算 + 对比

    Returns: {"attn": {...}, "lse": {...}}
    """
    spec = spec or flash_attn_inputs
    comparators = comparators or []
    layout = params.get("input_layout", params.get("layout_q", "BNSD"))

    # 1. 在 primary device 上生成输入
    inputs = spec.generate(params, device=primary.device, layout=layout)

    # 2. CPU golden
    print(f"  [golden] CPU 参考计算...")
    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
    golden_out = golden.compute(cpu_inputs, params)

    # 3. primary 计算
    print(f"  [primary] {primary.name} 计算...")
    primary_out = primary.compute(inputs, params)

    # 4. 精度对比
    cpu_out = golden_out["out"]
    dev_out = primary_out["out"].cpu() if primary_out["out"].is_cuda or str(primary_out["out"].device).startswith("npu") else primary_out["out"]

    print(f"  [compare] CPU vs {primary.name}")
    attn_result = check_result(f"CPU_vs_{primary.name}",
                               cpu_out.float(), dev_out.float(),
                               except_label="CPU", comp_label=primary.name)

    # LSE 对比
    lse_result = {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                  "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    if "lse" in primary_out and "lse" in golden_out:
        dev_lse = primary_out["lse"].cpu() if not isinstance(primary_out["lse"], torch.Tensor) or primary_out["lse"].is_cuda else primary_out["lse"]
        cpu_lse = golden_out["lse"]
        if isinstance(dev_lse, torch.Tensor):
            print(f"  [compare] LSE check")
            lse_result = check_result("LSE", cpu_lse.float(), dev_lse.float(),
                                      except_label="CPU_lse", comp_label=f"{primary.name}_lse")

    # 5. 额外对比 (GPU vs NPU 等)
    for comp in comparators:
        comp_inputs = {k: v.to(comp.device) for k, v in inputs.items()}
        comp_out = comp.compute(comp_inputs, params)
        comp_cpu = comp_out["out"].cpu()
        print(f"  [compare] {primary.name} vs {comp.name}")
        _ = check_result(f"{primary.name}_vs_{comp.name}",
                         dev_out.float(), comp_cpu.float(),
                         except_label=primary.name, comp_label=comp.name)

    return {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}


def _to_stats(val) -> dict:
    if isinstance(val, dict):
        return val
    if val:
        return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    return {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
            "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}
