"""编排器 — 串联数据生成 → 后端计算 → 精度对比 + golden 持久化 + 子进程隔离。"""

import gc
import os
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from core.data import InputSpec, flash_attn_inputs
from core.backends.base import Backend
from utils.compare import check_result
from utils.data import trans_bnsd_to_layout

_ctx = mp.get_context('spawn')


# ─── Golden 持久化 ───────────────────────────────────────────────────────────

def save_golden_case(case_name: str, inputs: dict,
                     golden_out: dict, params: dict, golden_dir: str):
    safe_name = case_name.replace("/", "_")
    case_dir = Path(golden_dir) / safe_name
    case_dir.mkdir(parents=True, exist_ok=True)

    inputs_save = {}
    for k, v in inputs.items():
        inputs_save[k] = v.cpu() if isinstance(v, torch.Tensor) else v
    for k, v in params.items():
        if k not in inputs_save:
            inputs_save[k] = v.cpu() if isinstance(v, torch.Tensor) else v
    torch.save(inputs_save, case_dir / "inputs.pt")

    golden_save = {}
    for k, v in golden_out.items():
        if isinstance(v, torch.Tensor):
            golden_save[k] = v.cpu().float()
    torch.save(golden_save, case_dir / "golden.pt")


def load_golden_case(case_name: str, golden_dir: str):
    safe_name = case_name.replace("/", "_")
    case_dir = Path(golden_dir) / safe_name

    if not case_dir.exists():
        raise FileNotFoundError(f"Golden 目录不存在: {case_dir}")

    raw = torch.load(case_dir / "inputs.pt", map_location="cpu", weights_only=False)
    golden = torch.load(case_dir / "golden.pt", map_location="cpu", weights_only=False)

    inputs = {}
    params = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v
        else:
            params[k] = v

    return inputs, golden, params


# ─── 子进程隔离 NPU 执行 ─────────────────────────────────────────────────────

def _npu_worker(inputs_path, result_path, params, device_id):
    import torch
    import torch_npu
    torch.npu.set_device(device_id)

    from core.backends.npu import NPUBackend
    primary = NPUBackend(device_id)

    inputs = torch.load(inputs_path, map_location="cpu")
    dev_inputs = {k: v.to(primary.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
    out = primary.compute(dev_inputs, params)

    result_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                  for k, v in out.items()}
    torch.save(result_cpu, result_path)


def run_npu_isolated(inputs, params, device_id, timeout=300):
    tmp_inputs = f"/tmp/_fa_inputs_{os.getpid()}.pt"
    tmp_result = f"/tmp/_fa_result_{os.getpid()}.pt"

    tensor_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    torch.save(tensor_inputs, tmp_inputs)

    p = _ctx.Process(target=_npu_worker,
                     args=(tmp_inputs, tmp_result, params, device_id))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join()
        _cleanup(tmp_inputs, tmp_result)
        return None, "TIMEOUT: kernel hang"

    _cleanup(tmp_inputs)

    if p.exitcode != 0:
        _cleanup(tmp_result)
        return None, f"CRASHED: exit code {p.exitcode}"

    if not os.path.exists(tmp_result):
        return None, "NO_OUTPUT: 子进程未写出结果文件"

    result = torch.load(tmp_result, map_location="cpu", weights_only=False)
    _cleanup(tmp_result)
    return result, None


def _cleanup(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def _npu_health_check(device_id, timeout=10):
    def _probe(result_path, dev_id):
        import torch, torch_npu
        torch.npu.set_device(dev_id)
        x = torch.ones(1, device=f"npu:{dev_id}")
        y = x + x
        torch.npu.synchronize()
        torch.save({"ok": True}, result_path)

    tmp = f"/tmp/_fa_health_{os.getpid()}.pt"
    p = _ctx.Process(target=_probe, args=(tmp, device_id))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()
        _cleanup(tmp)
        return False
    ok = os.path.exists(tmp)
    _cleanup(tmp)
    return ok


# ─── 核心运行逻辑 ────────────────────────────────────────────────────────────

def run_case(params: dict,
             primary: Backend,
             golden: Backend,
             spec: InputSpec = None,
             comparators: List[Backend] = None,
             golden_dir: str = None,
             case_name: str = None,
             verbose_diff: bool = False,
             visualize: bool = False,
             viz_dir: str = "./viz_output",
             fail_analysis: bool = False) -> dict:
    spec = spec or flash_attn_inputs
    comparators = comparators or []
    layout = params.get("input_layout", params.get("layout_q", "BNSD"))
    out_layout = params.get("layout_out", layout)

    inputs = spec.generate(params, device=primary.device, layout=layout,
                           layout_kv=params.get("layout_kv", layout))
    if str(params.get("layout_kv", "")).startswith("PA_"):
        inputs["block_table"] = params["block_table"].to(primary.device)
    elif "block_table" in params and params["block_table"] is not None:
        inputs["block_table"] = params["block_table"].to(primary.device)

    print(f"  [primary] {primary.name} 计算...")
    primary_out = primary.compute(inputs, params)

    print(f"  [golden] CPU 参考计算...")
    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
    golden_out = golden.compute(cpu_inputs, params)

    cpu_raw = golden_out["out"]
    dev_raw = primary_out["out"].cpu() if primary_out["out"].is_cuda or str(primary_out["out"].device).startswith("npu") else primary_out["out"]

    cpu_out = trans_bnsd_to_layout(cpu_raw, out_layout, **params) if out_layout != "BNSD" else cpu_raw
    dev_out = dev_raw

    print(f"  [compare] CPU vs {primary.name}")
    attn_result = check_result(f"CPU_vs_{primary.name}",
                               cpu_out.float(), dev_out.float(),
                               except_label="CPU", comp_label=primary.name,
                               verbose_diff=verbose_diff)

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
                                          except_label="CPU_lse", comp_label=f"{primary.name}_lse",
                                          verbose_diff=verbose_diff)

    for comp in comparators:
        comp_inputs = {k: v.to(comp.device) for k, v in inputs.items()}
        comp_out = comp.compute(comp_inputs, params)
        comp_cpu = comp_out["out"].cpu()
        print(f"  [compare] {primary.name} vs {comp.name}")
        _ = check_result(f"{primary.name}_vs_{comp.name}",
                         dev_out.float(), comp_cpu.float(),
                         except_label=primary.name, comp_label=comp.name)

    ret = {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}

    if golden_dir and case_name:
        save_golden_case(case_name, cpu_inputs, golden_out, params, golden_dir)

    attn_passed = ret["attn"].get("passed", True) if isinstance(ret["attn"], dict) else True
    if not attn_passed:
        if fail_analysis:
            try:
                from utils.compare import analyze_fail_distribution
                analyze_fail_distribution(
                    test_name=case_name or "unknown",
                    expect=cpu_out.float(),
                    result=dev_out.float(),
                    dump_dir=viz_dir,
                )
            except Exception as e:
                print(f"  [warn] fail_analysis 异常: {e}")
        if visualize:
            try:
                from utils.precision_visual import visualize_from_tensors
                visualize_from_tensors(
                    cpu_out.float().numpy(),
                    dev_out.float().numpy(),
                    case_name=case_name or "unknown",
                    out_dir=viz_dir,
                )
            except Exception as e:
                print(f"  [warn] visualize 异常: {e}")

    del inputs, cpu_inputs, primary_out, golden_out
    del cpu_raw, dev_raw, cpu_out, dev_out
    gc.collect()
    primary.clear_cache()

    return ret


def run_case_load(case_name: str, golden_dir: str, primary: Backend,
                  device_id: int = 0, timeout: int = 300,
                  verbose_diff: bool = False,
                  visualize: bool = False,
                  viz_dir: str = "./viz_output",
                  fail_analysis: bool = False,
                  use_isolation: bool = True) -> dict:
    inputs, golden_data, params = load_golden_case(case_name, golden_dir)

    if use_isolation and primary.name == "npu":
        npu_out, error = run_npu_isolated(inputs, params, device_id, timeout)
        if error:
            return {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                             "fail_cnt": -1, "total": -1, "fail_ratio": float('nan'),
                             "error": error},
                    "lse": {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                            "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}}
    else:
        dev_inputs = {k: v.to(primary.device) for k, v in inputs.items()}
        npu_out_raw = primary.compute(dev_inputs, params)
        npu_out = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in npu_out_raw.items()}

    out_layout = params.get("layout_out", params.get("layout_q", "BNSD"))

    cpu_raw = golden_data["out"]
    dev_raw = npu_out["out"]

    cpu_out = trans_bnsd_to_layout(cpu_raw, out_layout, **params) if out_layout != "BNSD" else cpu_raw

    attn_result = check_result("CPU_vs_NPU", cpu_out.float(), dev_raw.float(),
                               except_label="CPU", comp_label=primary.name,
                               verbose_diff=verbose_diff)

    lse_result = {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                  "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    if params.get("return_softmax_lse", 0) and "lse" in golden_data and "lse" in npu_out:
        dev_lse = npu_out["lse"]
        cpu_lse = golden_data["lse"]
        if isinstance(dev_lse, torch.Tensor) and isinstance(cpu_lse, torch.Tensor):
            lse_result = check_result("LSE", cpu_lse.float(), dev_lse.float(),
                                      except_label="CPU_lse", comp_label=f"{primary.name}_lse",
                                      verbose_diff=verbose_diff)

    ret = {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}

    attn_passed = ret["attn"].get("passed", True) if isinstance(ret["attn"], dict) else True
    if not attn_passed:
        if fail_analysis:
            try:
                from utils.compare import analyze_fail_distribution
                analyze_fail_distribution(
                    test_name=case_name,
                    expect=cpu_out.float(),
                    result=dev_raw.float(),
                    dump_dir=viz_dir,
                )
            except Exception as e:
                print(f"  [warn] fail_analysis 异常: {e}")
        if visualize:
            try:
                from utils.precision_visual import visualize_from_tensors
                visualize_from_tensors(
                    cpu_out.float().numpy(),
                    dev_raw.float().numpy(),
                    case_name=case_name,
                    out_dir=viz_dir,
                )
            except Exception as e:
                print(f"  [warn] visualize 异常: {e}")

    return ret


def _to_stats(val) -> dict:
    if isinstance(val, dict):
        return val
    if val:
        return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    return {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
            "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}
