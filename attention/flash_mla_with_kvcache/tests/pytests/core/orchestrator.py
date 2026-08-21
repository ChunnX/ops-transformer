# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""编排器 — 串联数据生成 → 后端计算 → 精度对比；含 NC 自洽、确定性、静态图(M9) 分支。"""

import gc
from typing import List

import torch

from core.data import InputSpec, flash_mla_with_kvcache_inputs
from core.backends.base import Backend
from utils.compare import check_result, get_tolerance
from utils.data import make_noncontiguous


def run_case(params: dict,
             primary: Backend,
             golden: Backend,
             spec: InputSpec = None,
             case_name: str = None,
             verbose_diff: bool = False,
             static_graph: bool = False) -> dict:
    spec = spec or flash_mla_with_kvcache_inputs
    layout = params.get("input_layout", params.get("layout_q", "TND"))
    output_dtype = params.get("Dtype", "fp16")
    atol, rtol = get_tolerance(output_dtype)

    cpu_inputs = spec.generate(params, device=torch.device("cpu"), layout=layout)
    cpu_inputs["block_table"] = params["block_table"]

    print(f"  [golden] CPU 参考计算...")
    golden_out = golden.compute(cpu_inputs, params)

    print(f"  [primary] {primary.name} 计算...")
    dev_inputs = {k: v.to(primary.device) if isinstance(v, torch.Tensor) else v
                  for k, v in cpu_inputs.items()}
    primary_out_raw = primary.compute(dev_inputs, params)
    primary_out = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                   for k, v in primary_out_raw.items()}
    del primary_out_raw
    primary.clear_cache()

    # --- 确定性（同一输入跑两次，位级一致；M9 基础证据）---
    det_result = None
    if static_graph:
        det_raw = primary.compute(dev_inputs, params)
        det_out = det_raw["out"].cpu()
        del det_raw
        primary.clear_cache()
        exact = torch.equal(primary_out["out"], det_out)
        d = (primary_out["out"].float() - det_out.float()).abs()
        det_result = {
            "passed": exact,
            "max_abs": 0.0 if exact else d.max().item(),
            "mean_abs": 0.0 if exact else d.mean().item(),
            "fail_cnt": 0 if exact else int((d > 0).sum().item()),
            "total": primary_out["out"].numel(),
            "fail_ratio": 0.0 if exact else float((d > 0).sum().item()) / primary_out["out"].numel(),
        }
        del det_out, d

    # --- 静态图（torch.export）vs 动态图 等价（M9）---
    static_result = None
    if static_graph:
        try:
            print(f"  [static] torch.export 静态子图执行...")
            static_raw = primary.compute_static(dev_inputs, params)
            static_out = static_raw["out"].cpu()
            del static_raw
            primary.clear_cache()
            exact = torch.equal(primary_out["out"], static_out)
            d = (primary_out["out"].float() - static_out.float()).abs()
            static_result = {
                "passed": exact,
                "max_abs": 0.0 if exact else d.max().item(),
                "mean_abs": 0.0 if exact else d.mean().item(),
                "fail_cnt": 0 if exact else int((d > 0).sum().item()),
                "total": primary_out["out"].numel(),
                "fail_ratio": 0.0 if exact else float((d > 0).sum().item()) / primary_out["out"].numel(),
            }
            del static_out, d
        except Exception as e:
            import traceback
            static_result = {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                             "fail_cnt": -1, "total": -1, "fail_ratio": float('nan'),
                             "error": f"{e}\n{traceback.format_exc()}"}

    # --- k_cache 非连续 self-consistency check（functional_nc）---
    nc_result = None
    nc_kv_dims = params.get("nc_kv_dims")
    if nc_kv_dims is not None:
        nc_dev_inputs = {}
        for k, v in dev_inputs.items():
            if isinstance(v, torch.Tensor) and k == "k_cache":
                nc_dev_inputs[k] = make_noncontiguous(v, nc_kv_dims)
            else:
                nc_dev_inputs[k] = v
        assert torch.equal(nc_dev_inputs["k_cache"], dev_inputs["k_cache"]), \
            "k_cache NC 后数据与连续时不一致"
        print(f"  [nc_check] nc_kv_dims={nc_kv_dims} "
              f"k_cache_contig={nc_dev_inputs['k_cache'].is_contiguous()} "
              f"k_cache_stride={nc_dev_inputs['k_cache'].stride()}")
        nc_raw = primary.compute(nc_dev_inputs, params)
        nc_cpu = nc_raw["out"].cpu()
        del nc_dev_inputs, nc_raw
        primary.clear_cache()
        contig_f = primary_out["out"].float()
        nc_f = nc_cpu.float()
        exact_match = torch.equal(contig_f, nc_f)
        diff = (contig_f - nc_f).abs()
        max_abs = diff.max().item()
        print(f"  [nc_check] NC vs Contiguous  exact_match={exact_match}  max_abs={max_abs:.8e}")
        nc_result = {
            "passed": exact_match,
            "max_abs": max_abs,
            "mean_abs": diff.mean().item(),
            "fail_cnt": 0 if exact_match else int((diff > 0).sum().item()),
            "total": contig_f.numel(),
            "fail_ratio": 0.0 if exact_match else float((diff > 0).sum().item()) / contig_f.numel(),
        }
        del nc_cpu, diff

    del dev_inputs

    dev_out = primary_out["out"]
    cpu_out = golden_out["out"].float()

    print(f"  [compare] CPU vs {primary.name}")
    attn_result = check_result(f"CPU_vs_{primary.name}", cpu_out, dev_out.float(),
                               except_label="CPU", comp_label=primary.name,
                               verbose_diff=verbose_diff, atol=atol, rtol=rtol)

    lse_result = {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                  "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    if params.get("return_softmax_lse", False) and "lse" in primary_out and "lse" in golden_out:
        dev_lse = primary_out["lse"]
        cpu_lse = golden_out["lse"]
        if isinstance(dev_lse, torch.Tensor) and isinstance(cpu_lse, torch.Tensor):
            cpu_lse = cpu_lse.float().clone()
            dev_lse = dev_lse.float().clone()
            cpu_lse[cpu_lse.isinf() & (cpu_lse > 0)] = float('-inf')
            dev_lse[dev_lse.isinf() & (dev_lse > 0)] = float('-inf')
            print(f"  [compare] LSE check  (shape: cpu={tuple(cpu_lse.shape)} dev={tuple(dev_lse.shape)})")
            lse_result = check_result("LSE", cpu_lse, dev_lse,
                                      except_label="CPU_lse", comp_label=f"{primary.name}_lse",
                                      verbose_diff=verbose_diff, atol=atol, rtol=rtol)

    # --- metadata section 断言（>1-section 用例）---
    meta_info = {}
    md = primary_out.get("metadata")
    if md is not None:
        m_arr = md.cpu().contiguous().to(torch.int32)
        sec_num = int(m_arr[0])
        is_fd = int(m_arr[1])
        meta_info = {"section_num": sec_num, "is_fd": is_fd}
        print(f"  [metadata] sectionNum={sec_num}  isFD={is_fd}")
        if params.get("expect_multi_section", False):
            passed = sec_num >= 2
            meta_info["passed"] = passed
            print(f"  [metadata] expect sectionNum>=2: {'✓' if passed else '✗'} (got {sec_num})")

    ret = {"attn": _to_stats(attn_result), "lse": _to_stats(lse_result)}
    if det_result is not None:
        ret["determinism"] = _to_stats(det_result)
    if static_result is not None:
        ret["static"] = _to_stats(static_result)
    if nc_result is not None:
        ret["nc"] = _to_stats(nc_result)
    if meta_info:
        ret["meta"] = meta_info

    del cpu_inputs, primary_out, golden_out
    del dev_out, cpu_out
    gc.collect()
    return ret


def _to_stats(val) -> dict:
    if isinstance(val, dict):
        return val
    if val:
        return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0,
                "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    return {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
            "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}