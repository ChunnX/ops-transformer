#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import sys
import argparse
import importlib
import itertools
import torch
import numpy as np
from backends.cpu_impl import tforward
from utils.data import generate_qkv, generate_npu_mask, trans_bnsd_to_layout, gen_block_table
from utils.compare import data_compare_benchmark_new, Result, check_result, analyze_fail_distribution
from utils.io import save_tensor_to_txt, load_tensor_from_txt

_DEFAULT_CASE_FILES = ["base", "fia_stc", "functional_stc",
                        "functional_redline_train_tnd", "functional_redline_train",
                        "functional_redline_infer", "performance_redline_infer"]

# ─────────────────── 动态导入 GPU / NPU 实现 ───────────────────
GPU_AVAILABLE = False
flash_attn_gpu = None
NPU_AVAILABLE = False
flash_attn_npu = None
flash_attn_metadata_only = None
flash_attn_npu_graph = None

try:
    from backends.gpu_impl import flash_attn_gpu as _gpu_func
    flash_attn_gpu = _gpu_func
    GPU_AVAILABLE = True
except ImportError:
    pass

try:
    from backends.npu_impl import flash_attn_npu as _npu_func
    from backends.npu_impl import flash_attn_metadata_only as _meta_func
    from backends.npu_impl import flash_attn_npu_graph as _graph_func
    flash_attn_npu = _npu_func
    flash_attn_metadata_only = _meta_func
    flash_attn_npu_graph = _graph_func
    NPU_AVAILABLE = True
except ImportError:
    pass

# ─────────────────── 用例加载 ───────────────────
def load_case_modules(module_names):
    all_cases = {}
    for mod_name in module_names:
        mod = importlib.import_module(mod_name)
        for case_name, case_dict in mod.TestCases.items():
            keys = list(case_dict.keys())
            values = [case_dict[k] for k in keys]
            combos = list(itertools.product(*values))
            if len(combos) == 1:
                all_cases[f"{mod_name}/{case_name}"] = dict(zip(keys, combos[0]))
            else:
                for i, combo in enumerate(combos):
                    all_cases[f"{mod_name}/{case_name}_{i}"] = dict(zip(keys, combo))
    return all_cases


def normalize_case(raw):
    """将 test_cases 模块字段转换为 call_flash_attn 所需 kwargs 格式。"""
    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    c["input_layout"] = layout_q
    c.setdefault("layout_kv",   c.get("layout_kv", layout_q))
    c.setdefault("layout_out",  c.get("layout_out", layout_q))
    c.setdefault("N2",          c.get("N1"))
    c.setdefault("S2",          c.get("S1"))
    c.setdefault("DV",          c.get("D"))
    c.setdefault("DRope",       0)
    
    c.setdefault("q_start_idx", 0)
    c.setdefault("win_left",   2147483647)
    c.setdefault("win_right",  2147483647)
    c.setdefault("prefix",      [])
    c.setdefault("q_range",     None)
    c.setdefault("k_range",     None)
    c.setdefault("v_range",     None)
    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)
    if layout_q == "TND":
        c.setdefault("seqused_q", [c["cu_seqlens_q"][i+1] - c["cu_seqlens_q"][i]
                                    for i in range(len(c["cu_seqlens_q"]) - 1)])
        if c.get("layout_kv", layout_q) not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            c.setdefault("cu_seqlens_kv", list(c["cu_seqlens_q"]))
            c.setdefault("seqused_kv", [c["cu_seqlens_kv"][i+1] - c["cu_seqlens_kv"][i]
                                         for i in range(len(c["cu_seqlens_kv"]) - 1)])
        c["B"] = 1
    layout_kv_val = c.get("layout_kv", layout_q)
    if layout_kv_val in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("cu_seqlens_kv", None)
        c.setdefault("block_size", c.get("block_size"))
        c.setdefault("block_table_shape", c.get("block_table_shape"))
        block_size = c.get("block_size", 1)
        block_table_shape = c.get("block_table_shape", [])
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)
        seqused_kv = c.get("seqused_kv")
        if ("block_table" in c.keys()) and (c.get("block_table") is not None):
            block_table = torch.tensor(c.get("block_table"), dtype=torch.int32)
            c["block_table"] = block_table
        else:
            b_val = c.get("B")
            if block_table_shape:
                b_val = block_table_shape[0]
                max_block_num_per_batch = block_table_shape[1]
            else:
                if layout_q == "TND":
                    b_val = len(c["cu_seqlens_q"]) - 1
                max_block_num_per_batch = (max(seqused_kv) + block_size - 1) // block_size
            block_table = torch.full((b_val, max_block_num_per_batch), -1, dtype=torch.int32)
            block_idx = 0
            for i in range(b_val):
                b_seq = seqused_kv[i] if len(seqused_kv) > i else seqused_kv[0]
                b_block_num = (b_seq + block_size - 1) // block_size
                for j in range(b_block_num):
                    block_table[i][j] = block_idx
                    block_idx += 1
            c["block_table"] = block_table
        c["actual_b"] = block_table.shape[0]
    return c


def resolve_case_ids(case_id_arg, all_cases):
    if case_id_arg == "all":
        return sorted(all_cases.keys())
    ids = [x.strip() for x in case_id_arg.split(",")]
    result = []
    missing = []
    for cid in ids:
        full_names = [k for k in all_cases if k.endswith(f"/{cid}") or k == cid or k.rsplit("/", 1)[-1].startswith(f"{cid}_")]
        if len(full_names) >= 1:
            result.extend(full_names)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] 以下 case 不存在: {missing}")
    return result


# ─────────────────── 核心测试编排 ───────────────────
def call_flash_attn(test_name, dump_tensors=False, dump_dir="./dump_output",
                    verbose_diff=False, visualize=False, viz_dir="./viz_output",
                    meta_only=False, compare_mode=False, graph_mode=False,
                    load_gpu_dump=None, load_npu_dump=None, fail_analysis=False,
                    **kwargs):
    b          = kwargs.get("B", 1)
    n1         = kwargs.get("N1")
    n2         = kwargs.get("N2", n1)
    sq         = kwargs.get("S1", -1)
    skv        = kwargs.get("S2", sq)
    d          = kwargs.get("D")
    d_v        = kwargs.get("DV", d)
    d_rope     = kwargs.get("DRope", 0)
    input_layout = kwargs.get("input_layout")
    layout_kv    = kwargs.get("layout_kv", input_layout)
    output_layout = kwargs.get('layout_out')
    q_start_idx  = kwargs.get("q_start_idx", 0)
    dtype = kwargs.get("Dtype", "bf16")
    pttype = torch.float16 if dtype == "fp16" else torch.bfloat16
    input_dtype = pttype
    mask_mode = kwargs.get("mask_mode", None)
    win_left  = kwargs.get("win_left",  2147483647)
    win_right = kwargs.get("win_right", 2147483647)
    prefix      = kwargs.get("prefix", [])
    q_range     = kwargs.get("q_range", None)
    k_range     = kwargs.get("k_range", None)
    v_range     = kwargs.get("v_range", None)
    lse_flag = kwargs.get("return_softmax_lse", 0)
    sq_gen = sq;  skv_gen = skv
    if input_layout == "TND":
        cu_q = kwargs["cu_seqlens_q"]
        sq = max(cu_q[i+1] - cu_q[i] for i in range(len(cu_q) - 1))
        sq_gen = cu_q[-1]
        if layout_kv in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            skv = max(kwargs["seqused_kv"])
            skv_gen = sum(kwargs["seqused_kv"])
        else:
            cu_kv = kwargs["cu_seqlens_kv"]
            skv = max(cu_kv[i+1] - cu_kv[i] for i in range(len(cu_kv) - 1))
            skv_gen = cu_kv[-1]
        kwargs["S1"] = sq
        kwargs["S2"] = skv
    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(b, n1, n2, sq_gen, skv_gen, d, d_v, d_rope,
                                                    input_layout, input_dtype,
                                                    q_range=q_range, k_range=k_range, v_range=v_range)

    # 1. CPU Golden
    print(f"[{test_name}] CPU 参考计算...")
    out_cpu, x_max, x_sum = tforward(qf, kf, v, **kwargs)
    out_cpu_out_layout = trans_bnsd_to_layout(out_cpu, output_layout, **kwargs)
    lse_out = torch.log(x_sum) + x_max
    if dump_tensors:
        dump_path = os.path.join(dump_dir, test_name)
        os.makedirs(dump_path, exist_ok=True)
        save_tensor_to_txt(q, os.path.join(dump_path, "q.txt"))
        save_tensor_to_txt(k, os.path.join(dump_path, "k.txt"))
        save_tensor_to_txt(v, os.path.join(dump_path, "v.txt"))
        save_tensor_to_txt(out_cpu_out_layout.float(), os.path.join(dump_path, "cpu_out.txt"))

    # 2. GPU / NPU
    gpu_out = None
    npu_out = None

    if kwargs.get("_use_gpu", False):
        if not GPU_AVAILABLE:
            print(f"[{test_name}] 警告: GPU 不可用，但要求 GPU 模式，尝试从 dump 加载。")
        if GPU_AVAILABLE:
            print(f"[{test_name}] GPU 计算...")
            atten_mask = generate_npu_mask(b, sq, skv, mask_mode, win_left, win_right, prefix)
            device = torch.cuda.current_device()
            q_gpu = q.cuda(device); k_gpu = k.cuda(device); v_gpu = v.cuda(device)
            q_rope_gpu = q_rope.cuda(device) if q_rope is not None else None
            k_rope_gpu = k_rope.cuda(device) if k_rope is not None else None
            mask_gpu = atten_mask.cuda(device) if atten_mask is not None else None
            gpu_out_bnsd = flash_attn_gpu(q_gpu, k_gpu, v_gpu, q_rope_gpu, k_rope_gpu, mask_gpu, **kwargs)
            gpu_out = trans_bnsd_to_layout(gpu_out_bnsd, output_layout, **kwargs).cpu()
            if dump_tensors:
                dump_path = os.path.join(dump_dir, test_name)
                os.makedirs(dump_path, exist_ok=True)
                save_tensor_to_txt(gpu_out.float(), os.path.join(dump_path, "gpu_out.txt"))
                print(f"[{test_name}] 已保存 q/k/v/gpu_out → {dump_path}/")
        else:
            print(f"[{test_name}] GPU 不可用，跳过计算。")
    if gpu_out is None or load_gpu_dump:
        gpu_dump_path = load_gpu_dump if load_gpu_dump else os.path.join(dump_dir, test_name, "gpu_out.txt")
        if os.path.exists(gpu_dump_path):
            print(f"[{test_name}] 从 dump 加载 GPU 结果: {gpu_dump_path}")
            gpu_out = load_tensor_from_txt(gpu_dump_path, target_dtype=torch.float32, target_device='cpu')
        else:
            print(f"[{test_name}] 未找到 GPU dump 文件: {gpu_dump_path}")

    if not kwargs.get("_use_gpu", False):
        if not NPU_AVAILABLE:
            print(f"[{test_name}] 警告: NPU 不可用，尝试从 dump 加载。")
        if NPU_AVAILABLE and (n1 >= n2 and n1 % n2 == 0):
            atten_mask = generate_npu_mask(b, sq, skv, mask_mode, win_left, win_right, prefix)
            if graph_mode:
                print(f"[{test_name}] NPU 图模式计算...")
                npu_out, lse_npu = flash_attn_npu_graph(q, k, v, q_rope, k_rope, atten_mask, **kwargs)
            else:
                print(f"[{test_name}] NPU 单算子模式计算...")
                npu_out, lse_npu = flash_attn_npu(q, k, v, q_rope, k_rope, atten_mask, **kwargs)
        elif NPU_AVAILABLE:
            print(f"[{test_name}] 跳过 NPU: N1={n1} 不满足 N1>=N2 且 N1%N2==0 (N2={n2})")
        else:
            print(f"[{test_name}] NPU 不可用，跳过计算。")
    if npu_out is None or load_npu_dump:
        npu_dump_path = load_npu_dump if load_npu_dump else os.path.join(dump_dir, test_name, "npu_out.txt")
        if os.path.exists(npu_dump_path):
            print(f"[{test_name}] 从 dump 加载 NPU 结果: {npu_dump_path}")
            npu_out = load_tensor_from_txt(npu_dump_path, target_dtype=torch.float32, target_device='cpu')
        else:
            print(f"[{test_name}] 未找到 NPU dump 文件: {npu_dump_path}")

    # 3. 精度对比
    passed_cpu_gpu = False
    passed_cpu_npu = False
    passed_gpu_npu = False

    if compare_mode:
        if gpu_out is not None and npu_out is not None:
            print(f"\n{'='*40} 三方对比模式（详细精度统计） {'='*40}")
            dtype_str = kwargs.get("Dtype", "fp16")
            params = {
                "op_name": "flash_attn", "case_name": test_name,
                "dtype_output": [dtype_str], "dtype_input": [dtype_str],
                "red_range": {
                    "fp32": "0.000001/0.00001/0.0001/0.0005",
                    "fp16": "0.001/0.002/0.005/0.01",
                    "bf16": "0.001/0.002/0.005/0.01",
                },
                "bm_cmp_std": {
                    "fp32": {"max_re_rtol": 10.0, "avg_re_rtol": 2.0, "rmse_rtol": 2.0,
                             "small_value": 1e-06, "small_value_atol": 0.0},
                    "fp16": {"max_re_rtol": 10.0, "avg_re_rtol": 2.0, "rmse_rtol": 2.0,
                             "small_value": 0.001, "small_value_atol": 0.001},
                    "bf16": {"max_re_rtol": 10.0, "avg_re_rtol": 2.0, "rmse_rtol": 2.0,
                             "small_value": 1e-07, "small_value_atol": 0.004},
                },
            }
            str1, str2, data = data_compare_benchmark_new(params, npu_out.float().numpy(),
                                                           gpu_out.float().numpy(),
                                                           out_cpu_out_layout.float().numpy(),
                                                           dtype_str, i=0)
            print("\n====== 三方对比结果 ======")
            print(f"状态: {str1}, 原因: {str2}")
            passed_cpu_gpu = (str1 == 'Pass')
            passed_cpu_npu = (str1 == 'Pass')
            passed_gpu_npu = (str1 == 'Pass')
        else:
            print(f"[{test_name}] 缺少 GPU 或 NPU 结果，无法执行三方对比，使用标准对比模式。")
            compare_mode = False

    if not compare_mode:
        if gpu_out is not None:
            print(f"\n{'='*40} 对比 CPU vs GPU {'='*40}")
            passed_cpu_gpu = check_result(test_name + "_CPU_vs_GPU", out_cpu_out_layout.float(), gpu_out.float(),
                                          except_label="CPU", comp_label="GPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 GPU 结果，跳过 CPU vs GPU 对比。")
        if npu_out is not None:
            print(f"\n{'='*40} 对比 CPU vs NPU {'='*40}")
            passed_cpu_npu = check_result(test_name + "_CPU_vs_NPU", out_cpu_out_layout.float(), npu_out.float(),
                                          except_label="CPU", comp_label="NPU", verbose_diff=verbose_diff)
            if fail_analysis and isinstance(passed_cpu_npu, dict) and not passed_cpu_npu["passed"]:
                analyze_fail_distribution(test_name, out_cpu_out_layout, npu_out, dump_dir=dump_dir, **kwargs)
        else:
            print(f"[{test_name}] 缺少 NPU 结果，跳过 CPU vs NPU 对比。")
        if gpu_out is not None and npu_out is not None:
            print(f"\n{'='*40} 对比 GPU vs NPU {'='*40}")
            passed_gpu_npu = check_result(test_name + "_GPU_vs_NPU", gpu_out.float(), npu_out.float(),
                                          except_label="GPU", comp_label="NPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 GPU 或 NPU 结果，跳过 GPU vs NPU 对比。")

    if visualize:
        try:
            from utils.precision_visual import visualize_from_tensors
            if npu_out is not None:
                visualize_from_tensors(out_cpu_out_layout.float(), npu_out.float(), case_name=test_name, out_dir=viz_dir)
            if gpu_out is not None:
                visualize_from_tensors(out_cpu_out_layout.float(), gpu_out.float(), case_name=test_name + "_gpu", out_dir=viz_dir)
        except ImportError:
            print("[WARN] precision_visual 导入失败，请确认 matplotlib 已安装")
        except Exception as exc:
            print(f"[WARN] 可视化异常: {exc}")

    passed_lse = 1
    if lse_flag == 1 and npu_out is not None:
        passed_lse = 0
        print(f"\n{'='*40} 对比 lse goldem vs lse npu {'='*40}")
        passed_lse = check_result("LSE CHECK", lse_out.float(), lse_npu.float(),
                                  except_label="lse_golden", comp_label="lse_npu", verbose_diff=verbose_diff)
    elif lse_flag == 1:
        print(f"[{test_name}] 缺少 NPU 结果，跳过 LSE 对比。")

    def _to_stats(val):
        if isinstance(val, dict):
            return val
        if val:
            return {"passed": True, "max_abs": 0.0, "mean_abs": 0.0, "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
        return None

    attn_stats = _to_stats(passed_cpu_npu) or _to_stats(passed_cpu_gpu)
    if attn_stats is None:
        attn_stats = {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                      "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}
    if lse_flag != 1:
        lse_stats = {"passed": True, "max_abs": 0.0, "mean_abs": 0.0, "fail_cnt": 0, "total": 0, "fail_ratio": 0.0}
    else:
        lse_stats = _to_stats(passed_lse)
        if lse_stats is None:
            lse_stats = {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                         "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}
    return {"attn": attn_stats, "lse": lse_stats}


def _build_result_row(name, res, max_name_len):
    a = res["attn"]
    l = res["lse"]
    a_tag = "✓ PASS" if a["passed"] else "✗ FAIL"
    l_tag = "✓ PASS" if l["passed"] else "✗ FAIL"
    a_max = f"{a['max_abs']:.6f}" if a['total'] > 0 else "N/A"
    a_fr  = f"{a['fail_ratio']*100:.4f}%" if a['total'] > 0 else "N/A"
    l_max = f"{l['max_abs']:.6f}" if l['total'] > 0 else "---"
    l_fr  = f"{l['fail_ratio']*100:.4f}%" if l['total'] > 0 else "---"
    return (f"│  {name:<{max_name_len}}  "
            f"{a_tag:>8}  {a_max:>12}  {a_fr:>10}  │  "
            f"{l_tag:>8}  {l_max:>12}  {l_fr:>10}  │")


def _print_results_table(results, total_cases, is_final=False):
    """Print formatted results table. Returns (pass_cnt, fail_cnt)."""
    SEP = "─" * 120
    max_name_len = max((len(n) for n in results.keys()), default=28)
    if is_final:
        title = f"  汇总结果  ({len(results)} cases)"
    else:
        title = f"  中间统计结果  ({len(results)}/{total_cases} cases)"
    print(f"\n┌{SEP}┐")
    print(f"│{title}")
    print(f"├{SEP}┤")
    hdr = (f"│  {'Case':<{max_name_len}}  "
           f"{'Attn':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │  "
           f"{'LSE':>8}  {'MaxAbsErr':>12}  {'FailRatio':>10}  │")
    print(hdr)
    print(f"├{SEP}┤")
    pass_cnt = fail_cnt = 0
    for name, res in results.items():
        print(_build_result_row(name, res, max_name_len))
        both_pass = res["attn"]["passed"] and res["lse"]["passed"]
        if both_pass: pass_cnt += 1
        else:         fail_cnt += 1
    print(f"├{SEP}┤")
    if is_final:
        print(f"│  通过: {pass_cnt}   失败: {fail_cnt}   共: {len(results)}")
    else:
        print(f"│  已执行: {len(results)}/{total_cases}  通过: {pass_cnt}  失败: {fail_cnt}")
    print(f"└{SEP}┘")
    return pass_cnt, fail_cnt


# ─────────────────── main ───────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttn 精度测试（CPU vs GPU vs NPU）")
    parser.add_argument("--case_files", type=str, default=",".join(_DEFAULT_CASE_FILES),
                        help=f"逗号分隔的 test case 模块名（不含 .py）")
    parser.add_argument("--case_id",      type=str,            default="all", help="case名称，多个用逗号分隔")
    parser.add_argument("--device_id",    type=int,            default=0,    help="NPU device id")
    parser.add_argument("--use_gpu",      action="store_true", help="使用 GPU 进行计算")
    parser.add_argument("--gpu-device",   type=int,            default=0,    help="GPU 设备 ID")
    parser.add_argument("--dump_tensors", action="store_true", help="保存 q/k/v 及输出为 txt")
    parser.add_argument("--dump_dir",     type=str,            default="./dump_output")
    parser.add_argument("--verbose_diff", action="store_true", help="逐元素输出精度对比表")
    parser.add_argument("--visualize",    action="store_true", help="生成精度热力图")
    parser.add_argument("--viz_dir",      type=str,            default="./viz_output")
    parser.add_argument("--meta_only",    action="store_true", help="只调用 metadata 算子")
    parser.add_argument("--load_gpu_dump", type=str, default=None, help="GPU dump 文件路径")
    parser.add_argument("--load_npu_dump", type=str, default=None, help="NPU dump 文件路径")
    parser.add_argument("--compare_mode", action="store_true", help="三方对比模式")
    parser.add_argument("--graph_mode",  action="store_true", help="图模式调用 NPU 算子")
    parser.add_argument("--fail_analysis", action="store_true", help="失败元素分布分析")
    parser.add_argument("--report_interval", type=int, default=20, help="每 N 条用例输出一次中间统计")
    args = parser.parse_args()

    if args.use_gpu:
        if not torch.cuda.is_available():
            print("[ERROR] CUDA 不可用。"); sys.exit(1)
        torch.cuda.set_device(args.gpu_device)
        print(f"[INFO] GPU 模式，使用设备 cuda:{args.gpu_device}")
    elif args.compare_mode:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_device)
            print(f"[INFO] Compare 模式，使用 GPU 设备 cuda:{args.gpu_device}")
        elif args.load_gpu_dump and args.load_npu_dump:
            print(f"[INFO] Compare 模式，从 dump 文件对比（无 GPU/NPU）")
        else:
            try:
                import torch_npu
                torch.npu.set_device(args.device_id)
                print(f"[INFO] Compare 模式，使用 NPU 设备 npu:{args.device_id}")
            except ImportError:
                print("[ERROR] compare_mode 需要至少一种设备或同时指定 --load_gpu_dump 和 --load_npu_dump")
                sys.exit(1)
    else:
        try:
            import torch_npu
            torch.npu.set_device(args.device_id)
            print(f"[INFO] NPU 模式，使用设备 npu:{args.device_id}")
        except ImportError:
            print("[ERROR] torch_npu 未安装。"); sys.exit(1)

    case_files = args.case_files
    module_names = ["test_cases." + x.strip() for x in case_files.split(",")]
    all_cases = load_case_modules(module_names)
    run_cases = resolve_case_ids(args.case_id, all_cases)
    if not run_cases:
        print("[ERROR] 没有可运行的 case。"); sys.exit(1)

    results = {}
    _int_cnt = 0
    for name in run_cases:
        _int_cnt += 1
        if NPU_AVAILABLE:
            torch.npu.synchronize()

        config = all_cases[name]
        try:
            kwargs = normalize_case(config)
        except Exception as e:
            import traceback
            print(f"[ERROR] {name} normalize_case 异常: {e}")
            traceback.print_exc()
            results[name] = {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                                       "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')},
                              "lse":  {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                                       "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}}
            if args.report_interval > 0 and _int_cnt % args.report_interval == 0 and _int_cnt < len(run_cases):
                _print_results_table(results, len(run_cases))
            continue
        kwargs["_use_gpu"] = args.use_gpu

        print(f"\n{'='*66}")
        print(f"  Case: {name}  "
              f"B={kwargs.get('B')} N1={kwargs.get('N1')} N2={kwargs.get('N2')} "
              f"S1={kwargs.get('S1')} S2={kwargs.get('S2')} D={kwargs.get('D')} "
              f"layout={kwargs.get('input_layout')} layout_out={kwargs.get('layout_out')} dtype={kwargs.get('Dtype')}")
        print(f"{'='*66}")

        try:
            passed = call_flash_attn(
                name, dump_tensors=args.dump_tensors, dump_dir=args.dump_dir,
                verbose_diff=args.verbose_diff, visualize=args.visualize, viz_dir=args.viz_dir,
                meta_only=args.meta_only, compare_mode=args.compare_mode, graph_mode=args.graph_mode,
                load_gpu_dump=args.load_gpu_dump, load_npu_dump=args.load_npu_dump,
                fail_analysis=args.fail_analysis, **kwargs)
        except Exception as e:
            import traceback
            print(f"[ERROR] {name} 运行异常: {e}")
            traceback.print_exc()
            passed = {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                               "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')},
                      "lse":  {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                               "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}}
        results[name] = passed
        if args.report_interval > 0 and _int_cnt % args.report_interval == 0 and _int_cnt < len(run_cases):
            _print_results_table(results, len(run_cases))

    fail_cnt = _print_results_table(results, len(run_cases), is_final=True)[1]
    sys.exit(0 if fail_cnt == 0 else 1)
