import os
import sys
import argparse
import torch
import math
import numpy as np
from cpu_impl import tforward
from test_utils import (generate_qkv, generate_pse, generate_npu_mask, trans_bnsd_to_layout, get_seqlen_list,
                         data_compare_benchmark_new, Result)
from test_case import TestCases
from test_case_fia_STC import TestCasesFIA

# ------------------ 动态导入 GPU / NPU 实现 ------------------
try:
    from gpu_impl import flash_attn_gpu
    GPU_AVAILABLE = True
except ImportError:
    flash_attn_gpu = None
    GPU_AVAILABLE = False

try:
    from npu_impl import flash_attn_npu, flash_attn_metadata_only
    NPU_AVAILABLE = True
except ImportError:
    flash_attn_npu = None
    flash_attn_metadata_only = None
    NPU_AVAILABLE = False

# ------------------ 辅助函数 ------------------
def save_tensor_to_txt(tensor, filepath):
    """将 tensor 展平后逐行保存为 txt，首行写入 shape 注释。"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    arr = tensor.detach().cpu().float().numpy().flatten()
    shape_str = "x".join(str(s) for s in tensor.shape)
    with open(filepath, "w") as fh:
        fh.write(f"# shape: {shape_str}  total: {arr.size}\n")
        for v in arr:
            fh.write(f"{v:.8f}\n")

def load_tensor_from_txt(filepath, target_dtype=torch.float32, target_device='cpu'):
    """从 save_tensor_to_txt 保存的文件恢复为 Tensor。"""
    with open(filepath, "r") as fh:
        header = fh.readline().strip()
        # 解析 shape 注释: # shape: 16x104x224x128  total: xxx
        if not header.startswith("# shape:"):
            raise ValueError(f"文件 {filepath} 格式不正确，缺少 shape 注释。")
        shape_str = header.split()[2]
        shape = tuple(int(s) for s in shape_str.split('x'))
        arr = np.loadtxt(fh, dtype=np.float32)
    expected = 1
    for s in shape:
        expected *= s
    if arr.size != expected:
        raise ValueError(f"形状 {shape} 与数值数 {arr.size} 不匹配。")
    tensor = torch.from_numpy(arr).reshape(shape).to(dtype=target_dtype, device=target_device)
    return tensor

# ------------------ 精度报告阈值 ------------------
RATIO_THRESHOLD = 0.005   # 单元素相对误差阈值
FAIL_RATIO_LIMIT = 0.005  # 超阈值元素占比上限


def normalize_case(raw):
    """将 test_case.py 字段转换为 call_flash_attn 所需 kwargs 格式。"""
    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    c["input_layout"] = layout_q
    c.setdefault("layout_kv",   c.get("layout_kv",  layout_q))
    c.setdefault("layout_out",  c.get("layout_out", layout_q))
    if "mask_mode" in c and "sparse_mode" not in c:
        c["sparse_mode"] = c.pop("mask_mode")
    c.setdefault("N2",          c.get("N1"))
    c.setdefault("S2",          c.get("S1"))
    c.setdefault("DV",          c.get("D"))
    c.setdefault("DRope",       0)
    c.setdefault("pse_type",    0)
    c.setdefault("pse_layout",  "none")
    c.setdefault("q_start_idx", 0)
    c.setdefault("kv_start_idx",0)
    c.setdefault("keep_prob",   1.0)
    c.setdefault("seed",        0)
    c.setdefault("offset",      0)
    c.setdefault("pre_tokens",  2147483647)
    c.setdefault("next_tokens", 2147483647)
    c.setdefault("prefix",      [])
    c.setdefault("q_range",     None)
    c.setdefault("k_range",     None)
    c.setdefault("v_range",     None)
    if layout_q == "TND":
        if isinstance(c.get("cu_seqlens_q"), list):
            # New 4-param style: cu_seqlens_q=(B+1,) cumulative with 0, seqused_q=(B,) individual actual
            cum_q_full  = list(c["cu_seqlens_q"])
            cum_kv_full = list(c.get("cu_seqlens_kv", cum_q_full))
            def _diffs(lst): return [lst[i+1] - lst[i] for i in range(len(lst) - 1)]
            ind_q  = list(c.get("seqused_q",  _diffs(cum_q_full)))
            ind_kv = list(c.get("seqused_kv", _diffs(cum_kv_full)))
            # cpu_impl slices by cu_seqlens boundaries (without leading 0)
            c["actual_seq_qlen"]  = cum_q_full[1:]    # [0, 128, 256, 512]
            c["actual_seq_kvlen"] = cum_kv_full[1:]
            c["seqlens_list_q"]   = cum_q_full[1:]    # [128, 128, 256]
            c["seqlens_list_kv"]  = cum_kv_full[1:]
            # NPU: pass full (B+1,) list as cuSeqlens, (B,) individual as seqused
            c["_npu_cu_q"]       = cum_q_full
            c["_npu_cu_kv"]      = cum_kv_full
            c["_npu_seqused_q"]  = ind_q
            c["_npu_seqused_kv"] = ind_kv
            # TND tensors are always [1, N, T, D]; request count is encoded in cu_seqlens lists
            c["B"] = 1
        else:
            # Legacy style: cu_seqused_q/kv as cumulative list (with or without leading 0)
            cu_q  = list(c.get("cu_seqused_q", []))
            cu_kv = list(c.get("cu_seqused_kv", cu_q))
            if cu_q  and cu_q[0]  == 0: cu_q  = cu_q[1:]
            if cu_kv and cu_kv[0] == 0: cu_kv = cu_kv[1:]
            c["seqlens_list_q"]  = cu_q
            c["seqlens_list_kv"] = cu_kv
            c["actual_seq_qlen"]  = cu_q
            c["actual_seq_kvlen"] = cu_kv
            c["B"] = 1
    return c

def check_result(test_name, expect, result, except_label = "CPU", comp_label="NPU", verbose_diff=False):
    """比较两个 tensor，输出精度报告。
    comp_label 用于标识对比的另一方（如 'GPU'/'NPU'）。
    """
    SEP = "─" * 64
    print(f"\n┌{SEP}┐")
    print(f"│  精度报告: {test_name}  ({except_label} vs {comp_label})")
    print(f"├{SEP}┤")
    if expect.shape != result.shape:
        print(f"│  [ERROR] shape不匹配: expect={tuple(expect.shape)}  {comp_label}={tuple(result.shape)}")
        print(f"└{SEP}┘")
        return False
    ef   = expect.float()
    rf   = result.float()
    diff = torch.abs(ef - rf)
    ref_abs = torch.abs(ef)
    rel_err = diff / (ref_abs + 1e-6)
    max_abs   = diff.max().item()
    mean_abs  = diff.mean().item()
    max_rel   = rel_err.max().item()
    mean_rel  = rel_err.mean().item()
    threshold = torch.max(ref_abs.mul(RATIO_THRESHOLD), torch.full_like(diff, 0.000025))
    fail_mask = diff > threshold
    fail_cnt  = int(fail_mask.sum().item())
    total     = expect.numel()
    fail_ratio = fail_cnt / total
    passed    = fail_ratio <= FAIL_RATIO_LIMIT
    print(f"│  Shape       : {tuple(expect.shape)}")
    print(f"│  MaxAbsErr   : {max_abs:.8f}")
    print(f"│  MeanAbsErr  : {mean_abs:.8f}")
    print(f"│  MaxRelErr   : {max_rel:.8f}")
    print(f"│  MeanRelErr  : {mean_rel:.8f}")
    print(f"│  FailElems   : {fail_cnt} / {total}  ({fail_ratio*100:.4f}%)")
    print(f"│  Threshold   : elemRelErr≤{RATIO_THRESHOLD*100:.2f}%  failRatio≤{FAIL_RATIO_LIMIT*100:.2f}%")
    print(f"│  结论        : {'✓ PASS' if passed else '✗ FAIL'}")
    if fail_cnt > 0:
        print(f"├{SEP}┤")
        if verbose_diff:
            all_idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).tolist()
            print(f"│  全部 {len(all_idxs)} 个超阈値元素 (relErr > {RATIO_THRESHOLD * 100:.2f}%):")
            print(f"│  {'idx':>10}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>12}")
            for i in all_idxs:
                print(f"│  {i:>10}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>12.6f}")
        else:
            idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)[:10].tolist()
            print(f"│  前{len(idxs)}个不通过元素:")
            print(f"│  {'idx':>8}  {except_label:>14}  {comp_label:>14}  {'absErr':>12}  {'relErr':>10}")
            for i in idxs:
                print(f"│  {i:>8}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>10.6f}")
    print(f"└{SEP}┘")
    return passed



# ------------------ 三方精度对比 ------------------
def call_flash_attn(test_name, dump_tensors=False, dump_dir="./dump_output",
                    verbose_diff=False, visualize=False, viz_dir="./viz_output",
                    meta_only=False, compare_mode=False,
                    load_gpu_dump=None, load_npu_dump=None,
                    **kwargs):
    # ... 参数提取（与原一致）...
    b          = kwargs.get("B", 1)
    n1         = kwargs.get("N1")
    n2         = kwargs.get("N2", n1)
    sq         = kwargs.get("S1", -1)
    skv        = kwargs.get("S2", sq)
    d          = kwargs.get("D")
    d_v        = kwargs.get("DV", d)
    d_rope     = kwargs.get("DRope", 0)
    input_layout = kwargs.get("input_layout")
    output_layout = kwargs.get('layout_out')
    pse_type   = int(kwargs.get("pse_type") if kwargs.get("pse_type") != '' else 0)
    pse_layout = kwargs.get("pse_layout", "none").lower()
    q_start_idx  = kwargs.get("q_start_idx", 0)
    kv_start_idx = kwargs.get("kv_start_idx", 0)
    dtype = kwargs.get("Dtype", "bf16")
    pttype = torch.float16 if dtype == "fp16" else torch.bfloat16
    input_dtype = pttype
    sparse_mode = kwargs.get("sparse_mode", None)
    pre_tokens  = kwargs.get("pre_tokens",  2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    prefix      = kwargs.get("prefix", [])
    q_range     = kwargs.get("q_range", None)
    k_range     = kwargs.get("k_range", None)
    v_range     = kwargs.get("v_range", None)
    pse_b = b;  pse_s1 = sq;  pse_s2 = skv
    sq_gen = sq;  skv_gen = skv
    if input_layout == "TND":
        sl_q  = list(kwargs.get("seqlens_list_q",  []))
        sl_kv = list(kwargs.get("seqlens_list_kv", sl_q))
        q_arr  = get_seqlen_list(sl_q)
        kv_arr = get_seqlen_list(sl_kv)
        sq     = int(q_arr.max());   skv    = int(kv_arr.max())
        sq_gen = int(q_arr.sum());   skv_gen = int(kv_arr.sum())
        kwargs["S1"] = sq
        kwargs["S2"] = skv
        pse_b = len(q_arr);  pse_s1 = sq;  pse_s2 = skv
        if pse_layout in ["bnhs", "1nhs"]: pse_s1 = max(1024, pse_s1)
    pse_cpu, pse_npu = generate_pse(pse_b, n1, pse_s1, pse_s2, pse_type, pse_layout,
                                    pttype, q_start_idx, kv_start_idx)
    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(b, n1, n2, sq_gen, skv_gen, d, d_v, d_rope,
                                                    input_layout, input_dtype,
                                                    q_range=q_range, k_range=k_range, v_range=v_range)

    # ------------------ 1. CPU Golden 计算 (总是执行) ------------------
    print(f"[{test_name}] CPU 参考计算...")
    out_cpu, _, _ = tforward(qf, kf, v, pse_cpu, **kwargs)
    out_cpu_out_layout = trans_bnsd_to_layout(out_cpu, output_layout)  # CPU 标准结果，用于对比
    if dump_tensors:
        dump_path = os.path.join(dump_dir, test_name)
        os.makedirs(dump_path, exist_ok=True)
        save_tensor_to_txt(q, os.path.join(dump_path, "q.txt"))
        save_tensor_to_txt(k, os.path.join(dump_path, "k.txt"))
        save_tensor_to_txt(v, os.path.join(dump_path, "v.txt"))
        save_tensor_to_txt(out_cpu_out_layout.float(), os.path.join(dump_path, "cpu_out.txt"))

    # ------------------ 2. 准备 GPU / NPU 结果张量 ------------------
    gpu_out = None
    npu_out = None

    # ---- GPU 计算或加载 dump ----
    if kwargs.get("_use_gpu", False):
        if not GPU_AVAILABLE:
            print(f"[{test_name}] 警告: GPU 不可用，但要求 GPU 模式，尝试从 dump 加载。")
        if GPU_AVAILABLE:
            print(f"[{test_name}] GPU 计算...")
            atten_mask = generate_npu_mask(b, sq, skv, sparse_mode, pre_tokens, next_tokens, prefix)
            device = torch.cuda.current_device()
            q_gpu = q.cuda(device)
            k_gpu = k.cuda(device)
            v_gpu = v.cuda(device)
            q_rope_gpu = q_rope.cuda(device) if q_rope is not None else None
            k_rope_gpu = k_rope.cuda(device) if k_rope is not None else None
            mask_gpu = atten_mask.cuda(device) if atten_mask is not None else None
            pse_gpu = pse_npu.cuda(device) if pse_npu is not None else None
            gpu_out_bnsd = flash_attn_gpu(q_gpu, k_gpu, v_gpu, q_rope_gpu, k_rope_gpu, mask_gpu, pse_gpu, **kwargs)
            gpu_out = trans_bnsd_to_layout(gpu_out_bnsd, output_layout).cpu()
            if dump_tensors:
                dump_path = os.path.join(dump_dir, test_name)
                os.makedirs(dump_path, exist_ok=True)
                # save_tensor_to_txt(q.cpu(), os.path.join(dump_path, "q.txt"))
                # save_tensor_to_txt(k.cpu(), os.path.join(dump_path, "k.txt"))
                # save_tensor_to_txt(v.cpu(), os.path.join(dump_path, "v.txt"))
                save_tensor_to_txt(gpu_out.float(), os.path.join(dump_path, "gpu_out.txt"))
                print(f"[{test_name}] 已保存 q/k/v/gpu_out → {dump_path}/")
        else:
            print(f"[{test_name}] GPU 不可用，跳过计算。")
    # 尝试从 dump 加载 GPU 结果（如果未计算）
    if gpu_out is None or load_gpu_dump:
        gpu_dump_path = load_gpu_dump if load_gpu_dump else os.path.join(dump_dir, test_name, "gpu_out.txt")
        if os.path.exists(gpu_dump_path):
            print(f"[{test_name}] 从 dump 加载 GPU 结果: {gpu_dump_path}")
            gpu_out = load_tensor_from_txt(gpu_dump_path, target_dtype=torch.float32, target_device='cpu')
        else:
            print(f"[{test_name}] 未找到 GPU dump 文件: {gpu_dump_path}")

    # ---- NPU 计算或加载 dump ----
    if not kwargs.get("_use_gpu", False):   # NPU 模式
        if not NPU_AVAILABLE:
            print(f"[{test_name}] 警告: NPU 不可用，尝试从 dump 加载。")
        if NPU_AVAILABLE:
            print(f"[{test_name}] NPU 计算...")
            atten_mask = generate_npu_mask(b, sq, skv, sparse_mode, pre_tokens, next_tokens, prefix)
            npu_out = flash_attn_npu(q, k, v, q_rope, k_rope, atten_mask, pse_npu, **kwargs)
            # flash_attn_npu 内部已将结果移至 CPU，这里 npu_out 即最终布局的输出
            if dump_tensors:
                dump_path = os.path.join(dump_dir, test_name)
                os.makedirs(dump_path, exist_ok=True)
                # save_tensor_to_txt(q,                 os.path.join(dump_path, "q.txt"))
                # save_tensor_to_txt(k,                 os.path.join(dump_path, "k.txt"))
                # save_tensor_to_txt(v,                 os.path.join(dump_path, "v.txt"))
                save_tensor_to_txt(npu_out.float(),   os.path.join(dump_path, "npu_out.txt"))
                print(f"[{test_name}] 已保存 q/k/v/npu_out → {dump_path}/")
        else:
            print(f"[{test_name}] NPU 不可用，跳过计算。")
    # 尝试从 dump 加载 NPU 结果（如果未计算）
    if npu_out is None or load_npu_dump:
        npu_dump_path = load_npu_dump if load_npu_dump else os.path.join(dump_dir, test_name, "npu_out.txt")
        if os.path.exists(npu_dump_path):
            print(f"[{test_name}] 从 dump 加载 NPU 结果: {npu_dump_path}")
            npu_out = load_tensor_from_txt(npu_dump_path, target_dtype=torch.float32, target_device='cpu')
        else:
            print(f"[{test_name}] 未找到 NPU dump 文件: {npu_dump_path}")

    # ------------------ 3. 精度对比 ------------------
    passed_cpu_gpu = False
    passed_cpu_npu = False
    passed_gpu_npu = False

    # 三方对比模式（使用详细精度统计）
    if compare_mode:
        if gpu_out is not None and npu_out is not None:
            print(f"\n{'='*40} 三方对比模式（详细精度统计） {'='*40}")
            
            dtype_str = kwargs.get("Dtype", "fp16")
            dtype_map = {"fp16": "fp16", "bf16": "bf16", "fp32": "fp32"}
            out_dtype_key = dtype_map.get(dtype_str, dtype_str)
            
            params = {
                "op_name": "flash_attn",
                "case_name": test_name,
                "dtype_output": [dtype_str],
                "dtype_input": [dtype_str],
                "red_range": {
                    "fp32": "0.000001/0.00001/0.0001/0.0005",
                    "fp16": "0.001/0.002/0.005/0.01",
                    "bf16": "0.001/0.002/0.005/0.01",
                },
                "bm_cmp_std": {
                    "fp32": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 1e-06,
                        "small_value_atol": 0.0
                    },
                    "fp16": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 0.001,
                        "small_value_atol": 0.001
                    },
                    "bf16": {
                        "max_re_rtol": 10.0,
                        "avg_re_rtol": 2.0,
                        "rmse_rtol": 2.0,
                        "small_value": 1e-07,
                        "small_value_atol": 0.004
                    }
                },
            }
            
            npu_np = npu_out.float().numpy()
            gpu_np = gpu_out.float().numpy()
            cpu_np = out_cpu_out_layout.float().numpy()
            
            str1, str2, data = data_compare_benchmark_new(params, npu_np, gpu_np, cpu_np, out_dtype_key, i=0)
            
            print("\n====== 三方对比结果 ======")
            print(f"状态: {str1}, 原因: {str2}")
            passed_cpu_gpu = (str1 == 'Pass')
            passed_cpu_npu = (str1 == 'Pass')
            passed_gpu_npu = (str1 == 'Pass')
        else:
            print(f"[{test_name}] 缺少 GPU 或 NPU 结果，无法执行三方对比，使用标准对比模式。")
            compare_mode = False
    
    # 标准对比模式
    if not compare_mode:
        # CPU vs GPU
        if gpu_out is not None:
            print(f"\n{'='*40} 对比 CPU vs GPU {'='*40}")
            passed_cpu_gpu = check_result(test_name + "_CPU_vs_GPU", out_cpu_out_layout.float(), gpu_out.float(), except_label="CPU", comp_label="GPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 GPU 结果，跳过 CPU vs GPU 对比。")

        # CPU vs NPU
        if npu_out is not None:
            print(f"\n{'='*40} 对比 CPU vs NPU {'='*40}")
            passed_cpu_npu = check_result(test_name + "_CPU_vs_NPU", out_cpu_out_layout.float(), npu_out.float(), except_label="CPU", comp_label="NPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 NPU 结果，跳过 CPU vs NPU 对比。")

        # GPU vs NPU
        if gpu_out is not None and npu_out is not None:
            print(f"\n{'='*40} 对比 GPU vs NPU {'='*40}")
            passed_gpu_npu = check_result(test_name + "_GPU_vs_NPU", gpu_out.float(), npu_out.float(), except_label="GPU", comp_label="NPU", verbose_diff=verbose_diff)
        else:
            print(f"[{test_name}] 缺少 GPU 或 NPU 结果，跳过 GPU vs NPU 对比。")

    # 可视化（如果指定）
    if visualize:
        try:
            from precision_visual import visualize_from_tensors
            if npu_out is not None:
                visualize_from_tensors(out_cpu_out_layout.float(), npu_out.float(), case_name=test_name, out_dir=viz_dir)
            if gpu_out is not None:
                visualize_from_tensors(out_cpu_out_layout.float(), gpu_out.float(), case_name=test_name + "_gpu", out_dir=viz_dir)
        except ImportError:
            print("[WARN] precision_visual 导入失败，请确认 matplotlib 已安装")
        except Exception as exc:
            print(f"[WARN] 可视化异常: {exc}")

    # 返回总体结果：至少一方对比通过即可（可根据需求调整逻辑）
    at_least_one_pass = passed_cpu_gpu or passed_cpu_npu or passed_gpu_npu
    return at_least_one_pass

# ------------------ 其它辅助函数（build_case_keys_pool 等）与原一致 ------------------
def build_case_keys_pool(case_choice):
    """返回 (keys_pool, conflict_warning_list)"""
    if case_choice in ("base", "TestCases"):
        return set(TestCases.keys()), []
    elif case_choice in ("fia", "TestCasesFIA"):
        return set(TestCasesFIA.keys()), []
    elif case_choice == "all":
        base_keys = set(TestCases.keys())
        fia_keys = set(TestCasesFIA.keys())
        conflict = base_keys & fia_keys
        if conflict:
            warn = f"警告: 以下 case 名称同时在 TestCases 和 TestCasesFIA 中存在，将只采用 TestCases 中的定义: {conflict}"
            keys_pool = base_keys | fia_keys
            return keys_pool, [warn]
        else:
            return base_keys | fia_keys, []
    else:
        raise ValueError(f"未知的 --case 值: {case_choice}")

def get_case_config(case_name):
    if case_name in TestCases:
        return TestCases[case_name]
    elif case_name in TestCasesFIA:
        return TestCasesFIA[case_name]
    else:
        raise KeyError(f"找不到 case '{case_name}' 的配置")

# ------------------ main ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttn 精度测试（CPU vs GPU vs NPU）")
    parser.add_argument("--case", type=str, default="all",
                        choices=["all", "base", "fia", "TestCases", "TestCasesFIA"],
                        help="选择测试集 (default: all)")
    parser.add_argument("--case_id",      type=str,            default="all",
                        help="case名称，多个用逗号分隔，'all'表示全部 (default: all)")
    parser.add_argument("--device_id",    type=int,            default=0,
                        help="NPU device id (default: 0)")
    parser.add_argument("--use_gpu",      action="store_true", help="使用 GPU 进行计算，不执行 NPU")
    parser.add_argument("--gpu-device",   type=int,            default=0, help="GPU 设备 ID (默认 0)")
    parser.add_argument("--dump_tensors", action="store_true",
                        help="保存 q/k/v 及算子输出为 txt 文件")
    parser.add_argument("--dump_dir",     type=str,            default="./dump_output",
                        help="txt 文件保存根目录 (default: ./dump_output)")
    parser.add_argument("--verbose_diff", action="store_true",
                        help="逐元素输出全部超阈值精度对比表")
    parser.add_argument("--visualize",    action="store_true",
                        help="生成 CPU vs NPU 精度热力图")
    parser.add_argument("--viz_dir",      type=str,            default="./viz_output")
    parser.add_argument("--meta_only",    action="store_true",
                        help="只调用 npu_flash_attn_metadata (NPU环境)")
    parser.add_argument("--load_gpu_dump", type=str, default=None,
                        help="指定 GPU dump 文件路径（用于离线对比）")
    parser.add_argument("--load_npu_dump", type=str, default=None,
                        help="指定 NPU dump 文件路径（用于离线对比）")
    parser.add_argument("--compare_mode", action="store_true",
                        help="启用三方对比模式（CPU vs GPU vs NPU），使用详细精度统计")
    args = parser.parse_args()

    # 设备初始化
    if args.use_gpu:
        if not torch.cuda.is_available():
            print("[ERROR] CUDA 不可用，无法使用 GPU 模式。")
            sys.exit(1)
        torch.cuda.set_device(args.gpu_device)
        print(f"[INFO] GPU 模式，使用设备 cuda:{args.gpu_device}")
    elif args.compare_mode:
        # compare_mode 下允许纯 CPU 运行（从 dump 文件对比），或 GPU 运行
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_device)
            print(f"[INFO] Compare 模式，使用 GPU 设备 cuda:{args.gpu_device}")
        elif args.load_gpu_dump and args.load_npu_dump:
            print(f"[INFO] Compare 模式，从 dump 文件对比（无 GPU/NPU）")
        else:
            # 尝试使用 NPU
            try:
                import torch_npu
                torch.npu.set_device(args.device_id)
                print(f"[INFO] Compare 模式，使用 NPU 设备 npu:{args.device_id}")
            except ImportError:
                print("[ERROR] compare_mode 需要至少一种设备（GPU/NPU）或同时指定 --load_gpu_dump 和 --load_npu_dump")
                sys.exit(1)
    else:
        # 默认 NPU 模式
        try:
            import torch_npu
            torch.npu.set_device(args.device_id)
            print(f"[INFO] NPU 模式，使用设备 npu:{args.device_id}")
        except ImportError:
            print("[ERROR] torch_npu 未安装，无法使用 NPU 模式。")
            print("[提示] 使用 --use_gpu 切换到 GPU 模式，或使用 --compare_mode --use_gpu 进行三方对比")
            sys.exit(1)

    case_keys_pool, conflicts = build_case_keys_pool(args.case)
    for w in conflicts:
        print(f"[WARN] {w}")

    if args.case_id == "all":
        run_cases = sorted(case_keys_pool)
    else:
        ids = [x.strip() for x in args.case_id.split(",")]
        missing = [x for x in ids if x not in case_keys_pool]
        if missing:
            print(f"[WARN] 以下 case 不存在于所选测试集中: {missing}")
        run_cases = [x for x in ids if x in case_keys_pool]

    if not run_cases:
        print("[ERROR] 没有可运行的 case，退出。")
        sys.exit(1)

    # 根据模式决定是否传递 _use_gpu 及加载 dump 行为
    # 自动推断是否需要从 dump 加载另一方结果
    results = {}
    for name in run_cases:
        config = get_case_config(name)
        kwargs = normalize_case(config)
        if args.use_gpu:
            kwargs["_use_gpu"] = True
            # 如果 NPU 不可用，自动尝试加载 npu dump（仅在 dump 功能开启或显式指定时加载）
            auto_load_npu = args.load_npu_dump or (args.dump_tensors and not NPU_AVAILABLE)
        else:
            kwargs["_use_gpu"] = False
            # 如果 GPU 不可用，自动尝试加载 gpu dump
            auto_load_gpu = args.load_gpu_dump or (args.dump_tensors and not GPU_AVAILABLE)

        print(f"\n{'='*66}")
        print(f"  Case: {name}  "
              f"B={kwargs.get('B')} N1={kwargs.get('N1')} N2={kwargs.get('N2')} "
              f"S1={kwargs.get('S1')} S2={kwargs.get('S2')} D={kwargs.get('D')} "
              f"layout={kwargs.get('input_layout')} layout_out={kwargs.get('layout_out')} dtype={kwargs.get('Dtype')}")
        print(f"{'='*66}")

        try:
            passed = call_flash_attn(
                name,
                dump_tensors=args.dump_tensors,
                dump_dir=args.dump_dir,
                verbose_diff=args.verbose_diff,
                visualize=args.visualize,
                viz_dir=args.viz_dir,
                meta_only=args.meta_only,
                compare_mode=args.compare_mode,
                load_gpu_dump=args.load_gpu_dump,
                load_npu_dump=args.load_npu_dump,
                **kwargs
            )
        except Exception as e:
            import traceback
            print(f"[ERROR] {name} 运行异常: {e}")
            traceback.print_exc()
            passed = False
        results[name] = passed

    # 汇总表格
    SEP = "─" * 50
    print(f"\n┌{SEP}┐")
    print(f"│  汇总结果  ({len(run_cases)} cases)")
    print(f"├{SEP}┤")
    print(f"│  {'Case':<28}  {'Result':>8}  │")
    print(f"├{SEP}┤")
    pass_cnt = fail_cnt = 0
    for name, ok in results.items():
        tag = "✓ PASS" if ok else "✗ FAIL"
        print(f"│  {name:<28}  {tag:>8}  │")
        if ok: pass_cnt += 1
        else:  fail_cnt += 1
    print(f"├{SEP}┤")
    print(f"│  通过: {pass_cnt}   失败: {fail_cnt}   共: {len(run_cases)}")
    print(f"└{SEP}┘")
    sys.exit(0 if fail_cnt == 0 else 1)