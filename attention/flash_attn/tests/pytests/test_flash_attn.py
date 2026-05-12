import os
import sys
import argparse
import torch
import torch_npu
import math
import numpy as np
from cpu_impl import tforward
from test_utils import generate_qkv, generate_pse, generate_npu_mask, trans_bnsd_to_layout, get_seqlen_list
from npu_impl import flash_attn_npu, flash_attn_metadata_only
from test_case import TestCases
from test_case_fia_STC import TestCasesFIA


def save_tensor_to_txt(tensor, filepath):
    """将 tensor 展平后逐行保存为 txt，首行写入 shape 注释。"""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    arr = tensor.detach().cpu().float().numpy().flatten()
    shape_str = "x".join(str(s) for s in tensor.shape)
    with open(filepath, "w") as fh:
        fh.write(f"# shape: {shape_str}  total: {arr.size}\n")
        for v in arr:
            fh.write(f"{v:.8f}\n")


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


def check_result(test_name, expect, result, verbose_diff=False):
    SEP = "─" * 64
    print(f"\n┌{SEP}┐")
    print(f"│  精度报告: {test_name}")
    print(f"├{SEP}┤")
    if expect.shape != result.shape:
        print(f"│  [ERROR] shape不匹配: CPU={tuple(expect.shape)}  NPU={tuple(result.shape)}")
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
            # all_idxs = fail_mask.view(-1).nonzero(as_tuple=False).squeeze(1).tolist()
            all_idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).tolist()
            print(f"│  全部 {len(all_idxs)} 个超阈値元素 (relErr > {RATIO_THRESHOLD * 100:.2f}%):")
            print(f"│  {'idx':>10}  {'CPU':>14}  {'NPU':>14}  {'absErr':>12}  {'relErr':>12}")
            for i in all_idxs:
                # print(f"│  {i:>10}  {ef.view(-1)[i].item():>+14.8f}  {rf.view(-1)[i].item():>+14.8f}"
                #       f"  {diff.view(-1)[i].item():>12.8f}  {rel_err.view(-1)[i].item():>12.6f}")
                print(f"│  {i:>10}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>12.6f}")
        else:
            # idxs = fail_mask.view(-1).nonzero(as_tuple=False).squeeze(1)[:10].tolist()
            idxs = fail_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)[:10].tolist()
            print(f"│  前{len(idxs)}个不通过元素:")
            print(f"│  {'idx':>8}  {'CPU':>14}  {'NPU':>14}  {'absErr':>12}  {'relErr':>10}")
            for i in idxs:
                # print(f"│  {i:>8}  {ef.view(-1)[i].item():>+14.8f}  {rf.view(-1)[i].item():>+14.8f}"
                #       f"  {diff.view(-1)[i].item():>12.8f}  {rel_err.view(-1)[i].item():>10.6f}")
                print(f"│  {i:>8}  {ef.reshape(-1)[i].item():>+14.8f}  {rf.reshape(-1)[i].item():>+14.8f}"
                      f"  {diff.reshape(-1)[i].item():>12.8f}  {rel_err.reshape(-1)[i].item():>10.6f}")
    print(f"└{SEP}┘")
    return passed


def call_flash_attn(test_name, dump_tensors=False, dump_dir="./dump_output",
                    verbose_diff=False, visualize=False, viz_dir="./viz_output",
                    meta_only=False, **kwargs):
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
    # sq_gen/skv_gen: TND 下为全部 token 总数（CPU golden 按累积切片需要），其他 layout 同 sq/skv
    sq_gen = sq;  skv_gen = skv
    if input_layout == "TND":
        sl_q  = list(kwargs.get("seqlens_list_q",  []))
        sl_kv = list(kwargs.get("seqlens_list_kv", sl_q))
        q_arr  = get_seqlen_list(sl_q)
        kv_arr = get_seqlen_list(sl_kv)
        sq     = int(q_arr.max());   skv    = int(kv_arr.max())    # max_seqlen → NPU metadata
        sq_gen = int(q_arr.sum());   skv_gen = int(kv_arr.sum())   # total_tokens → 生成 tensor
        # npu_impl 通过 S1/S2 读取 max_seqlen，TND case 中本无此键，需显式写入
        kwargs["S1"] = sq
        kwargs["S2"] = skv
        pse_b = len(q_arr);  pse_s1 = sq;  pse_s2 = skv
        if pse_layout in ["bnhs", "1nhs"]: pse_s1 = max(1024, pse_s1)
    pse_cpu, pse_npu = generate_pse(pse_b, n1, pse_s1, pse_s2, pse_type, pse_layout,
                                    pttype, q_start_idx, kv_start_idx)
    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(b, n1, n2, sq_gen, skv_gen, d, d_v, d_rope,
                                                    input_layout, input_dtype,
                                                    q_range=q_range, k_range=k_range, v_range=v_range)
    if meta_only:
        print(f"[{test_name}] --meta-only: 跳过 CPU golden，仅调用 npu_flash_attn_metadata")
        flash_attn_metadata_only(**kwargs)
        return True
    print(f"[{test_name}] CPU 参考计算...")
    out, _, _ = tforward(qf, kf, v, pse_cpu, **kwargs)
    atten_mask = generate_npu_mask(b, sq, skv, sparse_mode, pre_tokens, next_tokens, prefix)
    print(f"[{test_name}] NPU 算子执行...")
    npu_out = flash_attn_npu(q, k, v, q_rope, k_rope, atten_mask, pse_npu, **kwargs)
    out_trans = trans_bnsd_to_layout(out, output_layout)
    if dump_tensors:
        dump_path = os.path.join(dump_dir, test_name)
        os.makedirs(dump_path, exist_ok=True)
        save_tensor_to_txt(q,                 os.path.join(dump_path, "q.txt"))
        save_tensor_to_txt(k,                 os.path.join(dump_path, "k.txt"))
        save_tensor_to_txt(v,                 os.path.join(dump_path, "v.txt"))
        save_tensor_to_txt(out_trans.float(), os.path.join(dump_path, "cpu_out.txt"))
        save_tensor_to_txt(npu_out.float(),   os.path.join(dump_path, "npu_out.txt"))
        print(f"[{test_name}] 已保存 q/k/v/cpu_out/npu_out → {dump_path}/")
    passed = check_result(test_name, out_trans.float(), npu_out.float(), verbose_diff=verbose_diff)
    if visualize:
        try:
            from precision_visual import visualize_from_tensors
            visualize_from_tensors(out_trans.float(), npu_out.float(),
                                   case_name=test_name, out_dir=viz_dir)
        except ImportError:
            print("[WARN] precision_visual 导入失败，请确认 matplotlib 已安装")
        except Exception as exc:
            print(f"[WARN] 可视化异常: {exc}")
    return passed

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
            # 合并时，如果冲突，优先保留 TestCases 中的配置（你可以调整优先级）
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashAttn 精度测试（从 test_case.py 读取用例）")
    parser.add_argument("--case", type=str, default="all",
                        choices=["all", "base", "fia", "TestCases", "TestCasesFIA"],
                        help="选择测试集: all(两个都包含), base(仅TestCases), fia(仅TestCasesFIA) (default: all)")
    parser.add_argument("--case_id",      type=str,            default="all",
                        help="case名称，多个用逗号分隔，'all'表示全部 (default: all)")
    parser.add_argument("--device_id",    type=int,            default=0,
                        help="NPU device id (default: 0)")
    parser.add_argument("--dump_tensors", action="store_true",
                        help="将 q/k/v 及 cpu/npu 输出保存为 txt 文件")
    parser.add_argument("--dump_dir",     type=str,            default="./dump_output",
                        help="txt 文件保存根目录 (default: ./dump_output)")
    parser.add_argument("--verbose_diff", action="store_true",
                        help="逐元素输出全部超阈值精度对比表")
    parser.add_argument("--visualize",    action="store_true",
                        help="生成 CPU vs NPU 精度热力图（依赖 precision_visual.py + matplotlib）")
    parser.add_argument("--viz_dir",      type=str,            default="./viz_output",
                        help="热力图保存目录 (default: ./viz_output)")
    parser.add_argument("--meta-only",    action="store_true",
                        help="只调用 npu_flash_attn_metadata，跳过 CPU golden 和 npu_flash_attn")
    args = parser.parse_args()

    torch.npu.set_device(args.device_id)

    case_keys_pool, conflicts = build_case_keys_pool(args.case)
    for w in conflicts:
        print(f"[WARN] {w}")

    if args.case_id == "all":
        run_cases = sorted(case_keys_pool)   # 排序使执行顺序固定
    else:
        ids = [x.strip() for x in args.case_id.split(",")]
        missing = [x for x in ids if x not in case_keys_pool]
        if missing:
            print(f"[WARN] 以下 case 不存在于所选测试集中: {missing}")
        run_cases = [x for x in ids if x in case_keys_pool]

    if not run_cases:
        print("[ERROR] 没有可运行的 case，退出。")
        sys.exit(1)

    results = {}
    for name in run_cases:
        config = get_case_config(name)
        kwargs = normalize_case(config)
        print(f"\n{'='*66}")
        print(f"  Case: {name}  "
              f"B={kwargs.get('B')} N1={kwargs.get('N1')} N2={kwargs.get('N2')} "
              f"S1={kwargs.get('S1')} S2={kwargs.get('S2')} D={kwargs.get('D')} "
              f"layout={kwargs.get('input_layout')} layout_out={kwargs.get('layout_out')} dtype={kwargs.get('Dtype')}")
        
        print(f"{'='*66}")
        try:
            passed = call_flash_attn(name, dump_tensors=args.dump_tensors,
                                     dump_dir=args.dump_dir, verbose_diff=args.verbose_diff,
                                     visualize=args.visualize, viz_dir=args.viz_dir,
                                     meta_only=args.meta_only, **kwargs)
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