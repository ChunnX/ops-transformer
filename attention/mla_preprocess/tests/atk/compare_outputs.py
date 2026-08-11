#!/usr/bin/env python3
import os
import sys
import glob
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "compare_result.log")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def find_latest_output(base):
    dirs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith("db")],
                  key=os.path.getmtime)
    return dirs[-1] if dirs else None

def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)

def flatten_tensors(data):
    if isinstance(data, (list, tuple)):
        ts = [x for x in data if isinstance(x, torch.Tensor)]
        return ts
    if isinstance(data, torch.Tensor):
        return [data]
    return []

def print_stats(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype}")
    print(f"    min={t.float().min().item():.6e}  max={t.float().max().item():.6e}  "
          f"mean={t.float().mean().item():.6e}")

def compare_tensor(local, remote, label):
    print(f"\n{'='*60}")
    print(f"Output: {label}")
    print_stats("golden (cpu)", remote)
    print_stats("operator(npu)", local)

    if local.shape != remote.shape:
        print(f"  [SHAPE MISMATCH] local={tuple(local.shape)} remote={tuple(remote.shape)}")
        return

    l = local.float()
    r = remote.float()
    diff = (l - r).abs()
    eps = 1e-7
    rel_err = diff / (r.abs() + eps)
    mean_rel = rel_err.mean().item()
    max_rel = rel_err.max().item()
    max_abs = diff.max().item()
    print(f"  max_abs_diff={max_abs:.6e}  mean_rel_err={mean_rel:.6e}  max_rel_err={max_rel:.6e}")

    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    print(f"  相对误差分布:")
    total = rel_err.numel()
    for thd in thresholds:
        cnt = (rel_err <= thd).sum().item()
        pct = cnt / total * 100
        print(f"    rel_err <= {thd:.0e}: {cnt}/{total} ({pct:.2f}%)")

    pass_thd = 0.005
    pass_mare_ratio = 50
    mere_pass = (rel_err < pass_thd).sum().item()
    mare_pass = (rel_err < pass_thd * pass_mare_ratio).sum().item()
    print(f"  single_bm 判定 (fp16_thd/bf16_thd={pass_thd}, mare_ratio={pass_mare_ratio}):")
    print(f"    MERE (rel_err < {pass_thd}): {mere_pass}/{total} ({mere_pass/total*100:.2f}%)")
    print(f"    MARE (rel_err < {pass_thd * pass_mare_ratio}): {mare_pass}/{total} ({mare_pass/total*100:.2f}%)")

    n = min(20, l.numel())
    flat_l = l.flatten()
    flat_r = r.flatten()
    flat_d = diff.flatten()
    flat_rel = rel_err.flatten()
    print(f"  前{n}个元素对比 (idx | golden | operator | abs_diff | rel_err):")
    for i in range(n):
        print(f"    [{i:4d}] {flat_r[i].item():+.6e}  {flat_l[i].item():+.6e}  "
              f"{flat_d[i].item():.6e}  {flat_rel[i].item():.6e}")

    top_abs = torch.topk(flat_d, 10)
    print(f"  绝对diff最大的10个元素 (idx | golden | operator | abs_diff | rel_err):")
    for i in range(10):
        idx = top_abs.indices[i].item()
        print(f"    [{idx:7d}] {flat_r[idx].item():+.6e}  {flat_l[idx].item():+.6e}  "
              f"{flat_d[idx].item():.6e}  {flat_rel[idx].item():.6e}")

    top_rel = torch.topk(flat_rel, 10)
    print(f"  相对diff最大的10个元素 (idx | golden | operator | abs_diff | rel_err):")
    for i in range(10):
        idx = top_rel.indices[i].item()
        print(f"    [{idx:7d}] {flat_r[idx].item():+.6e}  {flat_l[idx].item():+.6e}  "
              f"{flat_d[idx].item():.6e}  {flat_rel[idx].item():.6e}")

def main():
    log_fh = open(LOG_FILE, "w")
    sys.stdout = Tee(sys.__stdout__, log_fh)

    base = os.path.join(SCRIPT_DIR, "atk_output")
    out_dir = find_latest_output(base)
    if not out_dir:
        print("未找到 atk_output 目录")
        sys.exit(1)
    print(f"输出目录: {out_dir}")

    cpu_base = os.path.join(out_dir, "output", "cpu_0")
    npu_base = os.path.join(out_dir, "output", "pyaclnn_0")
    if not os.path.isdir(cpu_base):
        print(f"cpu 目录不存在: {cpu_base}")
        sys.exit(1)
    save_names = [d for d in os.listdir(cpu_base) if os.path.isdir(os.path.join(cpu_base, d))]
    if not save_names:
        print(f"cpu_0 下无输出子目录: {cpu_base}")
        sys.exit(1)
    save_name = save_names[0]
    cpu_root = os.path.join(cpu_base, save_name)
    npu_root = os.path.join(npu_base, save_name)
    print(f"save_name: {save_name}")
    for case_id in os.listdir(cpu_root):
        cpu_dir = os.path.join(cpu_root, case_id)
        npu_dir = os.path.join(npu_root, case_id)
        if not os.path.isdir(cpu_dir) or not os.path.isdir(npu_dir):
            print(f"case {case_id}: 目录不完整, cpu={os.path.isdir(cpu_dir)} npu={os.path.isdir(npu_dir)}")
            continue

        cpu_files = sorted(glob.glob(os.path.join(cpu_dir, "*.pt")))
        npu_files = sorted(glob.glob(os.path.join(npu_dir, "*.pt")))
        print(f"\ncase {case_id}: cpu_files={len(cpu_files)} npu_files={len(npu_files)}")

        all_cpu = []
        all_npu = []
        for f in cpu_files:
            data = load_pt(f)
            all_cpu.extend(flatten_tensors(data))
        for f in npu_files:
            data = load_pt(f)
            all_npu.extend(flatten_tensors(data))

        n = min(len(all_cpu), len(all_npu))
        for i in range(n):
            compare_tensor(all_npu[i], all_cpu[i], f"output[{i}]")

    sys.stdout = sys.__stdout__
    log_fh.close()
    print(f"\n对比结果已保存到: {LOG_FILE}")

if __name__ == "__main__":
    main()
