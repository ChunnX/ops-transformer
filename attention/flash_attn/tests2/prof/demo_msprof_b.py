"""方案B 最小验证: subprocess 调 msprof CLI 包裹 python 子进程, 采集 flash_attn 耗时。

执行: python prof/demo_msprof_b.py [--case_id <id>] [--case_set <set>]

采样规则:
- decode (S1==1): 每轮前 L2 污染, 跑 5 次, 去最高最低, 剩 3 次取平均
- prefill (S1>1): 不污染, 跑 5 次, 去第一次, 剩 4 次取平均
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _l2_pollute(dtype, device, seed: int):
    """L2 缓存污染: 分配大 tensor 挤占 L2, 模拟冷缓存。每次用不同 seed。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    fb_cpu = torch.randn(32, 32768, 1, 128, dtype=dtype, generator=g)
    flush_buf = fb_cpu.to(device)
    _ = flush_buf.sum()
    torch.npu.synchronize()


def _perf_worker(case_set: str, case_id: str):
    """被 msprof 包裹的子进程入口 (通过 python -c 调用)。
    只跑: gen + warmup + npu×runs, 不做精度/记录。"""
    import torch
    import torch_npu  # noqa: F401  注册 npu backend
    from core.case_loader import load_cases, resolve_case_ids
    from core.gen_data import GenDataBackend
    from core.npu import NpuBackend

    cases, ids = load_cases(case_set)
    sel = resolve_case_ids(case_id, ids)
    if not sel:
        raise ValueError(f"case_id {case_id} not found in {case_set}")
    c = cases[ids.index(sel[0])]

    gen = GenDataBackend()
    inp = gen.gen_inputs(gen.normalize_params(c))
    devinp = {k: (v.to("npu") if isinstance(v, torch.Tensor) else v)
              for k, v in inp.items()}
    npu = NpuBackend()

    s1 = c.get("S1", 0)
    is_decode = (s1 == 1)
    dtype = inp["q"].dtype
    device = torch.device("npu")

    # 采样 5 次 (msprof 全程采集)
    for i in range(5):
        if is_decode:
            _l2_pollute(dtype, device, seed=100 + i)
        npu.npu_compute(devinp, "eager")
    torch.npu.synchronize()


_WORKER_SRC = (
    "import sys; sys.path.insert(0, '.'); "
    "from prof.demo_msprof_b import _perf_worker; "
    "_perf_worker({case_set!r}, {case_id!r})"
)


def run_msprof_wrap(prof_dir: str, case_set: str, case_id: str):
    """subprocess 调 msprof 包裹 python 子进程。"""
    os.makedirs(prof_dir, exist_ok=True)
    worker_src = _WORKER_SRC.format(case_set=case_set, case_id=case_id)
    cmd = [
        "msprof",
        f"--output={prof_dir}",
        "--aic-mode=task-based",
        sys.executable, "-c", worker_src,
    ]
    print(f"[demo] case_id={case_id} case_set={case_set}")
    print(f"[demo] cmd: msprof --output={prof_dir} --aic-mode=task-based python -c <worker>")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[demo] msprof stderr:", result.stderr[-2000:])
        result.check_returncode()
    print(f"[demo] msprof done, rc={result.returncode}")


def parse_op_summary(prof_dir: str, op_name: str = "FlashAttn") -> list[float]:
    """解析 op_summary_*.csv, 返回纯算子(非 Metadata)的耗时列表, 按采集顺序。"""
    prof_root = Path(prof_dir)
    csvs = sorted(prof_root.glob("PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
    if not csvs:
        print(f"[demo] WARN: no op_summary under {prof_dir}")
        for p in sorted(prof_root.rglob("*"))[:30]:
            print(f"     {p.relative_to(prof_root)}")
        return []

    import csv as csvmod
    durs = []
    with open(csvs[-1]) as f:
        reader = csvmod.DictReader(f)
        for r in reader:
            name = r.get("Op Name", r.get("Name", ""))
            if name == op_name:  # 严格匹配, 排除 FlashAttnMetadata
                durs.append(float(r.get("Task Duration(us)", 0) or 0))
    return durs


def aggregate(durs: list[float], is_decode: bool) -> dict:
    """按规则聚合 5 次采集结果。
    decode: 去最高最低 → 3 次平均;
    prefill: 去第一次 → 4 次平均。"""
    if len(durs) < 2:
        return {"raw": durs, "rule": "too few(<2)"}
    if is_decode:
        if len(durs) < 3:
            return {"raw": durs, "rule": "decode(<3)"}
        s = sorted(durs)
        trimmed = s[1:-1]  # 去最高最低
        return {"raw": durs, "rule": "decode(去最高最低)",
                "trimmed": trimmed, "avg": sum(trimmed) / len(trimmed)}
    else:
        if len(durs) < 2:
            return {"raw": durs, "rule": "prefill(<2)"}
        trimmed = durs[1:]  # 去第一次
        return {"raw": durs, "rule": "prefill(去第一次)",
                "trimmed": trimmed, "avg": sum(trimmed) / len(trimmed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_id", default="FP16_BNSD_01")
    ap.add_argument("--case_set", default="ci")
    args = ap.parse_args()

    from core.case_loader import load_cases, resolve_case_ids
    cases, ids = load_cases(args.case_set)
    sel = resolve_case_ids(args.case_id, ids)
    if not sel:
        print(f"[demo] case_id {args.case_id} not found"); sys.exit(1)
    c = cases[ids.index(sel[0])]
    s1 = c.get("S1", 0)
    is_decode = (s1 == 1)
    print(f"[demo] S1={s1} → {'decode' if is_decode else 'prefill'} 规则")

    prof_dir = tempfile.mkdtemp(prefix="msprof_demo_")
    run_msprof_wrap(prof_dir, args.case_set, sel[0])

    durs = parse_op_summary(prof_dir, "FlashAttn")
    print(f"\n[demo] FlashAttn 采集到 {len(durs)} 次: {durs}")

    agg = aggregate(durs, is_decode)
    print(f"[demo] 规则: {agg.get('rule')}")
    if "trimmed" in agg:
        print(f"[demo] 剔除后: {[f'{x:.3f}' for x in agg['trimmed']]}")
        print(f"[demo] 平均: {agg['avg']:.3f} us")


if __name__ == "__main__":
    main()
