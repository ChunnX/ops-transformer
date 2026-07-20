"""NPU 性能采集: subprocess 调 msprof CLI 包裹 python 子进程。

worker 函数定义在本模块内, 子进程通过 `python -c` import 调用,
无需独立脚本文件。采集规则:
- decode (S1==1): 每轮 L2 污染(不同seed), 跑 N 次, 去最高最低取平均
- prefill (S1>1): 不污染, 跑 N 次, 去第一次取平均
"""
import csv as csvmod
import os
import subprocess
import sys
from pathlib import Path

import torch

_WORKER_SRC_TMPL = (
    "import sys; sys.path.insert(0, {cwd!r}); "
    "from prof.npuprof import _perf_worker; "
    "_perf_worker({case_set!r}, {case_id!r}, {runs!r})"
)


def _l2_pollute(dtype, device, seed: int):
    """L2 缓存污染: 分配大 tensor 挤占 L2, 模拟冷缓存。每次用不同 seed。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    fb_cpu = torch.randn(32, 32768, 1, 128, dtype=dtype, generator=g)
    flush_buf = fb_cpu.to(device)
    _ = flush_buf.sum()
    torch.npu.synchronize()


def _perf_worker(case_set: str, case_id: str, runs: int):
    """被 msprof 包裹的子进程入口 (通过 python -c 调用)。
    只跑: gen + npu×runs, 不做精度/记录。"""
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

    for i in range(runs):
        if is_decode:
            _l2_pollute(dtype, device, seed=100 + i)
        npu.npu_compute(devinp, "eager")
    torch.npu.synchronize()


def run_msprof(case_id: str, cmd: list, prof_dir: str) -> str:
    """subprocess 调 msprof 包裹 cmd, 产物落 prof_dir。

    Args:
        case_id: 用例ID (用于日志)
        cmd: 被包裹命令 (已含 python + worker 调用)
        prof_dir: msprof --output 目录
    Returns:
        prof_dir (成功) 或 raise (失败)
    """
    os.makedirs(prof_dir, exist_ok=True)
    full_cmd = [
        "msprof",
        f"--output={prof_dir}",
        "--aic-mode=task-based",
    ] + cmd
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"msprof failed (rc={result.returncode}): {result.stderr[-1000:]}")


def run_msprof_case(case_id: str, case_set: str, prof_dir: str,
                    runs: int = 5) -> str:
    """便捷封装: 构造 worker cmd 并调 run_msprof。"""
    worker_src = _WORKER_SRC_TMPL.format(
        cwd=os.getcwd(), case_set=case_set, case_id=case_id, runs=runs)
    cmd = [sys.executable, "-c", worker_src]
    run_msprof(case_id, cmd, prof_dir)
    return prof_dir


def parse_prof_result(prof_dir: str, op_name: str = "FlashAttn") -> list[float]:
    """解析 op_summary_*.csv, 返回纯算子(非 Metadata)的耗时列表, 按采集顺序。"""
    prof_root = Path(prof_dir)
    csvs = sorted(prof_root.glob(
        "PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no op_summary under {prof_dir}")
    durs = []
    with open(csvs[-1]) as f:
        reader = csvmod.DictReader(f)
        for r in reader:
            name = r.get("Op Name", r.get("Name", ""))
            if name == op_name:  # 严格匹配, 排除 FlashAttnMetadata
                durs.append(float(r.get("Task Duration(us)", 0) or 0))
    return durs


def aggregate(durs: list[float], is_decode: bool) -> dict:
    """按规则聚合。
    decode: 去最高最低 → 3 次平均;
    prefill: 去第一次 → 4 次平均。"""
    if len(durs) < 2:
        return {"raw": durs, "rule": "too few(<2)"}
    if is_decode:
        if len(durs) < 3:
            return {"raw": durs, "rule": "decode(<3)"}
        s = sorted(durs)
        trimmed = s[1:-1]
        return {"raw": durs, "rule": "decode(去最高最低)",
                "trimmed": trimmed, "avg": sum(trimmed) / len(trimmed)}
    else:
        trimmed = durs[1:]
        return {"raw": durs, "rule": "prefill(去第一次)",
                "trimmed": trimmed, "avg": sum(trimmed) / len(trimmed)}
