#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GetSystemCycle 打点测试：在 kernel 内部插入 GetSystemCycle() 读取 SYS_CNT 寄存器，
精确测量 prologue（InitBuffer+SetGlobalBuffer+Tiling）的硬件周期数。

使用方法：
  1. 将 csrc/add/ascend950/add_cycle.cpp 复制为 csrc/add/ascend950/add.cpp
  2. 运行本脚本：python3 tests/add/cycle_test.py

原理：
  - kernel 在 InitBuffer 后、Tiling 后、kernel 结束时各调一次 GetSystemCycle()
  - 结果写入 GM buffer，host 侧读取后计算 prologue 和 wall-clock 周期数
  - 与 msprof aiv_time 交叉验证，标定 SYS_CNT 频率
"""

import os
import sys
import shutil
import subprocess
import statistics
import glob
import csv

# 工程根目录（tests/add/ 的上两级）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)

CANN_PATH = os.environ.get("CANN_PATH", "/usr/local/Ascend/cann-9.1.0")
NPU_SOC = os.environ.get("NPU_SOC_VERSION", "ascend950")


def run(cmd, env=None, timeout=300):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)


def build_and_install():
    """编译 add_cycle 版本的 wheel 并安装"""
    # 备份原始 add.cpp，替换为 cycle 版本
    add_cpp = os.path.join(PROJECT_ROOT, "csrc/add/ascend950/add.cpp")
    add_cycle = os.path.join(PROJECT_ROOT, "csrc/add/ascend950/add_cycle.cpp")
    backup = add_cpp + ".bak_commented"

    if not os.path.exists(backup):
        shutil.copy2(add_cpp, backup)
    shutil.copy2(add_cycle, add_cpp)

    # 清理构建目录
    for d in ["build", "dist"]:
        p = os.path.join(PROJECT_ROOT, d)
        if os.path.exists(p):
            shutil.rmtree(p)
    for e in glob.glob(os.path.join(PROJECT_ROOT, "*.egg-info")):
        shutil.rmtree(e)

    # 编译
    env = os.environ.copy()
    env["NPU_SOC_VERSION"] = NPU_SOC
    r = run(f"cd {PROJECT_ROOT} && python3 -m build --wheel --no-isolation 2>&1 | tail -5", env=env)
    if r.returncode != 0:
        print(f"BUILD FAIL:\n{r.stdout[-1000:]}")
        sys.exit(1)

    # 安装 .so
    so = os.path.join(PROJECT_ROOT, "build/_C.abi3.so")
    if not os.path.exists(so):
        import zipfile
        for w in glob.glob(os.path.join(PROJECT_ROOT, "dist/*.whl")):
            with zipfile.ZipFile(w) as zf:
                for n in zf.namelist():
                    if n.endswith(".so"):
                        zf.extract(n, os.path.join(PROJECT_ROOT, "build"))
    if not os.path.exists(so):
        print("BUILD FAIL: no .so found")
        sys.exit(1)

    site_packages = "/usr/local/lib/python3.9/site-packages/ascend_ops"
    os.makedirs(site_packages, exist_ok=True)
    shutil.copy2(so, os.path.join(site_packages, "_C.abi3.so"))
    print(f"Built and installed .so ({os.path.getsize(so)} bytes)")


def run_cycle_test():
    """Eager 模式运行 GetSystemCycle 测试"""
    test_script = """
import torch, torch_npu
torch.npu.set_device(0)
import ascend_ops

x = torch.randn(1000, 1000, dtype=torch.float16).npu()
y = torch.randn(1000, 1000, dtype=torch.float16).npu()

# 多次运行，每次 kernel 内部打印 cycle 数据
for i in range(10):
    z = torch.ops.ascend_ops.add(x, y)
    torch.npu.synchronize()
    if i == 0:
        ref = x + y
        err = float((z - ref).abs().max())
        print(f"Correctness: err={err:.6f}")

print("All 10 runs done")
"""
    env = os.environ.copy()
    r = run(f"source {CANN_PATH}/set_env.sh && python3 -c '{test_script}' 2>&1", env=env, timeout=60)
    print(r.stdout)
    return r.stdout


def run_msprof_crossval():
    """NPUGraph + msprof 交叉验证"""
    test_script = """
import torch, torch_npu
torch.npu.set_device(0)
import ascend_ops

x = torch.randn(1000, 1000, dtype=torch.float16).npu()
y = torch.randn(1000, 1000, dtype=torch.float16).npu()

g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    for _ in range(101):
        torch.ops.ascend_ops.add(x, y)
torch.npu.synchronize()
g.replay()
torch.npu.synchronize()
print("done")
"""
    mdir = os.path.join(PROJECT_ROOT, "results_cycle_msprof")
    if os.path.exists(mdir):
        shutil.rmtree(mdir)
    env = os.environ.copy()
    r = run(f"source {CANN_PATH}/set_env.sh && msprof --output={mdir} python3 -c '{test_script}' 2>&1",
            env=env, timeout=120)
    print(r.stdout[-500:])

    # 解析 msprof 结果
    csvs = glob.glob(os.path.join(mdir, "**/op_summary_*.csv"), recursive=True)
    if csvs:
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        adds = [r for r in rows if "add_kernel" in r.get("Op Name", "")]
        if adds:
            durs = [float(r["Task Duration(us)"].strip()) for r in adds]
            aivs = [float(r["aiv_time(us)"].strip()) for r in adds if r.get("aiv_time(us)", "").strip()]
            cycs = [float(r["aiv_total_cycles"].strip()) for r in adds if r.get("aiv_total_cycles", "").strip()]
            sd = durs[2:] if len(durs) > 2 else durs
            sa = aivs[2:] if len(aivs) > 2 else aivs
            sc = cycs[2:] if len(cycs) > 2 else cycs
            print(f"\nmsprof (skip2): dur={statistics.mean(sd):.3f}us aiv={statistics.mean(sa):.3f}us cyc={statistics.mean(sc):.0f}")
            return statistics.mean(sa), statistics.mean(sd)
    return None, None


def parse_cycle_output(output):
    """解析 kernel 打印的 cycle 数据"""
    block0_data = []
    for line in output.split("\n"):
        parts = line.split()
        if len(parts) >= 7 and parts[0] == "0":
            try:
                t0 = int(parts[1])
                t1 = int(parts[2])
                t2 = int(parts[3])
                t_end = int(parts[4])
                prologue = int(parts[5])
                total = int(parts[6])
                if t0 > 0:
                    block0_data.append((prologue, total, t1 - t0, t2 - t1))
            except:
                pass
    return block0_data


def main():
    print("=== GetSystemCycle 头开销精确测量 ===\n")

    # Step 1: 编译
    build_and_install()

    # Step 2: Eager 模式运行
    print("\n=== Eager mode GetSystemCycle test ===")
    output = run_cycle_test()

    # Step 3: 解析结果
    block0_data = parse_cycle_output(output)
    if not block0_data:
        print("No valid cycle data found!")
        return

    prologues = [d[0] for d in block0_data]
    totals = [d[1] for d in block0_data]
    initbufs = [d[2] for d in block0_data]
    tilings = [d[3] for d in block0_data]

    print(f"\n{'Run':<5} {'InitBuf':>10} {'Tiling':>10} {'Prologue':>10} {'Total':>10}")
    for i, (p, t, ib, ti) in enumerate(block0_data):
        print(f"{i:<5} {ib:>10} {ti:>10} {p:>10} {t:>10}")

    mean_p = statistics.mean(prologues)
    mean_t = statistics.mean(totals)
    mean_ib = statistics.mean(initbufs)
    mean_ti = statistics.mean(tilings)

    print(f"\n{'Mean':<5} {mean_ib:>10.1f} {mean_ti:>10.1f} {mean_p:>10.1f} {mean_t:>10.1f}")

    # Step 4: msprof 交叉验证
    print("\n=== msprof cross-validation ===")
    msprof_aiv, msprof_dur = run_msprof_crossval()

    if msprof_aiv and msprof_dur:
        freq_aiv = mean_t / msprof_aiv
        freq_dur = mean_t / msprof_dur
        print(f"\n=== Frequency Calibration ===")
        print(f"Block0 mean total: {mean_t:.0f} cycles")
        print(f"msprof aiv_time: {msprof_aiv:.3f} us -> freq = {freq_aiv:.0f} MHz")
        print(f"msprof TaskDur: {msprof_dur:.3f} us -> freq = {freq_dur:.0f} MHz")

        f = freq_aiv
        print(f"\n=== HEAD OVERHEAD (freq={f:.0f} MHz) ===")
        print(f"InitBuffer(×3)+SetGlobalBuffer(×3): {mean_ib:.0f} cyc = {mean_ib/f*1000:.0f} ns")
        print(f"Tiling calculation:                {mean_ti:.0f} cyc = {mean_ti/f*1000:.0f} ns")
        print(f"Prologue total (per block):        {mean_p:.0f} cyc = {mean_p/f:.3f} us")
        print(f"Wall-clock (kernel):              {mean_t:.0f} cyc = {mean_t/f:.3f} us")
        print(f"Dispatch+exit:                    {(mean_t-mean_p):.0f} cyc = {(mean_t-mean_p)/f:.3f} us")

        # 验证：GetSystemCycle wall-clock 应与 msprof TaskDuration 一致
        print(f"\n=== Cross-Validation ===")
        print(f"GetSystemCycle wall-clock: {mean_t:.0f} cyc / {f:.0f} MHz = {mean_t/f:.3f} us")
        print(f"msprof Task Duration:      {msprof_dur:.3f} us")
        print(f"Match: {'YES' if abs(mean_t/f - msprof_dur) < 0.3 else 'NO (diff=' + f'{abs(mean_t/f - msprof_dur):.3f}us)'}}")

        # 注释版头开销 (msprof)
        commented_aiv = 1.08  # 56 blocks
        print(f"\nCommented version aiv_time (56 blocks): ~{commented_aiv} us")
        print(f"Cycle prologue: {mean_p:.0f} cyc / {f:.0f} MHz = {mean_p/f:.3f} us")
        print(f"Match: {'YES' if abs(mean_p/f - commented_aiv) < 0.3 else 'NO'}")


if __name__ == "__main__":
    main()
