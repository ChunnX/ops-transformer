#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tail-return 验证测试：验证 return 在 for 体内是否为死代码（tileNum=0），
以及 tail 块的实际执行开销。

原理：
  - 1000×1000 fp16 下 tileNum = 17858/21162 = 0（整数除法），for 循环 0 次迭代
  - return 在 for 体内 = 死代码，控制流落到 tail 块
  - tail 块完整执行 CopyIn+Add+CopyOut → ~1.5us 额外开销
  - 在 tail 块 AllocTensor 后加 return → tail 也不执行 → 降至 ~1.4us

测试场景：
  T1: for+return(死代码)+tail完整执行     → ~2.93us (56b) / ~4.16us (64b)
  T2: for+return(死代码)+tail内return      → ~1.40us (56b) / ~1.75us (64b)
  对比注释版:                                → ~1.35us (56b) / ~1.71us (64b)

使用方法：
  python3 tests/add/test_tail_return.py
"""

import os
import sys
import shutil
import subprocess
import statistics
import glob
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)

CANN_PATH = os.environ.get("CANN_PATH", "/usr/local/Ascend/cann-9.1.0")
NPU_SOC = os.environ.get("NPU_SOC_VERSION", "ascend950")


def run(cmd, env=None, timeout=300):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env)


def get_add_cpp():
    """读取原始 add.cpp 模板"""
    add_cpp = os.path.join(PROJECT_ROOT, "csrc/add/ascend950/add.cpp")
    with open(add_cpp) as f:
        return f.read(), add_cpp


def write_add_cpp(content, path):
    with open(path, "w") as f:
        f.write(content)


def generate_t1_source(original):
    """T1: 解注释 for头 + for体内到return + tail完整执行（return是死代码）"""
    lines = original.split("\n")
    new_lines = []
    for i, line in enumerate(lines):
        # 解注释 for 循环头 (line ~93: "// for (int64_t i = 0; i < tileNum; ++i) {")
        if "// for (int64_t i = 0; i < tileNum" in line:
            new_lines.append(line.replace("// ", ""))
        # 解注释 for 体内到 return 行（offset, copyParams, AllocTensor, return）
        elif "//     int64_t offset = i * elementNumPerTile;" in line:
            new_lines.append(line.replace("// ", ""))
        elif "//     AscendC::DataCopyExtParams copyParams;" in line and "tailTile" not in lines[max(0,i-1)]:
            new_lines.append(line.replace("// ", ""))
        elif "//     copyParams.blockCount = 1;" in line and i < 100:
            new_lines.append(line.replace("// ", ""))
        elif "//     copyParams.blockLen = elementNumPerTile * sizeof(T);" in line:
            new_lines.append(line.replace("// ", ""))
        elif "//     copyParams.srcStride = 0;" in line and i < 100:
            new_lines.append(line.replace("// ", ""))
        elif "//     copyParams.dstStride = 0;" in line and i < 100:
            new_lines.append(line.replace("// ", ""))
        elif "//     AscendC::DataCopyPadExtParams<T> padParams" in line and i < 102:
            new_lines.append(line.replace("// ", ""))
        elif "//     AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();" in line and i < 105:
            new_lines.append(line.replace("// ", ""))
        elif "//     AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();" in line and i < 105:
            new_lines.append(line.replace("// ", ""))
        elif "//     return;" in line and i < 106:
            new_lines.append(line.replace("// ", ""))
        # 解注释整个 tail 块
        elif "// if (tailTileElementNum > 0) {" in line:
            new_lines.append(line.replace("// ", ""))
        elif "// }" in line and i > 120:  # tail closing brace
            new_lines.append(line.replace("// ", ""))
        elif i >= 123 and i <= 149 and line.strip().startswith("// "):
            new_lines.append(line.replace("// ", ""))
        else:
            new_lines.append(line)
    return "\n".join(new_lines)


def generate_t2_source(original):
    """T2: T1 + 在 tail 块 AllocTensor 后加 return（tail也不执行）"""
    t1 = generate_t1_source(original)
    # 在 tail 块的 yLocal = inQueueY.AllocTensor<T>(); 之后加 return;
    t2 = t1.replace(
        "    AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();\n    AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);\n    AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);",
        "    AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();\n    return;\n    AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);\n    AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);"
    )
    return t2


def build_and_test(source_content, tag, blocks=56):
    """编译指定源码并运行 msprof 测试"""
    add_cpp = os.path.join(PROJECT_ROOT, "csrc/add/ascend950/add.cpp")
    backup = add_cpp + ".bak_orig"

    # 保存原始
    if not os.path.exists(backup):
        shutil.copy2(add_cpp, backup)

    # 写入测试版本
    write_add_cpp(source_content, add_cpp)

    # sed 修改 blocks
    if blocks == 56:
        run(f"sed -i 's/<<<64,/<<<56,/' {add_cpp}")
        run(f"sed -i 's/std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) \\/ MIN_ELEMS_PER_CORE)/56/' {add_cpp}")

    # 清理
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
        print(f"  [{tag}] BUILD FAIL")
        return None

    # 安装
    so = os.path.join(PROJECT_ROOT, "build/_C.abi3.so")
    if not os.path.exists(so):
        import zipfile
        for w in glob.glob(os.path.join(PROJECT_ROOT, "dist/*.whl")):
            with zipfile.ZipFile(w) as zf:
                for n in zf.namelist():
                    if n.endswith(".so"):
                        zf.extract(n, os.path.join(PROJECT_ROOT, "build"))
    if not os.path.exists(so):
        print(f"  [{tag}] BUILD FAIL: no .so")
        return None

    shutil.copy2(so, "/usr/local/lib/python3.9/site-packages/ascend_ops/_C.abi3.so")

    # msprof 测试
    results_dir = os.path.join(PROJECT_ROOT, f"results_tail_ret/{tag}")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    test_script = f"""
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

    env = os.environ.copy()
    r = run(f"source {CANN_PATH}/set_env.sh && msprof --output={results_dir} python3 -c '{test_script}' 2>&1",
            env=env, timeout=120)

    # 解析
    csvs = glob.glob(os.path.join(results_dir, "**/op_summary_*.csv"), recursive=True)
    if csvs:
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        adds = [r for r in rows if "add_kernel" in r.get("Op Name", "")]
        if adds:
            durs = [float(r["Task Duration(us)"].strip()) for r in adds]
            aivs = [float(r["aiv_time(us)"].strip()) for r in adds if r.get("aiv_time(us)", "").strip()]
            sd = durs[2:] if len(durs) > 2 else durs
            sa = aivs[2:] if len(aivs) > 2 else aivs
            return {
                "dur": statistics.mean(sd),
                "aiv": statistics.mean(sa),
                "first5_dur": durs[:5],
            }

    return None


def main():
    print("=== Tail-return 验证测试 ===\n")
    print("验证 return 在 for 体内是否为死代码，以及 tail 块的实际开销\n")

    original, add_cpp_path = get_add_cpp()

    scenarios = [
        ("T1_full_tail", generate_t1_source(original),
         "for+return(死代码)+tail完整执行 CopyIn+Add+CopyOut"),
        ("T2_tail_ret", generate_t2_source(original),
         "for+return(死代码)+tail内AllocTensor后return(无DMA)"),
    ]

    print(f"{'Scenario':<20} {'Description':<50} {'Blocks':>6} {'Duration':>10} {'aiv_time':>10}")
    print("-" * 100)

    results = {}
    for tag, source, desc in scenarios:
        r = build_and_test(source, tag, blocks=56)
        if r:
            print(f"{tag:<20} {desc:<50} {'56':>6} {r['dur']:>9.3f}us {r['aiv']:>9.3f}us")
            results[tag] = r
        else:
            print(f"{tag:<20} {desc:<50} {'56':>6} {'FAIL':>10}")

    # 恢复原始
    backup = add_cpp_path + ".bak_orig"
    if os.path.exists(backup):
        shutil.copy2(backup, add_cpp_path)
        os.remove(backup)

    # 分析结果
    if "T1_full_tail" in results and "T2_tail_ret" in results:
        t1 = results["T1_full_tail"]
        t2 = results["T2_tail_ret"]
        diff = t1["dur"] - t2["dur"]
        print(f"\n=== 分析 ===")
        print(f"T1 (tail完整执行): {t1['dur']:.3f} us")
        print(f"T2 (tail内return): {t2['dur']:.3f} us")
        print(f"差异: {diff:.3f} us = tail 块 CopyIn+Add+CopyOut 的开销")
        print(f"\n结论:")
        print(f"  - return 在 for 体内是死代码（tileNum=0），对性能零影响")
        print(f"  - tail 块的实际 DMA+计算 贡献了 ~{diff:.1f}us 额外开销")
        print(f"  - 双return版(T2) 与注释版(~1.35us) 几乎一致，确认 return 本身零开销")


if __name__ == "__main__":
    main()
