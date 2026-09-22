"""bench_prep: prep_cache 数据生成基准 (设计文档 M2/M3, 验收 V2/V3/V8)。

只依赖 numpy/torch/prep_cache, 无 torch_npu 依赖, 本机无 NPU 可跑。
每规格: 冷生成(或本就命中则标注) → 二次 generate(应命中) → load 计时 →
shape/dtype 断言 → 首块/末块/中部随机块独立 rng 重算逐字节比对 → total%4 整除性记录。
"""

import argparse
import os
import platform
import resource
import shutil
import sys
import time

import numpy as np
import torch

import prep_cache

DTYPE = "fp8_e4m3fn"
USE_FP8 = True
SF_DTYPE = "e8m0"
WORLD_SIZE = 4
BLOCK_ROWS = prep_cache.DEFAULT_BLOCK_ROWS

# (name, total, hidden, seed); 全部 dtype=fp8_e4m3fn, use_fp8=1, sf_dtype=e8m0, world_size=4
SPECS = [
    ("functional", 1024, 576, 11001),
    ("vocab_small", 5138000, 576, 11001),
    ("vocab_medium", 10276000, 576, 11002),
    ("vocab_medium2", 20552000, 576, 11003),
    ("vocab_large1", 41104000, 576, 11004),
    ("vocab_large2", 82208000, 576, 11005),
    ("tensor_1x", 5138000, 576, 11001),   # 与 vocab_small 同 key, 验证缓存复用命中
    ("tensor_4x", 12845000, 2304, 11003),
    ("tensor_16x", 321128, 9216, 11005),
]


def _fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def _fs_type(path):
    """cache 目录所在文件系统类型 (/proc/mounts 最长前缀匹配)。"""
    try:
        real = os.path.realpath(path)
        best_mnt, best_fs = "", "?"
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt, fstype = os.path.realpath(parts[1]), parts[2]
                if real == mnt or real.startswith(mnt.rstrip(os.sep) + os.sep):
                    if len(mnt) > len(best_mnt):
                        best_mnt, best_fs = mnt, fstype
        return best_fs
    except Exception:
        return "?"


def _shm_used_bytes():
    try:
        return shutil.disk_usage("/dev/shm").used
    except Exception:
        return None


def run_spec(name, total, hidden, seed, cache_dir):
    row = {
        "name": name, "total": total, "hidden": hidden,
        "storage_gb": None, "sf_gb": None, "cold_s": None, "cold_hit": None,
        "hit_s": None, "load_s": None, "sample": "SKIP", "total_mod4": total % WORLD_SIZE,
    }
    errors = []

    r1 = prep_cache.generate(cache_dir, total, hidden, DTYPE, USE_FP8, SF_DTYPE, seed)
    row["cold_hit"] = bool(r1["cache_hit"])
    row["cold_s"] = r1["elapsed_s"]
    row["storage_gb"] = r1["storage_bytes"] / 1024 ** 3
    row["sf_gb"] = r1["sf_bytes"] / 1024 ** 3

    t0 = time.perf_counter()
    r2 = prep_cache.generate(cache_dir, total, hidden, DTYPE, USE_FP8, SF_DTYPE, seed)
    row["hit_s"] = time.perf_counter() - t0
    if not r2["cache_hit"]:
        errors.append("二次 generate 未命中缓存")

    t0 = time.perf_counter()
    storage, sf = prep_cache.load(cache_dir, total, hidden, DTYPE, USE_FP8, SF_DTYPE, seed)
    row["load_s"] = time.perf_counter() - t0

    sf_cols = (hidden + 31) // 32
    if tuple(storage.shape) != (total, hidden):
        errors.append(f"storage shape {tuple(storage.shape)} != {(total, hidden)}")
    if storage.dtype != torch.float8_e4m3fn:
        errors.append(f"storage dtype {storage.dtype} != float8_e4m3fn")
    if tuple(sf.shape) != (total, sf_cols):
        errors.append(f"sf shape {tuple(sf.shape)} != {(total, sf_cols)}")
    if sf.dtype != torch.float8_e8m0fnu:
        errors.append(f"sf dtype {sf.dtype} != float8_e8m0fnu")

    # 抽样校验: 首块、末块、中部随机 1 块 (每块 ≤ block_rows 行), 独立 rng 重算逐字节比对
    n_blk = prep_cache.num_blocks(total, BLOCK_ROWS)
    picks = {0, n_blk - 1}
    if n_blk > 2:
        picks.add(int(np.random.default_rng(20260922).integers(1, n_blk - 1)))
    s_raw = storage.view(torch.uint8).numpy()
    sf_raw = sf.view(torch.uint8).numpy()
    sample_ok = True
    for b in sorted(picks):
        r0 = b * BLOCK_ROWS
        r1_ = min(r0 + BLOCK_ROWS, total)
        exp_s = prep_cache.expected_storage_block(b, total, hidden, DTYPE, seed, block_rows=BLOCK_ROWS)
        exp_f = prep_cache.expected_sf_block(b, total, hidden, SF_DTYPE, seed, block_rows=BLOCK_ROWS)
        if not np.array_equal(s_raw[r0:r1_], exp_s):
            sample_ok = False
            errors.append(f"storage 块 {b} 与独立重算不一致")
        if not np.array_equal(sf_raw[r0:r1_], exp_f):
            sample_ok = False
            errors.append(f"sf 块 {b} 与独立重算不一致")
    row["sample"] = "PASS" if sample_ok else "FAIL"
    return row, errors


def build_report(rows, failures, env, shm_before, shm_after, rss_gb, gen_stats):
    lines = []
    lines.append("# prep_cache 数据生成基准报告 (M2+M3, V2/V3/V8)")
    lines.append("")
    lines.append(f"- 生成时间: {env['time']}")
    lines.append(f"- 数据规格: dtype={DTYPE}, use_fp8={int(USE_FP8)}, sf_dtype={SF_DTYPE}, "
                 f"world_size={WORLD_SIZE} (total%4 为 sliced 整除性, 仅报告)")
    lines.append("")
    lines.append("## 规格结果")
    lines.append("")
    lines.append("| 规格 | total | hidden | 存储GB | sfGB | 冷生成s | 命中s | load s | 抽样校验 | total%4 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cold = f"hit({_fmt(r['cold_s'])})" if r["cold_hit"] else _fmt(r["cold_s"])
        lines.append(
            f"| {r['name']} | {r['total']} | {r['hidden']} | {_fmt(r['storage_gb'])} "
            f"| {_fmt(r['sf_gb'])} | {cold} | {_fmt(r['hit_s'])} | {_fmt(r['load_s'])} "
            f"| {r['sample']} | {r['total_mod4']} |")
    lines.append("")
    lines.append("## 环境")
    lines.append("")
    lines.append(f"- host: {env['host']}")
    lines.append(f"- cpu_count: {env['cpu_count']}")
    lines.append(f"- cache_dir: {env['cache_dir']} (文件系统: {env['fs_type']}, "
                 f"可用 {env['free_gib']} GiB / 共 {env['total_gib']} GiB)")
    if shm_before is not None and shm_after is not None:
        lines.append(f"- /dev/shm 占用: 运行前 {shm_before / 1024 ** 2:.1f} MiB, "
                     f"运行后 {shm_after / 1024 ** 2:.1f} MiB "
                     f"(差值 {(shm_after - shm_before) / 1024 ** 2:+.1f} MiB, 应基本不变)")
    else:
        lines.append("- /dev/shm: 不可用(跳过)")
    lines.append(f"- 峰值 RSS (ru_maxrss, RUSAGE_SELF): {rss_gb:.2f} GB")
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(f"- 冷生成: {gen_stats['miss_count']} 次未命中, 总耗时 {gen_stats['cold_total_s']:.2f}s, "
                 f"写入总量 {gen_stats['written_gib']:.2f} GiB, "
                 f"平均生成速度 {gen_stats['gbps']:.2f} GiB/s")
    lines.append(f"- 缓存复用: {gen_stats['hit_count']} 个规格首次 generate 即命中; "
                 f"tensor_1x 与 vocab_small 同 key (total/hidden/seed 相同), "
                 f"{'已验证复用命中' if gen_stats['tensor_1x_reused'] else '未同时执行, 未验证'}")
    lines.append(f"- /dev/shm 前后占用{'基本不变 (未使用共享内存)' if shm_before is not None and abs(shm_after - shm_before) < 64 * 1024 ** 2 else '有明显变化, 请检查'}; "
                 f"峰值 RSS {rss_gb:.2f} GB (分块生成, 无整表 fp32 中间)")
    if failures:
        lines.append("")
        lines.append("## 失败原因")
        lines.append("")
        for msg in failures:
            lines.append(f"- {msg}")
    lines.append("")
    lines.append("ALL SPECS PASS" if not failures else "SOME SPECS FAILED")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="prep_cache 数据生成基准 (无 NPU 依赖)")
    parser.add_argument("--report", type=str, required=True, help="markdown 报告输出路径")
    parser.add_argument("--specs", type=str, default="all",
                        choices=["all", "functional", "vocab", "tensor"],
                        help="规格过滤: all(默认)/functional/vocab/tensor")
    parser.add_argument("--cache_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prep_cache"),
                        help="缓存目录, 默认取本脚本所在目录/prep_cache")
    args = parser.parse_args()

    def keep(name):
        if args.specs == "all":
            return True
        if args.specs == "functional":
            return name == "functional"
        return name.startswith(args.specs + "_")

    os.makedirs(args.cache_dir, exist_ok=True)
    shm_before = _shm_used_bytes()
    env = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "cache_dir": os.path.abspath(args.cache_dir),
        "fs_type": _fs_type(args.cache_dir),
    }
    du = shutil.disk_usage(args.cache_dir)
    env["free_gib"], env["total_gib"] = f"{du.free / 1024 ** 3:.2f}", f"{du.total / 1024 ** 3:.2f}"

    rows, failures = [], []
    for name, total, hidden, seed in SPECS:
        if not keep(name):
            continue
        print(f"[BENCH] {name}: total={total}, hidden={hidden}, seed={seed} ...", flush=True)
        try:
            row, errors = run_spec(name, total, hidden, seed, args.cache_dir)
        except Exception as e:
            row = {"name": name, "total": total, "hidden": hidden, "storage_gb": None, "sf_gb": None,
                   "cold_s": None, "cold_hit": None, "hit_s": None, "load_s": None,
                   "sample": "FAIL", "total_mod4": total % WORLD_SIZE}
            errors = [f"{type(e).__name__}: {e}"]
        rows.append(row)
        failures.extend(f"{name}: {msg}" for msg in errors)
        print(f"[BENCH] {name}: cold_hit={row['cold_hit']} cold_s={_fmt(row['cold_s'])} "
              f"hit_s={_fmt(row['hit_s'])} load_s={_fmt(row['load_s'])} sample={row['sample']} "
              f"({len(errors)} errors)", flush=True)

    shm_after = _shm_used_bytes()
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1024 ** 3  # Linux: KB

    miss_rows = [r for r in rows if r["cold_hit"] is False]
    hit_rows = [r for r in rows if r["cold_hit"] is True]
    cold_total_s = sum(r["cold_s"] for r in miss_rows if r["cold_s"] is not None)
    written_gib = sum((r["storage_gb"] or 0) + (r["sf_gb"] or 0) for r in miss_rows)
    tensor_1x = next((r for r in rows if r["name"] == "tensor_1x"), None)
    gen_stats = {
        "miss_count": len(miss_rows),
        "hit_count": len(hit_rows),
        "cold_total_s": cold_total_s,
        "written_gib": written_gib,
        "gbps": (written_gib / cold_total_s) if cold_total_s > 0 else 0.0,
        "tensor_1x_reused": bool(tensor_1x and tensor_1x["cold_hit"]),
    }

    report = build_report(rows, failures, env, shm_before, shm_after, rss_gb, gen_stats)
    report_dir = os.path.dirname(os.path.abspath(args.report))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    verdict = "ALL SPECS PASS" if not failures else "SOME SPECS FAILED"
    print(f"[BENCH] 报告已写入: {os.path.abspath(args.report)}")
    print(f"[BENCH] {verdict}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
