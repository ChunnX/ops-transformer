#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# perf_decode.py — mega_moe A3 ws16 decode 场景性能采集（perf-only，不做 CPU golden 比对）
#
# 与 golden_megamoe.py 的分工：
#   golden_megamoe.py  = 精度 + 单次 Duration（跑得慢，CPU golden 是大头）
#   perf_decode.py     = 纯性能，抓 kernel_details.csv 全量 pipe 指标（mac/mte1/mte2/fixpipe/vec/...）
#
# 关键设计：
#   1. 一次 fork 内重复调用 N 次算子，profiler 抓到 N 条 MegaMoe 记录 → 取中位数消除 ±5% 噪声。
#      （比"重复 fork N 次"快一个数量级，import + HCCL 建链只付一次）
#   2. 复用 golden_megamoe 的 CASES / make_inputs，保证 shape 与精度用例完全一致。
#   3. 输出 CSV + 终端表格，便于 baseline / 优化版对比。
#
# 用法：
#   source switch_vendor.sh master
#   python3 perf_decode.py --tag master                 # 全部 decode 用例
#   python3 perf_decode.py --tag opt-x --case dsv3_ws16_dec32
#   python3 perf_decode.py --tag master --repeat 7
# -----------------------------------------------------------------------------------------------------------
import os
import sys
import csv
import time
import glob
import json
import argparse
import statistics

import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_megamoe as GM

# kernel_details.csv 里要提取的列（Duration + AIC/AIV 各 pipe 占比）
METRICS = [
    "Duration(us)", "Block Num", "Mix Block Num", "aicore_time(us)",
    "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
    "aic_fixpipe_ratio", "aic_icache_miss_rate",
    "aiv_time(us)", "aiv_vec_ratio", "aiv_scalar_ratio",
    "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_icache_miss_rate",
    "cube_utilization(%)",
]

G_INPUT = None
G_CASE = None
G_DONE = None

# ---------------------------------------------------------------------------------------------
# 固定开销探针用例：固定 hidden/inter/tokens，只扫 epr（=权重量），其余全部不变。
# Duration 对权重量做线性回归，**截距即为与权重量无关的固定串行开销**。
# 这比用 5 个真实模型拟合干净得多（那里 hidden/inter/topk 全在变，混淆因素多）。
# 用 ws=16 是为了让跨卡同步规模与真实 decode 场景一致。
# ---------------------------------------------------------------------------------------------
PROBE_CASES = [
    dict(name="probe_epr1",  world_size=16, num_experts=16,  hidden=1024, intermediate_hidden=1024,
         topk=1, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=90),
    dict(name="probe_epr4",  world_size=16, num_experts=64,  hidden=1024, intermediate_hidden=1024,
         topk=4, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=91),
    dict(name="probe_epr8",  world_size=16, num_experts=128, hidden=1024, intermediate_hidden=1024,
         topk=8, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=92),
    dict(name="probe_epr16", world_size=16, num_experts=256, hidden=1024, intermediate_hidden=1024,
         topk=8, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=93),
]

# ---------------------------------------------------------------------------------------------
# 第二组探针：**固定 epr=8，只扫 hidden**（权重量随 hidden 增长，但专家迭代次数不变）。
#
# 为什么需要它：PROBE_CASES 里 epr 与权重量完全共线（权重 = epr × 6.29MB），
# 回归斜率里「每 MB 带宽时间」与「每专家固定开销」混在一起，无法分离。
# 本组斜率是**纯带宽项**；与 PROBE_CASES 的斜率相减，即得每专家固定开销
# （= dispatch 循环体的 SyncAll + CrossCoreSetFlag + GMM1 每专家 PipeBarrier/SetGlobalBuffer
#    + combine 每专家 CrossCoreWaitFlag 等的总和）。
#
# 权重量 = epr × (h×2×ih + ih×h) × 2B，epr=8/ih=1024 时 = 8×(h×2048 + 1024h)×2 = 49152·h 字节
#   h=1024 →  50.3 MB
#   h=2048 → 100.7 MB
#   h=4096 → 201.3 MB
# ---------------------------------------------------------------------------------------------
PROBE_H_CASES = [
    dict(name="probeH_h1024", world_size=16, num_experts=128, hidden=1024, intermediate_hidden=1024,
         topk=8, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=94),
    dict(name="probeH_h2048", world_size=16, num_experts=128, hidden=2048, intermediate_hidden=1024,
         topk=8, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=95),
    dict(name="probeH_h4096", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1024,
         topk=8, num_tokens=32, num_max_tokens_per_rank=64, input_mode="random",
         topk_ids_mode="unique", topk_weights_mode="random", seed=96),
]


def weight_bytes(c):
    """每卡权重总字节数：epr 个专家 × (w1[h,2ih] + w2[ih,h])，bf16=2B。
    decode 场景这是 MTE2 搬运量的主体，用来算带宽利用率。"""
    epr = c["num_experts"] // c["world_size"]
    h, ih = c["hidden"], c["intermediate_hidden"]
    return epr * (h * 2 * ih + ih * h) * 2


def make_inputs_fast(c):
    """perf 专用的轻量输入构造。

    与 golden 的 make_inputs 的区别（以及为什么可以这么做）：
      - golden 要比对数值，必须为 256 个专家各造一份随机权重 → dsv3 下 22.5GB CPU 内存 +
        海量 uniform_ 随机数生成，单 case 构造就要几分钟。
      - perf 只测耗时，权重"内容"不影响性能：GMM 的搬运量只由 shape 决定，L2 按地址索引
        （内容相同不会提高命中率）。所以这里让 epr 个专家共享同一个 CPU tensor 对象，
        .npu() 时每个引用仍会拷贝成地址独立的 NPU tensor，搬运量与 golden 完全一致。
      - topk_ids / topk_weights 仍复用 golden 的生成逻辑：它们决定每专家的 token 分布，
        是会影响性能的，必须与精度用例保持一致。
    """
    GM.assert_case(c)
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    h, ih = c["hidden"], c["intermediate_hidden"]
    tok, seed = c["num_tokens"], c["seed"]

    # epr 个专家共享同一份 CPU 权重（NPU 侧仍是 epr 份独立显存）
    w1_shared = torch.full((h, 2 * ih), 0.01, dtype=torch.bfloat16)
    w2_shared = torch.full((ih, h), 0.01, dtype=torch.bfloat16)

    per_rank = []
    for rank in range(ws):
        g = torch.Generator().manual_seed(seed * 100003 + rank * 7)
        x = torch.empty(tok, h, dtype=torch.float32).uniform_(-0.1, 0.1, generator=g).to(torch.bfloat16)
        per_rank.append({
            "x": x,
            "topk_ids": GM._topk_ids(c, g),
            "topk_weights": GM._topk_weights(c, g).to(torch.bfloat16),
            "l1": [w1_shared] * epr,
            "l2": [w2_shared] * epr,
        })
    return per_rank


def run_one_rank(rank):
    c = G_CASE
    ws = c["world_size"]
    repeat = c["repeat"]
    from cann_ops_transformer.ops import mega_moe, get_symm_buffer_for_mega_moe

    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws,
                            init_method=f"tcp://127.0.0.1:{c['port']}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(ws)))

    # HCCL warmup（必须，否则 get_symm_buffer 内 HcclCommGetHandleWithName 失败）
    warm = torch.ones(1, device="npu")
    dist.all_reduce(warm, group=ep_group)
    torch.npu.synchronize()

    d = G_INPUT[rank]
    x = d["x"].to(torch.bfloat16).npu()
    ids = d["topk_ids"].to(torch.int32).npu()
    w = d["topk_weights"].to(torch.float32).npu()   # 必须 float32，bf16 会精度崩
    l1 = [t.to(torch.bfloat16).npu() for t in d["l1"]]
    l2 = [t.to(torch.bfloat16).npu() for t in d["l2"]]

    epr = c["num_experts"] // ws
    max_recv = c["num_max_tokens_per_rank"] * ws * min(c["topk"], epr)
    sym = get_symm_buffer_for_mega_moe(
        ep_group, num_experts=c["num_experts"],
        num_max_tokens_per_rank=c["num_max_tokens_per_rank"],
        num_topk=c["topk"], hidden=c["hidden"],
        intermediate_hidden=c["intermediate_hidden"],
        max_recv_token_num=max_recv, dispatch_quant_mode=0, dispatch_quant_out_dtype=None)

    # warmup：首次调用含 gen_task 运行时编译，不能计入 profiling
    for _ in range(2):
        _ = mega_moe(x, ids, w, l1, l2, sym)
    torch.npu.synchronize()

    pdir = None
    if rank == 0:
        pdir = f"/tmp/perfdec_{c['tag']}_{c['name']}_{os.getpid()}"
        os.system(f"rm -rf {pdir}; mkdir -p {pdir}")
        ec = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=ec,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(pdir))
        prof.start()

    # 重复 repeat 次 → profiler 抓 repeat 条记录，后面取中位数
    for _ in range(repeat):
        y, etn = mega_moe(x, ids, w, l1, l2, sym)
    torch.npu.synchronize()

    if rank == 0:
        prof.stop()
        rows = []
        for kd in glob.glob(f"{pdir}/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"):
            with open(kd) as f:
                for row in csv.DictReader(f):
                    if "MegaMoe" in row.get("Name", ""):
                        rows.append({k: row.get(k) for k in METRICS})
            if rows:
                break
        with open(f"{c['dump_dir']}/perf_{c['name']}.json", "w") as f:
            json.dump(rows, f)

    G_DONE[rank].set()
    dist.destroy_process_group()


def run_case(c, repeat, tag, idx, timeout=600):
    global G_INPUT, G_CASE, G_DONE
    c = dict(c)
    c.update(port=51000 + idx, repeat=repeat, tag=tag,
             dump_dir=f"/tmp/perfdec_dump_{os.getpid()}")
    os.makedirs(c["dump_dir"], exist_ok=True)

    G_INPUT = make_inputs_fast(c)
    G_CASE = c
    G_DONE = [mp.Event() for _ in range(c["world_size"])]

    mp.set_start_method("fork", force=True)
    procs = [Process(target=run_one_rank, args=(r,)) for r in range(c["world_size"])]
    for p in procs:
        p.start()

    # 必须带 timeout：某个 rank 崩溃（HCCL 资源不足等）时不会 set event，
    # 无 timeout 的 wait() 会让主进程永久阻塞，整个采集卡死。
    deadline = time.time() + timeout
    for r, e in enumerate(G_DONE):
        if not e.wait(max(1, deadline - time.time())):
            print(f"    [timeout] rank{r} 未完成，判定该 case 失败")
            break
    for p in procs:
        if p.is_alive():
            p.terminate()
        p.join(timeout=30)

    jf = f"{c['dump_dir']}/perf_{c['name']}.json"
    rows = json.load(open(jf)) if os.path.exists(jf) else []
    if os.path.exists(jf):
        os.remove(jf)          # 防止下个 case 读到上个 case 的残留
    if not rows:
        return None

    def med(key):
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "N/A")]
        return statistics.median(vals) if vals else float("nan")

    out = {k: med(k) for k in METRICS}
    out["name"] = c["name"]
    out["samples"] = len(rows)
    out["all_dur"] = sorted(round(float(r["Duration(us)"]), 1) for r in rows)
    # 带宽利用率：权重字节数 / Duration，对比 A3 理论 20核×32B/cycle×1.85GHz = 1184 GB/s
    wb = weight_bytes(c)
    out["weight_GB"] = wb / 1e9
    out["eff_BW_GBs"] = wb / (out["Duration(us)"] * 1e-6) / 1e9
    out["BW_util_%"] = out["eff_BW_GBs"] / 1184.0 * 100
    return out


def main():
    ap = argparse.ArgumentParser(description="mega_moe decode 场景性能采集")
    ap.add_argument("--tag", required=True, help="版本标签，如 master / opt-xxx")
    ap.add_argument("--case", default=None, help="只跑单个用例")
    ap.add_argument("--repeat", type=int, default=5, help="单次 fork 内重复次数（取中位数）")
    ap.add_argument("--filter", default="ws16_dec", help="用例名过滤子串")
    ap.add_argument("--timeout", type=int, default=600, help="单 case 超时秒数（防某 rank 崩溃导致卡死）")
    ap.add_argument("--probe", action="store_true",
                    help="跑固定开销探针用例（只扫 epr，回归出与权重量无关的固定开销截距）")
    ap.add_argument("--probeh", action="store_true",
                    help="跑纯带宽探针（固定 epr=8 只扫 hidden），其斜率与 --probe 斜率相减即得每专家固定开销")
    ap.add_argument("--out", default=None, help="结果 CSV 路径")
    args = ap.parse_args()

    if args.probe:
        cases = list(enumerate(PROBE_CASES))
    elif args.probeh:
        cases = [(100 + i, c) for i, c in enumerate(PROBE_H_CASES)]
    elif args.case:
        cases = [(i, c) for i, c in enumerate(GM.CASES) if c["name"] == args.case]
        assert cases, f"未知用例 {args.case}"
    else:
        cases = [(i, c) for i, c in enumerate(GM.CASES) if args.filter in c["name"]]

    print(f"[perf_decode] tag={args.tag} repeat={args.repeat} cases={len(cases)}", flush=True)
    results = []
    for i, c in cases:
        t0 = time.time()
        r = run_case(c, args.repeat, args.tag, i, timeout=args.timeout)
        if r is None:
            print(f"  {c['name']:20s} FAILED (no profiling data)", flush=True)
            continue
        results.append(r)
        print(f"  {r['name']:20s} Dur={r['Duration(us)']:8.1f}us  "
              f"BW={r['BW_util_%']:5.1f}%  mte2={r['aic_mte2_ratio']:.3f} "
              f"mac={r['aic_mac_ratio']:.3f} vec={r['aiv_vec_ratio']:.3f}  "
              f"{r['all_dur']}  ({time.time()-t0:.0f}s)", flush=True)

    out = args.out or f"/tmp/perf_decode_{args.tag}.csv"
    if results:
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            wr.writeheader()
            for r in results:
                wr.writerow(r)
        print(f"\n[perf_decode] -> {out}")
        print(f"\n{'case':22s} {'Dur(us)':>9s} {'权重GB':>7s} {'有效BW':>8s} {'BW%':>6s} "
              f"{'mte2':>6s} {'mac':>6s} {'mte1':>6s} {'fix':>6s} {'vec':>6s}")
        for r in results:
            print(f"{r['name']:22s} {r['Duration(us)']:9.1f} {r['weight_GB']:7.3f} "
                  f"{r['eff_BW_GBs']:8.0f} {r['BW_util_%']:6.1f} {r['aic_mte2_ratio']:6.3f} "
                  f"{r['aic_mac_ratio']:6.3f} {r['aic_mte1_ratio']:6.3f} "
                  f"{r['aic_fixpipe_ratio']:6.3f} {r['aiv_vec_ratio']:6.3f}")


if __name__ == "__main__":
    main()
