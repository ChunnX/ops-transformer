#!/usr/bin/env python3
"""perf_all_cases.py — mega_moe A3 性能检测：torch_npu.profiler 抓 MegaMoe kernel Duration（rank0 子进程内）。
用法:
    python3 perf_all_cases.py <case_name>   # 单 case，打印 "<case>\\t<Duration_us>"
    python3 perf_all_cases.py --all         # 跑全部 15 case（每行一个结果）
前置（每个新 shell）:
    source <EXP>/cann/ascend-toolkit/set_env.sh
    source <EXP>/cann/vendors/custom_transformer/bin/set_env.bash
    export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<EXP>/.torch_extensions
说明: 性能不依赖输入数值，用 random 输入（shape 与 golden 15 用例一致即可做性能对比）。
"""
import sys, os, glob, csv
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

# 15 用例 shape config（与 golden_megamoe.py CASES 一致；nmtr=num_max_tokens_per_rank, ih=intermediate_hidden, tok=num_tokens）
CASES = [
    dict(name="coded_L0_topk1_fixed_ws2",  world_size=2, num_experts=2,   hidden=1024, ih=1024, topk=1,  tok=32,  nmtr=64),
    dict(name="coded_L1_topk1_unique_ws2", world_size=2, num_experts=4,   hidden=1024, ih=1024, topk=1,  tok=64,  nmtr=128),
    dict(name="coded_L1_topk1_unique_ws4", world_size=4, num_experts=4,   hidden=2048, ih=1024, topk=1,  tok=128, nmtr=256),
    dict(name="coded_L2_topk2_equal_ws2",  world_size=2, num_experts=8,   hidden=2048, ih=1024, topk=2,  tok=128, nmtr=256),
    dict(name="coded_L2_topk4_equal_ws4",  world_size=4, num_experts=8,   hidden=2048, ih=1024, topk=4,  tok=128, nmtr=256),
    dict(name="coded_L2_topk4_equal_ws8",  world_size=8, num_experts=16,  hidden=2048, ih=1024, topk=4,  tok=256, nmtr=512),
    dict(name="coded_L3_topk8_randw_ws2",  world_size=2, num_experts=8,   hidden=2048, ih=1024, topk=8,  tok=256, nmtr=512),
    dict(name="coded_L3_topk8_randw_ws8",  world_size=8, num_experts=16,  hidden=4096, ih=1536, topk=8,  tok=256, nmtr=1024),
    dict(name="probe_ex_like",             world_size=2, num_experts=8,   hidden=2048, ih=1024, topk=6,  tok=256, nmtr=1024),
    dict(name="dsv3_ws8",                  world_size=8, num_experts=256, hidden=7168, ih=2048, topk=8,  tok=512, nmtr=1024),
    dict(name="dsv2_ws8",                  world_size=8, num_experts=160, hidden=5120, ih=1536, topk=6,  tok=512, nmtr=1024),
    dict(name="qwen2_57b_ws8",             world_size=8, num_experts=64,  hidden=3584, ih=2560, topk=8,  tok=512, nmtr=1024),
    dict(name="qwen3_235b_ws8",            world_size=8, num_experts=128, hidden=4096, ih=1536, topk=8,  tok=1024, nmtr=2048),
    dict(name="boundary_maxdims_ws4",      world_size=4, num_experts=32,  hidden=8192, ih=3072, topk=8,  tok=512, nmtr=1024),
    dict(name="boundary_topk16_ws8",       world_size=8, num_experts=64,  hidden=4096, ih=2048, topk=16, tok=256, nmtr=512),
]
PORT_BASE = 50700
G = {}  # 跨进程共享：case / inputs / profile_dir（fork 继承）


def make_inputs(c):
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    h, ih, tok, topk = c["hidden"], c["ih"], c["tok"], c["topk"]
    per_rank = []
    for _ in range(ws):
        x = torch.empty(tok, h, dtype=torch.bfloat16).uniform_(-0.1, 0.1)
        ids = torch.stack([torch.randperm(exp)[:topk] for _ in range(tok)]).to(torch.int32)
        w = torch.rand(tok, topk, dtype=torch.bfloat16)
        l1 = [torch.empty(h, 2 * ih, dtype=torch.bfloat16).uniform_(-0.1, 0.1) for _ in range(epr)]
        l2 = [torch.empty(ih, h, dtype=torch.bfloat16).uniform_(-0.1, 0.1) for _ in range(epr)]
        per_rank.append({"x": x, "ids": ids, "w": w, "l1": l1, "l2": l2})
    return per_rank


def run_one_rank(rank):
    c, inputs, profile_dir = G["case"], G["inputs"], G["profile_dir"]
    from cann_ops_transformer.ops import mega_moe, get_symm_buffer_for_mega_moe
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=c["world_size"],
                            init_method=f"tcp://127.0.0.1:{c['port']}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(c["world_size"])))
    dist.all_reduce(torch.ones(1, device="npu"), group=ep_group)
    torch.npu.synchronize()

    d = inputs[rank]
    x = d["x"].npu(); ids = d["ids"].npu(); w = d["w"].npu()
    l1 = [t.npu() for t in d["l1"]]; l2 = [t.npu() for t in d["l2"]]
    epr = c["num_experts"] // c["world_size"]
    max_recv = c["nmtr"] * c["world_size"] * min(c["topk"], epr)
    sym_buffer = get_symm_buffer_for_mega_moe(
        ep_group, num_experts=c["num_experts"], num_max_tokens_per_rank=c["nmtr"],
        num_topk=c["topk"], hidden=c["hidden"], intermediate_hidden=c["ih"],
        max_recv_token_num=max_recv, dispatch_quant_mode=0, dispatch_quant_out_dtype=None)
    _ = mega_moe(x, ids, w, l1, l2, sym_buffer); torch.npu.synchronize()  # warmup，避免首次 gen_task 编译进 profiling

    if rank == 0:
        ec = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=ec,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_dir))
        profiler.start()
    mega_moe(x, ids, w, l1, l2, sym_buffer)
    torch.npu.synchronize()
    if rank == 0:
        profiler.stop()
    dist.destroy_process_group()


def run_case(case_name, port):
    c = next(x for x in CASES if x["name"] == case_name)
    c = dict(c); c["port"] = port
    G["case"] = c
    G["inputs"] = make_inputs(c)
    G["profile_dir"] = f"/tmp/perf_case_{case_name}"
    os.system(f"rm -rf {G['profile_dir']}"); os.makedirs(G["profile_dir"], exist_ok=True)
    mp.set_start_method("fork", force=True)
    procs = [Process(target=run_one_rank, args=(r,)) for r in range(c["world_size"])]
    for p in procs: p.start()
    for p in procs: p.join()
    for kd in glob.glob(f"{G['profile_dir']}/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"):
        with open(kd) as f:
            for row in csv.DictReader(f):
                if "MegaMoe" in row.get("Name", ""):
                    return float(row["Duration(us)"])
    return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        for i, c in enumerate(CASES):
            dur = run_case(c["name"], PORT_BASE + i)
            print(f"{c['name']}\t{dur}", flush=True)
    else:
        name = sys.argv[1] if len(sys.argv) > 1 else CASES[0]["name"]
        print(f"{name}\t{run_case(name, PORT_BASE)}")
