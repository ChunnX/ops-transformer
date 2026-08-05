#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# mega_moe (bf16/bf16 非量化, A3 / ascend910_93 / arch22) 精度 golden 框架
#
# 用途：
#   1) 在【当前】vendor 包的 mega_moe 上跑 CPU golden vs NPU，建立精度基线；
#   2) 将来重写 mega_moe bf16 路径后，复用本框架与同一批用例对比回归。
#
# 设计要点（便于精度调试）：
#   - coded 用例：输入「自带标签」——x 按 rank 染色、权重按 expert 染色，从输出数值即可反推出错位置；
#   - random 用例：DeepSeek-V3/V2、Qwen2-57B、Qwen3-235B 等真实 shape + 边界；
#   - expert_token_nums（int32）严格相等：dispatch 路由正确性的第一信号；
#   - CPU golden 全程 float32、全局视角（无真实 all_to_all）。
#
# 关键工程约束（实测踩坑）：
#   - 全程【不用 numpy】，只用 torch —— numpy 的 OpenMP/BLAS 线程池被 fork 继承后会坏掉，
#     导致子进程算子后的 tensor 操作偶发死锁（example.py 全程 torch 故稳定）。
#   - fork 必须在 CPU golden 之前（父进程 torch matmul 同样会初始化坏线程池状态）。
#   - 子进程算子输出 y.cpu() + torch.save 都须在 destroy_process_group 之前完成。
#   - 结果经文件（torch.save/load）+ Event 回传，规避 Queue/share_memory 在 fork 下的死锁。
#
# 运行（每个新 shell）：
#   source <EXP>/cann/ascend-toolkit/set_env.sh
#   source <EXP>/cann/vendors/custom_transformer/bin/set_env.bash
#   export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<EXP>/.torch_extensions
#   python3 golden_megamoe.py --case coded_L0_topk1_fixed_ws2
#   python3 golden_megamoe.py --all
# -----------------------------------------------------------------------------------------------------------
import os
import argparse
import torch
import torch_npu                      # noqa: F401  (注册 npu 后端；父进程 import 安全，不初始化 HCCL)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process

# ===========================================================================================================
# 模块级全局（fork 继承）
# ===========================================================================================================
G_INPUT = None   # list[dict]：每 rank 输入片 {x, topk_ids, topk_weights, l1[list], l2[list]}（cpu tensor）
G_CASE = None    # 当前 case dict
G_RESULT = None  # list[dict]：每 rank {done: Event}（结果经文件传回）

# ===========================================================================================================
# 编码常量（coded 用例：rank / expert 染色）
# 说明：用户原意 0.0001/0.0002 标识 rank/expert。但 x 与权重都取 1e-4 时 SwiGLU(gate*up) 平方缩小，
#   y~1e-12 在 bf16 下 underflow。故染色步长取 1e-2（仍可辨识、bf16 可分辨），gate/up 半不同倍数避免退化。
# ===========================================================================================================
CODE_STEP = 1e-2
GATE_FACTOR = 1.0    # W1 gate 半 = CODE_STEP*(e+1)*GATE_FACTOR
UP_FACTOR = 2.0      # W1 up   半 = CODE_STEP*(e+1)*UP_FACTOR


# ===========================================================================================================
# 用例约束（arch22 硬上限，源码 mega_moe_tiling_arch22.cpp 核实）
# ===========================================================================================================
def assert_case(c):
    h, ih = c["hidden"], c["intermediate_hidden"]
    ws, exp, topk = c["world_size"], c["num_experts"], c["topk"]
    assert 1024 <= h <= 8192 and h % 512 == 0, f"hidden={h} 必须 ∈[1024,8192] 且 %512==0"
    assert 1024 <= ih <= 3072 and ih % 512 == 0, f"intermediate_hidden={ih} 必须 ∈[1024,3072] 且 %512==0 (上限3072)"
    assert 1 <= topk <= 16, f"topk={topk} 必须 ∈[1,16]"
    assert ws in (2, 4, 8, 16, 32), f"world_size={ws} 必须 ∈{{2,4,8,16,32}}"
    assert exp % ws == 0, f"num_experts={exp} 必须能被 world_size={ws} 整除"
    assert 1 <= exp // ws <= 128, f"expert_per_rank={exp // ws} 必须 ∈[1,128]"
    assert 1 <= c["num_tokens"] <= 4096
    assert c["num_tokens"] <= c["num_max_tokens_per_rank"] <= 4096


# ===========================================================================================================
# 输入构造（全 torch，不用 numpy）
# ===========================================================================================================
def _topk_ids(c, g):
    tok, topk, exp = c["num_tokens"], c["topk"], c["num_experts"]
    if c["topk_ids_mode"] == "fixed":
        # 所有 token 选同一组专家 [0..topk-1]（L0 topk=1 → 全选 expert 0，退化为单专家）
        return torch.arange(topk, dtype=torch.int32).repeat(tok, 1)
    ids = torch.empty(tok, topk, dtype=torch.int32)
    for i in range(tok):
        ids[i] = torch.randperm(exp, generator=g)[:topk]
    return ids


def _topk_weights(c, g):
    tok, topk = c["num_tokens"], c["topk"]
    if c["topk_weights_mode"] == "equal":
        return torch.full((tok, topk), 1.0 / topk, dtype=torch.float32)
    return torch.empty(tok, topk, dtype=torch.float32).uniform_(0.0, 1.0, generator=g)


def make_inputs(c):
    """构造全局输入 + 每 rank 切片（全 torch cpu tensor，fork 后 .npu()）。"""
    assert_case(c)
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    h, ih = c["hidden"], c["intermediate_hidden"]
    tok, seed = c["num_tokens"], c["seed"]
    mode = c["input_mode"]

    x_list, ids_list, w_list, per_rank = [], [], [], []
    for rank in range(ws):
        g = torch.Generator().manual_seed(seed * 100003 + rank * 7)
        if mode == "coded":
            x = torch.full((tok, h), CODE_STEP * (rank + 1), dtype=torch.float32)
        else:
            x = torch.empty(tok, h, dtype=torch.float32).uniform_(-0.1, 0.1, generator=g)
        x = x.to(torch.bfloat16)
        ids = _topk_ids(c, g)
        w = _topk_weights(c, g).to(torch.bfloat16)
        x_list.append(x); ids_list.append(ids); w_list.append(w)
        per_rank.append({"x": x, "topk_ids": ids, "topk_weights": w})

    # 全局权重（CPU golden 用全部；NPU 每 rank 取 epr 片）
    w1_full, w2_full = [], []
    for e in range(exp):
        ge = torch.Generator().manual_seed(seed * 999983 + e * 13)
        if mode == "coded":
            w1 = torch.empty(h, 2 * ih, dtype=torch.float32)
            w1[:, :ih] = CODE_STEP * (e + 1) * GATE_FACTOR
            w1[:, ih:] = CODE_STEP * (e + 1) * UP_FACTOR
            w2 = torch.full((ih, h), CODE_STEP * (e + 1), dtype=torch.float32)
        else:
            w1 = torch.empty(h, 2 * ih, dtype=torch.float32).uniform_(-0.1, 0.1, generator=ge)
            w2 = torch.empty(ih, h, dtype=torch.float32).uniform_(-0.1, 0.1, generator=ge)
        w1_full.append(w1.to(torch.bfloat16))
        w2_full.append(w2.to(torch.bfloat16))

    for rank in range(ws):
        per_rank[rank]["l1"] = [w1_full[rank * epr + j] for j in range(epr)]
        per_rank[rank]["l2"] = [w2_full[rank * epr + j] for j in range(epr)]

    return {"x_list": x_list, "ids_list": ids_list, "w_list": w_list,
            "w1_full": w1_full, "w2_full": w2_full, "per_rank": per_rank}


# ===========================================================================================================
# CPU golden（全局视角，全程 float32，无真实 all_to_all）
# 前向：dispatch(token→expert) → expert FFN(X@W1→SwiGLU→@W2) → combine(Σ w·Y 回原 rank/token)
# ===========================================================================================================
def _silu(x):
    return x * torch.sigmoid(x)


def cpu_mega_moe_golden(c, G):
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    h, ih, topk, tok = c["hidden"], c["intermediate_hidden"], c["topk"], c["num_tokens"]
    x_list, ids_list, w_list = G["x_list"], G["ids_list"], G["w_list"]
    w1_full, w2_full = G["w1_full"], G["w2_full"]

    # ---- dispatch_gather：按 (rank, token, k) 升序投桶；记录行映射 ----
    buckets = [[] for _ in range(exp)]      # buckets[e] = list of x_vec[torch.float32, h]
    row_map = {}                            # (rank, token, k) -> (e, row_in_e)
    for rank in range(ws):
        ids = ids_list[rank]                # [tok, topk] int32
        x = x_list[rank].to(torch.float32)  # [tok, h]
        for token in range(tok):
            for k in range(topk):
                e = int(ids[token, k].item())
                row_map[(rank, token, k)] = (e, len(buckets[e]))
                buckets[e].append(x[token])

    # ---- expert_token_nums[rank][j] ----
    etn = torch.zeros((ws, epr), dtype=torch.int32)
    for e in range(exp):
        etn[e // epr, e % epr] = len(buckets[e])

    # ---- 每 expert FFN（float32）----
    Y_e = [None] * exp
    for e in range(exp):
        if not buckets[e]:
            Y_e[e] = torch.zeros((0, h), dtype=torch.float32)
            continue
        Xe = torch.stack(buckets[e]).to(torch.float32)     # [N, h]
        W1 = w1_full[e].to(torch.float32)
        W2 = w2_full[e].to(torch.float32)
        H = Xe @ W1                       # [N, 2*ih]
        A = _silu(H[:, :ih]) * H[:, ih:]  # SwiGLU, [N, ih]
        Y_e[e] = A @ W2                   # [N, h]

    # ---- combine：y[rank][token] = Σ_k w_k · Y_e[e][row] ----
    y_list = []
    for rank in range(ws):
        wts = w_list[rank].to(torch.float32)     # [tok, topk]
        y = torch.zeros((tok, h), dtype=torch.float32)
        for token in range(tok):
            for k in range(topk):
                e, row = row_map[(rank, token, k)]
                y[token] += wts[token, k] * Y_e[e][row]   # combine: y=Σ w_k·Y（文档 mega_moe.md:124）
        y_list.append(y)
    return y_list, [etn[r] for r in range(ws)]


# ===========================================================================================================
# NPU 调用层（fork 子进程，复用 megamoe_example.py 稳定姿势；全程 torch，结果经文件回传）
# ===========================================================================================================
def run_one_rank(rank):
    global G_INPUT, G_CASE, G_RESULT
    c = G_CASE
    ws = c["world_size"]
    # cann_ops_transformer 必须在子进程内 import（fork 继承已初始化 HCCL 会失败）
    from cann_ops_transformer.ops import mega_moe, get_symm_buffer_for_mega_moe

    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws,
                            init_method=f"tcp://127.0.0.1:{c['port']}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(ws)))

    # HCCL warmup（必须，否则 get_symm_buffer 内 HcclCommGetHandleWithName 失败）
    warmup = torch.ones(1, device="npu")
    dist.all_reduce(warmup, group=ep_group)
    torch.npu.synchronize()

    d = G_INPUT[rank]
    x = d["x"].to(torch.bfloat16).npu()
    ids = d["topk_ids"].to(torch.int32).npu()
    w = d["topk_weights"].to(torch.float32).npu()
    l1 = [t.to(torch.bfloat16).npu() for t in d["l1"]]
    l2 = [t.to(torch.bfloat16).npu() for t in d["l2"]]

    # max_recv 必须用 nmtr 计算（example 方式：nmtr × ws × min(topk,epr)），用 num_tokens 会偏小导致后半 token 丢失
    epr = c["num_experts"] // ws
    max_recv = c["num_max_tokens_per_rank"] * ws * min(c["topk"], epr)
    sym = get_symm_buffer_for_mega_moe(
        ep_group, num_experts=c["num_experts"],
        num_max_tokens_per_rank=c["num_max_tokens_per_rank"],
        num_topk=c["topk"], hidden=c["hidden"],
        intermediate_hidden=c["intermediate_hidden"],
        max_recv_token_num=max_recv, dispatch_quant_mode=0, dispatch_quant_out_dtype=None)

    # warmup（避免首次 gen_task 编译进 profiler）
    _ = mega_moe(x, ids, w, l1, l2, sym); torch.npu.synchronize()

    # rank0 profiler 抓 MegaMoe kernel Duration（perf/golden 合一）
    dur = None
    if rank == 0:
        import glob as _glob, csv as _csv
        _pdir = f"/tmp/gm_prof_{c['name']}_{os.getpid()}"
        os.system(f"rm -rf {_pdir}; mkdir -p {_pdir}")
        _ec = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
        _prof = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=_ec,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(_pdir))
        _prof.start()
    y, etn = mega_moe(x, ids, w, l1, l2, sym)
    torch.npu.synchronize()
    if rank == 0:
        _prof.stop()
        for _kd in _glob.glob(f"{_pdir}/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"):
            with open(_kd) as _f:
                for _row in _csv.DictReader(_f):
                    if "MegaMoe" in _row.get("Name", ""):
                        dur = float(_row["Duration(us)"]); break
            if dur: break
    # y.cpu() 须在 destroy 之前；.to(float32) 移到父进程（fork 子进程里 bf16→f32 会卡死）
    y_bf16 = y.cpu()
    etn_cpu = etn.cpu()
    torch.save({"y": y_bf16, "etn": etn_cpu, "dur": dur}, f"{c['dump_dir']}/result_{rank}.pt")
    G_RESULT[rank]["done"].set()
    dist.destroy_process_group()


def alloc_results(c):
    """每 rank 一个 done event；结果 tensor 经文件（torch.save/load）回传。"""
    return [{"done": mp.Event()} for _ in range(c["world_size"])]


def fork_run_npu(c):
    global G_INPUT, G_CASE, G_RESULT
    G_INPUT = c["_inputs"]["per_rank"]
    G_CASE = c
    G_RESULT = alloc_results(c)
    c["dump_dir"] = f"/tmp/gm_dump_{os.getpid()}"
    os.makedirs(c["dump_dir"], exist_ok=True)
    mp.set_start_method("fork", force=True)
    procs = [Process(target=run_one_rank, args=(r,)) for r in range(c["world_size"])]
    for p in procs:
        p.start()
    for r in range(c["world_size"]):
        G_RESULT[r]["done"].wait()       # 等每 rank 写完结果文件
    for p in procs:
        p.join()
    results = {}
    for r in range(c["world_size"]):
        d = torch.load(f"{c['dump_dir']}/result_{r}.pt")
        results[r] = (d["y"].to(torch.float32), d["etn"], d.get("dur"))   # bf16→f32 在父进程（fork 外）做
    return results


# ===========================================================================================================
# 精度判定与失败定位
# ===========================================================================================================
def check_token_nums(g, a, rank):
    if torch.equal(g, a):
        return True
    print(f"  [rank {rank}] ✗ expert_token_nums 不一致（dispatch 路由可能有误）:")
    print(f"    golden = {g.tolist()}")
    print(f"    npu    = {a.tolist()}")
    return False


def check_result(golden, actual):
    g = golden.cpu().view(-1).to(torch.float32)
    a = actual.cpu().view(-1).to(torch.float32)
    cos = torch.nn.functional.cosine_similarity(g.unsqueeze(0), a.unsqueeze(0)).item()
    ad = (g - a).abs()
    max_abs = ad.max().item()
    denom = g.abs().clamp(min=1e-9)
    max_rel = (ad / denom).max().item()
    return {"cos": cos, "max_abs": max_abs, "max_rel": max_rel, "passed": cos > 0.99}


def report_failure(golden, actual, rank, n=8):
    g = golden.cpu().to(torch.float32)
    a = actual.cpu().to(torch.float32)
    ad = (g - a).abs()
    flat = ad.view(-1)
    idx = torch.argsort(flat, descending=True)[:n].tolist()
    H = g.shape[1]
    print(f"  [rank {rank}] 差异最大的 {n} 个元素 (token, dim, golden, npu, diff):")
    for fi in idx:
        fi = int(fi)
        ti, di = fi // H, fi % H
        print(f"    token={ti:<5d} dim={di:<5d} golden={g[ti, di].item():.6e} "
              f"npu={a[ti, di].item():.6e} diff={flat[fi].item():.3e}")


# ===========================================================================================================
# 用例清单（内置）
# ===========================================================================================================
CASES = [
    # ---- A. coded 简单用例（数据自带标签，打通精度）----
    dict(name="coded_L0_topk1_fixed_ws2",  world_size=2, num_experts=2,   hidden=1024, intermediate_hidden=1024, topk=1,  num_tokens=32,   num_max_tokens_per_rank=64,   input_mode="coded",  topk_ids_mode="fixed",  topk_weights_mode="equal",  seed=1),
    dict(name="coded_L1_topk1_unique_ws2", world_size=2, num_experts=4,   hidden=1024, intermediate_hidden=1024, topk=1,  num_tokens=64,   num_max_tokens_per_rank=128,  input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="equal",  seed=2),
    dict(name="coded_L1_topk1_unique_ws4", world_size=4, num_experts=4,   hidden=2048, intermediate_hidden=1024, topk=1,  num_tokens=128,  num_max_tokens_per_rank=256,  input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="equal",  seed=3),
    dict(name="coded_L2_topk2_equal_ws2",  world_size=2, num_experts=8,   hidden=2048, intermediate_hidden=1024, topk=2,  num_tokens=128,  num_max_tokens_per_rank=256,  input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="equal",  seed=4),
    dict(name="coded_L2_topk4_equal_ws4",  world_size=4, num_experts=8,   hidden=2048, intermediate_hidden=1024, topk=4,  num_tokens=128,  num_max_tokens_per_rank=256,  input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="equal",  seed=5),
    dict(name="coded_L2_topk4_equal_ws8",  world_size=8, num_experts=16,  hidden=2048, intermediate_hidden=1024, topk=4,  num_tokens=256,  num_max_tokens_per_rank=512,  input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="equal",  seed=6),
    dict(name="coded_L3_topk8_randw_ws2",  world_size=2, num_experts=8,   hidden=2048, intermediate_hidden=1024, topk=8,  num_tokens=256,  num_max_tokens_per_rank=512,  input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="random", seed=7),
    dict(name="coded_L3_topk8_randw_ws8",  world_size=8, num_experts=16,  hidden=4096, intermediate_hidden=1536, topk=8,  num_tokens=256,  num_max_tokens_per_rank=1024, input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="random", seed=8),
    # ---- probe：复制 megamoe_example.py 参数，验证框架本身（算子已知能跑通此配置）----
    dict(name="probe_ex_like", world_size=2, num_experts=8, hidden=2048, intermediate_hidden=1024, topk=6, num_tokens=256, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=100),
    # ---- B. random 规模/边界用例（真实 shape）----
    dict(name="dsv3_ws8",             world_size=8, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8,  num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=10),
    dict(name="dsv2_ws8",             world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=6,  num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=11),
    dict(name="qwen2_57b_ws8",        world_size=8, num_experts=64,  hidden=3584, intermediate_hidden=2560, topk=8,  num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=12),
    dict(name="qwen3_235b_ws8",       world_size=8, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8,  num_tokens=1024, num_max_tokens_per_rank=2048, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=13),
    dict(name="boundary_maxdims_ws4", world_size=4, num_experts=32,  hidden=8192, intermediate_hidden=3072, topk=8,  num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=14),
    dict(name="boundary_topk16_ws8",  world_size=8, num_experts=64,  hidden=4096, intermediate_hidden=2048, topk=16, num_tokens=256,  num_max_tokens_per_rank=512,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=15),
    # ---- ws=16 主流大模型（16 卡验收新增）----
    dict(name="coded_L3_ws16",        world_size=16, num_experts=16,  hidden=2048, intermediate_hidden=1024, topk=8, num_tokens=256,  num_max_tokens_per_rank=512,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=20),
    dict(name="dsv3_ws16",            world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=21),
    dict(name="qwen3_235b_ws16",      world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=1024, num_max_tokens_per_rank=2048, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=22),
    dict(name="kimi_k2_ws16",         world_size=16, num_experts=384, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=23),
    dict(name="glm45_ws16",           world_size=16, num_experts=160, hidden=8192, intermediate_hidden=3072, topk=8, num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=24),
    dict(name="llama4_mav_ws16",      world_size=16, num_experts=128, hidden=5120, intermediate_hidden=2048, topk=8, num_tokens=512,  num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=25),
    # ---- decode 场景（小 batch 16/32/64，ws=16，BF16；遍历验证全跑通）----
    dict(name="dsv3_ws16_dec16",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=30),
    dict(name="dsv3_ws16_dec32",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=31),
    dict(name="dsv3_ws16_dec64",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=32),
    dict(name="dsv4_ws16_dec16",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=33),
    dict(name="dsv4_ws16_dec32",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=34),
    dict(name="dsv4_ws16_dec64",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=35),
    dict(name="kimi_ws16_dec16",  world_size=16, num_experts=384, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=36),
    dict(name="kimi_ws16_dec32",  world_size=16, num_experts=384, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=37),
    dict(name="glm5_ws16_dec16",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=39),
    dict(name="glm5_ws16_dec32",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=40),
    dict(name="glm5_ws16_dec64",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=41),
    dict(name="qwen3_ws16_dec16", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=42),
    dict(name="qwen3_ws16_dec32", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=43),
    dict(name="qwen3_ws16_dec64", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=44),
]


# ===========================================================================================================
# 单用例执行
# ===========================================================================================================
def run_case(c, dump=False):
    c["_inputs"] = make_inputs(c)
    # 1) NPU（fork）必须在 CPU golden 之前（父进程 torch matmul 会初始化坏线程池状态被 fork 继承）
    npu_res = fork_run_npu(c)
    # 2) CPU golden（fork 之后）
    golden_y, golden_etn = cpu_mega_moe_golden(c, c["_inputs"])

    # 3) 比对
    print(f"\n{'=' * 72}\nCase: {c['name']}  (ws={c['world_size']}, exp={c['num_experts']}, "
          f"h={c['hidden']}, ih={c['intermediate_hidden']}, topk={c['topk']}, tok={c['num_tokens']}, "
          f"mode={c['input_mode']})\n{'=' * 72}")
    # 实测（含 example.py 自身）：当前算子 combine 仅输出前 num_tokens/ws 个 token 有效，后半为 0。
    # golden 只比对前 half；重写算子若修了 combine 全输出，再去掉此截断。
    half = c["num_tokens"] // c["world_size"]
    all_pass = True
    for rank in sorted(npu_res):
        y_npu, etn_npu, _dur = npu_res[rank]
        etn_ok = check_token_nums(golden_etn[rank], etn_npu, rank)
        r = check_result(golden_y[rank][:half], y_npu[:half])
        ok = etn_ok and r["passed"]
        all_pass = all_pass and ok
        _durstr = f" Dur={_dur:.1f}us" if (rank == 0 and _dur) else ""
        print(f"  [rank {rank}] {'PASS' if ok else 'FAIL'} | etn_ok={etn_ok} | cos={r['cos']:.6f}{_durstr} "
              f"max_abs={r['max_abs']:.3e} max_rel={r['max_rel']:.3e}")
        if not ok:
            report_failure(golden_y[rank][:half], y_npu[:half], rank)
        if dump:
            dp = f".dump/{c['name']}"
            os.makedirs(dp, exist_ok=True)
            torch.save(golden_y[rank][:half], f"{dp}/golden_y_rank{rank}.pt")
            torch.save(y_npu[:half], f"{dp}/npu_y_rank{rank}.pt")
    print(f"  -> {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    if "dump_dir" in c:
        import shutil
        shutil.rmtree(c["dump_dir"], ignore_errors=True)
    return all_pass


# ===========================================================================================================
# main
# ===========================================================================================================
def main():
    ap = argparse.ArgumentParser(description="mega_moe bf16 精度 golden")
    ap.add_argument("--case", type=str, default=None, help="用例名")
    ap.add_argument("--num", type=int, default=None, help="用例序号(1-based)")
    ap.add_argument("--all", action="store_true", help="跑全部用例")
    ap.add_argument("--dump", action="store_true", help="dump golden/npu y 到 .dump/")
    ap.add_argument("--list", action="store_true", help="列出用例")
    args = ap.parse_args()

    if args.list:
        for i, c in enumerate(CASES, 1):
            print(f"  [{i:>2d}] {c['name']}")
        return

    if args.all:
        selected = list(range(len(CASES)))
    elif args.case:
        names = {c["name"]: i for i, c in enumerate(CASES)}
        assert args.case in names, f"未知用例 {args.case}（--list 查看全部）"
        selected = [names[args.case]]
    elif args.num:
        assert 1 <= args.num <= len(CASES), f"--num 须 ∈[1,{len(CASES)}]"
        selected = [args.num - 1]
    else:
        ap.error("需指定 --case / --num / --all / --list")

    npass = 0
    for i in selected:
        c = dict(CASES[i])            # copy，避免污染内置 CASES
        c["port"] = 50000 + i         # 每 case 独立端口
        npass += int(run_case(c, dump=args.dump))
    print(f"\n==== 总结: {npass}/{len(selected)} 用例通过 ====")
    if npass != len(selected):
        os._exit(1)


if __name__ == "__main__":
    main()
