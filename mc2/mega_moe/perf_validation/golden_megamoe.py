#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# mega_moe (A3 / ascend910_93 / arch22) 精度 golden 框架（BF16 + W8A8 + W4A8）
#
# 用途：
#   1) 在【当前】vendor 包的 mega_moe 上跑 NPU 小算子级联 golden vs 融合算子，建立精度基线；
#   2) 将来重写 mega_moe 后，复用本框架与同一批用例对比回归。
#
# 精度参考：**NPU 小算子级联**（本文件 npu_mega_moe_golden）——把 mega_moe 拆成 dispatch/GMM1/
#   SwiGLU/GMM2/combine 五段，全部用 torch_npu 小算子在 NPU 上算，与融合算子逐 rank 比 cos + etn。
#   CPU golden（cpu_mega_moe_golden）仅作「参考的参考」保留，--cpu-golden 才启用（很慢）。
#
# 设计要点（便于精度调试）：
#   - coded 用例：输入「自带标签」——x 按 rank 染色、权重按 expert 染色，从输出数值即可反推出错位置；
#   - random 用例：DeepSeek-V3/V2、Qwen2-57B、Qwen3-235B 等真实 shape + 边界；
#   - expert_token_nums（int32）严格相等：dispatch 路由正确性的第一信号。
#
# 进程模型：**spawn 常驻 worker**（按 world_size 分池，跨 case 复用）
#   旧模型每 case fork ws 个子进程 + 1 个 golden 进程，62 个 case 累计 730 进程、
#   668 次 HCCL 初始化，而被测 kernel 只有 76us~10ms —— 全量跑 140 分钟里 99.99% 是重复初始化。
#   现模型：set_device / init_process_group / warmup 全程只做 1 次，每 case 只重建 SymmBuffer。
#
# 关键工程约束（实测踩坑）：
#   - 全程【不用 numpy】，只用 torch —— numpy 的 OpenMP/BLAS 线程池在多进程下会坏掉。
#   - 必须 spawn 不能 fork：fork 继承父进程已初始化的 NPU 状态会直接报
#     "Cannot re-initialize NPU in forked subprocess"（master_npugolden_all.log 实录）。
#   - topk_weights 必须传 float32，传 bf16 会让随机权重精度崩塌（cos 0.76）。
#   - SymmBuffer 每 case 必须 destroy，否则常驻 worker 下 HCCL 显存累积必 OOM。
#   - 权重种子只与 expert id 有关（与 rank/ws 无关），这是「worker 各造自己那份」的正确性前提。
#
# 运行（每个新 shell）：
#   source <EXP>/cann/ascend-toolkit/set_env.sh
#   source <EXP>/cann-vendors/master/vendors/custom_transformer/bin/set_env.bash
#   export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<EXP>/.torch_extensions
#   python3 golden_megamoe.py --case coded_L0_topk1_fixed_ws2
#   python3 golden_megamoe.py --all
# -----------------------------------------------------------------------------------------------------------
import os
import time
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor
import torch
import torch_npu                      # noqa: F401  (注册 npu 后端；父进程 import 安全，不初始化 HCCL)
import torch.distributed as dist
import torch.multiprocessing as mp

# ===========================================================================================================
# 模块级全局
# ===========================================================================================================
WORLD = 0        # worker 进程的 world_size（spawn 时由 _worker_entry 显式设入，不靠继承）

# 参考实现：NPU 小算子级联（本文件 npu_mega_moe_golden）。CPU 版仅作「参考的参考」，两者判定等价性
# 已由三方对比验证（verify_3way.py）。--cpu-golden 可切回（很慢，仅排障用）。
USE_CPU_GOLDEN = False

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


# ===========================================================================================================
# 量化辅助（w8a8 / w4a8）
# 语义对齐官方 mega_moe.md 示例 + kernel（A3 arch22）：
#   - weight quant：scale = max(|W|)/qmax（int8: qmax=127；int4: qmax=7 对称量化），Wq = round(W/scale) clamp
#   - UINT64 scale 打包：fp32 位模式放低 32 位（sf.view(int32).to(int64) 符号扩展，正 scale 高 32 位=0）
#   - INT4 权重：每 2 个 int4 装 1 字节（(+8 偏移) → uint8 → 每 2 个组合 → view int8），NZ cast 后 view int32
#   - W4A8 必配 bias1/bias2（FP32）：bias = 8·(1ᵀ·(W_int4·s_w)) —— MSD 拆分常数项补偿（反量化域）
# ===========================================================================================================
def quantize_w(w_fp32, bits):
    """逐输出通道量化权重 [k,n] fp32 → (w_int, scale[n] fp32)。int4 对称量化 [-7,7]，int8 [-127,127]。"""
    qmax = 127 if bits == 8 else 7
    scale = w_fp32.abs().amax(dim=0) / qmax          # [n]
    scale = scale.clamp(min=1e-8)                    # 防全零列除零（对齐文档 1e-8）
    wq = torch.round(w_fp32 / scale.unsqueeze(0)).clamp(-qmax, qmax)
    return wq.to(torch.int8), scale


def pack_u64(scale_f32):
    """fp32 scale → UINT64（官方方式：fp32 位模式放低 32 位，符号扩展后高 32 位=0）。"""
    return scale_f32.float().contiguous().view(torch.int32).to(torch.int64).view(torch.uint64)


def pack_int4_to_int8(x):
    """int8（值 -7..7）每 2 个打包进 1 字节（官方方式：+8 偏移 → 每 2 个组合）。"""
    x = (x + 8).to(torch.uint8).reshape(x.shape[0], -1, 2)
    return ((x[..., 1] << 4) | x[..., 0]).view(torch.int8)


def bias_msd(w_int4, scale):
    """W4A8 的 MSD 常数项补偿：bias = 8·(1ᵀ·(W_int4·s_w))，[n] fp32。"""
    return (8.0 * (w_int4.float() * scale.unsqueeze(0)).sum(dim=0)).to(torch.float32)


def dispatch_quant(x):
    """kernel `moe_v2_fullload_dynamic_quant` 的 per-token dynamic quant（对齐舍入序）：
       max = |x|.amax(dim=1) → v = (x/max) CAST_TRUNC 到 fp16 → q = round(v·127) → clamp[-128,127]
       返回 (q int8 [tok,h], s [tok] fp32 = max/127)。"""
    xf = x.float()
    maxv = xf.abs().amax(dim=1).clamp(min=1e-9)      # [tok]
    v16 = (xf / maxv.unsqueeze(1)).to(torch.float16).float()
    q = torch.round(v16 * 127.0).clamp(-128, 127).to(torch.int8)
    return q, (maxv / 127.0)


def requant(A):
    """SwiGLU 输出 → int8（kernel `block_epilogue_w4a8post_pertoken_swiglu` 的 requant）：
       s2 = max(|A|)/127，q2 = round(A/s2)（fp16 中间）→ clamp[-128,127]。"""
    maxv = A.abs().amax(dim=1).clamp(min=1e-9)       # [N]
    v16 = (A / maxv.unsqueeze(1)).to(torch.float16).float()
    q = torch.round(v16 * 127.0).clamp(-128, 127).to(torch.int8)
    return q, (maxv / 127.0)


def int4_split(q):
    """int8 激活 → int4 高低半（MSD）：xh = floor(q/16)（算术右移）、xl = (q&0xF)-8。"""
    return q.to(torch.int32) >> 4, ((q.to(torch.int32) & 0x0F) - 8).to(torch.int8)


def unpack_int4(w_packed):
    """打包 int8（每字节 2 个 int4）→ int4 值矩阵 [k, n] int8。
    官方 pack 为 `(x[...,1] << 4) | x[...,0]`（低 4 位=偶数列、高 4 位=奇数列，值均带 +8 偏移），
    故解包需 -8 还原；高位用无符号右移（先转 uint8 再 >>4）避免符号扩展污染。"""
    u = w_packed.to(torch.uint8).to(torch.int32)
    low = (u & 0x0F) - 8                      # 偶数列
    high = ((u >> 4) & 0x0F) - 8              # 奇数列
    return torch.stack([low, high], dim=-1).reshape(w_packed.shape[0], -1).to(torch.int8)


# 权重构造的线程配置。
#
# 【实测教训：线程越多越慢】dsv3_ws16（256 专家）在默认 torch.set_num_threads(320) 下：
#     32 线程 21.0s / 64 线程 33.6s / 128 线程 61.8s / 256 线程 80.6s
#   原因是**嵌套并行超订**：外层每个 Python 线程调 uniform_/matmul 时，torch 又各自
#   起一套 intra-op 线程池（本机 320 核 → 320 线程），32×320 个线程抢 320 个核。
#
# 【解法】构造期把 intra-op 降到 1，让并行度只由外层线程池提供：
#     16 线程 11.0s / 32 线程 11.2s / 64 线程 13.8s   —— 28.3s → 11.0s（2.6x）
#   超过 32 收益转负（uniform_ 是内存带宽受限，线程再多只是加剧 cache 争用）。
_BUILD_THREADS = 32


class _SingleThreadedTorch:
    """构造权重期间把 torch intra-op 线程降到 1，退出时恢复。

    只包住权重构造——NPU 级联 golden 仍需要正常的多线程（它在 NPU 上算，
    但 H2D 拷贝和 CPU 侧的 index/argsort 会用到）。
    """

    def __enter__(self):
        self._saved = torch.get_num_threads()
        torch.set_num_threads(1)
        return self

    def __exit__(self, *exc):
        torch.set_num_threads(self._saved)
        return False


def _build_experts_parallel(c, expert_ids):
    """并行构造指定的一批专家权重（消除嵌套并行超订）。"""
    ids = list(expert_ids)
    with _SingleThreadedTorch(), \
            ThreadPoolExecutor(max_workers=max(1, min(_BUILD_THREADS, len(ids)))) as ex:
        return list(ex.map(lambda e: _build_expert(c, e), ids))


def _build_expert(c, e):
    """构造第 e 个专家的权重，返回 (w1, w2, s1, s2, b1, b2)（量化项在 bf16 时为 None）。

    ⚠ 种子只与 e 有关（seed*999983 + e*13），**与 rank / world_size 无关**——这是
    「worker 各自只造自己那 epr 个专家」与「rank0 造全量给 golden」逐位一致的前提。
    """
    h, ih = c["hidden"], c["intermediate_hidden"]
    seed, mode = c["seed"], c["input_mode"]
    quant = c.get("quant", "bf16")
    bits = None if quant == "bf16" else (8 if quant == "w8a8" else 4)

    ge = torch.Generator().manual_seed(seed * 999983 + e * 13)
    if mode == "coded":
        w1 = torch.empty(h, 2 * ih, dtype=torch.float32)
        w1[:, :ih] = CODE_STEP * (e + 1) * GATE_FACTOR
        w1[:, ih:] = CODE_STEP * (e + 1) * UP_FACTOR
        w2 = torch.full((ih, h), CODE_STEP * (e + 1), dtype=torch.float32)
    else:
        w1 = torch.empty(h, 2 * ih, dtype=torch.float32).uniform_(-0.1, 0.1, generator=ge)
        w2 = torch.empty(ih, h, dtype=torch.float32).uniform_(-0.1, 0.1, generator=ge)
    if quant == "bf16":
        return w1.to(torch.bfloat16), w2.to(torch.bfloat16), None, None, None, None
    w1q, s1 = quantize_w(w1, bits)
    w2q, s2 = quantize_w(w2, bits)
    b1 = b2 = None
    if quant == "w4a8":
        b1 = bias_msd(w1q, s1)                 # bias 用打包前的 int4 值
        b2 = bias_msd(w2q, s2)
        w1q = pack_int4_to_int8(w1q)           # 每 2 个 int4 装 1 字节
        w2q = pack_int4_to_int8(w2q)
    return w1q, w2q, s1, s2, b1, b2


def _build_rank_io(c, rank):
    """构造 rank 的 (x, topk_ids, topk_weights)。

    ⚠ 三者共用同一个 generator 且**消费顺序不可改**：coded 模式下 x 走 torch.full 不消费
    随机数，random 模式下 x 会消费——顺序一变，ids/weights 全变。
    """
    tok, h = c["num_tokens"], c["hidden"]
    seed, mode = c["seed"], c["input_mode"]
    g = torch.Generator().manual_seed(seed * 100003 + rank * 7)
    if mode == "coded":
        x = torch.full((tok, h), CODE_STEP * (rank + 1), dtype=torch.float32)
    else:
        x = torch.empty(tok, h, dtype=torch.float32).uniform_(-0.1, 0.1, generator=g)
    return x.to(torch.bfloat16), _topk_ids(c, g), _topk_weights(c, g).to(torch.bfloat16)


def _pack_rank_weights(c, built):
    """把 _build_expert 的结果列表打包成算子入参格式（l1/l2 + 量化 scale/bias）。"""
    quant = c.get("quant", "bf16")
    d = {"l1": [b[0] for b in built], "l2": [b[1] for b in built]}
    if quant != "bf16":
        d["l1_sf"] = [pack_u64(b[2]) for b in built]
        d["l2_sf"] = [pack_u64(b[3]) for b in built]
        if quant == "w4a8":
            d["l1_bias"] = [b[4] for b in built]
            d["l2_bias"] = [b[5] for b in built]
    return d


def build_rank_slice(c, rank):
    """【worker 侧】只构造本 rank 需要的那份输入——epr 个专家，不碰其余专家。

    这是 spawn 模式相对 fork 的关键收益：原先父进程串行造 exp 个专家再切片，现在
    ws 个 worker 各造 exp/ws 个，天然并行。rank0 另需全量（golden 用），走 make_inputs。
    """
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    x, ids, w = _build_rank_io(c, rank)
    built = _build_experts_parallel(c, range(rank * epr, (rank + 1) * epr))
    return {"x": x, "topk_ids": ids, "topk_weights": w, **_pack_rank_weights(c, built)}


def make_inputs(c):
    """构造全局输入 + 每 rank 切片（全 torch cpu tensor）。

    【谁调用】rank0 worker——NPU 小算子级联 golden（本文件 npu_mega_moe_golden）需要
    全部 exp 个专家的权重和所有 rank 的 x/ids/weights 才能算出全局参考值。
    其余 rank 只调 build_rank_slice，不付全量构造的代价。

    【不做跨 case 权重缓存】用例集里每个 case 的 seed 都不同（dsv3_ws16_t16/t32/t64
    分别是 30/31/32），刻意让各 case 的随机权重互相独立、覆盖面更广。若按 shape 共享
    权重可省 ~30s/大 case，但 cos/max_abs 数值会与历史日志失去可比性——不划算。

    quant ∈ {"bf16","w8a8","w4a8"}：量化时权重转 int8/int4 + UINT64 scale（+W4A8 bias）。
    """
    assert_case(c)
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    quant = c.get("quant", "bf16")

    x_list, ids_list, w_list, per_rank = [], [], [], []
    for rank in range(ws):
        x, ids, w = _build_rank_io(c, rank)
        x_list.append(x); ids_list.append(ids); w_list.append(w)
        per_rank.append({"x": x, "topk_ids": ids, "topk_weights": w})

    # 全局权重（golden 用全部；NPU 每 rank 取 epr 片）。量化时逐 expert 构造后立即量化释放 fp32。
    # 多线程并行构造：每专家 seed 独立（seed*999983 + e*13），彼此无依赖，可安全并行。
    # 大 shape 下这是 golden 的最大单项开销（dsv3_ws16: 256 专家 × 44M 元素，实测 11s）。
    _built = _build_experts_parallel(c, range(exp))

    w1_full = [b[0] for b in _built]    # 量化后：bf16 tensor / int8 / int32(int4)
    w2_full = [b[1] for b in _built]
    s1_full = [b[2] for b in _built] if quant != "bf16" else []   # 仅量化：scale fp32 [n]
    s2_full = [b[3] for b in _built] if quant != "bf16" else []
    b1_full = [b[4] for b in _built] if quant == "w4a8" else []   # 仅 w4a8：bias fp32 [n]
    b2_full = [b[5] for b in _built] if quant == "w4a8" else []

    for rank in range(ws):
        per_rank[rank].update(_pack_rank_weights(c, _built[rank * epr:(rank + 1) * epr]))

    return {"x_list": x_list, "ids_list": ids_list, "w_list": w_list,
            "w1_full": w1_full, "w2_full": w2_full, "s1_full": s1_full, "s2_full": s2_full,
            "b1_full": b1_full, "b2_full": b2_full, "per_rank": per_rank}


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
    quant = c.get("quant", "bf16")

    # ---- dispatch：按 (rank, token, k) 投桶；量化时先 per-token dynamic quant ----
    if quant == "bf16":
        xq_list, sx_list = None, None
    else:
        xq_list, sx_list = [], []          # xq: int8 [tok,h]，sx: [tok] per-token scale
        for rank in range(ws):
            xq, sx = dispatch_quant(x_list[rank])
            xq_list.append(xq)
            sx_list.append(sx)

    buckets = [[] for _ in range(exp)]      # buckets[e] = list of (x_row float32 或 (q_row, sx))
    row_map = {}                            # (rank, token, k) -> (e, row_in_e)
    for rank in range(ws):
        ids = ids_list[rank]                # [tok, topk] int32
        for token in range(tok):
            for k in range(topk):
                e = int(ids[token, k].item())
                row_map[(rank, token, k)] = (e, len(buckets[e]))
                if quant == "bf16":
                    buckets[e].append(x_list[rank][token].to(torch.float32))
                else:
                    buckets[e].append((xq_list[rank][token], sx_list[rank][token].item()))

    # ---- expert_token_nums[rank][j] ----
    etn = torch.zeros((ws, epr), dtype=torch.int32)
    for e in range(exp):
        etn[e // epr, e % epr] = len(buckets[e])

    # ---- 每 expert FFN（float32；量化时逐步对齐 kernel：dequant → SwiGLU → requant）----
    Y_e = [None] * exp
    for e in range(exp):
        if not buckets[e]:
            Y_e[e] = torch.zeros((0, h), dtype=torch.float32)
            continue
        if quant == "bf16":
            Xe = torch.stack([b for b in buckets[e]]).to(torch.float32)     # [N, h]
            W1 = w1_full[e].to(torch.float32)
            W2 = w2_full[e].to(torch.float32)
            H = Xe @ W1                       # [N, 2*ih]
            A = _silu(H[:, :ih]) * H[:, ih:]  # SwiGLU, [N, ih]
            Y_e[e] = A @ W2                   # [N, h]
        else:
            Xq = torch.stack([b[0] for b in buckets[e]]).to(torch.float32)  # [N, h] int8 量化行
            sx1 = torch.tensor([b[1] for b in buckets[e]], dtype=torch.float32)  # [N]
            if quant == "w8a8":
                W1 = w1_full[e].to(torch.float32)   # int8 值
                W2 = w2_full[e].to(torch.float32)
            else:  # w4a8：w1_full 是打包后的 int8，需解包成 int4 值矩阵
                W1 = unpack_int4(w1_full[e]).to(torch.float32)
                W2 = unpack_int4(w2_full[e]).to(torch.float32)
            sw1 = G["s1_full"][e].to(torch.float32)
            sw2 = G["s2_full"][e].to(torch.float32)
            if quant == "w8a8":
                C1 = (Xq @ W1) * sw1.unsqueeze(0)          # fixpipe (C+0)·s_w1
                H = C1 * sx1.unsqueeze(1)                  # per-token 激活 scale
                A = _silu(H[:, :ih]) * H[:, ih:]
                q2, sx2 = requant(A)
                C2 = (q2.float() @ W2) * sw2.unsqueeze(0)
                Y_e[e] = C2 * sx2.unsqueeze(1)
            else:  # w4a8（MSD）
                xh, xl = int4_split(Xq.to(torch.int8))
                xhf, xlf = xh.float(), xl.float()
                C1h = (xhf @ W1) * sw1.unsqueeze(0)
                C1l = (xlf @ W1) * sw1.unsqueeze(0)
                b1 = G["b1_full"][e].unsqueeze(0).float()
                H = (16.0 * C1h + C1l + b1) * sx1.unsqueeze(1)   # 序：权重 scale → ×16+low → +bias → 激活 scale
                A = _silu(H[:, :ih]) * H[:, ih:]
                q2, sx2 = requant(A)
                x2h, x2l = int4_split(q2)
                x2hf, x2lf = x2h.float(), x2l.float()
                C2h = (x2hf @ W2) * sw2.unsqueeze(0)
                C2l = (x2lf @ W2) * sw2.unsqueeze(0)
                b2 = G["b2_full"][e].unsqueeze(0).float()
                Y_e[e] = (16.0 * C2h + C2l + b2) * sx2.unsqueeze(1)

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
# NPU 小算子级联参考（原 npu_golden.py，已并入本文件成为单脚本）
# 把 mega_moe 拆成 5 个小算子段，全部用 torch_npu 在 NPU 上算（替代 CPU golden，快 ~12 万倍）：
#   ① dispatch  : argsort(expert_id) + index_select      —— 按专家把 token 行聚到一起
#   ② GMM1      : 逐专家 matmul（x @ W1）
#   ③ SwiGLU    : silu(gate) * up
#   ④ GMM2      : 逐专家 matmul（a @ W2）
#   ⑤ combine   : index_add_ 按权重散射累加回原 token
# 核心是把三层 Python 循环换成「排序 + gather + scatter-add」的向量化三件套，
# 零 .item()、零 Python 级循环（专家维除外，那是 exp 次 matmul，本来就要分组算）。
#
# 【数值约定】全程 float32 累加（与 CPU golden 一致），只在输入/权重侧保持 bf16 —— 与 kernel 的
# 「bf16 输入 → fp32 累加 → bf16 输出」一致。量化路径逐步复刻 kernel 的 scale 施加顺序。
#
# 用法：y_list, etn = npu_mega_moe_golden(case, G, device='npu:0')
#   入参与 cpu_mega_moe_golden 完全一致（复用 make_inputs 的产物 G），返回 (y_list, etn)：
#     y_list[rank] : [tok, h] float32（CPU tensor，便于与 NPU 算子输出比对）
#     etn          : [ws, epr] int32（CPU tensor）
# ===========================================================================================================
def _dispatch_quant(x):
    """per-token 动态对称量化到 int8。返回 (int8 量化值, per-token scale)。"""
    amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = amax / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127)
    return q, s.squeeze(1)


def _requant(a):
    """SwiGLU 输出 requant 到 int8（GMM2 的激活）。"""
    amax = a.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = amax / 127.0
    q = torch.clamp(torch.round(a / s), -127, 127)
    return q, s.squeeze(1)


def _int4_split(q):
    """w4a8 的 MSD 拆分：q = 16*high + low，high/low 各在 int4 范围。"""
    q = q.to(torch.float32)
    high = torch.floor((q + 8.0) / 16.0) - 0.5
    low = q - 16.0 * high
    return high, low


def npu_mega_moe_golden(c, G, device="npu:0"):
    """
    NPU 小算子拼接参考实现。

    入参与 cpu_mega_moe_golden 完全一致（复用 make_inputs 的产物 G），
    返回 (y_list, etn)：
      y_list[rank] : [tok, h] float32（CPU tensor，便于与 NPU 算子输出比对）
      etn          : [ws, epr] int32（CPU tensor）
    """
    ws, exp = c["world_size"], c["num_experts"]
    epr = exp // ws
    h, ih, topk, tok = c["hidden"], c["intermediate_hidden"], c["topk"], c["num_tokens"]
    quant = c.get("quant", "bf16")
    dev = torch.device(device)

    x_list, ids_list, w_list = G["x_list"], G["ids_list"], G["w_list"]
    w1_full, w2_full = G["w1_full"], G["w2_full"]

    # ---------------------------------------------------------------------------------------------------
    # ① dispatch —— 向量化替代三层 Python 循环
    #    全局视角：把 ws 个 rank 的 [tok, topk] 路由拉平成一维 (rank,token,k) 序列，
    #    再按 expert_id 稳定排序 → 同一专家的行天然连续，用 index_select 一次取出。
    # ---------------------------------------------------------------------------------------------------
    # X_all: [ws*tok, h]，行号 = rank*tok + token。保持 bf16 —— 与 kernel 输入同精度，
    # 且 matmul 直接吃 bf16 走 Cube 快路径。
    X_all_bf = torch.cat([x for x in x_list], dim=0).to(dev)          # bf16
    # ids_all: [ws*tok*topk]，元素 = expert id；位置 idx = (rank*tok + token)*topk + k
    ids_all = torch.cat([ids.reshape(-1) for ids in ids_list], dim=0).to(dev).long()
    n_slot = ids_all.numel()  # = ws*tok*topk

    # 每个 slot 对应的源行号（rank*tok+token）
    src_row = (torch.arange(n_slot, device=dev) // topk).long()

    # 稳定排序：先按 expert 排，同 expert 内保持 (rank,token,k) 原序 —— 与 CPU 版投桶顺序一致
    order = torch.argsort(ids_all.float(), stable=True)
    sorted_eid = ids_all[order]                 # [n_slot] 已按专家分组
    gathered_row = src_row[order]               # [n_slot] 每个槽位取哪一行 X

    # 每专家行数（与 CPU 版 etn 同语义）
    counts = torch.bincount(ids_all, minlength=exp)          # [exp]
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                         torch.cumsum(counts, dim=0)[:-1]])  # [exp] 每专家起始偏移

    etn = counts.reshape(ws, epr).to(torch.int32).cpu()

    # 按排序后的顺序聚出输入行：[n_slot, h]。bf16 版给 GMM 直算，fp32 版给量化路径用。
    Xg_bf = torch.index_select(X_all_bf, 0, gathered_row)    # bf16 [n_slot, h]

    if quant != "bf16":
        # per-token 动态量化在 dispatch 阶段做（与 kernel 一致：量化后再跨卡搬运）
        X_all_f = X_all_bf.to(torch.float32)
        Xq_all, sx_all = _dispatch_quant(X_all_f)            # [ws*tok, h], [ws*tok]
        Xg_q = torch.index_select(Xq_all, 0, gathered_row)   # [n_slot, h]
        sx_g = torch.index_select(sx_all, 0, gathered_row)   # [n_slot]

    # ---------------------------------------------------------------------------------------------------
    # ②③④ 逐专家 GMM1 → SwiGLU → GMM2
    #     专家维必须分开算（每个专家权重不同），但每专家内部是一次大 matmul，无 Python 级逐行循环。
    # ---------------------------------------------------------------------------------------------------
    Y_sorted = torch.zeros((n_slot, h), dtype=torch.float32, device=dev)

    cnt_cpu = counts.cpu().tolist()
    off_cpu = offsets.cpu().tolist()

    for e in range(exp):
        n = cnt_cpu[e]
        if n == 0:
            continue
        lo = off_cpu[e]
        hi = lo + n

        if quant == "bf16":
            # 【搬运 + 算力双优化】
            # 1) 只搬 bf16（不在 CPU 侧转 fp32）：H2D 流量减半，实测 2.4x
            # 2) matmul 直接吃 bf16：NPU Cube 原生支持 bf16 且内部按 fp32 累加，
            #    与 kernel 的「bf16 输入 → fp32 累加」同构；转 fp32 反而多一次全量转换。
            # SwiGLU 在 fp32 上做（与 CPU golden 一致，避免 silu 在 bf16 上的精度损失）。
            W1 = w1_full[e].to(dev, non_blocking=True)      # bf16 [h, 2*ih]
            W2 = w2_full[e].to(dev, non_blocking=True)      # bf16 [ih, h]
            H = (Xg_bf[lo:hi] @ W1).to(torch.float32)       # ② GMM1（bf16 in，fp32 out）
            A = _silu(H[:, :ih]) * H[:, ih:]                # ③ SwiGLU（fp32）
            Y_sorted[lo:hi] = (A.to(torch.bfloat16) @ W2).to(torch.float32)   # ④ GMM2

        elif quant == "w8a8":
            W1 = w1_full[e].to(dev, non_blocking=True).to(torch.float32)
            W2 = w2_full[e].to(dev, non_blocking=True).to(torch.float32)
            sw1 = G["s1_full"][e].to(dev, non_blocking=True).to(torch.float32)
            sw2 = G["s2_full"][e].to(dev, non_blocking=True).to(torch.float32)
            xq = Xg_q[lo:hi]
            sx1 = sx_g[lo:hi]
            C1 = (xq @ W1) * sw1.unsqueeze(0)           # fixpipe: (C+0)·s_w1
            H = C1 * sx1.unsqueeze(1)                   # per-token 激活 scale
            A = _silu(H[:, :ih]) * H[:, ih:]
            q2, sx2 = _requant(A)
            C2 = (q2 @ W2) * sw2.unsqueeze(0)
            Y_sorted[lo:hi] = C2 * sx2.unsqueeze(1)

        else:  # w4a8（MSD 拆分）
            W1 = unpack_int4(w1_full[e]).to(torch.float32).to(dev)
            W2 = unpack_int4(w2_full[e]).to(torch.float32).to(dev)
            sw1 = G["s1_full"][e].to(torch.float32).to(dev)
            sw2 = G["s2_full"][e].to(torch.float32).to(dev)
            b1 = G["b1_full"][e].to(torch.float32).to(dev).unsqueeze(0)
            b2 = G["b2_full"][e].to(torch.float32).to(dev).unsqueeze(0)
            xq = Xg_q[lo:hi]
            sx1 = sx_g[lo:hi]
            xh, xl = _int4_split(xq)
            C1h = (xh @ W1) * sw1.unsqueeze(0)
            C1l = (xl @ W1) * sw1.unsqueeze(0)
            H = (16.0 * C1h + C1l + b1) * sx1.unsqueeze(1)
            A = _silu(H[:, :ih]) * H[:, ih:]
            q2, sx2 = _requant(A)
            x2h, x2l = _int4_split(q2)
            C2h = (x2h @ W2) * sw2.unsqueeze(0)
            C2l = (x2l @ W2) * sw2.unsqueeze(0)
            Y_sorted[lo:hi] = (16.0 * C2h + C2l + b2) * sx2.unsqueeze(1)

    # ---------------------------------------------------------------------------------------------------
    # ⑤ combine —— index_add_ 向量化替代逐 token 累加
    #    y[rank][token] = Σ_k w_k · Y[该 (rank,token,k) 对应的专家输出行]
    # ---------------------------------------------------------------------------------------------------
    # 每个排序后槽位的 topk 权重
    W_all = torch.cat([w.to(torch.float32).reshape(-1) for w in w_list], dim=0).to(dev)  # [n_slot]
    w_g = W_all[order]                                   # 与 Y_sorted 对齐

    Y_weighted = Y_sorted * w_g.unsqueeze(1)             # [n_slot, h]
    Y_out = torch.zeros((ws * tok, h), dtype=torch.float32, device=dev)
    Y_out.index_add_(0, gathered_row, Y_weighted)        # 散射累加回 (rank,token)

    y_list = [Y_out[r * tok:(r + 1) * tok].cpu() for r in range(ws)]
    # etn 与 cpu_mega_moe_golden 同格式：list[ws]，每项 [epr] int32
    return y_list, [etn[r] for r in range(ws)]


# ===========================================================================================================
# NPU 调用层（spawn 常驻 worker：进程/HCCL 只初始化一次，Queue 驱动 case 循环）
#
# 【为什么从 fork-per-case 换成 spawn 常驻】
#   旧模型每个 case 都 fork ws 个子进程 + 再 fork 一个 golden 进程，跑完全部销毁。
#   62 个 case 累计 spawn 730 个进程、668 次 import torch extension + init_process_group，
#   而被测 kernel 单次只有 76us~10ms —— 初始化开销比被测对象大 4~5 个数量级。
#   实测 62 case 全量 140 分钟，其中 99.99% 是重复初始化。
#
#   常驻 worker：set_device / init_process_group / new_group / all_reduce warmup 全程只做 1 次，
#   每个 case 只重建 SymmBuffer（shape 相关，必须重建）和权重 NPU tensor。
#
# 【spawn 相对 fork 的两处必然改动】
#   1. 全局变量不再被继承 —— 输入不能靠 G_INPUT 传，改由 worker 侧 build_rank_slice 自建
#      （反而是收益：ws 个 worker 各造 exp/ws 个专家，天然并行）。
#   2. cann_ops_transformer 在每个 worker 内 import 一次即可（spawn 是干净解释器，无 fork 的
#      「继承已初始化 NPU 状态」问题 —— 这正是旧模型 master_npugolden_all.log 里
#      "Cannot re-initialize NPU in forked subprocess" 崩溃的根因）。
# ===========================================================================================================
def _read_kernel_duration(pdir):
    """从 profiler 输出目录读 MegaMoe 的 Duration(us)，读不到返回 None。"""
    import glob as _glob, csv as _csv
    for kd in _glob.glob(f"{pdir}/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"):
        with open(kd) as f:
            for row in _csv.DictReader(f):
                if "MegaMoe" in row.get("Name", ""):
                    return float(row["Duration(us)"])
    return None


def _to_npu_inputs(c, d):
    """CPU 输入片 → NPU tensor（含量化权重的 FRACTAL_NZ 转换）。返回 mega_moe 的入参 dict。"""
    quant = c.get("quant", "bf16")
    out = {
        "x": d["x"].to(torch.bfloat16).npu(),
        "ids": d["topk_ids"].to(torch.int32).npu(),
        # topk_weights 必须 float32：传 bf16 会让非均匀随机权重精度崩塌（cos 0.76），见 RUNBOOK 坑表
        "w": d["topk_weights"].to(torch.float32).npu(),
    }
    if quant == "bf16":
        out.update(l1=[t.to(torch.bfloat16).npu() for t in d["l1"]],
                   l2=[t.to(torch.bfloat16).npu() for t in d["l2"]],
                   l1_sf=None, l2_sf=None, l1_bias=None, l2_bias=None,
                   qmode=0, qout=None)
        return out

    # INT8/INT4 权重 → FRACTAL_NZ 格式（empty_with_format + copy_ 自动做 ND→NZ 转换；
    # 此环境 npu_format_cast 失效，不可用）。w4a8 已每 2 个 int4 打包 1 字节，NZ 后 view 成 int32。
    def _to_nz(wt, dtype):
        t = torch_npu.empty_with_format(wt.shape, dtype=dtype, device="npu", acl_format=29)
        t.copy_(wt.to(dtype).npu())
        return t

    l1 = [_to_nz(t, torch.int8) for t in d["l1"]]
    l2 = [_to_nz(t, torch.int8) for t in d["l2"]]
    if quant == "w4a8":
        l1 = [t.view(torch.int32) for t in l1]
        l2 = [t.view(torch.int32) for t in l2]
    out.update(l1=l1, l2=l2,
               l1_sf=[t.to(torch.uint64).npu() for t in d["l1_sf"]],
               l2_sf=[t.to(torch.uint64).npu() for t in d["l2_sf"]],
               l1_bias=[t.to(torch.float32).npu() for t in d["l1_bias"]] if quant == "w4a8" else None,
               l2_bias=[t.to(torch.float32).npu() for t in d["l2_bias"]] if quant == "w4a8" else None,
               qmode=2, qout=torch.int8)
    return out


def _worker_run_case(c, rank, ep_group, mega_moe, get_symm_buffer_for_mega_moe):
    """worker 侧跑一个 case：建输入 → 建 sym_buffer → warmup → profiler → 算子 → 回收。

    返回 (y_cpu_bf16, etn_cpu, dur_us)。sym_buffer 在 finally 里 destroy —— 常驻 worker
    下不回收会累积 HCCL 显存，62 个 case 必然 OOM。
    """
    ws = c["world_size"]
    t0 = time.time()
    d = build_rank_slice(c, rank)          # 只造本 rank 的 epr 个专家
    n = _to_npu_inputs(c, d)
    t_build = time.time() - t0

    # max_recv 必须用 nmtr 计算（example 方式：nmtr × ws × min(topk,epr)），
    # 用 num_tokens 会偏小导致后半 token 丢失
    epr = c["num_experts"] // ws
    max_recv = c["num_max_tokens_per_rank"] * ws * min(c["topk"], epr)
    sym = get_symm_buffer_for_mega_moe(
        ep_group, num_experts=c["num_experts"],
        num_max_tokens_per_rank=c["num_max_tokens_per_rank"],
        num_topk=c["topk"], hidden=c["hidden"],
        intermediate_hidden=c["intermediate_hidden"],
        max_recv_token_num=max_recv,
        dispatch_quant_mode=n["qmode"], dispatch_quant_out_dtype=n["qout"])
    pdir = None
    try:
        def _call():
            return mega_moe(n["x"], n["ids"], n["w"], n["l1"], n["l2"], sym,
                            l1_weights_sf=n["l1_sf"], l2_weights_sf=n["l2_sf"],
                            l1_bias=n["l1_bias"], l2_bias=n["l2_bias"])

        # warmup（避免首次 gen_task 运行时编译进 profiler）
        _call()
        torch.npu.synchronize()

        # rank0 profiler 抓 MegaMoe kernel Duration（perf/golden 合一）；GM_NO_PROF=1 禁用
        dur, prof = None, None
        do_prof = (rank == 0 and not os.environ.get("GM_NO_PROF"))
        if do_prof:
            pdir = f"/tmp/gm_prof_{c['name']}_{os.getpid()}"
            shutil.rmtree(pdir, ignore_errors=True)
            os.makedirs(pdir, exist_ok=True)
            ec = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                # data_simplification=True 丢掉 tensorboard 的中间态文件，只留 ASCEND_PROFILER_OUTPUT；
                # 我们只读 kernel_details.csv 的一个字段，完整 trace 是纯浪费（ws=16 导出要 10~18s）
                data_simplification=True,
                export_type=torch_npu.profiler.ExportType.Text)
            prof = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                experimental_config=ec,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(pdir))
            prof.start()

        y, etn = _call()
        torch.npu.synchronize()

        if do_prof:
            prof.stop()
            dur = _read_kernel_duration(pdir)

        y_cpu, etn_cpu = y.cpu(), etn.cpu()
        if os.environ.get("GM_DEBUG"):
            print(f"[gmdebug r{rank}] {c['name']} build={t_build:.1f}s "
                  f"total={time.time()-t0:.1f}s", flush=True)
        return y_cpu, etn_cpu, dur
    finally:
        # sym 不回收 → 常驻 worker 下 HCCL 显存逐 case 累积必 OOM
        sym.destroy()
        # profiler 目录不清 → /tmp 会被几百个 trace 撑爆（实测 438 个残留目录）
        if pdir:
            shutil.rmtree(pdir, ignore_errors=True)


def _enc(t):
    """tensor → bytes（走普通管道，绕开 /dev/shm）。

    torch.multiprocessing 的 Queue 默认 file_descriptor 策略：tensor 经 /dev/shm 零拷贝。
    但容器里 /dev/shm 常被限到 64MB，而 ws=16 + tok512 + h7168 的结果就有 117MB，
    直接 "No space left on device"（实测踩过）。序列化成 bytes 走管道，代价是一次拷贝，
    对 7MB 级张量可忽略。
    """
    import io
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _dec(b):
    """_enc 的逆操作。"""
    import io
    return torch.load(io.BytesIO(b), weights_only=False)


def worker_loop(rank, port, task_q, result_q):
    """常驻 worker 进程入口：初始化一次，然后循环消费 task_q 里的 case。

    task 协议：
      {"type":"run", "case":{...}, "need_ref":bool}  跑一个 case
      {"type":"exit"}                                 优雅退出
    result 协议：
      {"rank":r, "ok":True,  "y":.., "etn":.., "dur":.., "ref":{"y":..,"etn":..}|None}
      {"rank":r, "ok":False, "err":"..."}             异常（父进程据此判 case 失败，不影响后续 case）
    """
    # cann_ops_transformer 在 worker 内 import（spawn 是干净解释器，无 fork 的继承问题）
    from cann_ops_transformer.ops import mega_moe, get_symm_buffer_for_mega_moe

    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=WORLD,
                            init_method=f"tcp://127.0.0.1:{port}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(WORLD)))

    # HCCL warmup（必须，否则 get_symm_buffer 内 HcclCommGetHandleWithName 失败）
    warm = torch.ones(1, device="npu")
    dist.all_reduce(warm, group=ep_group)
    torch.npu.synchronize()

    try:
        while True:
            task = task_q.get()
            if task["type"] == "exit":
                break
            c = task["case"]
            try:
                y, etn, dur = _worker_run_case(c, rank, ep_group, mega_moe,
                                               get_symm_buffer_for_mega_moe)
                res = {"rank": rank, "ok": True, "y": _enc(y), "etn": _enc(etn),
                       "dur": dur, "ref": None}
                # rank0 顺带算 NPU 小算子级联参考值（golden）。
                # 常驻 worker 下这是安全的：通信域不销毁，不存在旧模型「golden 打乱跨 rank
                # 退出节奏」的问题；golden 与算子都在本卡串行，靠 synchronize 分隔。
                if task["need_ref"] and rank == 0:
                    G = make_inputs(c)                # golden 需要全部 exp 个专家
                    ry, retn = npu_mega_moe_golden(c, G, device="npu:0")
                    torch.npu.synchronize()
                    res["ref"] = {"y": _enc(ry), "etn": _enc(retn)}
                result_q.put(res)
            except Exception as e:
                import traceback
                result_q.put({"rank": rank, "ok": False,
                              "err": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def _free_port():
    """要一个当前空闲的 TCP 端口给 init_process_group。

    不用固定端口：切 world_size 时旧池刚退出，端口可能还在 TIME_WAIT，
    固定端口会偶发 "Address already in use"。
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WorkerPool:
    """常驻 worker 池。**同一时刻只存在一个池**，(world_size, quant) 任一变化时先拆旧池再建新池。

    【为什么不能多池并存】HCCL 除了我们指定的 init_method TCP 端口，还会在每张卡的
    网卡上绑一个固定内部端口（实测 16666）。ws=2 的 worker 若还活着占着 device 0~1，
    ws=8 的池起来就报
        EI0020: The IP address 192.168.2.199 and port 16666 have already been bound
    继而 hcclCommInitRootInfoConfig error code 7 → 整池卡死在 HcclAllreduce 直到超时。

    【为什么 quant 也要拆池】实测：常驻 worker 里先跑 bf16 再跑 w8a8，第一个 w8a8 用例
    必挂在 "aicore timeout"（error 507014）——全量跑里 3 次失败全是这个模式
    （dsv4_ws8_t16_w8a8 / glm45_ws8_t16_w8a8 / dsv32_ws16_t16_w8a8），而这 3 个单独跑全过。
    根因是 mega_moe 走 gen_task 运行时编译，量化模式切换后残留的 kernel/上下文状态与新
    tiling 不匹配。按 quant 分池后各自的 worker 只见过一种量化模式，彻底规避。

    【为什么仍然值得】main() 已按 (quant, ws) 把用例排序，最多只切十来次；
    同组的一串 case 共用一个池，初始化开销从「每 case 一次」降到「每组一次」。
    """

    def __init__(self):
        self._key = None         # (world_size, quant)
        self._procs = None
        self._task_q = None
        self._result_q = None
        self._mgr = None

    def _pool(self, ws, quant):
        if self._key == (ws, quant):
            return self._task_q, self._result_q
        # 切 ws 或 quant：先彻底拆掉旧池（释放 device / HCCL 端口 / kernel 上下文），再建新的
        self.shutdown()
        self._mgr = mp.Manager()
        self._task_q, self._result_q = self._mgr.Queue(), self._mgr.Queue()
        port = _free_port()
        ctx = mp.get_context("spawn")
        self._procs = []
        for r in range(ws):
            p = ctx.Process(target=_worker_entry,
                            args=(r, ws, port, self._task_q, self._result_q))
            p.start()
            self._procs.append(p)
        self._key = (ws, quant)
        return self._task_q, self._result_q

    def run_case(self, c, need_ref=True, timeout=900):
        """跑一个 case，返回 (results, ref)。results[rank] = (y_f32, etn, dur)。

        任一 rank 失败或超时 → 抛异常，由上层 per-case try/except 兜住，不影响后续 case。
        """
        ws = c["world_size"]
        task_q, result_q = self._pool(ws, c.get("quant", "bf16"))
        # 只传纯 dict（不含 tensor）——tensor 走 Queue 会 pickle 几十 GB
        cc = {k: v for k, v in c.items() if not k.startswith("_")}
        for _ in range(ws):
            task_q.put({"type": "run", "case": cc, "need_ref": need_ref})

        results, ref, errs = {}, None, []
        deadline = time.time() + timeout
        for _ in range(ws):
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError(f"{c['name']}: 等待 worker 结果超时（{timeout}s）")
            r = result_q.get(timeout=left)
            if not r["ok"]:
                errs.append(f"rank{r['rank']}: {r['err']}")
                continue
            results[r["rank"]] = (_dec(r["y"]).to(torch.float32), _dec(r["etn"]), r["dur"])
            if r["ref"] is not None:
                ref = {"y": _dec(r["ref"]["y"]), "etn": _dec(r["ref"]["etn"])}
        if errs:
            raise RuntimeError("worker 异常:\n" + "\n".join(errs))
        return results, ref

    def shutdown(self):
        """拆池并等进程真正退出——必须 join 到位，否则 NPU device / HCCL 端口不释放。"""
        if self._procs:
            for _ in self._procs:
                try:
                    self._task_q.put({"type": "exit"})
                except Exception:
                    pass
            for p in self._procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=15)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=10)
        if self._mgr is not None:
            self._mgr.shutdown()
        self._key = self._procs = self._task_q = self._result_q = self._mgr = None


def _worker_entry(rank, ws, port, task_q, result_q):
    """spawn 进程真正的入口：先把 WORLD 设进模块全局，再进 worker_loop。

    spawn 会重新 import 本模块（__name__ != "__main__"），模块级全局是初始值，
    故 world_size 必须显式传入而非依赖继承。
    """
    global WORLD
    WORLD = ws
    worker_loop(rank, port, task_q, result_q)


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
    # ---- 小 batch 场景（token 16/32/64，ws=16，BF16；遍历验证全跑通）----
    dict(name="dsv3_ws16_t16",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=30),
    dict(name="dsv3_ws16_t32",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=31),
    dict(name="dsv3_ws16_t64",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=32),
    dict(name="dsv4_ws16_t16",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=33),
    dict(name="dsv4_ws16_t32",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=34),
    dict(name="dsv4_ws16_t64",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=35),
    dict(name="kimi_ws16_t16",  world_size=16, num_experts=384, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=36),
    dict(name="kimi_ws16_t32",  world_size=16, num_experts=384, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=37),
    dict(name="glm5_ws16_t16",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=39),
    dict(name="glm5_ws16_t32",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=40),
    dict(name="glm5_ws16_t64",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=41),
    dict(name="qwen3_ws16_t16", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=16, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=42),
    dict(name="qwen3_ws16_t32", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=43),
    dict(name="qwen3_ws16_t64", world_size=16, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=64, num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=44),
    # ---- 量化场景（w8a8 = A8W8-INT / w4a8 = A8W4-INT，A3 arch22）----
    # qsmoke：小规模快速验证/定位（coded 用例数据自带标签，可反推出错位置）
    dict(name="w8a8_ws2",         world_size=2,  num_experts=8,   hidden=1024, intermediate_hidden=1024, topk=2, num_tokens=32, num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=50, quant="w8a8"),
    dict(name="w8a8_coded_ws2",   world_size=2,  num_experts=8,   hidden=1024, intermediate_hidden=1024, topk=2, num_tokens=64, num_max_tokens_per_rank=128, input_mode="coded",  topk_ids_mode="unique", topk_weights_mode="equal", seed=52, quant="w8a8"),
    dict(name="tok1_w8a8_ws2",    world_size=2,  num_experts=8,   hidden=1024, intermediate_hidden=1024, topk=2, num_tokens=1,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=56, quant="w8a8"),
    # ---- 模型矩阵（真实 shape × ws × quant × batch）----
    # DeepSeek-V3.2 (256, 7168, 2048, k8) — W8A8 是官方 w8a8-mtp 形态
    dict(name="dsv32_ws16_t16_w8a8",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=60, quant="w8a8"),
    dict(name="dsv32_ws16_t64_w8a8",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=61, quant="w8a8"),
    dict(name="dsv32_ws16_t512_w8a8", world_size=16, num_experts=256, hidden=7168, intermediate_hidden=2048, topk=8, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=62, quant="w8a8"),
    # DeepSeek-V4-Flash (256, 7168, 3072, k6) — W8A8/W4A8；BF16 仅 ws=16（EP=8 权重放不下）
    dict(name="dsv4_ws16_t16_w8a8",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=63, quant="w8a8"),
    dict(name="dsv4_ws16_t64_w8a8",  world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=64, quant="w8a8"),
    dict(name="dsv4_ws16_t512_w8a8", world_size=16, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=65, quant="w8a8"),
    # DSV4-Flash ws=8（EP=8 形态，量化后 64GB 可容；无 BF16）
    dict(name="dsv4_ws8_t16_w8a8",  world_size=8, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=73, quant="w8a8"),
    dict(name="dsv4_ws8_t64_w8a8",  world_size=8, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=74, quant="w8a8"),
    dict(name="dsv4_ws8_t512_w8a8", world_size=8, num_experts=256, hidden=7168, intermediate_hidden=3072, topk=6, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=75, quant="w8a8"),
    # GLM-4.5（真实 shape：160 专家 / h5120 / ih1536 / k8，ws=8=EP8）——旧 glm45_ws16(8192,3072) 保留为压力用例
    dict(name="glm45_ws8_t16_bf16",  world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=8, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=80, quant="bf16"),
    dict(name="glm45_ws8_t64_bf16",  world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=8, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=81, quant="bf16"),
    dict(name="glm45_ws8_t512_bf16", world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=8, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=82, quant="bf16"),
    dict(name="glm45_ws8_t16_w8a8",  world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=8, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=83, quant="w8a8"),
    dict(name="glm45_ws8_t64_w8a8",  world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=8, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=84, quant="w8a8"),
    dict(name="glm45_ws8_t512_w8a8", world_size=8, num_experts=160, hidden=5120, intermediate_hidden=1536, topk=8, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=85, quant="w8a8"),
    # GLM-5 / 5.2（256, 6144, 2048, k8 推断；w8a8/w4a8 覆盖）
    dict(name="glm5_ws16_t16_w8a8",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=86, quant="w8a8"),
    dict(name="glm5_ws16_t64_w8a8",  world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=87, quant="w8a8"),
    dict(name="glm5_ws16_t512_w8a8", world_size=16, num_experts=256, hidden=6144, intermediate_hidden=2048, topk=8, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=88, quant="w8a8"),
    # Qwen3-235B（128, 4096, 1536, k8）ws=8=EP8（官方推荐 TP8+EP）——W8A8
    dict(name="qwen3_ws8_t16_w8a8",  world_size=8, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=92, quant="w8a8"),
    dict(name="qwen3_ws8_t64_w8a8",  world_size=8, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=93, quant="w8a8"),
    dict(name="qwen3_ws8_t512_w8a8", world_size=8, num_experts=128, hidden=4096, intermediate_hidden=1536, topk=8, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=94, quant="w8a8"),
    # Kimi-K2.5（384, 7168, 2048, k8）ws=16——W4A8（官方 W4A8 形态）
    # Llama-4 Maverick（128, 5120, 2048, k8）ws=8——W8A8
    dict(name="llama4_ws8_t16_w8a8",  world_size=8, num_experts=128, hidden=5120, intermediate_hidden=2048, topk=8, num_tokens=16,  num_max_tokens_per_rank=64,  input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=98, quant="w8a8"),
    dict(name="llama4_ws8_t64_w8a8",  world_size=8, num_experts=128, hidden=5120, intermediate_hidden=2048, topk=8, num_tokens=64,  num_max_tokens_per_rank=128, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=99, quant="w8a8"),
    dict(name="llama4_ws8_t512_w8a8", world_size=8, num_experts=128, hidden=5120, intermediate_hidden=2048, topk=8, num_tokens=512, num_max_tokens_per_rank=1024, input_mode="random", topk_ids_mode="unique", topk_weights_mode="random", seed=100, quant="w8a8"),
]


# ===========================================================================================================
# 单用例执行
# ===========================================================================================================
def run_case(c, pool, dump=False):
    t0 = time.time()
    # 1) 跑算子 + 取参考值。默认参考 = NPU 小算子级联（npu_mega_moe_golden），由 rank0 worker
    #    在算完算子后于同一进程内顺带算出，随结果一起回传。
    npu_res, ref = pool.run_case(c, need_ref=not USE_CPU_GOLDEN)
    t_npu = time.time() - t0

    # 2) --cpu-golden 时才在父进程算 CPU 参考（很慢，仅排障用；父进程不碰 NPU 的约束不受影响）
    if USE_CPU_GOLDEN:
        golden_y, golden_etn = cpu_mega_moe_golden(c, make_inputs(c))
    else:
        assert ref is not None, "rank0 未回传 NPU 参考值（检查 worker 是否异常退出）"
        golden_y, golden_etn = ref["y"], ref["etn"]

    # 3) 比对
    print(f"\n{'=' * 72}\nCase: {c['name']}  (ws={c['world_size']}, exp={c['num_experts']}, "
          f"h={c['hidden']}, ih={c['intermediate_hidden']}, topk={c['topk']}, tok={c['num_tokens']}, "
          f"mode={c['input_mode']})\n{'=' * 72}")
    # 实测（含 example.py 自身）：当前算子 combine 仅输出前 num_tokens/ws 个 token 有效，后半为 0。
    # golden 只比对前 half；重写算子若修了 combine 全输出，再去掉此截断。
    # tok < ws 时整除得 0，切片会产出空张量并让 check_result 的 max() 抛异常（tok1_* 用例踩过）。
    # 至少比对 1 个 token：算子在 tok=1 时仍会输出该 token。
    half = max(1, c["num_tokens"] // c["world_size"])
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
    print(f"  -> {'ALL PASSED' if all_pass else 'SOME FAILED'}  "
          f"[耗时 {time.time() - t0:.1f}s (算子+golden {t_npu:.1f}s)]")
    return all_pass


# ===========================================================================================================
# main
# ===========================================================================================================
def main():
    ap = argparse.ArgumentParser(description="mega_moe bf16 精度 golden")
    ap.add_argument("--case", type=str, default=None, help="用例名")
    ap.add_argument("--num", type=int, default=None, help="用例序号(1-based)")
    ap.add_argument("--all", action="store_true", help="跑全部用例")
    ap.add_argument("--filter", type=str, default=None, help="按名字子串过滤用例（可多个逗号分隔）")
    ap.add_argument("--dump", action="store_true", help="dump golden/npu y 到 .dump/")
    ap.add_argument("--cpu-golden", action="store_true",
                    help="用 CPU 参考实现（默认用 NPU 小算子拼接版，快约 4x）")
    ap.add_argument("--list", action="store_true", help="列出用例")
    args = ap.parse_args()

    global USE_CPU_GOLDEN
    USE_CPU_GOLDEN = args.cpu_golden

    if args.list:
        for i, c in enumerate(CASES, 1):
            print(f"  [{i:>2d}] {c['name']}")
        return

    if args.filter:
        subs = [s.strip() for s in args.filter.split(",") if s.strip()]
        selected = [i for i, c in enumerate(CASES)
                    if any(s in c["name"] for s in subs)]
        assert selected, f"--filter {args.filter} 无匹配用例（--list 查看全部）"
    elif args.all:
        # w4a8 的 CPU golden 尚未与 NPU 对齐（int4 权重在 FRACTAL_NZ 下的 nibble 分组契约未确认，
        # 详见 PERF_TUNING_JOURNAL 量化章节），--all 默认跳过；需要时用 --filter w4a8 单独跑。
        selected = [i for i, c in enumerate(CASES) if c.get("quant") != "w4a8"]
    elif args.case:
        names = {c["name"]: i for i, c in enumerate(CASES)}
        assert args.case in names, f"未知用例 {args.case}（--list 查看全部）"
        selected = [names[args.case]]
    elif args.num:
        assert 1 <= args.num <= len(CASES), f"--num 须 ∈[1,{len(CASES)}]"
        selected = [args.num - 1]
    else:
        ap.error("需指定 --case / --num / --all / --list")

    # spawn 必须在任何 NPU 初始化之前设定，且整个进程只设一次。
    # 用 spawn 而非 fork：fork 会继承父进程状态，一旦父进程碰过 NPU，后续子进程直接报
    # "Cannot re-initialize NPU in forked subprocess"（master_npugolden_all.log 实录）。
    mp.set_start_method("spawn", force=True)

    # 按 (quant, world_size) 分组跑：同组的 case 复用同一个常驻 worker 池。
    # quant 必须参与分组且优先级最高——量化模式在同一 worker 内切换会触发 aicore timeout
    # （见 WorkerPool 文档），按 quant 聚簇能把拆池次数降到最少。
    order = sorted(selected,
                   key=lambda i: (CASES[i].get("quant", "bf16"), CASES[i]["world_size"], i))
    if order != selected:
        _grp = sorted({(CASES[i].get("quant", "bf16"), CASES[i]["world_size"]) for i in order})
        print(f"[info] 用例已按 (quant, world_size) 重排以复用 worker 池，共 {len(_grp)} 组: "
              + ", ".join(f"{q}/ws{w}" for q, w in _grp))

    pool = WorkerPool()
    npass = 0
    errs = []
    t_start = time.time()
    try:
        for i in order:
            c = dict(CASES[i])            # copy，避免污染内置 CASES
            # per-case 异常隔离：单个用例的脚本 bug 或算子崩溃不得吞掉后续用例。
            # （曾因 tok1 用例在 check_result 抛异常，让 --all 后 24 个用例零信息。）
            try:
                npass += int(run_case(c, pool, dump=args.dump))
            except Exception as e:
                errs.append((c["name"], f"{type(e).__name__}: {e}"))
                print(f"  [rank -] ERROR {c['name']}: {type(e).__name__}: {e}")
                # worker 异常后池状态可能已损坏（某 rank 挂了会让后续集合通信永久失配），
                # 拆掉重建——下个 case 的 _pool() 会自动起新池。
                pool.shutdown()
    finally:
        pool.shutdown()

    _el = time.time() - t_start
    print(f"\n==== 总结: {npass}/{len(order)} 用例通过  "
          f"总耗时 {_el / 60:.1f}min（均 {_el / max(1, len(order)):.0f}s/case）====")
    if errs:
        print(f"==== 异常中断 {len(errs)} 个 ====")
        for n, m in errs:
            print(f"  {n}: {m}")
    if npass != len(order):
        os._exit(1)


if __name__ == "__main__":
    main()
