#!/usr/bin/env python3
# capture 版: NPUGraph capture N 次 add/grouped_matmul + T0/T1 cycle buffer 读取
#
# 环境变量控制:
#   OP_TYPE=add|gemm   选择算子（默认 add）
#   OP_ITERS=101       capture 迭代次数（默认 101）
#   REPLAY_SHOTS=5     多拍回放次数（默认 5）：连续 replay 多次、每拍回读一次 buffer，
#                      每拍独立计算头开销指标后由汇总段取 median，滤除单拍随机波动
#                      （icache 冷热/核启动错峰/HBM 干扰）。每拍内部仍是完整 101 连发
#                      back-to-back 执行，不破坏执行流；拍间 host 回读耗时毫秒级可忽略
#
# 核数说明: 核数由安装的 wheel 决定（run_head_overhead.sh 通过 sed 改源码 + rebuild）
#   本脚本不控制核数，只负责 capture + cycle 数据导出
#
# T0/T1/T2 cycle 打点（配合 msprof + parse_stars_log.py 计算头开销）:
#   add:  T0 = kernel 第一行; T1 = prologue 之后（InitBuffer/SetGlobalBuffer/tiling 完成）; T2 = kernel 末尾（计算完成）
#   gemm: T0 = kernel 第一行; T1 = mm.Init/computeOp.Init/op.Init 完成后、第一个 MTE2 搬运前; T2 = op.Process 完成后
#   打点数据由算子侧写入持久 GM cycle buffer（capture 模式地址固化），replay 后由本脚本导出
#
# 导出格式（stderr，供后处理与 STARS 最近邻匹配）:
#   Shot<r> Block <bid>: T0=<abs> T1=<abs> T2=<abs> T1-T0=<diff> T2-T0=<diff>   (第 r 拍, r 从 1 起)
#   兼容旧格式 ReplayLast Block ...（同拍语义, 旧后处理可继续解析首拍）
#
# 注意:
#   - capture 模式 TASK_QUEUE_ENABLE 必须为 1（=2 与 NPUGraph capture 冲突）
#   - capture 阶段 kernel 只录不跑（cycle buffer 不写），每拍 replay 后 buffer 为该拍最后一次 launch 的数据
import os
import sys
import time
import statistics
import torch
import torch_npu
import ascend_ops

os.environ["TASK_QUEUE_ENABLE"] = "1"

# 可选设备号 (A3 环境默认 device 0, 若其 profiling 通道故障可换卡)
_dev = int(os.environ.get("DEVICE_ID", "0"))
if _dev != 0:
    torch.npu.set_device(_dev)
    print(f"using device {_dev}")

op_type = os.environ.get("OP_TYPE", "add")
op_iters = int(os.environ.get("OP_ITERS", "101"))
replay_shots = int(os.environ.get("REPLAY_SHOTS", "5"))
# SYS_CNT 频率 (cycle→us 换算): A5 实测 1000 MHz (时钟速率直测: 两次 replay 间隔 host 1s,
# 计数差 1e9; AIV 与 AIC 比值 1.000, SoC 全局计数器核类型无关)。
# 历史 283/587MHz 均为误标定 (依赖跨域匹配的间接推断), 已废弃。
# 环境变量 SYS_CNT_FREQ_MHZ 可整体覆盖; A3/910C 需上板重测
sys_cnt_freq = float(os.environ.get("SYS_CNT_FREQ_MHZ", "1000"))

# === 数据准备 + run 函数 ===
if op_type == "add":
    # add: fp16, 2 args (核数由 wheel 决定)
    # shape 均分设计: total = 336 x 21845 = lcm(56,48) x elementNumPerTile(fp16, UB 256KB/2/3/2)
    #   A5 (56 AIV): blockLength=131070=6 整 tile, 56 核均分
    #   A3 (48 AIV): blockLength=152915=7 整 tile, 48 核均分
    # 两平台均不触发 kernel 尾块处理分支 (tailTileElementNum=0), 消除尾块分支对
    # 头开销/块间方差的干扰 (旧 1000x1000: 末核 tail=17810 走尾块分支)
    # ADD_TOTAL 覆盖 shape (保持 21845 整数倍以维持 56/48 均分无尾块)
    _total = int(os.environ.get("ADD_TOTAL", str(336 * 21845)))  # 默认 7,339,920
    x = torch.randn(_total, dtype=torch.float16).npu()
    y = torch.randn(_total, dtype=torch.float16).npu()
    dummy = x
    def run():
        torch.ops.ascend_ops.add(x, y)
    read_cycle = lambda: torch.ops.ascend_ops.read_cycle(dummy).view(64, 3)

elif op_type == "gemm":
    # grouped_matmul: bf16 (Cube 核, 核数由 wheel 决定)
    # M=448 = 28(A5 Cube 满核) x 16(baseM): MN 切分满核调度 28 核;
    # A3 (24 Cube 核) 下 448 行同样切满 24 核, 无需平台分支
    # (K=16/N=16 保持小计算量, 聚焦头开销)
    # GMM_M/GMM_K/GMM_N 覆盖 shape (M 须为 16 的倍数且切满核)
    E, M, K, N, PERGROUP = 1, int(os.environ.get("GMM_M", "448")), int(os.environ.get("GMM_K", "16")), int(os.environ.get("GMM_N", "16")), int(os.environ.get("GMM_M", "448"))
    x_cpu = torch.rand(M, K, dtype=torch.bfloat16)
    weight_cpu = torch.rand(K, N, dtype=torch.bfloat16)
    x = [x_cpu.npu()]
    weight = [weight_cpu.npu()]
    dummy = x[0]
    gl = torch.zeros(E, dtype=torch.int64)
    total = PERGROUP
    for i in range(E):
        if total <= M:
            gl[i] = PERGROUP
            total += PERGROUP
        else:
            gl[i] = PERGROUP - (total - M)
            break
    group_list = gl.npu()
    def run():
        torch.ops.ascend_ops.grouped_matmul(
            x, weight, None, None, None, None, None,
            group_list, None, 0, 0, 0, 0, None
        )
    read_cycle = lambda: torch.ops.ascend_ops.gmm_read_cycle(dummy).view(64, 3)
else:
    raise ValueError(f"Unknown OP_TYPE: {op_type}")

print(f"capture {op_iters} iters ({op_type}, no warmup)...")
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    for _ in range(op_iters):
        run()

# gemm sync segfault 已修复 (grouped_matmul.cpp 悬垂引用修复), 统一走完整 replay 路径
torch.npu.synchronize()

# === 多拍回放: 每拍 replay(整图 101 连发 back-to-back) + 回读 cycle buffer ===
# 单拍快照受 icache 冷热/核启动错峰/HBM 干扰影响, 轮间波动 ~10-18%;
# 多拍独立样本 + 汇总段取 median 滤除离群拍 (预期波动 <3%)
# 注: 偶发个别 block 的 GM 标量值在单次 D2H 回读中滞后/为 0（D2H 可见性问题）：
#   - 0 值: 该 slot 本拍未写（55/56 现象）
#   - 旧值: 读到前几拍 kernel 写入的值（T2 写在 kernel 末尾最易滞后）
#     旧值形态: 整组旧(三元组自洽,仍可独立匹配 dispatch)/混合旧(slot 间拍次错位)
# 采用每 slot 独立收敛: 值连续 2 轮不变即接受(不管稳定在第几拍)。
# 拍次错位不影响单值指标——T0/T1/T2 各自与 STARS dispatch 最近邻独立匹配即自洽;
# 仅跨 slot 差值(T1-T0/T2-T0)要求同拍, 由下游软校验(单调)兜底
def read_cycle_merged(max_tries=12):
    accepted = {}   # (block, slot) -> value
    streak = {}     # (block, slot) -> 连续稳定轮数
    last = {}       # (block, slot) -> 上轮值
    for _ in range(max_tries):
        cyc = read_cycle()
        cur = {}
        for b in range(cyc.shape[0]):
            for j in range(3):
                v = int(cyc[b, j].item())
                if v > 0:
                    cur[(b, j)] = v
        for key, v in cur.items():
            if last.get(key) == v:
                streak[key] = streak.get(key, 0) + 1
                if streak[key] >= 2:
                    accepted[key] = v
            else:
                streak[key] = 1
        last = cur
        if cur and all(streak.get(k, 0) >= 2 for k in cur):
            break  # 全部 slot 稳定
        time.sleep(0.05)
    merged = {}
    for (b, j), v in accepted.items():
        if b not in merged:
            merged[b] = [0, 0, 0]
        merged[b][j] = v
    return {b: tuple(v) for b, v in merged.items()}

# warmup 一拍不计入样本（首拍含环境初始化, icache 冷）
print(f"warmup replay ({op_type}, {op_iters} iters)...")
g.replay()
torch.npu.synchronize()

shot_data = []   # 每拍: {block: (T0, T1, T2)}
t_all = time.time()
for shot in range(1, replay_shots + 1):
    t0 = time.time()
    g.replay()
    torch.npu.synchronize()
    elapsed = time.time() - t0
    merged = read_cycle_merged()
    shot_data.append(merged)
    print(f"shot {shot}/{replay_shots}: replay {elapsed*1000:.2f} ms, valid blocks {len(merged)}/64")
print(f"total {replay_shots} shots in {(time.time()-t_all)*1000:.1f} ms")

# === 导出各拍 T0/T1/T2 cycle 数据 (Shot<r> 前缀) ===
try:
    n_last = len(shot_data[-1]) if shot_data else 0
    print(f"valid blocks: {n_last}/64 (merged re-read, last shot)")
    shot_durs = []
    for idx, merged in enumerate(shot_data, start=1):
        valid = sorted(merged.items())
        if not valid:
            continue
        for b, (a, bb, c) in valid:
            print(f"Shot{idx} Block {b}: T0={a} T1={bb} T2={c} T1-T0={bb-a} T2-T0={c-a}", file=sys.stderr, flush=True)
        # 兼容旧后处理: 末拍同时输出 ReplayLast 前缀
        if idx == len(shot_data):
            for b, (a, bb, c) in valid:
                print(f"ReplayLast Block {b}: T0={a} T1={bb} T2={c} T1-T0={bb-a} T2-T0={c-a}", file=sys.stderr, flush=True)
        kernel_durs = [c - a for _, (a, bb, c) in valid]
        shot_durs.extend(kernel_durs)
    if shot_durs:
        print(f"shots T2-T0(kernel exec): mean={statistics.mean(shot_durs):.0f} cyc "
              f"({statistics.mean(shot_durs)/sys_cnt_freq:.3f} us) min={min(shot_durs)} max={max(shot_durs)} n={len(shot_durs)}")
    else:
        print("[WARN] cycle buffer 为空（kernel 未执行或打点未生效）")
except Exception as e:
    print(f"[WARN] 读取 cycle buffer 失败: {e}")
