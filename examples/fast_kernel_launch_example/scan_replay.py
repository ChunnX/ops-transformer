#!/usr/bin/env python3
# capture 版: NPUGraph capture N 次 add/grouped_matmul + T0/T1 cycle buffer 读取
#
# 环境变量控制:
#   OP_TYPE=add|gemm   选择算子（默认 add）
#   OP_ITERS=101       capture 迭代次数（默认 101）
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
#   ReplayLast Block <bid>: T0=<abs> T1=<abs> T2=<abs> T1-T0=<diff> T2-T0=<diff>
#
# 注意:
#   - capture 模式 TASK_QUEUE_ENABLE 必须为 1（=2 与 NPUGraph capture 冲突）
#   - capture 阶段 kernel 只录不跑（cycle buffer 不写），replay 后 buffer 为最后一次 replay 的数据
import os
import sys
import time
import statistics
import torch
import torch_npu
import ascend_ops

os.environ["TASK_QUEUE_ENABLE"] = "1"

op_type = os.environ.get("OP_TYPE", "add")
op_iters = int(os.environ.get("OP_ITERS", "101"))
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
    _total = 336 * 21845  # 7,339,920
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
    E, M, K, N, PERGROUP = 1, 448, 16, 16, 448
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
print(f"replay once ({op_type}, {op_iters} iters)...")
t0 = time.time()
g.replay()
torch.npu.synchronize()
elapsed = time.time() - t0
print(f"replay done in {elapsed*1000:.2f} ms")

# === 导出最后一次 replay 的 T0/T1/T2 cycle 数据 ===
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

try:
    merged = read_cycle_merged()
    valid = sorted(merged.items())
    n = len(valid)
    print(f"valid blocks: {n}/64 (merged re-read)")
    for b, (a, bb, c) in valid:
        print(f"ReplayLast Block {b}: T0={a} T1={bb} T2={c} T1-T0={bb-a} T2-T0={c-a}", file=sys.stderr, flush=True)
    if n > 0:
        kernel_durs = [c - a for _, (a, bb, c) in valid]
        print(f"last replay T2-T0(kernel exec): mean={statistics.mean(kernel_durs):.0f} cyc "
              f"({statistics.mean(kernel_durs)/sys_cnt_freq:.3f} us) min={min(kernel_durs)} max={max(kernel_durs)} n={n}")
    else:
        print("[WARN] cycle buffer 为空（kernel 未执行或打点未生效）")
except Exception as e:
    print(f"[WARN] 读取 cycle buffer 失败: {e}")
