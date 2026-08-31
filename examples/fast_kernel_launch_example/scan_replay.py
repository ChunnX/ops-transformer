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
# T0/T1 cycle 打点（配合 msprof + parse_stars_log.py 计算头开销）:
#   add:  T0 = kernel 第一行; T1 = prologue 之后（InitBuffer/SetGlobalBuffer/tiling 完成）
#   gemm: T0 = kernel 第一行; T1 = mm.Init/computeOp.Init/op.Init 完成后、第一个 MTE2 搬运前
#   打点数据由算子侧写入持久 GM cycle buffer（capture 模式地址固化），replay 后由本脚本导出
#
# 导出格式（stderr，供后处理与 STARS 最近邻匹配）:
#   ReplayLast Block <bid>: T0=<abs> T1=<abs> T1-T0=<diff>
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

# === 数据准备 + run 函数 ===
if op_type == "add":
    # add: fp16, 2 args (核数由 wheel 决定)
    x = torch.randn(1000, 1000, dtype=torch.float16).npu()
    y = torch.randn(1000, 1000, dtype=torch.float16).npu()
    dummy = x
    def run():
        torch.ops.ascend_ops.add(x, y)
    read_cycle = lambda: torch.ops.ascend_ops.read_cycle(dummy).view(64, 2)

elif op_type == "gemm":
    # grouped_matmul: bf16 (Cube 核, 核数由 wheel 决定)
    # 小矩阵避开计算开销，聚焦头开销
    E, M, K, N, PERGROUP = 4, 16, 16, 16, 16
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
    read_cycle = lambda: torch.ops.ascend_ops.gmm_read_cycle(dummy).view(64, 2)
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

# === 导出最后一次 replay 的 T0/T1 cycle 数据 ===
# 注: 偶发个别 block 的 GM 标量值在单次 D2H 回读中为 0（55/56 现象，缺失 block 随机），
# 采用重读合并：连续 max_tries 轮读取，非零 slot 逐轮合并，连续两轮无新增即收敛，
# 保证满核 block 全量可见（n = 核数）
def read_cycle_merged(max_tries=6):
    merged = {}
    stable = 0
    for _ in range(max_tries):
        cyc = read_cycle()
        new = 0
        for b in range(cyc.shape[0]):
            t0 = int(cyc[b, 0].item())
            t1 = int(cyc[b, 1].item())
            if t0 > 0 and t1 > 0 and b not in merged:
                merged[b] = (t0, t1)
                new += 1
        if new == 0:
            stable += 1
            if stable >= 2:
                break
        else:
            stable = 0
        time.sleep(0.05)
    return merged

try:
    merged = read_cycle_merged()
    valid = sorted(merged.items())
    n = len(valid)
    print(f"valid blocks: {n}/64 (merged re-read)")
    for b, (a, bb) in valid:
        print(f"ReplayLast Block {b}: T0={a} T1={bb} T1-T0={bb-a}", file=sys.stderr, flush=True)
    if n > 0:
        prologues = [bb - a for _, (a, bb) in valid]
        print(f"last replay T1-T0: mean={statistics.mean(prologues):.0f} cyc "
              f"({statistics.mean(prologues)/283.0:.3f} us) min={min(prologues)} max={max(prologues)} n={n}")
    else:
        print("[WARN] cycle buffer 为空（kernel 未执行或打点未生效）")
except Exception as e:
    print(f"[WARN] 读取 cycle buffer 失败: {e}")
