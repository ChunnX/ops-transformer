"""对 pypto FAG kernel 做 profiling，定位 8192 场景下的瓶颈。

输出各 pipe 的耗时占比，用于判断是 cube 计算、vector 计算、GM 访存
(atomicAdd)，还是 CV 跨核同步在主导耗时。
"""

import math
import os
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_score_grad_pypto import (  # noqa: E501
    decide_swizzle,
    flash_attention_score_grad,
    FagTiling,
)

B = int(os.environ.get("B", "1"))
N = int(os.environ.get("N", "1"))
S1 = int(os.environ.get("S1", "8192"))
S2 = int(os.environ.get("S2", "8192"))
D = 128
CORES = int(os.environ.get("CORES", "36"))
OUT = os.environ.get("PROF_OUT", "/root/.claude/jobs/c66577db/tmp/prof_pypto")


def main():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    q = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, S2, N, D, device=device, dtype=torch.float16)
    dy = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)
    y = torch.randn(B, S1, N, D, device=device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    smax = torch.zeros(B * N, S1, device=device, dtype=torch.float32)
    ssum = torch.ones(B * N, S1, device=device, dtype=torch.float32)
    dq_ws = torch.zeros(B, S1, N, D, device=device, dtype=torch.float32)
    dk_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    dv_ws = torch.zeros(B, S2, N, D, device=device, dtype=torch.float32)
    # fp16 最终输出：kernel 的 post 阶段写(scale + cast)
    dq_o = torch.zeros(B, S1, N, D, device=device, dtype=torch.float16)
    dk_o = torch.zeros(B, S2, N, D, device=device, dtype=torch.float16)
    dv_o = torch.zeros(B, S2, N, D, device=device, dtype=torch.float16)
    tiling = FagTiling(
        b=B, n=N, s1=S1, s2=S2, d=D, scale_value=scale, core_num=CORES,
        enable_swizzle=decide_swizzle(B, N, S1, S2, D, CORES),
    )

    def once():
        flash_attention_score_grad[None, CORES](
            q, k, v, dy, y, smax, ssum, dq_ws, dk_ws, dv_ws,
            dq_o, dk_o, dv_o, tiling)

    # warmup + 编译
    once()
    torch.npu.synchronize()

    exp_cfg = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exp_cfg,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(OUT),
    ) as prof:
        for _ in range(3):
            once()
            prof.step()
    torch.npu.synchronize()
    print(f"profiling written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
