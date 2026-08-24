"""对 pypto FAG kernel 做 profiling，定位 8192 场景下的瓶颈。

输出各 pipe 的耗时占比，用于判断是 cube 计算、vector 计算、GM 访存
(atomicAdd)，还是 CV 跨核同步在主导耗时。
"""

import math
import os
import sys

import pypto_pro.language as pl
import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attn_grad import (  # noqa: E501
    decide_swizzle,
    flash_attn_grad,
    FlashAttnGradTilingData,
)

B = int(os.environ.get("B", "1"))
N = int(os.environ.get("N", "1"))
S1 = int(os.environ.get("S1", "8192"))
S2 = int(os.environ.get("S2", "8192"))
# D 与 Dv 可不等长(Q/K 用 D，V/dO 用 Dv)；两者都要参与 tilingkey 分档
D = int(os.environ.get("D", "128"))
Dv = int(os.environ.get("DV", str(D)))
CORES = int(os.environ.get("CORES", "36"))
# 布局只影响 GM 视图，不改变计算量；两种布局的耗时应基本一致
LAYOUT = os.environ.get("FAG_LAYOUT", "BSND").upper()
IS_BNSD = LAYOUT == "BNSD"
# GQA：N 为 KV 的 head 数(N2)，Q 侧为 N1 = N2*G
G = int(os.environ.get("G", "1"))
N2 = N
N1 = N * G
OUT = os.environ.get("PROF_OUT", "/root/.claude/jobs/c66577db/tmp/prof_pypto")


def main():
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    # 按目标布局分配：BNSD 为 [B,N,S,D]
    _sq = (B, N1, S1, D) if IS_BNSD else (B, S1, N1, D)
    _skv = (B, N2, S2, D) if IS_BNSD else (B, S2, N2, D)
    # V/dO/y(attn_out) 的最后一维是 Dv，不是 D
    _sqv = (B, N1, S1, Dv) if IS_BNSD else (B, S1, N1, Dv)
    _skvv = (B, N2, S2, Dv) if IS_BNSD else (B, S2, N2, Dv)
    q = torch.randn(*_sq, device=device, dtype=torch.float16)
    k = torch.randn(*_skv, device=device, dtype=torch.float16)
    v = torch.randn(*_skvv, device=device, dtype=torch.float16)
    dy = torch.randn(*_sqv, device=device, dtype=torch.float16)
    y = torch.randn(*_sqv, device=device, dtype=torch.float16)
    scale = 1.0 / math.sqrt(D)
    # softmax_lse [B,N1,S1]：lse = max + log(sum(exp(s-max)))。
    # profiling 只关心耗时，取 0 即 max=0/sum=1 的等价情形。
    lse = torch.zeros(B, N1, S1, device=device, dtype=torch.float32)
    dq_ws = torch.zeros(*_sq, device=device, dtype=torch.float32)
    dk_ws = torch.zeros(*_skv, device=device, dtype=torch.float32)
    dv_ws = torch.zeros(*_skvv, device=device, dtype=torch.float32)
    # fp16 最终输出：kernel 的 post 阶段写(scale + cast)
    dq_o = torch.zeros(*_sq, device=device, dtype=torch.float16)
    dk_o = torch.zeros(*_skv, device=device, dtype=torch.float16)
    dv_o = torch.zeros(*_skvv, device=device, dtype=torch.float16)
    tiling = FlashAttnGradTilingData(
        b=B, s1=S1, s2=S2, n1=N1, n2=N2, d=D, dv=Dv, scaleValue=scale,
        gSize=G,
        viewD0Q=(B * N1) if IS_BNSD else B,
        viewD2Q=1 if IS_BNSD else N1,
        coefB0Q=N1 if IS_BNSD else 1,
        viewD0KV=(B * N2) if IS_BNSD else B,
        viewD2KV=1 if IS_BNSD else N2,
        coefB0KV=N2 if IS_BNSD else 1,
        coefN0=1 if IS_BNSD else 0,
        coefN2=0 if IS_BNSD else 1,
    )
    swizzle = decide_swizzle(B, N1, S1, S2, D, CORES, g_size=G)
    _force = os.environ.get("FAG_FORCE_SWIZZLE")
    if _force is not None:
        swizzle = int(_force)
    # dv 段按 Dv 宽(与 host 的 workspace 切分一致)
    ws_elems = (B * S1 * N1 * D) + (B * S2 * N2 * D) + (B * S2 * N2 * Dv)
    workspace = torch.zeros(ws_elems, device=device, dtype=torch.float32)

    def once():
        flash_attn_grad[
            None, CORES,
            # 新版 PTO 会严校验 launch key 必须覆盖 schema 全部字段，
            # 少一个就报 "do not match schema fields"，故 d_align/dv_align
            # 必须一起给出(取值即 tile 的分配列宽)。
            {"template": 0, "layout": 1 if IS_BNSD else 0,
             "has_atten_mask": 0, "swizzle": swizzle,
             "d_align": 128 if D <= 128 else 192,
             "dv_align": 128 if Dv <= 128 else 192},
            {"q": pl.DT_FP16},
        ](q, k, v, dy, y, lse,
          None, None, None, None, None, None, None,
          dq_o, dk_o, dv_o, workspace, tiling)

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
