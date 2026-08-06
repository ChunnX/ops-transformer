#!/usr/bin/env python3
"""mssanitizer Python 检测入口：小 shape 触发 Softmax CopyCompactFloatsUb tail DataCopy。"""
import os
import sys

import torch
import torch_npu

# 保证能 import 自定义 torch 扩展与 prepare_tensors
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(_ROOT, "torch_extension"))
sys.path.insert(0, "/data/kdy/code/analyse")

from attenres_lite import prepare_tensors
from cann_ops_transformer.ops.attn_res_fwd import attn_res_fwd


def main():
    device_id = int(os.environ.get("DEVICE_ID", "7"))
    torch_npu.npu.set_device(device_id)
    # B=N+1=3 → CopyCompactFloatsUb 只走 tail DataCopy(DataCopyParams)
    t, n, h = 2, 2, 7168
    ps, br, pw, nw, eps = prepare_tensors(t, n, h, seed=0)
    # need_backward=False：避开 DEBUG GetValue(invUb_)；Softmax 仍走 CopyCompactFloatsUb
    need_bwd = os.environ.get("NEED_BWD", "0") == "1"
    print(f"[mssan] call attn_res_fwd T={t} N={n} H={h} need_backward={need_bwd} device={device_id}",
          flush=True)
    out = attn_res_fwd(ps, br, pw, nw, eps, need_backward=need_bwd)
    torch.npu.synchronize()
    if isinstance(out, tuple):
        hidden, inv, probs = out
        print(f"[mssan] done hidden={tuple(hidden.shape)} inv={tuple(inv.shape)} probs={tuple(probs.shape)}",
              flush=True)
    else:
        print(f"[mssan] done hidden={tuple(out.shape)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
