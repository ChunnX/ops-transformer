#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 v1/v2b 级联 stage-2 失败: 分段定位(pre/post/res) + 中间量来源对照。

实验设计:
  A) NPU-mids:   反向 kernel 与 golden 均消费 NPU 正向中间量 (复现失败场景)
  B) torch-mids: 反向 kernel 与 golden 均消费 forward_ref torch 中间量 (隔离 NPU 中间量影响)
  若 B 过 A 挂 -> NPU 正向中间量噪声(含 sum/norm 不一致噪声)触发反向误差放大;
  若 B 也挂    -> 与中间量来源无关, 是该随机数据下 kernel 自身精度问题。
"""
import os
import sys

REPO = "/home/ops-transformer-ds-v4.1"
sys.path.insert(0, os.path.join(REPO, "build_out"))

import torch
import torch_npu  # noqa: F401

import verify_mhc_bwd as V
from cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn_premix_backward import (  # noqa: E402
    mhc_pre_sinkhorn_premix_backward,
)


def seg_stats(tag, got, gold):
    g, a = gold.float(), got.float()
    for name, sl in (("pre[:n]", slice(0, V.N)), ("post[n:2n]", slice(V.N, 2 * V.N)),
                     ("res[2n:]", slice(2 * V.N, None))):
        gg, aa = g[sl], a[sl]
        if gg.numel() == 0:
            continue
        err = (aa - gg).abs()
        print("    {:<10}.{:.<9} max_abs={:.3e} gold_absmax={:.3e} worst_rel={:.2e}".format(
            tag, name, err.max().item(), gg.abs().max().item(),
            err.max().item() / max(gg.abs().max().item(), 1e-12)))


def run_variant(label, data, has_pm, mids):
    (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
     pre_mix, grad_pre, hc_eps) = data
    h_pre, hcbn, inv_rms, sum_out, norm_out = mids
    golds = V.backward_golden(
        grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
        h_pre, hcbn, inv_rms, sum_out, norm_out, pre_mix, grad_pre, hc_eps)
    outs = mhc_pre_sinkhorn_premix_backward(
        grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
        h_pre, hcbn, inv_rms, sum_out, norm_out, pre_mix, grad_pre, hc_eps)
    torch.npu.synchronize()
    print("  == {} ==".format(label))
    ga, gga = outs[2].float(), golds[2].float()
    for i in range(3):
        print("    grad_alpha[{}]: got={:.<10g} gold={:.<10g} abs={:.3e} rel={:.2e}".format(
            i, ga[i].item(), gga[i].item(), abs(ga[i] - gga[i]).item(),
            abs(ga[i] - gga[i]).item() / max(abs(gga[i].item()), 1e-12)))
    seg_stats("grad_bias", outs[3], golds[3])
    seg_stats("grad_phi", outs[1], golds[1])
    V.compare("grad_x", outs[0], golds[0], V.TOL_BF16)
    V.compare("grad_phi", outs[1], golds[1], V.TOL_FP32)
    V.compare("grad_alpha", outs[2], golds[2], V.TOL_FP32)
    V.compare("grad_bias", outs[3], golds[3], V.TOL_FP32)
    if has_pm:
        V.compare("grad_pre_mix", outs[4], golds[4], V.TOL_FP32)


def run_scene(scene, has_pm, has_gpr, data):
    print("\n##### scene {} (pre_mix={}, grad_pre={}) #####".format(
        scene, "Y" if has_pm else "N", "Y" if has_gpr else "N"))
    (grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias,
     pre_mix, grad_pre, hc_eps) = data
    refs = V.forward_ref(x, phi, alpha, bias, V.N, V.ITERS, hc_eps)
    fwd = V.FWD_OP.mhc_pre_sinkhorn_premix(
        x, phi, alpha, bias, pre_mix, V.N, V.ITERS, hc_eps, hc_eps, True)
    torch.npu.synchronize()
    npu_mids = fwd[3:8]
    run_variant("A: NPU-mids (cascade)", data, has_pm, npu_mids)
    run_variant("B: torch-mids (control)", data, has_pm, refs)


def main():
    V.FWD_OP = V.mhc_pre_sinkhorn_premix_op_builder.load()
    # 按 verify 顺序重放 4 个场景数据生成(仅 RNG 消费者), 保证与失败 run 同数据
    datas = {}
    for scene, pm, gpr in (("v1", False, False), ("v1b", False, True),
                           ("v2", True, True), ("v2b", True, False)):
        datas[scene] = V.make_case(pm, gpr)
    run_scene("v1", False, False, datas["v1"])
    run_scene("v2b", True, False, datas["v2b"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
