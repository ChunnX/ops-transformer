# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
import argparse
import random
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests


np.random.seed(121)
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(npu_out, cpu_out, diff_thd=0.005, pct_thd=0.005, max_diff_hd=0.001):
    real_data = npu_out.flatten()
    data_compe = cpu_out.flatten()
    start = 0
    end = real_data.size - 1
    max_error = 0
    result = "Failed"
    if real_data.size != data_compe.size:
        return result, 0.0, max_error
    
    split_count = int(end - start + 1) if end != start else 1
    diff_abs = np.abs(np.subtract(real_data.astype(np.float32), data_compe.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 data_compe[diff_index].astype(np.float32), diff_thd)
    
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]
    error_cnt = err_diff.size

    fulfill_num = split_count - error_cnt
    fulfill_percent = float(fulfill_num) / float(split_count) * 100.0

    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"

    return result, fulfill_percent, max_error


def hc_split_sinkhorn_torch_full(
        mixes: torch.Tensor, 
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6,
        need_backward: bool = True):
    pre, post, comb = mixes.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
    comb = comb.unflatten(-1, (hc_mult, hc_mult))

    pre = F.sigmoid(pre * hc_scale[0] + hc_base[:hc_mult].unsqueeze(0).unsqueeze(0)) + eps
    post = 2 * F.sigmoid(post * hc_scale[1] + hc_base[hc_mult:2 * hc_mult].unsqueeze(0).unsqueeze(0))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult:].view(hc_mult, hc_mult).unsqueeze(0).unsqueeze(0)

    comb = comb.softmax(-1) + eps
    
    sum_out_list = []
    norm_out_list = []
    
    if need_backward:
        norm_out_list.append(comb)
    
    col_sum = comb.sum(-2, keepdim=True)
    
    if need_backward:
        sum_out_list.append(col_sum.squeeze(-2))
    
    comb = comb / (col_sum + eps)
    
    if need_backward:
        norm_out_list.append(comb)
    
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(-1, keepdim=True)
        
        if need_backward:
            sum_out_list.append(row_sum.squeeze(-1))
        
        comb = comb / (row_sum + eps)
        
        if need_backward:
            norm_out_list.append(comb)
        
        col_sum = comb.sum(-2, keepdim=True)
        
        if need_backward:
            sum_out_list.append(col_sum.squeeze(-2))
        
        comb = comb / (col_sum + eps)
        
        if need_backward:
            norm_out_list.append(comb)
    
    if need_backward:
        sum_out = torch.stack(sum_out_list, dim=0)
        norm_out = torch.stack(norm_out_list, dim=0)
        return pre, post, comb, sum_out, norm_out
    else:
        return pre, post, comb, None, None


def _hc_pre_full(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, 
                 hc_mult: int, hc_sinkhorn_iters: int, norm_eps: float, hc_eps: float, 
                 need_backward: bool = True):
    shape, dtype = x.size(), x.dtype
    if x.dim() == 4:
        x_flat = x.flatten(2).float()
    elif x.dim() == 3:
        x_flat = x.flatten(1).float()
    else:
        raise ValueError(f"x must be 3D or 4D, got {x.dim()}D")
    
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    inv_rms = rsqrt.unsqueeze(-1)
    
    mixes = F.linear(x_flat, hc_fn)
    hc_before_norm = mixes
    mixes = mixes * rsqrt

    pre, post, comb, sum_out, norm_out = hc_split_sinkhorn_torch_full(
        mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, need_backward)
    
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    
    if need_backward:
        return y.to(dtype), post, comb, pre, hc_before_norm, inv_rms, sum_out, norm_out
    else:
        return y.to(dtype), post, comb, None, None, None, None, None


def hc_split_sinkhorn_torch(
        mixes: torch.Tensor, 
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6):
    pre, post, comb, _, _ = hc_split_sinkhorn_torch_full(
        mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, need_backward=False)
    return pre, post, comb


def _hc_pre(x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, 
            hc_mult: int, hc_sinkhorn_iters: int, norm_eps: float, hc_eps: float):
    y, post, comb, _, _, _, _, _ = _hc_pre_full(
        x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps, need_backward=False)
    return y, post, comb


class TestCustomHcPre(TestCase):
    def test_hc_pre_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        d = 4096
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        np.random.seed(42)

        hc_scale = torch.tensor(np.random.uniform(0, 2, (3))).to(torch.float32)
        hc_base = torch.tensor(np.random.uniform(0, 2, (hc_mix))).to(torch.float32)
        hc_fn = torch.tensor(np.random.uniform(0, 2, (hc_mix, hc_mult * d))).to(torch.float32)
        x = torch.tensor(np.random.uniform(0, 2, (b, s, hc_mult, d))).to(torch.bfloat16)

        hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
        hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
        hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
        x_npu = x.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager test ========================')

        golden_yOut, golden_postOut, golden_comb_fragOut = _hc_pre(
            x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

        npu_yOut, npu_postOut, npu_comb_fragOut = torch.ops.custom.npu_hc_pre(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, hc_mult=hc_mult, 
            hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps)

        npu_yOut_cpu = npu_yOut.cpu().float().numpy()
        npu_postOut_cpu = npu_postOut.cpu().float().numpy()
        npu_comb_fragOut_cpu = npu_comb_fragOut.cpu().float().numpy()

        golden_yOut_cpu = golden_yOut.cpu().float().numpy()
        golden_postOut_cpu = golden_postOut.cpu().numpy()
        golden_comb_fragOut_cpu = golden_comb_fragOut.cpu().numpy()

        compare_y = data_compare(golden_yOut_cpu, npu_yOut_cpu)
        compare_post = data_compare(golden_postOut_cpu, npu_postOut_cpu)
        compare_comb_frag = data_compare(golden_comb_fragOut_cpu, npu_comb_fragOut_cpu)

        assert(compare_y[0] == "Pass")
        assert(compare_post[0] == "Pass")
        assert(compare_comb_frag[0] == "Pass")

    def test_hc_pre_eager_full_output(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        d = 4096
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6
        need_backward = True

        np.random.seed(42)

        hc_scale = torch.tensor(np.random.uniform(0, 2, (3))).to(torch.float32)
        hc_base = torch.tensor(np.random.uniform(0, 2, (hc_mix))).to(torch.float32)
        hc_fn = torch.tensor(np.random.uniform(0, 2, (hc_mix, hc_mult * d))).to(torch.float32)
        x = torch.tensor(np.random.uniform(0, 2, (b, s, hc_mult, d))).to(torch.bfloat16)

        hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
        hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
        hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
        x_npu = x.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA eager test (8 outputs) ========================')

        (golden_hin, golden_hPost, golden_hRes, golden_hPre, golden_hcBeforeNorm, 
         golden_invRms, golden_sumOut, golden_normOut) = _hc_pre_full(
            x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps, need_backward)

        hPre_npu = torch.empty((b, s, hc_mult), dtype=torch.float32, device="npu:%s" % DEVICE_ID)
        hcBeforeNorm_npu = torch.empty((b, s, hc_mix), dtype=torch.float32, device="npu:%s" % DEVICE_ID)
        invRms_npu = torch.empty((b, s, 1), dtype=torch.float32, device="npu:%s" % DEVICE_ID)
        sumOut_npu = torch.empty((hc_sinkhorn_iters * 2, b, s, hc_mult), dtype=torch.float32, device="npu:%s" % DEVICE_ID)
        normOut_npu = torch.empty((hc_sinkhorn_iters * 2, b, s, hc_mult, hc_mult), dtype=torch.float32, device="npu:%s" % DEVICE_ID)

        npu_hin, npu_hPost, npu_hRes, npu_hPre, npu_hcBeforeNorm, npu_invRms, npu_sumOut, npu_normOut = \
            torch.ops.custom.npu_hc_pre(
                x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, 
                hc_mult=hc_mult, hc_sinkhorn_iters=hc_sinkhorn_iters, 
                norm_eps=norm_eps, hc_eps=hc_eps, need_backward=need_backward,
                hPre=hPre_npu, hcBeforeNorm=hcBeforeNorm_npu, invRms=invRms_npu,
                sumOut=sumOut_npu, normOut=normOut_npu)

        compare_hin = data_compare(golden_hin.cpu().float().numpy(), npu_hin.cpu().float().numpy())
        compare_hPost = data_compare(golden_hPost.cpu().numpy(), npu_hPost.cpu().float().numpy())
        compare_hRes = data_compare(golden_hRes.cpu().numpy(), npu_hRes.cpu().float().numpy())
        compare_hPre = data_compare(golden_hPre.cpu().numpy(), npu_hPre.cpu().float().numpy())
        compare_hcBeforeNorm = data_compare(golden_hcBeforeNorm.cpu().numpy(), npu_hcBeforeNorm.cpu().float().numpy())
        compare_invRms = data_compare(golden_invRms.cpu().numpy(), npu_invRms.cpu().float().numpy())
        compare_sumOut = data_compare(golden_sumOut.cpu().numpy(), npu_sumOut.cpu().float().numpy())
        compare_normOut = data_compare(golden_normOut.cpu().numpy(), npu_normOut.cpu().float().numpy())

        print(f"hin: {compare_hin[0]}, {compare_hin[1]:.2f}%")
        print(f"hPost: {compare_hPost[0]}, {compare_hPost[1]:.2f}%")
        print(f"hRes: {compare_hRes[0]}, {compare_hRes[1]:.2f}%")
        print(f"hPre: {compare_hPre[0]}, {compare_hPre[1]:.2f}%")
        print(f"hcBeforeNorm: {compare_hcBeforeNorm[0]}, {compare_hcBeforeNorm[1]:.2f}%")
        print(f"invRms: {compare_invRms[0]}, {compare_invRms[1]:.2f}%")
        print(f"sumOut: {compare_sumOut[0]}, {compare_sumOut[1]:.2f}%")
        print(f"normOut: {compare_normOut[0]}, {compare_normOut[1]:.2f}%")

        assert(compare_hin[0] == "Pass")
        assert(compare_hPost[0] == "Pass")
        assert(compare_hRes[0] == "Pass")
        assert(compare_hPre[0] == "Pass")
        assert(compare_hcBeforeNorm[0] == "Pass")
        assert(compare_invRms[0] == "Pass")
        assert(compare_sumOut[0] == "Pass")
        assert(compare_normOut[0] == "Pass")

    def test_hc_pre_graph(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
                npu_yOut, npu_postOut, npu_comb_fragOut = torch.ops.custom.npu_hc_pre(
                    x, hc_fn, hc_scale, hc_base, hc_mult=hc_mult, 
                    hc_sinkhorn_iters=hc_sinkhorn_iters, norm_eps=norm_eps, hc_eps=hc_eps
                )
                return npu_yOut, npu_postOut, npu_comb_fragOut

        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        d = 4096
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        np.random.seed(42)

        hc_scale = torch.tensor(np.random.uniform(0, 2, (3))).to(torch.float32)
        hc_base = torch.tensor(np.random.uniform(0, 2, (hc_mix))).to(torch.float32)
        hc_fn = torch.tensor(np.random.uniform(0, 2, (hc_mix, hc_mult * d))).to(torch.float32)
        x = torch.tensor(np.random.uniform(0, 2, (b, s, hc_mult, d))).to(torch.bfloat16)

        hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
        hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
        hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
        x_npu = x.to("npu:%s" % DEVICE_ID)

        print(f'======================== PTA graph test ========================')

        golden_yOut, golden_postOut, golden_comb_fragOut = _hc_pre(
            x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

        npu_mode = Network().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_yOut, npu_postOut, npu_comb_fragOut = npu_mode(
            x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

        npu_yOut_cpu = npu_yOut.cpu().float().numpy()
        npu_postOut_cpu = npu_postOut.cpu().float().numpy()
        npu_comb_fragOut_cpu = npu_comb_fragOut.cpu().float().numpy()

        golden_yOut_cpu = golden_yOut.cpu().float().numpy()
        golden_postOut_cpu = golden_postOut.cpu().numpy()
        golden_comb_fragOut_cpu = golden_comb_fragOut.cpu().numpy()

        compare_y = data_compare(golden_yOut_cpu, npu_yOut_cpu)
        compare_post = data_compare(golden_postOut_cpu, npu_postOut_cpu)
        compare_comb_frag = data_compare(golden_comb_fragOut_cpu, npu_comb_fragOut_cpu)

        assert(compare_y[0] == "Pass")
        assert(compare_post[0] == "Pass")
        assert(compare_comb_frag[0] == "Pass")


if __name__ == "__main__":
    run_tests()
