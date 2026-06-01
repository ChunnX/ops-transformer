# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import numpy as np
import random
from torch_npu.testing.testcase import TestCase, run_tests
from common.examples import aclnn_example_runner
from .hc_pre_golden import create_hc_pre_inputs, data_compare, hc_pre_golden


np.random.seed(121)
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

def run_hc_pre_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
    hc_scale_npu = hc_scale.to("npu:%s" % DEVICE_ID)
    hc_base_npu = hc_base.to("npu:%s" % DEVICE_ID)
    hc_fn_npu = hc_fn.to("npu:%s" % DEVICE_ID)
    x_npu = x.to("npu:%s" % DEVICE_ID)

    golden_y_out, golden_post_out, golden_comb_frag_out = hc_pre_golden(
        x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    npu_y_out, npu_post_out, npu_comb_frag_out = aclnn_example_runner.hc_pre(
        x_npu, hc_fn_npu, hc_scale_npu, hc_base_npu, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps)

    compare_y = data_compare(golden_y_out.cpu().float().numpy(), npu_y_out.cpu().float().numpy())
    compare_post = data_compare(golden_post_out.cpu().numpy(), npu_post_out.cpu().float().numpy())
    compare_comb_frag = data_compare(golden_comb_frag_out.cpu().numpy(), npu_comb_frag_out.cpu().float().numpy(), pct_thd=0.02
    )

    return compare_y, compare_post, compare_comb_frag

class TestCustomHcPre(TestCase):
    def _run_and_check_case(self, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps):
        compare_y, compare_post, compare_comb_frag = run_hc_pre_case(
            x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)
        assert(compare_y[0] == "Pass")
        assert(compare_post[0] == "Pass")
        assert(compare_comb_frag[0] == "Pass")

    def test_hc_pre_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 192
        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 7168]
        for d in d_list:
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((b, s, hc_mult, d), hc_mix, hc_mult)
            print(f'======================== PTA eager test d={d} ========================')
            self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_large_bs_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6
        x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((8193, hc_mult, 4096), hc_mix, hc_mult)

        print('======================== PTA large-bs eager regression test ========================')
        self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_small_bs_eager(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6
        x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((3079, hc_mult, 4096), hc_mix, hc_mult)

        print('======================== PTA large-bs eager regression test ========================')
        self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_fusion_bs_eager(self):
        """Ascend950: bs > 512 and divisible by 256, routed to HcPre fused op."""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        # bs values: <=512 range + >=8192 and divisible by 256
        bs_list = [
            # bs <= 512 range (always fused on 950)
            1, 16, 32, 64, 128, 256, 512,
            # bs >= 8192, divisible by 256 (fused)
            8192,      # power of 2, minimum fused
            10240,     # 20 * 512
            12288,     # 24 * 512
            16384,     # power of 2
            20480,     # 40 * 512
            24576,     # 48 * 512
            32768,     # power of 2
            40960,     # 80 * 512
            49152,     # 96 * 512
            64000,     # upper bound, 250 * 256
        ]

        for bs in bs_list:
            seed = bs
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(
                (bs, hc_mult, 4096), hc_mix, hc_mult, seed=seed)
            print(f'=== Ascend950 fusion bs eager: bs={bs}, d=4096 ===')
            self._run_and_check_case(
                x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_ascend950_fusion_bs_extend_d_eager(self):
        """Ascend950: bs > 512 and divisible by 256, with d=7168 (extended D)."""
        torch_npu.npu.set_device(int(DEVICE_ID))
        soc_name = torch.npu.get_device_properties().name
        if not soc_name.startswith("Ascend950"):
            self.skipTest("This regression case only applies to Ascend950.")

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        bs_list = [
            1, 128, 256, 512, 8192, 16384, 32768, 64000,
        ]

        for bs in bs_list:
            seed = bs + 7168
            x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs(
                (bs, hc_mult, 7168), hc_mix, hc_mult, seed=seed)
            print(f'=== Ascend950 fusion bs eager (d=7168): bs={bs} ===')
            self._run_and_check_case(
                x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

    def test_hc_pre_generalized_d_eager(self):
        """测试 d=4096/7168，所有芯片通用"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        hc_mix = 24
        hc_mult = 4
        hc_sinkhorn_iters = 20
        hc_eps = 1e-6
        norm_eps = 1e-6

        d_list = [4096, 7168]

        rng = random.Random(2025)
        bs_candidates = []
        for _ in range(8):
            bs_candidates.append(rng.randint(1, 32768))
        bs_candidates += [1, 128, 1024, 4096, 32768]
        seen = set()
        bs_list = []
        for v in bs_candidates:
            if v not in seen:
                seen.add(v)
                bs_list.append(v)

        for d in d_list:
            for bs in bs_list:
                seed = d * 100 + bs
                x, hc_fn, hc_scale, hc_base = create_hc_pre_inputs((bs, hc_mult, d), hc_mix, hc_mult, seed=seed)
                print(f'=== hc_pre generalized D eager: d={d}, bs={bs} ===')
                self._run_and_check_case(x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, norm_eps, hc_eps)

if __name__ == "__main__":
    run_tests()
