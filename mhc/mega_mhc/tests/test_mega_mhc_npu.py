# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Hardware acceptance tests. Run after installing the operator and Torch package.
Tolerance is a provisional baseline (HF32 is intentionally looser), not a measured
performance or accuracy guarantee. No 910 bitwise equality is asserted.
"""

import unittest
import importlib
import os
import ctypes

try:
    import torch
    import torch_npu  # registers the NPU backend

    cann_ops_transformer = importlib.import_module(
        os.getenv("MEGA_MHC_TORCH_PACKAGE", "cann_ops_transformer")
    )
    AVAILABLE = torch.npu.is_available()
except ImportError:
    AVAILABLE = False


def reference(
    h_out,
    residual,
    h_post,
    h_res,
    phi,
    alpha,
    bias,
    gamma_1,
    gamma_0,
    gamma_2=None,
    norm_eps=1e-6,
    mhc_pre_norm_eps=1e-6,
    hc_eps=1e-6,
):
    """Independent unfused FP32 reference, preserving every stage's input-type cast."""
    dtype = h_out.dtype

    def rms(x, gamma):
        f = x.float()
        return (
            f * torch.rsqrt(f.square().mean(-1, keepdim=True) + norm_eps) * gamma
        ).to(dtype)

    x1 = rms(h_out, gamma_0)
    post = h_post.unsqueeze(-1) * x1.float().unsqueeze(-2)
    for source in range(4):
        post = post + h_res[..., source, :].unsqueeze(-1) * residual[
            ..., source, :
        ].float().unsqueeze(-2)
    x2 = post.to(dtype)
    flat = x2.flatten(-2)
    if gamma_2 is not None:
        flat = rms(flat, gamma_2)
        x2 = flat.reshape_as(x2)
    f = flat.float()
    projected = (f @ phi.T) * torch.rsqrt(
        f.square().mean(-1, keepdim=True) + mhc_pre_norm_eps
    )
    pre = torch.sigmoid(alpha[0] * projected[..., :4] + bias[:4]) + hc_eps
    post_new = 2 * torch.sigmoid(alpha[1] * projected[..., 4:8] + bias[4:8])
    res_new = (alpha[2] * projected[..., 8:] + bias[8:]).reshape(
        *h_out.shape[:-1], 4, 4
    )
    h_in = torch.zeros_like(h_out, dtype=torch.float32)
    for head in range(4):
        h_in = h_in + pre[..., head, None] * x2[..., head, :].float()
    return rms(h_in.to(dtype), gamma_1), post_new, res_new, x2


@unittest.skipUnless(
    AVAILABLE, "requires installed CANN operator, Torch extension and Ascend 950"
)
class TestMegaMhcNpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Use the selected idle device and refuse to run a 950-only kernel on another architecture."""
        device = int(os.getenv("MEGA_MHC_NPU_DEVICE", "1"))
        if "950" not in torch.npu.get_device_name(device):
            raise unittest.SkipTest("mega_mhc acceptance requires Ascend 950")
        torch.npu.set_device(device)
        torch.set_num_threads(4)
        limit = os.getenv("MEGA_MHC_TEST_AIC_LIMIT")
        if limit is not None:
            # Process-local CANN quota, applied before dispatch; production tiling
            # still obtains the available cores through the platform interface.
            aic_limit = int(limit)
            if aic_limit <= 0:
                raise ValueError("MEGA_MHC_TEST_AIC_LIMIT must be positive")
            runtime = ctypes.CDLL("libascendcl.so")
            runtime.aclrtGetDeviceResLimit.argtypes = [
                ctypes.c_int32,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            runtime.aclrtSetDeviceResLimit.argtypes = [
                ctypes.c_int32,
                ctypes.c_int,
                ctypes.c_uint32,
            ]

            def set_limit(kind, value):
                status = runtime.aclrtSetDeviceResLimit(device, kind, value)
                if status != 0:
                    raise RuntimeError(
                        f"aclrtSetDeviceResLimit({kind}, {value}) failed: {status}"
                    )

            for kind, value in ((0, aic_limit), (1, 2 * aic_limit)):
                previous = ctypes.c_uint32()
                status = runtime.aclrtGetDeviceResLimit(
                    device, kind, ctypes.byref(previous)
                )
                if status != 0 or value > previous.value:
                    raise RuntimeError(
                        f"invalid test resource limit: kind={kind}, value={value}, "
                        f"available={previous.value}, status={status}"
                    )
                cls.addClassCleanup(set_limit, kind, previous.value)
                set_limit(kind, value)
            print(
                f"TEST_RESOURCE_LIMIT AIC={aic_limit} AIV={2 * aic_limit}", flush=True
            )
        if not callable(getattr(cann_ops_transformer, "mega_mhc", None)):
            raise RuntimeError(
                f"mega_mhc was not collected in {cann_ops_transformer.__file__}"
            )

    def test_precision_and_fixed_configuration_determinism(self):
        """Cover dtype/mode/stages, short/tail/multiwave M, both ranks, and strided inputs."""
        torch.manual_seed(17)
        for dtype in (torch.float16, torch.bfloat16):
            for d in (2560, 5120, 6144, 7168):
                for mid in (False, True):
                    for mode in (0, 1):
                        for leading in (
                            (1,),
                            (15,),
                            (16,),
                            (17,),
                            (31,),
                            (32,),
                            (33,),
                            (64,),
                            (128,),
                            (160,),
                            (192,),
                            (512,),
                            (768,),
                            (1024,),
                            (1536,),
                            (1537,),
                            (2048,),
                            (4096,),
                            (3072,),
                            (3073,),
                            (2, 17),
                        ):
                            with self.subTest(
                                dtype=dtype, d=d, mid=mid, mode=mode, leading=leading
                            ):
                                print(
                                    f"CASE dtype={dtype} D={d} mid={mid} mode={mode} leading={leading}",
                                    flush=True,
                                )
                                # Noncontiguous h_out checks the ACLNN Contiguous path.
                                h_out = torch.randn((*leading, d * 2), dtype=dtype)[
                                    ..., ::2
                                ]
                                inputs = [
                                    h_out,
                                    torch.randn((*leading, 4, d), dtype=dtype),
                                    torch.randn((*leading, 4)) * 0.1,
                                    torch.randn((*leading, 4, 4)) * 0.1,
                                    torch.randn((24, 4 * d)) / (4 * d) ** 0.5,
                                    torch.tensor([0.1, 0.2, 0.3]),
                                    torch.randn(24) * 0.1,
                                    0.9 + 0.2 * torch.rand(d),
                                    0.9 + 0.2 * torch.rand(d),
                                ]
                                gamma2 = 0.9 + 0.2 * torch.rand(4 * d) if mid else None
                                expected = reference(*inputs, gamma_2=gamma2)
                                device_inputs = [x.npu() for x in inputs]
                                # Device conversion may make a view contiguous; force a strided NPU view too.
                                doubled = torch.stack(
                                    (device_inputs[0], device_inputs[0]), dim=-1
                                )
                                device_inputs[0] = doubled[..., 0]
                                kwargs = {
                                    "gamma_2": gamma2.npu() if mid else None,
                                    "compute_mode": mode,
                                }
                                actual = cann_ops_transformer.mega_mhc(
                                    *device_inputs[:8],
                                    None,
                                    *device_inputs[8:],
                                    **kwargs,
                                )
                                repeated = cann_ops_transformer.mega_mhc(
                                    *device_inputs[:8],
                                    None,
                                    *device_inputs[8:],
                                    **kwargs,
                                )
                                torch.npu.synchronize()
                                tolerance = (
                                    0.02
                                    if dtype == torch.bfloat16 or mode == 1
                                    else 0.005
                                )
                                self.assertEqual(len(actual), 5)
                                self.assertIsNone(actual[4])
                                for result, repeat, want in zip(
                                    actual[:4], repeated[:4], expected
                                ):
                                    host_result = result.cpu()
                                    torch.testing.assert_close(
                                        host_result,
                                        want,
                                        atol=tolerance,
                                        rtol=tolerance,
                                    )
                                    self.assertTrue(
                                        torch.equal(host_result, repeat.cpu())
                                    )
                                print("CASE_PASS", flush=True)

    def test_independent_pre_norm_epsilon(self):
        """A distinct Pre epsilon changes gates but must not change the upstream x2."""
        torch.manual_seed(71)
        m, d = 34, 2560
        for dtype in (torch.float16, torch.bfloat16):
            for mid in (False, True):
                inputs = [
                    torch.randn(m, d, dtype=dtype),
                    torch.randn(m, 4, d, dtype=dtype),
                    torch.randn(m, 4) * 0.1,
                    torch.randn(m, 4, 4) * 0.1,
                    torch.randn(24, 4 * d) / (4 * d) ** 0.5,
                    torch.tensor([0.1, 0.2, 0.3]),
                    torch.randn(24) * 0.1,
                    0.9 + 0.2 * torch.rand(d),
                    0.9 + 0.2 * torch.rand(d),
                ]
                # Cover three-dimensional inputs in addition to the flat-token matrix tests.
                inputs[:4] = [x.reshape(2, 17, *x.shape[1:]) for x in inputs[:4]]
                gamma2 = 0.9 + 0.2 * torch.rand(4 * d) if mid else None
                expected = reference(
                    *inputs,
                    gamma_2=gamma2,
                    norm_eps=1e-3,
                    mhc_pre_norm_eps=0.25,
                    hc_eps=0.01,
                )
                device_inputs = [x.npu() for x in inputs]
                for mode in (0, 1):
                    kwargs = dict(
                        gamma_2=gamma2.npu() if mid else None,
                        norm_eps=1e-3,
                        hc_eps=0.01,
                        compute_mode=mode,
                    )
                    actual = cann_ops_transformer.mega_mhc(
                        *device_inputs[:8],
                        None,
                        *device_inputs[8:],
                        mhc_pre_norm_eps=0.25,
                        **kwargs,
                    )
                    default_pre = cann_ops_transformer.mega_mhc(
                        *device_inputs[:8], None, *device_inputs[8:], **kwargs
                    )
                    torch.npu.synchronize()
                    tolerance = 0.02 if dtype == torch.bfloat16 or mode == 1 else 0.005
                    self.assertEqual(len(actual), 5)
                    self.assertIsNone(actual[4])
                    for got, want in zip(actual[:4], expected):
                        torch.testing.assert_close(
                            got.cpu(), want, atol=tolerance, rtol=tolerance
                        )
                    self.assertTrue(torch.equal(actual[3], default_pre[3]))
                    self.assertFalse(torch.equal(actual[1], default_pre[1]))

    def test_zero_small_large_and_epsilon(self):
        """Exercise RMS scaling and zero hc epsilon across all 16 dtype/D/stage/mode families."""
        torch.manual_seed(29)
        for dtype in (torch.float16, torch.bfloat16):
            for d in (2560, 5120):
                for mid in (False, True):
                    for mode in (0, 1):
                        for scale, epsilon in (
                            (0.0, 1e-6),
                            (1e-6, 1e-12),
                            (1.0, 1e-3),
                            (1000.0, 1e-6),
                        ):
                            with self.subTest(
                                dtype=dtype,
                                d=d,
                                mid=mid,
                                mode=mode,
                                scale=scale,
                                epsilon=epsilon,
                            ):
                                print(
                                    f"EDGE dtype={dtype} D={d} mid={mid} mode={mode} "
                                    f"scale={scale} epsilon={epsilon}",
                                    flush=True,
                                )
                                inputs = [
                                    (torch.randn(2, d) * scale).to(dtype),
                                    (torch.randn(2, 4, d) * scale).to(dtype),
                                    torch.full((2, 4), 0.2),
                                    torch.eye(4).expand(2, 4, 4).contiguous() * 0.9,
                                    torch.randn(24, 4 * d) / (4 * d) ** 0.5,
                                    torch.tensor([0.1, 0.2, 0.3]),
                                    torch.randn(24) * 0.1,
                                    torch.ones(d),
                                    torch.ones(d),
                                ]
                                gamma2 = torch.ones(4 * d) if mid else None
                                expected = reference(
                                    *inputs,
                                    gamma_2=gamma2,
                                    norm_eps=epsilon,
                                    hc_eps=0.0,
                                )
                                actual = cann_ops_transformer.mega_mhc(
                                    *[x.npu() for x in inputs],
                                    gamma_2=gamma2.npu() if mid else None,
                                    norm_eps=epsilon,
                                    hc_eps=0.0,
                                    compute_mode=mode,
                                )
                                torch.npu.synchronize()
                                tolerance = (
                                    0.02
                                    if dtype == torch.bfloat16 or mode == 1
                                    else 0.005
                                )
                                for result, want in zip(actual[:4], expected):
                                    host_result = result.cpu()
                                    self.assertTrue(
                                        torch.isfinite(host_result).all().item()
                                    )
                                    torch.testing.assert_close(
                                        host_result,
                                        want,
                                        atol=tolerance,
                                        rtol=tolerance,
                                    )
                                print("EDGE_PASS", flush=True)


if __name__ == "__main__":
    unittest.main()
