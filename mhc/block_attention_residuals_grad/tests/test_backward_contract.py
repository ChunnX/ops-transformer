# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证反向接口的参数约束、空 Tensor 行为及正反向绑定。

在 NPU 主机安装本次算子包和 torch_extension，并加载对应环境后运行：
    python test_backward_contract.py -v

Meta 用例只验证接口的元数据检查，不执行 NPU 计算。非法参数也可能被
Python 提前拦截，因此这些测试不能代替 ACLNN、tiling、InferShape 的分层测试。
自动反向与独立反向调用相同的底层算子，其结果一致不代表独立的数学精度验证。

文档对外仅声明支持 valid_block_num=-1、proj_weight=[1,H]；本脚本还保留
实现兼容性探测（valid_block_num=N），不将其视为文档承诺。
"""

import unittest

import torch
import torch_npu

from cann_ops_transformer import (
    block_attention_residuals,
    block_attention_residuals_backward,
)


def inputs(t, n, h, device, dtype=torch.float32):
    """构造七个反向输入；T为token数，N为历史block数，H为隐藏维度。

    返回顺序：partial_block、block_res、proj_weight、norm_weight、
    grad_hidden_states、inv_norm、probs。后两个缓存固定使用FP32。
    此处缓存为合成数据，仅用于接口与边界测试，不是一般输入的前向参考结果。
    """
    return (
        torch.ones((t, h), device=device, dtype=dtype),
        torch.ones((t, n, h), device=device, dtype=dtype),
        torch.ones((1, h), device=device, dtype=dtype),
        torch.ones((h,), device=device, dtype=dtype),
        torch.ones((t, h), device=device, dtype=dtype),
        torch.ones((t, n + 1), device=device, dtype=torch.float32),
        torch.full((t, n + 1), 1 / (n + 1), device=device, dtype=torch.float32),
    )


class BackwardContractTest(unittest.TestCase):
    def test_public_forward_backward_matches_independent_interface(self):
        """验证包顶层正向入口能触发反向，并得到与独立反向接口一致的四个梯度。"""
        # 从用户实际使用的包顶层调用，可发现导出路径绕过自动反向封装的问题。
        # 使用文档声明的 [1,H] 权重形状。
        args = list(inputs(2, 3, 64, "npu")[:4])
        args = [x.detach().requires_grad_(True) for x in args]
        hidden = block_attention_residuals(*args, valid_block_num=3)
        self.assertIsInstance(hidden, torch.Tensor)
        self.assertTrue(hidden.requires_grad)
        grad_hidden = torch.randn_like(hidden)
        hidden.backward(grad_hidden)
        # 重新调用底层正向并保存缓存，为独立反向提供真实 inv_norm/probs。
        from cann_ops_transformer.ops.mhc.block_attention_residuals.block_attention_residuals import (
            block_attention_residuals_op_builder,
        )

        forward_module = block_attention_residuals_op_builder.load()
        _, inv_norm, probs = forward_module.block_attention_residuals(
            *[x.detach() for x in args], 3, 1e-6, True
        )
        expected = block_attention_residuals_backward(
            args[0].detach(),
            args[1].detach(),
            args[2].detach().reshape(1, 64),
            args[3].detach(),
            grad_hidden,
            inv_norm,
            probs,
            valid_block_num=3,
        )
        for x, grad in zip(args, expected):
            self.assertIsNotNone(x.grad)
            torch.testing.assert_close(x.grad, grad.reshape_as(x))

    def test_public_forward_no_grad_returns_tensor(self):
        """无梯度模式仍应返回单个主输出，不能泄露内部的三个输出元组。"""
        with torch.no_grad():
            hidden = block_attention_residuals(
                *inputs(2, 3, 64, "npu")[:4], valid_block_num=3
            )
        self.assertIsInstance(hidden, torch.Tensor)
        self.assertEqual(tuple(hidden.shape), (2, 64))

    def test_default_and_explicit_n(self):
        """探测当前实现兼容性：默认-1与显式N应产生相同梯度。"""
        args = inputs(2, 3, 64, "npu")
        default = block_attention_residuals_backward(*args)
        explicit = block_attention_residuals_backward(*args, valid_block_num=3)
        for actual, expected in zip(default, explicit):
            torch.testing.assert_close(actual, expected)

    def test_reject_invalid_attribute_including_empty_input(self):
        """N=3时拒绝-2、0、1、4；T=0也不能跳过属性校验。"""
        for device in ("meta", "npu"):
            for t in (0, 2):
                for value in (-2, 0, 1, 4):
                    with self.subTest(device=device, t=t, value=value):
                        with self.assertRaisesRegex(
                            RuntimeError, r"valid_block_num|validBlockNum"
                        ):
                            torch.ops.cann_ops_transformer.block_attention_residuals_backward(
                                *inputs(t, 3, 64, device), valid_block_num=value
                            )

    def test_empty_tokens_zero_weight_gradients(self):
        """T=0时输入梯度为空，但两个权重梯度非空，必须写入0。"""
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            args = inputs(0, 3, 64, "npu", dtype)
            outputs = block_attention_residuals_backward(*args)
            self.assertEqual(
                [tuple(x.shape) for x in outputs], [(0, 64), (0, 3, 64), (1, 64), (64,)]
            )
            for output in outputs[2:]:
                torch.testing.assert_close(output, torch.zeros_like(output))

    def test_empty_hidden(self):
        """H=0时四个梯度均为空，覆盖T/N也为0的组合及非法属性。"""
        for device in ("meta", "npu"):
            for t, n in ((2, 3), (0, 3), (2, 0), (0, 0)):
                with self.subTest(device=device, t=t, n=n):
                    outputs = block_attention_residuals_backward(
                        *inputs(t, n, 0, device)
                    )
                    self.assertEqual(
                        [tuple(x.shape) for x in outputs],
                        [(t, 0), (t, n, 0), (1, 0), (0,)],
                    )
                    self.assertTrue(all(x.numel() == 0 for x in outputs))
                    with self.assertRaisesRegex(
                        RuntimeError, r"valid_block_num|validBlockNum"
                    ):
                        block_attention_residuals_backward(
                            *inputs(t, n, 0, device), valid_block_num=n + 1
                        )

    def test_zero_blocks_still_computes_partial_gradient(self):
        """仅N=0时partial仍非空，不能把整个计算当作空任务跳过。"""
        # 只有partial参与融合：输出对partial的导数为1，权重梯度为0。
        # 此处传0是测试实现接受N的兼容行为，文档对外仅声明支持-1。
        args = inputs(2, 0, 64, "npu")
        outputs = block_attention_residuals_backward(*args, valid_block_num=0)
        torch.testing.assert_close(outputs[0], args[4])
        self.assertEqual(outputs[1].numel(), 0)
        for output in outputs[2:]:
            torch.testing.assert_close(output, torch.zeros_like(output))


if __name__ == "__main__":
    unittest.main()
