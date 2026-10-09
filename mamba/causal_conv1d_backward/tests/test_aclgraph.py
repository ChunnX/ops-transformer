# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.

"""ACLGraph capture/replay verification for causal_conv1d_backward.

两层验证（与 engram_gate_dsl/tests/test_aclgraph.py 同构）：
1. torch.npu.NPUGraph（torch_npu 原生 aclgraph 捕获）：eager 路径的 aclnn kernel
   可被图捕获，重放时对新输入数据结果正确。
2. torchair npugraph_ex（依赖 torchair，缺失时跳过）：经 torch_extension 自定义
   算子边界（``torch.ops.cann_ops_transformer.causal_conv1d_backward``，converter
   由 ``graph_convert_causal_conv1d_backward.py`` 注册）被 dynamo 全图捕获并重放，
   GE 侧以 REG_OP(CausalConv1dBackward) 执行。

golden 为标准 06 §七 的 qsl 版参考实现（tests/golden.py）。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden  # noqa: E402

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")
if not torch.npu.is_available():  # pragma: no cover - host dependent
    pytest.skip("no NPU visible", allow_module_level=True)

from cann_ops_transformer.ops.mamba import causal_conv1d_backward  # noqa: E402


def _make_inputs(b, l, c, width, num_docs, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(b, l, c, generator=generator) * 0.5).to(torch.bfloat16)
    weight = (torch.randn(c, 1, width, generator=generator) * 0.1).to(torch.bfloat16)
    grad_y = (torch.randn(b, l, c, generator=generator) * 0.1).to(torch.bfloat16)
    positions = golden._positions_per_row_docs(b, l, num_docs)
    qsl = golden._query_start_loc(positions)
    return x, weight, grad_y, qsl


def _golden(grad_y, x, weight, qsl):
    # 返回 (grad_x [B,L,C], grad_weight [W,C]) —— grad_weight 与调用方传入的
    # weight 视图（[C,1,W] 的 transpose(0,1)）同形，由适配层负责转回 [C,1,W]。
    tokens = x.shape[0] * x.shape[1]
    c = x.shape[-1]
    gx, gw = golden.causal_conv1d_backward(
        grad_y.cpu().reshape(tokens, c),
        x.cpu().reshape(tokens, c),
        weight.cpu()[:, 0, :].transpose(0, 1),
        qsl.cpu(),
    )
    return gx.reshape(x.shape), gw


# 模型形态用例（KDA 短卷积训练缩小版）+ 变长/多文档边界
@pytest.mark.npu
@pytest.mark.parametrize(
    "b,l,c,width,num_docs",
    [
        pytest.param(2, 64, 1024, 4, 2, id="model_shape_packed"),
        pytest.param(1, 32, 256, 4, 4, id="multi_docs"),
        pytest.param(1, 8, 256, 2, 1, id="width2_single_doc"),
    ],
)
def test_causal_conv1d_backward_aclgraph_capture_and_replay(b, l, c, width, num_docs):
    torch.npu.set_device(0)
    static_grad_y, static_weight, static_x, static_qsl = (
        t.npu() for t in _make_inputs(b, l, c, width, num_docs, 20260926)
    )

    def _op(gy, w_c1w, x, qsl):
        # 框架参数 [C,1,W] → 算子 [W,C]（非连续转置视图直传，标准 §九 接入写法）
        return causal_conv1d_backward(gy, x, w_c1w[:, 0, :].transpose(0, 1), qsl)

    warmup_stream = torch.npu.Stream()
    warmup_stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(warmup_stream):
        for _ in range(3):
            _op(static_grad_y, static_weight, static_x, static_qsl)
    torch.npu.current_stream().wait_stream(warmup_stream)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        static_gx, static_gw = _op(static_grad_y, static_weight, static_x, static_qsl)
    torch.npu.synchronize()

    for replay_idx in range(3):
        grad_y, weight, x, qsl = _make_inputs(b, l, c, width, num_docs, 20260927 + replay_idx)
        static_grad_y.copy_(grad_y.npu())
        static_weight.copy_(weight.npu())
        static_x.copy_(x.npu())
        static_qsl.copy_(qsl.npu())
        torch.npu.synchronize()

        graph.replay()
        torch.npu.synchronize()

        gx_expected, gw_expected = _golden(grad_y, x, weight, qsl)
        assert torch.allclose(
            static_gx.cpu().float(), gx_expected.float(), atol=2e-2, rtol=2e-2
        ), (
            f"replay {replay_idx}: grad_x max error "
            f"{(static_gx.cpu().float() - gx_expected.float()).abs().max().item()}"
        )
        assert torch.allclose(
            static_gw.cpu().float(), gw_expected.float(), atol=2e-2, rtol=2e-2
        ), (
            f"replay {replay_idx}: grad_weight max error "
            f"{(static_gw.cpu().float() - gw_expected.float()).abs().max().item()}"
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "b,l,c,width,num_docs",
    [
        pytest.param(2, 64, 1024, 4, 2, id="model_shape_packed"),
        pytest.param(1, 32, 256, 4, 4, id="multi_docs"),
    ],
)
def test_causal_conv1d_backward_torchair_npugraph_ex(b, l, c, width, num_docs):
    torchair = pytest.importorskip("torchair", reason="torchair not installed")
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_device(0)

    class _Model(torch.nn.Module):
        def forward(self, grad_y, weight_c1w, x, qsl):
            # 框架参数 [C,1,W] → 算子 [W,C]（非连续转置视图直传，标准 §九 接入写法）
            return causal_conv1d_backward(
                grad_y, x, weight_c1w[:, 0, :].transpose(0, 1), qsl)

    static_grad_y, static_weight, static_x, static_qsl = (
        t.npu() for t in _make_inputs(b, l, c, width, num_docs, 20260928)
    )
    args = (static_grad_y, static_weight, static_x, static_qsl)

    eager_model = _Model().npu()
    eager_model(*args)
    torch.npu.synchronize()

    for tensor in args:
        torch._dynamo.mark_static(tensor)

    config = CompilerConfig()
    config.mode = "npugraph_ex"
    backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    compiled_model = torch.compile(
        eager_model, fullgraph=True, backend=backend, dynamic=False
    )

    for replay_idx in range(2):
        grad_y, weight, x, qsl = _make_inputs(b, l, c, width, num_docs, 20260929 + replay_idx)
        static_grad_y.copy_(grad_y.npu())
        static_weight.copy_(weight.npu())
        static_x.copy_(x.npu())
        static_qsl.copy_(qsl.npu())
        torch.npu.synchronize()

        gx_result, gw_result = compiled_model(*args)
        torch.npu.synchronize()

        gx_expected, gw_expected = _golden(grad_y, x, weight, qsl)
        assert torch.allclose(
            gx_result.cpu().float(), gx_expected.float(), atol=2e-2, rtol=2e-2
        ), (
            f"replay {replay_idx}: grad_x max error "
            f"{(gx_result.cpu().float() - gx_expected.float()).abs().max().item()}"
        )
        assert torch.allclose(
            gw_result.cpu().float(), gw_expected.float(), atol=2e-2, rtol=2e-2
        ), (
            f"replay {replay_idx}: grad_weight max error "
            f"{(gw_result.cpu().float() - gw_expected.float()).abs().max().item()}"
        )
