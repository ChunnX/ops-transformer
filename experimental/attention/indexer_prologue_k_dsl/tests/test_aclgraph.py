# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLGraph capture/replay coverage for the installed ds41 operator."""

import pytest
import torch

from golden import indexer_prologue_k_golden, make_inputs

torch_npu = pytest.importorskip("torch_npu", reason="torch_npu not installed")
if not torch.npu.is_available():  # pragma: no cover - host dependent
    pytest.skip("no NPU visible", allow_module_level=True)

try:
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
except Exception as error:  # pragma: no cover - environment dependent
    pytest.skip(f"torchair is unavailable: {error}", allow_module_level=True)


class _IndexerPrologueKGraph(torch.nn.Module):
    def __init__(self, storage_mode: int, combined_block_size: int):
        super().__init__()
        self.storage_mode = storage_mode
        self.combined_block_size = combined_block_size

    def forward(
        self,
        latent,
        wk,
        norm_weight,
        rope_sin,
        rope_cos,
        k_cache,
        k_scale_cache,
        cache_index,
    ):
        from cann_ops_transformer.ops.ds41 import indexer_prologue_k

        return indexer_prologue_k(
            latent,
            wk,
            norm_weight,
            rope_sin,
            rope_cos,
            k_cache,
            k_scale_cache,
            cache_index=cache_index,
            storage_mode=self.storage_mode,
            norm_eps=1e-6,
            combined_block_size=self.combined_block_size,
        )


def _to_npu(inputs):
    scale_cache = inputs["k_scale_cache"]
    return (
        inputs["latent"].npu(),
        torch_npu.npu_format_cast(
            inputs["wk"].npu(), torch_npu.Format.FRACTAL_NZ
        ),
        inputs["norm_weight"].npu(),
        inputs["rope_sin"].npu(),
        inputs["rope_cos"].npu(),
        inputs["k_cache"].npu(),
        None if scale_cache is None else scale_cache.npu(),
        inputs["cache_index"].npu(),
    )


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "storage_mode,group,with_scale",
    [(0, -1, True), (0, -1, False), (1, 4, False), (1, 8, False)],
)
def test_indexer_prologue_k_aclgraph_capture_and_replay(
    storage_mode, group, with_scale
):
    warmup_inputs = make_inputs(
        72,
        512,
        128,
        64,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=700 + storage_mode * 10 + max(group, 0),
    )
    warmup_args = _to_npu(warmup_inputs)
    eager_model = _IndexerPrologueKGraph(storage_mode, group).npu()
    eager_model(*warmup_args)
    torch.npu.synchronize()

    inputs = make_inputs(
        72,
        512,
        128,
        64,
        storage_mode=storage_mode,
        combined_block_size=group,
        with_scale_cache=with_scale,
        seed=800 + storage_mode * 10 + max(group, 0),
    )
    expected_cache, expected_scale = indexer_prologue_k_golden(
        inputs,
        storage_mode=storage_mode,
        norm_eps=1e-6,
        combined_block_size=group,
    )
    graph_args = _to_npu(inputs)
    for tensor in graph_args:
        if tensor is not None:
            torch._dynamo.mark_static(tensor)

    config = CompilerConfig()
    config.mode = "npugraph_ex"
    # FRACTAL_NZ is private storage metadata. The default input clone uses
    # empty_like and degrades it to ND, which the DSL provider must reject.
    config.debug.aclgraph.clone_input = False
    backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    compiled_model = torch.compile(
        _IndexerPrologueKGraph(storage_mode, group).npu(),
        fullgraph=True,
        backend=backend,
        dynamic=False,
    )

    result = compiled_model(*graph_args)
    torch.npu.synchronize()
    assert result is graph_args[5]
    assert torch.equal(result.cpu(), expected_cache)
    if graph_args[6] is not None:
        assert torch.equal(graph_args[6].cpu(), expected_scale)

    # Restore the output buffers. The second call must replay the captured
    # kernels and mutate them back to the golden values.
    graph_args[5].copy_(inputs["k_cache"].npu())
    if graph_args[6] is not None:
        graph_args[6].copy_(inputs["k_scale_cache"].npu())
    torch.npu.synchronize()
    replay_result = compiled_model(*graph_args)
    torch.npu.synchronize()
    assert replay_result is graph_args[5]
    assert torch.equal(replay_result.cpu(), expected_cache)
    if graph_args[6] is not None:
        assert torch.equal(graph_args[6].cpu(), expected_scale)
