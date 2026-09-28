#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NPU regression coverage for FIA executor caching and unchanged fallback paths."""
from concurrent.futures import ThreadPoolExecutor
import math
import pytest
torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope='module', autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip('Ascend NPU required')
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    yield
    torch.npu.synchronize()


def device_view(cpu, geometry):
    shape = tuple(cpu.shape)
    guards = []
    if geometry == 'dim0':
        backing = torch.full((shape[0], shape[1] + 1, *shape[2:]), 73., dtype=cpu.dtype, device='npu')
        view = backing[:, :shape[1]]
        guards.append((backing[:, shape[1]:], 73.))
    elif geometry == 'inner':
        backing = torch.full((*shape[:-1], shape[-1] + 3), 73., dtype=cpu.dtype, device='npu')
        view = backing[..., :shape[-1]]
        guards.append((backing[..., shape[-1]:], 73.))
    elif geometry == 'offset':
        backing = torch.full((cpu.numel() + 17,), 73., dtype=cpu.dtype, device='npu')
        view = backing[17:].reshape(shape)
        guards.append((backing[:17], 73.))
    else:
        return cpu.npu(), guards
    view.copy_(cpu)
    return view, guards


def make_case(seed, dtype=torch.bfloat16, geometry='contiguous', lse=False, dim=128, alias=False):
    generator = torch.Generator().manual_seed(seed)
    batch, heads, blocks, block_size = 2, 4, 4, 128
    q_cpu = torch.randn(batch, heads, dim, dtype=dtype, generator=generator)
    k_cpu = torch.randn(batch * blocks, block_size, dim, dtype=dtype, generator=generator)
    v_cpu = k_cpu if alias else torch.randn(k_cpu.shape, dtype=dtype, generator=generator)
    table_cpu = torch.randperm(batch * blocks, generator=generator).reshape(batch, blocks).int()
    q, q_guards = device_view(q_cpu, 'offset' if geometry == 'offset' else 'contiguous')
    k, k_guards = device_view(k_cpu, geometry)
    v, v_guards = (k, []) if alias else device_view(v_cpu, geometry)
    table = table_cpu.npu()
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()
    params = dict(query=q, key=k, value=v, block_table=table, block_size=block_size,
                  input_layout='TND', actual_seq_lengths=[1, 2], actual_seq_lengths_kv=[512, 511],
                  num_heads=heads, num_key_value_heads=1, scale=1 / math.sqrt(dim),
                  atten_mask=mask, sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647,
                  softmax_lse_flag=lse)
    guards = q_guards + k_guards + v_guards
    metadata = [(t, t.shape, t.stride(), t.storage_offset()) for t in (q, k, v)]
    return params, (q_cpu, k_cpu, v_cpu, table_cpu), guards, metadata


def reference(cpu, lengths):
    q, k, v, table = cpu
    outputs, lses = [], []
    for b, length in enumerate(lengths):
        key = k[table[b].long()].flatten(0, 1)[:length].float()
        value = v[table[b].long()].flatten(0, 1)[:length].float()
        scores = q[b].float() @ key.T / math.sqrt(q.shape[-1])
        outputs.append(scores.softmax(-1) @ value)
        lses.append(scores.logsumexp(-1))
    return torch.stack(outputs), torch.stack(lses)


def check_result(output, cpu, lengths, guards, metadata, lse=False):
    expected, expected_lse = reference(cpu, lengths)
    torch.testing.assert_close(output[0].cpu().float(), expected, atol=0.003, rtol=0.025)
    if lse:
        torch.testing.assert_close(output[1].cpu().float().reshape_as(expected_lse), expected_lse,
                                   atol=0.003, rtol=0.003)
    for guard, poison in guards:
        assert torch.all(guard.cpu() == poison), 'operator modified padding outside a view'
    for tensor, shape, stride, offset in metadata:
        assert (tensor.shape, tensor.stride(), tensor.storage_offset()) == (shape, stride, offset)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('geometry', ['contiguous', 'dim0', 'inner', 'offset'])
def test_dynamic_lengths_and_new_addresses(dtype, geometry):
    # Revisit equal lengths with new Q/K/V and block-table addresses/content.
    for index, lengths in enumerate(([127, 193], [400, 511], [127, 193], [1, 257])):
        params, cpu, guards, metadata = make_case(101 + index, dtype, geometry)
        output = torch_npu.npu_fused_infer_attention_score(**{**params, 'actual_seq_lengths_kv': lengths})
        check_result(output, cpu, lengths, guards, metadata)


@pytest.mark.parametrize('variant', ['lse', 'd64', 'kv_alias'])
def test_unsupported_cache_domains_keep_original_behavior(variant):
    for index in range(2):
        params, cpu, guards, metadata = make_case(401 + index, lse=variant == 'lse',
                                                dim=64 if variant == 'd64' else 128,
                                                alias=variant == 'kv_alias')
        output = torch_npu.npu_fused_infer_attention_score(**params)
        check_result(output, cpu, params['actual_seq_lengths_kv'], guards, metadata, variant == 'lse')


def test_graph_update_mixed_geometry_and_every_layer():
    cases = [make_case(501 + i, geometry=g) for i, g in enumerate(
        ['contiguous', 'dim0', 'offset', 'contiguous', 'inner'])]
    outputs, workspaces, handles, events = [], [], [], []
    capture_stream = torch.npu.Stream()
    update_stream = torch.npu.Stream()
    for params, *_ in cases:
        # The existing max-workspace helper rejects noncontiguous caches before
        # the handwritten normalization path. Query it on dense equivalents and
        # reserve bounded extra space for fallback copies; the operator inputs
        # below remain noncontiguous and are checked against the CPU reference.
        dense_params = {**params, **{name: params[name].contiguous().clone() for name in ('query', 'key', 'value')}}
        base = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**dense_params)
        extra_bytes = 4 * sum(params[name].numel() * params[name].element_size() for name in ('query', 'key', 'value')) + 4096
        workspaces.append(torch.empty(base.numel() + math.ceil(extra_bytes / base.element_size()),
                                      dtype=base.dtype, device=base.device))
        outputs.append([torch.empty_like(params['query']), torch.empty(1, dtype=params['query'].dtype, device='npu')])
    def execute(i, lengths):
        return torch_npu.npu_fused_infer_attention_score.out(
            **{**cases[i][0], 'actual_seq_lengths_kv': lengths}, workspace=workspaces[i], out=outputs[i])
    for i in range(len(cases)):
        execute(i, [512, 511])
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=capture_stream):
        for i in range(len(cases)):
            event = torch.npu.ExternalEvent()
            event.wait(capture_stream)
            event.reset(capture_stream)
            torch.npu.graph_task_group_begin(capture_stream)
            execute(i, [512, 511])
            handles.append(torch.npu.graph_task_group_end(capture_stream))
            events.append(event)
    for lengths in ([127, 193], [400, 511], [127, 193], [1, 257]):
        with torch.npu.stream(update_stream):
            for i, handle in enumerate(handles):
                torch.npu.graph_task_update_begin(update_stream, handle)
                execute(i, lengths)
                torch.npu.graph_task_update_end(update_stream)
                events[i].record(update_stream)
        graph.replay()
        torch.npu.synchronize()
        for i, (_, cpu, guards, metadata) in enumerate(cases):
            check_result(outputs[i], cpu, lengths, guards, metadata)


def test_two_threads_and_streams():
    def run(seed):
        torch.npu.set_device(0)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            for index, lengths in enumerate(([127, 193], [400, 511], [127, 193])):
                params, cpu, guards, metadata = make_case(seed + index)
                output = torch_npu.npu_fused_infer_attention_score(**{**params, 'actual_seq_lengths_kv': lengths})
                check_result(output, cpu, lengths, guards, metadata)
        stream.synchronize()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, seed) for seed in (701, 801)]
        for future in futures:
            future.result()


def test_alias_cache_domain_transitions():
    for index, alias in enumerate((False, True, False, True)):
        params, cpu, guards, metadata = make_case(901 + index, alias=alias)
        lengths = [193, 257]
        result = torch_npu.npu_fused_infer_attention_score(**{**params, 'actual_seq_lengths_kv': lengths})
        check_result(result, cpu, lengths, guards, metadata)


def test_nonzero_output_offset():
    params, cpu, guards, metadata = make_case(1001)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**params)
    output_cpu = torch.zeros_like(cpu[0])
    output, output_guards = device_view(output_cpu, 'offset')
    lse = torch.empty(1, dtype=output.dtype, device='npu')
    for lengths in ([193, 257], [400, 511], [193, 257]):
        result = torch_npu.npu_fused_infer_attention_score.out(
            **{**params, 'actual_seq_lengths_kv': lengths}, workspace=workspace, out=[output, lse])
        check_result(result, cpu, lengths, guards + output_guards, metadata)


def test_cache_eviction_with_new_lengths_and_addresses():
    # Run with ACLNN_CACHE_LIMIT=32 to force eviction while checking every call.
    for index in range(40):
        params, cpu, guards, metadata = make_case(1101 + index)
        lengths = [16 + index, 160 + index]
        result = torch_npu.npu_fused_infer_attention_score(**{**params, 'actual_seq_lengths_kv': lengths})
        check_result(result, cpu, lengths, guards, metadata)
