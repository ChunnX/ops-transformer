# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Installed binary precision tests, including the metadata producer."""
import importlib
import json
import os

import pytest
import torch
import torch_npu  # noqa: F401


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    from cannbotdsl.core.compiler import passes
    from cannbotdsl.core.toolchain.ascendc import AscendCCompiler

    package = importlib.import_module(os.environ.get("KDA_PACKAGE", "cann_ops_transformer_custom"))
    metadata = importlib.import_module(package.__name__ + ".ops.attention.flash_kda_metadata.op_kernel.flash_kda_metadata")
    report = tmp_path_factory.mktemp("native-report") / "report.json"

    def forbid_compile(*args, **kwargs):
        raise AssertionError("An installed operator tried to compile at runtime")

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CANNBOTDSL_NATIVE_BINARY_MODE", "require")
        patch.setenv("CANNBOTDSL_CACHE_MEM_DISABLE", "1")
        patch.setenv("CANNBOTDSL_CACHE_DISK_DISABLE", "1")
        patch.setenv("CANNBOTDSL_NATIVE_BINARY_REPORT", str(report))
        patch.setattr(passes, "_run_cannir_passes", forbid_compile)
        patch.setattr(AscendCCompiler, "compile_ascendc_to_so", forbid_compile)
        patch.setattr(metadata, "compile_aicpu_kernel", forbid_compile)
        torch.npu.set_device(int(os.environ.get("KDA_NPU_DEVICE", "0")))
        yield package, metadata
        counts = json.loads(report.read_text())["counts"]
        assert counts.get("installed_hit", 0) > 0
        assert counts.get("backend_pipeline", 0) == counts.get("backend_compile", 0) == 0
        assert "/_native/" in metadata._AICPU_COMPILED.so_path
        print("Native runtime events:", counts)


def _reference(q, k, v, g, beta, state, a_log, bias, scale, lower_bound):
    """Independent FP32 token recurrence, state layout [Nv, Dv, Dk]."""
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    repeats = v.shape[1] // k.shape[1]
    q, k = (x.repeat_interleave(repeats, dim=1) for x in (q, k))
    decay = (lower_bound * torch.sigmoid(a_log.exp()[None, :, None] * (g + bias))).exp()
    beta = beta.sigmoid()
    state = state.clone()
    outputs = []
    for token in range(q.shape[0]):
        state = state * decay[token].unsqueeze(-2)
        memory = (state * k[token].unsqueeze(-2)).sum(-1)
        delta = (v[token] - memory) * beta[token].unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k[token].unsqueeze(-2)
        outputs.append((state * (q[token] * scale).unsqueeze(-2)).sum(-1))
    return torch.stack(outputs), state


@pytest.mark.parametrize("layout,lengths,nq,nv", [
    ("BNSD", (64,), 3, 3),
    ("BSND", (128,), 2, 4),
    ("BSND", (128, 128), 2, 4),
    ("BNSD", (65, 65), 2, 6),
    ("TND", (1, 63, 65, 129), 2, 4),
    ("TND", (257, 64, 7), 2, 6),
    ("BSND", (2049,), 8, 32),
])
def test_flash_kda(native, layout, lengths, nq, nv):
    package, metadata_impl = native
    generator = torch.Generator().manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    total, batch, dim = sum(lengths), len(lengths), 128
    q, k = randn(total, nq, dim), randn(total, nq, dim)
    v, g, beta = randn(total, nv, dim), randn(total, nv, dim), randn(total, nv)
    state = torch.rand(batch, nv, dim, dim, generator=generator) * 0.1
    a_log = torch.linspace(-1.0, 0.5, nv)
    bias = torch.randn(nv, dim, generator=generator)
    scale, lower_bound = dim ** -0.5, -5.0
    expected_outputs, expected_states = [], []
    offset = 0
    for index, length in enumerate(lengths):
        sl = slice(offset, offset + length)
        output, final = _reference(q[sl], k[sl], v[sl], g[sl], beta[sl], state[index],
                                   a_log, bias, scale, lower_bound)
        expected_outputs.append(output)
        expected_states.append(final)
        offset += length

    def to_layout(tensor):
        if layout != "TND":
            tensor = tensor.reshape(batch, lengths[0], *tensor.shape[1:])
            if layout == "BNSD":
                tensor = tensor.transpose(1, 2)
        return tensor.contiguous().npu()

    q, k, v, g, beta = map(to_layout, (q, k, v, g, beta))
    initial = state.npu()
    cu = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32).npu() if layout == "TND" else None
    metadata = package.flash_kda_metadata(q, v, initial, layout, cu)
    out, final = package.flash_kda(q, k, v, g, beta, scale, initial, a_log.npu(), bias.npu(),
                                 lower_bound, layout, metadata)
    torch.npu.synchronize()
    decoded = metadata_impl.decode_metadata(metadata.cpu().tolist())
    assert decoded["status"] == metadata_impl.STATUS_OK
    assert decoded["batch"] == batch
    if nv * sum((length + 63) // 64 for length in lengths) > metadata_impl.WORKSPACE_SLOTS:
        assert decoded["stage12_round_num"] > 1
    torch.testing.assert_close(initial.cpu(), state, rtol=0, atol=0)
    out = out.cpu().float()
    if layout == "BNSD":
        out = out.transpose(1, 2)
    out = out.reshape(total, nv, dim)
    for label, actual, expected in (
        ("output", out, torch.cat(expected_outputs)),
        ("state", final.cpu(), torch.stack(expected_states)),
    ):
        assert torch.isfinite(actual).all()
        error = ((actual - expected).abs() / (1 + expected.abs())).flatten()
        p999 = torch.quantile(error, 0.999).item()
        print(layout, lengths, label, "p99.9:", p999, "max:", error.max().item())
        # Keep the existing FlashKDA test's 99.9th percentile / 5e-3 criterion.
        assert p999 <= 5e-3


def test_flash_kda_meta(native):
    package, metadata_impl = native
    q = torch.empty((2, 4, 65, 128), dtype=torch.bfloat16, device="meta")
    v = torch.empty((2, 8, 65, 128), dtype=torch.bfloat16, device="meta")
    state = torch.empty((2, 8, 128, 128), device="meta")
    metadata = package.flash_kda_metadata(q, v, state, "BNSD")
    assert metadata.shape == (metadata_impl.metadata_capacity_upper_bound(2, 65, is_packed=False),)
    out, final = package.flash_kda(q, q, v, v, v[..., 0], 128 ** -0.5, state,
                                 torch.empty(8, device="meta"), torch.empty(8, 128, device="meta"),
                                 -5.0, "BNSD", metadata)
    assert out.shape == v.shape and final.shape == state.shape
