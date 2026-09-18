import contextlib
import io

import pytest
import torch
import torch_npu

from cann_ops_transformer.ops import compressor
from compressor_golden import check_result, cpu_compressor


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("cache_mode", [1, 2])
@pytest.mark.parametrize("has_ape", [False, True])
def test_split_k_after_prefill(dtype, coff, cache_mode, has_ape):
    torch_npu.npu.set_device(0)
    torch.manual_seed(7317)
    width = coff * 128
    wkv = torch.randn(width, 1024).to(dtype) * 0.1
    wgate = torch.randn(width, 1024).to(dtype) * 0.1
    ape = torch.randn(4, width) * 0.1 if has_ape else None
    device_wkv, device_wgate = wkv.npu(), wgate.npu()
    device_ape = None if ape is None else ape.npu()
    prefill_state = torch.randn(5, 1024, 2 * width, device="npu") * 0.1
    prefill_x = (torch.randn(1003, 1024, dtype=dtype) * 0.1).npu()
    compressor(
        prefill_x, device_wkv, device_wgate, prefill_state,
        ape=device_ape,
        state_block_table=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device="npu"),
        cu_seqlens=torch.tensor([0, 1003], dtype=torch.int32, device="npu"),
        start_pos=torch.tensor([129], dtype=torch.int32, device="npu"),
        cmp_ratio=4, coff=coff, cache_mode=1,
    )
    torch.npu.synchronize()
    blocks = 3 if cache_mode == 1 else 1
    table = torch.tensor([[1, 2]] if cache_mode == 1 else [0], dtype=torch.int32)
    reference = torch.randn(blocks, 128, 2 * width) * 0.1
    backing = torch.full((blocks, 129, 2 * width), 17.0, device="npu")
    state = backing[:, :128]
    state.copy_(reference)
    device_table = table.npu()
    for start in range(125, 130):
        x = torch.randn(1, 1, 1024).to(dtype) * 0.1
        positions = torch.tensor([start], dtype=torch.int32)
        kv = reference[:, :, :width].contiguous()
        score = reference[:, :, width:].contiguous()
        update_kv = torch.zeros_like(kv, dtype=torch.bool)
        update_score = torch.zeros_like(score, dtype=torch.bool)
        with contextlib.redirect_stdout(io.StringIO()):
            expected, valid, *_ = cpu_compressor(
                x, wkv, wgate, kv, score, update_kv, update_score, ape,
                block_table=table, start_pos=positions,
                cmp_ratio=4, coff=coff, cache_mode=cache_mode,
            )
        before = state.cpu().clone()
        actual = compressor(
            x.npu(), device_wkv, device_wgate, state,
            ape=device_ape, state_block_table=device_table,
            start_pos=positions.npu(), cmp_ratio=4, coff=coff, cache_mode=cache_mode,
        ).cpu()
        if expected[valid].numel():
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                _, verdict = check_result(
                    expected[valid].float(), actual[valid].float(), str(dtype).split(".")[-1]
                )
            assert verdict == "Pass", capture.getvalue()
        reference = torch.cat((kv, score), dim=-1)
        actual_state = state.cpu()
        torch.testing.assert_close(actual_state, reference, rtol=1e-4, atol=1e-5)
        updated = torch.cat((update_kv, update_score), dim=-1)
        assert torch.equal(actual_state[~updated], before[~updated])
        assert torch.all(backing[:, 128:].cpu() == 17)
