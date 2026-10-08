"""Check host view mapping and rejection before a Native provider is submitted."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


def load_operator():
    name = "flash_kda_precision_test"
    if name not in sys.modules:
        source = Path(__file__).parents[1] / "op_kernel/flash_kda.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
@pytest.mark.parametrize("length", [64, 65])
def test_layout_and_provider_bindings(monkeypatch, layout, length):
    module = load_operator()
    calls = []
    monkeypatch.setattr(module.cannbotdsl, "get_platform_info", lambda: SimpleNamespace(cube_core_num=24))
    monkeypatch.setattr(module, "_compile_flash_kda", lambda: lambda *args: calls.append(args))
    qshape = (2, 1, length, 128) if layout == "BNSD" else (2, length, 1, 128) if layout == "BSND" else (length, 1, 128)
    vshape = (2, 2, length, 128) if layout == "BNSD" else (2, length, 2, 128) if layout == "BSND" else (length, 2, 128)
    q = torch.empty(qshape, dtype=torch.bfloat16)
    v = torch.empty(vshape, dtype=torch.bfloat16)
    beta = torch.empty(vshape[:-1], dtype=torch.bfloat16)
    state = torch.empty((2, 2, 128, 128), dtype=torch.float32)
    metadata = torch.empty(256, dtype=torch.int32)
    out, final = module.flash_kda_aclnn(q, q, v, v, beta, 0.125, state,
                                       torch.empty(2), torch.empty(2, 128), -1.0, layout, metadata)
    assert out.shape == v.shape and final.shape == state.shape
    args = calls[0]
    assert args[0].data_ptr() == q.data_ptr() and args[1].data_ptr() == v.data_ptr()
    assert args[0].shape == ((1 if layout == "TND" else 2), 1, length, 128)
    assert args[14].shape[2] == length
    assert args[15] is state and args[16] is final and args[18] is metadata
    assert args[-1] == 24 and args[5].shape == (24, 2, 2, 64)
    assert args[17].shape == (16, 16) and args[17].dtype == torch.float16


def test_invalid_layout_rejected_before_launch(monkeypatch):
    module = load_operator()
    monkeypatch.setattr(module, "_compile_flash_kda", lambda: pytest.fail("invalid inputs reached Native"))
    tensor = torch.empty(1)
    with pytest.raises(AssertionError):
        module.flash_kda_aclnn(tensor, tensor, tensor, tensor, tensor, 1.0, tensor,
                              tensor, tensor, -1.0, "invalid", tensor)
