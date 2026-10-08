"""Compare the installed ACLNN DSO with the DSL entry on the same NPU inputs."""
import ctypes as C
import os
from pathlib import Path

import pytest
import torch

from test_aclnn_adapter import load_operator
from test_flash_kda import load_source, require_npu


@pytest.mark.parametrize("layout", ["BNSD", "BSND", "TND"])
def test_builtin_aclnn_matches_dsl(layout):
    library = os.environ.get("FLASH_KDA_ACLNN_LIBRARY")
    opbase = os.environ.get("FLASH_KDA_NNOPBASE_LIBRARY")
    metadata_source = os.environ.get("FLASH_KDA_METADATA_SOURCE")
    if not all((library, opbase, metadata_source)):
        pytest.skip("Set FLASH_KDA_ACLNN_LIBRARY, FLASH_KDA_NNOPBASE_LIBRARY and FLASH_KDA_METADATA_SOURCE")
    require_npu()
    module = load_operator()
    metadata_module = load_source("flash_kda_metadata_aclnn_test", metadata_source)
    qshape = (1, 1, 65, 128) if layout == "BNSD" else (1, 65, 1, 128) if layout == "BSND" else (65, 1, 128)
    q = torch.randn(qshape, dtype=torch.bfloat16).npu()
    k, v, g = [torch.randn_like(q) for _ in range(3)]
    beta = torch.randn(qshape[:-1], dtype=torch.bfloat16).npu()
    initial = torch.zeros((1, 1, 128, 128), device=q.device)
    a_log = torch.zeros(1, device=q.device)
    bias = torch.zeros(1, 128, device=q.device)
    cu = torch.tensor([0, 65], dtype=torch.int32, device=q.device) if layout == "TND" else None
    metadata = metadata_module.flash_kda_metadata(q, v, initial, layout, cu)
    expected, expected_state = module.flash_kda(q, k, v, g, beta, 0.125, initial, a_log, bias, -1.0, layout, metadata)
    out, final = torch.empty_like(v), torch.empty_like(initial)
    base = C.CDLL(str(Path(opbase).resolve()), mode=C.RTLD_GLOBAL)
    api = C.CDLL(str(Path(library).resolve()))
    ptr = C.c_void_p
    i64 = C.POINTER(C.c_int64)
    base.aclCreateTensor.argtypes = [i64, C.c_uint64, C.c_int, i64, C.c_int64, C.c_int, i64, C.c_uint64, ptr]
    base.aclCreateTensor.restype = ptr
    base.aclDestroyTensor.argtypes = [ptr]
    descriptors = []
    def descriptor(tensor):
        shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
        stride = (C.c_int64 * tensor.ndim)(*tensor.stride())
        dtype = {torch.bfloat16: 27, torch.float32: 0, torch.int32: 3}[tensor.dtype]
        result = base.aclCreateTensor(shape, tensor.ndim, dtype, stride, 0, 2,
                                     shape, tensor.ndim, tensor.data_ptr())
        assert result
        descriptors.append(result)
        return result
    workspace_size = C.c_uint64()
    executor = ptr()
    api.aclnnFlashKDAGetWorkspaceSize.argtypes = [ptr] * 5 + [C.c_double] + [ptr] * 3 + [C.c_double, C.c_char_p] + [ptr] * 3 + [C.POINTER(C.c_uint64), C.POINTER(ptr)]
    api.aclnnFlashKDAGetWorkspaceSize.restype = C.c_int
    api.aclnnFlashKDA.argtypes = [ptr, C.c_uint64, ptr, ptr]
    api.aclnnFlashKDA.restype = C.c_int
    try:
        rc = api.aclnnFlashKDAGetWorkspaceSize(
            *[descriptor(t) for t in (q, k, v, g, beta)], 0.125,
            *[descriptor(t) for t in (initial, a_log, bias)], -1.0, layout.encode(),
            *[descriptor(t) for t in (metadata, out, final)], C.byref(workspace_size), C.byref(executor))
        assert rc == 0, f"GetWorkspaceSize status={rc}"
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=q.device)
        rc = api.aclnnFlashKDA(workspace.data_ptr(), workspace_size.value, executor,
                              torch.npu.current_stream().npu_stream)
        assert rc == 0, f"execution status={rc}"
        torch.npu.synchronize()
        torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(final, expected_state, atol=0.0, rtol=0.0)
    finally:
        for tensor in descriptors:
            base.aclDestroyTensor(tensor)
        metadata_module.clear_metadata_caches()
