from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_custom_device_buffer_is_zero_initialized_after_malloc():
    source = (REPO_ROOT / "torch_extension/cann_ops_transformer/ops/csrc/comm_context.cpp").read_text()

    malloc_pos = source.index("aclrtMalloc(customDeviceBuffer_")
    memset_pos = source.index("aclrtMemset(*customDeviceBuffer_", malloc_pos)
    reg_pos = source.index("HcclCommMemRegFunc", malloc_pos)

    assert malloc_pos < memset_pos < reg_pos
    assert "aclrtMemset(*customDeviceBuffer_, bufferSizeBytes, 0, bufferSizeBytes)" in source
