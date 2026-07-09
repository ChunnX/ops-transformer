from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_symm_buffer_uses_context_manager_ccl_buffer_size():
    source = (REPO_ROOT / "torch_extension/cann_ops_transformer/ops/mega_moe.py").read_text()

    assert "required_ccl_buffer_size = _get_mega_moe_ccl_buffer_size(" in source
    assert "customCclBufferSize=required_ccl_buffer_size" in source
    assert "self.context = self._ctx_manager.create_context()" in source
    assert "self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size" in source
    assert "customCclBufferSize=self.ccl_buffer_size" not in source
