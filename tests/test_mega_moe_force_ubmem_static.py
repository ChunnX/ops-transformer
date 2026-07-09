from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_symm_buffer_forces_ubmem_context_and_mte_topology():
    source = (REPO_ROOT / "torch_extension/cann_ops_transformer/ops/mega_moe.py").read_text()

    assert 'commAlg="ub-mem"' in source
    assert "self.topo_type = 0" in source
    assert 'commAlg="urma"' not in source
    assert "self.topo_type = self._ctx_manager.topo_type" not in source


def test_arch35_tiling_key_forces_mte_kernel():
    source = (
        REPO_ROOT / "mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp"
    ).read_text()

    assert "int64_t topoType = TILINGKEY_TPL_MTE;" in source
    assert "int64_t topoType = *attrs->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));" not in source
