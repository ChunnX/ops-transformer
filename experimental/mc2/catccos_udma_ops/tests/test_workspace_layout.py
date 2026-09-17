from pathlib import Path


ROOT = Path(__file__).parents[1]
KERNEL_INCLUDE = (
    ROOT
    / "csrc"
    / "catccos_overlay"
    / "include"
    / "catccos"
    / "dgemm"
    / "kernel"
)


def _read(name: str) -> str:
    return (KERNEL_INCLUDE / name).read_text(encoding="utf-8")


def test_workspace_has_disjoint_data_and_per_operator_control_planes():
    source = _read("udma_workspace_layout.hpp")

    assert "kDataBytes = 640UL * 1024 * 1024" in source
    assert "kControlBase = 700UL * 1024 * 1024" in source
    assert "kControlBytes = 4096" in source
    assert "kTotalBytes = kControlBase + kControlBytes" in source
    assert "kGmmA2avControlOffset = kControlBase" in source
    assert "kGmmA2avControlBytes = 2048" in source
    assert "kA2avGmmControlOffset =" in source
    assert "kGmmA2avControlOffset + kGmmA2avControlBytes" in source
    assert "kA2avGmmControlBytes = 2048" in source
    assert "kControlAlignment = 64" in source
    assert "kDataBytes <= UdmaWorkspaceLayout::kControlBase" in source


def test_alltoallv_grouped_matmul_uses_only_its_control_slice():
    source = _read("ascend950_alltoallv_grouped_matmul.hpp")

    assert "WorkspaceLayout::kA2avGmmSyncMmadOffset" in source
    assert "syncCommFinish" not in source
    assert "WorkspaceLayout::kA2avGmmReadyOffset" in source
    assert "WorkspaceLayout::kGmmA2avSyncMmadOffset" not in source
    assert "WorkspaceLayout::kGmmA2avSyncCommOffset" not in source
    assert "symmetricOffset += IPC_BUFF_MAX_SIZE" not in source


def test_grouped_matmul_alltoallv_uses_only_its_control_slice():
    source = _read("grouped_matmul_alltoallv_tla.hpp")

    assert "WorkspaceLayout::kGmmA2avSyncMmadOffset" in source
    assert "WorkspaceLayout::kGmmA2avSyncCommOffset" in source
    assert "WorkspaceLayout::kA2avGmmSyncMmadOffset" not in source
    assert "WorkspaceLayout::kA2avGmmSyncCommOffset" not in source
    assert "symmetricOffset += SYMMETRIC_DATA_BYTES" not in source


def test_both_kernels_keep_payload_at_the_workspace_base():
    a2av_gmm = _read("ascend950_alltoallv_grouped_matmul.hpp")
    gmm_a2av = _read("grouped_matmul_alltoallv_tla.hpp")

    assert "auto gmSymmetric = args.ptrSymmetric;" in a2av_gmm
    assert "auto gmSymmetric = args.ptrSymmetric;" in gmm_a2av
    assert "SYMMETRIC_DATA_BYTES <= WorkspaceLayout::kDataBytes" in gmm_a2av
    assert "IPC_BUFF_MAX_SIZE <= WorkspaceLayout::kDataBytes" in a2av_gmm
