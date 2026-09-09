# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
ASCEND950_ROOT = (
    PROJECT_ROOT / "csrc" / "all_to_allv_grouped_mat_mul_udma" / "ascend950"
)


def test_business_shapes_select_tuned_tiling_profiles():
    source = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert (
        "KernelVariant::kM0_128, 128, 256, 256, 12, 128, 128, 4, 4"
        in source
    )
    assert (
        "KernelVariant::kM0_256, 256, 128, 256, 6, 128, 128, 4, 4"
        in source
    )
    assert "worldSize == 8 && expertNum == 256" in source
    assert (
        "isEightRank256ExpertCase && m == 131072 && k == 2048 && n == 1024"
        in source
    )
    assert (
        "isEightRank256ExpertCase && m == 131072 && k == 2048 && n == 512"
        in source
    )
    assert "static_cast<uint32_t>(worldSize)" in source
    assert "static_cast<uint32_t>(expertNum)" in source
    assert "return kDefaultTiling" in source


def test_tiling_selection_is_rank_invariant_for_imbalanced_routes():
    source = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    select_tiling = source[
        source.index("const TilingProfile &SelectTiling("):
        source.index("uint32_t RoundUp")
    ]
    assert "receivedRows" not in select_tiling
    assert "outputRows" not in select_tiling
    assert "rank-invariant" in select_tiling
    assert "SelectTiling(\n        m, k, n," in source


def test_export_business_shapes_use_separate_tuned_profiles():
    source = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert "kM131072K2048N1024NonExportTiling" in source
    assert "kM131072K2048N1024ExportTiling" in source
    assert (
        "KernelVariant::kM0_256N0_256, 256, 256, 256, 5, 32, 32, 1, 16"
        in source
    )
    assert "1280, 3, 16, 1, 24, 8" in source
    assert "kM131072K2048N512NonExportTiling" in source
    assert "kM131072K2048N512ExportTiling" in source
    assert (
        "KernelVariant::kM0_256, 256, 128, 256, 5, 32, 64, 1, 16"
        in source
    )
    assert "1280, 2, 8, 1, 0, 8" in source
    assert "execution mode is not part of the migrated kernel ABI" in source
    assert "bool outputAllToAllV" in source
    assert "? kM131072K2048N1024ExportTiling" in source
    assert ": kM131072K2048N1024NonExportTiling" in source
    assert "? kM131072K2048N512ExportTiling" in source
    assert ": kM131072K2048N512NonExportTiling" in source
    assert "allToAllVOutputOptional.has_value());" in source


def test_m0_256_profile_has_a_dedicated_kernel_instance():
    device = (
        ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma_device.h"
    ).read_text(encoding="utf-8")
    host = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert "using AllToAllVGroupedMatMulUdmaM0_256Config" in device
    assert "256, 128, 256>;" in device
    assert "TypeTag<AllToAllVGroupedMatMulUdmaM0_256Config>" in host
    assert "using AllToAllVGroupedMatMulUdmaM0_256ExportConfig" in device
    assert "TypeTag<AllToAllVGroupedMatMulUdmaM0_256ExportConfig>" in host
    assert "Bf16AllToAllVGroupedMatMulUdmaConfig<256, 256, 256" in host
    assert "true, 3, 2" in host


def test_selected_profile_drives_all_effective_runtime_tiling_fields():
    source = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert "tiling.commBlockM" in source
    assert "tiling.localCopyTileRows" in source
    assert "tiling.commInterval" in source
    assert "tiling.stageRows" in source
    assert "tiling.workspaceStages" in source
    assert "tiling.mSplit" in source
    assert "tiling.exportCoreCount" in source
    assert "tiling.selfCopyCores" in source
    assert "requiredWorkspaceBytes <= kSymmetricDataBytes" in source
    assert "does not consume either legacy host field" in source


def test_delivery_commit_compute_scheduler_and_physical_tiling_are_wired():
    host = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    device = (
        ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma_device.h"
    ).read_text(encoding="utf-8")
    scheduler = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "block"
        / "block_scheduler_alltoallv_allgather.hpp"
    ).read_text(encoding="utf-8")
    assert "uint32_t WorkspaceStages = 2, uint32_t MSplit = 1" in device
    assert "BlockMmadSchedulerAllToAllVAllGather<MoeConstraints, MSplit>" in device
    assert "scheduler.UpdateMmadContext(localExpertIdx);" in (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "ascend950_alltoallv_grouped_matmul.hpp"
    ).read_text(encoding="utf-8")
    assert "MatrixSwizzle<M_SPLIT, 0>::GetCoord" in scheduler
    assert "tiling.stageRows" in host
    assert "tiling.workspaceStages" in host
    assert "launchByMSplit" in host
    assert "true, 3, 4" in host
