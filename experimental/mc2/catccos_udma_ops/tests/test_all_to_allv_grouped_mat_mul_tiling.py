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


def test_selected_profile_drives_all_effective_runtime_tiling_fields():
    source = (ASCEND950_ROOT / "all_to_allv_grouped_mat_mul_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert "tiling.commBlockM" in source
    assert "tiling.commTileM / 2" in source
    assert "tiling.commInterval" in source
    assert "tiling.m0 * tiling.k0 * kLoops" in source
    assert "does not consume either legacy host field" in source
