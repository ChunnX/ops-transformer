# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
ASCEND950_ROOT = (
    PROJECT_ROOT / "csrc" / "grouped_mat_mul_all_to_allv_udma" / "ascend950"
)


def test_kernel_keeps_inputs_and_output_in_regular_npu_memory():
    source = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert 'reinterpret_cast<uint8_t *>(x.data_ptr())' in source
    assert 'reinterpret_cast<uint8_t *>(weight.data_ptr())' in source
    assert "at::empty({outputRows64, n64}, x.options())" in source
    assert "runtime.IsSymmetric(output.data_ptr())" not in source
    assert "AllocateSymmetricTensor(x, {m64, n64}" not in source
    assert "aclrtMemcpyAsync" not in source


def test_validation_uses_regular_reusable_output():
    accuracy = (PROJECT_ROOT / "validation" / "accuracy_worker.py").read_text(
        encoding="utf-8"
    )
    performance = (PROJECT_ROOT / "validation" / "performance_worker.py").read_text(
        encoding="utf-8"
    )
    assert "reusable_out = torch.empty((m, n)" in accuracy
    assert "reusable_out = udma.empty_symmetric" not in accuracy
    assert "out = udma.empty_symmetric" not in performance


def test_kernel_configuration_matches_bf16_udma_get_example():
    source = (
        ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma_device.h"
    ).read_text(encoding="utf-8")
    assert "MoeConstraints<1, 8, 32>" in source
    assert "BlockMmadSchedulerGmmAllToAllVTargetOrder" in source
    assert "BlockCommSchedulerGmmAllToAllVTargetOrder" in source
    assert "kWorkspaceStages, kSymmetricDataBytes, true" in source
    assert "CopyDirect::Get" in source
    assert "CopyTransport::Mte" in source
    assert "200UL * 1024 * 1024 * sizeof(uint16_t)" in source


def test_business_profiles_use_compile_time_no_reuse_delivery_kernel():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    device = (
        ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma_device.h"
    ).read_text(encoding="utf-8")
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")

    assert "const bool useNoReuseDelivery" in host
    assert "worldSize == 8 && expertNum == 256 && n64 == 2048" in host
    assert "(k64 == 1024 || k64 == 512)" in host
    assert "useNoReuseKernel" in host
    assert "GroupedMatMulAllToAllVUdmaNoReuseConfig" in host
    assert "kNoReuseM0_128WorkspaceStages = 19" in device
    assert "kNoReuseM0_256WorkspaceStages = 22" in device
    assert "NoReuseWorkspaceStages(tiling.kernelVariant)" in host
    assert (
        "kNoReuseM0_128WorkspaceStages, kNoReuseSymmetricDataBytes, true"
        in device
    )
    assert (
        "kNoReuseM0_256WorkspaceStages, kNoReuseSymmetricDataBytes, true"
        in device
    )
    assert "kNoReuseSymmetricDataBytes" in device
    assert "640UL * 1024 * 1024" in device
    assert "NoReuseStaging = false" in device
    assert "kNoReuseSymmetricDataBytes, true" in device
    assert "static constexpr bool NO_REUSE_STAGING" in kernel
    assert "if constexpr (!NO_REUSE_STAGING)" in kernel


def test_no_reuse_host_and_device_use_the_same_profile_stage_count():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )

    assert "selectedWorkspaceStages == SelectedConfig::kWorkspaceStages" in host


def test_no_reuse_delivery_retires_gets_once_instead_of_waiting_each_round():
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    target_order = kernel[kernel.index("void RunTargetOrderAivWithBlock("):]

    completion_signal = re.search(
        r"syncCommFinish, 1,\s+ACLSHMEM_SIGNAL_ADD, srcRankIdx", target_order
    ).start()
    signal_guard = target_order.rfind(
        "if constexpr (!NO_REUSE_STAGING)", 0, completion_signal
    )
    assert signal_guard >= 0

    completion_wait = target_order.index(
        "syncCommFinish, ACLSHMEM_CMP_EQ, receiveAccum"
    )
    wait_guard = target_order.rfind(
        "if constexpr (!NO_REUSE_STAGING)", 0, completion_wait
    )
    assert wait_guard > completion_signal

    final_retirement_comment = target_order.index(
        "Retire all remote reads once"
    )
    final_retirement_guard = target_order.rfind(
        "if constexpr (NO_REUSE_STAGING)", 0, final_retirement_comment
    )
    final_retirement_barrier = target_order.index(
        "aclshmemx_barrier_all_vec();", final_retirement_comment
    )
    assert final_retirement_guard > completion_wait
    assert final_retirement_barrier > final_retirement_guard


def test_delivery_producer_sync_is_selected_at_build_time():
    root_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    device_cmake = (
        ASCEND950_ROOT / "CMakeLists.txt"
    ).read_text(encoding="utf-8")
    setup_source = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")

    assert "option(CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER" in root_cmake
    assert "CATCCOS_UDMA_GMM_A2AV_PRODUCER_SYNC_FLAG" in root_cmake
    assert "CATCCOS_UDMA_GMM_A2AV_PRODUCER_SYNC_FLAG" in device_cmake
    assert '"CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER", "1"' in setup_source
    assert "static constexpr bool PRODUCER_FULL_BARRIER" in kernel
    assert "if constexpr (NO_REUSE_STAGING && PRODUCER_FULL_BARRIER)" in kernel
    assert "if constexpr (PRODUCER_FULL_BARRIER)" in kernel
    assert "AscendC::SyncAll<true>();" in kernel


def test_kernel_applies_upstream_non_comm_core_deadlock_fix():
    source = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert "if (isCommCore)" in source
    assert "aclshmem_ptr(params.syncMmadFinish, srcRankIdx)" in source
    assert source.index("if (isCommCore)") < source.index(
        "aclshmem_ptr(params.syncMmadFinish, srcRankIdx)"
    )


def test_python_facade_dispatches_both_allocation_modes():
    source = (PROJECT_ROOT / "ops_transformer_catccos_udma" / "ops.py").read_text(
        encoding="utf-8"
    )
    assert "grouped_mat_mul_all_to_allv_udma(" in source
    assert "grouped_mat_mul_all_to_allv_udma_out(" in source


def test_unbalanced_route_uses_distinct_input_and_output_row_counts():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert "const int64_t inputRows64 = x.size(0);" in host
    assert "outputRows64 = localTokensPerExpert.sum().item<int64_t>();" in host
    assert "outputRows64 = output.size(0);" in host
    assert "static_cast<uint32_t>(outputRows64)" in host
    assert "uint32_t outputRows{0};" in kernel
    assert "LayoutTagA layoutTagA{args.gemmShape.m(), args.gemmShape.k()}" in kernel
    assert "args.outputRows == 0 ? args.gemmShape.m() : args.outputRows" in kernel
    assert "output.size(0) == x.size(0)" not in host


def test_reusable_out_path_does_not_add_route_sum_synchronization():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    out_branch = host[host.index("if (outputOptional.has_value())"):
                      host.index("} else {", host.index("if (outputOptional.has_value())"))]
    assert "outputRows64 = output.size(0);" in out_branch
    assert ".sum().item" not in out_branch


def test_target_order_scheduler_uses_final_upstream_coordinate_fixes():
    overlay = PROJECT_ROOT / "csrc" / "catccos_overlay" / "include" / "catccos"
    comm_scheduler = (
        overlay
        / "comm"
        / "block"
        / "comm_block_scheduler_gmm_alltoallv_target_order.hpp"
    ).read_text(encoding="utf-8")
    mm_scheduler = (
        overlay
        / "dgemm"
        / "block"
        / "block_scheduler_gmm_alltoallv_target_order.hpp"
    ).read_text(encoding="utf-8")
    kernel = (
        overlay / "dgemm" / "kernel" / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")

    assert "MatrixCoord{uint32_t{1}, uint32_t{1}}" in comm_scheduler
    assert "blockOffset.m() / scheduler->blockShape.m()" in mm_scheduler
    assert "blockOffset.n()" in mm_scheduler
    assert "MatrixCoord blockOffsetB = blockOffset.GetCoordKN();" in kernel
    assert "RunTargetOrderAic(params)" in kernel
    assert "RunTargetOrderAiv(params)" in kernel


def test_target_order_workspace_check_tracks_full_n_staging():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert "groupsPerCommInRank" in host
    assert "stageRows" in host
    assert "static_cast<uint64_t>(worldSize) * static_cast<uint64_t>(n64)" in host
    assert "tilePerCommInRank >= nLoops" in host
    assert "TargetOrderLocalCopyRows" in host
    assert "TargetOrderLocalCopyColumns" in host
    assert "tiling.commBlockM, static_cast<uint32_t>(n64)" in host
    assert "commTileShape{localCopyRows, localCopyColumns}" in host


def test_ep8_256_expert_business_shapes_use_rank_invariant_tuned_profiles():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    assert (
        "KernelVariant::kM0_128, 128, 256, 256, 14, 64, 64, 4, 4"
        in host
    )
    assert (
        "KernelVariant::kM0_256, 256, 128, 256, 12, 16, 16, 1, 20"
        in host
    )
    assert "worldSize == 8 && expertNum == 256" in host
    assert (
        "isEightRank256ExpertCase && k == 1024 && n == 2048"
        in host
    )
    assert (
        "isEightRank256ExpertCase && k == 512 && n == 2048"
        in host
    )
    select_tiling = host[
        host.index("const TilingProfile &SelectTiling("):
        host.index("uint32_t TargetOrderLocalCopyColumns")
    ]
    assert "uint32_t m" not in select_tiling
    assert "m == 131072" not in select_tiling
    assert "may differ across" in select_tiling
    assert "rank-invariant" in select_tiling
    assert "SelectTiling(\n        k, n," in host
    assert "return kDefaultTiling;" in host


def test_tuned_profile_drives_kernel_and_all_runtime_tiling_fields():
    host = (ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma.cpp").read_text(
        encoding="utf-8"
    )
    device = (
        ASCEND950_ROOT / "grouped_mat_mul_all_to_allv_udma_device.h"
    ).read_text(encoding="utf-8")

    assert "using GroupedMatMulAllToAllVUdmaM0_256Config" in device
    assert "256, 128, 256>;" in device
    assert "TypeTag<GroupedMatMulAllToAllVUdmaM0_256Config>" in host
    assert "tiling.commInterval" in host
    assert "tiling.commTileM / 2" in host
    assert "tiling.commBlockM" in host
    assert "tiling.commDataSplit, tiling.commNpuSplit" in host
    assert "tiling.n0" in host
    assert "tiling.m0" in host
    assert "tiling.k0 == SelectedConfig::kK0" in host


def test_workspace_reuse_guard_avoids_unsigned_underflow():
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")
    assert (
        "commLoops >= WORKSPACE_STAGES &&\n"
        "                commIdx < commLoops - WORKSPACE_STAGES"
    ) in kernel


def test_get_void_completion_precedes_acknowledgement():
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")

    assert "quietStatus" not in kernel
    assert "GET CQ failure" not in kernel
    legacy = kernel[:kernel.index("private:")]
    target_order = kernel[kernel.index("void RunTargetOrderAivWithBlock("):]
    for path in (legacy, target_order):
        quiet = path.index("blockRemoteCopy.FinalizeBlockLoop();")
        fence = path.index("aclshmem_fence();", quiet)
        completion = path.index("ACLSHMEM_SIGNAL_ADD, srcRankIdx", fence)
        assert quiet < fence < completion


def test_device_printf_is_compiled_only_into_explicit_diagnostic_wheel():
    root_cmake = (PROJECT_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    setup_source = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")

    assert "option(CATCCOS_UDMA_DEVICE_DIAGNOSTICS" in root_cmake
    assert 'set(CATCCOS_UDMA_ASCEND_DIAGNOSTIC_FLAGS "-DASCENDC_DUMP=0")' in root_cmake
    assert "-DCATCCOS_UDMA_DEVICE_DIAGNOSTICS=1 -DASCENDC_DUMP=1" in root_cmake
    assert 'cmake_boolean_from_env("CATCCOS_UDMA_DEVICE_DIAGNOSTICS")' in setup_source
    for match in re.finditer("AscendC::printf\\(", kernel):
        guard = kernel.rfind(
            "#if defined(CATCCOS_UDMA_DEVICE_DIAGNOSTICS)", 0, match.start()
        )
        end = kernel.find("#endif", match.end())
        assert guard >= 0
        assert end > match.end()


def test_balanced_route_experiment_runtime_selects_local_or_global_sync():
    kernel = (
        PROJECT_ROOT
        / "csrc"
        / "catccos_overlay"
        / "include"
        / "catccos"
        / "dgemm"
        / "kernel"
        / "grouped_matmul_alltoallv_tla.hpp"
    ).read_text(encoding="utf-8")

    assert "UsesProducerBarrier(params.diagnosticMode)" in kernel
    assert "UsesRecycleBarrier(params.diagnosticMode)" in kernel
    assert kernel.count("AscendC::SyncAll<true>();") == 3
    assert "Original protocol: only this rank's active AIVs" in kernel
