from pathlib import Path


ARCH_920R1 = Path(__file__).resolve().parents[2] / "op_kernel" / "arch-920r1"


def test_uint16_and_uint32_always_use_score_gm_topk():
    kernel = (ARCH_920R1 / "stem_indexer_kernel.h").read_text(encoding="utf-8")
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    assert "std::is_same_v<SCORE_T, uint16_t> || std::is_same_v<SCORE_T, uint32_t>" in service
    assert "if constexpr (SIVector<SIT>::ENABLE_SCORE_GM_TOPK)" in kernel
    assert "if (s2BlockNum > 1U)" not in kernel
    assert "vectorService.ProcessVec1ScoreGm(runInfo);" in kernel
    assert "vectorService.ProcessScoreGmTopK(runInfo);" in kernel


def test_score_gm_storage_uses_selected_score_type():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    assert "GlobalTensor<SCORE_T> scoreWorkspaceGm;" in service
    assert "StemIndexerTopk<SCORE_T, SCORE_GM_TRUNK_LEN> scoreGmTopkOp_;" in service
    assert "sizeof(SCORE_T)" in service[service.index("SCORE_GM_MERGE_SLOT_BYTES"):service.index("private:")]
    assert "(__gm__ SCORE_T *)workspace" in service
    assert "LocalTensor<SCORE_T> scoreLocal" in service
    assert "LocalTensor<SCORE_T> batchHistoryLocal" in service


def test_score_gm_dependent_reinterpret_casts_use_template_disambiguator():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    dependent_cast_lines = [
        line.strip()
        for line in service.splitlines()
        if "fastBuf[" in line and "ReinterpretCast<uint32_t>()" in line
    ]

    assert len(dependent_cast_lines) == 4
    assert all(".template ReinterpretCast<uint32_t>()" in line for line in dependent_cast_lines)


def test_a6_uint32_score_gm_uses_compact_buffers_within_single_384k_ub():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    assert "SCORE_GM_HISTORY_ALLOC_BYTES" in service
    assert "SCORE_GM_GLOBAL_ALLOC_BYTES" in service
    assert "VECTOR_UB_CAPACITY_BYTES = 384U * 1024U" in service
    assert "static_assert(VECTOR_UB_USED_BYTES <= VECTOR_UB_CAPACITY_BYTES" in service
    assert "InitBuffer(historyValueBuf_, SCORE_GM_HISTORY_ALLOC_BYTES)" in service
    assert "InitBuffer(globalIndexBuf_, SCORE_GM_GLOBAL_ALLOC_BYTES)" in service


def test_a6_score_gm_workspace_bytes_follow_topk_score_precision():
    tiling = (ARCH_920R1.parents[1] / "op_host" / "stem_indexer_tiling.cpp").read_text(encoding="utf-8")

    assert "tilingInfo->topkScorePrecision == TOPK_SCORE_PRECISION_UINT32" in tiling
    assert "sizeof(uint32_t)" in tiling[tiling.index("size_t workspaceSize"):tiling.index("workspaces[0]")]
    assert "scoreElementBytes" in tiling


def test_score_gm_write_is_ordered_before_mte2_reload():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    set_dependency = "SetFlag<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_);"
    wait_dependency = "WaitFlag<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_);"
    store_begin = service.index("inline void SIVector<SIT>::StoreScoreToWorkspace")
    store_end = service.index("inline void SIVector<SIT>::ProcessVec1ScoreGm", store_begin)
    store = service[store_begin:store_end]
    store_compact = "".join(store.split())
    store_to_gm = "DataCopy(scoreWorkspaceGm[scoreGmOffset], scoreLocal"

    assert store.index(store_to_gm) < store.index(set_dependency)
    assert (
        "if(info.isLastS2InnerLoop){"
        "SetFlag<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_);"
        "}"
    ) in store_compact

    process_topk = service.index("inline void SIVector<SIT>::ProcessScoreGmTopK")
    first_batch = "RunScoreGmTopKBatch(info, rowIdx4, topkNum4, validEnd4"
    wait_offset = service.index(wait_dependency, process_topk)
    batch_offset = service.index(first_batch, process_topk)
    assert wait_offset < batch_offset


def test_a6_cube_d512_l1_layout_and_loops_are_consistent():
    service = (ARCH_920R1 / "stem_indexer_service_cube.h").read_text(encoding="utf-8")

    assert "static constexpr uint64_t KEY_BUF_NUM = 4;" in service
    assert "static constexpr uint64_t QUERY_BUF_NUM = 4;" in service
    assert "static constexpr uint64_t D_BASIC_BLOCK_L1 = 512;" in service
    assert "static constexpr uint64_t D_BASIC_BLOCK_L0 = 256;" in service
    assert "QUERY_BUFFER_OFFSET = M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1" in service
    assert "KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1" in service
    assert "InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(Q_T))" in service
    assert "InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(K_T))" in service
    assert "kGmOffset += D_BASIC_BLOCK_L1" in service
    assert "kL1Offset < D_BASIC_BLOCK_L1; kL1Offset += D_BASIC_BLOCK_L0" in service
    assert "keyL1BufIdx_ & (KEY_BUF_NUM - 1U)" in service
    assert "dL1Idx & (QUERY_BUF_NUM - 1U)" in service


def test_a6_cube_all_hard_events_are_dynamic_and_balanced():
    service = (ARCH_920R1 / "stem_indexer_service_cube.h").read_text(encoding="utf-8")

    fixed_events = (
        "KEY_MTE1_MTE2_EVENT",
        "QUERY_MTE1_MTE2_EVENT",
        "M_MTE1_EVENT",
        "MTE2_MTE1_EVENT",
        "MTE1_M_EVENT",
        "FIX_M_EVENT",
        "M_FIX_EVENT",
    )
    assert all(event not in service for event in fixed_events)

    alloc_begin = service.index("inline void SIMatmul<SIT>::AllocEventID")
    free_begin = service.index("inline void SIMatmul<SIT>::FreeEventID", alloc_begin)
    alloc = service[alloc_begin:free_begin]
    free = service[free_begin:]
    alloc_compact = "".join(alloc.split())
    free_compact = "".join(free.split())

    array_events = (
        ("keyMte1Mte2Event_[slot]", "MTE1_MTE2", True),
        ("queryMte1Mte2Event_[slot]", "MTE1_MTE2", True),
        ("mMte1Event_[slot]", "M_MTE1", True),
        ("fixMEvent_[slot]", "FIX_M", True),
        ("mFixEvent_[slot]", "M_FIX", False),
    )
    scalar_events = (
        ("mte2Mte1Event_", "MTE2_MTE1"),
        ("mte1MEvent_", "MTE1_M"),
    )
    for member, event_type, has_initial_token in array_events:
        assert f"{member}=static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::{event_type}>())" in alloc_compact
        assert f"ReleaseEventID<HardEvent::{event_type}>({member})" in free_compact
        if has_initial_token:
            assert f"SetFlag<HardEvent::{event_type}>({member})" in alloc
    for member, event_type in scalar_events:
        assert f"{member}=static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::{event_type}>())" in alloc_compact
        assert f"ReleaseEventID<HardEvent::{event_type}>({member})" in free_compact
        assert f"SetFlag<HardEvent::{event_type}>({member})" in service
        assert f"WaitFlag<HardEvent::{event_type}>({member})" in service

    assert "SetFlag<HardEvent::M_FIX>(mFixEvent_[slot])" not in alloc
    assert "SetFlag<HardEvent::MTE2_MTE1>(mte2Mte1Event_)" not in alloc
    assert "SetFlag<HardEvent::MTE1_M>(mte1MEvent_)" not in alloc
    assert "SetFlag<HardEvent::M_FIX>(mFixEvent_[l0cSlot])" in service
    assert "WaitFlag<HardEvent::M_FIX>(mFixEvent_[l0cSlot])" in service


def test_a6_vector_all_hard_events_are_dynamic_and_balanced():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    fixed_events = (
        "VEC1_V_MTE2_EVENT",
        "VEC1_MTE2_V_EVENT",
        "TOPK_MTE3_MTE2_EVENT",
        "TOPK_V_MTE3_EVENT",
        "TOPK_MTE3_V_EVENT",
    )
    assert all(event not in service for event in fixed_events)

    alloc_begin = service.index("inline void SIVector<SIT>::AllocEventID")
    free_begin = service.index("inline void SIVector<SIT>::FreeEventID", alloc_begin)
    alloc = service[alloc_begin:free_begin]
    free = service[free_begin:service.index("inline void SIVector<SIT>::InitSparseIndicesToNegOne", free_begin)]
    alloc_compact = "".join(alloc.split())
    free_compact = "".join(free.split())
    events = (
        ("vec1VMte2Event_[slot]", "V_MTE2"),
        ("vec1Mte2VEvent_[slot]", "MTE2_V"),
        ("topkMte3Mte2Event_", "MTE3_MTE2"),
        ("topkVMte3Event_", "V_MTE3"),
        ("topkMte3VEvent_", "MTE3_V"),
    )
    for member, event_type in events:
        assert f"{member}=static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::{event_type}>())" in alloc_compact
        assert f"ReleaseEventID<HardEvent::{event_type}>({member})" in free_compact

    assert "SetFlag<HardEvent::V_MTE2>(vec1VMte2Event_[slot])" in alloc
    assert "SetFlag<HardEvent::MTE3_V>(topkMte3VEvent_)" in alloc
    assert "SetFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[slot])" not in alloc
    assert "SetFlag<HardEvent::MTE3_MTE2>(topkMte3Mte2Event_)" not in alloc
    assert "SetFlag<HardEvent::V_MTE3>(topkVMte3Event_)" not in alloc
    assert "SetFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[pingpong])" in service
    assert "WaitFlag<HardEvent::MTE2_V>(vec1Mte2VEvent_[pingpong])" in service
    assert "SetFlag<HardEvent::V_MTE3>(topkVMte3Event_)" in service
    assert "WaitFlag<HardEvent::V_MTE3>(topkVMte3Event_)" in service


def test_initial_blocks_are_masked_before_first_score_tile_store():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    store_begin = service.index("inline void SIVector<SIT>::StoreScoreToWorkspace")
    store_end = service.index("inline void SIVector<SIT>::ProcessVec1ScoreGm", store_begin)
    store = service[store_begin:store_end]

    first_tile_guard = "if (info.s2Idx == 0U && initialBlocks_ > 0U)"
    mask_initial_blocks = (
        "TopkUtils::PadMrgRowsInitBlockVF(scoreLocal, S2_BASIC_BLOCK, info.actMBaseSize, 0U, initialBlocks_);"
    )
    store_to_gm = "DataCopy(scoreWorkspaceGm[scoreGmOffset], scoreLocal"

    assert first_tile_guard in store
    assert mask_initial_blocks in store
    assert store.index(first_tile_guard) < store.index(mask_initial_blocks) < store.index(store_to_gm)


def test_a6_uint32_initial_block_padding_uses_strided_duplicate_without_row_loop():
    utils = (ARCH_920R1 / "vf" / "stem_indexer_topk_utils.h").read_text(encoding="utf-8")

    wrapper_begin = utils.index("inline void PadMrgRowsInitBlockVF(LocalTensor")
    wrapper_end = utils.index("inline void PadMrgFourRowsFirstS2RangeVF", wrapper_begin)
    wrapper = "".join(utils[wrapper_begin:wrapper_end].split())

    assert "ifconstexpr(std::is_same_v<SCORE_T,uint32_t>)" in wrapper
    assert "for(" not in wrapper
    assert "dstRepStride=static_cast<uint8_t>(rowStride*sizeof(uint32_t)/32U)" in wrapper
    assert "Duplicate(mrgValueLocal[topkSelectNumAlign],static_cast<uint32_t>(0),mask," in wrapper
    assert "repeatTimes,1U,dstRepStride);" in wrapper
    assert "else{PadMrgRowsInitBlockVFImpl<SCORE_T>(" in wrapper


def test_direct_output_skips_zero_length_index_vector_and_mte3():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    direct_begin = service.index("inline void SIVector<SIT>::WriteDirectIndices")
    direct_end = service.index("inline void SIVector<SIT>::CopyOutBatchRows", direct_begin)
    direct = service[direct_begin:direct_end]
    direct_compact = "".join(direct.split())

    guarded_create = (
        "if(outputLen>0U){"
        "CreateVecIndex(indicesOutLocal.ReinterpretCast<int32_t>(),0,outputLen);"
        "}"
    )
    guarded_indices_copy = (
        "if(outputLen>0U){"
        "DataCopyPad(sparseIndicesGm[baseOutOffset+"
        "static_cast<uint64_t>(gIdx)*constInfo_.qSeqSize*constInfo_.kSeqSize],"
        "indicesOutLocal.ReinterpretCast<int32_t>(),copyOut);"
        "}"
    )

    assert guarded_create in direct_compact
    assert "DataCopyPad(sparseSeqLenGm" in direct_compact
    assert guarded_indices_copy in direct_compact


def test_output_cleanup_partitions_match_launched_vector_cores():
    service = (ARCH_920R1 / "stem_indexer_service_vector.h").read_text(encoding="utf-8")

    init_begin = service.index("inline void SIVector<SIT>::InitOutputSingleCore")
    init_end = service.index("inline void SIVector<SIT>::InitParams", init_begin)
    init = service[init_begin:init_end]

    # The failing device launches 54 logical Vector cores. Dividing cleanup into the
    # compile-time maximum of 64 leaves exactly the padding tails reported by the two
    # mixed-zero batch failures.
    align_elems = 128  # 512 bytes / sizeof(int32_t)

    def uncovered_size(total_size, partition_count, launched_core_count):
        single_core_size = (total_size + partition_count - 1) // partition_count
        single_core_size = (single_core_size + align_elems - 1) // align_elems * align_elems
        return max(total_size - launched_core_count * single_core_size, 0)

    launched_core_count = 54
    si_gen_1258_size = 3 * 64 * 1 * 256
    si_gen_1349_size = 3 * 64 * 8 * 64
    assert uncovered_size(si_gen_1258_size, 64, launched_core_count) == 7680
    assert uncovered_size(si_gen_1349_size, 64, launched_core_count) == 15360
    assert uncovered_size(si_gen_1258_size, launched_core_count, launched_core_count) == 0
    assert uncovered_size(si_gen_1349_size, launched_core_count, launched_core_count) == 0

    redline_q_blocks = 30720 // 128
    redline_k_blocks = 30720 // 128
    redline_size = 40 * 32 * redline_q_blocks * redline_k_blocks
    fixed_partition_size = (redline_size // 64 + align_elems - 1) // align_elems * align_elems
    first_uncovered_element = fixed_partition_size * launched_core_count
    first_uncovered_row = first_uncovered_element // redline_k_blocks
    rows_per_batch = 32 * redline_q_blocks
    redline_batch, row_in_batch = divmod(first_uncovered_row, rows_per_batch)
    redline_q_head, redline_q_block = divmod(row_in_batch, redline_q_blocks)
    assert (redline_batch, redline_q_head, redline_q_block) == (33, 24, 0)

    assert "constInfo_.usedCoreNum" in init
    assert "AIV_CORE_NUM" not in init
