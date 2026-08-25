from pathlib import Path


ARCH_920R1 = Path(__file__).resolve().parents[2] / "op_kernel" / "arch-920r1"


def test_uint16_always_uses_score_gm_topk():
    kernel = (ARCH_920R1 / "stem_indexer_kernel.h").read_text(encoding="utf-8")

    assert "if constexpr (SIVector<SIT>::ENABLE_SCORE_GM_TOPK)" in kernel
    assert "if (s2BlockNum > 1U)" not in kernel
    assert "vectorService.ProcessVec1ScoreGm(runInfo);" in kernel
    assert "vectorService.ProcessScoreGmTopK(runInfo);" in kernel


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
