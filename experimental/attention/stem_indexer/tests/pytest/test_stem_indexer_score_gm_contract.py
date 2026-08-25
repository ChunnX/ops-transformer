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

    set_dependency = (
        "SetFlag<HardEvent::MTE3_MTE2>(TOPK_MTE3_MTE2_EVENT);"
    )
    wait_dependency = (
        "WaitFlag<HardEvent::MTE3_MTE2>(TOPK_MTE3_MTE2_EVENT);"
    )
    process_topk = "inline void SIVector<SIT>::ProcessScoreGmTopK"
    first_batch = "RunScoreGmTopKBatch(info, rowIdx4, topkNum4, validEnd4"

    assert set_dependency in service
    assert wait_dependency in service
    process_offset = service.index(process_topk)
    wait_offset = service.index(wait_dependency, process_offset)
    batch_offset = service.index(first_batch, process_offset)
    assert wait_offset < batch_offset


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
