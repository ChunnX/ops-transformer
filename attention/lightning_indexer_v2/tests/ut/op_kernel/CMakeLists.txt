if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(lightning_indexer_v2_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/lightning_indexer_v2_tiling.cpp
        )
    AddOpTestCase(lightning_indexer_v2 "Ascend910B1" "-DORIG_DTYPE_Q=DT_BF16" "${lightning_indexer_v2_tiling_files}")
    if (TARGET lightning_indexer_v2_Ascend910B1_cases_obj)
        target_include_directories(lightning_indexer_v2_Ascend910B1_cases_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host
        )
    endif()
endif()
