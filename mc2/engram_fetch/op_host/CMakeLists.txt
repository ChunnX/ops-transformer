# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        engram_fetch_def.cpp
        engram_fetch_wait_def.cpp
    )
    
# Tiling sources will be auto-discovered by add_modules_sources_with_soc
    
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON 
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api 
        OP_MC2_ENABLE ON 
        OPTYPE engram_fetch ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(engram_fetch_depends mc2/common mc2/3rd PARENT_SCOPE)
else()
    add_mc2_modules_sources(OPTYPE engram_fetch ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE) 
endif()