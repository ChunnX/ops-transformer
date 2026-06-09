# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    # ACLNN API files
    target_sources(op_api_aclnnInner PRIVATE
        aclnn_engram_fetch.cpp
        aclnn_engram_fetch_wait.cpp
    )
endif()