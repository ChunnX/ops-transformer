# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 fk2323
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# -----------------------------------------------------------------------------------------------------------

# The official non-master PR UT entry point invokes the regular CMake tree.
# Keep the implementation under experimental/attention, but expose it to that
# UT entry point without changing the shared build scripts.
if(ENABLE_TEST AND NOT ENABLE_EXPERIMENTAL)
    add_subdirectory(
        "${OPS_TRANSFORMER_DIR}/experimental/attention/fmm016_streaming_top_k_blocked_v4"
        "${CMAKE_CURRENT_BINARY_DIR}/fmm016_streaming_top_k_blocked_v4"
    )
endif()
