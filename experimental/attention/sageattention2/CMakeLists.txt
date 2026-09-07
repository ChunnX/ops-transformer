# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# One public L2 API and three registered internal operators.
add_subdirectory(op_host)
foreach(stage stats quant core)
    if(NOT sage_attention2_${stage} IN_LIST COMPILED_OPS)
        add_subdirectory(../sage_attention2_${stage}/op_host ${CMAKE_CURRENT_BINARY_DIR}/${stage})
    endif()
endforeach()
