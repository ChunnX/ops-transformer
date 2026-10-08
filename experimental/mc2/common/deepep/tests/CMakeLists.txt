# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

find_package(Threads REQUIRED)
add_executable(mc2_session_test session_test.cpp ../csrc/runtime/session.cpp)
target_include_directories(mc2_session_test PRIVATE stubs ../csrc)
target_link_libraries(mc2_session_test PRIVATE Threads::Threads)
foreach(scenario normal invalid low-latency ep16-small-workspace udma megamoe
        init-fault allocation-fault memset-fault foreign abandoned close-fault)
    add_test(NAME mc2_session_${scenario} COMMAND mc2_session_test "${scenario}")
endforeach()
