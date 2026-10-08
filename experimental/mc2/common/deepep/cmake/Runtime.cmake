# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Adapted from DeepEP; original copyright and license notices follow.

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
include_guard(GLOBAL)
get_filename_component(MC2_DEEPEP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
get_filename_component(MC2_ROOT "${MC2_DEEPEP_ROOT}/../.." ABSOLUTE)

function(mc2_add_deepep_runtime)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "MC2 DeepEP operators require Linux, CANN 9.2.0 and Ascend950")
    endif()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/output" CACHE PATH "Install prefix" FORCE)
    endif()
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit")
    set(SHMEM_ROOT "$ENV{SHMEM_ROOT}" CACHE PATH "Pinned SHMEM SDK")
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")
    find_package(pybind11 CONFIG REQUIRED)
    set(identity_args "")
    if(MC2_BUILD_MEGAMOE)
        set(timer_enabled OFF)
        if(DEEPEP_MEGAMOE_TIMER)
            set(timer_enabled ON)
        endif()
        list(APPEND identity_args --catlass "${CATLASS_ROOT}" --timer "${timer_enabled}")
    endif()
    execute_process(COMMAND "${Python_EXECUTABLE}"
        "${MC2_DEEPEP_ROOT}/scripts/check_dependencies.py"
        --cann "${ASCEND_HOME_PATH}" --shmem "${SHMEM_ROOT}"
        --output "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" ${identity_args}
        COMMAND_ERROR_IS_FATAL ANY)
    include("${MC2_DEEPEP_ROOT}/cmake/Sdk.cmake")
    find_program(MC2_BISHENG bisheng HINTS
        "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin"
        "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin" NO_DEFAULT_PATH REQUIRED)

    set(dispatch_dir "${MC2_ROOT}/low_latency_dispatch")
    set(device_sources "${dispatch_dir}/op_kernel/low_latency_dispatch.cpp")
    set(host_sources "${dispatch_dir}/op_host/low_latency_dispatch.cpp")
    set(kernel_roots "${dispatch_dir}/op_kernel")
    set(host_definitions "")
    if(MC2_BUILD_MEGAMOE)
        list(APPEND host_definitions DEEPEP_BUILD_MEGAMOE=1)
    endif()
    if(MC2_BUILD_COMBINE)
        set(combine_dir "${MC2_ROOT}/low_latency_combine")
        list(APPEND device_sources "${combine_dir}/op_kernel/low_latency_combine.cpp")
        list(APPEND host_sources "${combine_dir}/op_host/low_latency_combine.cpp")
        list(APPEND kernel_roots "${combine_dir}/op_kernel")
        list(APPEND host_definitions DEEPEP_BUILD_LOW_LATENCY_COMBINE=1)
    endif()
    set(device_include
        "${MC2_DEEPEP_ROOT}/include" "${dispatch_dir}/op_kernel"
        "${ASCEND_HOME_PATH}/include"
        "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
        "${ASCEND_HOME_PATH}/compiler/tikcpp"
        "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
        "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl"
        "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
        "${SHMEM_ROOT}/include" "${SHMEM_ROOT}/src"
        "${SHMEM_ROOT}/src/device" "${SHMEM_ROOT}/src/device_simt")
    find_path(MC2_RUNTIME_INCLUDE rt.h HINTS
        "${ASCEND_HOME_PATH}/include/runtime"
        "${ASCEND_HOME_PATH}/include/experiment/runtime"
        "${ASCEND_HOME_PATH}/pkg_inc/runtime" NO_DEFAULT_PATH)
    if(MC2_RUNTIME_INCLUDE)
        list(APPEND device_include "${MC2_RUNTIME_INCLUDE}")
    endif()
    set(device_flags -O2 -std=c++17 -fPIC -shared -xasc --npu-arch=dav-3510
        --cce-auto-infer-kernel-type=false --cce-fatobj-link
        -DCATLASS_ARCH=3510 -DUSE_SIMT -DL2_CACHE_HINT
        -Wl,-soname,libmc2_deepep_kernels.so -Wl,-z,relro,-z,now,-z,noexecstack)
    foreach(path IN LISTS device_include)
        list(APPEND device_flags "-I${path}")
    endforeach()
    set(kernel_headers "")
    foreach(root IN LISTS kernel_roots)
        file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${root}/*.h" "${root}/*.hpp")
        list(APPEND kernel_headers ${headers})
    endforeach()
    set(kernel_library "${CMAKE_CURRENT_BINARY_DIR}/libmc2_deepep_kernels.so")
    add_custom_command(OUTPUT "${kernel_library}"
        COMMAND "${MC2_BISHENG}" ${device_flags} ${device_sources}
            "-L${ASCEND_HOME_PATH}/lib64"
            "-L${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
            "-L${SHMEM_ROOT}/lib" -lruntime -lascendcl -lshmem -ldl -lstdc++ -lm
            -o "${kernel_library}"
        DEPENDS ${device_sources} ${kernel_headers}
            "${MC2_DEEPEP_ROOT}/include/mc2_kernel_launch.hpp" "${MC2_SHMEM_LIBRARY}"
        VERBATIM)
    add_custom_target(mc2_deepep_device DEPENDS "${kernel_library}")
    add_library(mc2_deepep_kernels SHARED IMPORTED)
    set_target_properties(mc2_deepep_kernels PROPERTIES IMPORTED_LOCATION "${kernel_library}")
    pybind11_add_module(mc2_deepep_runtime
        "${MC2_DEEPEP_ROOT}/csrc/bindings.cpp"
        "${MC2_DEEPEP_ROOT}/csrc/runtime/session.cpp" ${host_sources})
    set_target_properties(mc2_deepep_runtime PROPERTIES OUTPUT_NAME _C
        INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
    target_include_directories(mc2_deepep_runtime PRIVATE
        "${MC2_DEEPEP_ROOT}/csrc" "${MC2_DEEPEP_ROOT}/include"
        "${dispatch_dir}/op_kernel" "${ASCEND_HOME_PATH}/include" "${SHMEM_ROOT}/include")
    target_compile_definitions(mc2_deepep_runtime PRIVATE ${host_definitions})
    target_compile_options(mc2_deepep_runtime PRIVATE -O2 -Wall -Wextra -fstack-protector-strong)
    target_link_options(mc2_deepep_runtime PRIVATE -Wl,-z,relro,-z,now,-z,noexecstack)
    target_link_libraries(mc2_deepep_runtime PRIVATE
        mc2_ascendcl mc2_shmem mc2_deepep_kernels ${CMAKE_DL_LIBS})
    add_dependencies(mc2_deepep_runtime mc2_deepep_device)
    install(TARGETS mc2_deepep_runtime LIBRARY DESTINATION python/ops_transformer_mc2)
    install(FILES "${kernel_library}" "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json"
        DESTINATION python/ops_transformer_mc2)
    install(DIRECTORY "${MC2_DEEPEP_ROOT}/python/" "${dispatch_dir}/python/"
        DESTINATION python FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
    install(DIRECTORY "${MC2_DEEPEP_ROOT}/licenses/"
        DESTINATION python/ops_transformer_mc2/licenses)
endfunction()
