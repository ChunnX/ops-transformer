cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510 for A5 (Ascend 950)")

find_package(ASC REQUIRED)

project(moe_distribute_ops_a5 LANGUAGES ASC CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# CANN headers for host-side ACL/HCCL
if(DEFINED ENV{ASCEND_HOME_PATH})
    file(GLOB CANN_ARCH_DIR "$ENV{ASCEND_HOME_PATH}/*-linux")
    foreach(D ${CANN_ARCH_DIR})
        if(IS_DIRECTORY "${D}/include")
            include_directories("${D}/include")
        endif()
        if(IS_DIRECTORY "${D}/asc/include")
            include_directories("${D}/asc/include")
        endif()
    endforeach()
endif()

add_executable(dispatch dispatch.asc dispatch_main.asc)
add_executable(combine combine.asc combine_main.asc)

target_compile_options(dispatch PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
target_compile_options(combine PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
