cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-2201 (910/910C), dav-3510 (950/910PR)")

find_package(ASC REQUIRED)

project(pinned_rw LANGUAGES ASC CXX)

add_executable(demo
    pinned_rw.asc
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)