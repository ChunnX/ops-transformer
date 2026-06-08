cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
set(PKG_NAME AscendOps)
project(${PKG_NAME} LANGUAGES ASC AICPU CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Torch must be found BEFORE ascend.cmake sets bisheng as compiler
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(SAVED_CXX ${CMAKE_CXX_COMPILER})  # save g++ before override
include(cmake/ascend.cmake)
include(cmake/func.cmake)

set(NPU_ARCH "ascend950" CACHE STRING "NPU architecture")
set(EXTENSION_MODULE_NAME "ascend_ops" CACHE STRING "Extension module name")

if(NOT DEFINED ASCEND_HOME)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend installation path")
  else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend installation path")
  endif()
endif()

# Derive CANN arch directory from uname -m
execute_process(COMMAND uname -m OUTPUT_VARIABLE UNAME_M OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CANN_ARCH "${UNAME_M}-linux")
message(STATUS "CANN arch directory: ${CANN_ARCH}")

# Shared includes: torch (via find_package), torch_npu (via cmake), Ascend (via ascend.cmake)
set(INCLUDE_DIRECTORIES
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_kernel
)

set(COMPILE_OPTIONS
    -O3 -fPIC -w
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
    -DTORCH_MODE
)

set(OBJECTS_LIST "" CACHE INTERNAL "List of objects")
add_subdirectory(csrc)

# _C needs g++ (torch headers), not bisheng
set(CMAKE_CXX_COMPILER ${SAVED_CXX})
set("EXTENSION_CPP" ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp)
file(GLOB TORCH_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch/*.cpp")
add_library(_C SHARED ${EXTENSION_CPP} ${OBJECTS_LIST} ${TORCH_SOURCES})
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch
    ${TORCH_NPU_PATH}/include/third_party/acl/inc
    ${ASCEND_DIR}/${CANN_ARCH}/include
    ${ASCEND_DIR}/${CANN_ARCH}/pkg_inc
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON PREFIX "" SUFFIX ".abi3.so" OUTPUT_NAME "_C")
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${TORCH_NPU_LIB_PATH} ${ASCEND_DIR}/${CANN_ARCH}/lib64)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_npu ascendcl platform register tiling_api runtime hccl hcomm ascendc_runtime)

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/${EXTENSION_MODULE_NAME}/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying _C.abi3.so to ${EXTENSION_MODULE_NAME}/")
