cmake_minimum_required(VERSION 3.16)
project(allto_allv_grouped_mat_mul_shmem_pta LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

foreach(required_var ASCEND_HOME_PATH CUST_OPAPI_LIBRARY TORCH_NPU_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "${required_var} is required")
    endif()
endforeach()
if(NOT EXISTS "${CUST_OPAPI_LIBRARY}")
    message(FATAL_ERROR "CUST_OPAPI_LIBRARY does not exist: ${CUST_OPAPI_LIBRARY}")
endif()
if(NOT EXISTS "${ASCEND_HOME_PATH}/include/acl/acl.h")
    message(FATAL_ERROR "AscendCL header not found: ${ASCEND_HOME_PATH}/include/acl/acl.h")
endif()
if(NOT EXISTS "${TORCH_NPU_ROOT}/include")
    message(FATAL_ERROR "torch_npu include directory not found: ${TORCH_NPU_ROOT}/include")
endif()

get_filename_component(OPS_TRANSFORMER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
set(OPS_TRANSFORMER_EXTERNAL_INCLUDE_DIR "${OPS_TRANSFORMER_ROOT}/common/include/external")
if(NOT EXISTS "${OPS_TRANSFORMER_ROOT}/common/include/external/aclnn_util.h")
    message(FATAL_ERROR
        "ACLNN API helper header not found: ${OPS_TRANSFORMER_EXTERNAL_INCLUDE_DIR}/aclnn_util.h")
endif()

if(DEFINED SHMEM_HOME_PATH AND NOT "${SHMEM_HOME_PATH}" STREQUAL "")
    set(SHMEM_INCLUDE_DIRS
        "${SHMEM_HOME_PATH}/shmem/include"
        "${SHMEM_HOME_PATH}/shmem/src"
        "${SHMEM_HOME_PATH}/shmem/src/device")
    set(SHMEM_LIBRARY_HINTS "${SHMEM_HOME_PATH}/shmem/lib")
elseif(DEFINED SHMEM_HOME AND NOT "${SHMEM_HOME}" STREQUAL "")
    set(SHMEM_INCLUDE_DIRS
        "${SHMEM_HOME}/include"
        "${SHMEM_HOME}/src/device")
    set(SHMEM_LIBRARY_HINTS "${SHMEM_HOME}/build/lib" "${SHMEM_HOME}/lib")
else()
    message(FATAL_ERROR "SHMEM_HOME_PATH (source SDK layout) or SHMEM_HOME (HCCL-integrated layout) is required")
endif()
foreach(shmem_include IN LISTS SHMEM_INCLUDE_DIRS)
    if(NOT EXISTS "${shmem_include}")
        message(FATAL_ERROR "SHMEM include directory not found: ${shmem_include}")
    endif()
endforeach()

find_library(SHMEM_LIBRARY NAMES shmem HINTS ${SHMEM_LIBRARY_HINTS} NO_DEFAULT_PATH)
find_library(SHMEM_UTILS_LIBRARY NAMES shmem_utils HINTS ${SHMEM_LIBRARY_HINTS} NO_DEFAULT_PATH)
if(NOT SHMEM_LIBRARY)
    message(FATAL_ERROR "libshmem.so not found under: ${SHMEM_LIBRARY_HINTS}")
endif()
if(NOT SHMEM_UTILS_LIBRARY)
    message(FATAL_ERROR "libshmem_utils.so not found under: ${SHMEM_LIBRARY_HINTS}")
endif()

get_filename_component(CUST_OPAPI_LIBRARY_DIR "${CUST_OPAPI_LIBRARY}" DIRECTORY)
set(CUST_OPAPI_INCLUDE_DIR_HINT "${CUST_OPAPI_INCLUDE_DIR}")
unset(CUST_OPAPI_INCLUDE_DIR CACHE)
unset(CUST_OPAPI_INCLUDE_DIR)
if(NOT "${CUST_OPAPI_INCLUDE_DIR_HINT}" STREQUAL "")
    set(CUST_OPAPI_INCLUDE_DIR "${CUST_OPAPI_INCLUDE_DIR_HINT}")
    if(NOT EXISTS "${CUST_OPAPI_INCLUDE_DIR}/aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h")
        message(FATAL_ERROR
            "V2 aclnn header not found under CUST_OPAPI_INCLUDE_DIR: ${CUST_OPAPI_INCLUDE_DIR}")
    endif()
else()
    find_path(CUST_OPAPI_INCLUDE_DIR
        NAMES aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h
        HINTS
            "${CUST_OPAPI_LIBRARY_DIR}/../include"
            "${CUST_OPAPI_LIBRARY_DIR}/../../include"
        NO_DEFAULT_PATH)
    if(NOT CUST_OPAPI_INCLUDE_DIR)
        message(FATAL_ERROR
            "aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h was not found; set CUST_OPAPI_INCLUDE_DIR")
    endif()
endif()
set(CUST_OPAPI_V2_HEADER
    "${CUST_OPAPI_INCLUDE_DIR}/aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h")
file(READ "${CUST_OPAPI_V2_HEADER}" CUST_OPAPI_V2_HEADER_CONTENT)
string(FIND "${CUST_OPAPI_V2_HEADER_CONTENT}" "shmemContext" SHMEM_CONTEXT_INDEX)
string(FIND "${CUST_OPAPI_V2_HEADER_CONTENT}" "commMode" LEGACY_COMM_MODE_INDEX)
if(SHMEM_CONTEXT_INDEX EQUAL -1 OR NOT LEGACY_COMM_MODE_INDEX EQUAL -1)
    message(FATAL_ERROR
        "PTA bridge requires the SHMEM V2 ABI (shmemContext and no commMode), but selected header is: "
        "${CUST_OPAPI_V2_HEADER}. Set CUST_OPAPI_INCLUDE_DIR to the include directory installed with "
        "CUST_OPAPI_LIBRARY.")
endif()

find_library(TORCH_NPU_LIBRARY NAMES torch_npu HINTS "${TORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH)
find_library(ASCENDCL_LIBRARY NAMES ascendcl HINTS "${ASCEND_HOME_PATH}/lib64" NO_DEFAULT_PATH)
if(NOT TORCH_NPU_LIBRARY)
    message(FATAL_ERROR "libtorch_npu.so not found under: ${TORCH_NPU_ROOT}/lib")
endif()
if(NOT ASCENDCL_LIBRARY)
    message(FATAL_ERROR "libascendcl.so not found under: ${ASCEND_HOME_PATH}/lib64")
endif()

add_library(allto_allv_grouped_mat_mul_shmem_pta SHARED
    allto_allv_grouped_mat_mul_shmem_torch.cpp)
target_compile_features(allto_allv_grouped_mat_mul_shmem_pta PRIVATE cxx_std_17)
set_target_properties(allto_allv_grouped_mat_mul_shmem_pta PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${TORCH_NPU_ROOT}/lib;${ASCEND_HOME_PATH}/lib64;${CUST_OPAPI_LIBRARY_DIR};${SHMEM_LIBRARY_HINTS}")

target_include_directories(allto_allv_grouped_mat_mul_shmem_pta PRIVATE
    "${CUST_OPAPI_INCLUDE_DIR}"
    "${OPS_TRANSFORMER_EXTERNAL_INCLUDE_DIR}"
    ${SHMEM_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    "${TORCH_NPU_ROOT}/include"
    "${ASCEND_HOME_PATH}/include")

separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(allto_allv_grouped_mat_mul_shmem_pta PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_link_libraries(allto_allv_grouped_mat_mul_shmem_pta PRIVATE
    ${TORCH_LIBRARIES}
    Python3::Python
    "${TORCH_NPU_LIBRARY}"
    "${ASCENDCL_LIBRARY}"
    "${CUST_OPAPI_LIBRARY}"
    "${SHMEM_LIBRARY}"
    "${SHMEM_UTILS_LIBRARY}")
