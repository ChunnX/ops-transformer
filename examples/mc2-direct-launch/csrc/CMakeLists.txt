set(KERNEL_SOURCES
    dispatch_entry.cpp
    combine_entry.cpp
)

set_source_files_properties(${KERNEL_SOURCES} PROPERTIES
    COMPILE_FLAGS "--npu-arch=dav-3510 -xasc -fPIC")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(mc2_kernels OBJECT ${KERNEL_SOURCES})
target_include_directories(mc2_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${ASCEND_INCLUDE_DIRS}
)

set(EXTENSION_MODULE_NAME mc2_direct_launch)

add_library(${EXTENSION_MODULE_NAME} SHARED mc2_torch.cpp)
target_compile_definitions(${EXTENSION_MODULE_NAME} PRIVATE
    TORCH_EXTENSION_NAME=${EXTENSION_MODULE_NAME})
target_include_directories(${EXTENSION_MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${Python3_INCLUDE_DIRS}
)
target_link_libraries(${EXTENSION_MODULE_NAME} PRIVATE
    mc2_kernels
    ${TORCH_LIBRARIES}
    ${TORCH_NPU_LIBRARIES}
)
set_target_properties(${EXTENSION_MODULE_NAME} PROPERTIES PREFIX "")
