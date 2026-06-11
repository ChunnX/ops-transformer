# Kernel 文件用 -xasc 编译（bisheng 已在 project() 前设为全局编译器）
set(KERNEL_SOURCES
    dispatch_entry.cpp
    combine_entry.cpp
)

set_source_files_properties(
    ${KERNEL_SOURCES} PROPERTIES
    COMPILE_FLAGS "--npu-arch=dav-3510 -xasc"
)

add_library(mc2_kernels OBJECT ${KERNEL_SOURCES})
target_include_directories(mc2_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${INCLUDE_DIRECTORIES}
)

# Torch 集成模块
add_library(${EXTENSION_MODULE_NAME} SHARED mc2_torch.cpp)
target_compile_definitions(${EXTENSION_MODULE_NAME} PRIVATE
    TORCH_EXTENSION_NAME=${EXTENSION_MODULE_NAME}
)
target_include_directories(${EXTENSION_MODULE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${INCLUDE_DIRECTORIES}
)
target_link_libraries(${EXTENSION_MODULE_NAME} PRIVATE
    mc2_kernels
    ${TORCH_LIBRARIES}
    ${TORCH_NPU_LIBRARIES}
)
set_target_properties(${EXTENSION_MODULE_NAME} PROPERTIES PREFIX "")
