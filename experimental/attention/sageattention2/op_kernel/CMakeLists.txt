include(${CMAKE_CURRENT_LIST_DIR}/../cmake/SageAttnDeps.cmake)

# 设备核源码与共享 include 列表（本目录两个 target 的唯一定义点）
set(SAGE_ATTN_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sageattention2.cpp
)

set(SAGE_ATTN_KERNEL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
    ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant
    ${SAGE_ATTN_CATLASS_INCLUDE_DIR}
)

if (DEFINED CMAKE_ASC_COMPILER)
    set_source_files_properties(${SAGE_ATTN_KERNEL_SOURCES} PROPERTIES LANGUAGE ASC)
endif()

if (BUILD_OPEN_PROJECT)
    if(TARGET op_kernel_binary)
        target_sources(op_kernel_binary PRIVATE ${SAGE_ATTN_KERNEL_SOURCES})
        target_include_directories(op_kernel_binary PRIVATE ${SAGE_ATTN_KERNEL_INCLUDE_DIRS})
    endif()
elseif(NOT TARGET sageattention2_kernel)
    # 独立构建支持 (在容器内或配合官方 CATLASS 毕昇编译器直接编译设备核)
    add_library(sageattention2_kernel OBJECT ${SAGE_ATTN_KERNEL_SOURCES})
    target_include_directories(sageattention2_kernel PUBLIC ${SAGE_ATTN_KERNEL_INCLUDE_DIRS})
    target_compile_features(sageattention2_kernel PUBLIC cxx_std_17)
endif()
