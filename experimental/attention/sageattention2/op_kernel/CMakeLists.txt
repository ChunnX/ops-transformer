set(SAGE_ATTN_KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sageattention2.cpp
)

if (DEFINED CMAKE_ASC_COMPILER)
    set_source_files_properties(${SAGE_ATTN_KERNEL_SOURCES} PROPERTIES LANGUAGE ASC)
endif()

if (BUILD_OPEN_PROJECT)
    if(TARGET op_kernel_binary)
        target_sources(op_kernel_binary PRIVATE ${SAGE_ATTN_KERNEL_SOURCES})
        target_include_directories(op_kernel_binary PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant
        )
    endif()
endif()

# 独立构建支持 (在容器内或配合 CATLASS 毕昇编译器直接编译设备核)
if(NOT TARGET op_kernel_binary AND NOT TARGET sageattention2_kernel)
    add_library(sageattention2_kernel OBJECT ${SAGE_ATTN_KERNEL_SOURCES})
    target_include_directories(sageattention2_kernel PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
        ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant
    )
    target_compile_features(sageattention2_kernel PUBLIC cxx_std_17)
endif()
