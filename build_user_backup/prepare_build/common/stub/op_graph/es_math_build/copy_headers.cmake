# Auto-generated script for copying headers
file(GLOB H_FILES "/home/developer/codes/ops-transformer/build/prepare_build/common/stub/op_graph/es_math_build/generated_code/*.h")
foreach(h_file ${H_FILES})
    get_filename_component(filename ${h_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(${h_file} "/home/developer/codes/ops-transformer/build/prepare_build/es_packages_math/include/es_math/${filename}" COPYONLY)
    message(STATUS "Copied header: ${filename}")
endforeach()
