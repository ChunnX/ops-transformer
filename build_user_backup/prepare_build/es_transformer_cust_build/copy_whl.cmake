# Auto-generated script for copying whl files
file(GLOB WHL_FILES "/home/developer/codes/ops-transformer/build/prepare_build/es_transformer_cust_build/python_package/dist/*.whl")
foreach(whl_file ${WHL_FILES})
    get_filename_component(filename ${whl_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(${whl_file} "/home/developer/codes/ops-transformer/build/prepare_build/es_packages/whl/${filename}" COPYONLY)
    message(STATUS "Copied wheel: ${filename}")
endforeach()
