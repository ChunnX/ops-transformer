# Auto-generated script for copying Python files
file(GLOB PY_FILES "/home/developer/codes/ops-transformer/build/prepare_build/es_transformer_cust_build/generated_code/*.py")
foreach(py_file ${PY_FILES})
    get_filename_component(filename ${py_file} NAME)
    # 使用 configure_file 替代 COPY_FILE (兼容 CMake 3.16)
    configure_file(${py_file} "/home/developer/codes/ops-transformer/build/prepare_build/es_transformer_cust_build/python_package/es_transformer_cust/${filename}" COPYONLY)
    message(STATUS "Copied: ${filename}")
endforeach()
