# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Split selected operators by their declared compilation frontend.
string(REPLACE "," ";" ASCENDC_OPS "${ASCEND_OP_NAME}")
set(NATIVE_OPS)
if(NOT BUILD_OPEN_PROJECT OR ENABLE_TEST)
  return()
endif()
file(GLOB _native_entries CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*/*/op_kernel/*.py")
foreach(_entry IN LISTS _native_entries)
  get_filename_component(_kernel_dir "${_entry}" DIRECTORY)
  get_filename_component(_op_dir "${_kernel_dir}" DIRECTORY)
  get_filename_component(_op "${_op_dir}" NAME)
  get_filename_component(_category_dir "${_op_dir}" DIRECTORY)
  get_filename_component(_category "${_category_dir}" NAME)
  should_add_module("${_category}" _include_module)
  if(NOT _include_module OR (NOT ASCEND_OP_NAME MATCHES "^[Aa][Ll][Ll]$" AND NOT _op IN_LIST ASCENDC_OPS))
    continue()
  endif()
  if(NOT _entry STREQUAL "${_kernel_dir}/${_op}.py")
    continue()
  endif()
  file(STRINGS "${_entry}" _hook REGEX "^@(register\\(|aicpu_kernel)")
  if(_hook)
    list(APPEND NATIVE_OPS "${_op}")
  endif()
endforeach()
if(NOT NATIVE_OPS)
  return()
endif()

if(NOT ASCEND_COMPUTE_UNIT STREQUAL "ascend950" OR
   (DEFINED ENABLE_OPS_KERNEL AND NOT ENABLE_OPS_KERNEL))
  if(ASCEND_OP_NAME MATCHES "^[Aa][Ll][Ll]$")
    set(NATIVE_OPS)
    return()
  endif()
  message(FATAL_ERROR "Selected Native operators currently require --pkg --soc=ascend950")
endif()
list(REMOVE_ITEM ASCENDC_OPS ${NATIVE_OPS})
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_native_stage "${CMAKE_CURRENT_BINARY_DIR}/native")
string(JOIN "," _native_names ${NATIVE_OPS})
if(ENABLE_BUILT_IN)
  set(_native_vendor builtin)
  set(_native_install "opp/built-in/op_impl/cannbotdsl")
else()
  set(_native_vendor "${VENDOR_NAME}")
  set(_native_install "packages/vendors/${VENDOR_NAME}_transformer/op_impl/cannbotdsl")
endif()
add_custom_target(ops_transformer_native ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${_native_stage}/python"
  COMMAND ${CMAKE_COMMAND} -E env "TORCH_EXTENSION_OPS=${_native_names}"
    "TORCH_EXTENSION_VENDOR=${_native_vendor}" CANNBOTDSL_NPU_ARCH=dav-3510
    "${Python3_EXECUTABLE}" setup.py build_py --build-lib "${_native_stage}/python"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/torch_extension"
  VERBATIM)
install(DIRECTORY "${_native_stage}/python/" DESTINATION "${_native_install}/python"
  COMPONENT ops-transformer PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/native_set_env.bash"
  DESTINATION "${_native_install}" RENAME set_env.bash COMPONENT ops-transformer)
