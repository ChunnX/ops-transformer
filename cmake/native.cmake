# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Split selected operators by their declared compilation frontend.
string(REPLACE "," ";" ASCENDC_SELECTED_OPS "${ASCEND_OP_NAME}")
if(NOT ASCENDC_SELECTED_OPS)
  set(ASCENDC_SELECTED_OPS ALL)
endif()
set(NATIVE_OPS)
if(NOT BUILD_OPEN_PROJECT OR ENABLE_TEST OR PREPARE_BUILD OR BUILD_OPS_RTY_KERNEL
   OR TESTS_EXAMPLE_OPS_TEST OR TESTS_UT_OPS_TEST)
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
  if(NOT _include_module OR (NOT ASCENDC_SELECTED_OPS MATCHES "^[Aa][Ll][Ll]$" AND NOT _op IN_LIST ASCENDC_SELECTED_OPS))
    continue()
  endif()
  if(NOT _entry STREQUAL "${_kernel_dir}/${_op}.py")
    continue()
  endif()
  file(STRINGS "${_entry}" _hook REGEX "^@(register\\(|aicpu_kernel)")
  if(_hook)
    if(_hook MATCHES "@aicpu_kernel" AND DEFINED ENABLE_AICPU AND NOT ENABLE_AICPU)
      if(NOT ASCENDC_SELECTED_OPS MATCHES "^[Aa][Ll][Ll]$")
        message(FATAL_ERROR "Selected Native operator ${_op} requires ENABLE_AICPU=ON")
      endif()
      continue()
    endif()
    list(APPEND NATIVE_OPS "${_op}")
  endif()
endforeach()
if(NOT NATIVE_OPS)
  return()
endif()

if(NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT OR
   (DEFINED ENABLE_OPS_KERNEL AND NOT ENABLE_OPS_KERNEL))
  if(ASCENDC_SELECTED_OPS MATCHES "^[Aa][Ll][Ll]$")
    set(NATIVE_OPS)
    return()
  endif()
  message(FATAL_ERROR "Selected Native operators currently require --pkg --soc=ascend950")
endif()
list(REMOVE_ITEM ASCENDC_SELECTED_OPS ${NATIVE_OPS})
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
    "${Python3_EXECUTABLE}" setup.py build_native --build-lib "${_native_stage}/python"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/torch_extension"
  VERBATIM)
install(DIRECTORY "${_native_stage}/python/" DESTINATION "${_native_install}/python"
  COMPONENT ops-transformer PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/native_set_env.bash"
  DESTINATION "${_native_install}" RENAME set_env.bash COMPONENT ops-transformer)
