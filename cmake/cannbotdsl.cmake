# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Split selected operators by publication model. Legacy @register and
# @aicpu_kernel sources keep using the Python resource builder. A
# @cannbotdsl.entry source is built as a frozen OperatorBundle and attached to
# the built-in ACLNN DSO.
string(REPLACE "," ";" ASCENDC_BUILD_OPS "${ASCEND_OP_NAME}")
if(NOT ASCENDC_BUILD_OPS)
  set(ASCENDC_BUILD_OPS ALL)
endif()
set(CANNBOTDSL_OPS)
set(CANNBOTDSL_LEGACY_OPS)
set(CANNBOTDSL_BUNDLE_OPS)
set(CANNBOTDSL_BUNDLE_SOURCES)
if(NOT BUILD_OPEN_PROJECT OR ENABLE_TEST OR PREPARE_BUILD OR BUILD_OPS_RTY_KERNEL
   OR TESTS_EXAMPLE_OPS_TEST OR TESTS_UT_OPS_TEST)
  return()
endif()

file(GLOB _cannbotdsl_entries CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*/*/op_kernel/*.py")
foreach(_entry IN LISTS _cannbotdsl_entries)
  get_filename_component(_kernel_dir "${_entry}" DIRECTORY)
  get_filename_component(_op_dir "${_kernel_dir}" DIRECTORY)
  get_filename_component(_op "${_op_dir}" NAME)
  get_filename_component(_category_dir "${_op_dir}" DIRECTORY)
  get_filename_component(_category "${_category_dir}" NAME)
  should_add_module("${_category}" _include_module)
  if(NOT _include_module OR
     (NOT ASCENDC_BUILD_OPS MATCHES "^[Aa][Ll][Ll]$" AND
      NOT _op IN_LIST ASCENDC_BUILD_OPS))
    continue()
  endif()
  if(NOT _entry STREQUAL "${_kernel_dir}/${_op}.py")
    continue()
  endif()

  file(STRINGS "${_entry}" _legacy_hook REGEX "^@(register\\(|aicpu_kernel)")
  file(STRINGS "${_entry}" _bundle_hook REGEX "^@cannbotdsl\\.entry\\(")
  if(_legacy_hook)
    if(_legacy_hook MATCHES "@aicpu_kernel" AND DEFINED ENABLE_AICPU AND NOT ENABLE_AICPU)
      if(NOT ASCENDC_BUILD_OPS MATCHES "^[Aa][Ll][Ll]$")
        message(FATAL_ERROR "Selected Native operator ${_op} requires ENABLE_AICPU=ON")
      endif()
      continue()
    endif()
    list(APPEND CANNBOTDSL_LEGACY_OPS "${_op}")
    list(APPEND CANNBOTDSL_OPS "${_op}")
  elseif(_bundle_hook)
    if(NOT ENABLE_BUILT_IN)
      if(NOT ASCENDC_BUILD_OPS MATCHES "^[Aa][Ll][Ll]$")
        message(FATAL_ERROR
          "Selected frozen CANNBotDSL bundles currently require a built-in package")
      endif()
      continue()
    endif()
    list(APPEND CANNBOTDSL_BUNDLE_OPS "${_op}")
    list(APPEND CANNBOTDSL_BUNDLE_SOURCES "${_entry}")
    list(APPEND CANNBOTDSL_OPS "${_op}")
  endif()
endforeach()
if(NOT CANNBOTDSL_OPS)
  return()
endif()

if(NOT "ascend950" IN_LIST ASCEND_COMPUTE_UNIT OR
   (DEFINED ENABLE_OPS_KERNEL AND NOT ENABLE_OPS_KERNEL))
  if(ASCENDC_BUILD_OPS MATCHES "^[Aa][Ll][Ll]$")
    set(CANNBOTDSL_OPS)
    return()
  endif()
  message(FATAL_ERROR "Selected CANNBotDSL operators currently require --pkg --soc=ascend950")
endif()
list(REMOVE_ITEM ASCENDC_BUILD_OPS ${CANNBOTDSL_OPS})
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Keep one stable umbrella target so the root build can distinguish a package
# containing only CANNBotDSL operators from a genuinely empty selection.
add_custom_target(ops_transformer_cannbotdsl ALL)

if(CANNBOTDSL_LEGACY_OPS)
  set(_cannbotdsl_stage "${CMAKE_CURRENT_BINARY_DIR}/cannbotdsl")
  string(JOIN "," _cannbotdsl_names ${CANNBOTDSL_LEGACY_OPS})
  if(ENABLE_BUILT_IN)
    set(_cannbotdsl_vendor builtin)
    set(_cannbotdsl_install "opp/built-in/op_impl/cannbotdsl")
  else()
    set(_cannbotdsl_vendor "${VENDOR_NAME}")
    set(_cannbotdsl_install
      "packages/vendors/${VENDOR_NAME}_transformer/op_impl/cannbotdsl")
  endif()
  set(_cannbotdsl_legacy_python "${Python3_EXECUTABLE}")
  if(CANNBOTDSL_PYTHON_EXECUTABLE)
    set(_cannbotdsl_legacy_python "${CANNBOTDSL_PYTHON_EXECUTABLE}")
  endif()
  add_custom_target(ops_transformer_cannbotdsl_legacy
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_cannbotdsl_stage}/python"
    COMMAND ${CMAKE_COMMAND} -E env
      "TORCH_EXTENSION_OPS=${_cannbotdsl_names}"
      "TORCH_EXTENSION_VENDOR=${_cannbotdsl_vendor}"
      CANNBOTDSL_NPU_ARCH=dav-3510
      "${_cannbotdsl_legacy_python}" setup.py build_cannbotdsl
      --build-lib "${_cannbotdsl_stage}/python"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/torch_extension"
    VERBATIM)
  add_dependencies(ops_transformer_cannbotdsl
    ops_transformer_cannbotdsl_legacy)
  install(DIRECTORY "${_cannbotdsl_stage}/python/"
    DESTINATION "${_cannbotdsl_install}/python"
    COMPONENT ops-transformer
    PATTERN "__pycache__" EXCLUDE PATTERN "*.pyc" EXCLUDE)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/cannbotdsl_set_env.bash"
    DESTINATION "${_cannbotdsl_install}" RENAME set_env.bash
    COMPONENT ops-transformer)
endif()

if(CANNBOTDSL_BUNDLE_OPS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cannbotdsl_aclnn.cmake")
endif()
