# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# FlashKDA v2 OperatorBundle -> framework ACLNN stage -> built-in DSO.
set(CANNBOTDSL_SOURCE_DIR "" CACHE PATH
  "CANNBotDSL checkout containing framework/cmake")
set(CANNBOTDSL_PYTHONPATH "" CACHE STRING
  "Python path containing the v2 cannbotdsl package and cannbot_framework")
set(CANNBOTDSL_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
  "Python interpreter used to build the frozen FlashKDA bundle")
set(CANNBOTDSL_NNOPBASE_INCLUDE_DIR "" CACHE PATH
  "Patched nnopbase include directory containing opdev/external_op_task.h")
set(CANNBOTDSL_NNOPBASE_LIBRARY "" CACHE FILEPATH
  "Patched libnnopbase.so exporting op::AddExternalOpTask")
foreach(_required CANNBOTDSL_SOURCE_DIR CANNBOTDSL_PYTHONPATH
    CANNBOTDSL_PYTHON_EXECUTABLE CANNBOTDSL_NNOPBASE_INCLUDE_DIR
    CANNBOTDSL_NNOPBASE_LIBRARY)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required for built-in FlashKDA ACLNN")
  endif()
endforeach()
if(NOT EXISTS
    "${CANNBOTDSL_SOURCE_DIR}/framework/cmake/CANNBotFrameworkOperatorPackage.cmake")
  message(FATAL_ERROR "CANNBOTDSL_SOURCE_DIR has no ACLNN framework CMake package")
endif()
if(NOT EXISTS
    "${CANNBOTDSL_NNOPBASE_INCLUDE_DIR}/opdev/external_op_task.h" OR
   NOT EXISTS "${CANNBOTDSL_NNOPBASE_LIBRARY}")
  message(FATAL_ERROR
    "FlashKDA ACLNN requires matching External Task nnopbase headers and library")
endif()

# The framework stage must exist before CMake can declare its generated object
# library. Reuse it across reconfigure when the source/toolchain inputs match.
set(_flash_kda_stage "${CMAKE_CURRENT_BINARY_DIR}/cannbotdsl/FlashKda.stage")
set(_flash_kda_stamp "${_flash_kda_stage}.input")
set(_flash_kda_source
  "${CMAKE_CURRENT_SOURCE_DIR}/attention/flash_kda/op_kernel/flash_kda.py")
set(_flash_kda_builder
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cannbotdsl_aclnn.py")
file(SHA256 "${_flash_kda_source}" _flash_kda_source_hash)
file(SHA256 "${_flash_kda_builder}" _flash_kda_builder_hash)
string(SHA256 _flash_kda_input
  "${_flash_kda_source_hash};${_flash_kda_builder_hash};${CANNBOTDSL_SOURCE_DIR};${CANNBOTDSL_PYTHONPATH};${CMAKE_SYSTEM_PROCESSOR}")
set(_rebuild_flash_kda TRUE)
if(EXISTS "${_flash_kda_stamp}" AND EXISTS "${_flash_kda_stage}/stage.json")
  file(READ "${_flash_kda_stamp}" _flash_kda_existing_input)
  if(_flash_kda_existing_input STREQUAL _flash_kda_input)
    set(_rebuild_flash_kda FALSE)
  endif()
endif()
if(_rebuild_flash_kda)
  file(REMOVE_RECURSE "${_flash_kda_stage}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${CANNBOTDSL_PYTHONPATH}:$ENV{PYTHONPATH}"
      TORCH_DEVICE_BACKEND_AUTOLOAD=0
      PYTHONDONTWRITEBYTECODE=1
      "${CANNBOTDSL_PYTHON_EXECUTABLE}" "${_flash_kda_builder}"
      --source "${_flash_kda_source}"
      --parent-opp "${CMAKE_CURRENT_SOURCE_DIR}"
      --output "${_flash_kda_stage}"
      --processor "${CMAKE_SYSTEM_PROCESSOR}"
    RESULT_VARIABLE _flash_kda_result
    OUTPUT_VARIABLE _flash_kda_stdout
    ERROR_VARIABLE _flash_kda_stderr)
  if(NOT _flash_kda_result EQUAL 0)
    message(FATAL_ERROR
      "FlashKDA frozen bundle generation failed (${_flash_kda_result})\n"
      "stdout:\n${_flash_kda_stdout}\nstderr:\n${_flash_kda_stderr}")
  endif()
  file(WRITE "${_flash_kda_stamp}" "${_flash_kda_input}")
endif()

# Called after custom_build.cmake has resolved nnopbase. Keep the generated
# implementation in the framework's canonical op_api/lib layout so its
# DSO-relative Provider lookup remains unchanged. This target is sufficient
# for a FlashKDA-only package and is reused by a mixed package.
function(cannbotdsl_add_aclnn_library)
  if(TARGET cannbotdsl_flash_kda_aclnn)
    return()
  endif()
  if(NOT TARGET nnopbase)
    add_library(nnopbase SHARED IMPORTED GLOBAL)
  endif()
  get_target_property(_nnopbase_imported nnopbase IMPORTED)
  if(NOT _nnopbase_imported)
    message(FATAL_ERROR
      "FlashKDA ACLNN requires an imported nnopbase target so the patched ABI can be selected")
  endif()
  set_target_properties(nnopbase PROPERTIES
    IMPORTED_LOCATION "${CANNBOTDSL_NNOPBASE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${CANNBOTDSL_NNOPBASE_INCLUDE_DIR};${ASCEND_CANN_PACKAGE_PATH}/include;${ASCEND_CANN_PACKAGE_PATH}/include/aclnn;${ASCEND_CANN_PACKAGE_PATH}/include/experiment")
  set(CANNBotDSL_NPU_ARCH dav-3510)
  include(
    "${CANNBOTDSL_SOURCE_DIR}/framework/cmake/CANNBotFrameworkOperatorPackage.cmake")
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ops-transformer)
  cannbot_framework_add_frozen_bundle(
    cannbotdsl_flash_kda_bundle DIRECTORY "${_flash_kda_stage}")
  add_library(cannbotdsl_flash_kda_aclnn SHARED)
  target_link_libraries(cannbotdsl_flash_kda_aclnn PRIVATE
    cannbotdsl_flash_kda_bundle)
  target_link_options(cannbotdsl_flash_kda_aclnn PRIVATE
    "LINKER:--require-defined=aclnnFlashKda"
    "LINKER:--require-defined=aclnnFlashKdaGetWorkspaceSize"
    "LINKER:-soname,$ORIGIN/../../opp/built-in/op_api/lib/libopapi_cannbot_transformer.so")
  set_target_properties(cannbotdsl_flash_kda_aclnn PROPERTIES
    OUTPUT_NAME opapi_cannbot_transformer
    NO_SONAME TRUE)
  install(TARGETS cannbotdsl_flash_kda_aclnn
    LIBRARY DESTINATION "opp/built-in/op_api/lib"
    COMPONENT ops-transformer)
endfunction()

# gen_opapi_symbol calls this for mixed packages. Retaining the helper as a
# DT_NEEDED dependency makes aclnnFlashKda* visible through dlsym on the
# public libopapi_transformer handle as well.
function(cannbotdsl_attach_aclnn target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "FlashKDA ACLNN requires an opapi target")
  endif()
  cannbotdsl_add_aclnn_library()
  target_link_libraries("${target}" PRIVATE
    "-Wl,--no-as-needed" cannbotdsl_flash_kda_aclnn "-Wl,--as-needed")
endfunction()
