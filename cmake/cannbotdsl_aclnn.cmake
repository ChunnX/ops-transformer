# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Generic CANNBotDSL OperatorBundle -> framework ACLNN -> built-in DSO bridge.
set(CANNBOTDSL_SOURCE_DIR "" CACHE PATH
  "CANNBotDSL checkout containing framework/cmake")
set(CANNBOTDSL_PYTHONPATH "" CACHE STRING
  "Python path containing cannbotdsl and cannbot_framework")
set(CANNBOTDSL_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH
  "Python interpreter used to build frozen CANNBotDSL bundles")
set(CANNBOTDSL_NNOPBASE_INCLUDE_DIR "" CACHE PATH
  "Patched nnopbase include directory containing opdev/external_op_task.h")
set(CANNBOTDSL_NNOPBASE_LIBRARY "" CACHE FILEPATH
  "Patched libnnopbase.so exporting op::AddExternalOpTask")
set(CANNBOTDSL_NPU_ARCH "dav-3510" CACHE STRING
  "CANNBotDSL target architecture for selected frozen bundles")
set(CANNBOTDSL_AIC_COUNT 24 CACHE STRING "CANNBotDSL target AIC count")
set(CANNBOTDSL_AIV_COUNT 48 CACHE STRING "CANNBotDSL target AIV count")
set(CANNBOTDSL_BARRIER_LIMIT 16 CACHE STRING
  "CANNBotDSL target grid-barrier limit")
set(CANNBOTDSL_UB_BYTES 262144 CACHE STRING "CANNBotDSL target UB bytes")

foreach(_required CANNBOTDSL_SOURCE_DIR CANNBOTDSL_PYTHONPATH
    CANNBOTDSL_PYTHON_EXECUTABLE CANNBOTDSL_NNOPBASE_INCLUDE_DIR
    CANNBOTDSL_NNOPBASE_LIBRARY CANNBOTDSL_NPU_ARCH)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required for frozen CANNBotDSL ACLNN bundles")
  endif()
endforeach()
if(NOT EXISTS
    "${CANNBOTDSL_SOURCE_DIR}/framework/cmake/CANNBotFrameworkOperatorPackage.cmake")
  message(FATAL_ERROR "CANNBOTDSL_SOURCE_DIR has no framework CMake package")
endif()
if(NOT EXISTS
    "${CANNBOTDSL_NNOPBASE_INCLUDE_DIR}/opdev/external_op_task.h" OR
   NOT EXISTS "${CANNBOTDSL_NNOPBASE_LIBRARY}")
  message(FATAL_ERROR
    "Frozen CANNBotDSL ACLNN bundles require matching External Task nnopbase headers and library")
endif()

set(_cannbotdsl_builder
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cannbotdsl_bundle.py")
file(SHA256 "${_cannbotdsl_builder}" _cannbotdsl_builder_hash)
set(CANNBOTDSL_ACLNN_STAGES)
set(CANNBOTDSL_ACLNN_OPERATORS)
list(LENGTH CANNBOTDSL_BUNDLE_OPS _cannbotdsl_op_count)
list(LENGTH CANNBOTDSL_BUNDLE_SOURCES _cannbotdsl_source_count)
if(NOT _cannbotdsl_op_count EQUAL _cannbotdsl_source_count)
  message(FATAL_ERROR "CANNBotDSL bundle operator/source lists differ")
endif()

if(_cannbotdsl_op_count GREATER 0)
  math(EXPR _cannbotdsl_last "${_cannbotdsl_op_count} - 1")
  foreach(_index RANGE ${_cannbotdsl_last})
    list(GET CANNBOTDSL_BUNDLE_OPS ${_index} _op)
    list(GET CANNBOTDSL_BUNDLE_SOURCES ${_index} _source)
    set(_stage "${CMAKE_CURRENT_BINARY_DIR}/cannbotdsl/${_op}.stage")
    set(_stamp "${_stage}.input")
    file(SHA256 "${_source}" _source_hash)
    string(SHA256 _input_hash
      "${_source_hash};${_cannbotdsl_builder_hash};${CANNBOTDSL_SOURCE_DIR};${CANNBOTDSL_PYTHONPATH};${CANNBOTDSL_NPU_ARCH};${CANNBOTDSL_AIC_COUNT};${CANNBOTDSL_AIV_COUNT};${CANNBOTDSL_BARRIER_LIMIT};${CANNBOTDSL_UB_BYTES};${CMAKE_SYSTEM_PROCESSOR}")
    set(_rebuild TRUE)
    if(EXISTS "${_stamp}" AND EXISTS "${_stage}/stage.json")
      file(READ "${_stamp}" _existing_hash)
      if(_existing_hash STREQUAL _input_hash)
        set(_rebuild FALSE)
      endif()
    endif()
    if(_rebuild)
      file(REMOVE_RECURSE "${_stage}")
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=${CANNBOTDSL_PYTHONPATH}:$ENV{PYTHONPATH}"
          TORCH_DEVICE_BACKEND_AUTOLOAD=0
          PYTHONDONTWRITEBYTECODE=1
          "${CANNBOTDSL_PYTHON_EXECUTABLE}" "${_cannbotdsl_builder}"
          --source "${_source}"
          --parent-opp "${CMAKE_CURRENT_SOURCE_DIR}"
          --output "${_stage}"
          --processor "${CMAKE_SYSTEM_PROCESSOR}"
          --npu-arch "${CANNBOTDSL_NPU_ARCH}"
          --aic-count "${CANNBOTDSL_AIC_COUNT}"
          --aiv-count "${CANNBOTDSL_AIV_COUNT}"
          --barrier-limit "${CANNBOTDSL_BARRIER_LIMIT}"
          --ub-bytes "${CANNBOTDSL_UB_BYTES}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr)
      if(NOT _result EQUAL 0)
        message(FATAL_ERROR
          "CANNBotDSL bundle generation failed for ${_op} (${_result})\n"
          "stdout:\n${_stdout}\nstderr:\n${_stderr}")
      endif()
      file(WRITE "${_stamp}" "${_input_hash}")
    endif()
    file(READ "${_stage}/stage.json" _manifest)
    string(JSON _operator GET "${_manifest}" operator)
    if(_operator IN_LIST CANNBOTDSL_ACLNN_OPERATORS)
      message(FATAL_ERROR "Duplicate CANNBotDSL ACLNN operator: ${_operator}")
    endif()
    list(APPEND CANNBOTDSL_ACLNN_STAGES "${_stage}")
    list(APPEND CANNBOTDSL_ACLNN_OPERATORS "${_operator}")
  endforeach()
endif()

# Called after custom_build.cmake resolves nnopbase. All selected frozen
# bundles share one product-owned helper DSO, for both bundle-only and mixed
# operator packages.
function(cannbotdsl_add_aclnn_library)
  if(TARGET cannbotdsl_aclnn)
    return()
  endif()
  if(NOT TARGET nnopbase)
    add_library(nnopbase SHARED IMPORTED GLOBAL)
  endif()
  get_target_property(_nnopbase_imported nnopbase IMPORTED)
  if(NOT _nnopbase_imported)
    message(FATAL_ERROR
      "Frozen CANNBotDSL ACLNN bundles require an imported nnopbase target")
  endif()
  set_target_properties(nnopbase PROPERTIES
    IMPORTED_LOCATION "${CANNBOTDSL_NNOPBASE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES
      "${CANNBOTDSL_NNOPBASE_INCLUDE_DIR};${ASCEND_CANN_PACKAGE_PATH}/include;${ASCEND_CANN_PACKAGE_PATH}/include/aclnn;${ASCEND_CANN_PACKAGE_PATH}/include/experiment")
  set(CANNBotDSL_NPU_ARCH "${CANNBOTDSL_NPU_ARCH}")
  include(
    "${CANNBOTDSL_SOURCE_DIR}/framework/cmake/CANNBotFrameworkOperatorPackage.cmake")
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ops-transformer)
  add_library(cannbotdsl_aclnn SHARED)
  list(LENGTH CANNBOTDSL_ACLNN_STAGES _stage_count)
  math(EXPR _last_stage "${_stage_count} - 1")
  foreach(_index RANGE ${_last_stage})
    list(GET CANNBOTDSL_ACLNN_STAGES ${_index} _stage)
    list(GET CANNBOTDSL_ACLNN_OPERATORS ${_index} _operator)
    string(MAKE_C_IDENTIFIER "${_operator}" _operator_id)
    set(_bundle_target "cannbotdsl_${_operator_id}_bundle")
    cannbot_framework_add_frozen_bundle(
      "${_bundle_target}" DIRECTORY "${_stage}")
    target_link_libraries(cannbotdsl_aclnn PRIVATE "${_bundle_target}")
    target_link_options(cannbotdsl_aclnn PRIVATE
      "LINKER:--require-defined=aclnn${_operator}"
      "LINKER:--require-defined=aclnn${_operator}GetWorkspaceSize")
  endforeach()
  target_link_options(cannbotdsl_aclnn PRIVATE
    "LINKER:-soname,$ORIGIN/../../opp/built-in/op_api/lib/libopapi_cannbot_transformer.so")
  set_target_properties(cannbotdsl_aclnn PROPERTIES
    OUTPUT_NAME opapi_cannbot_transformer
    NO_SONAME TRUE)
  install(TARGETS cannbotdsl_aclnn
    LIBRARY DESTINATION "opp/built-in/op_api/lib"
    COMPONENT ops-transformer)
endfunction()

function(cannbotdsl_attach_aclnn target)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR "Cannot attach CANNBotDSL ACLNN bundles to ${target}")
  endif()
  cannbotdsl_add_aclnn_library()
  target_link_libraries("${target}" PRIVATE
    "-Wl,--no-as-needed" cannbotdsl_aclnn "-Wl,--as-needed")
endfunction()
