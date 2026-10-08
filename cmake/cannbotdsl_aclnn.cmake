# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Consume only installed cannbot-dsl/opkit wheels; no DSL source checkout.
set(CANNBOTDSL_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python with cannbot-dsl and opkit wheels")
set(CANNBOTDSL_NNOPBASE_INCLUDE_DIR "" CACHE PATH "Matching DirectInvoke nnopbase headers")
set(CANNBOTDSL_NNOPBASE_LIBRARY "" CACHE FILEPATH "Matching DirectInvoke libnnopbase.so")
set(CANNBOTDSL_NPU_ARCH "dav-3510" CACHE STRING "Native provider target")
if(NOT EXISTS "${CANNBOTDSL_NNOPBASE_INCLUDE_DIR}/opdev/direct_invoke_task.h"
   OR NOT EXISTS "${CANNBOTDSL_NNOPBASE_LIBRARY}")
  message(FATAL_ERROR "OpKit requires matching DirectInvoke nnopbase headers and library")
endif()
execute_process(COMMAND "${CANNBOTDSL_PYTHON_EXECUTABLE}" -c
  "import hashlib,importlib.metadata as m; print(hashlib.sha256(''.join(m.distribution(n).read_text('RECORD') for n in ('cannbot-dsl','opkit')).encode()).hexdigest())"
  RESULT_VARIABLE _wheel_result OUTPUT_VARIABLE _wheel_hash ERROR_VARIABLE _wheel_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _wheel_result EQUAL 0)
  message(FATAL_ERROR "Install cannbot-dsl and opkit wheels in CANNBOTDSL_PYTHON_EXECUTABLE: ${_wheel_error}")
endif()
set(_builder "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cannbotdsl.py")
file(SHA256 "${_builder}" _builder_hash)
set(CANNBOTDSL_ACLNN_STAGES)
foreach(_source IN LISTS CANNBOTDSL_SOURCES)
  get_filename_component(_op "${_source}" NAME_WE)
  set(_stage "${CMAKE_CURRENT_BINARY_DIR}/cannbotdsl/${_op}.stage")
  set(_stamp "${_stage}.input")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_source}" "${_builder}")
  file(SHA256 "${_source}" _source_hash)
  string(SHA256 _input_hash "${_source_hash};${_builder_hash};${_wheel_hash};${CANNBOTDSL_PYTHON_EXECUTABLE};${CANNBOTDSL_NPU_ARCH};${ASCEND_CANN_PACKAGE_PATH};${CMAKE_SYSTEM_PROCESSOR}")
  set(_existing_hash "")
  if(EXISTS "${_stamp}" AND EXISTS "${_stage}/package.json")
    file(READ "${_stamp}" _existing_hash)
  endif()
  if(NOT _existing_hash STREQUAL _input_hash)
    file(REMOVE_RECURSE "${_stage}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
      "ASCEND_HOME_PATH=${ASCEND_CANN_PACKAGE_PATH}" TORCH_DEVICE_BACKEND_AUTOLOAD=0
      PYTHONDONTWRITEBYTECODE=1 "${CANNBOTDSL_PYTHON_EXECUTABLE}" "${_builder}"
      --source "${_source}" --output "${_stage}" --npu-arch "${CANNBOTDSL_NPU_ARCH}"
      RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR "OpKit generation failed for ${_op}: ${_stdout}\n${_stderr}")
    endif()
    file(WRITE "${_stamp}" "${_input_hash}")
  endif()
  list(APPEND CANNBOTDSL_ACLNN_STAGES "${_stage}")
endforeach()

set(CANN_ROOT "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "CANN SDK for OpKit" FORCE)
set(OPBASE_INCLUDE_DIR "${CANNBOTDSL_NNOPBASE_INCLUDE_DIR}" CACHE PATH "DirectInvoke headers for OpKit" FORCE)
set(OPBASE_LIBRARY "${CANNBOTDSL_NNOPBASE_LIBRARY}" CACHE STRING "DirectInvoke library for OpKit" FORCE)
list(GET CANNBOTDSL_ACLNN_STAGES 0 _first_stage)
add_subdirectory("${_first_stage}/common/hostapi" "cannbotdsl/hostapi")
add_library(cannbotdsl_aclnn SHARED)
foreach(_stage IN LISTS CANNBOTDSL_ACLNN_STAGES)
  file(READ "${_stage}/package.json" _package)
  string(JSON _count LENGTH "${_package}" operators)
  math(EXPR _last "${_count} - 1")
  foreach(_index RANGE ${_last})
    string(JSON _operator GET "${_package}" operators ${_index})
    string(JSON _module GET "${_package}" modules "${_operator}")
    if(TARGET "operator_${_operator}")
      message(FATAL_ERROR "Duplicate OpKit operator ${_operator}")
    endif()
    add_subdirectory("${_stage}/${_module}" "cannbotdsl/${_operator}")
    target_link_libraries(cannbotdsl_aclnn PRIVATE "operator_${_operator}")
    get_filename_component(_header "${_module}" NAME)
    install(FILES "${_stage}/${_module}/op_api/aclnn_op.h"
      DESTINATION "${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop"
      RENAME "aclnn_${_header}.h" COMPONENT ops-transformer)
  endforeach()
  # Provider plans already contain paths relative to op_impl/cannbotdsl.
  install(DIRECTORY "${_stage}/op_impl/cannbotdsl/"
    DESTINATION "opp/built-in/op_impl/cannbotdsl" COMPONENT ops-transformer)
endforeach()
target_link_options(cannbotdsl_aclnn PRIVATE
  "LINKER:-soname,$ORIGIN/../../opp/built-in/op_api/lib/libopapi_cannbot_transformer.so")
set_target_properties(cannbotdsl_aclnn PROPERTIES OUTPUT_NAME opapi_cannbot_transformer NO_SONAME TRUE)
install(TARGETS cannbotdsl_aclnn LIBRARY DESTINATION "opp/built-in/op_api/lib" COMPONENT ops-transformer)
