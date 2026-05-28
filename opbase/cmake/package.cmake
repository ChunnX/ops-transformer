# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set_target_properties(ops_base PROPERTIES OUTPUT_NAME "ops_base")
#### CPACK to package run #####
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Detected architecture: x86_64")
    set(ARCH x86_64)
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    message(STATUS "Detected architecture: ARM64")
    set(ARCH aarch64)
else ()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()
# 打印路径
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")

set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/opbase/scripts)
install(DIRECTORY ${script_prefix}/
    DESTINATION share/info/opbase/script
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
)
set(SCRIPTS_FILES
    ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
    ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
    ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
)

install(FILES ${SCRIPTS_FILES}
    DESTINATION share/info/opbase/script
    COMPONENT opbase
)
set(COMMON_FILES
    ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
    ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
    ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
)

set(PACKAGE_FILES
    ${COMMON_FILES}
    ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
)
set(LATEST_MANGER_FILES
    ${COMMON_FILES}
    ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
    ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
    ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
)
set(CONF_FILES
    ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
    ${CMAKE_SOURCE_DIR}/src/nnopbase/common/op_info_record/dump_tool_config.ini
)
install(FILES ${CMAKE_BINARY_DIR}/version.opbase.info
    DESTINATION share/info/opbase
    RENAME version.info
    COMPONENT opbase
)
install(FILES ${CONF_FILES}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
    COMPONENT opbase
)
install(FILES ${PACKAGE_FILES}
    DESTINATION share/info/opbase/script
    COMPONENT opbase
)

set(pkg_inc_src ${CMAKE_SOURCE_DIR}/pkg_inc)
install(DIRECTORY ${pkg_inc_src}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
    PATTERN "aicpu_common" EXCLUDE
)
set(aicpu_common ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common)
install(DIRECTORY ${aicpu_common}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/aicpu_common
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)
install(TARGETS ops_base
    LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)
set(AICPU_LIBS
    aicpu_context
    aicpu_nodedef
    aicpu_context_host
    aicpu_nodedef_host
)

install(TARGETS ${AICPU_LIBS}
    ARCHIVE DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)

install(FILES ${PROTOBUF_STATIC_PKG_DIR}/lib/libbase_ascend_protobuf.a
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)
install(FILES ${PROTOBUF_HOST_STATIC_PKG_DIR}/lib/libhost_ascend_protobuf.a
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
    COMPONENT opbase
)
install(TARGETS aicpu_cust_log
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/device/lib64
    COMPONENT opbase
)

set(aicpu_headers_src 
    ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common/context/cpu_proto/cpu_attr_value.h
    ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common/context/cpu_proto/cpu_tensor_shape.h
    ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common/context/cpu_proto/cpu_tensor.h
    ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common/context/common/cpu_context.h
    ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common/context/common/cpu_types.h
    ${CMAKE_SOURCE_DIR}/pkg_inc/op_common/aicpu_common/context/cust_op/cust_cpu_utils.h
    )
install(FILES ${aicpu_headers_src}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aicpu
    COMPONENT opbase
)

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")
get_filename_component(COMPILER_PATH $ENV{ASCEND_HOME_PATH}/compiler REALPATH)
get_filename_component(VER_PATH "${COMPILER_PATH}" DIRECTORY)
message(STATUS "VERSION PATH: ${VER_PATH}")

set(aclnn_source ${CMAKE_SOURCE_DIR}/include/nnopbase/aclnn)
install(DIRECTORY ${aclnn_source}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)
set(opdev_source ${CMAKE_SOURCE_DIR}/include/nnopbase/opdev)
install(DIRECTORY ${opdev_source}/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn/opdev
    COMPONENT opbase
    FILE_PERMISSIONS
    OWNER_READ OWNER_WRITE
    GROUP_READ GROUP_EXECUTE
)

set(aclnnop_source ${CMAKE_SOURCE_DIR}/include/aclnnop)
install(DIRECTORY ${aclnnop_source}/
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop
        COMPONENT opbase
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
)
install(DIRECTORY ${aclnnop_source}/
        DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnnop/level2
        COMPONENT opbase
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ GROUP_EXECUTE
)

install(TARGETS nnopbase
   LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
   COMPONENT opbase
)

install(TARGETS dummy_tls
   LIBRARY DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/lib64
   COMPONENT opbase
)

install(TARGETS stub_nnopbase
    LIBRARY DESTINATION ${ARCH}-linux/devlib/linux/${ARCH}
    COMPONENT opbase
)

install(FILES ${CMAKE_BINARY_DIR}/version.opbase.info
    DESTINATION opp
    RENAME version.info
    COMPONENT opbase
)

set_cann_cpack_config(opbase SHARE_INFO_NAME opbase)