# Install script for directory: /home/developer/codes/ops-transformer

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/developer/codes/ops-transformer/build_out")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "--version")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/lib" TYPE SHARED_LIBRARY FILES "/home/developer/codes/ops-transformer/build/prepare_build/libcust_opapi.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/lib/libcust_opapi.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/lib/libcust_opapi.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/lib/libcust_opapi.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/lib/linux/aarch64" TYPE SHARED_LIBRARY FILES "/home/developer/codes/ops-transformer/build/prepare_build/libcust_opsproto_rt2.0.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64" TYPE SHARED_LIBRARY FILES "/home/developer/codes/ops-transformer/build/prepare_build/libcust_opmaster_rt2.0.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling" TYPE FILE FILES "/home/developer/codes/ops-transformer/build/prepare_build/compat/liboptiling.so")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/common/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mc2/all_gather_matmul_v2/op_host/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mc2/all_gather_matmul_v2/op_graph/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mc2/all_gather_matmul_v3/op_host/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/attention/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mhc/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mamba/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mc2/common/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/developer/codes/ops-transformer/build/prepare_build/mc2/3rd/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/inc" TYPE FILE OPTIONAL FILES
    "/home/developer/codes/ops-transformer/build/autogen/inner/all_gather_matmul_v2_proto.h"
    "/home/developer/codes/ops-transformer/build/autogen/inner/all_gather_matmul_v3_proto.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/es/include" TYPE DIRECTORY OPTIONAL FILES "/home/developer/codes/ops-transformer/build/prepare_build/es_packages/include/es_transformer_cust/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_proto/es/lib/linux/aarch64" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/prepare_build/es_packages/lib64/libes_transformer_cust.so")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v2.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v2_a2.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v2_a3.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v2_apt.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v3.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v3_a2.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v3_a3.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/dynamic" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/impl/dynamic/all_gather_matmul_v3_apt.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/common/include/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/gmm/common/cgmct")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/catlass/include" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/catlass/include")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/ops-tensor/include/blaze")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/tensor_api/impl" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/ops-tensor/include/tensor_api/impl/tensor_api")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/tensor_api/include" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/ops-tensor/include/tensor_api/include/tensor_api")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/tensor_api/impl" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/ops-tensor/include/tensor_api/impl/c_api")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common/tensor_api/include" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/ops-tensor/include/tensor_api/include/c_api")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/common/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/batch_mat_mul_v3" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/batch_mat_mul_v3/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/grouped_matmul" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/grouped_matmul/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/mat_mul_v3" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/mat_mul_v3/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/norm_common" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/norm_common/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/quant_batch_matmul_v3" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/quant_batch_matmul_v3/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/rms_norm" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/rms_norm/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/template_linear_algebra" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/template_linear_algebra/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd/weight_quant_batch_matmul_v2" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/mc2/3rd/weight_quant_batch_matmul_v2/op_kernel/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY FILES "/home/developer/codes/ops-transformer/third_party/opbase/include/op_common/log")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY OPTIONAL FILES "/home/developer/codes/ops-transformer/third_party/opbase/pkg_inc/op_common/atvoss")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/common" TYPE DIRECTORY OPTIONAL FILES "/home/developer/codes/ops-transformer/third_party/opbase/include/op_common/op_kernel")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/all_gather_matmul_v2" TYPE FILE OPTIONAL FILES
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/agm_v2_common.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/agm_v2_copy_gm_to_l1.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode_block_epilogue_dequant.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode_dequant.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode_padding.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode_tiling.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_aiv_mode_util.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_tiling_data.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_v2_apt_tiling_key.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/all_gather_matmul_v2_tiling_key.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/block_mmad_preload_fixpipe.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/copy_l0c_to_gm.h"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/gather_matmul.hpp"
    "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/gather_moe_distribute_base.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/all_gather_matmul_v2" TYPE DIRECTORY OPTIONAL FILES "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_kernel/arch35")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd" TYPE DIRECTORY OPTIONAL FILES "")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/all_gather_matmul_v3" TYPE DIRECTORY OPTIONAL FILES "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v3/op_kernel/arch35")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/3rd" TYPE DIRECTORY OPTIONAL FILES "")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_impl/ai_core/tbe/config/ascend950" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/autogen/aic-ascend950-ops-info.json")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE DIRECTORY PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ FILES "/home/developer/codes/ops-transformer/build/prepare_build/scripts/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer" TYPE FILE FILES "/home/developer/codes/ops-transformer/build/prepare_build/version.info")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/include/aclnnop" TYPE FILE OPTIONAL FILES
    "/home/developer/codes/ops-transformer/build/prepare_build/autogen/aclnn_all_gather_matmul_v2.h"
    "/home/developer/codes/ops-transformer/build/prepare_build/autogen/aclnn_all_gather_matmul_v3.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/include/aclnnop" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/build/prepare_build/aclnn_ops_transformer_custom.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/packages/vendors/custom_transformer/op_api/include/aclnnop" TYPE FILE OPTIONAL FILES "/home/developer/codes/ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.h")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/developer/codes/ops-transformer/build/prepare_build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
