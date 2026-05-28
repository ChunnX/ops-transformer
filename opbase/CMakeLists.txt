cmake_minimum_required(VERSION 3.16)
include(cmake/fetch_cann_cmake.cmake)
project(opbase VERSION 1.0.0)
init_cann_project()
include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 17)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    message(STATUS "Build opbase with dependency cann pkgs installed")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build opbase with dependency cann pkgs installed" TRUE)
else()
    message(STATUS "Build opbase with cann source exist")
    option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build opbase need cann source code" FALSE)
endif()

include(cmake/config.cmake)
include(cmake/intf_pub_linux.cmake)
include(cmake/variables.cmake)
message(STATUS "Variables in ${CMAKE_PROJECT_NAME} project:")
cmake_print_variables(BUILD_WITH_DEPENDENCY_SOURCE)
cmake_print_variables(CANN_3RD_LIB_PATH)
cmake_print_variables(CANN_3RD_PKG_PATH)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    message(STATUS "ENV: $ENV{ASCEND_HOME_PATH}")
    if(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        if ("$ENV{USER}" STREQUAL "root")
            set(ASCEND_HOME_PATH /usr/local/Ascend/cann)
        else()
            set(ASCEND_HOME_PATH $ENV{HOME}/Ascend/cann)
        endif() 
    endif()

    if(DEFINED ENV{TOOLCHAIN_DIR})
        set(TOOLCHAIN_DIR $ENV{TOOLCHAIN_DIR})
    else()
        if(MINRC)
            set(TOOLCHAIN_DIR /usr)
        else()
            set(TOOLCHAIN_DIR ${ASCEND_HOME_PATH}/toolkit/toolchain/hcc)
        endif()
    endif()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/protobuf.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/eigen.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/nlohmann_json.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/func.cmake)

    include(version.cmake)
    set(ASCEND_INSTALL_PATH "${ASCEND_HOME_PATH}")
    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
        check_cann_pkg_build_deps("opbase")
    endif()
    add_cann_version_info_targets()

    include_directories(
        ${OPS_BASE_INCLUDE}
        ${OPS_BASE_INCLUDE}/op_common
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_HOME_PATH}/include/external
        ${ASCEND_HOME_PATH}/include/base
        ${ASCEND_HOME_PATH}/include/toolchain
        ${ASCEND_HOME_PATH}/pkg_inc
        ${ASCEND_HOME_PATH}/pkg_inc/aicpu
        ${ASCEND_HOME_PATH}/pkg_inc/base
        ${ASCEND_HOME_PATH}/pkg_inc/aicpu/aicpu_schedule
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched/common
    )

    add_library(ops_base SHARED ${OPS_BASE_SRC})
    target_compile_options(ops_base
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base
        PRIVATE
        ${ASCEND_HOME_PATH}/lib64/libexe_graph.so
        ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
        ${ASCEND_HOME_PATH}/lib64/libgraph.so
        -Wl,--no-as-needed
        ${ASCEND_HOME_PATH}/lib64/libregister.so
        -Wl,--as-needed
        -Wl,--whole-archive
        ${ASCEND_HOME_PATH}/lib64/libtiling_api.a
        -Wl,--no-whole-archive
        $<BUILD_INTERFACE:intf_pub_cxx17>
        ${ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    )

    add_subdirectory(aicpu_common/context)
    target_compile_definitions(ops_base PRIVATE
        BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG
    )
    if(DEFINED PRODUCT_SIDE AND "${PRODUCT_SIDE}" STREQUAL "device")
        message(STATUS ">>> Detected PRODUCT_SIDE=device, enabling device-specific features")
        target_compile_definitions(ops_base PRIVATE PRODUCT_SIDE_IS_DEVICE=1)
    endif()

    if(ENABLE_UT OR ENABLE_ST)
        add_subdirectory(tests)
    else()
        add_subdirectory(src/nnopbase)
        include(cmake/package.cmake)
    endif()
else()
    message(STATUS "Build ops_base_objs only from other source code")

    add_library(ops_base_infer_objs OBJECT ${OPS_BASE_INFER_SRC})
    add_library(ops_base_tiling_objs OBJECT ${OPS_BASE_TILING_SRC})
    add_library(ops_base_util_objs OBJECT ${OPS_BASE_UTIL_SRC})

    set(OPBASE_METADEF_INC
        ${METADEF_INCLUDE}
        ${METADEF_INCLUDE}/common
        ${METADEF_INCLUDE}/common/util
        ${METADEF_INCLUDE}/external
        ${METADEF_INCLUDE}/external/graph
        ${METADEF_INCLUDE}/external/exe_graph
        ${METADEF_INCLUDE}/exe_graph
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched/common
        ${TOP_DIR}/runtime/include/external
        )

    target_include_directories(ops_base_infer_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
        ${C_SEC_INCLUDE}
    )
    target_compile_options(ops_base_infer_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_infer_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog)

    target_include_directories(ops_base_tiling_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
        ${C_SEC_INCLUDE}
    )
    target_compile_options(ops_base_tiling_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_tiling_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog
        tiling_api)

    target_include_directories(ops_base_util_objs
        PRIVATE
        ${OPS_BASE_INCLUDE}
        ${OPBASE_METADEF_INC}
        ${ASCEND_DIR}/include/toolchain
        ${ASCEND_DIR}/pkg_inc/base
        ${C_SEC_INCLUDE}
    )
    target_compile_options(ops_base_util_objs
        PRIVATE
        -fPIC
        -fvisibility=hidden
    )
    target_link_libraries(ops_base_util_objs
        PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
        $<$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>:$<BUILD_INTERFACE:alog_headers>>
        $<$<NOT:$<BOOL:${BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG}>>:$<BUILD_INTERFACE:slog_headers>>
        alog
        tiling_api)
endif()