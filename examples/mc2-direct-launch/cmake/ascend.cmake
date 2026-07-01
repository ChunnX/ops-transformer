message(STATUS "Using Ascend toolkit path: ${ASCEND_HOME_PRE}")
set(CMAKE_PREFIX_PATH ${ASCEND_HOME_PRE}/)

set(BISHENG "${ASCEND_HOME_PRE}/${ARCH_PRE}-linux/ccec_compiler/bin/bisheng" CACHE FILEPATH "Path to Bisheng compiler")
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})

set(ASCEND_INCLUDE_DIRS
    ${ASCEND_HOME_PRE}/include
    ${ASCEND_HOME_PRE}/${ARCH_PRE}-linux/asc/include
    ${ASCEND_HOME_PRE}/${ARCH_PRE}-linux/ascendc/include
    ${ASCEND_HOME_PRE}/${ARCH_PRE}-linux/include
)
message(STATUS "ASCEND_INCLUDE_DIRS: ${ASCEND_INCLUDE_DIRS}")
