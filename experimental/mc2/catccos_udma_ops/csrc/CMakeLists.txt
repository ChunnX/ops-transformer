# Deliberately enumerate source files: an MTE build must not discover or
# instantiate a UDMA kernel by recursively scanning the source tree.
unset(CMAKE_CXX_FLAGS)
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})
set(CATCCOS_ASC_FLAGS "--npu-arch=dav-3510 -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=true -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false -xasc")

if(CATCCOS_BACKEND STREQUAL "mte")
    set(ASC_SOURCES
        mte/all_to_allv_grouped_mat_mul_mte.cpp
        mte/grouped_mat_mul_all_to_allv_mte.cpp
        mte/mte_peer_probe.cpp)
    set(HOST_SOURCES mte/mte_runtime.cpp mte/mte_runtime_bindings.cpp)
    set(BACKEND_OBJECT_TARGET catccos_mte_objects)
else()
    set(ASC_SOURCES
        all_to_allv_grouped_mat_mul_udma/ascend950/all_to_allv_grouped_mat_mul_udma.cpp
        all_to_allv_grouped_mat_mul_udma/ascend950/udma_runtime.cpp
        grouped_mat_mul_all_to_allv_udma/ascend950/grouped_mat_mul_all_to_allv_udma.cpp)
    set(HOST_SOURCES "")
    set(BACKEND_OBJECT_TARGET catccos_udma_objects)
    string(APPEND CATCCOS_ASC_FLAGS " ${CATCCOS_UDMA_ASCEND_FLAGS}")
endif()

set_source_files_properties(${ASC_SOURCES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "${CATCCOS_ASC_FLAGS}")
add_library(${BACKEND_OBJECT_TARGET} OBJECT ${ASC_SOURCES} ${HOST_SOURCES})
target_include_directories(${BACKEND_OBJECT_TARGET} PRIVATE
    ${INCLUDE_DIRECTORIES})
target_compile_options(${BACKEND_OBJECT_TARGET} PRIVATE ${COMPILE_OPTIONS})
if(CATCCOS_BACKEND STREQUAL "mte")
    target_include_directories(${BACKEND_OBJECT_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mte)
    target_compile_definitions(${BACKEND_OBJECT_TARGET} PRIVATE
        CATCCOS_MTE_WITH_TUNING=$<BOOL:${CATCCOS_MTE_WITH_TUNING}>)
else()
    target_include_directories(${BACKEND_OBJECT_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/all_to_allv_grouped_mat_mul_udma/ascend950)
endif()
set(OBJECTS_LIST $<TARGET_OBJECTS:${BACKEND_OBJECT_TARGET}> PARENT_SCOPE)
