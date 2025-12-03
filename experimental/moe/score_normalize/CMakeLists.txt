message(STATUS "BUILD_TORCH_OPS ON in score_normalize")
# SCORE_NORMALIZE operation sources
file(GLOB SCORE_NORMALIZE_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(SCORE_NORMALIZE_SOURCES ${SCORE_NORMALIZE_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${SCORE_NORMALIZE_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(score_normalize_objects OBJECT ${SCORE_NORMALIZE_SOURCES})

target_compile_options(score_normalize_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(score_normalize_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()