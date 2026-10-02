# CMakeLists.txt — SageAttention2 宿主仿真模块构建配置（仅在 ENABLE_TEST / 宿主单测时构建）

add_library(sa2_sim INTERFACE)
target_include_directories(sa2_sim INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/../op_host
)
