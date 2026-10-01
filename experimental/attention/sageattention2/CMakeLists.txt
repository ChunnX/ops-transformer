# ==============================================================================
# SageAttention2 实验算子构建入口：op_host（含算子 def / tiling / op_api 挂接）。
# kernel 侧编译选项见 op_host/CMakeLists.txt 的 SAGE_HOST_COMPILE_OPTIONS。
# ==============================================================================
add_subdirectory(op_host)
