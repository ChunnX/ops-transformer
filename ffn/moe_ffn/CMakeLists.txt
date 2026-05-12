add_ops_compile_options(
        OP_NAME MoeFFN
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        op_host/moe_ffn.cpp
)       

target_sources(optiling PRIVATE
        op_host/moe_ffn.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)