add_ops_compile_options(
        OP_NAME GroupedBiasAddGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
op_host/grouped_bias_add_grad.cpp
)

target_sources(optiling PRIVATE
        op_host/grouped_bias_add_grad.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
         op_host/grouped_bias_add_grad.cpp
)

install(FILES op_kernel/grouped_bias_add_grad.cpp
        DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)