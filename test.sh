export ASCEND_GLOBAL_LOG_LEVEL=0 # 0为DEBUGE级别，1为INFO，2为WARNING，3为ERROR，4为关闭
export ASCEND_SLOG_PRINT_TO_STDOUT=1 # 1为输出，0为不输出

bash build.sh --ops=grouped_matmul --opapi -u
./build/tests/ut/framework_normal/op_api/transformer_op_api_ut --gtest_filter=*wq950_s8s4_bad_groupsize_64* | tee 1.log
