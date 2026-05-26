cann_path="/usr/local/Ascend" # 更改cann包环境的路径
g++ "aclnn_dispatch_v3.cpp" -o dispatchDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                    -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi -lop_common -lpthread -lhccl