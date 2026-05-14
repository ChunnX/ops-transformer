E=4
M=40000
K=512
N=2048
type="fp16"
dataform="nz"
quantGroupSize=512
G=0
G_num=0     # 有G_num项，每项是G
tuningNum=$((M / E))
deviceId=7
withOffset=false
fileFlag=FILE     #NOFILE情况下，不运行标杆直接跑算子，多用于profiling

code_path=/data/huangjunzhe/gmm/ops-transformer
ascend_path=/data/huangjunzhe/Ascend.GDN/cann-9.0.0
export TMPDIR=/data/huangjunzhe/tmp
test_path=$(pwd)
#------------------------------------------------------------------------------------------------

export A8W4_DATA_PATH=${test_path}/data
source ${ascend_path}/set_env.sh

alias log='export ASCEND_SLOG_PRINT_TO_STDOUT=1; export ASCEND_GLOBAL_LOG_LEVEL=0'
alias unlog='unset ASCEND_SLOG_PRINT_TO_STDOUT; unset ASCEND_GLOBAL_LOG_LEVEL'

if [ "$1" = "clean" ]; then
    # compile gmm code
    unset ASCEND_SLOG_PRINT_TO_STDOUT; unset ASCEND_GLOBAL_LOG_LEVEL
    rm -rf log
    rm -rf log_multi
    rm -rf log_single
    rm -rf test_gmms
    rm -rf test_gmms_multi
    rm -rf test_gmms_all
    rm -rf data/*
    exit 0
fi

if [ "$1" = "compile" ]; then
    # compile gmm code
    unset ASCEND_SLOG_PRINT_TO_STDOUT; unset ASCEND_GLOBAL_LOG_LEVEL
    cd ${code_path}
    bash build.sh --pkg --ops=grouped_matmul # --soc=ascend910_93
    [ $? -ne 0 ] && exit 1
    bash ${code_path}/build/cann-ops-transformer-custom_linux-aarch64.run
    [ $? -ne 0 ] && exit 1
fi
echo "E=${E}, M=${M}, K=${K}, N=${N}, type=${type}, G=${G}, G_num=${G_num}, quantGroupSize=${quantGroupSize}"
export LD_LIBRARY_PATH=${ascend_path}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
cd ${test_path}

echo "===========Starting MSD.py==========="
python3 gmm_a8w4_gendata.py ${E} ${M} ${K} ${N} ${type} ${G} ${G_num} ${quantGroupSize} ${tuningNum} ${withOffset}

echo "===========Compiling & running NPU op==========="
g++ -std=c++17 test_gmms_all.cpp -L${ascend_path}/lib64 -D${fileFlag} -lascendcl -lnnopbase -L${ascend_path}/opp/vendors/custom_transformer/op_api/lib -lcust_opapi -I${ascend_path}/include/ -o test_gmms_all
[ $? -ne 0 ] && exit 1
cmd="./test_gmms_all ${E} ${M} ${K} ${N} ${type} ${dataform} ${quantGroupSize} ${tuningNum} ${deviceId} ${withOffset}"
if [ "$1" = "log" ]; then
    # run with logs
    export ASCEND_SLOG_PRINT_TO_STDOUT=1; export ASCEND_GLOBAL_LOG_LEVEL=0
    $cmd multi > log_multi
    $cmd single > log_single
elif [ "$1" = "prof" ]; then
    msprof $cmd multi
    msprof $cmd single
else
    $cmd multi
    $cmd single
fi

echo "===========Comparing results with golden==========="
[ "$type" = "fp16" ] && dtype="np.float16" || dtype="bfloat16"
python -c "import numpy as np; from bfloat16 import bfloat16; print('a8w4_nz_test_golden: ', np.fromfile('${A8W4_DATA_PATH}/a8w4_nz_test_golden.bin', dtype=${dtype}))"
python -c "import numpy as np; from bfloat16 import bfloat16; print('a8w4_nz_test_result_multi: ', np.fromfile('${A8W4_DATA_PATH}/a8w4_nz_test_result_multi.bin', dtype=${dtype}))"
python -c "import numpy as np; from bfloat16 import bfloat16; print('a8w4_nz_test_result: ', np.fromfile('${A8W4_DATA_PATH}/a8w4_nz_test_result.bin', dtype=${dtype}))"
# python -c "import numpy as np; print('multi - golden: ', (np.fromfile('${A8W4_DATA_PATH}/a8w4_nz_test_result_multi.bin', dtype=${dtype}) - np.fromfile('${A8W4_DATA_PATH}/a8w4_nz_test_golden.bin', dtype=${dtype})).reshape($M,$N)[17])"
python -c "import numpy as np; a,b=[np.fromfile(f'${A8W4_DATA_PATH}/a8w4_nz_test_{name}.bin',dtype=np.uint16) for name in ['result_multi','golden']]; min_len=min(len(a),len(b)); mismatch=np.sum(a[:min_len]!=b[:min_len]); print(f'[二进制对比] Golden与Multi匹配长度: {min_len}, 错误数: {mismatch}, 准确率: {((min_len-mismatch)/min_len*100):.4f}%' if min_len>0 else '文件为空')"
python -c "import numpy as np; a,b=[np.fromfile(f'${A8W4_DATA_PATH}/a8w4_nz_test_{name}.bin',dtype=np.uint16) for name in ['result','golden']]; min_len=min(len(a),len(b)); mismatch=np.sum(a[:min_len]!=b[:min_len]); print(f'[二进制对比] Golden与Single匹配长度: {min_len}, 错误数: {mismatch}, 准确率: {((min_len-mismatch)/min_len*100):.4f}%' if min_len>0 else '文件为空')"
md5sum ${A8W4_DATA_PATH}/a8w4_nz_test_golden.bin
md5sum ${A8W4_DATA_PATH}/a8w4_nz_test_result*.bin
