# find torch_npu
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
set(TORCH_NPU_INCLUDE_PATH "${TORCH_NPU_PATH}/include")
set(TORCH_NPU_LIB_PATH "${TORCH_NPU_PATH}/lib")
set(TORCH_NPU_LIBRARIES "${TORCH_NPU_LIB_PATH}/libtorch_npu.so")
message(STATUS "Torch NPU path: ${TORCH_NPU_PATH}")
message(STATUS "Torch NPU include: ${TORCH_NPU_INCLUDE_PATH}")
