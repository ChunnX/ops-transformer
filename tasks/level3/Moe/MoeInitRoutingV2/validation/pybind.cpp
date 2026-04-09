#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> custom_pybind_api(
    const at::Tensor& x,
    const at::Tensor& expertIdx,
    int64_t activeNum,
    int64_t expertCapacity,
    int64_t expertNum,
    int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag,
    bool expertTokensBeforeCapacityFlag
) {
    // 获取形状信息
    int64_t numRows = x.size(0);
    int64_t hiddenSize = x.size(1);
    int64_t k = expertIdx.dim() == 2 ? expertIdx.size(1) : 1;
    
    // 计算输出形状
    std::vector<int64_t> expandedX_shape;
    std::vector<int64_t> expandedRowIdx_shape = {numRows * k};
    std::vector<int64_t> expertTokens_shape = {expertNum};
    
    if (dropPadMode == 0) {
        // Dropless 模式
        int64_t active_num = activeNum == 0 ? numRows * k : std::min(activeNum, numRows * k);
        expandedX_shape = {active_num, hiddenSize};
    } else {
        // Drop/Pad 模式
        expandedX_shape = {expertNum, expertCapacity, hiddenSize};
    }
    
    // 创建输出张量
    // expandedX: (active_num, hidden_size) 或 (expert_num, expert_capacity, hidden_size)
    c10::TensorOptions x_options = x.options().dtype(x.scalar_type());
    at::Tensor expandedX = at::empty(expandedX_shape, x_options);
    
    // expandedRowIdx: (numRows * k,)
    c10::TensorOptions idx_options = x.options().dtype(torch::kInt32);
    at::Tensor expandedRowIdx = at::empty(expandedRowIdx_shape, idx_options);
    
    // expertTokensCountOrCumsum: (expert_num,)
    at::Tensor expertTokensCountOrCumsum;
    if (expertTokensCountOrCumsumFlag > 0) {
        expertTokensCountOrCumsum = at::empty(expertTokens_shape, x.options().dtype(torch::kInt32));
    } else {
        expertTokensCountOrCumsum = at::empty({0}, x.options().dtype(torch::kInt32));
    }
    
    // expertTokensBeforeCapacity: (expert_num,)
    at::Tensor expertTokensBeforeCapacity;
    if (expertTokensBeforeCapacityFlag) {
        expertTokensBeforeCapacity = at::empty(expertTokens_shape, x.options().dtype(torch::kInt32));
    } else {
        expertTokensBeforeCapacity = at::empty({0}, x.options().dtype(torch::kInt32));
    }
    
    // 调用 NPU 算子
    EXEC_NPU_CMD(aclnnCustomOp, x, expertIdx, activeNum, expertCapacity, expertNum, dropPadMode, 
                 expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, 
                 expandedX, expandedRowIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity);
    
    return {expandedX, expandedRowIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `moe_init_routing_v2`
    m.def("moe_init_routing_v2", &custom_pybind_api, "");
}
