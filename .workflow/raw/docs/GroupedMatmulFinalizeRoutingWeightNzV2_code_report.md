# GroupedMamtulFinalizeRoutingWeightNzV2算子的代码检查报告

## 错误信息与函数名不匹配

文件：gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp
行号：1175
代码：
``` c++
        if (tmpWeight->GetDataType() == DataType::DT_INT4 && pertokenScaleOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR,
                    "GroupedMatmulFinalizeRoutingWeightNz does not support nullptr for pertokenScale.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
```
问题: `aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize` 函数调用应该都是 GroupedMatmulFinalizeRoutingWeightNzV2 相关的报错，但上面的报错信息填写的是 GroupedMatmulFinalizeRoutingWeightNz，缺少V2。请检查该函数调用链，找到所有相关的错误报错信息，并进行修改。

## 拼写错误

文件：gmm/grouped_matmul_finalize_routing/op_api/aclnn_grouped_matmul_finalize_routing.cpp
问题：代码中存在多处拼写错误，请检查并纠正：
- first写为fisrt
- parameter写为paramater
- DT_INT32写为INT_32
- viewDimMultiply写为viewDimMutltiply

文件：gmm/grouped_matmul_finalize_routing/op_host/grouped_matmul_finalize_routing_infershape.cpp
- FLOAT8_E8M0写为FlOAT8_E8M0
- is not supported写为is not support
- must bigger than 0
- should be less or equal to