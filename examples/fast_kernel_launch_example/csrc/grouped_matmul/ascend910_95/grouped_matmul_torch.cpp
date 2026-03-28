// /**
//  * This program is free software, you can redistribute it and/or modify it.
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This file is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
//  * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// /*!
//  * \file grouped_matmul_torch.cpp
//  * \brief
//  */

// #include <ATen/Operators.h>
// #include <torch/all.h>
// #include <torch/library.h>
// #include "acl/acl.h"
// #include "torch_npu/csrc/core/npu/NPUStream.h"
// #include "torch_npu/csrc/core/npu/DeviceUtils.h"
// #include "torch_npu/csrc/framework/OpCommand.h"
// #include "tiling/platform/platform_ascendc.h"
// #include "kernel_operator.h"
// #include <stdio.h>  // TODO: remove

// // TODO: include grouped_matmul_tiling_data_apt.h
// #pragma pack(push, 8)
// struct GMMArray {
//     int32_t mList[128] = {0};
//     int32_t kList[128] = {0};
//     int32_t nList[128] = {0};
// };
// #pragma pack(pop)
// #pragma pack(push, 8)
// struct GMMBaseParams {
//     uint32_t groupNum = 0;
//     uint32_t coreNum = 0;
//     uint32_t activeType = 0;
//     uint32_t ubBaseK = 0;
//     uint32_t ubBaseN = 0;
//     uint32_t ubCalSize = 0;
//     uint32_t ubResetBytes = 0;
//     uint32_t singleWeight = 0;
//     uint32_t singleX = 0;
//     uint32_t singleY = 0;
//     int32_t groupType = 0;
//     uint32_t singleN = 0;
//     uint32_t quantParam = 0;
//     uint32_t groupListType = 0;
//     uint32_t m = 0;
//     uint32_t hasBias = 0;
//     uint64_t workspaceSize = 0;
//     uint64_t totalInGroup = 0;
//     uint64_t k = 0;
//     uint64_t n = 0;
//     uint64_t vBaseM = 0;
//     uint64_t parallNum = 0;
//     uint64_t quantGroupNum = 0;
//     uint64_t isPreTiling = 0;
//     uint32_t withOffset = 0;
//     uint32_t isOutputDisableL2Cache = 0;
// };
// #pragma pack(pop)

// #pragma pack(push,8)
// struct A8W4HPTiling {
//     uint32_t group_num = 0;
//     int8_t group_type = 0;
//     uint32_t required_core_num = 0;
//     float format_in = 0;
//     float format_out = 0;
//     uint32_t numAic = 0;
//     uint32_t numAiv = 0;
//     uint64_t szUb = 0;
//     uint64_t szL0A = 0;
//     uint64_t szL0C = 0;
//     uint8_t pattern = 0;
//     uint8_t kernel_index = 0;
//     uint32_t splitTimes = 0;
//     int8_t output_type = 0;
//     uint32_t ori_in0_shape[2] = {0};
//     uint32_t ori_in1_shape[2] = {0};
//     uint32_t ori_out_shape[2] = {0};
//     uint32_t single_core_tiling[3] = {0};
//     uint32_t single_core_base_tiling[2] = {0};
//     uint32_t splitRecord[3] = {0};
//     uint64_t workspaceOffset = 0;
// };
// #pragma pack(pop)

// #pragma pack(push, 8)
// struct GMMTilingData {
//     GMMBaseParams gmmBaseParams;
//     GMMArray gmmArray;
//     TCubeTiling mmTilingData;
//     A8W4HPTiling hpTilingData;
// };
// #pragma pack(pop)

// // TODO: DTYPE_xx FORMAT_xx ORIG_DTYPE_xx按照模板参数往下传
// #define FORMAT_ND 0
// #define FORMAT_FRACTAL_NZ 1
// #define DTYPE_X half
// #define DTYPE_WEIGHT half
// #define DTYPE_BIAS half
// #define DTYPE_SCALE half
// #define DTYPE_Y half
// #define FORMAT_X FORMAT_ND
// #define FORMAT_WEIGHT FORMAT_ND
// #define FORMAT_BIAS FORMAT_ND
// #define FORMAT_SCALE FORMAT_ND
// #define FORMAT_Y FORMAT_ND
// #define ORIG_DTYPE_X DT_FLOAT16
// #define ORIG_DTYPE_WEIGHT DT_FLOAT16
// #define ORIG_DTYPE_BIAS DT_FLOAT16
// #define ORIG_DTYPE_SCALE DT_FLOAT16
// #define ORIG_DTYPE_Y DT_FLOAT16


// // __global__ __aicore__ void printf(__gm__ const char* fmt, ...) {}  // TODO: remove
// #include "gmm/grouped_matmul/op_kernel/grouped_matmul.cpp"

// // TODO: 资料修改，例如dir_structure.md
// // TODO: examples/CMake跳过该目录

// namespace ascend_ops {
// namespace GroupedMatmul {
// const static int64_t IN_NOT_SPLIT_OUT_NOT_SPLIT = 0;
// const static int64_t IN_SPLIT_OUT_NOT_SPLIT = 1;
// const static int64_t IN_NOT_SPLIT_OUT_SPLIT = 2;
// const static int64_t IN_SPLIT_OUT_SPLIT = 3;
// const static int64_t INT4_NUMS_IN_INT32 = 8;
// const static int64_t DEFAULT_SPLIT = -1;
// const static int64_t M_SPLIT = 0;
// const static int64_t K_SPLIT = 2;

// #define PARAM_LIST GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, \
//                    GM_ADDR

// __global__ __aicore__ void GroupedMatmulKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
//                                                GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
//                                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
//                                                const GMMTilingData tiling)
// {
//     using FuncType = void (*)(PARAM_LIST);
//     FuncType func = &grouped_matmul<GMM_TPL_FLOAT16,                           // D_T_A
//                                     GMM_TPL_FLOAT16,                           // D_T_B
//                                     GMM_TPL_FLOAT16,                           // D_T_Y
//                                     0,                                         // TRANS_A
//                                     0,                                         // TRANS_B
//                                     GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT,      // GROUP_LIST_TYPE
//                                     0,                                         // IS_STATIC_TILING_API
//                                     GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_NONE,  // A8W4_KERNEL_TEMPLATE
//                                     GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, // A16W8_KERNEL_TEMPLATE
//                                     GROUPED_MATMUL_CUBE_ONLY,                  // AIV_AIC_RATIO
//                                     false                                      // IS_ENABLE_FIXED_AXIS
//                                     >(PARAM_LIST);

//     func(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, groupList, perTokenScale, y, workspace,
//          &tiling);
// }

// template <typename T>
// void GroupedMatmulApi(aclrtStream stream, const torch::TensorList x, const torch::TensorList weight,
//                       const c10::optional<torch::TensorList> bias, const c10::optional<torch::TensorList> scale,
//                       const c10::optional<torch::TensorList> offset, const c10::optional<torch::TensorList> antiquant_scale,
//                       const c10::optional<torch::TensorList> antiquant_offset,
//                       const c10::optional<torch::TensorList> per_token_scale, const c10::optional<torch::Tensor> &group_list,
//                       const c10::optional<torch::TensorList> activation_input,
//                       const c10::optional<torch::TensorList> activation_quant_scale,
//                       const c10::optional<torch::TensorList> activation_quant_offset, const torch::TensorList y,
//                       int64_t split_item, int64_t group_type, int64_t group_list_type, int64_t act_type,
//                       const std::vector<int64_t> &tuning_config)
// {
//     printf("enter into GroupedMatmulApi\n");

//     size_t E = 1;
//     uint32_t blockDim = 24;
//     GMMTilingData tilingData;
//     tilingData.gmmBaseParams.groupNum = E;
//     tilingData.gmmBaseParams.coreNum = blockDim;
//     tilingData.gmmBaseParams.groupType = 0;

//     tilingData.gmmArray.mList[0] = 345;
//     tilingData.gmmArray.kList[0] = -1;
//     tilingData.gmmArray.nList[0] = 567;

//     tilingData.mmTilingData.usedCoreNum = blockDim;
//     tilingData.mmTilingData.M = 345;
//     tilingData.mmTilingData.N = 567;
//     tilingData.mmTilingData.Ka = 1280;
//     tilingData.mmTilingData.Kb = 1280;
//     tilingData.mmTilingData.singleCoreM = 345;
//     tilingData.mmTilingData.singleCoreN = 256;
//     tilingData.mmTilingData.singleCoreK = 1280;
//     tilingData.mmTilingData.baseM = 128;
//     tilingData.mmTilingData.baseN = 256;
//     tilingData.mmTilingData.baseK = 64;
//     tilingData.mmTilingData.depthA1 = 8;
//     tilingData.mmTilingData.depthB1 = 8;
//     tilingData.mmTilingData.stepM = 1;
//     tilingData.mmTilingData.stepN = 1;
//     tilingData.mmTilingData.stepKa = 4;
//     tilingData.mmTilingData.stepKb = 4;
//     tilingData.mmTilingData.isBias = 0;
//     tilingData.mmTilingData.transLength = 0;
//     tilingData.mmTilingData.iterateOrder = 0;
//     tilingData.mmTilingData.dbL0A = true;
//     tilingData.mmTilingData.dbL0B = true;
//     tilingData.mmTilingData.dbL0C = false;

//     // TODO: tensor list怎么获取地址？只传第一个肯定是不可以的
//     __gm__ uint8_t* xPtr = (__gm__ uint8_t*)x[0].data_ptr<T>();
//     __gm__ uint8_t* weightPtr = (__gm__ uint8_t*)weight[0].data_ptr<T>();
//     __gm__ uint8_t* biasPtr = nullptr;
//     __gm__ uint8_t* scalePtr = nullptr;
//     __gm__ uint8_t* offsetPtr = nullptr;
//     __gm__ uint8_t* antiquantScalePtr = nullptr;
//     __gm__ uint8_t* antiquantOffsetPtr = nullptr;
//     __gm__ uint8_t* groupListPtr = (__gm__ uint8_t*)group_list.value().data_ptr<int64_t>();
//     __gm__ uint8_t* perTokenScalePtr = nullptr;
//     __gm__ uint8_t* yPtr = (__gm__ uint8_t*)y[0].data_ptr<T>();
//     __gm__ uint8_t* workspace = nullptr;
//     GroupedMatmulKernel<<<blockDim, nullptr, stream>>>(xPtr, weightPtr, biasPtr, scalePtr, offsetPtr, antiquantScalePtr,
//                                                        antiquantOffsetPtr, groupListPtr, perTokenScalePtr, yPtr,
//                                                        workspace, tilingData);
// }


// void check_dims(int64_t split_item, size_t num_x, size_t num_weight, size_t num_group_list)
// {
//     TORCH_CHECK(num_x > 0 && num_weight > 0,
//                 "Invalid inputs: neither x nor weight could be empty.");
//     TORCH_CHECK(split_item == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item == IN_SPLIT_OUT_NOT_SPLIT ||
//                 split_item == IN_NOT_SPLIT_OUT_SPLIT || split_item == IN_SPLIT_OUT_SPLIT,
//                 "Invalid value of split_item [", split_item, "], which should only be one of 0/1/2/3.");
//     if (split_item == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item == IN_SPLIT_OUT_NOT_SPLIT) {
//         if (num_group_list > 0) {
//             TORCH_CHECK(num_x == 1 && num_weight == num_group_list, "Invalid inputs. "
//                         "When split_item = 0 or 1 and input group_list is not None, "
//                         "the following two conditions are supposed to be satisfied: "
//                         "(1) length of x equals 1; (2) length of weight equals that of group_list. "
//                         "Actual lengths: x [", num_x, "], weight [", num_weight, "], "
//                         "group_list [", num_group_list, "].");
//         } else {
//             TORCH_CHECK(num_x == num_weight,
//                         "When split_item = 0 or 1 and input group_list is None, "
//                         "the num of x tensors must equal the num of weight tensors."
//                         "Actual lengths: x [", num_x, "], weight [", num_weight, "].");
//         }
//     }
// }

// void create_new_tensor_multi_dim(std::vector<torch::Tensor> &y, const torch::Tensor &x_i, int64_t n,
//                                  c10::TensorOptions options)
// {
//     auto x_sizes = x_i.sizes();
//     std::vector<int64_t> y_sizes(x_sizes.begin(), x_sizes.end());
//     y_sizes.at(x_sizes.size() - 1) = static_cast<int64_t>(n);

//     y.emplace_back(torch::empty(y_sizes, options));
// }

// void create_new_tensor(std::vector<torch::Tensor> &y, int64_t dim_m, int64_t dim_n, c10::TensorOptions options)
// {
//     y.emplace_back(torch::empty({dim_m, dim_n}, options));
// }

// void create_new_tensor_batch(std::vector<torch::Tensor> &y, int64_t batch, int64_t dim_m, int64_t dim_n,
//                              c10::TensorOptions options)
// {
//     y.emplace_back(torch::empty({batch, dim_m, dim_n}, options));
// }

// void calculate_dim_m(size_t& dim_m, size_t num_x, const torch::TensorList x)
// {
//     for (size_t i = 0; i < num_x; i++) {
//         dim_m += x[i].sizes()[0];
//     }
// }

// // TODO: 要使用Torch::Tensor，并且把接口改成引用
// // Motivation for adapting this interface for each Torch version separately:
// // 1. Optional TensorList is only supported in Torch2.1 and later versions.
// //    Thus, "Tensor[] bias" is used in Torch1.11 and Torch2.0, while
// //    "Tensor[]? bias=None" is used in Torch2.1 and later versions.
// // 2. Even if "Int[]? group_list=None" is used for all Torch versions, the
// //    auto-generated data type for optional IntList group_list in Torch2.1
// //    is different from those in Torch1.11 and Torch2.0.
//  std::vector<torch::Tensor> GroupedMatmulNpu(const torch::TensorList x,
//                                            const torch::TensorList weight,
//                                            const c10::optional<torch::TensorList> bias,
//                                            const c10::optional<torch::TensorList> scale,
//                                            const c10::optional<torch::TensorList> offset,
//                                            const c10::optional<torch::TensorList> antiquant_scale,
//                                            const c10::optional<torch::TensorList> antiquant_offset,
//                                            const c10::optional<torch::TensorList> per_token_scale,
//                                            const c10::optional<torch::Tensor>& group_list,
//                                            const c10::optional<torch::TensorList> activation_input,
//                                            const c10::optional<torch::TensorList> activation_quant_scale,
//                                            const c10::optional<torch::TensorList> activation_quant_offset,
//                                            c10::optional<int64_t> split_item,
//                                            c10::optional<int64_t> group_type,
//                                            c10::optional<int64_t> group_list_type,
//                                            c10::optional<int64_t> act_type,
//                                            const c10::OptionalIntArrayRef tuning_config,
//                                            c10::optional<at::ScalarType> output_dtype)
// {
//     printf("enter into GroupedMatmulNpu\n");
//     // TODO: 检查所有输入是否都在npu上，包括可选输入
//     TORCH_CHECK(torch_npu::utils::is_npu(x[0]), "Input tensor must be on NPU device");

//     TORCH_CHECK(group_type.has_value(),
//                 "Requires manual passing group_type, current is None.");
//     int64_t group_type_value = group_type.value();
//     // TODO: demo先支持M分组，打通流程后再支持其他场景
//     TORCH_CHECK(group_type_value == DEFAULT_SPLIT || group_type_value == M_SPLIT || group_type_value == K_SPLIT,
//                 "Use Tensor input with current cann version, "
//                 "The group type must be -1, 0 or 2, but now is [", group_type_value, "]");

//     // TODO: at namespace改torch
//     auto num_x = x.size();
//     bool singleWeight = weight.size() == 1 && weight[0].sizes().size() == 3;
//     auto num_weight = singleWeight ? static_cast<size_t>(weight[0].size(0)) : static_cast<size_t>(weight.size());
//     auto group_list_real = group_list.value_or(torch::Tensor());
//     auto num_group_list = group_list_real.size(0);
//     int64_t split_item_value = split_item.value_or(0);
//     check_dims(split_item_value, num_x, num_weight, num_group_list);

//     std::vector<torch::Tensor> y;
//     c10::TensorOptions options = x[0].options().dtype(output_dtype.value_or(x[0].scalar_type()));

//     // TODO: 转置属性获取
//     size_t dim_num_w = weight[0].sizes().size();
//     size_t n0 = static_cast<size_t>(weight[0].size(dim_num_w - 1));

//     if (split_item_value == IN_NOT_SPLIT_OUT_NOT_SPLIT || split_item_value == IN_SPLIT_OUT_NOT_SPLIT) {
//         if (num_group_list > 0) {
//             y.reserve(num_group_list);
//             int64_t glr_value_0 = group_list_real[0].item<int64_t>();
//             TORCH_CHECK(glr_value_0 >= 0,
//                 "group_list[0] should be larger than or equal to 0, but now is ", glr_value_0, "." );
//             create_new_tensor(y, glr_value_0, n0, options);
//             int64_t glr_value_pre = glr_value_0;
//             for (int i = 1; i < num_group_list; i++) {
//                 int64_t glr_value_cur = group_list_real[i].item<int64_t>();
//                 TORCH_CHECK(glr_value_cur - glr_value_pre >= 0,
//                     "group_list[", i, "] - group_list[", i - 1, "] should be larger than or equal to 0, but now is ",
//                     glr_value_cur - glr_value_pre, ".");
//                 size_t ni = singleWeight ? n0 : weight[i].size(dim_num_w - 1);
//                 create_new_tensor(y, glr_value_cur - glr_value_pre, ni, options);
//                 glr_value_pre = glr_value_cur;
//             }
//         } else {
//             y.reserve(num_x);
//             for (size_t i = 0; i < num_x; i++) {
//                 size_t ni = singleWeight ? n0 : weight[i].size(dim_num_w - 1);
//                 create_new_tensor_multi_dim(y, x[i], ni, options);
//             }
//         }  // 校验NO_SPLIT时为特殊场景（groupList为空）或num_x > 1
//     } else if (split_item_value == IN_NOT_SPLIT_OUT_SPLIT || split_item_value == IN_SPLIT_OUT_SPLIT) {
//         if (num_x > 1) {
//             size_t dim_m = 0;
//             for (size_t i = 0; i < num_x; i++) {
//                 dim_m += static_cast<size_t>(x[i].size(0));
//             }
//             weight[0].dtype() == at::ScalarType::Int ?
//                 create_new_tensor(y, dim_m, n0 * INT4_NUMS_IN_INT32, options) :
//                 create_new_tensor(y, dim_m, n0, options);
//         } else if (num_x == 1) {
//             if (group_type_value == K_SPLIT) {
//                     TORCH_CHECK(num_weight == 1,
//                         "When group_list is 2(K_SPLIT) and split_item is 2/3, the length of weight must equal x.");
//                     create_new_tensor_batch(y, num_group_list, x[0].size(0), n0, options);
//             } else {
//             weight[0].dtype() == at::ScalarType::Int ?
//                 create_new_tensor(y, x[0].size(0), n0 * INT4_NUMS_IN_INT32, options) :
//                 create_new_tensor(y, x[0].size(0), n0, options);
//             }
//         }
//     }
//     torch::TensorList result = torch::TensorList(y);

//     auto bias_real = bias.value_or(torch::TensorList());
//     auto scale_real = scale.value_or(torch::TensorList());
//     auto offset_real = offset.value_or(torch::TensorList());
//     auto antiquant_scale_real = antiquant_scale.value_or(torch::TensorList());
//     auto antiquant_offset_real = antiquant_offset.value_or(torch::TensorList());
//     auto per_token_scale_real = per_token_scale.value_or(torch::TensorList());
//     auto activation_input_real = activation_input.value_or(torch::TensorList());
//     auto activation_quant_scale_real = activation_quant_scale.value_or(torch::TensorList());
//     auto activation_quant_offset_real = activation_quant_offset.value_or(torch::TensorList());
//     auto act_out = torch::TensorList();  // TODO: 起到啥作用？
//     auto dynamic_quant_scale_out = torch::TensorList(); // TODO: 冗余
//     int64_t group_list_type_value = group_list_type.value_or(0);
//     int64_t act_type_value = act_type.value_or(0);
//     auto tuning_config_real = tuning_config.value_or(at::IntArrayRef{});  // TODO: convert to std::vector<int64_t>

//     auto stream = c10_npu::getCurrentNPUStream().stream(false);

//     // TODO: 宏每个参数的含义是什么
//     auto acl_call = [=]() -> int {
//         AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x[0].scalar_type(), "GroupedMatmulNpu", [&] {
//             GroupedMatmulApi<scalar_t>(
//                 stream, x, weight, bias_real, scale_real, offset_real, antiquant_scale_real, antiquant_offset_real,
//                 per_token_scale_real, group_list_real, activation_input_real, activation_quant_scale_real,
//                 activation_quant_offset_real, result, split_item_value, group_type_value, group_list_type_value,
//                 act_type_value, std::vector<int64_t>());  // TODO: tuning_config_real要能传进去
//         });
//         return 0;
//     };

//     at_npu::native::OpCommand::RunOpApiV2("GroupedMatmul", acl_call);
//     return y;
// }

// TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
// {
//     // TODO: tuning_config默认值
//     m.def(
//           "grouped_matmul(Tensor[] x, Tensor[] weight, Tensor[]? bias=None, Tensor[]? scale=None, "
//           "Tensor[]? offset=None, Tensor[]? antiquant_scale=None, Tensor[]? antiquant_offset=None, "
//           "Tensor[]? per_token_scale=None, Tensor? group_list=None, Tensor[]? activation_input=None, "
//           "Tensor[]? activation_quant_scale=None, Tensor[]? activation_quant_offset=None, int? split_item=0,"
//           "int? group_type=-1, int? group_list_type=0, int? act_type=0, int[]? tuning_config=None, "
//           "ScalarType? output_dtype=None) -> Tensor[]");
// }

// // Register Ascend implementations for grouped_matmul
// TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
// {
//     m.impl("grouped_matmul", GroupedMatmulNpu);
// }

// } // namespace GroupedMatmul
// } // namespace ascend_ops
