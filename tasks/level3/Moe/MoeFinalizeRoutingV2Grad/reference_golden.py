# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import torch
import numpy
from atk.configs.dataset_config import InputDataset
from atk.tasks.dataset.base_dataset import OpsDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("function_aclnn_moe_finalize_routing_v2_grad")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # print(f"------------{input_data=}-----------------")
        if input_data.kwargs['grad_y'].dtype == torch.float16:
            tmp_dtype = "float16"
        elif input_data.kwargs['grad_y'].dtype == torch.bfloat16:
            tmp_dtype = "bfloat16"
        elif input_data.kwargs['grad_y'].dtype == torch.float32:
            tmp_dtype = "float32"
        else:
            raise TypeError(f"Unsupported dtype: {dtype}")  
        output_dtype = [tmp_dtype]

        grad_y = input_data.kwargs.get("grad_y").cpu().float().numpy()
        expanded_row_idx = input_data.kwargs.get("expanded_row_idx").cpu().numpy()
        expanded_x = input_data.kwargs.get("expanded_x").cpu().float().numpy()
        scales = input_data.kwargs.get("scales").cpu().float().numpy()
        expert_idx = input_data.kwargs.get("expert_idx").cpu().numpy()
        bias = input_data.kwargs.get("bias").cpu().float().numpy()


        drop_pad_mode = input_data.kwargs.get("drop_pad_mode")
        active_num = input_data.kwargs.get("active_num")
        expert_num = input_data.kwargs.get("expert_num")
        expert_capacity = input_data.kwargs.get("expert_capacity")

        grad_y_shape = grad_y.shape
        expanded_row_idx_shape = expanded_row_idx.shape

        row = grad_y_shape[0]
        hidden = grad_y_shape[1]
        row_topk = expanded_row_idx_shape[0]
        topk = 1
        if scales is not None:
            scales_shape = scales.shape
            topk = scales_shape[1]

        expandedX_dim_0 = row_topk
        if drop_pad_mode == 0 and active_num > 0 and active_num < row_topk:
            expandedX_dim_0 = active_num
        elif drop_pad_mode == 1 and expert_num != 0 and expert_capacity != 0:
            expandedX_dim_0 = expert_num * expert_capacity


        if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
            grad_y = grad_y.astype("float32")
        grad_y_tensor = torch.from_numpy(grad_y)
        grad_y_dtype = grad_y_tensor.dtype
        grad_y_tensor = grad_y_tensor.unsqueeze(1).expand(row, topk, hidden).reshape(row_topk, hidden)
        expanded_row_idx = expanded_row_idx.astype("int64")
        expanded_row_idx_tensor = torch.from_numpy(expanded_row_idx)
        sorted, indices = torch.sort(expanded_row_idx_tensor, dim=-1)

        zeros = torch.zeros((1, hidden), dtype=grad_y_dtype)

        if scales is None:
            # grad_expanded_x
            if drop_pad_mode == 0:
                if expandedX_dim_0 < row_topk:
                    indices = indices[0:expandedX_dim_0]
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices)
            else:
                first_index = torch.nonzero(sorted.ne(-1))[0][0]
                sorted = sorted[first_index:]
                indices = indices[first_index:]
                indices2 = torch.full((expandedX_dim_0,), row_topk)
                indices2.index_put_((sorted,), indices)
                grad_y_tensor = torch.cat((grad_y_tensor, zeros), dim=0)
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices2)

            # grad_scales
            grad_scales_golden_tensor = torch.ones((row_topk, 1), dtype=grad_y_dtype)
        else:
            scales = scales.reshape(row_topk)
            if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
                scales = scales.astype("float32")
            scales_tensor = torch.from_numpy(scales)
            scales_tensor = scales_tensor.unsqueeze(1).expand(-1, hidden)

            # grad_expanded_x
            if drop_pad_mode == 0:
                if expandedX_dim_0 < row_topk:
                    indices = indices[0:expandedX_dim_0]
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices) * scales_tensor.index_select(0,
                                                                                                                    indices)
            else:
                first_index = torch.nonzero(sorted.ne(-1))[0][0]
                sorted = sorted[first_index:]
                indices = indices[first_index:]
                indices2 = torch.full((expandedX_dim_0,), row_topk)
                indices2.index_put_((sorted,), indices)
                grad_y_tensor_2 = torch.cat((grad_y_tensor, zeros), dim=0)
                scales_tensor = torch.cat((scales_tensor, zeros), dim=0)
                grad_expanded_x_golden_tensor = (grad_y_tensor_2.index_select(0, indices2) * scales_tensor.index_select(0, indices2)).reshape(expert_num, expert_capacity, hidden)

            # grad_scales
            if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
                expanded_x = expanded_x.astype("float32")
            expanded_x_tensor = torch.from_numpy(expanded_x)

            if drop_pad_mode == 0:
                if expandedX_dim_0 < row_topk:
                    expanded_x_tensor = torch.cat((expanded_x_tensor, zeros), dim=0)
                    expanded_row_idx_tensor = torch.where(expanded_row_idx_tensor >= expandedX_dim_0,
                                                        torch.tensor(expandedX_dim_0), expanded_row_idx_tensor)
            else:
                expanded_x_tensor = expanded_x_tensor.reshape(expandedX_dim_0, hidden)
                expanded_x_tensor = torch.cat((expanded_x_tensor, zeros), dim=0)
                expanded_row_idx_tensor = torch.where(expanded_row_idx_tensor == -1,
                                                    torch.tensor(expandedX_dim_0, dtype=torch.int64),
                                                    expanded_row_idx_tensor)
                expanded_row_idx_tensor = torch.where(expanded_row_idx_tensor >= expandedX_dim_0,
                                                    torch.tensor(expandedX_dim_0, dtype=torch.int64),
                                                    expanded_row_idx_tensor)

            add_result = expanded_x_tensor.index_select(0, expanded_row_idx_tensor)

            if bias is not None:
                if output_dtype[0] == "bfloat16" or output_dtype[0] == "float16":
                    bias = bias.astype("float32")
                bias_tensor = torch.from_numpy(bias)
                expert_idx = expert_idx.reshape(row_topk)
                expert_idx_tensor = torch.from_numpy(expert_idx)
                add_result = add_result + bias_tensor.index_select(0, expert_idx_tensor)

            # print(f"------------{bias_tensor.dtype=}-----------------")
            # print(f"------------{add_result.dtype=}-----------------")
            # print(f"------------{grad_y_tensor.dtype=}-----------------")
            # print(f"------------{(add_result * grad_y_tensor)=}-----------------")

            grad_scales_golden_tensor = torch.sum(add_result * grad_y_tensor, dim=1).reshape(row, topk)

        if output_dtype[0] == "bfloat16":
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.to(torch.bfloat16)
            grad_scales_golden = grad_scales_golden_tensor.to(torch.bfloat16)
        elif output_dtype[0] == "float16":
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.to(torch.float16)
            grad_scales_golden = grad_scales_golden_tensor.to(torch.float16)
        else:
            grad_expanded_x_golden = grad_expanded_x_golden_tensor    #保持原类型
            grad_scales_golden = grad_scales_golden_tensor

        return grad_expanded_x_golden, grad_scales_golden

    def init_by_input_data(self, input_data: InputDataset):
        # input_data.kwargs['expanded_row_idx'] = torch.arange(15).to(input_data.kwargs['expanded_row_idx'].dtype).to(input_data.kwargs['expanded_row_idx'].device)

        OpsDataset.seed_everything()
        expert_num = input_data.kwargs.get("expert_num")
        sorted_indices_after = torch.randint(low=0, high=input_data.kwargs['expanded_row_idx'].shape[0],
                                             size=(input_data.kwargs['expanded_row_idx'].shape)).to(torch.int32)
        sort_sorted_indices_after = torch.argsort(sorted_indices_after, stable = True)
        # print(f"------------{sort_sorted_indices_after=}-----------------")
        input_data.kwargs['expanded_row_idx'] = sort_sorted_indices_after.to(input_data.kwargs['expanded_row_idx'].dtype).to(input_data.kwargs['expanded_row_idx'].device)
        
        sorted_indices_after = torch.randint(low=0, high=expert_num,
                                        size=(input_data.kwargs['expanded_row_idx'].shape)).to(torch.int32)
        expert_idx = sorted_indices_after.numpy()
        flat_expert_idx = expert_idx.reshape(-1)  # [total_routing]
        sorted_row_idx = numpy.argsort(flat_expert_idx, kind="stable")  # indices
        sorted_expert_idx = flat_expert_idx[sorted_row_idx]  # sorted values

        capacity = input_data.kwargs.get("expert_capacity")
        drop_pad_mode = input_data.kwargs.get("drop_pad_mode")

        if drop_pad_mode == 1:
            # ========================
            # Drop/PAD Mode (3D Output)
            # ========================
            # Apply capacity limit per expert
            count = 0
            last = -1
            for i in range(len(sorted_expert_idx)):
                if sorted_expert_idx[i] != last:
                    count = 1
                    last = sorted_expert_idx[i]
                else:
                    count += 1
                    if count > capacity:
                        sorted_expert_idx[i] = -1
                        sorted_row_idx[i] = -1

            # Build output index mapping
            sort_row_tmp = numpy.full((expert_num * capacity), -1, dtype=int)
            offset = 0
            last_expert = -1
            for i in range(len(sorted_row_idx)):
                if sorted_row_idx[i] != -1:
                    curr_expert = sorted_expert_idx[i]
                    if curr_expert != last_expert:
                        offset = 0
                        last_expert = curr_expert
                    pos = curr_expert * capacity + offset
                    sort_row_tmp[pos] = sorted_row_idx[i]
                    offset += 1

            # expanded_row_idx: map original row -> new position
            expanded_row_idx = numpy.full(input_data.kwargs['expanded_row_idx'].shape[0], -1, dtype=numpy.int32)
            for new_pos, old_pos in enumerate(sort_row_tmp):
                if old_pos != -1:
                    expanded_row_idx[old_pos] = new_pos
            tem = torch.from_numpy(expanded_row_idx)
            input_data.kwargs['expanded_row_idx'] = tem.to(input_data.kwargs['expanded_row_idx'].dtype).to(input_data.kwargs['expanded_row_idx'].device)
        

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *expandedRowIdx, const aclTensor *expandedXOptional, const aclTensor *scalesOptional, const aclTensor *expertIdxOptional, const aclTensor *biasOptional, int64_t dropPadMode, int64_t activeNum, int64_t expertNum, int64_t expertCapacity, const aclTensor *gradExpandedXOut, const aclTensor *gradScalesOut, uint64_t *workspaceSize, aclOpExecutor **executor)"
