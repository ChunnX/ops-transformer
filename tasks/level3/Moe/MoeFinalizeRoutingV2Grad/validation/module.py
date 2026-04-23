import torch
import torch_npu
import kernel_gen_ops


class Model(torch.nn.Module):
    def __init__(self, dropPadMode=0, active_num=0, expert_num=0, expert_capacity=0, *args, **kwargs):
        super(Model, self).__init__()
        self.dropPadMode = int(dropPadMode)
        self.active_num = int(active_num)
        self.expert_num = int(expert_num)
        self.expert_capacity = int(expert_capacity)

    def forward(self, gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional):
        grad_y = gradY.cpu().float().numpy()
        expanded_row_idx = expandedRowIdx.cpu().numpy()
        expanded_x = (
            expandedXOptional.cpu().float().numpy()
            if expandedXOptional is not None and expandedXOptional.numel() > 0
            else None
        )
        scales = (
            scalesOptional.cpu().float().numpy()
            if scalesOptional is not None and scalesOptional.numel() > 0
            else None
        )
        expert_idx = (
            expertIdxOptional.cpu().numpy()
            if expertIdxOptional is not None and expertIdxOptional.numel() > 0
            else None
        )
        bias = (
            biasOptional.cpu().float().numpy()
            if biasOptional is not None and biasOptional.numel() > 0
            else None
        )

        grad_y_shape = grad_y.shape
        expanded_row_idx_shape = expanded_row_idx.shape
        row = grad_y_shape[0]
        hidden = grad_y_shape[1]
        row_topk = expanded_row_idx_shape[0]
        # 无 scales 时仍须按 RK=row*topk 将 grad_y 扩到 row_topk；topk 由 row 与 expanded 长度推导
        topk = (row_topk // row) if row > 0 else 1
        if scales is not None:
            topk = scales.shape[1]

        expandedX_dim_0 = row_topk
        if self.dropPadMode == 0 and self.active_num > 0 and self.active_num < row_topk:
            expandedX_dim_0 = self.active_num
        elif self.dropPadMode == 1 and self.expert_num != 0 and self.expert_capacity != 0:
            expandedX_dim_0 = self.expert_num * self.expert_capacity

        grad_y_tensor = torch.from_numpy(grad_y)
        grad_y_dtype = grad_y_tensor.dtype
        grad_y_tensor = grad_y_tensor.unsqueeze(1).expand(row, topk, hidden).reshape(row_topk, hidden)
        expanded_row_idx_tensor = torch.from_numpy(expanded_row_idx.astype("int64"))
        sorted_idx, indices = torch.sort(expanded_row_idx_tensor, dim=-1)
        zeros = torch.zeros((1, hidden), dtype=grad_y_dtype)

        if scales is None:
            if self.dropPadMode == 0:
                if expandedX_dim_0 < row_topk:
                    indices = indices[0:expandedX_dim_0]
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices)
            else:
                first_index = torch.nonzero(sorted_idx.ne(-1))[0][0]
                sorted_idx = sorted_idx[first_index:]
                indices = indices[first_index:]
                indices2 = torch.full((expandedX_dim_0,), row_topk, dtype=torch.int64)
                indices2.index_put_((sorted_idx,), indices)
                grad_y_tensor = torch.cat((grad_y_tensor, zeros), dim=0)
                grad_expanded_x_golden_tensor = grad_y_tensor.index_select(0, indices2)
            # 与算子 infershape 一致：无 scales 时输出 grad_scales 形状为 (row, 1)
            grad_scales_golden_tensor = torch.ones((row, 1), dtype=grad_y_dtype)
        else:
            scales = scales.reshape(row_topk)
            scales_tensor = torch.from_numpy(scales.astype("float32")).unsqueeze(1).expand(-1, hidden)
            if self.dropPadMode == 0:
                if expandedX_dim_0 < row_topk:
                    indices = indices[0:expandedX_dim_0]
                grad_expanded_x_golden_tensor = (
                    grad_y_tensor.index_select(0, indices) * scales_tensor.index_select(0, indices)
                )
            else:
                first_index = torch.nonzero(sorted_idx.ne(-1))[0][0]
                sorted_idx = sorted_idx[first_index:]
                indices = indices[first_index:]
                indices2 = torch.full((expandedX_dim_0,), row_topk, dtype=torch.int64)
                indices2.index_put_((sorted_idx,), indices)
                grad_y_tensor_2 = torch.cat((grad_y_tensor, zeros), dim=0)
                scales_tensor = torch.cat((scales_tensor, zeros), dim=0)
                grad_expanded_x_golden_tensor = (
                    grad_y_tensor_2.index_select(0, indices2) * scales_tensor.index_select(0, indices2)
                ).reshape(self.expert_num, self.expert_capacity, hidden)

            if expanded_x is None:
                raise ValueError("expandedXOptional is required when scales are present")
            expanded_x_tensor = torch.from_numpy(expanded_x.astype("float32"))
            if self.dropPadMode == 0:
                if expandedX_dim_0 < row_topk:
                    expanded_x_tensor = torch.cat((expanded_x_tensor, zeros), dim=0)
                    expanded_row_idx_tensor = torch.where(
                        expanded_row_idx_tensor >= expandedX_dim_0,
                        torch.tensor(expandedX_dim_0, dtype=torch.int64),
                        expanded_row_idx_tensor,
                    )
            else:
                expanded_x_tensor = expanded_x_tensor.reshape(expandedX_dim_0, hidden)
                expanded_x_tensor = torch.cat((expanded_x_tensor, zeros), dim=0)
                expanded_row_idx_tensor = torch.where(
                    expanded_row_idx_tensor == -1,
                    torch.tensor(expandedX_dim_0, dtype=torch.int64),
                    expanded_row_idx_tensor,
                )
                expanded_row_idx_tensor = torch.where(
                    expanded_row_idx_tensor >= expandedX_dim_0,
                    torch.tensor(expandedX_dim_0, dtype=torch.int64),
                    expanded_row_idx_tensor,
                )

            add_result = expanded_x_tensor.index_select(0, expanded_row_idx_tensor)
            if bias is not None and expert_idx is not None:
                bias_tensor = torch.from_numpy(bias.astype("float32"))
                expert_idx_tensor = torch.from_numpy(expert_idx.reshape(row_topk))
                add_result = add_result + bias_tensor.index_select(0, expert_idx_tensor)

            grad_scales_golden_tensor = torch.sum(add_result * grad_y_tensor, dim=1).reshape(row, topk)

        if gradY.dtype == torch.bfloat16:
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.to(torch.bfloat16)
            grad_scales_golden = grad_scales_golden_tensor.to(torch.bfloat16)
        elif gradY.dtype == torch.float16:
            grad_expanded_x_golden = grad_expanded_x_golden_tensor.to(torch.float16)
            grad_scales_golden = grad_scales_golden_tensor.to(torch.float16)
        else:
            grad_expanded_x_golden = grad_expanded_x_golden_tensor
            grad_scales_golden = grad_scales_golden_tensor

        return grad_expanded_x_golden.contiguous(), grad_scales_golden.contiguous()


class ModelNew(torch.nn.Module):
    def __init__(self, dropPadMode=0, active_num=0, expert_num=0, expert_capacity=0, *args, **kwargs):
        super(ModelNew, self).__init__()
        self.dropPadMode = int(dropPadMode)
        self.active_num = int(active_num)
        self.expert_num = int(expert_num)
        self.expert_capacity = int(expert_capacity)

    def forward(self, gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional):
        target = gradY.device if gradY.device.type == "npu" else torch.device("npu", torch.npu.current_device())
        gradY = gradY.to(target)
        expandedRowIdx = expandedRowIdx.to(target)
        if expandedXOptional is not None:
            expandedXOptional = expandedXOptional.to(target)
        if scalesOptional is not None:
            scalesOptional = scalesOptional.to(target)
        if expertIdxOptional is not None:
            expertIdxOptional = expertIdxOptional.to(target)
        if biasOptional is not None:
            biasOptional = biasOptional.to(target)
        grad_expanded_x_out, grad_scales_out = kernel_gen_ops.moe_finalize_routing_v2_grad(
            gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional,
            self.dropPadMode, self.active_num, self.expert_num, self.expert_capacity
        )
        return grad_expanded_x_out, grad_scales_out


expert_matcher = {
    "float32": {
        "rtol": 1.e-5,
        "atol": 1.e-5,
        "total_atol": 1.e-4
    },
    "float16": {
        "rtol": 1.e-3,
        "atol": 1.e-3,
        "total_atol": 1.e-3
    },
    "bfloat16": {
        "rtol": 1.e-2,
        "atol": 1.e-2,
        "total_atol": 1.e-2
    }
}


def get_model(data_format="channels_first", **kwargs):
    dropPadMode = kwargs.get("dropPadMode", kwargs.get("row", {}).get("dropPadMode", 0))
    active_num = kwargs.get("active_num", kwargs.get("activeNum", kwargs.get("row", {}).get("active_num", kwargs.get("row", {}).get("activeNum", 0))))
    expert_num = kwargs.get("expert_num", kwargs.get("expertNum", kwargs.get("row", {}).get("expert_num", kwargs.get("row", {}).get("expertNum", 0))))
    expert_capacity = kwargs.get("expert_capacity", kwargs.get("expertCapacity", kwargs.get("row", {}).get("expert_capacity", kwargs.get("row", {}).get("expertCapacity", 0))))
    
    if hasattr(dropPadMode, "item"):
        dropPadMode = dropPadMode.item()
    if hasattr(active_num, "item"):
        active_num = active_num.item()
    if hasattr(expert_num, "item"):
        expert_num = expert_num.item()
    if hasattr(expert_capacity, "item"):
        expert_capacity = expert_capacity.item()
    return Model(
        dropPadMode=dropPadMode,
        active_num=active_num,
        expert_num=expert_num,
        expert_capacity=expert_capacity
    )


def get_model_new(data_format="channels_first", **kwargs):
    dropPadMode = kwargs.get("dropPadMode", kwargs.get("row", {}).get("dropPadMode", 0))
    active_num = kwargs.get("active_num", kwargs.get("activeNum", kwargs.get("row", {}).get("active_num", kwargs.get("row", {}).get("activeNum", 0))))
    expert_num = kwargs.get("expert_num", kwargs.get("expertNum", kwargs.get("row", {}).get("expert_num", kwargs.get("row", {}).get("expertNum", 0))))
    expert_capacity = kwargs.get("expert_capacity", kwargs.get("expertCapacity", kwargs.get("row", {}).get("expert_capacity", kwargs.get("row", {}).get("expertCapacity", 0))))
    
    if hasattr(dropPadMode, "item"):
        dropPadMode = dropPadMode.item()
    if hasattr(active_num, "item"):
        active_num = active_num.item()
    if hasattr(expert_num, "item"):
        expert_num = expert_num.item()
    if hasattr(expert_capacity, "item"):
        expert_capacity = expert_capacity.item()
    return ModelNew(
        dropPadMode=dropPadMode,
        active_num=active_num,
        expert_num=expert_num,
        expert_capacity=expert_capacity
    )


def get_matcher(dtype="float32", **kwargs):
    return expert_matcher.get(dtype, expert_matcher["float32"])