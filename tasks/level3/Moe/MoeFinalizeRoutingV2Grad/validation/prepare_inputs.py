import torch
import numpy as np


def generate_inputs(
    batch_size,
    hidden_size,
    k,
    use_bias,
    dtype,
    dropPadMode,
    active_num,
    expert_num,
    expert_capacity,
    use_scale,
):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    row = batch_size
    row_topk = row * k

    gradY = torch.randn(row, hidden_size, dtype=torch_dtype)
    if use_scale:
        scalesOptional = torch.randn(row, k, dtype=torch_dtype)
    else:
        scalesOptional = torch.zeros((0, 0), dtype=torch_dtype)

    if use_bias and expert_num > 0:
        expertIdxOptional = torch.randint(0, expert_num, (row, k), dtype=torch.int32)
        biasOptional = torch.randn(expert_num, hidden_size, dtype=torch_dtype)
    else:
        expertIdxOptional = torch.zeros((0, 0), dtype=torch.int32)
        biasOptional = torch.zeros((0, 0), dtype=torch_dtype)

    if dropPadMode == 1 and expert_num > 0 and expert_capacity > 0:
        expandedXOptional = torch.randn(expert_num, expert_capacity, hidden_size, dtype=torch_dtype)
        if expertIdxOptional.numel() > 0:
            flat_expert_idx = expertIdxOptional.reshape(-1).numpy()
        else:
            flat_expert_idx = np.random.randint(0, expert_num, size=row_topk, dtype=np.int32)
        sorted_row_idx = np.argsort(flat_expert_idx, kind="stable")
        sorted_expert_idx = flat_expert_idx[sorted_row_idx].copy()

        count = 0
        last = -1
        for i in range(len(sorted_expert_idx)):
            if sorted_expert_idx[i] != last:
                count = 1
                last = sorted_expert_idx[i]
            else:
                count += 1
                if count > expert_capacity:
                    sorted_expert_idx[i] = -1
                    sorted_row_idx[i] = -1

        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=np.int32)
        offset = 0
        last_expert = -1
        for i in range(len(sorted_row_idx)):
            if sorted_row_idx[i] != -1:
                curr_expert = sorted_expert_idx[i]
                if curr_expert != last_expert:
                    offset = 0
                    last_expert = curr_expert
                pos = curr_expert * expert_capacity + offset
                sort_row_tmp[pos] = sorted_row_idx[i]
                offset += 1

        expanded_row_idx = np.full(row_topk, -1, dtype=np.int32)
        for new_pos, old_pos in enumerate(sort_row_tmp):
            if old_pos != -1:
                expanded_row_idx[old_pos] = new_pos
        ec = expert_num * expert_capacity
        expanded_row_idx = np.clip(expanded_row_idx, -1, ec - 1)
        expandedRowIdx = torch.from_numpy(expanded_row_idx)
    else:
        max_rows = row_topk
        if active_num > 0 and active_num < row_topk:
            max_rows = active_num
        expandedXOptional = torch.randn(max_rows, hidden_size, dtype=torch_dtype)
        # 与 reference_golden 中 init 一致：expanded_row_idx 为 0..RK-1 的排列
        expandedRowIdx = torch.randperm(row_topk, dtype=torch.int64).to(torch.int32)

    inputs = {
        "gradY": gradY,
        "expandedRowIdx": expandedRowIdx,
        "expandedXOptional": expandedXOptional,
        "scalesOptional": scalesOptional,
        "expertIdxOptional": expertIdxOptional,
        "biasOptional": biasOptional,
        "dropPadMode": torch.tensor(dropPadMode, dtype=torch.int64),
        "activeNum": torch.tensor(active_num, dtype=torch.int64),
        "expertNum": torch.tensor(expert_num, dtype=torch.int64),
        "expertCapacity": torch.tensor(expert_capacity, dtype=torch.int64),
    }
    return inputs


def get_init_inputs(row, device=None):
    """
    precision 阶段以 Model(*get_init_inputs(row)) 构造 CPU 金标模型，必须与 CSV 中属性一致，
    否则 Model 会落在默认 dropPadMode=0 / expert_num=0，与真实输入不匹配。
    """
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    drop_pad_mode = int(row.get("dropPadMode", 0))
    active_num = int(row.get("active_num", 0))
    expert_num = int(row.get("expert_num", 0))
    expert_capacity = int(row.get("expert_capacity", 0))
    return [drop_pad_mode, active_num, expert_num, expert_capacity]

def get_inputs(row, device=None):
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    batch_size = int(row.get("batch_size", 16))
    hidden_size = int(row.get("hidden_size", 256))
    k = int(row.get("k", 2))
    use_bias = bool(int(row.get("use_bias", 1)))
    dtype = row.get("dtype", "float16")
    dropPadMode = int(row.get("dropPadMode", 0))
    active_num = int(row.get("active_num", 0))
    expert_num = int(row.get("expert_num", 0))
    expert_capacity = int(row.get("expert_capacity", 0))
    use_scale = bool(int(row.get("use_scale", 1)))

    inputs_dict = generate_inputs(
        batch_size,
        hidden_size,
        k,
        use_bias,
        dtype,
        dropPadMode,
        active_num,
        expert_num,
        expert_capacity,
        use_scale,
    )

    if device is not None:
        if hasattr(device, "type") and device.type == "cuda":
            device_id = device.index if hasattr(device, "index") else 0
            device = torch.device(f"npu:{device_id}")
        for key in inputs_dict:
            inputs_dict[key] = inputs_dict[key].to(device)

    return [
        inputs_dict["gradY"],
        inputs_dict["expandedRowIdx"],
        inputs_dict["expandedXOptional"],
        inputs_dict["scalesOptional"],
        inputs_dict["expertIdxOptional"],
        inputs_dict["biasOptional"]
    ]