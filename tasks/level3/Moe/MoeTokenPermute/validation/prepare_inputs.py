import ast
import torch


def get_inputs(row, device=None):
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    raw_shape = row.get("tokens_shape", "[4, 128]")
    if isinstance(raw_shape, str):
        tokens_shape = ast.literal_eval(raw_shape.replace(" ", ""))
    else:
        tokens_shape = list(raw_shape)

    topk = int(row.get("topk", 2))
    num_out_tokens = int(row.get("num_out_tokens", 0))
    dtype_str = row.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    padded_mode = bool(int(row.get("padded_mode", 0)))

    num_tokens = int(tokens_shape[0])
    hidden = int(tokens_shape[1])
    tokens = torch.rand((num_tokens, hidden), dtype=dtype, device=device)

    if padded_mode:
        # padded：indices 直接作为 tokens 行号参与 index_select，须在 [0, num_tokens)
        high = max(num_tokens, 1)
        if topk > 1:
            indices = torch.randint(0, high, size=(num_tokens, topk), dtype=torch.int32, device=device)
        else:
            indices = torch.randint(0, high, size=(num_tokens,), dtype=torch.int32, device=device)
    else:
        total_slots = num_tokens * topk
        high = max(total_slots, 1)
        if topk > 1:
            indices = torch.randint(0, high, size=(num_tokens, topk), dtype=torch.int32, device=device)
        else:
            indices = torch.randint(0, high, size=(num_tokens,), dtype=torch.int32, device=device)

    return (tokens, indices, num_out_tokens, padded_mode)


def get_init_inputs(row, device=None):
    return []


def custom_check_precision(row, outputs, outputs_new):
    from npukernelbench.stages.precision.comparator import check_precision

    if hasattr(row, "to_dict"):
        row = row.to_dict()
    dtype_str = row.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    outs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    outs_new = list(outputs_new) if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    check_outputs = []
    check_outputs_new = []
    for o, on in zip(outs, outs_new):
        if o is None or on is None:
            continue
        check_outputs.append(o)
        check_outputs_new.append(on)
    if dtype == torch.float32:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
    if dtype == torch.float16:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.001, max_rel_error=0.001)
    if dtype == torch.bfloat16:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.004, max_rel_error=0.004)
    return check_precision(check_outputs, check_outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
