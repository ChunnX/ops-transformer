import numpy as np
import torch

def parse_bool_param(param):
    if isinstance(param, str) and param.lower() in ['true', 't', '1']:
        return True
    if param == 1:
        return True
    return False

def gen_input_data(shape, dtype_str, input_range):
    """
    生成AdamW优化器所需的输入数据。
    """
    # 映射数据类型
    dtype_map = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int64": torch.int64,
    }

    dtype = dtype_map.get(dtype_str.lower(), torch.float32)

    # 随机生成数据
    np.random.seed(5)  # 保持一致性
    var = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    m = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    v = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    max_grad_norm = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)
    grad = torch.tensor(np.random.uniform(input_range[0], input_range[1], shape), dtype=dtype)

    return var, m, v, max_grad_norm, grad

def get_inputs(param, device=None):
    """
    根据参数生成 ApplyAdamWV2 算子的输入数据。
    """
    # 解析基本参数
    shape = eval(param.get('shape', '[2, 2, 2]'))
    dtype_str = str(param.get('dtype', 'float16')).lower()
    input_range = [0.1, 1]  # 与原始代码保持一致

    # 生成基础变量
    var_ref, m_ref, v_ref, max_grad_norm_ref, grad = gen_input_data(shape, dtype_str, input_range)
    # CSV 的 step 表示「第几步」(1-based)，与 PyTorch state_steps（先 +=1 再算 bias）及 kernel（内部 +1）对齐：存 T-1
    t = int(param.get("step", 1))
    step_val = max(0, t - 1)
    step = torch.tensor([step_val], dtype=torch.int64)

    # 如果指定了设备，将张量移到对应设备
    if device:
        var_ref = var_ref.to(device)
        m_ref = m_ref.to(device)
        v_ref = v_ref.to(device)
        max_grad_norm_ref = max_grad_norm_ref.to(device)
        grad = grad.to(device)
        step = step.to(device)

    # 参数配置
    lr = float(param.get('lr', '0.01'))
    beta1 = float(param.get('beta1', '0.9'))
    beta2 = float(param.get('beta2', '0.99'))
    weight_decay = float(param.get('weight_decay', '5e-3'))
    eps = float(param.get('eps', '1e-6'))
    amsgrad = parse_bool_param(param.get('amsgrad', False))
    maximize = parse_bool_param(param.get('maximize', False))

    return var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize

def custom_check_precision(row, ref_output, test_output):
    """低精度与多输出：略放宽阈值；Adam 多步累加在 BF16/FP16 上绝对误差易偏大。"""
    dtype_str = str(row.get("dtype", "float32")).lower()
    if dtype_str in ("float16", "half", "bfloat16", "bf16"):
        max_abs, max_rel = 5e-2, 5e-2
    else:
        max_abs, max_rel = 1e-2, 1e-2
    from framework.utils import check_precision

    ref_f = [t.float().cpu() for t in ref_output]
    test_f = [t.float().cpu() for t in test_output]
    return check_precision(ref_f, test_f, max_abs, max_rel)


def get_init_inputs(param, device=None):
    """
    提取模型初始化参数。ApplyAdamWV2 不需要特殊的初始化参数。
    """
    return []  # 不需要特殊的初始化输入