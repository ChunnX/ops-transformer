import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the QuantizedBatchNorm operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1, 1, 1, 1]'))
    num_features = param.get('num_features', 1)
    dtype_str = param.get('dtype', 'int8')
    
    # Quantized input
    if dtype_str == 'int8':
        input_dtype = torch.int8
        quant_min, quant_max = -128, 127
    elif dtype_str == 'uint8':
        input_dtype = torch.uint8
        quant_min, quant_max = 0, 255
    else:
        input_dtype = torch.int32
        quant_min, quant_max = -2147483648, 2147483647
    
    # Generate quantized input
    x = torch.randint(quant_min, quant_max + 1, input_shape, device=device, dtype=input_dtype)
    
    # Running statistics (float32)
    mean = torch.rand(num_features, device=device, dtype=torch.float32)
    var = torch.rand(num_features, device=device, dtype=torch.float32) + 1e-3
    
    # Quantization parameters - ensure they are created on the correct device and are contiguous
    input_scale = torch.tensor([0.1], device=device, dtype=torch.float32).contiguous()
    input_zero_point = torch.tensor([0], device=device, dtype=torch.int32).contiguous()
    output_scale = torch.tensor([0.1], device=device, dtype=torch.float32).contiguous()
    output_zero_point = torch.tensor([0], device=device, dtype=torch.int32).contiguous()
    
    # Optional weight and bias
    affine = param.get('affine', True)
    weight = None
    bias = None
    if affine:
        weight = torch.rand(num_features, device=device, dtype=torch.float32)
        bias = torch.rand(num_features, device=device, dtype=torch.float32)

    return (x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point, weight, bias)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (num_features, eps) for the model.
    """
    num_features = param.get('num_features', 1)
    eps = float(param.get('epsilon', 1e-5))
    
    return [num_features, eps]


def custom_check_precision(param, outputs, outputs_new):
    """Quantized output: allow atol=1 for rounding difference."""
    out = outputs[0].int() if outputs[0].dtype in (torch.int8, torch.uint8) else outputs[0]
    out_new = outputs_new[0].int() if outputs_new[0].dtype in (torch.int8, torch.uint8) else outputs_new[0]
    abs_diff = torch.abs(out - out_new)
    rel_diff = abs_diff.float() / (torch.abs(out).float() + 1e-7)
    is_pass = 1 if torch.all(abs_diff <= 1) else 0
    return is_pass, abs_diff.flatten(), rel_diff.flatten()
