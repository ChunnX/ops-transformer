import torch
import torch_npu
import ascend_ops
import logging
import math

logging.basicConfig(level=logging.INFO)

# Configuration
gbH = 4
gbW = 64
blockDim = 4
dtype = torch.float16  # NPU kernel expects float16 input

logging.info(f"Testing gategelu_quant kernel")
logging.info(f"Configuration: gbH={gbH}, gbW={gbW}, blockDim={blockDim}, dtype={dtype}")


# CPU reference implementation
# NOTE: This kernel computes: out = clamp(gelu(in_one) * in_two * scale, clamp_val)
# where in_one = input[:, 0:gbW/2], in_two = input[:, gbW/2:gbW]
def gategelu_quant_cpu(input, scale, clamp=True, clamp_val=128.0):
    """
    CPU implementation of gategelu_quant for verification
    - input: (gbH, gbW) float16, split into in_one and in_two
    - scale: (gbW/2,) float32
    - output: (gbH, gbW/2) int8
    """
    output = torch.empty(gbH, gbW // 2, dtype=torch.int8)
    in_one = input[:, 0:gbW // 2]  # First half
    in_two = input[:, gbW // 2:]   # Second half

    for h in range(gbH):
        for w in range(gbW // 2):
            # GeLU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            x = in_one[h, w].item()
            gelu = 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

            # Multiply with in_two (gate)
            gated = gelu * in_two[h, w].item()

            # Scale
            s = scale[w].item()
            scaled = gated * s

            # Clamp before quantize
            if clamp:
                scaled = max(-clamp_val, min(clamp_val, scaled))

            # Clamp to int8 range to avoid overflow
            scaled = max(-128.0, min(127.0, scaled))

            # Quantize to int8
            output[h, w] = int(round(scaled))

    return output


# Test with CPU reference comparison
torch.manual_seed(42)
input = torch.randn(gbH, gbW, dtype=dtype) * 10
scale = torch.ones(gbW // 2, dtype=torch.float32) * 10  # kernel expects float32 scale

input_npu = input.npu()
scale_npu = scale.npu()
out_npu = torch.empty(gbH, gbW // 2, dtype=torch.int8).npu()

torch.ops.ascend_ops.gategelu_quant(blockDim, input_npu, scale_npu, out_npu, False, 128.0)

torch.npu.empty_cache()

# CPU result
out_cpu = gategelu_quant_cpu(input, scale, clamp=False, clamp_val=128.0)

logging.info(f"------------------------------------------Result Comparison-------------------------------------------------------------------------------")
logging.info(f"out_npu shape: {out_npu.cpu().shape}")
logging.info(f"out_cpu shape: {out_cpu.shape}")
logging.info(f"out_npu has non-zero values: {(out_npu.cpu() != 0).any().item()}")
logging.info(f"out_cpu has non-zero values: {(out_cpu != 0).any().item()}")

# Compare results
out_npu_cpu = out_npu.cpu()
match = (out_npu_cpu == out_cpu).all().item()
logging.info(f"out_npu sample: {out_npu_cpu[0, :16]}")
logging.info(f"out_cpu sample: {out_cpu[0, :16]}")
logging.info(f"All values match: {match}")

# Calculate absolute error (int8 quantization difference)
abs_error = (out_npu_cpu.float() - out_cpu.float()).abs()
logging.info(f"------------------------------------------Absolute Error-------------------------------------------------------------------------------")
logging.info(f"Max absolute error: {abs_error.max().item()}")
logging.info(f"Mean absolute error: {abs_error.mean().item()}")

# Calculate relative error based on floating point values before quantization
# We recompute the floating point values from CPU using float32 to match NPU precision
in_one = input[:, 0:gbW // 2].float()  # First half, convert to float32
in_two = input[:, gbW // 2:].float()   # Second half, convert to float32
scale_f = scale.float()  # float32

# GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
x = in_one
gelu = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
gated = gelu * in_two
scaled = gated * scale_f

# Clamp disabled in this test
out_cpu_float = scaled  # float32, matches NPU precision

# NPU output as float for comparison
out_npu_float = out_npu_cpu.float()

# Calculate relative error using floating point values
# Only consider non-zero CPU values (avoid division by zero)
valid_mask = out_cpu_float.abs() > 1e-6
rel_error = torch.zeros_like(abs_error)
rel_error[valid_mask] = abs_error[valid_mask] / (out_cpu_float[valid_mask].abs() + 1e-8)

logging.info(f"------------------------------------------Relative Error (Float)-------------------------------------------------------------------------------")
logging.info(f"Max relative error: {rel_error.max().item():.8f}")
logging.info(f"Mean relative error: {rel_error[valid_mask].mean().item():.8f}")
logging.info(f"Valid samples for relative error: {valid_mask.sum().item()} / {valid_mask.numel()}")

logging.info(f"------------------------------------------Test Result-------------------------------------------------------------------------------")
# Quantization operator precision standards:
# - Max absolute error <= 1 (int8 quantization error)
# - Max relative error <= 0.01 (1%) for non-zero values
abs_passed = abs_error.max().item() <= 1
rel_passed = rel_error.max().item() <= 0.01  # 1% relative error threshold
test_passed = abs_passed and rel_passed
logging.info(f"Absolute error check (max <= 1): {abs_passed}")
logging.info(f"Relative error check (max <= 0.01): {rel_passed}")
logging.info(f"CPU comparison test PASSED: {test_passed}")
if not test_passed:
    if not abs_passed:
        logging.info(f"ERROR: Max absolute error {abs_error.max().item()} exceeds tolerance")
    if not rel_passed:
        logging.info(f"ERROR: Max relative error {rel_error.max().item():.8f} exceeds 0.01")
else:
    logging.info(f"NOTE: All checks passed (quantization error within tolerance)")