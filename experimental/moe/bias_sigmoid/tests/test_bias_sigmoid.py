import logging
import torch
import torch_npu
import npu_ops_transformer_ext

logging.basicConfig(level=logging.INFO)

ROWS = 1024
COLS = 254
BLOCK_DIM = 40
EPSILON = 1e-5
supported_dtypes = {torch.bfloat16, torch.float16}

for dtype in supported_dtypes:
    logging.info(f"start testing dtype : {'BF16' if dtype == torch.bfloat16 else 'HALF'}")
    expertX = torch.randint(1, 10, (ROWS, COLS)).to(dtype)
    expertBias = torch.randint(1, 10, (COLS,)).to(dtype)

    expertX_fp32 = expertX.float()
    expertBias_fp32 = expertBias.float()
    sigmoid_out_cpu = torch.sigmoid(expertX_fp32)
    bias_out_cpu = sigmoid_out_cpu + expertBias_fp32.unsqueeze(0)

    sigmoid_out_cpu = sigmoid_out_cpu.to(dtype)
    bias_out_cpu = bias_out_cpu.to(dtype)

    sigmoid_out_npu = torch.empty_like(expertX).npu()
    bias_out_npu = torch.empty_like(expertX).npu()
    torch.ops.npu_ops_transformer_ext.bias_sigmoid(BLOCK_DIM, expertX.npu(), expertBias.npu(), sigmoid_out_npu, bias_out_npu)

    sigmoid_abs_error = torch.abs(sigmoid_out_npu.cpu() - sigmoid_out_cpu)
    sigmoid_rel_error = sigmoid_abs_error / (torch.abs(sigmoid_out_npu.cpu()) + EPSILON)

    bias_abs_error = torch.abs(bias_out_npu.cpu() - bias_out_cpu)
    bias_rel_error = bias_abs_error / (torch.abs(bias_out_npu.cpu()) + EPSILON)

    rotl, atol = torch.testing._comparison.get_tolerances(dtype, rtol=None, atol=None)
    logging.info(f"compare CPU Result vs NPU Sigmoid Result: {torch.allclose(sigmoid_out_cpu, sigmoid_out_npu, rotl, atol)}\n\n")
    logging.info(f"compare CPU Result vs NPU Bias Result: {torch.allclose(bias_out_cpu, bias_out_npu, rotl, atol)}\n\n")