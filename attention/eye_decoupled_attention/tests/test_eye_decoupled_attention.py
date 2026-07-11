import torch
import torch_npu
from ascend_ops.pytorch_binding import EyeDecoupledAttention

def main():
    print("Running EyeDecoupledAttention Example on Ascend NPU...")
    x = torch.randn(1, 432, 64, dtype=torch.float16).npu()
    pos = torch.randn(1, 432, 64, dtype=torch.float16).npu()
    attn = EyeDecoupledAttention(64, hb=12, wb=4, bn_h=3, bn_w=3).npu()
    
    y = attn(x, pos)
    print(f"Output shape: {y.shape}")
    print("Example executed successfully!")

if __name__ == "__main__":
    main()
