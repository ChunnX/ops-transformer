import torch
import torch_npu
from ascend_ops.pytorch_binding import CausalTopoGCN

def main():
    print("Running CausalTopoGCN Example on Ascend NPU...")
    x_gcn = torch.randn(1, 9, 2048, dtype=torch.float16).npu()
    gcn = CausalTopoGCN(2048, 2048, num_classes=3).npu()
    
    logits = gcn(x_gcn)
    print(f"Output logits shape: {logits.shape}")
    print("Example executed successfully!")

if __name__ == "__main__":
    main()
