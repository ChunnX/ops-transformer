import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, struct, math
torch_npu.npu.set_device(0); torch.manual_seed(42); np.random.seed(42)
q = torch.randn(8192, 64, 128, dtype=torch.bfloat16).npu()
w = torch.randn(8192, 64, dtype=torch.bfloat16).npu()
k = torch.randn(10591, 1024, 1, 128, dtype=torch.bfloat16).npu()
bt = torch.tensor([[i for i in range(8)]], dtype=torch.int32).npu()
asq = torch.tensor([8192.0], dtype=torch.int32).npu()
ask = torch.tensor([8192.0], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(q, k, w, actual_seq_lengths_query=asq, actual_seq_lengths_key=ask, block_table=bt, layout_query="TND", layout_key="PA_BSND", sparse_count=512, sparse_mode=3)
torch.npu.synchronize()

vals = r[600, 0].cpu().tolist()

print("=== ProcessVec debug for cuS1Idx=600 ===")
# First 11 values are int32
labels_int = ['cuS1Idx', 'cuBaseS2Idx', 'cuRealAcSeq', 'radixCuRealAcSeq', 'cuS2Len', 'radixCuS2Len', 'actS1Size', 'actS2Size', 'cuS1BeginIdxPerAiv', 'innerS1Idx', 'cuS1ProcNumPerAiv']
for i in range(11):
    print(f"  {labels_int[i]:25s} = {vals[i]}")

# Values 11~20 are reduceOutInner[0..9] as float bits
print("\n=== reduceOutInner[0..9] ===")
for i in range(10):
    f = struct.unpack('f', struct.pack('i', vals[11+i]))[0]
    print(f"  reduceOutInner[{i}] = {f:.4f}")

# Values 21~24 are reduceOutInner[0,87,88,511]
print("\n=== reduceOutInner key positions (before DataCopyPad) ===")
for i, idx in enumerate([0, 87, 88, 511]):
    f = struct.unpack('f', struct.pack('i', vals[21+i]))[0]
    tag = "INVALID" if f > 1e29 else ""
    print(f"  reduceOutInner[{idx}] = {f:.4f}  {tag}")

print("\n=== GM readback after DataCopyPad ===")
for i, idx in enumerate([0, 87, 88, 511]):
    f = struct.unpack('f', struct.pack('i', vals[21+i]))[0]
    tag = "INVALID" if f > 1e29 else ""
    print(f"  GM[600*1024+{idx}] = {f:.4f}  {tag}")
