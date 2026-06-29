import torch
import torch_npu
import torchair
import torch.nn as nn
import npu_ops_transformer
import pandas as pd

AIC_CORE_NUM = 36
AIV_CORE_NUM = 72
FA_METADATA_SIZE = 8
FD_METADATA_SIZE = 8

qSeqLens = torch.tensor([384], dtype=torch.int32).npu()
kvSeqLens = torch.tensor([32768], dtype=torch.int32).npu()
qHeads = 1
kvHeads = 1
dimQkflat = 128
stemBlockSize = 128
causal=True
windowSize=4

metadata = torch.ops.npu_ops_transformer.npu_stem_indexer_metadata(
    q_seq_lens=qSeqLens,
    kv_seq_lens=kvSeqLens,
    q_heads=qHeads,
    kv_heads=kvHeads,
    causal=causal,
    dim_qkflat=dimQkflat,
    stem_block_size=stemBlockSize,
    window_size=windowSize
)

if isinstance(metadata, torch.Tensor):
    metadata_np = metadata.cpu().numpy().flatten()
else:
    metadata_np = np.array(metadata).flatten()

fa_data = [[0 for _ in range(FA_METADATA_SIZE)] for _ in range(AIC_CORE_NUM)]
fd_data = [[0 for _ in range(FD_METADATA_SIZE)] for _ in range(AIV_CORE_NUM)]

# 按8元素一组解析aic部分
fa_length = AIC_CORE_NUM * FA_METADATA_SIZE
fd_length = AIV_CORE_NUM * FD_METADATA_SIZE
if len(metadata_np) < fa_length + fd_length:
    raise ValueError(f"算子输出数据长度({len(metadata_np)})不足，期望至少{fa_length} + {fd_length}")

for i in range(AIC_CORE_NUM):
    for j in range(FA_METADATA_SIZE):
        fa_data[i][j] = metadata_np[i * FA_METADATA_SIZE + j]

for i in range(AIV_CORE_NUM):
    for j in range(FD_METADATA_SIZE):
        fd_data[i][j] = metadata_np[fa_length + i * FD_METADATA_SIZE + j]

print("=============================== FA =========================")
for aic in range (AIC_CORE_NUM):
    print(fa_data[aic])
print("=============================== FD =========================")
for aiv in range (AIV_CORE_NUM):
    print(fd_data[aiv])