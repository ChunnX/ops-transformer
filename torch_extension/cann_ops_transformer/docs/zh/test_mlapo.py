import torch
import torch_npu
from cann_ops_transformer.ops.mla_preprocess import mla_preprocess

torch_npu.npu.set_device(0)
dtype = torch.bfloat16

tokenNum, hiddenSize = 8, 7168
headNum, nopeDim = 32, 128
keyTotalDim, kRopeDim = 576, 64
blockNum, blockSize = 192, 128

# 准备输入
inp = torch.randn(tokenNum, hiddenSize, dtype=dtype, device="npu")
g0 = torch.randn(hiddenSize, dtype=dtype, device="npu") + 1.0
b0 = torch.randn(hiddenSize, dtype=dtype, device="npu") * 0.1
qs0 = torch.randn(1, dtype=dtype, device="npu").abs() + 0.005
qo0 = torch.zeros(1, dtype=torch.int8, device="npu")

# 权重必须为 FRACTAL_NZ 格式
wdq = torch.randint(-3, 3, (1536 + keyTotalDim, hiddenSize), dtype=torch.int8)
wdq_nz = torch_npu.empty_with_format(wdq.shape, dtype=torch.int8, device="npu",
                                      acl_format=torch_npu.Format.FRACTAL_NZ)
wdq_nz.copy_(wdq)
ds0 = torch.randn(1536 + keyTotalDim, dtype=torch.float32, device="npu").abs() + 0.001
bi0 = torch.randint(-2, 2, (1536 + keyTotalDim,), dtype=torch.int32, device="npu")

g1 = torch.randn(1536, dtype=dtype, device="npu") + 1.0
b1 = torch.randn(1536, dtype=dtype, device="npu") * 0.1
qs1 = torch.randn(1, dtype=dtype, device="npu").abs() + 0.005
qo1 = torch.zeros(1, dtype=torch.int8, device="npu")

wu = torch.randint(-3, 3, (headNum * (nopeDim + kRopeDim), 1536), dtype=torch.int8)
wu_nz = torch_npu.empty_with_format(wu.shape, dtype=torch.int8, device="npu",
                                     acl_format=torch_npu.Format.FRACTAL_NZ)
wu_nz.copy_(wu)
ds1 = torch.randn(headNum * (nopeDim + kRopeDim), dtype=torch.float32, device="npu").abs() + 0.001
bi1 = torch.randint(-2, 2, (headNum * (nopeDim + kRopeDim),), dtype=torch.int32, device="npu")

g2 = torch.randn(512, dtype=dtype, device="npu") + 1.0
cos = torch.randn(tokenNum, kRopeDim, dtype=dtype, device="npu")
sin = torch.randn(tokenNum, kRopeDim, dtype=dtype, device="npu")
wk = torch.randn(headNum, nopeDim, 512, dtype=dtype, device="npu")

# kvCache: [blockNum, blockSize, 1, keyTotalDim]
kv_cache = torch.zeros(blockNum, blockSize, 1, keyTotalDim, dtype=dtype, device="npu")
slot_mapping = torch.tensor([2350, 23251, 11000, 13627, 21896, 6332, 2534, 11263],
                            dtype=torch.int32, device="npu")
kv_dim = keyTotalDim - kRopeDim  # = 512

kv_cache = torch.zeros(blockNum, blockSize, 1, kv_dim, dtype=dtype, device="npu")
kp_cache = torch.zeros(blockNum, blockSize, 1, kRopeDim, dtype=dtype, device="npu")
import math

# 构造交替存储的 key/value cache，模拟分页注意力中的内存布局
hidden = blockSize * 1 * kv_dim        # = 65536
hidden_rope = blockSize * 1 * kRopeDim  # = 8192

# kvCache：分配双倍空间，用 as_strided 在相邻 block 间制造间隙
buf = torch.zeros(2 * blockNum * hidden, dtype=dtype, device="npu")
kv = buf.view(2, blockNum, blockSize, 1, kv_dim)
kv.as_strided_(size=kv.shape, stride=(hidden, 2 * hidden, *kv.stride()[2:]))
kv_cache_nc, _ = kv.unbind(0)

# kvCacheRope：同理
buf_rope = torch.zeros(2 * blockNum * hidden_rope, dtype=dtype, device="npu")
kvr = buf_rope.view(2, blockNum, blockSize, 1, kRopeDim)
kvr.as_strided_(size=kvr.shape, stride=(hidden_rope, 2 * hidden_rope, *kvr.stride()[2:]))
kp_cache_nc, _ = kvr.unbind(0)

print(f"kvCache is_contiguous: {kv_cache_nc.is_contiguous()}")     # False
print(f"kvCache stride[0]:    {kv_cache_nc.stride(0)}")            # 131072（2 × 65536）

kv_cache_nz = torch_npu.empty_with_format(
    (blockNum, 16, blockSize, 32), dtype=torch.int8, device="npu",
    acl_format=torch_npu.Format.FRACTAL_NZ)
kp_cache_nz = torch_npu.empty_with_format(
    (blockNum, 4, blockSize, 16), dtype=dtype, device="npu",
    acl_format=torch_npu.Format.FRACTAL_NZ)

ctkv_scale = torch.randn(1, dtype=dtype, device="npu").abs() + 0.01
qnope_scale = torch.randn(headNum, dtype=dtype, device="npu").abs() + 0.001

q_out, kv_out, qr_out, kr_out = mla_preprocess(
    inp, g0, b0, qs0, qo0,
    wdq_nz, ds0, bi0,
    g1, b1, qs1, qo1,
    wu_nz, ds1, bi1,
    g2, cos, sin, wk,
    kv_cache_nz, slot_mapping,
    kv_cache_rope=kp_cache_nz,
    ctkv_scale=ctkv_scale,
    q_nope_scale=qnope_scale,
    cache_mode=2,
)
torch.npu.synchronize()

print(f"qOut:  {q_out.shape}")   # [8, 32, 512]，dtype=int8
print(f"kvOut: {kv_out.shape}")  # [192, 16, 128, 32]，dtype=int8
print(f"krOut: {kr_out.shape}")  # [192, 4, 128, 16]，dtype=bfloat16