#!/usr/bin/env python3
"""
生成 C++ aclnn 测试所需的输入数据

关键：对 q 和 k 进行 L2 normalize
"""

import torch
import torch_npu
import numpy as np
import struct
import os

DEVICE_ID = 0
torch_npu.npu.set_device(DEVICE_ID)

# UT 默认参数
B = 2
seqlen = 64
T = B * seqlen  # 128
nk = 4
nv = 4
dk = 64
dv = 64
scale = 1 / (dk ** 0.5)  # 0.125 for dk=64

print(f"参数: B={B}, seqlen={seqlen}, T={T}, nk={nk}, nv={nv}, dk={dk}, dv={dv}")

# 生成随机数据
np.random.seed(42)

q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
beta = torch.rand((T, nv), dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
g = torch.rand((T, nv), dtype=torch.float32, device=f"npu:{DEVICE_ID}") * -1.0

initial_state = torch.rand((B, nv, dv, dk), dtype=torch.bfloat16, device=f"npu:{DEVICE_ID}")
actual_seq_lengths = torch.tensor([seqlen] * B, dtype=torch.int32, device=f"npu:{DEVICE_ID}")

# **关键：L2 normalize**
q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(torch.bfloat16)
k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(torch.bfloat16)

# 保存到文件
output_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(output_dir, "..", "..", "test_inputs.bin")

print(f"\n保存数据到: {data_file}")

# BF16 编码为 uint16
def bf16_to_uint16(tensor):
    if tensor.dtype == torch.float32:
        tensor = tensor.to(torch.bfloat16)
    return tensor.view(torch.int16).short().cpu().numpy()

def float_to_uint32(tensor):
    return tensor.view(torch.int32).int().cpu().numpy()

def bf16_to_float32(tensor):
    return tensor.float()

# 转换数据
q_data = bf16_to_uint16(q)
k_data = bf16_to_uint16(k)
v_data = bf16_to_uint16(v)
beta_data = bf16_to_uint16(beta)
g_data = float_to_uint32(g)
initial_state_data = bf16_to_uint16(initial_state)
seqlens_data = actual_seq_lengths.cpu().numpy()

print(f"数据大小:")
print(f"  q: {q_data.shape} ({q_data.nbytes} bytes)")
print(f"  k: {k_data.shape} ({k_data.nbytes} bytes)")
print(f"  v: {v_data.shape} ({v_data.nbytes} bytes)")
print(f"  beta: {beta_data.shape} ({beta_data.nbytes} bytes)")
print(f"  g: {g_data.shape} ({g_data.nbytes} bytes)")
print(f"  initial_state: {initial_state_data.shape} ({initial_state_data.nbytes} bytes)")
print(f"  seqlens: {seqlens_data.shape} ({seqlens_data.nbytes} bytes)")

# 写入二进制文件
with open(data_file, 'wb') as f:
    # 写入参数信息
    f.write(struct.pack('q', B))          # int64
    f.write(struct.pack('q', T))          # int64
    f.write(struct.pack('q', nk))         # int64
    f.write(struct.pack('q', nv))         # int64
    f.write(struct.pack('q', dk))         # int64
    f.write(struct.pack('q', dv))         # int64
    f.write(struct.pack('f', scale))      # float32
    
    # 写入数据
    f.write(q_data.tobytes())
    f.write(k_data.tobytes())
    f.write(v_data.tobytes())
    f.write(beta_data.tobytes())
    f.write(g_data.tobytes())
    f.write(initial_state_data.tobytes())
    f.write(seqlens_data.tobytes())

print(f"\n文件已生成: {data_file}")
print(f"文件大小: {os.path.getsize(data_file)} bytes")

# 计算并保存 CPU golden 结果
print("\n计算 CPU golden 结果...")

# 导入 golden 实现
from chunk_gated_delta_rule_neumann_golden import cgdr_golden

o_golden, state_golden = cgdr_golden(
    q, k, v, g, beta, scale,
    initial_state, actual_seq_lengths,
    use_float64=False
)

print(f"Golden 输出:")
print(f"  out shape: {o_golden.shape}")
print(f"  state shape: {state_golden.shape}")
print(f"  out range: [{o_golden.min().item():.4f}, {o_golden.max().item():.4f}]")
print(f"  state range: [{state_golden.min().item():.4f}, {state_golden.max().item():.4f}]")

# 保存 golden 结果
golden_file = os.path.join(output_dir, "..", "..", "golden_output.bin")

o_golden_data = bf16_to_uint16(o_golden)
state_golden_data = bf16_to_uint16(state_golden)

with open(golden_file, 'wb') as f:
    f.write(o_golden_data.tobytes())
    f.write(state_golden_data.tobytes())

print(f"\nGolden 文件已生成: {golden_file}")
print(f"Golden 文件大小: {os.path.getsize(golden_file)} bytes")

# 显示部分样本数据
print("\n样本数据（用于 C++ 验证）:")
print(f"q[0:3]: {q_data[0:3]}")
print(f"k[0:3]: {k_data[0:3]}")
print(f"v[0:3]: {v_data[0:3]}")
print(f"out golden[0:3]: {o_golden_data[0:3]}")