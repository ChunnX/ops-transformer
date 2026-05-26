#!/usr/bin/env python3
"""
检查官方 npu_lightning_indexer_grad 算子的详细参数
"""

import torch
import torch_npu

print("=" * 70)
print("检查官方 npu_lightning_indexer_grad 算子")
print("=" * 70)

# 检查算子存在
print("\n1. 确认算子存在:")
print(f"   npu_lightning_indexer_grad: {'✓ 存在' if hasattr(torch_npu, 'npu_lightning_indexer_grad') else '✗ 不存在'}")

# 尝试获取帮助信息
print("\n2. 尝试获取帮助信息:")
try:
    help(torch_npu.npu_lightning_indexer_grad)
except Exception as e:
    print(f"   无法获取帮助: {e}")

# 尝试错误调用获取参数信息
print("\n3. 通过错误调用获取参数信息:")
try:
    # 尝试空参数调用
    result = torch_npu.npu_lightning_indexer_grad()
except TypeError as e:
    print(f"   TypeError (空参数): {e}")
except RuntimeError as e:
    print(f"   RuntimeError: {e}")

try:
    # 尝试单参数调用
    result = torch_npu.npu_lightning_indexer_grad(torch.randn(1, 10, 64, 128))
except TypeError as e:
    print(f"   TypeError (单参数): {e}")
except RuntimeError as e:
    print(f"   RuntimeError: {e}")

# 查看源代码（如果能找到）
print("\n4. 尝试查看算子定义:")
try:
    import inspect
    source = inspect.getsource(torch_npu.npu_lightning_indexer_grad)
    print(source[:500] if source else "无法获取源代码")
except Exception as e:
    print(f"   无法获取源代码: {e}")

# 检查是否有文档字符串
print("\n5. 检查文档字符串:")
doc = torch_npu.npu_lightning_indexer_grad.__doc__
if doc:
    print(f"   {doc}")
else:
    print("   无文档字符串")

print("\n" + "=" * 70)