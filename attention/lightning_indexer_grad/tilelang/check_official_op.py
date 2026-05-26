#!/usr/bin/env python3
"""
检查官方 lightning_indexer_grad 算子的参数和用法
"""

import torch
import torch_npu

print("=" * 70)
print("检查官方 torch_npu 算子")
print("=" * 70)

# 检查基本加载
print("\n1. torch_npu 版本:")
print(f"   torch_npu 版本: {torch_npu.__version__ if hasattr(torch_npu, '__version__') else '未知'}")

# 检查目标算子
print("\n2. 检查目标算子是否存在:")
target_ops = [
    'npu_sparse_lightning_indexer_grad_kl_loss',
    'npu_sparse_lightning_indexer_grad',
    'lightning_indexer_grad',
]

for op_name in target_ops:
    exists = hasattr(torch_npu, op_name)
    print(f"   {op_name}: {'✓ 存在' if exists else '✗ 不存在'}")

# 检查所有 lightning 相关算子
print("\n3. 所有 lightning 相关算子:")
lightning_ops = [name for name in dir(torch_npu) if 'lightning' in name.lower()]
if lightning_ops:
    for op in lightning_ops:
        print(f"   - {op}")
else:
    print("   未找到 lightning 相关算子")

# 检查所有 sparse 相关算子
print("\n4. 所有 sparse 相关算子:")
sparse_ops = [name for name in dir(torch_npu) if 'sparse' in name.lower() and not name.startswith('_')]
if sparse_ops:
    for op in sparse_ops[:20]:  # 只显示前20个
        print(f"   - {op}")
else:
    print("   未找到 sparse 相关算子")

# 尝试获取算子签名
print("\n5. 尝试获取算子签名:")
target = 'npu_sparse_lightning_indexer_grad_kl_loss'
if hasattr(torch_npu, target):
    try:
        import inspect
        func = getattr(torch_npu, target)
        sig = inspect.signature(func)
        print(f"   {target} 签名: {sig}")
    except Exception as e:
        print(f"   无法获取签名: {e}")
        
        # 尝试直接调用看错误信息
        try:
            print(f"\n   尝试调用（预期失败，获取参数信息）:")
            func()
        except TypeError as e:
            print(f"   TypeError: {e}")
        except Exception as e:
            print(f"   Error: {e}")

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)