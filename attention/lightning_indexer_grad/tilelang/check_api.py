#!/usr/bin/env python3
"""
检查官方 lightning_indexer_grad 算子的参数和用法
"""

import torch
import torch_npu

print("=" * 70)
print("检查官方 lightning_indexer_grad 算子")
print("=" * 70)

# 检查找到的算子
print("\n找到的算子:")
print("  - npu_lightning_indexer_grad ✓")
print("  - npu_sparse_lightning_indexer_grad_kl_loss ✓")

# 尝试获取 npu_lightning_indexer_grad 的签名
print("\n检查 npu_lightning_indexer_grad 参数:")
try:
    import inspect
    func = torch_npu.npu_lightning_indexer_grad
    # 尝试直接调用看错误信息
    try:
        result = func()
    except TypeError as e:
        print(f"  参数信息 (从错误中提取): {e}")
    except Exception as e:
        print(f"  其他错误: {e}")
except Exception as e:
    print(f"  检查失败: {e}")

# 尝试获取 npu_sparse_lightning_indexer_grad_kl_loss 的签名
print("\n检查 npu_sparse_lightning_indexer_grad_kl_loss 参数:")
try:
    func = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss
    try:
        result = func()
    except TypeError as e:
        print(f"  参数信息 (从错误中提取): {e}")
    except Exception as e:
        print(f"  其他错误: {e}")
except Exception as e:
    print(f"  检查失败: {e}")

print("\n" + "=" * 70)
print("完成")
print("=" * 70)