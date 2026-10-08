#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
kimik3 场景正反向串联性能测试
测试 causal_conv1d_fn (前向) + causal_conv1d_grad (反向) 的端到端性能
"""

import torch
import torch_npu
import time
import sys
from pathlib import Path

# 添加项目路径
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

def test_forward_only():
    """测试前向算子可用性"""
    print("=" * 80)
    print("阶段 1: 测试前向算子")
    print("=" * 80)

    try:
        from torch_extension.causal_conv1d_fn import _causal_conv1d_fn_op_builder
        print(f"✅ 前向算子已加载: {_causal_conv1d_fn_op_builder.name}")
    except Exception as e:
        print(f"❌ 前向算子加载失败: {e}")
        return False

    # kimik3 场景参数
    B, T, dim = 2, 512, 12288
    kW = 4
    dtype = torch.float16
    device = "npu:0"

    print(f"\n测试配置: B={B}, T={T}, dim={dim}, kW={kW}, dtype={dtype}")

    try:
        # 准备输入
        x = torch.randn(B, T, dim, dtype=dtype, device=device)
        weight = torch.randn(kW, dim, dtype=dtype, device=device)
        bias = torch.randn(dim, dtype=dtype, device=device)
        conv_states = torch.zeros(B, kW-1, dim, dtype=dtype, device=device)
        query_start_loc = torch.tensor([0, T, 2*T], dtype=torch.int32, device=device)

        # 前向推理
        y = torch.ops.ascend_ops.causal_conv1d_fn(
            x, weight, bias, conv_states, query_start_loc,
            activation="silu"
        )

        print(f"✅ 前向推理成功")
        print(f"   输入 shape: {x.shape}")
        print(f"   输出 shape: {y.shape}")
        print(f"   输出范围: [{y.min():.4f}, {y.max():.4f}]")

        return True

    except Exception as e:
        print(f"❌ 前向推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_binding():
    """测试反向算子绑定"""
    print("\n" + "=" * 80)
    print("阶段 2: 测试反向算子绑定")
    print("=" * 80)

    # 检查反向算子是否已注册
    try:
        # 尝试调用反向算子（通过 torch.ops）
        if hasattr(torch.ops, 'ascend_ops') and hasattr(torch.ops.ascend_ops, 'causal_conv1d_grad'):
            print("✅ 反向算子已注册: torch.ops.ascend_ops.causal_conv1d_grad")
            return True
        else:
            print("❌ 反向算子未注册")
            print("   需要编译并安装 causal_conv1d_grad 算子")
            return False
    except Exception as e:
        print(f"❌ 反向算子检查失败: {e}")
        return False

def test_autograd_chain():
    """测试 autograd 链路（前向→反向）"""
    print("\n" + "=" * 80)
    print("阶段 3: 测试 autograd 链路")
    print("=" * 80)

    # kimik3 场景参数
    B, T, dim = 2, 512, 12288
    kW = 4
    dtype = torch.float16
    device = "npu:0"

    print(f"\n测试配置: B={B}, T={T}, dim={dim}, kW={kW}, dtype={dtype}")

    try:
        # 准备输入（requires_grad=True）
        x = torch.randn(B, T, dim, dtype=dtype, device=device, requires_grad=True)
        weight = torch.randn(kW, dim, dtype=dtype, device=device, requires_grad=True)
        bias = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)
        conv_states = torch.zeros(B, kW-1, dim, dtype=dtype, device=device)
        query_start_loc = torch.tensor([0, T, 2*T], dtype=torch.int32, device=device)

        # 前向推理
        y = torch.ops.ascend_ops.causal_conv1d_fn(
            x, weight, bias, conv_states, query_start_loc,
            activation="silu"
        )

        print(f"✅ 前向推理成功")
        print(f"   输出 requires_grad: {y.requires_grad}")

        if not y.requires_grad:
            print("❌ 输出不支持梯度回传（autograd 未绑定）")
            return False

        # 反向传播
        loss = y.sum()
        loss.backward()

        # 检查梯度
        print(f"✅ 反向传播成功")
        print(f"   dx 存在: {x.grad is not None}")
        print(f"   dweight 存在: {weight.grad is not None}")
        print(f"   dbias 存在: {bias.grad is not None}")

        if x.grad is not None:
            print(f"   dx 范围: [{x.grad.min():.6f}, {x.grad.max():.6f}]")
        if weight.grad is not None:
            print(f"   dweight 范围: [{weight.grad.min():.6f}, {weight.grad.max():.6f}]")
        if bias.grad is not None:
            print(f"   dbias 范围: [{bias.grad.min():.6f}, {bias.grad.max():.6f}]")

        return True

    except Exception as e:
        print(f"❌ autograd 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_forward_backward():
    """性能基准测试（正反向串联）"""
    print("\n" + "=" * 80)
    print("阶段 4: 性能基准测试")
    print("=" * 80)

    # kimik3 场景参数矩阵
    test_configs = [
        (1, 512, 12288),
        (1, 1024, 12288),
        (1, 2048, 12288),
        (1, 4096, 12288),
        (1, 8192, 12288),
        (2, 512, 12288),
        (2, 1024, 12288),
        (2, 2048, 12288),
        (4, 512, 12288),
        (4, 1024, 12288),
    ]

    kW = 4
    dtype = torch.float16
    device = "npu:0"
    warmup = 10
    repeat = 50

    print(f"\n性能测试配置: kW={kW}, dtype={dtype}, warmup={warmup}, repeat={repeat}\n")
    print(f"{'Batch':<6} {'SeqLen':<7} {'Dim':<7} {'Forward(us)':<13} {'Backward(us)':<14} {'Total(us)':<11}")
    print("-" * 80)

    for B, T, dim in test_configs:
        try:
            # 准备输入
            x = torch.randn(B, T, dim, dtype=dtype, device=device, requires_grad=True)
            weight = torch.randn(kW, dim, dtype=dtype, device=device, requires_grad=True)
            bias = torch.randn(dim, dtype=dtype, device=device, requires_grad=True)
            conv_states = torch.zeros(B, kW-1, dim, dtype=dtype, device=device)
            query_start_loc = torch.arange(0, (B+1)*T, T, dtype=torch.int32, device=device)

            # Warmup
            for _ in range(warmup):
                y = torch.ops.ascend_ops.causal_conv1d_fn(
                    x, weight, bias, conv_states, query_start_loc, activation="silu"
                )
                loss = y.sum()
                loss.backward()
                x.grad = None
                weight.grad = None
                bias.grad = None

            torch_npu.synchronize()

            # 测试前向
            start = time.perf_counter()
            for _ in range(repeat):
                y = torch.ops.ascend_ops.causal_conv1d_fn(
                    x, weight, bias, conv_states, query_start_loc, activation="silu"
                )
            torch_npu.synchronize()
            fwd_time = (time.perf_counter() - start) / repeat * 1e6

            # 测试反向
            loss = y.sum()
            start = time.perf_counter()
            for _ in range(repeat):
                loss.backward(retain_graph=True)
            torch_npu.synchronize()
            bwd_time = (time.perf_counter() - start) / repeat * 1e6

            total_time = fwd_time + bwd_time

            print(f"{B:<6} {T:<7} {dim:<7} {fwd_time:<13.2f} {bwd_time:<14.2f} {total_time:<11.2f}")

        except Exception as e:
            print(f"{B:<6} {T:<7} {dim:<7} ERROR: {e}")

def main():
    """主测试流程"""
    print("kimik3 场景正反向串联性能测试")
    print("=" * 80)

    # 阶段 1: 测试前向
    if not test_forward_only():
        print("\n❌ 前向算子不可用，退出测试")
        return 1

    # 阶段 2: 检查反向算子
    backward_available = test_backward_binding()

    # 阶段 3: 测试 autograd 链路
    if backward_available:
        if not test_autograd_chain():
            print("\n⚠️  autograd 链路未打通，只能测试前向性能")
            backward_available = False

    # 阶段 4: 性能测试
    if backward_available:
        benchmark_forward_backward()
    else:
        print("\n" + "=" * 80)
        print("当前状态: 仅前向算子可用")
        print("下一步: 编译并安装 causal_conv1d_grad 反向算子")
        print("=" * 80)
        print("\n编译步骤:")
        print("1. cd /workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/ops-registry-invoke/operators/causal_conv1d_grad")
        print("2. bash build.sh")
        print("3. 重新运行本测试脚本")

    print("\n✅ 测试完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())
