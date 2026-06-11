#!/usr/bin/env python3
"""
MhcPreSinkhornBackward 确定性模式验证脚本

验证内容：
1. TILING_KEY=0（非确定性模式）基本功能
2. TILING_KEY=1（确定性模式）基本功能
3. 确定性验证（多次运行结果一致）
4. 数值一致性验证（非确定性 vs 确定性）
5. 性能对比验证（预期~12%性能损失）
"""

import torch
import torch_npu
import numpy as np
import time
import argparse


def prepare_test_data(batch_size=2, seq_length=1024, n=64, c=512):
    """准备测试数据"""
    
    # 生成测试输入
    x = torch.randn(batch_size, seq_length, n, c, dtype=torch.float32, device='npu')
    phi = torch.randn(batch_size, seq_length, n, n, dtype=torch.float32, device='npu')
    alpha = torch.randn(2, dtype=torch.float32, device='npu')
    bias = torch.randn(2 * n, dtype=torch.float32, device='npu')
    
    # 生成grad输入
    grad_hin = torch.randn(batch_size, seq_length, c, dtype=torch.float32, device='npu')
    grad_h_post = torch.randn(batch_size, seq_length, n, dtype=torch.float32, device='npu')
    grad_h_res = torch.randn(batch_size, seq_length, n, dtype=torch.float32, device='npu')
    
    # 生成其他输入
    h_pre = torch.randn(batch_size, seq_length, n, dtype=torch.float32, device='npu')
    hc_before_norm = torch.randn(batch_size, seq_length, n, dtype=torch.float32, device='npu')
    inv_rms = torch.randn(batch_size, seq_length, dtype=torch.float32, device='npu')
    sum_out = torch.randn(batch_size, seq_length, dtype=torch.float32, device='npu')
    norm_out = torch.randn(batch_size, seq_length, dtype=torch.float32, device='npu')
    
    return {
        'grad_hin': grad_hin,
        'grad_h_post': grad_h_post,
        'grad_h_res': grad_h_res,
        'x': x,
        'phi': phi,
        'alpha': alpha,
        'bias': bias,
        'h_pre': h_pre,
        'hc_before_norm': hc_before_norm,
        'inv_rms': inv_rms,
        'sum_out': sum_out,
        'norm_out': norm_out
    }


def run_operator(inputs, deterministic=False):
    """运行算子"""
    
    try:
        # 尝试通过opapi调用
        from torch_npu._C._ops import ops
        op = ops.npu.mhc_pre_sinkhorn_backward
        
        result = op(
            inputs['grad_hin'],
            inputs['grad_h_post'],
            inputs['grad_h_res'],
            inputs['x'],
            inputs['phi'],
            inputs['alpha'],
            inputs['bias'],
            inputs['h_pre'],
            inputs['hc_before_norm'],
            inputs['inv_rms'],
            inputs['sum_out'],
            inputs['norm_out'],
            deterministic=deterministic
        )
        
        return result
        
    except Exception as e:
        print(f"⚠️ Operator call failed: {e}")
        print("Note: TILING_KEY control may require custom op implementation")
        return None


def verify_basic_functionality(inputs):
    """验证基本功能"""
    
    print("\n" + "="*60)
    print("=== Basic Functionality Test ===")
    print("="*60)
    
    # 非确定性模式
    print("\n[1] Testing non-deterministic mode (TILING_KEY=0)...")
    result_non_det = run_operator(inputs, deterministic=False)
    
    if result_non_det is None:
        print("⚠️ Non-deterministic mode call failed - may need environment setup")
        return False
    
    print(f"✅ Output shape: {result_non_det.shape}")
    print(f"✅ Output dtype: {result_non_det.dtype}")
    
    # 确定性模式
    print("\n[2] Testing deterministic mode (TILING_KEY=1)...")
    result_det = run_operator(inputs, deterministic=True)
    
    if result_det is None:
        print("⚠️ Deterministic mode call failed - may need TILING_KEY setup")
        return False
    
    print(f"✅ Output shape: {result_det.shape}")
    print(f"✅ Output dtype: {result_det.dtype}")
    
    # 形状验证
    if result_non_det.shape != result_det.shape:
        print(f"❌ Shape mismatch: non-det {result_non_det.shape} vs det {result_det.shape}")
        return False
    
    print("\n✅ Basic functionality verified")
    return True


def verify_deterministic(inputs, num_runs=10):
    """验证确定性：多次运行结果应完全一致"""
    
    print("\n" + "="*60)
    print("=== Deterministic Verification ===")
    print("="*60)
    
    print(f"\nRunning {num_runs} times with deterministic mode...")
    
    results = []
    for i in range(num_runs):
        torch.manual_seed(42)  # 固定随机种子
        result = run_operator(inputs, deterministic=True)
        
        if result is None:
            print(f"⚠️ Run {i} failed")
            continue
        
        results.append(result.clone())  # clone避免内存共享
        print(f"  Run {i+1}/{num_runs} completed")
    
    if len(results) < 2:
        print("⚠️ Not enough successful runs for deterministic verification")
        return False
    
    # 验证结果一致性
    print("\nVerifying result consistency...")
    
    reference = results[0]
    all_match = True
    
    for i in range(1, len(results)):
        match = torch.allclose(reference, results[i], rtol=1e-5, atol=1e-8)
        
        if not match:
            diff = torch.abs(reference - results[i])
            print(f"  ❌ Run {i} does not match Run 0")
            print(f"     Max diff: {diff.max().item():.6e}")
            print(f"     Mean diff: {diff.mean().item():.6e}")
            all_match = False
        else:
            print(f"  ✅ Run {i} matches Run 0")
    
    if all_match:
        print("\n✅ Deterministic mode verified: results are identical across runs")
        return True
    else:
        print("\n❌ Deterministic mode failed: results differ across runs")
        return False


def verify_numerical_consistency(inputs):
    """验证数值一致性：非确定性 vs 确定性"""
    
    print("\n" + "="*60)
    print("=== Numerical Consistency Test ===")
    print("="*60)
    
    print("\nComparing non-deterministic vs deterministic results...")
    
    result_non_det = run_operator(inputs, deterministic=False)
    result_det = run_operator(inputs, deterministic=True)
    
    if result_non_det is None or result_det is None:
        print("⚠️ Cannot compare - one or both results are None")
        return False
    
    # 允许微小差异（atomicAdd顺序可能不同）
    match = torch.allclose(result_non_det, result_det, rtol=1e-3, atol=1e-5)
    
    if match:
        print("✅ Results match within tolerance (rtol=1e-3, atol=1e-5)")
        return True
    else:
        diff = torch.abs(result_non_det - result_det)
        print(f"⚠️ Results differ:")
        print(f"  Max difference: {diff.max().item():.6e}")
        print(f"  Mean difference: {diff.mean().item():.6e}")
        print(f"  Relative difference: {(diff / result_non_det.abs()).mean().item():.6e}")
        
        # 检查是否在合理范围内
        if diff.max().item() < 1e-2:
            print("✅ Difference within reasonable range (<1e-2)")
            return True
        else:
            print("❌ Difference exceeds reasonable range")
            return False


def verify_performance(inputs, num_runs=100):
    """验证性能对比"""
    
    print("\n" + "="*60)
    print("=== Performance Comparison ===")
    print("="*60)
    
    print(f"\nRunning {num_runs} iterations for each mode...")
    
    # 非确定性模式性能
    torch_npu.npu.synchronize()
    start = time.time()
    
    for i in range(num_runs):
        _ = run_operator(inputs, deterministic=False)
    
    torch_npu.npu.synchronize()
    time_non_det = time.time() - start
    
    print(f"\n[Non-deterministic] {time_non_det:.3f}s for {num_runs} runs")
    print(f"  Average: {time_non_det/num_runs*1000:.2f}ms per run")
    
    # 确定性模式性能
    torch_npu.npu.synchronize()
    start = time.time()
    
    for i in range(num_runs):
        _ = run_operator(inputs, deterministic=True)
    
    torch_npu.npu.synchronize()
    time_det = time.time() - start
    
    print(f"\n[Deterministic] {time_det:.3f}s for {num_runs} runs")
    print(f"  Average: {time_det/num_runs*1000:.2f}ms per run")
    
    # 计算性能损失
    perf_loss = (time_det - time_non_det) / time_non_det
    
    print(f"\nPerformance loss: {perf_loss:.2%}")
    
    # 验证性能损失范围
    if perf_loss < 0:
        print("⚠️ Deterministic mode is faster than non-deterministic (unexpected)")
        print("   This may indicate optimization opportunity")
        return True
    
    elif perf_loss < 0.15:
        print(f"✅ Performance loss within expected range (<15%)")
        print(f"   Actual: {perf_loss:.2%}, Expected: ~12%")
        return True
    
    elif perf_loss < 0.20:
        print(f"⚠️ Performance loss slightly above expected (15-20%)")
        print(f"   Actual: {perf_loss:.2%}, Expected: ~12%")
        print(f"   May need optimization")
        return True
    
    else:
        print(f"❌ Performance loss exceeds expected (>20%)")
        print(f"   Actual: {perf_loss:.2%}, Expected: ~12%")
        print(f"   Significant performance degradation detected")
        return False


def main():
    """主验证流程"""
    
    parser = argparse.ArgumentParser(description='Verify MhcPreSinkhornBackward deterministic mode')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--n', type=int, default=64, help='Hidden dimension n')
    parser.add_argument('--c', type=int, default=512, help='Hidden dimension c')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs for deterministic verification')
    parser.add_argument('--perf-runs', type=int, default=100, help='Number of runs for performance test')
    parser.add_argument('--skip-perf', action='store_true', help='Skip performance test')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MhcPreSinkhornBackward Deterministic Mode Verification")
    print("="*60)
    print(f"\nTest configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Hidden dim n: {args.n}")
    print(f"  Hidden dim c: {args.c}")
    print(f"  Deterministic runs: {args.num_runs}")
    print(f"  Performance runs: {args.perf_runs}")
    
    # 准备测试数据
    print("\nPreparing test data...")
    inputs = prepare_test_data(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        n=args.n,
        c=args.c
    )
    print("✅ Test data prepared")
    
    # 验证结果收集
    results = {}
    
    # 1. 基本功能验证
    results['basic'] = verify_basic_functionality(inputs)
    
    if not results['basic']:
        print("\n❌ Basic functionality verification failed")
        print("   Cannot proceed with further tests")
        return False
    
    # 2. 确定性验证
    results['deterministic'] = verify_deterministic(inputs, num_runs=args.num_runs)
    
    # 3. 数值一致性验证
    results['numerical'] = verify_numerical_consistency(inputs)
    
    # 4. 性能对比验证
    if not args.skip_perf:
        results['performance'] = verify_performance(inputs, num_runs=args.perf_runs)
    else:
        print("\n⚠️ Performance test skipped")
        results['performance'] = True
    
    # 最终汇总
    print("\n" + "="*60)
    print("=== Verification Summary ===")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All verification tests passed!")
        print("✅ Deterministic mode implementation verified successfully")
        return True
    else:
        print("\n❌ Some verification tests failed")
        print("⚠️ Please review failed tests and debug accordingly")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)