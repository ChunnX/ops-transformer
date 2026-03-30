# MhcPreSinkhorn 算子测试指南

## 测试概述

本文档提供了 mhc_pre_sinkhorn 算子的完整测试指南，包括功能测试、精度测试和性能测试。

## 测试环境要求

### 硬件要求
- NPU 设备：Ascend910B / Ascend910_93 / Ascend950DT / Ascend950PR
- 内存：至少 32GB
- 存储：至少 100GB

### 软件要求
- CANN 版本：8.0.RC3 或更高
- Python 版本：3.8 或更高
- PyTorch 版本：2.0 或更高（可选，用于对比测试）

## 1. 功能测试

### 1.1 基础功能测试

#### 测试用例 1：n=4，3D 输入
```python
import torch
import torch_npu

# 准备测试数据
T = 8
n = 4
h_data = torch.randn(T, n, n, dtype=torch.float32).npu()

# 调用算子
eps = 1e-6
num_iters = 20
out_flag = 0

h_res_sinkhorn = torch.empty_like(h_data)
norm_out = None
sum_out = None

# 调用 ACLNN 接口（需要封装）
result = mhc_pre_sinkhorn(h_data, eps, num_iters, out_flag, 
                          h_res_sinkhorn, norm_out, sum_out)

# 验证输出
assert result.shape == h_data.shape
assert result.dtype == torch.float32kt
```

#### 测试用例 2：n=6，4D 输入
```python
B = 2
S = 4
n = 6
h_data = torch.randn(B, S, n, n, dtype=torch.float32).npu()

# 调用算子
h_res_sinkhorn = torch.empty_like(h_data)
result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                          h_res_sinkhorn, None, None)

# 验证输出
assert result.shape == (B, S, n, n)
```

#### 测试用例 3：n=8，不同迭代次数
```python
for num_iters in [1, 10, 20, 50, 100]:
    h_data = torch.randn(8, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    result = mhc_pre_sinkhorn(h_data, 1e-6, num_iters, 0, 
                              h_res_sinkhorn, None, None)
    
    # 验证双随机性
    row_sums = result.sum(dim=-1)
    col_sums = result.sum(dim=-2)
    
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-5)
```

### 1.2 边界条件测试

#### 测试用例 4：最小迭代次数
```python
h_data = torch.randn(4, 4, 4, dtype=torch.float32).npu()
h_res_sinkhorn = torch.empty_like(h_data)

result = mhc_pre_sinkhorn(h_data, 1e-6, 1, 0, 
                          h_res_sinkhorn, None, None)
# 验证结果有效性
```

#### 测试用例 5：最大迭代次数
```python
h_data = torch.randn(8, 8, 8, dtype=torch.float32).npu()
h_res_sinkhorn = torch.empty_like(h_data)

result = mhc_pre_sinkhorn(h_data, 1e-6, 100, 0, 
                          h_res_sinkhorn, None, None)
# 验证结果有效性
```

#### 测试用例 6：不同 eps 值
```python
for eps in [1e-8, 1e-6, 1e-4, 1e-2]:
    h_data = torch.randn(4, 4, 4, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    result = mhc_pre_sinkhorn(h_data, eps, 20, 0, 
                              h_res_sinkhorn, None, None)
    
    # 验证数值稳定性
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
```

### 1.3 特殊值处理测试

#### 测试用例 7：包含 inf 的输入
```python
h_data = torch.randn(4, 4, 4, dtype=torch.float32).npu()
h_data[0, 0, 0] = float('inf')
h_res_sinkhorn = torch.empty_like(h_data)

result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                          h_res_sinkhorn, None, None)
# 验证输出为 nan
assert torch.isnan(result).any()
```

#### 测试用例 8：包含 nan 的输入
```python
h_data = torch.randn(4, 4, 4, dtype=torch.float32).npu()
h_data[0, 0, 0] = float('nan')
h_res_sinkhorn = torch.empty_like(h_data)

result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                          h_res_sinkhorn, None, None)
# 验证输出为 nan
assert torch.isnan(result).any()
```

## 2. 精度测试

### 2.1 双随机性验证

#### 测试用例 9：行归一化验证
```python
def test_row_normalization():
    h_data = torch.randn(8, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                              h_res_sinkhorn, None, None)
    
    # 计算行和
    row_sums = result.sum(dim=-1)
    
    # 验证行和接近 1
    expected = torch.ones_like(row_sums)
    max_error = torch.max(torch.abs(row_sums - expected)).item()
    
    assert max_error < 1e-5, f"Row normalization error: {max_error}"
    print(f"Row normalization max error: {max_error}")
```

#### 测试用例 10：列归一化验证
```python
def test_col_normalization():
    h_data = torch.randn(8, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                              h_res_sinkhorn, None, None)
    
    # 计算列和
    col_sums = result.sum(dim=-2)
    
    # 验证列和接近 1
    expected = torch.ones_like(col_sums)
    max_error = torch.max(torch.abs(col_sums - expected)).item()
    
    assert max_error < 1e-5, f"Column normalization error: {max_error}"
    print(f"Column normalization max error: {max_error}")
```

### 2.2 非负性验证

#### 测试用例 11：元素非负性验证
```python
def test_non_negativity():
    h_data = torch.randn(8, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                              h_res_sinkhorn, None, None)
    
    # 验证所有元素 >= 0
    min_value = result.min().item()
    
    assert min_value >= -1e-6, f"Negative value found: {min_value}"
    print(f"Minimum value: {min_value}")
```

### 2.3 数值稳定性验证

#### 测试用例 12：数值稳定性测试
```python
def test_numerical_stability():
    h_data = torch.randn(8, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    result = mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                              h_res_sinkhorn, None, None)
    
    # 验证无 inf 或 nan
    has_nan = torch.isnan(result).any().item()
    has_inf = torch.isinf(result).any().item()
    
    assert not has_nan, "NaN found in output"
    assert not has_inf, "Inf found in output"
    print("Numerical stability test passed")
```

### 2.4 对比测试（CPU 基线）

#### 测试用例 13：与 CPU 实现对比
```python
def sinkhorn_cpu(h_data, eps=1e-6, num_iters=20):
    """CPU 实现的 Sinkhorn 算法"""
    result = h_data.clone()
    
    for _ in range(num_iters):
        # 列归一化
        col_sums = result.sum(dim=-2, keepdim=True) + eps
        result = result / col_sums
        
        # 行归一化
        row_sums = result.sum(dim=-1, keepdim=True) + eps
        result = result / row_sums
    
    return result

def test_cpu_comparison():
    h_data = torch.randn(4, 4, 4, dtype=torch.float32)
    
    # NPU 计算
    h_data_npu = h_data.clone().npu()
    h_res_sinkhorn = torch.empty_like(h_data_npu)
    result_npu = mhc_pre_sinkhorn(h_data_npu, 1e-6, 20, 0, 
                                  h_res_sinkhorn, None, None)
    result_npu = result_npu.cpu()
    
    # CPU 计算
    result_cpu = sinkhorn_cpu(h_data, 1e-6, 20)
    
    # 对比结果
    max_diff = torch.max(torch.abs(result_npu - result_cpu)).item()
    mean_diff = torch.mean(torch.abs(result_npu - result_cpu)).item()
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    assert max_diff < 1e-4, f"Max difference too large: {max_diff}"
```

## 3. 性能测试

### 3.1 不同数据规模测试

#### 测试用例 14：小规模数据性能
```python
def test_small_scale_performance():
    import time
    
    h_data = torch.randn(8, 4, 4, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    # 预热
    for _ in range(10):
        mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    
    # 正式测试
    start_time = time.time()
()
    num_iterations = 100
    for _ in range(num_iterations):
        mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Small scale average time: {avg_time * 1000:.2f} ms")
```

#### 测试用例 15：中等规模数据性能
```python
def test_medium_scale_performance():
    import time
    
    h_data = torch.randn(64, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    # 预热
    for _ in range(10):
        mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    
    # 正式测试
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Medium scale average time: {avg_time * 1000:.2f} ms")
```

#### 测试用例 16：大规模数据性能
```python
def test_large_scale_performance():
    import time
    
    h_data = torch.randn(256, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    # 预热
    for _ in range(10):
        mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    
    # 正式测试
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        mhc_pre_sinkhorn(h_data, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"Large scale average time: {avg_time * 1000:.2f} ms")
```

### 3.2 不同迭代次数性能

#### 测试用例 17：迭代次数性能分析
```python
def test_iteration_performance():
    import time
    
    h_data = torch.randn(64, 8, 8, dtype=torch.float32).npu()
    h_res_sinkhorn = torch.empty_like(h_data)
    
    for num_iters in [1, 5, 10, 20, 50, 100]:
        # 预热
        for _ in range(5):
            mhc_pre_sinkhorn(h_data, 1e-6, num_iters, 0, 
                             h_res_sinkhorn, None, None)
        
        # 正式测试
        start_time = time.time()
        num_iterations = 50
        for _ in range(num_iterations):
            mhc_pre_sinkhorn(h_data, 1e-6, num_iters, 0, 
                             h_res_sinkhorn, None, None)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        print(f"Iterations={num_iters}, Average time: {avg_time * 1000:.2f} ms")
```

### 3.3 性能对比测试

#### 测试用例 18：NPU vs CPU 性能对比
```python
def test_npu_vs_cpu_performance():
    import time
    
    h_data = torch.randn(64, 8, 8, dtype=torch.float32)
    
    # CPU 测试
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        sinkhorn_cpu(h_data, 1e-6, 20)
    end_time = time.time()
    cpu_avg_time = (end_time - start_time) / num_iterations
    
    # NPU 测试
    h_data_npu = h_data.clone().npu()
    h_res_sinkhorn = torch.empty_like(h_data_npu)
    
    start_time = time.time()
    for _ in range(num_iterations):
        mhc_pre_sinkhorn(h_data_npu, 1e-6, 20, 0, 
                         h_res_sinkhorn, None, None)
    end_time = time.time()
    npu_avg_time = (end_time - start_time) / num_iterations
    
    speedup = cpu_avg_time / npu_avg_time
    print(f"CPU average time: {cpu_avg_time * 1000:.2f} ms")
    print(f"NPU average time: {npu_avg_time * 1000:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
```

## 4. 综合测试

### 4.1 完整测试套件

```python
def run_all_tests():
    print("=" * 60)
    print("MhcPreSinkhorn 算子完整测试套件")
    print("=" * 60)
    
    # 功能测试
    print("\n[1] 功能测试")
    print("-" * 60)
    test_basic_functionality()
    test_edge_cases()
    test_special_values()
    
    # 精度测试
    print("\n[2] 精度测试")
    print("-" * 60)
    test_row_normalization()
    test_col_normalization()
    test_non_negativity()
    test_numerical_stability()
    test_cpu_comparison()
    
    # 性能测试
    print("\n[3] 性能测试")
    print("-" * 60)
    test_small_scale_performance()
    test_medium_scale_performance()
    test_large_scale_performance()
    test_iteration_performance()
    test_npu_vs_cpu_performance()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
```

## 5. 测试报告模板

### 测试报告格式

```
MhcPreSinkhorn 算子测试报告
===========================

测试日期：时间戳
测试环境：NPU 型号、CANN 版本
测试人员：姓名

1. 功能测试结果
----------------
测试用例 1：通过/失败
测试用例 2：通过/失败
...
功能测试通过率：XX%

2. 精度测试结果
----------------
行归一化最大误差：X.XXXXe-XX
列归一化最大误差：X.XXXXe-XX
最小元素值：X.XXXXe-XX
数值稳定性：通过/失败
CPU 对比最大差异：X.XXXXe-XX

3. 性能测试结果
----------------
小规模数据平均时间：XX.XX ms
中等规模数据平均时间：XX.XX ms
大规模数据平均时间：XX.XX ms
NPU vs CPU 加速比：XX.XXx

4. 总体评估
------------
功能完整性：通过/失败
精度达标：通过/失败
性能达标：通过/失败
总体评价：优秀/良好/一般/需改进

5. 问题记录
------------
问题描述 1
问题描述 2
...

6. 改进建议
------------
改进建议 1
改进建议 2
...
```

## 6. 注意事项

### 6.1 测试前准备
1. 确保 NPU 设备正常工作
2. 确保算子已正确编译和安装
3. 准备充足的测试数据
4. 备份重要数据

### 6.2 测试过程中
1. 监控 NPU 利用率和内存使用情况
2. 记录所有测试结果和错误信息
3. 遇到异常时及时停止并分析原因
4. 定期保存测试进度

### 6.3 测试后整理
1. 整理测试数据和分析结果
2. 生成测试报告
3. 归档测试日志和输出
4. 总结测试经验和改进建议

## 7. 常见问题排查

### 问题 1：编译错误
- 检查 CANN 版本是否匹配
- 检查编译选项是否正确
- 查看编译日志定位错误

### 问题 2：运行时错误
- 检查输入数据格式和维度
- 检查参数范围是否合法
- 查看 NPU 错误日志

### 问题 3：精度不达标
- 增加 Sinkhorn 迭代次数
- 调整 eps 参数
- 检查数值稳定性

### 问题 4：性能不达标
- 检查 NPU 利用率
- 优化 Tiling 策略
- 考虑使用双缓冲或流水线

## 总结

本测试指南提供了 mhc_pre_sinkhorn 算子的完整测试方案，包括功能测试、精度测试和性能测试。按照本指南进行测试，可以全面验证算子的正确性、精度和性能，确保算子满足设计要求。
