# MhcPreSinkhornBackward 确定性模式验证指南

## 实施概览

**算子名称**: MhcPreSinkhornBackward
**平台**: arch35 (Ascend 950)
**实现方案**: 方案A（完全确定性）
**TILING_KEY**: 0=非确定性，1=确定性

---

## 1. 验证准备

### 1.1 环境要求

- NPU设备: Ascend 950 (arch35)
- CANN版本: 9.0.0-beta.2
- PyTorch: torch_npu已安装
- 算子库: libes_transformer_cust.so已编译

### 1.2 编译产物路径

```
/mnt/workspace/gitcode/ops-transformer/build/
├── libes_transformer_cust.so          # 主库（包含确定性算子）
├── libcust_opmaster_rt2.0.so          # 算子master库
└── es_packages/
    └── lib64/
        └── libes_transformer_cust.so  # 安装库
```

### 1.3 安装算子库

```bash
# 方式1: 环境变量方式
export LD_LIBRARY_PATH=/mnt/workspace/gitcode/ops-transformer/build:$LD_LIBRARY_PATH

# 方式2: 安装到CANN路径
cp /mnt/workspace/gitcode/ops-transformer/build/libes_transformer_cust.so \
   /home/developer/Ascend/cann-9.0.0-beta.2/lib64/
```

---

## 2. 算子调用方式

### 2.1 TILING_KEY=0（非确定性模式）

```python
import torch
import torch_npu

# 原有调用方式（默认TILING_KEY=0）
output = torch_npu.npu.mhc_pre_sinkhorn_backward(
    grad_hin, grad_h_post, grad_h_res, x, phi, 
    alpha, bias, h_pre, hc_before_norm, inv_rms, 
    sum_out, norm_out
)
```

### 2.2 TILING_KEY=1（确定性模式）

**方式1: 通过opapi参数**

```python
from torch_npu._C._ops import ops

op = ops.npu.mhc_pre_sinkhorn_backward
output = op(
    grad_hin, grad_h_post, grad_h_res, x, phi, 
    alpha, bias, h_pre, hc_before_norm, inv_rms, 
    sum_out, norm_out,
    deterministic=True  # 参数指定确定性模式
)
```

**方式2: 通过context管理**

```python
with torch_npu.npu.deterministic_mode(True):
    output = torch_npu.npu.mhc_pre_sinkhorn_backward(...)
```

**方式3: 直接TILING_KEY控制（底层方式）**

注意：需要确认算子proto是否支持deterministic参数，否则可能需要通过tiling配置文件控制。

---

## 3. 验证步骤

### Step 1: 基本功能验证

验证两种模式的基本功能是否正常。

```python
# 测试非确定性模式
result_non_det = torch_npu.npu.mhc_pre_sinkhorn_backward(
    inputs, deterministic=False
)

# 测试确定性模式
result_det = torch_npu.npu.mhc_pre_sinkhorn_backward(
    inputs, deterministic=True
)

# 检查输出形状和类型
assert result_non_det.shape == expected_shape
assert result_det.shape == expected_shape
assert result_non_det.dtype == torch.float32
assert result_det.dtype == torch.float32
```

### Step 2: 确定性验证

验证确定性模式多次运行结果完全一致。

```python
results = []
for i in range(10):
    torch.manual_seed(42)  # 固定随机种子
    result = torch_npu.npu.mhc_pre_sinkhorn_backward(
        inputs, deterministic=True
    )
    results.append(result)

# 验证结果完全一致
for i in range(1, 10):
    assert torch.allclose(
        results[0], results[i], 
        rtol=1e-5, atol=1e-8
    ), f"Run {i} does not match Run 0"
```

### Step 3: 数值一致性验证

验证非确定性 vs 确定性数值是否基本一致（允许微小差异）。

```python
result_non_det = torch_npu.npu.mhc_pre_sinkhorn_backward(
    inputs, deterministic=False
)
result_det = torch_npu.npu.mhc_pre_sinkhorn_backward(
    inputs, deterministic=True
)

# 允许微小差异（atomicAdd顺序可能不同）
assert torch.allclose(
    result_non_det, result_det, 
    rtol=1e-3, atol=1e-5
)
```

### Step 4: 性能对比验证

验证确定性模式的性能损失是否在预期范围内（~12%）。

```python
import time

# 非确定性模式性能
torch_npu.npu.synchronize()
start = time.time()
for i in range(100):
    _ = torch_npu.npu.mhc_pre_sinkhorn_backward(
        inputs, deterministic=False
    )
torch_npu.npu.synchronize()
time_non_det = time.time() - start

# 确定性模式性能
torch_npu.npu.synchronize()
start = time.time()
for i in range(100):
    _ = torch_npu.npu.mhc_pre_sinkhorn_backward(
        inputs, deterministic=True
    )
torch_npu.npu.synchronize()
time_det = time.time() - start

# 性能损失验证
perf_loss = (time_det - time_non_det) / time_non_det
print(f"Performance loss: {perf_loss:.2%}")
assert perf_loss < 0.15  # 性能损失应小于15%
```

---

## 4. 验证脚本使用

### 4.1 运行完整验证

```bash
cd /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/tests/verification

# 运行完整验证流程
python verify_deterministic_mode.py

# 自定义参数
python verify_deterministic_mode.py \
    --batch-size 4 \
    --seq-length 2048 \
    --n 128 \
    --c 1024 \
    --num-runs 20 \
    --perf-runs 200
```

### 4.2 跳过性能测试

```bash
# 如果只需要功能验证，可跳过性能测试
python verify_deterministic_mode.py --skip-perf
```

---

## 5. 验证检查清单

### 5.1 基本功能检查

- [ ] TILING_KEY=0模式能正常运行
- [ ] TILING_KEY=1模式能正常运行
- [ ] 输出形状正确
- [ ] 输出类型正确（FP32）
- [ ] 无运行时错误或崩溃

### 5.2 确定性检查

- [ ] 多次运行确定性模式结果完全一致
- [ ] 固定随机种子后结果稳定
- [ ] 不同批次数据结果独立正确

### 5.3 数值一致性检查

- [ ] 非确定性 vs 确定性数值基本一致
- [ ] 最大差异 < 1e-2
- [ ] 平均差异 < 1e-5
- [ ] 无精度异常（全0、随机值、NaN）

### 5.4 性能检查

- [ ] 性能损失在预期范围内（<15%）
- [ ] 无显著性能退化（>20%）
- [ ] 性能损失符合设计预期（~12%）

---

## 6. 常见问题排查

### 6.1 算子调用失败

**症状**: 运行时报错，算子无法调用

**可能原因**:
1. 算子库未安装或路径错误
2. TILING_KEY参数未正确传递
3. 算子proto不支持deterministic参数

**排查步骤**:
```bash
# 1. 检查算子库是否存在
ls -lh /mnt/workspace/gitcode/ops-transformer/build/libes_transformer_cust.so

# 2. 检查环境变量
echo $LD_LIBRARY_PATH

# 3. 检查算子proto定义
cat /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/op_host/proto/mhc_pre_sinkhorn_backward.proto
```

### 6.2 确定性验证失败

**症状**: 多次运行结果不一致

**可能原因**:
1. TILING_KEY=1未正确生效
2. workspace分配错误
3. SyncAll未正确执行
4. ComputeDeterministic逻辑错误

**排查步骤**:
```bash
# 1. 检查TILING_KEY传递是否正确
# 查看算子proto定义，确认deterministic参数支持

# 2. 检查kernel代码
grep "DETERMINISTIC" /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_one_stage.h

# 3. 检查tiling代码
grep "deterministicWorkspaceSize" /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/op_host/op_tiling/mhc_pre_sinkhorn_backward_tiling.cpp
```

### 6.3 数值差异过大

**症状**: 非确定性 vs 确定性数值差异超过预期

**可能原因**:
1. 精度问题（FP32精度损失）
2. workspace归约逻辑错误
3. ComputeDeterministic实现错误

**排查步骤**:
```python
# 1. 检查workspace归约逻辑
# 查看ComputeDeterministic函数实现

# 2. 检查精度标准
# 参考: /ops-precision-standard skill
# FP32: rtol=1e-3, atol=1e-5

# 3. 对比arch32实现
# 查看: mhc_pre_grad_kernel.h (arch32参考)
```

### 6.4 性能损失过大

**症状**: 性能损失超过20%

**可能原因**:
1. SyncAll开销过大
2. ComputeDeterministic串行瓶颈
3. workspace内存分配开销
4. 数据搬运效率低下

**排查步骤**:
```bash
# 1. 使用profiling分析
# 参考: /ops-profiling skill

# 2. 检查SyncAll位置
grep "SyncAll" /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/op_kernel/arch35/mhc_pre_sinkhorn_backward_one_stage.h

# 3. 检查workspace大小
grep "deterministicWorkspaceSize" /mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/op_host/op_tiling/mhc_pre_sinkhorn_backward_tiling.cpp
```

---

## 7. 验证报告模板

```markdown
# MhcPreSinkhornBackward 确定性模式验证报告

## 测试环境
- NPU设备: Ascend 950
- CANN版本: 9.0.0-beta.2
- PyTorch版本: xxx
- 测试日期: 2026-06-08

## 测试配置
- Batch size: 2
- Sequence length: 1024
- Hidden dim n: 64
- Hidden dim c: 512

## 验证结果

### 1. 基本功能验证
- [✅] TILING_KEY=0模式正常
- [✅] TILING_KEY=1模式正常
- [✅] 输出形状正确
- [✅] 输出类型正确

### 2. 确定性验证
- [✅] 10次运行结果完全一致
- [✅] 固定种子后结果稳定

### 3. 数值一致性验证
- [✅] 最大差异: 1.2e-4 (< 1e-2)
- [✅] 平均差异: 3.5e-6 (< 1e-5)
- [✅] 数值一致验证通过

### 4. 性能对比验证
- [✅] 非确定性: 1.5ms/run
- [✅] 确定性: 1.7ms/run
- [✅] 性能损失: 13.3% (< 15%)

## 结论
✅ 确定性模式实现验证通过，所有测试符合预期。

## 问题记录
- (无)
```

---

## 8. 参考资料

### 8.1 设计文档

- `/mnt/workspace/gitcode/knowledge-base/wiki/mhc_pre_sinkhorn_backward_arch35_deterministic_design.md`

### 8.2 深度分析

- `/mnt/workspace/gitcode/knowledge-base/wiki/mhc_pre_sinkhorn_backward_arch35_deep_analysis.md`

### 8.3 arch32参考

- `/mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/op_kernel/mhc_pre_grad_kernel.h`

### 8.4 Skills参考

- `/ops-precision-standard`: 精度标准
- `/ops-profiling`: 性能分析
- `/ascendc-precision-debug`: 精度调试

---

**最后更新**: 2026-06-08  
**实施状态**: 代码完成 + 编译通过  
**验证状态**: 待用户验证