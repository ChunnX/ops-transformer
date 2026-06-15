# Golden vs Verification脚本逻辑差异分析

## 执行时间
**分析时间**: 2026-06-09 11:00
**分析人**: Sisyphus (OpenCode Agent)

---

## 概览

**对比文件**:
- Golden实现: `/mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/tests/assets/golden.py`
- 验证脚本: `/mnt/workspace/gitcode/ops-transformer/mhc/mhc_pre_sinkhorn_backward/tests/verification/verify_deterministic_mode.py`

**核心问题**: 验证脚本**无法正确调用golden进行精度验证**，存在多个致命差异。

---

## 关键差异清单

### ❌ 差异1：alpha参数数量不匹配（致命）

**Golden定义**:
```python
# golden.py line 278
grad_alpha = torch.tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res], 
                          dtype=torch.float32)
# grad_alpha包含3个scalar: [pre, post, res]
```

**验证脚本定义**:
```python
# verify_deterministic_mode.py line 26
alpha = torch.randn(2, dtype=torch.float32, device='npu')
# alpha只有2个元素！
```

**影响**:
- 参数数量不匹配，NPU算子调用可能失败
- Golden计算grad_alpha_res时需要alpha[2]，但验证脚本只有alpha[0]和alpha[1]
- **无法正确验证精度**

---

### ❌ 差异2：sum_out和norm_out形状不匹配（致命）

**Golden需要**:
```python
# golden.py line 117-118
sum_out: 正向保存的每一步行和与列和，形状[iters, 2, B, S, n]
norm_out: 正向保存的每一步行归一化与列归一化结果，形状[iters, 2, B, S, n, n]

# SinkhornGrad函数 line 124
iters = sum_out.shape[0]  # 需要iters维度
```

**验证脚本生成**:
```python
# verify_deterministic_mode.py line 38-39
sum_out = torch.randn(batch_size, seq_length, dtype=torch.float32, device='npu')
norm_out = torch.randn(batch_size, seq_length, dtype=torch.float32, device='npu')
# 只有2维：[B, S]，缺少[iters, 2, n]维度！
```

**影响**:
- SinkhornGrad函数需要iters维度进行反向迭代，验证脚本缺少这个维度
- SinkhornGrad(line 126-142)反向迭代iters步，但验证脚本的sum_out/norm_out无法提供迭代数据
- **无法正确计算Sinkhorn梯度**
- **无法正确验证精度**

---

### ❌ 差异3：验证脚本没有调用Golden（致命）

**验证脚本现状**:
```python
# verify_deterministic_mode.py
# 没有导入golden模块！
# 没有调用mhc_pre_sinkhorn_backward_golden函数！
# 只调用NPU算子，没有golden对比验证！
```

**检查结果**:
```bash
grep -n "golden" verify_deterministic_mode.py  # (no output)
grep -n "import golden" verify_deterministic_mode.py  # (no output)
```

**影响**:
- **验证脚本无法验证精度**
- 只验证确定性（多次运行一致性），不验证数值正确性
- **无法确认算子实现是否正确**

---

### ⚠️ 差异4：返回值处理不一致

**Golden返回**:
```python
# golden.py line 341
return grad_x, grad_phi, grad_alpha, grad_bias
# 返回4个tensor
```

**验证脚本返回**:
```python
# verify_deterministic_mode.py line 81
return result  # result是什么？未拆分grad_x/grad_phi/grad_alpha/grad_bias
```

**影响**:
- 验证脚本没有拆分返回值，无法单独验证每个梯度
- 无法针对性检查grad_alpha是否正确（特别是grad_alpha_res）
- **精度验证不完整**

---

### ⚠️ 差异5：测试数据形状参数不一致

**Golden需要的完整参数**:
```python
def mhc_pre_sinkhorn_backward_golden(
    grad_hin: torch.Tensor,      # [B, S, C]
    grad_h_post: torch.Tensor,   # [B, S, n]
    grad_h_res: torch.Tensor,    # [B, S, n, n]
    x: torch.Tensor,             # [B, S, n, C]
    phi: torch.Tensor,           # [B, S, n, n]
    alpha: torch.Tensor,         # 应为[3]或3个scalar
    bias: torch.Tensor,          # [3*n]（应为3段：[:n], [n:2n], [2n:3n])
    h_pre: torch.Tensor,         # [B, S, n]
    hc_before_norm: torch.Tensor,# [B, S, 3*n]（应为3段）
    inv_rms: torch.Tensor,       # [B, S]
    sum_out: torch.Tensor,       # [iters, 2, B, S, n] ❌验证脚本缺少
    norm_out: torch.Tensor,      # [iters, 2, B, S, n, n] ❌验证脚本缺少
    hc_eps: float = 1e-6
)
```

**验证脚本生成的参数**:
```python
def prepare_test_data(batch_size=2, seq_length=1024, n=64, c=512):
    x = torch.randn(batch_size, seq_length, n, c)  # ✅形状正确
    phi = torch.randn(batch_size, seq_length, n, n)  # ✅形状正确
    alpha = torch.randn(2)  # ❌只有2个，应为3个
    bias = torch.randn(2 * n)  # ❌只有2*n，应为3*n
    h_pre = torch.randn(batch_size, seq_length, n)  # ✅形状正确
    hc_before_norm = torch.randn(batch_size, seq_length, n)  # ❌只有n，应为3*n
    inv_rms = torch.randn(batch_size, seq_length)  # ✅形状正确
    sum_out = torch.randn(batch_size, seq_length)  # ❌缺少[iters, 2, n]
    norm_out = torch.randn(batch_size, seq_length)  # ❌缺少[iters, 2, n, n]
```

**影响**:
- 多个参数形状不正确，无法正确调用golden
- **无法正确验证精度**

---

## Golden完整逻辑（10步骤）

**golden.py实现**（line 171-341）:

### 步骤1: 输入聚合梯度（line 216-222）
```python
grad_x_from_hin = grad_hin_fp32[..., None, :] * h_pre[..., None]
grad_h_pre = (grad_hin_fp32[..., None, :] * x_fp32).sum(-1)
```

### 步骤2: 计算正向中间变量（line 207-217）
```python
norm_out_forward = hc_before_norm * inv_rms
z_pre = norm_out_forward[..., :n] * alpha[0] + bias[:n]
z_post = norm_out_forward[..., n:2*n] * alpha[1] + bias[n:2*n]
z_res = (norm_out_forward[..., 2*n:] * alpha[2] + bias[2*n:]).reshape(B, S, n, n)
```

### 步骤3: 三路激活梯度（line 223-234）
```python
grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)
grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False)
sk_grad = SinkhornGrad(grad_h_res, sum_out, norm_out)  # ❌验证脚本缺少sum_out/norm_out
grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
```

### 步骤4: bias梯度（line 259-246）
```python
grad_bias = torch.cat([
    grad_z_pre.sum((0, 1)),
    grad_z_post.sum((0, 1)),
    grad_z_res.sum((0, 1))
])
```

### 步骤5: alpha梯度（line 272-278）
```python
grad_alpha_pre = (norm_out_forward[..., :n] * grad_z_pre).sum()
grad_alpha_post = (norm_out_forward[..., n:2*n] * grad_z_post).sum()
grad_alpha_res = (norm_out_forward[..., 2*n:] * grad_z_res).sum()  # ❌验证脚本缺少alpha[2]
grad_alpha = torch.tensor([grad_alpha_pre, grad_alpha_post, grad_alpha_res])
```

### 步骤6: 线性变换梯度（line 264-270）
```python
grad_norm_out_pre = grad_z_pre * alpha[0]
grad_norm_out_post = grad_z_post * alpha[1]
grad_norm_out_res = grad_z_res * alpha[2]
grad_norm_out = torch.cat([grad_norm_out_pre, grad_norm_out_post, grad_norm_out_res], dim=-1)
```

### 步骤7: 乘法梯度（line 277-281）
```python
grad_hc_before_norm = grad_norm_out * inv_rms
grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)
```

### 步骤8: RMSNorm倒数梯度（line 286-289）
```python
x_flat = x_fp32.flatten(2)
grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)
```

### 步骤9: 矩阵乘法梯度（line 295-304）
```python
grad_x_from_matmul = grad_hc_before_norm @ phi
grad_hc_before_norm_flat = grad_hc_before_norm.flatten(0, 1)
x_flat_flat = x_flat.flatten(0, 1)
grad_phi = grad_hc_before_norm_flat.T @ x_flat_flat
```

### 步骤10: 汇总梯度（line 308-320）
```python
grad_x = (
    grad_x_from_hin.view(x.shape)
    + grad_x_from_rms.view(x.shape)
    + grad_x_from_matmul.view(x.shape)
).to(x.dtype)

return grad_x, grad_phi, grad_alpha, grad_bias
```

---

## 验证脚本现状分析

**verify_deterministic_mode.py功能**:
- ✅ 准备测试数据（但形状不正确）
- ✅ 调用NPU算子（通过torch_npu._C._ops）
- ✅ 验证确定性（多次运行一致性）
- ❌ **没有调用golden进行精度验证**
- ❌ **没有验证数值正确性**

**验证脚本缺失功能**:
- 没有导入golden模块
- 没有调用mhc_pre_sinkhorn_backward_golden
- 没有对比golden输出与NPU算子输出
- 无法验证grad_alpha的3个scalar是否正确
- 无法验证Sinkhorn梯度是否正确（缺少sum_out/norm_out）

---

## 影响评估

### 命命影响（无法验证）

| 差异 | 影响 | 严重性 |
|------|------|--------|
| alpha参数数量不匹配（2 vs 3） | NPU算子调用可能失败；无法计算grad_alpha_res | 🔴 **致命** |
| sum_out/norm_out形状不匹配 | SinkhornGrad无法计算；无法验证Sinkhorn梯度 | 🔴 **致命** |
| 验证脚本没有调用golden | 无法验证精度；无法确认算子实现正确性 | 🔴 **致命** |

### 严重影响（验证不完整）

| 差异 | 影响 | 严重性 |
|------|------|--------|
| 返回值未拆分 | 无法单独验证grad_x/grad_phi/grad_alpha/grad_bias | 🟠 **严重** |
| bias参数数量不匹配（2*n vs 3*n） | 无法计算z_res；无法计算grad_z_res | 🟠 **严重** |
| hc_before_norm形状不匹配（n vs 3*n） | 无法计算z_pre/z_post/z_res | 🟠 **严重** |

### 中等影响（验证受限）

| 差异 | 影响 | 严重性 |
|------|------|--------|
| 测试数据随机生成 | 无法重现测试结果；无法调试 | 🟡 **中等** |

---

## 根因分析

### 为什么验证脚本没有调用golden？

**原因1**: 验证脚本设计目标不同
- verify_deterministic_mode.py：验证确定性模式功能（多次运行一致性）
- golden.py：验证数值精度（与参考实现对比）

**原因2**: 开发阶段不统一
- golden.py：从outputs_logs仓库迁移，可能由其他人开发
- verify_deterministic_mode.py：之前创建，没有考虑golden验证

**原因3**: 测试数据准备不正确
- sum_out/norm_out需要正向Sinkhorn迭代的中间数据
- 验证脚本只生成随机数据，无法提供真实的sum_out/norm_out

---

## 解决方案

### 方案1: 修改验证脚本，集成golden验证（推荐）

**步骤**:
1. 修正prepare_test_data函数：
   - alpha: torch.randn(3)（改为3个）
   - bias: torch.randn(3 * n)（改为3*n）
   - hc_before_norm: torch.randn(batch_size, seq_length, 3 * n)（改为3*n）
   - sum_out: 需要生成Sinkhorn正向迭代的中间数据（或从正向算子获取）
   - norm_out: 同上

2. 导入golden模块：
   ```python
   import sys
   sys.path.append('../assets')
   from golden import mhc_pre_sinkhorn_backward_golden
   ```

3. 添加golden验证函数：
   ```python
   def verify_accuracy(inputs, deterministic=False):
       # 调用NPU算子
       result_npu = run_operator(inputs, deterministic=deterministic)
       
       # 调用golden
       result_golden = mhc_pre_sinkhorn_backward_golden(
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
           inputs['norm_out']
       )
       
       # 对比精度
       grad_x_npu, grad_phi_npu, grad_alpha_npu, grad_bias_npu = result_npu
       grad_x_golden, grad_phi_golden, grad_alpha_golden, grad_bias_golden = result_golden
       
       assert torch.allclose(grad_x_npu, grad_x_golden, rtol=1e-3, atol=1e-5)
       assert torch.allclose(grad_phi_npu, grad_phi_golden, rtol=1e-3, atol=1e-5)
       assert torch.allclose(grad_alpha_npu, grad_alpha_golden, rtol=1e-3, atol=1e-5)
       assert torch.allclose(grad_bias_npu, grad_bias_golden, rtol=1e-3, atol=1e-5)
   ```

### 方案2: 创建独立的精度验证脚本（备选）

**创建新文件**: `verify_accuracy_with_golden.py`

**优点**:
- 与确定性验证分离，职责清晰
- 可以专门针对精度验证进行优化
- 可以使用正向算子生成sum_out/norm_out

**缺点**:
- 需要额外维护一个脚本
- 测试流程更复杂

---

## 建议

### 立即修复（致命问题）

1. ✅ **修正alpha参数**: 从2个改为3个
2. ✅ **修正bias参数**: 从2*n改为3*n
3. ✅ **修正hc_before_norm参数**: 从n改为3*n
4. ⚠️ **生成正确的sum_out/norm_out**: 需要从正向Sinkhorn算子获取（或使用固定测试数据）

### 中期优化（完善验证）

1. ✅ **集成golden验证**: 修改verify_deterministic_mode.py，添加golden对比
2. ✅ **拆分返回值**: 单独验证grad_x/grad_phi/grad_alpha/grad_bias
3. ✅ **添加固定测试数据**: 使用固定seed，方便重现和调试

### 长期改进（自动化）

1. ✅ **集成正向算子**: 通过正向Sinkhorn算子生成sum_out/norm_out
2. ✅ **CI集成**: 自动调用golden验证，CI流程中强制精度检查
3. ✅ **测试数据管理**: 使用测试数据文件，避免随机生成

---

## 附录：参数形状对比表

| 参数 | Golden需要 | 验证脚本生成 | 是否匹配 |
|------|-----------|------------|---------|
| grad_hin | [B, S, C] | [B, S, C] | ✅ |
| grad_h_post | [B, S, n] | [B, S, n] | ✅ |
| grad_h_res | [B, S, n, n] | [B, S, n, n] | ✅ |
| x | [B, S, n, C] | [B, S, n, C] | ✅ |
| phi | [B, S, n, n] | [B, S, n, n] | ✅ |
| **alpha** | **[3]** | **[2]** | **❌ 致命** |
| **bias** | **[3*n]** | **[2*n]** | **❌ 致命** |
| h_pre | [B, S, n] | [B, S, n] | ✅ |
| **hc_before_norm** | **[B, S, 3*n]** | **[B, S, n]** | **❌ 致命** |
| inv_rms | [B, S] | [B, S] | ✅ |
| **sum_out** | **[iters, 2, B, S, n]** | **[B, S]** | **❌ 致命** |
| **norm_out** | **[iters, 2, B, S, n, n]** | **[B, S]** | **❌ 致命** |

---

**报告完成时间**: 2026-06-09 11:05  
**严重性**: 🔴 **致命** - 无法验证精度  
**建议**: 立即修正验证脚本，集成golden验证