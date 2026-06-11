# MhcPreSinkhornBackward 计算流程图

## 整体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                    输入数据 (11个Tensor)                            │
├────────────────────────────────────────────────────────────────────┤
│ grad_hin [B,S,D]        grad_h_post [B,S,N]    grad_h_res [B,S,N,N]│
│ x [B,S,N,D]             phi [N²+2N,N*D]        alpha [3]           │
│ bias [N²+2N]            h_pre [B,S,N]          hc_before_norm [...]│
│ inv_rms [B,S,1]         sum_out [iters,2,B,S,N]  norm_out [...]   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                    计算流程 (16个步骤)                              │
├────────────────────────────────────────────────────────────────────┤
│  【llh负责】步骤1-5: pre/post支路初始化和梯度                      │
│  【hbj负责】步骤6-8: res支路Sinkhorn/alpha/bias梯度                │
│  【lhp负责】步骤9-15: 线性变换组装+RMSNorm+矩阵乘法                │
│  【zyf负责】步骤16: grad_alpha/grad_bias组装+汇总                 │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                    输出数据 (4个Tensor)                            │
├────────────────────────────────────────────────────────────────────┤
│ grad_x [B,S,N,D]        grad_phi [N²+2N,N*D]                      │
│ grad_alpha [3]          grad_bias [N²+2N]                         │
└────────────────────────────────────────────────────────────────────┘
```

---

## 详细流程图（按责任边界划分）

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【llh负责】步骤1-5：pre/post支路初始化和梯度
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
输入变量:
├─ grad_hin [B,S,D] = [2,8,64]
├─ x [B,S,N,D] = [2,8,4,64]
├─ h_pre [B,S,N] = [2,8,4]
├─ hc_before_norm [B,S,N²+2N] = [2,8,24]
├─ inv_rms [B,S,1] = [2,8,1]
├─ alpha [3]
├─ bias [N²+2N] = [24]
├─ grad_h_post [B,S,N] = [2,8,4]
└─ hc_eps = 1e-6

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤1：输入聚合的梯度（h_in门控）                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_x_from_hin = grad_hin[..., None, :] * h_pre[..., None]
    ↓ Shape: [B,S,D] * [B,S,N] → [B,S,N,D] = [2,8,4,64]
    
    grad_h_pre = (grad_hin[..., None, :] * x).sum(-1)
    ↓ Shape: [B,S,D] * [B,S,N,D] → sum(-1) → [B,S,N] = [2,8,4]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤2：计算正向的中间变量（先完整计算，再拆分）                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    norm_out_forward = hc_before_norm * inv_rms
    ↓ Shape: [B,S,N²+2N] * [B,S,1] → [B,S,N²+2N] = [2,8,24]
    
    拆分为三部分:
    ├─ norm_out_forward_part1 = norm_out_forward[..., :N]        [B,S,N] = [2,8,4]
    ├─ norm_out_forward_part2 = norm_out_forward[..., N:2N]      [B,S,N] = [2,8,4]
    └─ norm_out_forward_part3 = norm_out_forward[..., 2N:]       [B,S,N²] = [2,8,16]
    
    计算三路的线性输出:
    ├─ z_pre = norm_out_forward_part1 * alpha[0] + bias[:N]      [B,S,N] = [2,8,4]
    ├─ z_post = norm_out_forward_part2 * alpha[1] + bias[N:2N]   [B,S,N] = [2,8,4]
    └─ z_res = (norm_out_forward_part3 * alpha[2] + bias[2N:])   [B,S,N²] → reshape → [B,S,N,N] = [2,8,4,4]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤3：pre/post两路激活函数的梯度                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_z_pre = SigmoidGrad(z_pre, grad_h_pre, is_pre=True, hc_eps=hc_eps)
    ↓ Shape: [B,S,N] * [B,S,N] → [B,S,N] = [2,8,4]
    ↓ 公式: dy * (sigmoid(z) - eps) * (1 - sigmoid(z))
    
    grad_z_post = 2 * SigmoidGrad(z_post, grad_h_post, is_pre=False, hc_eps=hc_eps)
    ↓ Shape: 2 * [B,S,N] → [B,S,N] = [2,8,4]
    ↓ 公式: 2 * dy * sigmoid(z) * (1 - sigmoid(z))

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤4：pre/post两路缩放因子alpha的梯度                         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_alpha_pre = (norm_out_forward_part1 * grad_z_pre).sum()
    ↓ Shape: [B,S,N] * [B,S,N] → sum() → scalar
    
    grad_alpha_post = (norm_out_forward_part2 * grad_z_post).sum()
    ↓ Shape: [B,S,N] * [B,S,N] → sum() → scalar

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤5：pre/post两路偏置bias的梯度（先分计算）                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_bias_part1 = grad_z_pre.sum((0, 1))
    ↓ Shape: [B,S,N] → sum((0,1)) → [N] = [4]
    
    grad_bias_part2 = grad_z_post.sum((0, 1))
    ↓ Shape: [B,S,N] → sum((0,1)) → [N] = [4]

中间变量（供后续使用）:
├─ grad_x_from_hin [B,S,N,D] = [2,8,4,64]  ──────→ 步骤16（zyf）
├─ grad_z_pre [B,S,N] = [2,8,4]            ──────→ 步骤9
├─ grad_z_post [B,S,N] = [2,8,4]           ──────→ 步骤9
├─ grad_alpha_pre scalar                   ──────→ 步骤16（zyf）
├─ grad_alpha_post scalar                  ──────→ 步骤16（zyf）
├─ grad_bias_part1 [N] = [4]               ──────→ 步骤13（zyf）
├─ grad_bias_part2 [N] = [4]               ──────→ 步骤13（zyf）
└─ z_res [B,S,N,N] = [2,8,4,4]             ──────→ 步骤6（hbj/lhp）
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【hbj负责】步骤6-8：res支路Sinkhorn/alpha/bias梯度
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
输入变量（来自步骤1-5）:
├─ z_res [B,S,N,N] = [2,8,4,4]
├─ norm_out_forward_part3 [B,S,N²] = [2,8,16]
├─ grad_h_res [B,S,N,N] = [2,8,4,4]
├─ sum_out [iters,2,B,S,N] = [3,2,2,8,4]
└─ norm_out [iters,2,B,S,N,N] = [3,2,2,8,4,4]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤6：res支路激活函数的梯度计算                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    sk_grad = SinkhornGrad(grad_h_res, sum_out, norm_out)
    ↓ Shape: [B,S,N,N] → [B,S,N,N] = [2,8,4,4]
    ↓ 功能: Sinkhorn迭代反向传播梯度
    
    grad_z_res = ExpGrad(z_res, sk_grad).flatten(2)
    ↓ Shape: [B,S,N,N] * [B,S,N,N] → flatten(2) → [B,S,N²] = [2,8,16]
    ↓ 公式: 数值稳定的exp激活梯度

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤7：res支路缩放因子alpha的梯度                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_alpha_res = (norm_out_forward_part3 * grad_z_res).sum()
    ↓ Shape: [B,S,N²] * [B,S,N²] → sum() → scalar

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤8：res支路偏置bias的梯度                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_bias_part3 = grad_z_res.sum((0, 1))
    ↓ Shape: [B,S,N²] → sum((0,1)) → [N²] = [16]

中间变量（供后续使用）:
├─ grad_z_res [B,S,N²] = [2,8,16]          ──────→ 步骤10
├─ grad_alpha_res scalar                   ──────→ 步骤16（zyf）
└─ grad_bias_part3 [N²] = [16]             ──────→ 步骤13（zyf）
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【lhp负责】步骤9-15：线性变换组装+RMSNorm+矩阵乘法
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
输入变量（来自步骤1-8）:
├─ grad_z_pre [B,S,N] = [2,8,4]
├─ grad_z_post [B,S,N] = [2,8,4]
├─ grad_z_res [B,S,N²] = [2,8,16]
├─ alpha [3]
├─ hc_before_norm [B,S,N²+2N] = [2,8,24]
└─ inv_rms [B,S,1] = [2,8,1]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤9：pre/post两路线性变换的梯度                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_norm_out_pre = grad_z_pre * alpha[0]
    ↓ Shape: [B,S,N] * scalar → [B,S,N] = [2,8,4]
    
    grad_norm_out_post = grad_z_post * alpha[1]
    ↓ Shape: [B,S,N] * scalar → [B,S,N] = [2,8,4]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤10：res支路线性变换的梯度                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_norm_out_res = grad_z_res * alpha[2]
    ↓ Shape: [B,S,N²] * scalar → [B,S,N²] = [2,8,16]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤15：矩阵乘法的梯度 + grad_x汇总                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_norm_out = torch.cat([
        grad_norm_out_pre,    [B,S,N] = [2,8,4]
        grad_norm_out_post,   [B,S,N] = [2,8,4]
        grad_norm_out_res     [B,S,N²] = [2,8,16]
    ], dim=-1)
    ↓ Shape: cat([N, N, N²], dim=-1) → [B,S,N²+2N] = [2,8,24]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤12：完整的三路乘法梯度                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_hc_before_norm = grad_norm_out * inv_rms
    ↓ Shape: [B,S,N²+2N] * [B,S,1] → [B,S,N²+2N] = [2,8,24]
    
    grad_inv_rms = (hc_before_norm * grad_norm_out).sum(-1, keepdim=True)
    ↓ Shape: [B,S,N²+2N] * [B,S,N²+2N] → sum(-1) → [B,S,1] = [2,8,1]

中间变量（供后续使用）:
├─ grad_hc_before_norm [B,S,N²+2N] = [2,8,24] ──→ 步骤15（lhp）
└─ grad_inv_rms [B,S,1] = [2,8,1]             ──→ 步骤14（lhp）
```

---

### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【zyf负责】步骤16：grad_alpha和grad_bias组装
### ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
输入变量（来自步骤1-15）:
├─ grad_bias_part1 [N] = [4]           (来自步骤5)
├─ grad_bias_part2 [N] = [4]           (来自步骤5)
├─ grad_bias_part3 [N²] = [16]         (来自步骤8)
├─ grad_alpha_pre scalar               (来自步骤4)
├─ grad_alpha_post scalar              (来自步骤4)
├─ grad_alpha_res scalar               (来自步骤7)
└─ (已完成grad_x和grad_phi计算，步骤15已执行)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤13：偏置bias的梯度组装                                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_bias = torch.cat([
        grad_bias_part1,    [N] = [4]
        grad_bias_part2,    [N] = [4]
        grad_bias_part3     [N²] = [16]
    ])
    ↓ Shape: cat([N, N, N²]) → [N²+2N] = [24]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤14：RMSNorm倒数的梯度                                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    x_flat = x.flatten(2)
    ↓ Shape: [B,S,N,D] → flatten(2) → [B,S,N*D] = [2,8,256]
    
    grad_x_from_rms = RMSNormGrad(x_flat, inv_rms, grad_inv_rms)
    ↓ Shape: [B,S,N*D] * [B,S,1] * [B,S,1] → [B,S,N*D] = [2,8,256]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤15：矩阵乘法的梯度 + grad_x汇总                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_x_from_matmul = grad_hc_before_norm @ phi
    ↓ Shape: [B,S,N²+2N] @ [N²+2N,N*D] → [B,S,N*D] = [2,8,256]
    
    grad_hc_before_norm_flat = grad_hc_before_norm.flatten(0, 1)
    ↓ Shape: [B,S,N²+2N] → flatten(0,1) → [B*S,N²+2N] = [16,24]
    
    x_flat_flat = x_flat.flatten(0, 1)
    ↓ Shape: [B,S,N*D] → flatten(0,1) → [B*S,N*D] = [16,256]
    
    grad_phi = grad_hc_before_norm_flat.T @ x_flat_flat
    ↓ Shape: [N²+2N,B*S] @ [B*S,N*D] → [N²+2N,N*D] = [24,256]
    
    grad_x = (
        grad_x_from_hin.view(x.shape),      [B,S,N,D] = [2,8,4,64]
        + grad_x_from_rms.view(x.shape),    [B,S,N*D] → [B,S,N,D]
        + grad_x_from_matmul.view(x.shape)  [B,S,N*D] → [B,S,N,D]
    ).to(x.dtype)
    ↓ Shape: [B,S,N,D] + [B,S,N,D] + [B,S,N,D] → grad_x[B,S,N,D] = [2,8,4,64]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 步骤16：组装grad_alpha和grad_bias                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    grad_alpha = torch.tensor([
        grad_alpha_pre,    scalar
        grad_alpha_post,   scalar
        grad_alpha_res     scalar
    ], dtype=torch.float32)
    ↓ Shape: [3]
    
    grad_bias = torch.cat([
        grad_bias_part1,    [N] = [4]
        grad_bias_part2,    [N] = [4]
        grad_bias_part3     [N²] = [16]
    ])
    ↓ Shape: cat([N, N, N²]) → [N²+2N] = [24]
```

---

## 输出结果

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 最终输出 (4个Tensor)                                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
├─ grad_x      [B,S,N,D]    = [2,8,4,64]     ✅
├─ grad_phi    [N²+2N,N*D]  = [24,256]       ✅
├─ grad_alpha  [3]          = [3]            ✅
└─ grad_bias   [N²+2N]      = [24]           ✅
```

---

## 依赖关系总结

### 关键依赖链

```
步骤1 (llh) ──→ grad_x_from_hin ────────────────────────────────→ 步骤16 (zyf)
步骤1 (llh) ──→ grad_h_pre ──────────────────────────────────→ 步骤3 (llh)

步骤2 (llh) ──→ z_res ────────────────────────────────────────→ 步骤6 (hbj/lhp)
步骤2 (llh) ──→ norm_out_forward_part1/2/3 ──→ 步骤3-8

步骤3 (llh) ──→ grad_z_pre/post ──────────────────────────────→ 步骤9

步骤4 (llh) ──→ grad_alpha_pre/post ──────────────────────────→ 步骤16 (zyf)

步骤5 (llh) ──→ grad_bias_part1/2 ────────────────────────────→ 步骤13 (zyf)

步骤6 (hbj/lhp) ──→ grad_z_res ───────────────────────────────→ 步骤10
步骤7 (hbj/lhp) ──→ grad_alpha_res ───────────────────────────→ 步骤16 (zyf)
步骤8 (hbj/lhp) ──→ grad_bias_part3 ───────────────────────────→ 步骤13 (zyf)

步骤9 ──→ grad_norm_out_pre/post ─────────────────────────────→ 步骤11
步骤10 ──→ grad_norm_out_res ─────────────────────────────────→ 步骤11

步骤11 ──→ grad_norm_out ─────────────────────────────────────→ 步骤12
步骤12 ──→ grad_hc_before_norm ───────────────────────────────→ 步骤15 (zyf)
步骤12 ──→ grad_inv_rms ───────────────────────────────────────→ 步骤14 (zyf)

步骤13 (zyf) ──→ grad_bias (最终输出)
步骤14 (zyf) ──→ grad_x_from_rms ─────────────────────────────→ 步骤16 (zyf)
步骤15 (zyf) ──→ grad_phi (最终输出)
步骤15 (zyf) ──→ grad_x_from_matmul ──────────────────────────→ 步骤16 (zyf)
步骤16 (zyf) ──→ grad_x, grad_alpha (最终输出)
```

---

## 责任边界总结

| 负责人 | 步骤范围 | 关键变量 | 核心功能 |
|--------|---------|---------|---------|
| **llh** | 步骤1-5 | grad_x_from_hin, grad_z_pre/post, grad_alpha_pre/post, grad_bias_part1/2 | pre/post支路初始化和梯度计算 |
| **hbj** | 步骤6-8 | grad_z_res, grad_alpha_res, grad_bias_part3 | res支路Sinkhorn+exp激活梯度、alpha/bias梯度 |
| **lhp** | 步骤9-15 | grad_norm_out, grad_hc_before_norm, grad_phi, grad_x | 线性变换组装+RMSNorm+矩阵乘法+grad_x汇总 |
| **zyf** | 步骤16 | grad_alpha, grad_bias | grad_alpha和grad_bias组装 |

---

## 关键中间变量分类

### 输入变量 (11个)
```
grad_hin, grad_h_post, grad_h_res, x, phi, alpha, bias, 
h_pre, hc_before_norm, inv_rms, sum_out, norm_out
```

### 中间变量 (按步骤)
```
步骤1: grad_x_from_hin, grad_h_pre
步骤2: norm_out_forward, norm_out_forward_part1/2/3, z_pre/post/res
步骤3: grad_z_pre, grad_z_post
步骤4: grad_alpha_pre, grad_alpha_post
步骤5: grad_bias_part1, grad_bias_part2
步骤6: sk_grad, grad_z_res
步骤7: grad_alpha_res
步骤8: grad_bias_part3
步骤9: grad_norm_out_pre, grad_norm_out_post
步骤10: grad_norm_out_res
步骤11: grad_norm_out
步骤12: grad_hc_before_norm, grad_inv_rms
步骤14: x_flat, grad_x_from_rms
步骤15: grad_hc_before_norm_flat, x_flat_flat, grad_phi, grad_x_from_matmul
```

### 输出变量 (4个)
```
grad_x, grad_phi, grad_alpha, grad_bias
```

---

## Shape变化关键点

### 1. phi形状理解（最关键）
```
phi: [N²+2N, N*D] = [24, 256]
├─ N²+2N = 16+8 = 24  (三路输出维度: pre[N] + post[N] + res[N²])
└─ N*D = 4*64 = 256  (输入特征维度)

正向: hc_before_norm = x_flat @ phi.T + bias
├─ x_flat: [T, N*D] = [16, 256]
├─ phi.T: [N*D, N²+2N] = [256, 24]
└─ 输出: [T, N²+2N] = [16, 24]
```

### 2. 三路拆分逻辑
```
norm_out_forward [B,S,N²+2N] = [2,8,24]
├─ part1: [:N]           = [B,S,N] = [2,8,4]     (pre支路)
├─ part2: [N:2N]         = [B,S,N] = [2,8,4]     (post支路)
└─ part3: [2N:]          = [B,S,N²] = [2,8,16]   (res支路)
```

### 3. flatten操作
```
x [B,S,N,D] → flatten(2) → x_flat [B,S,N*D]
├─ [2,8,4,64] → [2,8,256]

x_flat [B,S,N*D] → flatten(0,1) → x_flat_flat [B*S,N*D]
├─ [2,8,256] → [16,256]

z_res [B,S,N,N] → flatten(2) → grad_z_res [B,S,N²]
├─ [2,8,4,4] → [2,8,16]
```

---

## 重要公式总结

### SigmoidGrad公式
```
is_pre=True:  dy * (sigmoid(z) - eps) * (1 - sigmoid(z))
is_pre=False: dy * sigmoid(z) * (1 - sigmoid(z))
```

### RMSNormGrad公式
```
grad_x = - (inv_rms ** 3) * x * grad_inv_rms / NC
```

### ExpGrad公式（数值稳定）
```
y = exp(x - x_max)
sum_all = (y * y_grad).sum()
grad = y * y_grad - is_max * sum_all
```

### SinkhornGrad公式（迭代反向）
```
for i in reversed(range(iters)):
    grad_x_row_normed = x_grad / col_sum - (x_grad * x_row_normed / (col_sum ** 2)).sum()
    x_grad = grad_x_row_normed / row_sum - (grad_x_row_normed * x_row_normed / row_sum).sum()
```

---

**生成时间**: 2025-06-10  
**验证状态**: ✅ golden_v2.py功能完全正常  
**关键约束**: phi [N²+2N, N*D] = [24, 256] (官方定义)