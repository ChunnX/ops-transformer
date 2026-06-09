# SparseLightningIndexerGradKLLoss 泛化精度测试

## 测试目的
验证 TileLang 实现在不同配置下的精度表现，对比官方 Ascend C 算子。

## 官方算子约束说明
经实测发现，官方算子当前版本**强制要求K=2048**（与文档声称的1024~8192不符），其他K值会导致参数校验失败。

## 测试用例（共5个，全部K=2048）

| Case | 配置 | 特点 |
|------|------|------|
| 0 | S1=1, S2=2048, K=2048, N1=128, Nidx1=64, BSND | 现有配置（已验证） |
| 1 | S1=128, S2=128, K=2048, N1=128, Nidx1=64, TND | 官方风格TND布局 |
| 2 | S1=64, S2=512, K=2048, N1=128, Nidx1=64, BSND | 短序列 |
| 3 | S1=512, S2=2048, K=2048, N1=128, Nidx1=64, BSND | 中序列，典型配置 |
| 4 | S1=2048, S2=8192, K=2048, N1=128, Nidx1=64, BSND | 长序列 |

## 关键改进

### 1. TND入参构造（从BSND squeeze）
```python
# 先构造BSND格式，再squeeze掉B维，确保数据一致性
query_bsnd = torch.randn(B, S1, N1, DQuery).npu()
if layout == "TND":
    query = query_bsnd.squeeze(0)  # (S1, N1, DQuery)
```

### 2. sparse_indices构造（统一使用官方示例逻辑）
```python
# 根据causal mask动态计算有效key数量
for s1_idx in range(S1):
    s2_real_size = S2 - S1 + s1_idx + 1  # 当前query能看到的key数量
    if s2_real_size <= 0:
        s2_real_size = S2  # 异常情况fallback
    s2_real_len = min(s2_real_size, K)
    
    # 使用randperm生成不重复索引
    indices = torch.randperm(s2_real_size)[:s2_real_len]
    sparse_indices[..., :s2_real_len] = indices.to(torch.int32)
    if s2_real_len < K:
        sparse_indices[..., s2_real_len:] = -1  # 超出部分填-1
```

### 3. softmax_max/sum正确计算
```python
# 从实际QK计算（修正官方示例使用随机值的错误）
bmm1 = torch.matmul(query, key.transpose())
bmm1_scaled = bmm1 * scale_value
softmax_max = bmm1_scaled.max(dim=-1).values
softmax_sum = torch.exp(bmm1_scaled - softmax_max).sum(dim=-1)
```

## 运行方法

```bash
cd ops-transformer/attention/sparse_lightning_indexer_grad_kl_loss/tilelang/precision_test_general
python precision_test_general.py
```

## 精度标准
- 相对误差 < 2e-3 为达标
- 四个输出（dQueryIndex、dKeyIndex、dWeights、Loss）全部达标才算通过

## 输出说明
测试会输出每个case的精度对比结果和总体统计：
- max_diff：最大绝对误差
- mean_diff：平均绝对误差  
- rel_err：平均相对误差
- Loss额外显示TileLang和官方的具体值