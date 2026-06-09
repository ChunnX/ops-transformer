# SparseLightningIndexerGradKLLoss 泛化精度测试

## 测试目的
验证 TileLang 实现在不同配置下的精度表现，对比官方 Ascend C 算子。

## 测试用例（共6个）

| Case | 配置 | 特点 |
|------|------|------|
| 0 | S1=1, S2=2048, K=2048, N1=128, Nidx1=64, BSND | 现有配置（已验证） |
| 1 | S1=128, S2=128, K=2048, N1=64, Nidx1=64, TND | 官方风格TND布局，动态填充 |
| 2 | S1=64, S2=512, K=1024, N1=64, Nidx1=32, BSND | 短序列，最小TopK |
| 3 | S1=512, S2=2048, K=2048, N1=64, Nidx1=32, BSND | 中序列，典型配置 |
| 4 | S1=2048, S2=8192, K=4096, N1=128, Nidx1=64, BSND | 长序列，动态填充 |
| 5 | S1=512, S2=2048, K=8192, N1=64, Nidx1=32, BSND | 大TopK边界 |

## 关键改进（修正官方示例错误）

### 1. weights范围限制
```python
weights = torch.randn(...) * 0.0167  # 固定范围[-0.05, 0.05]
```

### 2. sparse_indices不重复索引
```python
# 使用randperm替代randint（避免重复）
indices = torch.randperm(s2_real_size)[:s2_real_len]
```

### 3. softmax_max/sum正确计算
```python
# 从实际QK计算（修正官方使用随机值的错误）
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