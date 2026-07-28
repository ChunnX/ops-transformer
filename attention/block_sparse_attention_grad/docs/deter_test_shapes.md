# block_sparse_attention_grad arch22 确定性白盒测试用例

## 测试环境

| 项目 | 值 |
|------|-----|
| 芯片 | Ascend 910B3 (arch22/DAV_2201) |
| CANN | 9.0.0 |
| soc | ascend910b |
| dtype | float16 |
| block_shape | (128, 128) |
| deter | ON (`torch.use_deterministic_algorithms(True)`) |

## 测试维度

| 维度 | 取值 | 说明 |
|------|------|------|
| B | 1, 2, 4 | batch size |
| N | 1, 2, 4, 6, 8 | query head num |
| kvH | 1, 2, 4, 8 | kv head num (kvH==N 为 MHA, kvH<N 为 GQA) |
| S | 128, 192, 256, 384, 512, 1024 | sequence length |
| D | 128 | head dim (固定) |
| sparsity | 0.5, 1.0 | block sparse mask 稀疏度 |
| groupSize | N/kvH | GQA 分组数 |

## Q block 数与 SetAtomicAdd 累加次数

| S | Q blocks (ceil(S/128)) | SetAtomicAdd 累加次数 | 确定性风险 |
|---|----------------------|---------------------|-----------|
| 128 | 1 | 0 | 无累加, 确定性最佳 |
| 192 | 2 | 1 | 轻度累加 |
| 256 | 2 | 1 | 轻度累加 |
| 384 | 3 | 2 | 中度累加 |
| 512 | 4 | 3 | 中度累加 |
| 1024 | 8 | 7 | 重度累加, 确定性风险最高 |

## 测试用例

### 一、MHA 基础 (kvH == N, groupSize = 1)

| 编号 | 名称 | B | N | S | D | kvH | sp | Qblocks | 覆盖场景 |
|------|------|---|---|---|---|-----|-----|---------|---------|
| M-01 | MHA_S128_B1N2_sp1.0 | 1 | 2 | 128 | 128 | 2 | 1.0 | 1 | 单Q block, 无累加, 全掩码 |
| M-02 | MHA_S128_B1N2_sp0.5 | 1 | 2 | 128 | 128 | 2 | 0.5 | 1 | 单Q block, 稀疏掩码 |
| M-03 | MHA_S128_B1N1_sp1.0 | 1 | 1 | 128 | 128 | 1 | 1.0 | 1 | 单头边界 |
| M-04 | MHA_S128_B2N2_sp1.0 | 2 | 2 | 128 | 128 | 2 | 1.0 | 1 | 多batch, 单Q block |
| M-05 | MHA_S256_B1N4_sp1.0 | 1 | 4 | 256 | 128 | 4 | 1.0 | 2 | 2 Q blocks, 全掩码 |
| M-06 | MHA_S256_B1N4_sp0.5 | 1 | 4 | 256 | 128 | 4 | 0.5 | 2 | 2 Q blocks, 稀疏 |
| M-07 | MHA_S256_B2N2_sp1.0 | 2 | 2 | 256 | 128 | 2 | 1.0 | 2 | 多batch + 2 Q blocks |
| M-08 | MHA_S192_B4N4_sp0.5 | 4 | 4 | 192 | 128 | 4 | 0.5 | 2 | 非对齐S(192), 多batch |
| M-09 | MHA_S384_B1N8_sp0.5 | 1 | 8 | 384 | 128 | 8 | 0.5 | 3 | 3 Q blocks(非2幂), 8头 |
| M-10 | MHA_S512_B2N8_sp0.5 | 2 | 8 | 512 | 128 | 8 | 0.5 | 4 | 4 Q blocks, 多核满载 |
| M-11 | MHA_S1024_B1N4_sp0.5 | 1 | 4 | 1024 | 128 | 4 | 0.5 | 8 | 8 Q blocks, 重累加 |
| M-12 | MHA_S1024_B1N4_sp1.0 | 1 | 4 | 1024 | 128 | 4 | 1.0 | 8 | 8 Q blocks, 全掩码 |
| M-13 | MHA_S1024_B2N8_sp0.5 | 2 | 8 | 1024 | 128 | 8 | 0.5 | 8 | 8 Q blocks + 多batch + 多核 |

### 二、GQA (kvH < N, groupSize > 1)

| 编号 | 名称 | B | N | S | D | kvH | sp | groupSize | Qblocks | 覆盖场景 |
|------|------|---|---|---|---|-----|-----|-----------|---------|---------|
| G-01 | GQA_S128_B1N4_kvH2_sp1.0 | 1 | 4 | 128 | 128 | 2 | 1.0 | 2 | 1 | GQA单Q block |
| G-02 | GQA_S128_B1N8_kvH4_sp0.5 | 1 | 8 | 128 | 128 | 4 | 0.5 | 2 | 1 | GQA单Q block, 稀疏 |
| G-03 | GQA_S256_B2N8_kvH4_sp0.5 | 2 | 8 | 256 | 128 | 4 | 0.5 | 2 | 2 | GQA 2 Q blocks, 多batch |
| G-04 | GQA_S512_B4N6_kvH2_sp0.5 | 4 | 6 | 512 | 128 | 2 | 0.5 | 3 | 4 | GQA groupSize=3, 4 Q blocks |
| G-05 | GQA_S1024_B1N8_kvH4_sp0.5 | 1 | 8 | 1024 | 128 | 4 | 0.5 | 2 | 8 | GQA 8 Q blocks, 重累加 |
| G-06 | GQA_S1024_B2N8_kvH2_sp1.0 | 2 | 8 | 1024 | 128 | 2 | 1.0 | 4 | 8 | GQA groupSize=4, 全掩码 |

### 三、边界与异常

| 编号 | 名称 | B | N | S | D | kvH | sp | Qblocks | 覆盖场景 |
|------|------|---|---|---|---|-----|-----|---------|---------|
| E-01 | EDGE_S128_B1N2_sp1.0_min | 1 | 2 | 128 | 128 | 2 | 1.0 | 1 | 最小有效shape |
| E-02 | EDGE_S128_B1N1_sp1.0_single | 1 | 1 | 128 | 128 | 1 | 1.0 | 1 | 单头单batch |
| E-03 | EDGE_S256_B1N2_sp0.0_sparse | 1 | 2 | 256 | 128 | 2 | 0.0 | 2 | 全零掩码(无有效KV block) |

## 预期结果

| 场景 | 精度 (vs CPU) | 确定性 (bitwise) | 说明 |
|------|--------------|-----------------|------|
| S=128 (1 Q block) | ✅ 全通过 | ✅ 全通过 | 无 SetAtomicAdd 累加 |
| S=256 (2 Q blocks) | ✅ 全通过 | ✅ 大部分通过 | 轻度累加, FIX buffer 可排空 |
| S=384~512 (3~4 Q blocks) | dV 可能失败 | ✅ 大部分通过 | dV mmad3 先于 dK, buffer 积压 |
| S=1024 (8 Q blocks) | dV 失败 | ✅ 通过 | 重累加, dV 精度受损但确定性保持 |
| GQA (groupSize>1) | 同对应S的MHA | 同对应S的MHA | GQA 不影响 SetAtomicAdd 累加次数 |

## 测试方法

```python
import torch
torch.use_deterministic_algorithms(True)  # 启用确定性

# 精度: NPU 结果 vs CPU golden, atol=1e-3, rtol=1e-3
torch.allclose(npu_dk, cpu_dk, rtol=1e-3, atol=1e-3)

# 确定性: 两次 NPU 运行 bitwise 比较
torch.equal(dk_run1, dk_run2)
```
