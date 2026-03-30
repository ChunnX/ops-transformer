# MHC_Pre_Sinkhorn ATK 测试指南

## 概述

本文档提供 mhc_pre_sinkhorn 算子的 ATK 测试使用指南。

## 文件结构

```
mhc_pre_sinkhorn/
└── ST/
    ├── aclnn_mhc_pre_sinkhorn.yaml    # YAML 配置文件
    ├── generate_mhc_pre_sinkhorn.py   # 测试用例生成脚本
    ├── MhcPreSinkhorn.py              # 测试执行脚本
    ├── test_npu_hc_pre.py             # PTA 测试脚本（标杆实现）
    └── ATK_TEST_GUIDE.md              # 本文档
```

## 测试参数说明

### 输入参数

| 参数名 | 类型 | 形状 | 数据类型 | 说明 |
|--------|------|------|---------|------|
| x | tensor | (bs, seq_len, n, c) | BF16 | 输入张量 |
| phi | tensor | (hcMix, n*c) | FP32 | 权重矩阵 |
| alpha | tensor | (3,) | FP32 | 缩放系数 |
| bias | tensor | (hcMix,) | FP32 | 偏置向量 |
| hcMult | attr | - | int | 多头数量，取值范围 [4, 8] |
| numIters | attr | - | int | Sinkhorn迭代次数，固定为 20 |
| hcEps | attr | - | float | hc epsilon值，固定为 1e-6 |
| normEps | attr | - | float | 归一化epsilon值，固定为 1e-6 |
| needBackward | attr | - | bool | 是否需要反向传播 |

### 输出参数

| 参数名 | 类型 | 形状 | 数据类型 | 说明 |
|--------|------|------|---------|------|
| hin | tensor | (bs, seq_len, c) | BF16 | Attention/MLP层输入 |
| hPost | tensor | (bs, seq_len, n) | FP32 | 后处理变换矩阵 |
| hRes | tensor | (bs, seq_len, n*n) | FP32 | 未做Sinkhorn的组合矩阵 |
| hPre | tensor | (bs, seq_len, n) | FP32 | sigmoid(H_pre) + hc_eps |
| hcBeforeNorm | tensor | (bs, seq_len, hcMix) | FP32 | x @ phi 矩阵乘法结果 |
| invRms | tensor | (bs, seq_len, 1) | FP32 | RmsNorm的1/r结果 |
| sumOut | tensor | (numIters*2, bs, seq_len, n) | FP32 | Sinkhorn迭代中间sum |
| normOut | tensor | (numIters*2, bs, seq_len, n, n) | FP32 | Sinkhorn迭代中间norm |

**注意**: hPre, hcBeforeNorm, invRms, sumOut, normOut 仅在 needBackward=True 时有效。

## 执行测试

### 1. 生成测试用例

```bash
cd /data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/ST
atk case -f aclnn_mhc_pre_sinkhorn.yaml -p generate_mhc_pre_sinkhorn.py
```

### 2. 执行测试

```bash
atk node --backend pyaclnn --devices 3 node --backend cpu task -c *.json --task accuracy -p executor*
```

## 形状约束

生成脚本 `generate_mhc_pre_sinkhorn.py` 会自动处理以下形状约束：

1. **n 值约束**: n 必须在 {4, 6, 8} 范围内，否则自动调整到最近的合法值
2. **phi 形状**: 根据 n 和 c 自动计算，形状为 (n*n + 2*n, n*c)
3. **alpha 形状**: 固定为 (3,)
4. **bias 形状**: 根据 n 自动计算，形状为 (n*n + 2*n,)

## 标杆实现

标杆实现位于 `test_npu_hc_pre.py`，核心函数：

- `hc_split_sinkhorn_torch_full`: 完整的 Sinkhorn 迭代实现
- `_hc_pre_full`: 完整的算子标杆实现，支持 8 个输出

## 注意事项

1. **数值稳定性**: 所有除法操作都添加了 eps 参数防止除零
2. **矩阵乘法优化**: 使用 `torch.nn.functional.linear` 实现矩阵乘法
3. **可选输出处理**: needBackward=False 时，可选输出返回 None
4. **数据类型**: x 为 BF16，其他张量为 FP32

## 质量检查清单

- [x] YAML 文件格式正确，所有必需字段都已填写
- [x] 生成脚本继承自 CaseGenerator 并实现了 after_case_config
- [x] 测试执行脚本继承自 BaseApi 并实现了 __call__
- [x] 所有输入参数的约束都已正确定义
- [x] 特殊值（inf、nan）的处理逻辑已实现
- [x] 可选输出（needBackward）的处理逻辑已实现
- [x] 数值稳定性（eps 参数）已考虑
- [x] 矩阵乘法使用 torch.nn.functional.linear 实现
