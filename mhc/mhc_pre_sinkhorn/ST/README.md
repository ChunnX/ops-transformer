# MhcPreSinkhorn ATK 测试脚本

## 概述

本目录包含 mhc_pre_sinkhorn 算子的 ATK 测试脚本，用于验证 NPU 实现的正确性和精度。

## 文件清单

```
ST/
目录结构
├── aclnn_mhc_pre_sinkhorn.yaml           # YAML 配置文件
├── generate_mhc_pre_sinkhorn.py          # 测试用例生成脚本
├── MhcPreSinkhorn.py                     # 测试执行脚本
├── ATK_TEST_GUIDE.md                     # 测试使用指南
└── README.md                            # 本文档
```

## 快速开始

### 1. 生成测试用例

```bash
cd /data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn/ST
atk case -f aclnn_mhc_pre_sinkhorn.yaml -p generate_mhc_pre_sinkhorn.py
```

### 2. 执行测试

```bash
atk node --backend pyaclnn --devices 3 node --backend cpu task -c *.json --task accuracy -p executor*
```

### 3. 查看测试报告

测试报告位于：`atk_output/MhcPreSinkhorn_*/report/MhcPreSinkhorn_reports_*.xlsx`

## 文件说明

### 1. aclnn_mhc_pre_sinkhorn.yaml

**作用**：定义算子的输入输出约束和测试用例生成规则

**关键内容**：
- 定义 4 个输入参数（h_res、eps、num_iters、out_flag）
- h_res 支持 3D 和 4D 输入
- 最后 n 个维度约束为 {4, 6, 8}
- 参数范围设置合理

### 2. generate_mhc_pre_sinkhorn.py

**作用**：根据 YAML 配置生成测试用例

**关键功能**：
- 继承 `CaseGenerator` 类
- 实现 `after_case_config` 方法
- 动态调整形状，确保 n 值在 {4, 6, 8} 范围内

### 3. MhcPreSinkhorn.py

**作用**：实现算子的 CPU 参考实现，用于验证 NPU 实现的正确性

**关键功能**：
- 继承 `BaseApi` 类
- 实现 `__call__` 方法
- 使用 `torch.nn.functional.linear` 实现矩阵乘法
- 实现完整的 Sinkhorn 算法逻辑

**技术亮点**：
- 使用 `torch.nn.functional.linear` 优化矩阵乘法
- 支持动态形状调整
- 完整的中间结果处理
- 数值稳定性保证

### 4. ATK_TEST_GUIDE.md

**作用**：提供完整的测试使用指南

**内容**：
- 测试脚本说明
- 使用流程
- 测试用例说明
- 精度标准
- 常见问题排查
- 性能测试方法

## 技术特性

### 1. 使用 torch.nn.functional.linear 实现矩阵乘法

**优势**：
- 性能优化：利用 PyTorch 优化的矩阵乘法实现
- GPU 加速：可以充分利用 GPU 并行计算能力
- 代码一致性：与 PyTorch 生态系统的使用方式一致

### 2. 动态形状调整

**功能**：
- 在 `after_case_config` 中动态调整形状
- 确保 n 值在 {4, 6, 8} 范围内
- 自动调整到最近的合法 n 值

### 3. 完整的中间结果处理

**功能**：
- 根据 `out_flag` 参数决定是否存储中间结果
- 支持两种输出格式（out_flag=0 和 out_flag=1）
- 易于调试和验证

### 4. 数值稳定性保证

**功能**：
- 使用 `eps` 参数防止除零
- 在归一化操作中添加 `eps` 确保数值稳定
- 使用 `F.softmax` 进行稳定的 softmax 计算

## 测试覆盖范围

### 1. 数据类型测试
- ✅ fp32

### 2. 维度测试
- ✅ 3D 输入：[T, n, n]
- ✅ 4D 输入：[B, S, n, n]

### 3. n 值测试
- ✅ n = 4
- ✅ n = 6
- ✅ n = 8

### 4. 参数范围测试
- ✅ eps: [1e-8, 1e-4]
- ✅ num_iters: [1, 100]
- ✅ out_flag: [0, 1]

### 5. 特殊值测试
- ✅ inf 值处理
- ✅ nan 值处理
- ✅ 边界值处理

## 可复用技能

本测试脚本的生成流程已总结为可复用技能：`atk-test-generator`

**技能位置**：`/data/hey/git/.trae/skills/atk-test-generator/SKILL.md`

**技能功能**：
1. 学习参考工程的测试脚本结构
2. 生成符合 ATK 规范的 YAML 配置文件
3. 生成测试用例生成脚本
4. 生成测试执行脚本（推荐使用 torch.nn.functional.linear）
5. 提供完整的测试执行命令
6. 包含反模式清单和常见陷阱处理

**使用场景**：当您需要为一个新的 Ascend C 算子创建 ATK 测试时，使用本技能可以快速生成高质量的测试脚本。

## 常见问题

### Q1: 如何修改测试用例数量？

**A**: 修改 `aclnn_mhc_pre_sinkhorn.yaml` 中的 `dtype_numbers` 字段：
```yaml
dtype_numbers: 100  # 修改为所需的数量
```

### Q2: 如何添加新的数据类型支持？

**A**: 在 `aclnn_mhc_pre_sinkhorn.yaml` 的 `dtypes.values` 中添加：
```yaml
dtypes:
  values: [fp16, fp32, bf16]  # 添加所需的数据类型
```

### Q3: 如何调整参数范围？

**A**: 在 `aclnn_mhc_pre_sinkhorn.yaml` 的 `ranges.valid.values` 中修改：
```yaml
ranges:
  valid:
    values: [ [ 1e-8, 1e-4 ] ]  # 修改为所需的范围
```

### Q4: 测试失败怎么办？

**A**: 
1. 查看测试日志：`atk_output/MhcPreSinkhorn_*/log/atk.log`
2. 检查错误信息和堆栈跟踪
3. 验证 NPU 实现与 CPU 实现的一致性
4. 检查参数约束和形状调整逻辑

### Q5: 如何提高测试性能？

**A**:
1. 减少 `dtype_numbers` 的值
2. 使用更小的测试数据规模
3. 减少 `num_iters` 的值
4. 使用 `--backend cpu` 而不是 `--backend pyaclnn` 进行快速测试

## 参考资源

- **ATK 文档**：ATK 测试框架官方文档
- **Python 标杆脚本**：`../mhc_pre_sinkhorn_python_reference.py`
- **详细设计文档**：`../docs/mhc_pre_sinkhorn_DETAILED_DESIGN.md`
- **优化总结文档**：`../OPTIMIZATION_SUMMARY.md`
- **可复用技能**：`/data/hey/git/.trae/skills/atk-test-generator/SKILL.md`

## 总结

本 ATK 测试脚本提供了完整的测试流程，包括：

1. ✅ YAML 配置文件定义算子约束
2. ✅ 测试用例生成脚本生成测试用例
3. ✅ 测试执行脚本实现 CPU 参考实现
4. ✅ 完使用 `torch.nn.functional.linear` 优化矩阵乘法
5. ✅ 完整的测试执行流程
6. ✅ 详细的测试结果分析

通过使用本测试脚本，您可以：

- 快速生成测试用例
- 执行完整的精度测试
- 分析测试结果
- 定位和修复问题
- 验证 NPU 实现的正确性

---

**文档版本**: 1.0
**最后更新**: 2026-03-27
