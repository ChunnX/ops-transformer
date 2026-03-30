# MhcPreSinkhorn ATK 测试脚本使用指南

## 概述

本文档提供了 mhc_pre_sinkhorn 算子的 ATK 测试脚本完整使用指南，包括测试用例生成、测试执行和结果分析。

## 文件清单

```
mhc_pre_sinkhorn/
├── aclnn_mhc_pre_sinkhorn.yaml           # YAML 配置文件
├── generate_mhc_pre_sinkhorn.py          # 测试用例生成脚本
├── MhcPreSinkhorn.py                     # 测试执行脚本（CPU 参考实现）
└── ATK_TEST_GUIDE.md                     # 本文档
```

## 测试脚本说明

### 1. YAML 配置文件 (aclnn_mhc_pre_sinkhorn.yaml)

**作用**：定义算子的输入输出约束和测试用例生成规则

**关键内容**：
- `api`: API 类型（pytorch）
- `api_type`: 算子类型（ascend_function_mhc_pre_sinkhorn）
- `aclnn_name`: ACLNN 接口名称（MhcPreSinkhorn）
- `generate`: 生成器名称（ascend_generate_mhc_pre_sinkhorn）
- `inputs`: 输入参数定义
  - `h_res`: 输入张量（tensor 类型，fp32，3D 或 4D）
  - `eps`: 防除零参数（attr 类型，float，范围 [1e-8, 1e-4]）
  - `num_iters`: 迭代次数（attr 类型，int，范围 [1, 100]）
  - `out_flag`: 输出标志位（attr 类型，int，范围 [0, 1]）

### 2. 测试用例生成脚本 (generate_mhc_pre_sinkhorn.py)

**作用**：根据 YAML 配置生成测试用例

**关键功能**：
- 继承 `CaseGenerator` 类
- 实现 `after_case_config` 方法
- 确保最后两个维度相等且在 {4, 6, 8} 范围内

**特殊处理**：
```python
def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
    h_res_shape = case_config.inputs[0].shape
    
    # 确保最后两个维度相等且在 {4, 6, 8}
    n = h_res_shape[-1]
    if n not in [4, 6, 8]:
        valid_n_values = [4, 6, 8]
        n = min(valid_n_values, key=lambda x: abs(x - n))
        h_res_shape = list(h_res_shape)
        h_res_shape[-1] = n
        h_res_shape[-2] = n
        h_res_shape = tuple(h_res_shape)
    
    case_config.inputs[0].shape = h_res_shape
    
    return case_config
```

### 3. 测试执行脚本 (MhcPreSinkhorn.py)

**作用**：实现算子的 CPU 参考实现，用于验证 NPU 实现的正确性

**关键功能**：
- 继承 `BaseApi` 类
- 实现 `__call__` 方法
- 实现完整的 Sinkhorn 算法逻辑

**算法实现**：
```python
# 第一次迭代：Softmax（按行）+ 列归一化
current = softmax_func(h_res_np, axis=-1, eps=eps)
current, col_sums = col_normalize(current, eps=eps)

# 后续迭代：行归一化 + 列归一化
for iter_idx in range(1, num_iters):
    current, row_sums = row_normalize(current, eps=eps)
)
    current, col_sums = col_normalize(current, eps=eps)
```

**中间结果处理**：
- 当 `out_flag=1` 时，存储中间结果到 `norm_out` 和 `sum_out`
- 当 `out_flag=0` 时，返回 `None` 作为中间结果

## 使用流程

### 步骤 1：生成测试用例

**命令**：
```bash
cd /data/hey/git/ops-transformer/mhc/mhc_pre_sinkhorn
atk case -f aclnn_mhc_pre_sinkhorn.yaml -p generate_mhc_pre_sinkhorn.py
```

**说明**：
- `-f`: 指定 YAML 配置文件
- `-p`: 指定测试用例生成脚本

**输出**：
- 生成 JSON 格式的测试用例文件
- 文件名格式：`MhcPreSinkhorn_*.json`

### 步骤 2：执行测试

**命令**：
```bash
atk node --backend pyaclnn --devices 3 node --backend cpu task -c *.json --task accuracy -p executor*
```

**说明**：
- `--backend pyaclnn`: 使用 NPU 后端
- `--devices 3`: 指定 NPU 设备 ID
- `--backend cpu`: 使用 CPU 后端（参考实现）
- `-c *.json`: 指定测试用例文件
- `--task accuracy`: 执行精度测试任务
- `-p executor*`: 指定执行器模式

**输出**：
- 测试日志：`atk_output/MhcPreSinkhorn_*/log/atk.log`
- 测试报告：`atk_output/MhcPreSinkhorn_*/report/MhcPreSinkhorn_reports_*.xlsx`

### 步骤 3：查看测试结果

**查看日志**：
```bash
cat atk_output/MhcPreSinkhorn_*/log/atk.log
```

**查看报告**：
- 打开 Excel 文件：`atk_output/MhcPreSinkhorn_*/report/MhcPreSinkhorn_reports_*.xlsx`
- 报告包含：
  - 测试用例信息
  - 精度对比结果
  - 性能数据
  - 错误信息（如果有）

## 测试用例说明

### 生成的测试用例类型

1. **不同数据类型**：fp32
2. **不同维度**：3D 和 4D
3. **不同 n 值**：4、6、8
4. **不同迭代次数**：1-100
5. **不同 eps 值**：1e-8 到 1e-4
6. **不同 out_flag**：0 和 1

### 测试用例数量

- 默认生成 100 个测试用例（`dtype_numbers: 100`）
- 额外测试用例数量：0（`extra_numbers: 0`）

## 精度标准

### 默认精度标准

- **绝对误差容限**：1e-5
- **相对误差容限**：1e-3
- **双随机性验证**：行和列的和都接近 1（误差 < 1e-5）

### 精度验证方法

1. **逐元素对比**：对比 NPU 和 CPU 计算结果的每个元素
2. **双随机性验证**：验证输出矩阵的行和列都接近 1
3. **非负性验证**：验证所有元素都 >= 0

## 常见问题排查

### 问题 1：测试用例生成失败

**可能原因**：
- YAML 配置文件格式错误
- 生成脚本语法错误
- 参数约束设置不合理

**解决方法**：
1. 检查 YAML 文件格式
2. 检查查生成脚本语法
3. 调整参数约束范围

### 问题 2：测试执行失败

**可能原因**：
- NPU 设备不可用
- ACLNN 接口未正确注册
- 测试用例参数不合法

**解决方法**：
1. 检查 NPU 设备状态
2. 确认 ACLNN 接口已正确编译和安装
3. 检查测试用例参数

### 问题 3：精度测试不通过

**可能原因**：
- NPU 实现与 CPU 实现不一致
- 数值稳定性问题
- eps 参数设置不合理

**解决方法**：
1. 检查 NPU 实现逻辑
2. 调整 eps 参数
3. 增加迭代次数
4. 检查数值稳定性

### 问题 4：中间结果输出错误

**可能原因**：
- `out_flag` 参数处理不正确
- 中间结果存储逻辑错误
- 维度转换错误

**解决方法**：
1. 检查 `out_flag` 参数处理逻辑
2. 验证中间结果存储格式
3. 检查维度转换函数

## 性能测试

### 性能指标

- **AI Core 利用率**：> 80%
- **内存带宽利用率**：> 70%
- **计算延迟**：< 1000 us/op

### 性能测试方法

1. **基准测试**：使用 CPU 实现作为基准
2. **NPU 测试**：使用 NPU 实现进行测试
3. **性能对比**：对比 NPU 和 CPU 的性能差异

### 性能优化建议

1. **优化 Tiling 策略**：提高内存利用率
2. **优化并行计算**：提高 AI Core 利用率
3. **优化数据搬运**：减少 GM ↔ UB 搬运次数

## 测试报告分析

### 报告内容

1. **测试用例信息**
   - 输入参数
   - 输出形状
   - 数据类型

2. **精度对比结果**
   - 最大误差
   - 平均误差
   - 通过/失败状态

3. **性能数据**
   - 执行时间
   - AI Core 利用率
   - 内存带宽利用率

4. **错误信息**
   - 错误类型
   - 错误详情
   - 堆栈信息

### 报告解读

- **通过**：所有测试用例都通过精度验证
- **失败**：存在测试用例未通过精度验证
- **警告**：存在潜在问题但不影响测试结果

## 总结

本测试脚本提供了完整的 ATK 测试流程，包括：

1. ✅ YAML 配置文件定义算子约束
2. ✅ 测试用例生成脚本生成测试用例
3. ✅ 测试执行脚本实现 CPU 参考实现
4. ✅ 完整的测试执行流程
5. ✅ 详细的测试结果分析

通过遵循本指南，您可以：

- 快速生成测试用例
- 执行完整的精度测试
- 分析测试结果
- 定位和修复问题

## 参考资源

- **ATK 文档**：ATK 测试框架官方文档
- **Python 标杆脚本**：`mhc_pre_sinkhorn_python_reference.py`
- **详细设计文档**：`mhc_pre_sinkhorn_DETAILED_DESIGN.md`
- **优化总结文档**：`OPTIMIZATION_SUMMARY.md`
