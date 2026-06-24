# MlaPrologV3 算子测试框架

## 文件结构

pytest/

- test.py                        # pytest泛化测试用例运行主程序
- testcases.py                   # 泛化测试用例入参配置
- check_valid_param.py           # 入参检查及精度对比
- prologv3_no_quant_pa_bsnd.py   # CPU侧算子逻辑实现获取golden，npu算子直调
- pytest.ini                     # 创建ci单算子和graph图模式的测试标记

## 功能说明

基于pytest测试框架，实现MlaPrologV3算子的功能验证：

- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

### 当前实现范围

✅ **已实现**：基础MlaPrologV3非量化PA场景 (weight_quant_mode=0, kv_cache_quant_mode=0)
❌ **未实现**：MlaPrologV3 部分量化、全量化、FP8/HIF8/MXFP8 场景

### 参数限制

- **数据格式**：BF16
- **B**：Batch表示输入样本批量大小，取值范围为1~65536
- **S1**：q_seq表示query序列长度，取值范围为1~16
- **He**：隐藏层维度，取值为7168
- **Hcq**：q低秩矩阵维度，取值为1536
- **N1**：q_head_num表示多头数，取值范围为8、16、32、64、128
- **Hckv**：kv低秩矩阵维度，取值为512
- **D**：head_dim，取值为128
- **Dr**：rope_head_dim，取值为64
- **N2**：kv_head_num，取值为1
- **BlockSize**：PagedAttention块大小，取值范围为16、128
- **cache_mode**: PA_BSND / PA_NZ
- **weight_quant_mode**：权重量化模式 (0-5)，当前仅实现0
- **tile_size**：tile大小，取值为128

### V3 新增参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| weight_quant_mode | int | 权重量化模式 0-5 | 0 |
| kv_cache_quant_mode | int | KV Cache量化模式 0-3 | 0 |
| query_quant_mode | int | Query量化模式 0-1 | 0 |
| ckvkr_repo_mode | int | CkvKr回写模式 0-1 | 0 |
| quant_scale_repo_mode | int | 量化Scale回写模式 0-1 | 0 |
| tile_size | int | PerTile规约tile大小 | 128 |
| qc_qr_scale | float | Query尺度矫正因子 | 1.0 |
| kc_scale | float | KV尺度矫正因子 | 1.0 |
| query_norm_flag | bool | 是否返回queryNorm | False |

## 环境配置

### 前置要求

1. 确认torch_npu为最新版本
2. source CANN包

## 使用方法

在pytest文件夹路径下执行：

### 运行方式：运行泛化测试用例

单算子直调+图模式

```bash
python3 -m pytest -rA -s test.py
```

单算子直调

```bash
python3 -m pytest -rA -s test.py -v -m ci
```

图模式

```bash
python3 -m pytest -rA -s test.py -v -m graph
```
