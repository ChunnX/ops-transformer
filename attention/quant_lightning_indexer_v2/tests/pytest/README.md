# quant_lightning_indexer_v2算子测试框架
## 功能说明
基于pytest测试框架，实现quant_lightning_indexer_v2算子的功能验证：
- **CPU侧**：复现算子功能用以生成golden数据
- **NPU侧**：通过torch_npu进行算子直调获取实际数据
- **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

## 当前实现范围
### 参数限制

- **数据格式**:
  - **query_layout**：BSND、TND
  - **key_layout**: PA_BSND

### 环境配置

#### 前置要求
1、 确认torch_npu为最新版本  
2、 参考[Attention融合算子使用说明](../../../docs/zh/op_api_list.md)激活CANN包和自定义算子包

#### custom包调用
支持custom包调用

## 文件结构
#### pytest文件结构说明
- test_run.sh                                        # 执行脚本
- quant_lightning_indexer_v2_golden.py               # cpu侧算子golden实现
- quant_lightning_indexer_v2_acl_graph.py            # aclgraph模式调用
- result_compare_method.py                           # cpu golden与npu输出精度对比
- pytest.ini                                         # 创建测试标记

单用例测试：
- test_quant_lightning_indexer_v2_single.py          # pytest测试单用例运行主程序 
- test_quant_lightning_indexer_v2_paramset.py        # 单用例入参配置

批量测试：
- test_quant_lightning_indexer_v2_batch.py           # 用例批量测试主程序并生成excel文件保存结果
- ./batch/                                           # 批量测试辅助脚本目录
 
## 使用方法
在pytest文件夹路径下执行：

### 运行测试用例
#### 单用例调测
1、手动配置test_quant_lightning_indexer_v2_paramset.py的参数

2、执行指令：
``` bash
bash test_run.sh single
```
#### 用例的批量生成与测试
1、excel路径下存放用例excel表格
##### 批跑用例表格
批跑测试通过Excel文件配置测试用例参数，由`tests/pytest/test_run.sh batch`自动读取`tests/pytest/excel/`目录下的Excel文件并执行批量测试。用户需自行在该目录下创建Excel文件，格式如下表所示（共33列）：

| 列名 | 说明 |
|:-----|:-----|
| Testcase_Name | 用例名称 |
| batch_size | Batch大小 |
| q_seq | Query的Sequence Length |
| k_seq | Key的Sequence Length |
| q_t_size | Query的Total Tokens（TND布局使用，BSND时填0） |
| k_t_size | Key的Total Tokens（TND布局使用，PA_BSND时填0） |
| q_head_num | Query的Head Num |
| k_head_num | Key的Head Num（仅支持1） |
| head_dim | Head Dim（仅支持128） |
| block_size | PageAttention场景下每个block的token数（需为16的整数倍，最大1024） |
| block_num | PageAttention场景下的block总数 |
| qk_dtype | Query和Key的数据类型 |
| dequant_dtype | Dequant Scale的数据类型 |
| actual_seq_dtype | Actual Seq Lengths的数据类型 |
| cu_seqlens_q | Query的cu_seqlens（TND布局使用，BSND时填None） |
| cu_seqlens_k | Key的cu_seqlens（TND布局使用，PA_BSND时填None） |
| seqused_q | Query的seqused（BSND布局使用，TND时填None） |
| seqused_k | Key的seqused（PA_BSND/BSND布局使用） |
| cmp_residual_k | Key的压缩残余长度（cmp_ratio>1且mask_mode!=0时必填，否则填None） |
| max_seqlen_q | Query的最大序列长度（-1表示不限制） |
| quant_mode | 量化模式（A3: 2; 950: 1或4） |
| layout_query | Query数据排布格式 |
| layout_key | Key数据排布格式 |
| sparse_count | TopK保留的索引数量 |
| sparse_mode | Sparse模式（0: defaultMask, 3: rightDownCausal） |
| query_datarange | Query数据范围（逗号分隔的最小值,最大值） |
| key_datarange | Key数据范围（逗号分隔的最小值,最大值） |
| weights_datarange | Weights数据范围（逗号分隔的最小值,最大值） |
| q_scale_datarange | Query Dequant Scale数据范围（逗号分隔的最小值,最大值） |
| k_scale_datarange | Key Dequant Scale数据范围（逗号分隔的最小值,最大值） |
| cmp_ratio | Key的压缩倍数 |
| return_value | 是否输出sparse_values（0: 不输出, 1: 输出，仅950支持） |
| output_idx_offset | 输出索引偏移量 |

批跑测试用例表格格式示例如下：

| Testcase_Name | batch_size | q_seq | k_seq | q_t_size | k_t_size | q_head_num | k_head_num | head_dim | block_size | block_num | qk_dtype | dequant_dtype | actual_seq_dtype | cu_seqlens_q | cu_seqlens_k | seqused_q | seqused_k | cmp_residual_k | max_seqlen_q | quant_mode | layout_query | layout_key | sparse_count | sparse_mode | query_datarange | key_datarange | weights_datarange | q_scale_datarange | k_scale_datarange | cmp_ratio | return_value | output_idx_offset |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|:---|:---|:---|:---|:---|:---|:---:|:---:|:---|:---|:---:|:---:|:---|:---|:---|:---|:---|:---:|:---:|:---:|
| test001 | 4 | 2 | 0 | 0 | 0 | 64 | 1 | 128 | 128 | 4 | INT8 | FP16 | | | | | 2,2,2,2 | | -1 | 2 | BSND | PA_BSND | 512 | 3 | -128,127 | -128,127 | 0,0.01 | 0,10 | 0,10 | 1 | 0 | 0 |

2、test_run.sh中设置读取的用例excel表格路径和pt文件存放路径

3、执行指令：
``` bash
bash test_run.sh batch
```
