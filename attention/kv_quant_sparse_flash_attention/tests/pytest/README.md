# kv_quant_sparse_flash_attention pytest 测试框架



## 功能说明

基于pytest测试框架，实现 kv_quant_sparse_flash_attention 算子的功能验证：

-   **CPU侧**：复现算子功能用以生成golden数据
-   **NPU侧**：通过torch_npu进行算子直调获取实际数据
-   **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

支持三条主流程：

- `single`：基于 `kv_quant_sparse_flash_attention_paramset.py` 的固定参数直接构造输入并拉起 NPU 单算子执行。
- `batch_save`：从 Excel 读取参数，生成包含 CPU golden 的 `.pt` 用例文件。
- `batch_exec`：从已有 `.pt` 文件批量回放执行 NPU 算子并对比精度。



## 当前实现范围

### 参数说明

以下参数约束已经下沉到框架校验和本文档中，便于统一维护：

- `layout_query`: 支持 `BSND`、`TND`
- `layout_kv`: 支持 `BSND`、`TND`、`PA_BSND`
- 非 PA 场景要求 `layout_query == layout_kv`
- `q_type`: 仅支持 `torch.float16`、`torch.bfloat16`
- `kv_dtype`: 支持 `hifloat8`、`float8_e4m3fn`，也兼容 `None` 作为 `float8_e4m3fn` 默认生成路径
- `N1`: 仅支持 `1/2/4/8/16/32/48/64`
- `N2`: 仅支持 `1`
- `sparse_mode`: 仅支持 `0` 和 `3`
- `sparse_block_size`: 当前仅支持 `1`
- `key_quant_mode` / `value_quant_mode`: 仅支持 `2`
- `tile_size`: 当前仅支持 `128`
- `quant_scale_repo_mode`: 当前仅支持 `1`
- `attention_mode`: 支持 `0`、`2`；当取 `2` 时，`rope_head_dim` 必须为 `64`
- `block_size`: 仅在 `PA_BSND` 生效，且要求为 16 的倍数
- `actual_seq_q` / `actual_seq_kv`: 如果传入，长度必须等于 `B`

更完整的算子定义和输入约束，请同步参考：

- `attention/kv_quant_sparse_flash_attention/README.md`

### 环境配置

#### 前置要求

1、 确认torch_npu为最新版本
2、 参考[Attention融合算子Experimental使用说明](https://gitcode.com/cann/ops-transformer/blob/master/attention/Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包

#### custom包调用

支持custom包调用



## 文件结构

```text
pytest/
├── README.md
├── pytest.ini			# 创建测试标记
├── test_run.sh			# 执行脚本
├── check_valid_param.py			# 参数约束拦截
├── kv_quant_sparse_flash_attention_golden.py		# tensor转换/cpu侧算子golden实现
├── kv_quant_sparse_flash_attention_paramset.py		# 单用例入参配置
├── result_compare_method.py		# 输出精度对比
├── utils.py			# 参数解析/cpu npu执行入口
├── test_kv_quant_sparse_flash_attention_single.py	# 单用例运行主程序
├── test_kv_quant_sparse_flash_attention_batch.py	# 从 pt 文件批量执行 NPU 测试
└── batch/
    ├── kv_quant_sparse_flash_attention_process.py	# npu接口
    ├── test_kv_quant_sparse_flash_attention_pt_save.py		# 从 Excel 批量生成 pt 文件
    └── excel/
        ├── example.xlsx		# 示例 Excel 用例文件
        └── gen_example_xlsx.py		# 生成示例 Excel 的脚本
```



## 使用方法

在 `attention/kv_quant_sparse_flash_attention/test/pytest` 目录下执行：

### single

手动配置kv_quant_sparse_flash_attention_paramset.py的参数

```bash
bash test_run.sh single
```

### batch_save

从 `./batch/excel/example.xlsx` 读取参数，生成包含 CPU golden 的 `.pt` 用例文件到 `./pt_files/`。

```bash
bash test_run.sh batch_save
```

### batch_exec

从 `./pt_files/` 下的 `.pt` 文件批量回放执行 NPU 算子并对比精度。

```bash
bash test_run.sh batch_exec
```

下面给一个可直接参考的 Excel 用例，列名需与 batch 框架读取字段保持一致：

| Testcase_Prefix | Testcase_Number | layout_query | layout_kv | q_type | kv_dtype | B | S1 | S2 | N1 | N2 | D | K | scale_value | key_quant_mode | value_quant_mode | sparse_block_size | tile_size | rope_head_dim | sparse_mode | attention_mode | quant_scale_repo_mode | block_size | block_num | actual_seq_q | actual_seq_kv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tnd_sample | 1 | TND | TND | torch.bfloat16 | hifloat8 | 2 | 8 | 8 | 16 | 1 | 512 | 4 | 0.04166666666666666 | 2 | 2 | 1 | 128 | 64 | 3 | 2 | 1 | 256 |  | [5,8] | [6,8] |



## 结果文件

- `result.xlsx`：记录每个用例的关键信息、执行状态与 `fulfill_percent`
- `./pt_files/*.pt`：batch 流程生成的中间测试用例
