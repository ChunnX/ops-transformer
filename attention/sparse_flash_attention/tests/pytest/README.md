# sparse_flash_attention pytest 测试框架



## 功能说明

基于pytest测试框架，实现 sparse_flash_attention 算子的功能验证：

-   **CPU侧**：复现算子功能用以生成golden数据
-   **NPU侧**：通过torch_npu进行算子直调获取实际数据
-   **精度对比**：进行CPU与NPU结果的精度对比验证算子功能

保留 single 和 batch 两条主流程：

- `single`：基于 `sparse_flash_attention_paramset.py` 的固定参数直接构造输入并拉起 NPU 单算子执行。
- `batch`：先从 Excel 读取参数生成 `.pt` 用例，再批量回放执行 NPU 算子。



## 当前实现范围

### 参数约束
当前框架已在 `check_valid_param.py` 中统一拦截以下关键约束：

- `layout_query`：支持 `BSND`、`TND`
- `layout_kv`：支持 `BSND`、`TND`、`PA_BSND`
- 非 `PA_BSND` 场景要求 `layout_query == layout_kv`
- `q_type`：仅支持 `torch.float16`、`torch.bfloat16`
- `N2`：仅支持 `1`
- `g = N1 / N2`：仅支持 `1/2/4/8/16/32/64/128`
- `D`：当前仅支持 `512`
- `rope_head_dim`：当前仅支持 `64`
- `attention_mode`：当前仅支持 `2`
- `sparse_mode`：当前仅支持 `0`、`3`
- `sparse_block_size`：需位于 `[1, 128]` 且为 2 的幂
- `K`：不能超过 `ceil(S2 / sparse_block_size)`
- `PA_BSND` 场景：`block_size` 需为正整数且 16 对齐，`block_num` 需覆盖实际 KV 长度

更完整的算子定义可参考 `attention/sparse_flash_attention/README.md`。

### 环境配置

#### 前置要求

1、 确认torch_npu为最新版本
2、 参考[Attention融合算子Experimental使用说明](https://gitcode.com/cann/ops-transformer/blob/master/attention/Attention融合算子Experimental使用说明.md)激活CANN包和自定义算子包

#### custom包调用

支持custom包调用



## 目录结构
```text
pytest/
├── README.md
├── pytest.ini
├── test_run.sh
├── check_valid_param.py                  # 参数约束拦截
├── sparse_flash_attention_golden.py      # 输入构造 / pt 用例生成
├── sparse_flash_attention_paramset.py    # single 模式参数集
├── result_compare_method.py              # 输出精度对比
├── utils.py                              # Excel 解析 / 参数展开 / 执行入口
├── test_sparse_flash_attention_single.py # single 模式主程序
├── test_sparse_flash_attention_batch.py  # batch 回放主程序
└── batch/
    ├── sparse_flash_attention_process.py # NPU 直调与 graph 调用
    └── test_sparse_flash_attention_pt_save.py
```



## 使用方法
在 `attention/sparse_flash_attention/tests/pytest` 目录下执行：

### single
手动修改 `sparse_flash_attention_paramset.py` 中的参数后执行：

```bash
bash test_run.sh single
```

### batch
batch 流程依赖 `./excel/example.xlsx`，执行时会先生成 `.pt` 用例，再回放执行。

```bash
bash test_run.sh batch
```

## 结果文件
- `result.xlsx`：记录每个用例的关键信息、执行状态与 `fulfill_percent`
- `./pt_files/*.pt`：batch 流程生成的中间测试用例

