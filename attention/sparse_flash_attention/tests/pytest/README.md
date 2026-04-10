# sparse_flash_attention pytest 测试框架

## 功能说明
基于 pytest 测试框架，完成 `sparse_flash_attention` 算子的用例构造、NPU 执行与结果记录：

- `single`：基于 `sparse_flash_attention_paramset.py` 的固定参数直接构造输入并执行 NPU 单算子。
- `batch`：先从 Excel 读取参数生成 `.pt` 用例，再批量回放执行 NPU 算子。
- 当前 `cpu_output` 仍未补齐，因此结果文件会记录为 `SkipCompare`，现阶段重点是验证参数组织、数据落盘和 NPU 连通性。

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

## 参数约束
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

## Excel 字段
`batch` 模式默认读取 `./excel/example.xlsx` 的 `decode` sheet，必填列如下：

| Testcase_Prefix | Testcase_Number | layout_query | layout_kv | q_type | B | S1 | S2 | N1 | N2 | D | K | scale_value | sparse_block_size | rope_head_dim | sparse_mode | attention_mode | return_softmax_lse | block_size | block_num | actual_seq_q | actual_seq_kv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

说明：
- `actual_seq_q` / `actual_seq_kv` 支持形如 `[5, 8]` 的列表字符串。
- `block_num` 允许为空；当 `layout_kv == PA_BSND` 且该列为空时，框架会按 `actual_seq_kv` 自动推导。
- Excel 某列同时存在空值和整数时，pandas 可能将该列读成浮点；框架已在解析阶段统一回收为整数，避免 `block_num` 一类字段落成 `float`。

## 使用方法
在 `attention/sparse_flash_attention/tests/pytest` 目录下执行：

### single
手动修改 `sparse_flash_attention_paramset.py` 中的参数后执行：

```bash
bash test_run.sh single
```

### batch
`batch` 流程会先生成 `.pt` 用例，再执行批量回放：

```bash
bash test_run.sh batch
```

## 结果文件
- `result.xlsx`：记录每个用例的关键信息、执行状态与 `fulfill_percent`
- `./pt_files/*.pt`：batch 流程生成的中间测试用例

## 本次整改说明
本目录已按 `attention/kv_quant_sparse_flash_attention/tests/pytest` 的结构做编程规范整改，重点包括：

- 补齐统一版权头和更清晰的函数注释
- 收敛 `single` / `batch` 脚本的线程入口、异常处理与路径收集逻辑
- 将 Excel 解析、结果落盘、NPU 调用入口拆成更明确的辅助函数
- README 同步更新为 sparse_flash_attention 当前真实能力，避免沿用 QSFA 拷贝后的过时描述
