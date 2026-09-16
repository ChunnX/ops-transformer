# fused_scatter_copy_quant_sparse_flash_attention算子测试框架

## 功能说明

基于pytest测试框架，实现FusedScatterCopyQuantSparseFlashAttention算子的最小功能验证：

- **Scatter Copy校验**：逐字节比较HBM C8 KV Cache和CPU golden。
- **Attention校验**：独立解码C8 KV数据，并比较融合算子输出和CPU Attention golden。
- **路径覆盖**：`miss_count=8`覆盖融合miss路径，`miss_count=513`覆盖完整Scatter Copy fallback路径。

## 当前实现范围

- Batch Size：1。
- Query Head：8。
- MTP：MTP0，即单请求包含1个Query。
- Query数据类型：BFLOAT16。
- 稀疏索引shape：`[1, 1, 2048]`。
- Copy索引shape：`[1, 1, 32768]`。
- KV Cache数据格式：C8 INT8字节视图，shape为`[18, 128, 1, 656]`。

## 环境配置

1. 安装支持Ascend 950的CANN和TorchNPU。
2. 编译并安装FusedScatterCopyQuantSparseFlashAttention自定义算子包。
3. 激活自定义算子包环境。

## 文件结构

- `test_run.sh`：测试执行脚本。
- `test_fused_scatter_copy_quant_sparse_flash_attention.py`：NPU算子调用和结果校验。
- `fused_scatter_copy_quant_sparse_flash_attention_golden.py`：测试数据和CPU golden生成。
- `pytest.ini`：pytest标记配置。

## 使用方法

在当前目录执行：

```bash
bash test_run.sh
```

指定NPU设备：

```bash
DEVICE_ID=6 bash test_run.sh
```
