# sparse_flash_attention pytest 框架

## 当前目标
`pytest_sfa/pytest` 已从 QSFA 拷贝骨架切换为 SFA 框架，当前优先保证 `single` / `batch` / `graph` / `npu_only` 的入参组织和算子调用链路改为 `torch_npu.npu_sparse_flash_attention(...)`。

## 当前实现状态
- 已替换文件名、测试入口名、脚本名中的 `kv_quant_sparse_flash_attention` 为 `sparse_flash_attention`
- `utils.py` 已改为使用 SFA 参数集合：`layout_query/layout_kv/q_type/B/S1/S2/N1/N2/D/K/...`
- `batch/sparse_flash_attention_process.py` 已切换到 `torch_npu.npu_sparse_flash_attention`
- `sparse_flash_attention_golden.py` 当前负责组装 `.pt` 用例，CPU golden 暂时返回 `None`，因此结果记录为 `SkipCompare`
- 已按 `sparse_flash_attention_tiling.cpp` 增加参数拦截：`N2==1`、`D==512`、`rope_head_dim==64`、`attention_mode==2`、`sparse_mode in {0,3}`、`PA_BSND` 额外约束等

## 参数结构
- `single` 参数来自 `sparse_flash_attention_paramset.py`
- `batch` 参数来自 `excel/example.xlsx`
- 客户侧仍使用简写字段：`B/S1/S2/N1/N2/D/K`
- `utils.sfa()` 负责将这些简写参数转换为算子原型需要的 tensor 输入

## 数据生成
- `query/key/value` 使用 `FP16/BF16` 随机张量
- `query_rope/key_rope` 独立生成并作为算子可选输入传入
- `sparse_indices` 按 `sparse_mode` 和 `actual_seq_*` 生成
- `BSND/TND/PA_BSND` 三类布局都提供了基础输入拼装能力

## 用法
```bash
cd pytest_sfa/pytest
bash test_run.sh single
bash test_run.sh batch
bash test_run.sh graph
bash test_run.sh npu_only
```

## 说明
- `single` / `batch` 当前以先打通 NPU 调用为目标，CPU golden 后续可再补齐
- `perf_run.sh` 仍依赖 `msprof` 与本地 profiling 目录结构；若环境缺少 `msprof`，脚本会自动降级为普通 batch 回放
