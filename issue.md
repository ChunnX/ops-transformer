# 问题
## 问题1
attention/lightning_indexer/README.md和attention/sparse_flash_attention/README.md有问题
现在我要求修改这个文档
文档内部的“## 参数说明”表格存在问题，这两个文档的参数表格字段有：参数名、输入/输出/属性、描述、使用说明、数据类型、数据格式、维度(shape)、非连续Tensor
但要求的字段只允许有：参数名、输入/输出/属性、描述、数据类型、数据格式
所以请帮我做出修改这两个文档：将原来使用说明、维度(shape)、非连续Tensor整合进描述里面，具体可以参考attention/kv_quant_sparse_flash_attention/README.md文档，精简一点，非连续Tensor字段为“x”时表诉为“不支持非连续”

## 处理记录
- 2026-04-30：已按要求修改 `attention/lightning_indexer/README.md` 和 `attention/sparse_flash_attention/README.md` 的“参数说明”表格，仅保留“参数名、输入/输出/属性、描述、数据类型、数据格式”六列，并将原“使用说明”“维度(shape)”“非连续Tensor”信息整合到“描述”列，其中非连续Tensor为“x”的条目统一表述为“不支持非连续”。
