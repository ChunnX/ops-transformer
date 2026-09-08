# TurboQuantSparseFlashAttention兼容接口

标准接口为`cann_ops_transformer.turbo_quant_sparse_flash_attention`。
本可选Python包将旧`custom_ops`入口转发至标准接口。

## 构建与安装

先按[《源码构建》](../../../docs/zh/install/compile.md)安装算子包，并按[《PyTorch扩展构建与安装》](../../../torch_extension/README.md#build--installation)安装标准wheel，再在仓库根目录执行：

```bash
cd attention/turbo_quant_sparse_flash_attention/torch_ops_extension
bash build_and_install.sh
```

## 调用说明

```python
import torch
import custom_ops  # noqa: F401

result = torch.ops.custom.npu_turbo_quant_sparse_flash_attention(...)
```

以上片段展示旧入口名称，完整参数及可运行样例见[《PyTorch接口》](../../../torch_extension/cann_ops_transformer/docs/zh/turbo_quant_sparse_flash_attention.md)。

导入`custom_ops`后，也可通过`torch_npu.npu_turbo_quant_sparse_flash_attention`访问该接口。

从旧原生扩展升级后，应重启Python进程，避免在同一进程加载两个版本。新应用建议直接使用标准接口。
