# FusedQuantLightningIndexerManage 测试

## 测试环境

测试依赖支持 Ascend 950 的 CANN、PyTorch、TorchNPU，以及已安装的自定义 OPP 包和单算子 Torch 扩展。

从仓库根目录构建并安装：

```bash
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=fused_quant_lightning_indexer_manage -j8
./build_out/cann-ops-transformer-custom_linux-aarch64.run

bash build.sh --torch_extension \
  --ops=fused_quant_lightning_indexer_manage --vendor_name=custom
python3 -m pip install \
  build_out/cann_ops_transformer_custom-1.0.0-py3-none-any.whl \
  --force-reinstall --no-deps
```

按照安装程序输出设置 `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH`。一个 Python 进程只能加载一套 `custom_transformer` OPP；不要同时配置外层目录及其 `vendors/custom_transformer` 子目录。

## 运行测试

进入当前目录执行：

```bash
DEVICE_ID=0 bash test_run.sh
```

等价命令：

```bash
DEVICE_ID=0 python3 -m pytest -rA -s \
  test_fused_quant_lightning_indexer_manage.py -v -m ci
```

测试使用官方 `npu_quant_lightning_indexer` 作为 TopK 参考，覆盖：

- `-3` 非卸载状态；
- `-2` 首次填充；
- `-1` 稳态 Miss、淘汰和 Pool 更新；
- 非法请求的原子失败；
- Batch 内混合状态及 Pool 行隔离；
- 262144-token/18-bit Source ID 边界；
- NPUGraph 重放与 eager 结果一致性。

每类场景是独立 pytest case，可通过 `-k` 单独执行，例如：

```bash
python3 -m pytest -rA -s \
  test_fused_quant_lightning_indexer_manage.py \
  -v -m ci -k first_fill
```
