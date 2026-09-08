# TurboQuantSparseFlashAttention 构建与安装

## 算子包与 PyTorch 包

在仓库根目录执行以下命令。A2 使用 `ascend910b`，A3 使用 `ascend910_93`。

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --pkg --ops=turbo_quant_sparse_flash_attention --soc=ascend910b --vendor_name=custom -j8
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=/your/operator/install
source /your/operator/install/vendors/custom_transformer/bin/set_env.bash
python3 -m pip install build
bash build.sh --torch_extension
python3 -m pip install --force-reinstall --no-deps build_out/cann_ops_transformer-*.whl
```

ACLNN 头文件、Host 和 Kernel 必须来自同一次构建。升级时完整更新算子包；增量构建也需重新生成 ACLNN 入口。

标准内置包使用安装目录中的 `libopapi_transformer.so`；以上自定义验证包使用 `libcust_opapi.so`。接口调用示例见 [ACLNN 接口](aclnnTurboQuantSparseFlashAttention.md)，验证程序的编译运行方式见[测试说明](../tests/README.md)。

旧 Python 调用入口的安装方式见[兼容说明](../torch_ops_extension/README.md)。

## Python 包导入检查

安装后检查实际加载的包路径：

```bash
python3 -c 'import cann_ops_transformer; print(cann_ops_transformer.__file__)'
```

使用虚拟环境时，该路径应指向新安装的 wheel。CANN 的 `set_env.sh` 可能通过 `PYTHONPATH` 优先加载环境自带的同名包；如加载了旧包，将虚拟环境的 `site-packages` 加到 `PYTHONPATH` 首位后重启 Python 进程。
