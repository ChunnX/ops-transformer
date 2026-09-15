# FlashMlaMetadata UT

| 目录 / 文件 | 测试范围 | 执行目标 |
| --- | --- | --- |
| `op_api/test_aclnn_flash_mla_metadata.cpp` | 内部 aclnn 属性、序列和输出描述、量化限制、平台资源、可选输入 | `transformer_op_api_ut` |
| `op_kernel_aicpu/test_flash_mla_metadata_aicpu.cpp` | 通过 SDK `CpuKernelContext` 和注册器调用真实 CPU kernel，检查序列实长、调度结果、metadata 内容和错误发布 | Host 模式为 `transformer_op_host_ut`；全量 UT 或 `OP_KERNEL_AICPU_UT=ON` 时为 `flash_mla_metadata_aicpu_ut` |
| `torch_extension/test_flash_mla_metadata.py` | 真实 Python schema、Meta/FakeTensor、参数透传、默认值及 C++ binding 的量化模式检查 | Python unittest |

从仓库根目录运行：

```bash
bash build.sh --ophost_test --opapi_test \
  --ops=flash_mla,flash_mla_metadata --soc=ascend910b,ascend910_93 --cov
```

CPU kernel 用例链接 SDK 的 Host context/nodedef 库，在 Host 上执行，覆盖
dense/TND/PA、FP16/BF16/INT8、广播、空 batch、非法序列值和重复生成 metadata，
以及 INT8 多 token 分核与负载均衡、tree mask、滑窗调度和最大 batch 边界。
无需另建测试运行器；设备间同流执行与 AICore 数值结果由设备测试验证。

构建选项及消费端协议测试见 [FlashMla UT](../../../flash_mla/tests/ut/README.md)。

Python 测试需要 PyTorch、CANN 头文件、C++ 编译器和 Ninja，无需 NPU：

```bash
python3 attention/flash_mla_metadata/tests/ut/torch_extension/test_flash_mla_metadata.py
```

编译缓存默认位于临时目录，可用 `FLASH_MLA_METADATA_UT_BUILD_DIR` 指定目录。
测试加载生产 Python 注册代码，仅替换扩展加载和设备执行边界。
`input_layer` 的八种布局、输出后缀、`quant_mode=0/1` 与非法模式在 Python/API/AICPU 分层验证；
消费端 Host UT 还验证 FP16/BF16 共用非量化 metadata、PA 三种 KV 存储布局的调度一致性，
以及量化模式不匹配时拒绝合并 metadata。

安装配套版本的算子包与 Torch 扩展后，可执行设备回归：

```bash
python3 attention/flash_mla_metadata/tests/st/test_flash_mla_metadata.py --device 0
```

使用独立包名时添加 `--package <package_name>`。设备用例覆盖八种输入/输出布局，
复用同一份非量化 metadata 验证 FP16/BF16 和 PA 的 BSH/BNSD/NZ cache；
非量化注意力和 LSE 与 CPU 参考比较，INT8 全量化与均匀注意力 CPU 参考比较。
