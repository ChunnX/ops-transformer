# FlashMla UT

Host 测试复用仓库 GoogleTest 工程，由 `tests/ut/CMakeLists.txt` 自动收集。
Torch 输出形状与描述转换测试使用 Python unittest，在 CPU 上编译并执行生产 C++ 中的对应函数。

| 目录 / 文件 | 测试范围 | 执行目标 |
| --- | --- | --- |
| `torch_extension/test_flash_mla.py` | 输出/LSE 形状、分配前形状检查，切片 offset、逻辑 shape 与完整 storage 容量、原 FIA cache 连续化策略及 seq/metadata 描述检查 | Python unittest |
| `op_host/test_flash_mla_infershape.cpp` | 八种输入/输出布局的 shape 推导、LSE 和 FP16/BF16/INT8 dtype 推导 | `transformer_op_host_ut` |
| `op_host/test_flash_mla_tiling.cpp` | 注册 tiling、静态 key/payload/workspace、dense/PA/NZ/TND、量化、mask 和边界校验；单布局/量化模式 metadata 与消费端合并 | `transformer_op_host_ut` |
| `op_host/test_flash_mla_metadata_protocol_tiling.cpp` | metadata 协议与签名、动态字段合并、PA 行跨度和 workspace 容量/溢出 | `transformer_op_host_ut` |

在仓库根目录构建并执行两个算子的 UT：

```bash
bash build.sh --ophost_test --opapi_test \
  --ops=flash_mla,flash_mla_metadata --soc=ascend910b,ascend910_93 --cov
```

已有构建目录可添加 `--incremental`，离线依赖添加
`--cann_3rd_lib_path=/path/to/cache`。仅编译时添加 `--noexec`，随后仍需运行测试。
直接运行 Host 可执行文件时，将 `BUILD_PATH` 设为该次构建目录，使公共入口能够加载注册库。

Torch 测试需要 PyTorch、CANN 头文件、C++ 编译器和 Ninja，不需要 NPU。它替换 ACL 描述创建与执行边界，
不在测试代码中复制生产参数检查或描述转换实现：

```bash
python3 attention/flash_mla/tests/ut/torch_extension/test_flash_mla.py
```

编译缓存默认位于临时目录，可用 `FLASH_MLA_UT_BUILD_DIR` 指定复用目录。
Metadata 的 API 测试使用仓内 launcher/runtime 桩，以 RAII 管理张量描述与 executor，并恢复平台桩。
Host 用例执行真实推导、tiling 和共享协议代码；不在 UT 中执行 AICore 数值计算。
元数据生产端的真实 CPU kernel 测试见 [FlashMlaMetadata UT](../../../flash_mla_metadata/tests/ut/README.md)。

FlashMla 的 aclnn 由 OpDef 自动生成，不再测试已删除的手写入口。原 API 的属性、dtype、
shape、量化、mask、seq 项数、metadata shape 和平台边界由 `test_flash_mla_tiling.cpp` 覆盖；
seq 连续性/切片、metadata offset/对齐以及 cache 适配由 Torch 用例覆盖。
必选指针和 dtype/format 声明由生成接口的运行时框架处理。
