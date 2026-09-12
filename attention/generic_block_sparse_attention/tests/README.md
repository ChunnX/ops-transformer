# PA / 非 PA 回归验证

在仓库根目录运行 Host 和 ACLNN UT：

```bash
conda activate hxb_dev
bash build.sh --ophost_test --ops=generic_block_sparse_attention,generic_block_sparse_attention_metadata --soc=ascend950
bash build.sh --opapi_test --ops=generic_block_sparse_attention --soc=ascend950
```

独立 CPU 测试直接编译生产 KV 解析器，覆盖长度、地址、因果尾块、FD 分段、非法索引及 TilingKey 注册，
并验证稀疏块宽度、PA 页宽度和计算 tile 不相等时的地址层切分；需要 `g++`，不依赖 NPU：

```bash
python attention/generic_block_sparse_attention/tests/ut/op_kernel/test_kv_resolver.py
```

安装当前源码生成的算子包与 Torch 扩展后执行设备精度测试：

```bash
python -m pytest -q attention/generic_block_sparse_attention/tests/st/test_pa_non_pa.py
```

如果使用按 vendor 命名的独立 Torch 包，可设置
`GBSA_TEST_MODULE=cann_ops_transformer_<vendor>`。
`ASCEND_CUSTOM_OPP_PATH` 应指向本次构建包的 vendor 目录，避免调用其他版本。

设备测试比较同一份逻辑输入的 PA、TND 与 CPU 标杆，覆盖多 batch、MHA/GQA/MQA、
非对齐的 TND batch 起点、存储 padding、PA dim0 非连续 KV（TND KV 全连续）、尾块、重复/乱序/非法索引、
过滤后空任务，以及 FD 阈值 11/12。A5 额外覆盖 FP8 的 FP16/BF16 输出。
当前接口不支持 `return_softmax_lse=True`，因此设备测试仅比较 attention 输出。

共享设备环境下可用`GBSA_TEST_DEVICE=6`指定空闲卡；默认设备为0。

ACLNN UT 覆盖非 PA 的 Key/Value 任一维非连续输入被拒绝；Host UT 覆盖直接图调用的相同约束。

## 最新提交原有验证记录

- 环境：conda `hxb_dev`，Python 3.11，PyTorch/torch_npu 2.12，CANN 9.1.0。
- Ascend950PR（设备6）：16组 PA/TND/CPU 精度对比全部通过，包含 FP8 两种输出类型。
- Host/Tiling/InferShape/Metadata调度：53项通过；Metadata接口校验：3项通过。
- Torch Meta/Fake：5项通过；CPU解析器与注册key：2项通过。
- A2/A5 PA及TND共28个Cube/Vector实例通过语法检查；A2的12个实例全部生成机器码，A5完成4种dtype算子二进制构建。
- A2未进行设备精度测试；本次未做PA性能基线对比。

本地验证使用独立vendor目录合并AICPU和AICore产物。仓库当前`--opkernel_aicpu`
模式会关闭AICore打包，需要分别构建并合并两部分；该构建行为未在本次修改。

## FIA 寻址方式与尺寸解耦回归

TND 的 `blockSize` 置 0，`kStride0/vStride0` 表示 token stride；MMAD 使用独立的
`kvBaseTile=128`。Host UT 同时断言这些值，避免实现再次依赖“逻辑页 stride”。
地址层测试包含稀疏块/页/tile 为 192/256/128、32/256/128、页小于 tile、
跨页因果尾部，以及超过 256 个计算 tile 的范围迭代。
这些不同尺寸的组合仅在 CPU 地址层验证，完整算子仍只开放原有 128 尺寸。

本次解耦后在同一 conda 环境重新验证：53项 Host/InferShape/Metadata调度 UT、
2项CPU解析器/注册key测试、28个Kernel语法检查及12个A2机器码检查全部通过；
重编A5算子后，设备6的16组精度测试全部通过（22.24秒）。

构建时发现仓库现有增量规则未给源码复制和Kernel编译的`.done`标记添加源码依赖。
修改Kernel后，使用完整构建，或同时清除以下GBSA专属标记后再增量构建：

```text
build/binary/ascend950/src/generic_block_sparse_attention/generic_block_sparse_attention_ascend950_src_copy.done
build/binary/ascend950/gen/generic_block_sparse_attention_ascend950_{0,1,2,3}.done
```

验证前已确认构建源码副本与工作区一致、4个A5二进制均已重新生成，且验证vendor目录
内的二进制SHA256与构建结果一致。仅删除编译标记而保留源码复制标记仍会编译旧源码。
