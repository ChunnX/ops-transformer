# FlashKDA（CANNBotDSL frozen bundle）

本目录不再维护 Transformer 私有的 FlashKDA kernel/driver。唯一实现
`op_kernel/flash_kda.py` 直接同步自：

```text
/home/z00947297/run_toolkit/cannbot-arena/
  samples/native_example/attention/flash_kda/op_kernel/flash_kda.py
```

该源文件使用新框架的 `@cannbotdsl.entry(name="FlashKda")` 和
`build_flash_kda(target)`：构建时先生成不可变 `OperatorBundle`，再由
`cannbot_framework.stage` 生成 ACLNN L0/L2/adapter，最终生成 built-in
`libopapi_cannbot_transformer.so`。mixed 包中的主 `libopapi_transformer.so` 以带
`$ORIGIN` 的 DT_NEEDED 保留该依赖，因此无需修改全局 `LD_LIBRARY_PATH`，通过主 opapi
handle 查找符号时也可见。Transformer 仓不再复制 Python eager driver、手写 provider
cache 或第二套 ACLNN host。

## 公开接口

生成的两阶段接口为：

```text
aclnnFlashKdaGetWorkspaceSize(...)
aclnnFlashKda(...)
```

当前公开合约只认证连续 BNSD：

- `q/k`: BF16 `[B, NK, S, 128]`
- `v/g`: BF16 `[B, NV, S, 128]`
- `beta`: BF16 `[B, NV, S]`
- `initial_state`: FP32 `[B, NV, 128, 128]`
- `A_log`: FP32 `[NV]`
- `dt_bias`: FP32 `[NV, 128]`
- `B=1..32`、`S=64..65536` 且为 64 的倍数、`NK/NV=1..96`
- `scale=0..1`、`lower_bound=-5..0`

BSND/TND、metadata 参数及旧 Torch dispatcher 接口不属于该 frozen Entry 合约；
对应的 Transformer 老实现已经删除。

## built-in RUN 包构建

框架和 CANNBotDSL 必须来自相互匹配的 v2 版本；nnopbase 必须包含 External Task
ABI v1（`opdev/external_op_task.h` 和 `op::AddExternalOpTask`）。示例：

```bash
cmake -S . -B build/builtin \
  -DENABLE_BUILT_IN=ON \
  -DPACKAGE_TYPE=run \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  '-DASCEND_OP_NAME=flash_kda;flash_kda_metadata' \
  -DCANNBOTDSL_SOURCE_DIR=/path/to/cannbot-dsl \
  -DCANNBOTDSL_PYTHON_EXECUTABLE=/path/to/python3 \
  -DCANNBOTDSL_PYTHONPATH=/path/to/v2/site-packages:/path/to/cannbot-dsl/framework/python \
  -DCANNBOTDSL_NNOPBASE_INCLUDE_DIR=/path/to/opbase/include/nnopbase \
  -DCANNBOTDSL_NNOPBASE_LIBRARY=/path/to/libnnopbase.so
cmake --build build/builtin --target package -j16
```

构建链：

```text
arena FlashKDA Entry
  -> build_operator(Target dav-3510)
  -> frozen OperatorBundle
  -> cannbot_framework.stage
  -> generated ACLNN objects
  -> libopapi_cannbot_transformer.so
  -> libopapi_transformer.so (DT_NEEDED)
  -> built-in RUN package
```

安装内容沿用 CANN built-in 布局：

```text
<arch>-linux/lib64/libopapi_transformer.so
<arch>-linux/include/aclnnop/aclnn_FlashKda.h
opp/built-in/op_api/lib/libopapi_cannbot_transformer.so
opp/built-in/op_impl/cannbotdsl/FlashKda/...
```

bundle/stage 在 CMake configure 阶段生成，因为框架的 CMake API需要读取冻结
`stage.json` 后才能声明 object target。输入哈希未变化时会复用已有 stage。默认
`libopapi_transformer.so` 直接依赖 External Task ABI；目标 CANN 安装也必须提供匹配的
patched nnopbase，不能在运行时退回旧系统库。
