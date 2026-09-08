# FlashKDA

FlashKDA 接收 BF16 的 `q/k/v/g/beta`，返回 BF16 输出和 FP32 末态；支持
`BNSD`、`BSND`、`TND` 和 GQA，当前 head dimension 为 128，目标为 Ascend 950。
`TND` 的序列边界由 `flash_kda_metadata` 的 `cu_seqlens` 输入指定。

## 构建和使用

先加载 CANN 9.2 环境，将已安装 **CANNBotDSL run 包**中的
`cann/python/site-packages` 加入 `PYTHONPATH`。不需要安装 CANNBotDSL wheel。

在 transformer 仓的 `torch_extension` 目录执行：

```bash
TORCH_EXTENSION_OPS=flash_kda,flash_kda_metadata \
CANNBOTDSL_NPU_ARCH=dav-3510 \
python setup.py bdist_wheel
```

该命令同时编译两个算子的二进制，并生成
`dist/cann_ops_transformer_custom-1.0.0-cp312-cp312-linux_x86_64.whl`。
这是包含算子二进制的 Torch 扩展包，CANNBotDSL 编译器仍来自 run 安装。
将扩展包安装到应用的 Python 路径后：

```python
import cann_ops_transformer_custom as ops

metadata = ops.flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens)
out, final_state = ops.flash_kda(
    q, k, v, g, beta, scale, initial_state,
    A_log, dt_bias, lower_bound, layout_qkv, metadata,
)
```

`initial_state`、`A_log` 和 `dt_bias` 使用 FP32；所有输入放在同一 NPU 上且连续。
`initial_state` 为 `[B, Nv, 128, 128]`，`A_log` 为 `[Nv]`，`dt_bias` 为
`[Nv, 128]`；`lower_bound` 取 `[-5, 0]`。非 TND 布局可传 `cu_seqlens=None`。
公开参数和计算逻辑沿用 `op_kernel/flash_kda.py`。

## 二进制接入

- `op_kernel/flash_kda.py` 保留计算实现，在文件末尾用框架示例中的
  `@register("flash_kda")` 收集原有动态编译入口，生成 Native Provider。
  不另设 shape 清单，不拆分或复制算子实现。
- `torch_extension` 负责注册 Torch schema、NPU 调用和 Meta 推导，同时注册
  包内 `_native` 资源。它不生成 C++ 占位文件，也不经过 ACLNN 包装。
- 框架 Native 加载器按编译契约匹配二进制。设置
  `CANNBOTDSL_NATIVE_BINARY_MODE=require` 可要求匹配安装的二进制，缺失时直接报错。
  Python 保留原有参数检查、工作区分配和调度入口；设备计算来自预编译 `.so`。
- `flash_kda_metadata` 的 AICPU 编译链独立于 Native Provider，详见相邻目录说明。

## 验证

只维护 `test/flash_kda_test.py`，其中同时验证 metadata 和 FlashKDA 的联动：

```bash
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 \
CANNBOTDSL_NPU_ARCH=dav-3510 \
python -m pytest attention/flash_kda/test/flash_kda_test.py -v -s
```

选择空闲的设备编号后，在仓根目录执行。测试使用已安装的扩展包；自定义包名时
设置 `KDA_PACKAGE`。测试禁用编译缓存并拦截 AIC/AIV 和 AICPU 编译调用，确认
二进制命中；精度通过独立 FP32 token 递推校验，保留原测试的
`abs(actual-reference)/(1+abs(reference))` 的 P99.9 不超过 `5e-3` 标准。
覆盖三种布局、非整块序列、2:1/3:1 GQA、多 batch、跨轮工作区复用和 Meta。
