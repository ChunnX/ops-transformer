# FlashKDA Metadata

`flash_kda_metadata` 在 AICPU 上生成 FlashKDA 的 int32 调度元数据。
原有容量估算、序列解析、代价模型、分组和任务划分均保留在
`op_kernel/flash_kda_metadata.py`。

```python
metadata = ops.flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens=None)
```

`q/v` 使用 BF16，`initial_state` 为 FP32 的 `[B, Nv, 128, 128]`。
`layout_qkv` 为 `BNSD`、`BSND` 或 `TND`；TND 必须提供同设备、连续的
int32 `cu_seqlens`，长度为 `B+1`，首项为 0，末项为总 token 数。
返回的 metadata 直接传给 `ops.flash_kda`。

构建命令和调用示例见 [FlashKDA](../flash_kda/README.md)。构建时，Transformer 的
`torch_extension/setup.py` 识别 `@aicpu_kernel`，统一调用框架已有
`compile_aicpu_kernel` 编译原 AICPU kernel，随 Torch
扩展发布 `flash_kda_metadata.so`。Torch 集成层通过 DSL 的
`register_aicpu_resources` 登记包内资源。运行时，DSL 的
`compile_aicpu_kernel` 优先加载已登记的二进制；未登记的源码调用保留原编译行为。
算子不再定义路径处理或加载函数，原有公开入口继续执行参数检查、分配和 launch。

当前框架的 `package.native` 自动收集 AIC/AIV Provider，尚未收集 AICPU 产物；
AICPU 的导出由 Transformer 构建集成负责，资源绑定与加载由 DSL AICPU 运行时负责。
Torch 扩展加载不到该 `.so` 会报错，不回退到即时编译。

不单独设置 test 目录；metadata 的联动与 Meta 检查集中在
`../flash_kda/test/flash_kda_test.py`。

两个算子的 custom/builtin run 构建、安装及联动测试命令见相邻
[FlashKDA 说明](../flash_kda/README.md)。metadata 的 AICPU 二进制随同一个 run 直接安装。
