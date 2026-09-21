# FlashKDA Metadata

`flash_kda_metadata` 在 AICPU 上生成 FlashKDA 的 int32 调度元数据。源码
`op_kernel/flash_kda_metadata.py` 原样同步自：

```text
/home/z00947297/RUN/cannbot-arena-zyh/
  samples/flash_kda_metadata/flash_kda_metadata.py
```

同步版本 SHA-256 为
`6b555448eae83e8fd89d3c7c1205bd40d2a701240f470a950abba7a24c23878f`。
Transformer 不修改其中的容量估算、序列解析、设备信息查询、代价模型、分组、任务划分
和即时编译回退逻辑。

```python
metadata = ops.flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens=None)
```

`q/v` 使用 BF16，`initial_state` 为 FP32 的 `[B, Nv, 128, 128]`。
`layout_qkv` 为 `BNSD`、`BSND` 或 `TND`；TND 必须提供同设备、连续的
int32 `cu_seqlens`，长度为 `B+1`，首项为 0，末项为总 token 数。
该接口作为独立 Torch API 交付。新版 frozen `aclnnFlashKda` 有自己的 Entry 合约，
不把 metadata 暴露为 ACLNN 参数。

构建命令和调用示例见 [FlashKDA](../flash_kda/README.md)。构建时，Transformer 的
`torch_extension/setup.py` 识别 `@aicpu_kernel`，统一调用框架已有
`compile_aicpu_kernel` 编译原 AICPU kernel，随 Torch
扩展发布 `flash_kda_metadata.so`。产品侧 Torch wrapper 只把该预编译对象绑定到 arena
模块公开使用的 `_AICPU_COMPILED` 缓存；arena 源文件本身保持逐字节一致。原有公开入口
继续执行参数检查、分配和 launch，源码环境仍保留 arena 原有的即时编译行为。

当前 frozen OperatorBundle 框架负责 AIC/AIV Entry 的 ACLNN 导出；AICPU metadata
仍使用 CANNBotDSL 的通用 `@aicpu_kernel` 编译接口。这是两个不同执行后端，不在公共
框架代码中为 FlashKDA 增加特殊判断。

两个算子的 custom/builtin run 构建、安装及联动测试命令见相邻
[FlashKDA 说明](../flash_kda/README.md)。metadata 的 AICPU 二进制随同一个 run 直接安装。
