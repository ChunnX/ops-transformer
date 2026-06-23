# Package And Deployment Runbook

`--opkernel` 编译新的 AI Core kernel binary，但它不会自动更新运行时 vendor 目录中的所有产物。

## Use Full Package When

以下情况默认使用 `--pkg`：

- 修改 `op_host/*tiling*`。
- 修改 tiling key、block dim、workspace 或 TilingData layout。
- 修改 `op_host/*def*`、infer shape、dtype/format 约束或 op info。
- 修改 `op_host_aclnn*`、aclnn L0/L2 参数校验或分发。
- 修改 `op_graph/` 或 graph 接入。
- 第一次建立 baseline，或无法确认 runtime 是否加载新产物。

命令：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
./build_out/cann-ops-transformer-custom_linux-aarch64.run
```

## Use Kernel-Only Fast Path When

以下情况可以考虑 `--opkernel` 快速路径：

- 改动仅限 `op_kernel/`。
- 或只修改 binary config / kernel compile 参数。
- 已经安装过兼容 custom package。
- 能显式同步 `build/binary/ascend910b/bin` 中的新 binary/config 到运行时 vendor 目录。
- 能记录同步路径和校验证据。

命令：

```bash
bash build.sh --opkernel --soc=ascend910b --ops=<op_name> -j16 -O3
```

如果无法确认同步结果，退回 full package。

## Vendor Risk

custom package 的示例 vendor 是 `custom_transformer`。风险包括：

- 旧 package 覆盖当前 package。
- 当前 example 链接了旧 `cust_opapi`。
- runtime 加载系统 OPP 原始实现而不是 custom 实现。
- 只同步 kernel binary，但 op info/config 仍旧。

profiling 前要确认：

- custom vendor 目录存在。
- `op_api/lib` 中有目标 `cust_opapi`。
- `op_impl/ai_core/tbe` 下有目标 kernel/config。
- `ASCEND_CUSTOM_OPP_PATH` 或 OPP vendor 优先级指向当前安装产物。

## MC2 Deployment Notes

MC2 算子可能额外需要：

- HCCL runtime。
- 多卡可见性。
- rank/world size 配置。
- 通信拓扑稳定。

MC2 的 package 正确加载只是前置条件，不足以证明性能数据有效。
