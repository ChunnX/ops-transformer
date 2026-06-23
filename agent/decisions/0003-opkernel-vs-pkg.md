# 0003: opkernel Fast Path Versus Full Package

## Status

Accepted.

## Context

`build.sh --opkernel` 能快速编译 AI Core kernel binary，但运行时 example/profiling 加载的是安装目录中的 op_api、op_host tiling、op info、kernel config 和 binary。只编译 kernel 不代表运行目录已更新。

## Decision

默认规则：

- 涉及 `op_host`、tiling key、TilingData、workspace、opdef、dtype/format、infer shape、`op_host_aclnn*`、`op_graph` 或 package metadata 的改动，必须走 full package。
- 仅涉及 `op_kernel/` 或 binary config 的改动，可以用 `--opkernel` 做快速编译，但必须显式同步新 binary/config 到运行时 vendor 目录，并记录同步证据。
- 如果无法确认运行加载的是新产物，不能写成有效性能结果。
- `scripts/st_accuracy run` 的真实设备侧精度验证必须走 full package/install gate；`--opkernel` 快速路径不能替代该 ST 精度 gate。

## Consequences

- 后续 profiling 记录必须包含 build/install/sync 命令。
- `--opkernel` 结果必须附带 vendor 目录加载证据。
- 首次 baseline 建议始终使用 `--pkg`。
