# Build Runbook

本 runbook 面向后续单算子优化的日常构建。完整端到端静态分析见 [../build-flow.md](../build-flow.md)。

## Preflight

确认环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
which bisheng
python3 --version
gcc --version
cmake --version
```

确认目标：

```bash
op=<op_name>
find attention ffn gmm mc2 moe posembedding -maxdepth 2 -type d -name "$op"
find attention ffn gmm mc2 moe posembedding -path "*/$op/op_kernel" -type d
```

## Full Package Build

用于首次 baseline、涉及 host/tiling/op_api/op_graph/op info 的改动，或无法确认 runtime 加载边界时：

```bash
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=custom -j16 -O3
./build_out/cann-ops-transformer-custom_linux-aarch64.run
```

主要输入：

- `<domain>/<op>/op_host*`
- `<domain>/<op>/op_graph`
- `<domain>/<op>/op_kernel`
- `common/` 和依赖目录
- `CMakeLists.txt`、`cmake/`

主要输出：

- `build/` 中间产物
- `build/binary/ascend910b/`
- `build_out/cann-ops-transformer-custom_linux-aarch64.run`
- 安装后的 custom OPP vendor 目录

## Kernel-Only Build

仅改 `op_kernel/` 或 binary config 时可用于快速迭代：

```bash
bash build.sh --opkernel --soc=ascend910b --ops=<op_name> -j16 -O3
```

注意：

- 这不等价于运行时加载新实现。
- 必须同步新 binary/config 到安装目录，或退回 full package。
- 同步路径和校验证据必须写入实验记录。

## UT Build

按改动范围选择：

```bash
# host tiling / infer shape
bash build.sh -u --ophost_test --soc=ascend910b --ops=<op_name> -j16

# aclnn API
bash build.sh -u --opapi_test --soc=ascend910b --ops=<op_name> -j16

# graph 接入
bash build.sh -u --opgraph_test --soc=ascend910b --ops=<op_name> -j16

# AscendC kernel UT
bash build.sh -u --opkernel_test --soc=ascend910b --ops=<op_name> -j16

# 只编译不执行
bash build.sh -u --noexec --soc=ascend910b --ops=<op_name> -j16
```

`docs/zh/context/build.md` 说明 `--ophost_test` 与 `-u --ophost` 等价，`--opapi_test` 与 `-u --opapi` 等价，`--opkernel_test` 与 `-u --opkernel` 等价。

## Useful Flags

`ops-transformer` 当前静态可见的相关开关：

- `--soc=ascend910b`
- `--ops=<op_name>`
- `--vendor_name=custom`
- `--experimental`
- `--cann_3rd_lib_path=<path>`
- `--oom`
- `--kernel_template_input=<args>`
- `--bisheng_flags=<flags>`
- `--op_debug_config`
- `--ops-compile-options`

不要从 `ops-math` 文档直接套用 `--dump_cce`、`--mssanitizer`；它们不是当前 `ops-transformer build.sh` 静态暴露的入口。

## Build Logs And Artifacts

常见路径：

- `build/autogen/`
- `build/binary/ascend910b/gen/`
- `build/binary/ascend910b/bin/`
- `build_out/`
- `output/`

`build.sh` 会清理 `build/` 和 `output/`。不要把手工记录或唯一证据放在这些目录。
