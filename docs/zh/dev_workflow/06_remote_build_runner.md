# 06 远端编译调用程序

## 目标

在远端真实 CANN 环境中编译一次调用程序（runner），后续算子内部实现更新时重复使用。

## 对齐源码

```bash
cd "$HOME/ops-transformer"
git fetch origin --prune
git switch --detach "$OPS_COMMIT"
test "$(git rev-parse HEAD)" = "$(cat "$OPS_ARTIFACT_DIR/git-commit.txt")"
```

加载环境：

```bash
source "$OPS_CANN_ENV"
source "$(cat "$OPS_ARTIFACT_DIR/custom-env-path.txt")"
```

## 方案 A：使用仓库自带 example

先查看目标算子的 `examples/` 目录及 README，确认调用模式和包模式。常见 eager 自定义包示例：

```bash
bash build.sh \
  --run_example "$OPS_OPERATOR" eager cust \
  --vendor_name="$OPS_VENDOR"
```

该命令可能编译后直接运行。如需保留可执行文件，根据构建日志确认真实输出路径，不要仅通过文件名猜测。

项目示例总览见 [快速入门](../../QUICKSTART.md)。

## 方案 B：独立 runner 项目

长期重复运行时，建议维护独立调用项目：

```text
~/ops-runner/<operator>/
├── CMakeLists.txt
├── src/
│   └── main.cpp
├── test_data/
└── scripts/
    └── run.sh
```

在远端编译：

```bash
cd "$HOME/ops-runner/$OPS_OPERATOR"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

记录 runner 使用的源码与环境：

```bash
mkdir -p "$HOME/ops-runner/$OPS_OPERATOR/metadata"
printf '%s\n' "$OPS_COMMIT" \
  > "$HOME/ops-runner/$OPS_OPERATOR/metadata/operator-commit.txt"
```

## 何时需要重新编译

通常不需重新编译 runner：

- Kernel 内部实现变更。
- Tiling 策略变更。
- 不改变接口的精度或性能修复。

需要重新编译 runner：

- aclnn 接口签名或头文件变更。
- 输入输出数量、类型或属性变更。
- 链接库名称或路径变更。
- CANN 发生 ABI 不兼容升级。

## 成功判据

- runner 在远端编译通过。
- runner 能在已加载自定义算子环境的 shell 中启动。
- runner 对应的算子 commit 已记录。

## 下一步

继续执行 [07 运行、调试与采集](07_remote_run_and_profile.md)。
