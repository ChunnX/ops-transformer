# 07 远端运行、调试与采集

## 目标

用已编译的 runner 重复运行算子，并为每次运行保存独立、可追溯的日志和性能数据。

## 创建本次结果目录

```bash
export OPS_RESULT_ROOT="$OPS_REMOTE_ROOT/results/$OPS_OPERATOR"
export OPS_RUN_ID="$(date +%Y%m%d-%H%M%S)-${OPS_COMMIT:0:12}"
export OPS_RESULT_DIR="$OPS_RESULT_ROOT/$OPS_RUN_ID"
mkdir -p "$OPS_RESULT_DIR"
```

加载环境：

```bash
source "$OPS_CANN_ENV"
source "$(cat "$OPS_ARTIFACT_DIR/custom-env-path.txt")"
```

记录元数据：

```bash
cp "$OPS_ARTIFACT_DIR/git-commit.txt" "$OPS_RESULT_DIR/"
cp "$OPS_ARTIFACT_DIR/build-info.txt" "$OPS_RESULT_DIR/"
cp "$OPS_ARTIFACT_DIR/SHA256SUMS" "$OPS_RESULT_DIR/"
uname -m > "$OPS_RESULT_DIR/remote-architecture.txt"
npu-smi info > "$OPS_RESULT_DIR/npu-info.txt"
env | sort | grep -E '^(ASCEND|CANN|LD_LIBRARY_PATH)' \
  > "$OPS_RESULT_DIR/runtime-env.txt"
```

## 功能运行

先将完整命令写入 `run-command.txt`，再执行。例如：

```bash
printf '%s\n' '<complete runner command>' \
  > "$OPS_RESULT_DIR/run-command.txt"

<complete-runner-command> \
  > "$OPS_RESULT_DIR/stdout.log" \
  2> "$OPS_RESULT_DIR/stderr.log"
```

记录返回码：

```bash
printf '%s\n' "$?" > "$OPS_RESULT_DIR/exit-code.txt"
```

如果需要严格保留失败命令的返回码，应在运行脚本中立即将 `$?` 赋值给专用变量，不要在中间插入其他命令。

## 调试和性能采集

本项目的支持范围、环境变量和命令会随 SoC 与 CANN 变化，以当前分支指南为准：

- [CANN 仿真器调试](../debug/cann_sim.md)
- [算子调试与性能采集](../debug/op_debug_prof.md)

建议将原始采集数据放在：

```text
<result-dir>/profiler/
<result-dir>/dump/
<result-dir>/logs/
```

不要把这些原始大文件提交到 Git。

## 重复运行新算子包

如果本次只修改 Kernel 或 Tiling 且接口/ABI 不变：

1. 本地重新编译、提交并上传新 commit 产物。
2. 远端用新 `OPS_COMMIT` 执行环境检查和安装。
3. 打开新 shell，加载新包的 `set_env.bash`。
4. 直接运行现有 runner，无需重新编译。

## 成功判据

- 每次运行都有唯一 `OPS_RUN_ID`。
- 结果目录包含 commit、构建信息、运行命令和输出日志。
- 不同算子包版本的结果不会互相覆盖。

## 下一步

回到本地执行 [08 下载与分析结果](08_download_results.md)。
