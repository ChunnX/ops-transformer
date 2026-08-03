# 04 远端环境检查

## 目标

在安装算子包前，确认远端运行机的 CPU 架构、SoC、CANN、Driver 和 Firmware，并与本地构建信息对照。

## 获取源码

首次使用：

```bash
git clone https://gitcode.com/xwqtju/ops-transformer.git "$HOME/ops-transformer"
cd "$HOME/ops-transformer"
git remote add upstream https://gitcode.com/cann/ops-transformer.git
git remote set-url --push upstream DISABLED
```

已克隆时：

```bash
cd "$HOME/ops-transformer"
git fetch origin --prune
```

从产物目录读取 commit：

```bash
export OPS_REMOTE_ROOT="$HOME/ops-workspace"
export OPS_OPERATOR="<operator_name>"
export OPS_COMMIT="<commit-id>"
export OPS_ARTIFACT_DIR="$OPS_REMOTE_ROOT/artifacts/$OPS_OPERATOR/$OPS_COMMIT"
cat "$OPS_ARTIFACT_DIR/git-commit.txt"
```

将远端源码对齐到该 commit：

```bash
cd "$HOME/ops-transformer"
git fetch origin --prune
git switch --detach "$OPS_COMMIT"
```

`--detach` 适合纯运行工作区，可防止在远端误产生开发提交。

## 检查远端环境

```bash
export OPS_CANN_ENV="/usr/local/Ascend/cann/set_env.sh"
source "$OPS_CANN_ENV"

uname -m
npu-smi info
echo "$ASCEND_HOME_PATH"
```

将远端信息保存到本次产物目录：

```bash
uname -m > "$OPS_ARTIFACT_DIR/remote-architecture.txt"
npu-smi info > "$OPS_ARTIFACT_DIR/remote-npu-info.txt"
env | sort | grep -E '^(ASCEND|CANN)' \
  > "$OPS_ARTIFACT_DIR/remote-cann-env.txt"
```

## 兼容性判断

至少确认：

- `build-architecture.txt` 与 `remote-architecture.txt` 兼容。
- `build-info.txt` 的 SoC 与 `npu-smi info` 显示的远端 SoC 对应。
- 本地构建用 CANN 与远端 CANN ABI 兼容。
- 远端 Driver/Firmware 支持当前 CANN。

无法确认 ABI 时，不要盲目安装；改用与远端一致的容器构建，或在远端构建算子包。

## 成功判据

- 远端源码 HEAD 等于 `OPS_COMMIT`。
- 能正常执行 `npu-smi info`。
- CPU 架构、SoC 和 CANN 兼容性已确认。

## 下一步

继续执行 [05 远端安装算子包](05_remote_install_operator.md)。
