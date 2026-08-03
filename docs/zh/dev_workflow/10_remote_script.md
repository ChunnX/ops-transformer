# 10 远端一键执行脚本

## 适用场景

远端服务器有 NPU，但不使用 AI Agent 编码。本地负责修改源码、文档和脚本；远端只需要 `git pull` 并执行：

```text
scripts/dev_workflow/remote_operator.sh
```

脚本不保存服务器密码、Token 或 SSH 私钥。

## 首次准备

```bash
git clone https://gitcode.com/xwqtju/ops-transformer.git "$HOME/ops-transformer"
cd "$HOME/ops-transformer"
scripts/dev_workflow/remote_operator.sh --help
```

已克隆时执行 `git pull --ff-only`。然后设置本次参数：

```bash
export OPS_SCRIPT="$HOME/ops-transformer/scripts/dev_workflow/remote_operator.sh"
export OPS_WORKSPACE="$HOME/ops-workspace"
export OPS_OPERATOR="<operator_name>"
export OPS_COMMIT="<full-commit-id>"
export OPS_ARTIFACT_DIR="$OPS_WORKSPACE/artifacts/$OPS_OPERATOR/$OPS_COMMIT"
```

## 1. 环境检查

```bash
"$OPS_SCRIPT" check-env --output-dir "$OPS_WORKSPACE/environment"
```

CANN 环境脚本不在默认路径时增加 `--cann-env "<actual-set-env-path>"`。报告保存在 `~/ops-workspace/environment/remote_environment.txt`。

## 2. 校验与安装算子包

```bash
"$OPS_SCRIPT" verify --artifact-dir "$OPS_ARTIFACT_DIR"

"$OPS_SCRIPT" install \
  --artifact-dir "$OPS_ARTIFACT_DIR" \
  --install-root "$HOME/cann-operator-dev/$OPS_OPERATOR/$OPS_COMMIT"
```

存在多个 `.run` 包时增加 `--package "<exact-package-name>.run"`。当前安装包不支持 `--install-path` 时省略 `--install-root`，并参考 [安装说明](05_remote_install_operator.md)。

## 3. 编译 runner

首次部署或接口/ABI 变化时执行：

```bash
cd "$HOME/ops-transformer"
git fetch origin --prune
git switch --detach "$OPS_COMMIT"

find "$HOME/cann-operator-dev/$OPS_OPERATOR/$OPS_COMMIT" -name set_env.bash -print
export OPS_CUSTOM_ENV="<path-returned-above>"

"$OPS_SCRIPT" build \
  --source-dir "$HOME/ops-transformer" \
  --artifact-dir "$OPS_ARTIFACT_DIR" \
  --custom-env "$OPS_CUSTOM_ENV" \
  -- bash build.sh --run_example "$OPS_OPERATOR" eager cust --vendor_name=xwqtju_dev
```

`--` 后是完整编译命令，脚本不使用 `eval`。日志保存为 `<artifact-dir>/runner-build.log`。

## 4. 运行并采集

```bash
"$OPS_SCRIPT" run \
  --artifact-dir "$OPS_ARTIFACT_DIR" \
  --result-root "$OPS_WORKSPACE/results/$OPS_OPERATOR" \
  --custom-env "$OPS_CUSTOM_ENV" \
  -- <runner-path> <runner-arguments>
```

每次自动生成独立的 `<timestamp>-<commit-prefix>` 结果目录，包含 commit、SHA256、运行命令、NPU 信息、环境、stdout、stderr 和退出码。脚本返回 runner 的真实退出码。

## 日常迭代

只修改 Kernel 或 Tiling、接口/ABI 不变时：

1. `git pull --ff-only` 获取最新脚本。
2. 上传新 commit 的算子包。
3. 执行 `verify` 和 `install`。
4. 用已有 runner 执行 `run`，无需再次 `build`。

人工操作背景和故障处理保留在 [操作手册总入口](README.md)。
