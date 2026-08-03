# 算子本地构建与远端运行操作手册

本手册用于以下开发模式：

```text
本地构建机：修改源码 → 编译算子包 → 上传算子包
远端运行机：安装算子包 → 编译一次调用程序 → 反复运行与采集
结果通道：远端采集结果 → 下载到本地分析
源码通道：两端通过 GitCode Fork 对齐到同一 commit
```

## 执行顺序

| 顺序 | 执行位置 | 文档 | 目的 |
|---|---|---|---|
| 1 | 本地 | [本地环境检查](01_local_environment.md) | 确认构建机具备编译条件 |
| 2 | 本地 | [编译算子包](02_local_build_operator.md) | 生成与远端匹配的 `.run` 包 |
| 3 | 本地 | [发布与上传产物](03_upload_artifact.md) | 将产物、commit 和校验值一起上传 |
| 4 | 远端 | [远端环境检查](04_remote_environment.md) | 检查 SoC、CANN 和 CPU 架构 |
| 5 | 远端 | [安装算子包](05_remote_install_operator.md) | 校验并安装本次算子包 |
| 6 | 远端 | [编译调用程序](06_remote_build_runner.md) | 在真实运行环境编译 runner |
| 7 | 远端 | [运行、调试与采集](07_remote_run_and_profile.md) | 重复运行并保留可追溯结果 |
| 8 | 本地 | [下载与分析结果](08_download_results.md) | 拉回日志和 profiling 数据 |
| 9 | 两端 | [常见问题](09_troubleshooting.md) | 处理版本、动态库和 Git 问题 |
| 10 | 远端 | [一键执行脚本](10_remote_script.md) | 在无 AI Agent 的 NPU 服务器上执行固定流程 |
| 11 | 两端 | [报错回传链路](11_diagnostic_bridge.md) | 通过私有结果仓库传递报错与修复指令 |

## 无 AI Agent 远端服务器

远端服务器不需要 AI coding 能力。本地完成源码和脚本修改后推送到 GitCode；远端只负责拉取和执行。人工流程稳定后，优先使用 `scripts/dev_workflow/remote_operator.sh`，命令说明见 [远端一键执行脚本](10_remote_script.md)。

## 统一参数

下列变量在文档中反复使用。每次打开新 shell 后先根据实际环境设置：

```bash
cd "<实际的ops-transformer仓库路径>"
export OPS_REPO="$(git rev-parse --show-toplevel)"
export OPS_BASE="$(dirname "$OPS_REPO")"
export OPS_SCRIPT="$OPS_REPO/scripts/dev_workflow/remote_operator.sh"
export OPS_WORKSPACE="$OPS_BASE/ops-workspace"
export OPS_INSTALL_BASE="$OPS_BASE/cann-operator-dev"

export OPS_OPERATOR="<operator_name>"
export OPS_SOC="ascend910b"
export OPS_VENDOR="xwqtju_dev"
export OPS_REMOTE="<user>@<remote-host>"
export OPS_REMOTE_ROOT="<远端ops-workspace绝对路径>"
export OPS_CANN_ENV="/usr/local/Ascend/cann/set_env.sh"
```

不要假设仓库一定在 `$HOME/ops-transformer`。尤其是以 `root` 登录但仓库位于普通用户目录时，`$HOME` 会解析为 `/root`。应先进入实际仓库，再使用 `git rev-parse --show-toplevel` 获取路径。

`OPS_REMOTE_ROOT` 在本地 shell 中应当填远端的绝对路径，例如 `/home/user/cann/ops-workspace`，不要依赖远端 shell 对 `$HOME` 的二次展开。


## 不可混用的三类内容

- GitCode：源码、测试、运行脚本、配置模板和小型结果摘要。
- 产物目录：`.run` 包、调用程序和每次构建的元数据。
- 结果目录：原始 profiler、Dump、日志和大型输出。

不要将 Token、密码、SSH 私钥、编译产物或原始 profiling 数据提交到 Git。

## 可追溯性约定

每个算子包和每份运行结果至少记录：

```text
Git commit ID
Git branch
operator name
SoC
CPU architecture
CANN version
build command
run command
artifact SHA256
```

完整的 Git Fork 和上游同步策略见 [个人 Fork 开发与上游同步工作流](../develop/personal_fork_workflow.md)。
