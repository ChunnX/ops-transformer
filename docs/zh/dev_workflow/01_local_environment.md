# 01 本地环境检查

## 目标

确认本地构建机可以为目标 SoC 生成算子包，并记录将与远端对比的环境信息。

## 前提条件

- 已安装项目要求的 CANN toolkit 和编译工具链。
- 已克隆 `https://gitcode.com/xwqtju/ops-transformer.git`。
- 已知道远端运行机的 SoC 型号和 CPU 架构。

## 操作

设置本次开发参数：

```bash
export OPS_REPO="/home/user/桌面/workspace/cann/ops-transformer-xwqtju"
export OPS_OPERATOR="<operator_name>"
export OPS_SOC="ascend910b"
export OPS_VENDOR="xwqtju_dev"
export OPS_CANN_ENV="/usr/local/Ascend/cann/set_env.sh"
```

加载 CANN 环境：

```bash
source "$OPS_CANN_ENV"
cd "$OPS_REPO"
```

检查基本信息：

```bash
uname -m
git status --short --branch
git rev-parse HEAD
echo "$ASCEND_HOME_PATH"
bash build.sh --help
```

SoC 对照：

| 远端产品 | `OPS_SOC` |
|---|---|
| Atlas A2 | `ascend910b` |
| Atlas A3 | `ascend910_93` |
| Ascend 950 | `ascend950` |

对比远端时，CPU 架构和 CANN ABI 必须兼容。“远端负责编译调用程序”并不能解决算子包本身的架构或 ABI 不兼容。

## Git 准备

从最新上游创建任务分支：

```bash
git fetch upstream --prune
git switch master
git merge --ff-only upstream/master
git push origin master
git switch -c "feat/${OPS_OPERATOR}-development"
```

如果任务分支已存在：

```bash
git fetch origin --prune
git switch "feat/${OPS_OPERATOR}-development"
git pull --ff-only
```

## 成功判据

- `build.sh --help` 可以正常输出参数。
- `git status` 显示处于正确任务分支。
- 已确认本地 CPU 架构与远端兼容。
- `OPS_SOC` 对应远端实际 SoC。

## 下一步

继续执行 [02 编译算子包](02_local_build_operator.md)。
